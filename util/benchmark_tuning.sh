#!/bin/bash
# =============================================================================
# FlashInfer Tuning Sweep Script
# =============================================================================
# Runs the same mode with different FlashInfer tuning parameters to find
# optimal settings for your hardware.
#
# Usage:
#   ./benchmark_tuning.sh                   # Run default sweep on i2s-fp8
#   ./benchmark_tuning.sh i2s-tuned         # Run sweep on specific mode
#   QUICK=1 ./benchmark_tuning.sh           # Quick sweep (fewer params)
#
# Output:
#   Results saved to: tuning_sweep_YYYYMMDD_HHMMSS/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_SCRIPT="$SGLANG_DIR/run_server_ternary.sh"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_concurrency.py"
PYTHON="${SGLANG_DIR}/.venv/bin/python"

# Mode to test (default: i2s-fp8)
MODE="${1:-i2s-fp8}"

# Server settings
SERVER_PORT="${SERVER_PORT:-30080}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"

# Benchmark settings
# NOTE: REQUESTS must be >= max concurrency level to actually test that concurrency
CONCURRENCY="${CONCURRENCY:-1,4,16,64}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-64}"  # Must be >= max(CONCURRENCY)
MAX_TOKENS="${MAX_TOKENS:-256}"

# Tuning parameters to sweep
if [ "${QUICK:-0}" == "1" ]; then
    # Quick sweep
    PREFILL_TILE_SIZES=(4096 8192)
    DECODE_TILE_SIZES=(2048 4096)
    DISABLE_SPLIT_KV_THRESHOLDS=(0 8)
else
    # Full sweep
    PREFILL_TILE_SIZES=(2048 4096 8192 16384)
    DECODE_TILE_SIZES=(1024 2048 4096 8192)
    DISABLE_SPLIT_KV_THRESHOLDS=(0 4 8 16)
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SGLANG_DIR/tuning_sweep_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "FlashInfer Tuning Sweep"
echo "=============================================================="
echo "Mode: $MODE"
echo "Output: $OUTPUT_DIR"
echo "Prefill tile sizes: ${PREFILL_TILE_SIZES[*]}"
echo "Decode tile sizes: ${DECODE_TILE_SIZES[*]}"
echo "Disable split-KV thresholds: ${DISABLE_SPLIT_KV_THRESHOLDS[*]}"
echo "=============================================================="

# Functions (same as benchmark_modes.sh)
wait_for_server() {
    local port=$1
    local timeout=$2
    local start_time=$(date +%s)
    
    echo -n "Waiting for server..."
    while true; do
        if curl -s "http://${SERVER_HOST}:${port}/health" > /dev/null 2>&1; then
            echo " ready!"
            return 0
        fi
        local elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -ge $timeout ]; then
            echo " timeout!"
            return 1
        fi
        echo -n "."
        sleep 2
    done
}

kill_existing_server() {
    local port=$1
    local pids=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Results tracking
RESULTS_FILE="$OUTPUT_DIR/sweep_results.csv"
echo "prefill_tile,decode_tile,disable_split_thresh,c1_tps,c4_tps,c16_tps,c64_tps,c1_lat,c64_lat" > "$RESULTS_FILE"

run_count=0
total_runs=$(( ${#PREFILL_TILE_SIZES[@]} * ${#DECODE_TILE_SIZES[@]} * ${#DISABLE_SPLIT_KV_THRESHOLDS[@]} ))

for prefill_tile in "${PREFILL_TILE_SIZES[@]}"; do
    for decode_tile in "${DECODE_TILE_SIZES[@]}"; do
        for disable_thresh in "${DISABLE_SPLIT_KV_THRESHOLDS[@]}"; do
            run_count=$((run_count + 1))
            
            CONFIG_NAME="p${prefill_tile}_d${decode_tile}_t${disable_thresh}"
            echo ""
            echo "=============================================================="
            echo "Run $run_count/$total_runs: $CONFIG_NAME"
            echo "=============================================================="
            
            # Kill existing server
            kill_existing_server $SERVER_PORT
            
            # Set tuning parameters
            export SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE=$prefill_tile
            export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=$decode_tile
            export SGLANG_FLASHINFER_DECODE_DISABLE_SPLIT_KV_BELOW_BS=$disable_thresh
            
            # Start server
            LOG_FILE="$OUTPUT_DIR/${CONFIG_NAME}_server.log"
            cd "$SGLANG_DIR"
            nohup bash "$SERVER_SCRIPT" "$MODE" > "$LOG_FILE" 2>&1 &
            SERVER_PID=$!
            
            # Wait for server
            if ! wait_for_server "$SERVER_PORT" "$STARTUP_TIMEOUT"; then
                echo "Server failed to start"
                kill -9 $SERVER_PID 2>/dev/null || true
                echo "$prefill_tile,$decode_tile,$disable_thresh,FAIL,FAIL,FAIL,FAIL,FAIL,FAIL" >> "$RESULTS_FILE"
                continue
            fi
            
            # Run benchmark
            RESULT_FILE="$OUTPUT_DIR/${CONFIG_NAME}.json"
            "$PYTHON" "$BENCHMARK_SCRIPT" \
                --host "$SERVER_HOST" \
                --port "$SERVER_PORT" \
                --concurrency-levels "$CONCURRENCY" \
                --requests "$REQUESTS_PER_LEVEL" \
                --max-tokens "$MAX_TOKENS" \
                --warmup 3 \
                --label "$CONFIG_NAME" \
                --output "$RESULT_FILE" \
                2>&1 | tee "$OUTPUT_DIR/${CONFIG_NAME}_bench.log"
            
            # Extract key metrics
            if [ -f "$RESULT_FILE" ]; then
                metrics=$("$PYTHON" << PYEXTRACT
import json
with open("$RESULT_FILE") as f:
    data = json.load(f)
results = {r["concurrency"]: r for r in data.get("results", [])}
c1 = results.get(1, {})
c4 = results.get(4, {})
c16 = results.get(16, {})
c64 = results.get(64, {})
print(f"{c1.get('tokens_per_sec_total', 0):.1f},"
      f"{c4.get('tokens_per_sec_total', 0):.1f},"
      f"{c16.get('tokens_per_sec_total', 0):.1f},"
      f"{c64.get('tokens_per_sec_total', 0):.1f},"
      f"{c1.get('latency_p50', 0):.1f},"
      f"{c64.get('latency_p50', 0):.1f}")
PYEXTRACT
)
                echo "$prefill_tile,$decode_tile,$disable_thresh,$metrics" >> "$RESULTS_FILE"
            else
                echo "$prefill_tile,$decode_tile,$disable_thresh,ERR,ERR,ERR,ERR,ERR,ERR" >> "$RESULTS_FILE"
            fi
            
            # Stop server
            kill -TERM $SERVER_PID 2>/dev/null || true
            sleep 2
            kill -9 $SERVER_PID 2>/dev/null || true
            kill_existing_server $SERVER_PORT
            
            sleep "$COOLDOWN_SEC"
        done
    done
done

# Generate summary
echo ""
echo "=============================================================="
echo "TUNING SWEEP RESULTS"
echo "=============================================================="
echo ""
cat "$RESULTS_FILE" | column -t -s','
echo ""

# Find best config for each metric
"$PYTHON" << PYFINAL
import csv
from pathlib import Path

results_file = Path("$RESULTS_FILE")
rows = []
with open(results_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            row['c64_tps_val'] = float(row['c64_tps']) if row['c64_tps'] not in ('FAIL', 'ERR') else 0
            row['c1_lat_val'] = float(row['c1_lat']) if row['c1_lat'] not in ('FAIL', 'ERR') else 99999
            rows.append(row)
        except:
            pass

if rows:
    # Best for throughput (c64)
    best_tps = max(rows, key=lambda x: x['c64_tps_val'])
    print(f"Best for HIGH CONCURRENCY (c64 TPS): {best_tps['c64_tps']} TPS")
    print(f"  Config: prefill={best_tps['prefill_tile']}, decode={best_tps['decode_tile']}, thresh={best_tps['disable_split_thresh']}")
    print()
    
    # Best for latency (c1)
    best_lat = min(rows, key=lambda x: x['c1_lat_val'])
    print(f"Best for LOW LATENCY (c1 p50): {best_lat['c1_lat']} ms")
    print(f"  Config: prefill={best_lat['prefill_tile']}, decode={best_lat['decode_tile']}, thresh={best_lat['disable_split_thresh']}")
    print()

print(f"Full results: $RESULTS_FILE")
PYFINAL

echo "=============================================================="
echo "Sweep complete! Results in: $OUTPUT_DIR"
echo "=============================================================="
