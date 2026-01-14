#!/bin/bash
# =============================================================================
# Multi-Mode Benchmark Script
# =============================================================================
# Starts the server with each quantization mode, runs concurrency benchmarks,
# and saves results to a timestamped output file.
#
# Usage:
#   ./benchmark_modes.sh                    # Run all modes
#   ./benchmark_modes.sh fp16 i2s-fp8       # Run specific modes
#   CONCURRENCY="1,4,16,64" ./benchmark_modes.sh  # Custom concurrency levels
#
# Output:
#   Results saved to: benchmark_results_YYYYMMDD_HHMMSS/
#     - summary.json      (all results in JSON)
#     - summary.csv       (CSV for spreadsheets)
#     - <mode>.json       (per-mode detailed results)
#     - <mode>.log        (server logs for debugging)
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
SERVER_SCRIPT="$SGLANG_DIR/run_server_ternary.sh"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_concurrency.py"
PYTHON="${SGLANG_DIR}/.venv/bin/python"

# Server settings
SERVER_PORT="${SERVER_PORT:-30080}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"  # 5 minutes max startup time
COOLDOWN_SEC="${COOLDOWN_SEC:-10}"          # Seconds between modes

# Benchmark settings
# NOTE: REQUESTS_PER_LEVEL must be >= max concurrency level to actually test that concurrency
CONCURRENCY="${CONCURRENCY:-1,2,4,8,16,32,64}"
REQUESTS_PER_LEVEL="${REQUESTS_PER_LEVEL:-64}"  # --requests, must be >= max(CONCURRENCY)
MAX_TOKENS="${MAX_TOKENS:-256}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-5}"  # --warmup

# Available modes (order matters - run baseline first)
ALL_MODES=(
    "fp16"
    "i2s"
    "i2s-tuned"
    "i2s-fp8"
    "i2s-fp8-full"
)

# Parse command line arguments for specific modes
if [ $# -gt 0 ]; then
    MODES=("$@")
else
    MODES=("${ALL_MODES[@]}")
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SGLANG_DIR/benchmark_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================="
echo "Multi-Mode Benchmark"
echo "=============================================================="
echo "Output directory: $OUTPUT_DIR"
echo "Modes to test: ${MODES[*]}"
echo "Concurrency levels: $CONCURRENCY"
echo "Requests per level: $REQUESTS_PER_LEVEL"
echo "Max tokens: $MAX_TOKENS"
echo "=============================================================="

# Function to check if server is ready
wait_for_server() {
    local port=$1
    local timeout=$2
    local start_time=$(date +%s)
    
    echo -n "Waiting for server on port $port..."
    while true; do
        if curl -s "http://${SERVER_HOST}:${port}/health" > /dev/null 2>&1; then
            echo " ready!"
            return 0
        fi
        
        local elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -ge $timeout ]; then
            echo " timeout after ${timeout}s!"
            return 1
        fi
        
        echo -n "."
        sleep 2
    done
}

# Function to kill any existing server on the port
kill_existing_server() {
    local port=$1
    local pids=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing existing processes on port $port: $pids"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Function to start server and return PID
start_server() {
    local mode=$1
    local log_file=$2
    
    echo "Starting server with mode: $mode"
    
    # Kill any existing server
    kill_existing_server $SERVER_PORT
    
    # Start server in background
    cd "$SGLANG_DIR"
    nohup bash "$SERVER_SCRIPT" "$mode" > "$log_file" 2>&1 &
    local server_pid=$!
    
    echo "Server PID: $server_pid"
    echo $server_pid
}

# Function to stop server
stop_server() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping server (PID: $pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
    fi
    
    # Also kill by port just in case
    kill_existing_server $SERVER_PORT
}

# Function to run benchmark for a mode
run_benchmark() {
    local mode=$1
    local output_file=$2
    
    echo "Running benchmark for mode: $mode"
    
    "$PYTHON" "$BENCHMARK_SCRIPT" \
        --host "$SERVER_HOST" \
        --port "$SERVER_PORT" \
        --concurrency-levels "$CONCURRENCY" \
        --requests "$REQUESTS_PER_LEVEL" \
        --max-tokens "$MAX_TOKENS" \
        --warmup "$WARMUP_REQUESTS" \
        --label "$mode" \
        --output "$output_file" \
        2>&1 | tee -a "$OUTPUT_DIR/${mode}_benchmark.log"
    
    return ${PIPESTATUS[0]}
}

# Initialize summary arrays
declare -A MODE_RESULTS

# Main benchmark loop
echo ""
echo "Starting benchmark run..."
echo ""

for mode in "${MODES[@]}"; do
    echo "=============================================================="
    echo "Testing mode: $mode"
    echo "=============================================================="
    
    LOG_FILE="$OUTPUT_DIR/${mode}_server.log"
    RESULT_FILE="$OUTPUT_DIR/${mode}.json"
    
    # Start server
    SERVER_PID=$(start_server "$mode" "$LOG_FILE")
    
    # Wait for server to be ready
    if ! wait_for_server "$SERVER_PORT" "$STARTUP_TIMEOUT"; then
        echo "ERROR: Server failed to start for mode $mode"
        echo "Check log file: $LOG_FILE"
        stop_server "$SERVER_PID"
        MODE_RESULTS[$mode]="FAILED_TO_START"
        continue
    fi
    
    # Run benchmark
    if run_benchmark "$mode" "$RESULT_FILE"; then
        MODE_RESULTS[$mode]="SUCCESS"
        echo "Benchmark completed for mode: $mode"
    else
        MODE_RESULTS[$mode]="BENCHMARK_FAILED"
        echo "ERROR: Benchmark failed for mode $mode"
    fi
    
    # Stop server
    stop_server "$SERVER_PID"
    
    # Cooldown between modes
    if [ "$mode" != "${MODES[-1]}" ]; then
        echo "Cooling down for ${COOLDOWN_SEC}s..."
        sleep "$COOLDOWN_SEC"
    fi
    
    echo ""
done

# Generate summary
echo "=============================================================="
echo "Generating summary..."
echo "=============================================================="

# Create combined JSON summary
"$PYTHON" << 'PYTHON_SCRIPT' - "$OUTPUT_DIR" "${MODES[@]}"
import sys
import json
import os
from pathlib import Path

output_dir = Path(sys.argv[1])
modes = sys.argv[2:]

# Collect all results
all_results = {}
for mode in modes:
    result_file = output_dir / f"{mode}.json"
    if result_file.exists():
        with open(result_file) as f:
            all_results[mode] = json.load(f)
    else:
        all_results[mode] = {"error": "No results file"}

# Write combined JSON
with open(output_dir / "summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

# Generate CSV summary
csv_lines = [
    "Mode,Concurrency,Requests,OK,Failed,Total_TPS,Per_Req_TPS,Lat_p50,Lat_p95,Lat_p99,Lat_avg"
]

for mode, data in all_results.items():
    if "error" in data:
        csv_lines.append(f"{mode},ERROR,,,,,,,,,")
        continue
    
    results = data.get("results", [])
    for r in results:
        csv_lines.append(
            f"{mode},"
            f"{r.get('concurrency', '')},"
            f"{r.get('num_requests', '')},"
            f"{r.get('successful_requests', '')},"
            f"{r.get('failed_requests', '')},"
            f"{r.get('tokens_per_sec_total', 0):.1f},"
            f"{r.get('tokens_per_sec_per_request', 0):.1f},"
            f"{r.get('latency_p50', 0):.1f},"
            f"{r.get('latency_p95', 0):.1f},"
            f"{r.get('latency_p99', 0):.1f},"
            f"{r.get('latency_avg', 0):.1f}"
        )

with open(output_dir / "summary.csv", "w") as f:
    f.write("\n".join(csv_lines))

# Print comparison table
print("\n" + "=" * 120)
print("BENCHMARK COMPARISON")
print("=" * 120)

# Get all concurrency levels
all_concurrencies = set()
for mode, data in all_results.items():
    if "results" in data:
        for r in data["results"]:
            all_concurrencies.add(r["concurrency"])

concurrencies = sorted(all_concurrencies)

# Print header
header = f"{'Mode':<15}"
for c in concurrencies:
    header += f" | C={c:>3} TPS"
print(header)
print("-" * 120)

# Print each mode
for mode in modes:
    data = all_results.get(mode, {})
    if "error" in data:
        print(f"{mode:<15} | ERROR")
        continue
    
    results = {r["concurrency"]: r for r in data.get("results", [])}
    row = f"{mode:<15}"
    for c in concurrencies:
        if c in results:
            tps = results[c].get("tokens_per_sec_total", 0)
            row += f" | {tps:>8.1f}"
        else:
            row += f" | {'N/A':>8}"
    print(row)

print("=" * 120)

# Print latency comparison at key concurrency levels
key_concurrencies = [1, 8, 32, 64]
key_concurrencies = [c for c in key_concurrencies if c in concurrencies]

if key_concurrencies:
    print("\nLATENCY (p50 ms) COMPARISON")
    print("-" * 80)
    header = f"{'Mode':<15}"
    for c in key_concurrencies:
        header += f" | C={c:>3} p50"
    print(header)
    print("-" * 80)
    
    for mode in modes:
        data = all_results.get(mode, {})
        if "error" in data:
            continue
        results = {r["concurrency"]: r for r in data.get("results", [])}
        row = f"{mode:<15}"
        for c in key_concurrencies:
            if c in results:
                lat = results[c].get("latency_p50", 0)
                row += f" | {lat:>8.1f}"
            else:
                row += f" | {'N/A':>8}"
        print(row)
    print("-" * 80)

print(f"\nResults saved to: {output_dir}")
print(f"  - summary.json (all data)")
print(f"  - summary.csv (for spreadsheets)")
print(f"  - <mode>.json (detailed per-mode results)")
print(f"  - <mode>_server.log (server logs)")
PYTHON_SCRIPT

# Print final status
echo ""
echo "=============================================================="
echo "Benchmark Complete!"
echo "=============================================================="
echo "Mode Results:"
for mode in "${MODES[@]}"; do
    status="${MODE_RESULTS[$mode]:-UNKNOWN}"
    echo "  $mode: $status"
done
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "=============================================================="
