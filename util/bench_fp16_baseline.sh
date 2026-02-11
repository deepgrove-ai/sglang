#!/bin/bash
# FP16 Baseline Benchmark — matches ternary concurrency sweep
# Launches FP16 server on GPU 0, runs bench_serving at C=1,8,16,32,48,56,64
# Outputs structured results for comparison with ternary numbers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_DIR="$(dirname "$SCRIPT_DIR")"
cd "$SGLANG_DIR"
source .venv/bin/activate

MODEL_PATH="/scratch/mangrove"
SERVER_PORT=30090
LOG="/tmp/fp16_bench_server.log"
RESULTS="/tmp/fp16_bench_results.txt"

# Kill any existing server on this port
pkill -9 -f "sglang.*$SERVER_PORT" 2>/dev/null || true
sleep 2

echo "========================================" | tee "$RESULTS"
echo "FP16 Baseline Benchmark" | tee -a "$RESULTS"
echo "Model: $MODEL_PATH" | tee -a "$RESULTS"
echo "Date: $(date)" | tee -a "$RESULTS"
echo "========================================" | tee -a "$RESULTS"

# Launch FP16 server (no quantization)
echo "Launching FP16 server on port $SERVER_PORT..."
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name mangrove-fp16 \
    --port "$SERVER_PORT" \
    --tp-size 1 \
    --mem-fraction-static 0.85 \
    --enable-p2p-check \
    --trust-remote-code \
    --attention-backend flashinfer \
    --chunked-prefill-size 1024 \
    --cuda-graph-max-bs 128 \
    --enable-hierarchical-cache \
    --hicache-ratio 4.0 \
    > "$LOG" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server readiness
echo "Waiting for FP16 server to be ready..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://127.0.0.1:${SERVER_PORT}/health" 2>/dev/null | grep -q "ok"; then
        echo "Server ready after ${WAITED}s"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died. Check $LOG"
        tail -50 "$LOG"
        exit 1
    fi
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  Still waiting... ${WAITED}s"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "ERROR: Server did not become ready in ${MAX_WAIT}s"
    tail -50 "$LOG"
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# Quick smoke test
echo "Running smoke test..."
SMOKE=$(curl -s "http://127.0.0.1:${SERVER_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"mangrove-fp16","prompt":"Hello","max_tokens":5,"temperature":0}')
echo "Smoke test: $SMOKE"

# Run benchmarks at each concurrency level
CONCURRENCIES="1 8 16 32 48 56 64"
INPUT_LEN=512
OUTPUT_LEN=128
NUM_PROMPTS_PER_C=128

echo "" | tee -a "$RESULTS"
echo "Benchmark Parameters: input_len=$INPUT_LEN, output_len=$OUTPUT_LEN, num_prompts=$NUM_PROMPTS_PER_C" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"
printf "%-6s  %-12s  %-10s  %-12s  %-12s  %-12s  %-12s\n" \
    "C" "TPS(total)" "TPS/user" "Med TTFT" "P99 TTFT" "Med TPOT" "P99 TPOT" | tee -a "$RESULTS"
printf "%-6s  %-12s  %-10s  %-12s  %-12s  %-12s  %-12s\n" \
    "---" "----------" "--------" "--------" "--------" "--------" "--------" | tee -a "$RESULTS"

for C in $CONCURRENCIES; do
    echo ""
    echo "=== Benchmarking C=$C ==="
    
    # Adjust num_prompts: at least 2x concurrency, min 64
    NP=$((C * 4))
    [ $NP -lt 64 ] && NP=64
    [ $NP -gt 256 ] && NP=256
    
    BENCH_OUT="/tmp/fp16_bench_c${C}.json"
    
    python3 -m sglang.bench_serving \
        --backend sglang \
        --host 127.0.0.1 \
        --port "$SERVER_PORT" \
        --model mangrove-fp16 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NP" \
        --request-rate "$((C * 3))" \
        --output-file "$BENCH_OUT" \
        2>&1 | tee "/tmp/fp16_bench_c${C}.log"
    
    # Parse results
    if [ -f "$BENCH_OUT" ]; then
        PARSED=$(python3 -c "
import json, sys
with open('$BENCH_OUT') as f:
    d = json.load(f)
tps = d.get('output_throughput', 0)
tps_user = tps / $C if $C > 0 else tps
med_ttft = d.get('median_ttft_ms', 0)
p99_ttft = d.get('p99_ttft_ms', 0)
med_tpot = d.get('median_tpot_ms', 0)
p99_tpot = d.get('p99_tpot_ms', 0)
med_itl = d.get('median_itl_ms', 0)
p99_itl = d.get('p99_itl_ms', 0)
# Use ITL if TPOT not available
if med_tpot == 0 and med_itl > 0:
    med_tpot = med_itl
    p99_tpot = p99_itl
print(f'{tps:.0f}|{tps_user:.0f}|{med_ttft:.0f}|{p99_ttft:.0f}|{med_tpot:.1f}|{p99_tpot:.1f}')
" 2>/dev/null)
        
        if [ -n "$PARSED" ]; then
            IFS='|' read -r TPS TPS_U MTTFT P99TTFT MTPOT P99TPOT <<< "$PARSED"
            printf "%-6s  %-12s  %-10s  %-12s  %-12s  %-12s  %-12s\n" \
                "C=$C" "${TPS}" "${TPS_U}" "${MTTFT}ms" "${P99TTFT}ms" "${MTPOT}ms" "${P99TPOT}ms" | tee -a "$RESULTS"
        else
            echo "C=$C  PARSE ERROR" | tee -a "$RESULTS"
        fi
    else
        echo "C=$C  NO OUTPUT FILE" | tee -a "$RESULTS"
    fi
    
    # Brief cooldown
    sleep 3
done

echo "" | tee -a "$RESULTS"
echo "========================================" | tee -a "$RESULTS"
echo "Benchmark complete." | tee -a "$RESULTS"
echo "Results saved to: $RESULTS" | tee -a "$RESULTS"
echo "Server log: $LOG" | tee -a "$RESULTS"

# Extract KV token info from server log
KV_TOKENS=$(grep -o "max_total_num_tokens=[0-9]*" "$LOG" | head -1 | cut -d= -f2)
echo "KV tokens allocated: ${KV_TOKENS:-unknown}" | tee -a "$RESULTS"

# Cleanup
echo "Stopping FP16 server..."
kill -9 $SERVER_PID 2>/dev/null || true
pkill -9 -f "sglang.*$SERVER_PORT" 2>/dev/null || true
echo "Done."
