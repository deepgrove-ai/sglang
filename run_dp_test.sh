#!/bin/bash
# Data Parallel Test for 2x B200 GPUs
#
# Each GPU runs a full model copy, requests are load-balanced between them.
# Simpler than PD disaggregation, no KV transfer needed.
#
# Usage:
#   ./run_dp_test.sh              # Start DP server
#   ./run_dp_test.sh stop         # Stop server
#   ./run_dp_test.sh benchmark    # Run benchmark

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

# Configuration
MODEL_PRESET="${MODEL_PRESET:-qwen3-moe}"
DP_SIZE="${DP_SIZE:-2}"
SERVER_PORT=30080
LOAD_BALANCE="${LOAD_BALANCE:-round_robin}"  # round_robin, shortest_queue, minimum_tokens

# Model paths
case "$MODEL_PRESET" in
    qwen3-moe)
        MODEL_ID="/mnt/data/30ba3b"
        MODEL_NAME="qwen3-moe-dp${DP_SIZE}"
        ;;
    klear-20b)
        MODEL_ID="/home/ubuntu/raghav/20ba1b"
        MODEL_NAME="klear-20b-dp${DP_SIZE}"
        ;;
    *)
        echo "Error: Unknown model preset '$MODEL_PRESET'"
        exit 1
        ;;
esac

# Environment setup
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/python:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============== i2s-tuned FlashInfer optimizations ==============
export SGLANG_FLASHINFER_USE_TENSOR_CORE="${SGLANG_FLASHINFER_USE_TENSOR_CORE:-true}"
export SGLANG_FLASHINFER_WORKSPACE_SIZE="${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((1024*1024*1024))}"
export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE="${SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE:-4096}"

# ============== Ternary quantization settings ==============
export SGLANG_TERNARY_LOG_LEVEL="${SGLANG_TERNARY_LOG_LEVEL:-INFO}"
export SGLANG_TERNARY_CACHE="${SGLANG_TERNARY_CACHE:-1}"
export SGLANG_TERNARY_CACHE_WRITE="${SGLANG_TERNARY_CACHE_WRITE:-1}"
export TERNARY_MOE_DECODE_ONLY="${TERNARY_MOE_DECODE_ONLY:-1}"
export TERNARY_QUANTIZE_LM_HEAD="${TERNARY_QUANTIZE_LM_HEAD:-1}"
export TERNARY_FUSE_RMSNORM_QKV="${TERNARY_FUSE_RMSNORM_QKV:-1}"
export TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE="${TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE:-1}"
export TERNARY_ENABLE_PDL="${TERNARY_ENABLE_PDL:-1}"
export TERNARY_USE_JAX_PALLAS="${TERNARY_USE_JAX_PALLAS:-1}"
export TERNARY_JAX_MIN_BATCH="${TERNARY_JAX_MIN_BATCH:-32}"
export TERNARY_JAX_MAX_BATCH="${TERNARY_JAX_MAX_BATCH:-128}"
export SGLANG_ENABLE_PINNED_OUTPUT_COPY="${SGLANG_ENABLE_PINNED_OUTPUT_COPY:-1}"
export SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES="${SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES:-1048576}"

# Memory allocation
REQUEST_GPU_MEMORY="${REQUEST_GPU_MEMORY:-0.90}"

# Speculative decoding
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"

kill_existing() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Killing existing process on port $port..."
        lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

start_server() {
    echo "========================================"
    echo "Starting Data Parallel Server"
    echo "Model: $MODEL_ID"
    echo "DP Size: $DP_SIZE GPUs"
    echo "Port: $SERVER_PORT"
    echo "Load Balance: $LOAD_BALANCE"
    echo "========================================"
    
    kill_existing $SERVER_PORT
    
    # Build command
    CMD="python -m sglang_router.launch_server \
        --model-path $MODEL_ID \
        --served-model-name $MODEL_NAME \
        --port $SERVER_PORT \
        --dp $DP_SIZE \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --trust-remote-code \
        --quantization ternary \
        --attention-backend flashinfer \
        --chunked-prefill-size 1024 \
        --load-balance-method $LOAD_BALANCE"
    
    # Add speculative decoding if enabled
    if [[ "$ENABLE_SPECULATIVE" == "1" ]]; then
        CMD="$CMD --speculative-algorithm NGRAM --speculative-num-draft-tokens 4"
        echo "Speculative decoding: NGRAM with 4 draft tokens"
    fi
    
    echo ""
    echo "Running command:"
    echo "$CMD"
    echo "========================================"
    
    eval "$CMD"
}

stop_server() {
    echo "Stopping DP server..."
    kill_existing $SERVER_PORT
    # Also kill any sglang worker processes
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang_router" 2>/dev/null || true
    echo "Server stopped."
}

run_benchmark() {
    echo "========================================"
    echo "Running benchmark against DP server"
    echo "Endpoint: http://127.0.0.1:$SERVER_PORT"
    echo "========================================"
    
    # Quick smoke test
    echo ""
    echo "=== Quick smoke test ==="
    curl -s -X POST "http://127.0.0.1:$SERVER_PORT/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Hello, how are you?",
            "sampling_params": {"temperature": 0, "max_new_tokens": 50}
        }' | python -m json.tool
    
    echo ""
    echo "=== Running concurrency benchmark ==="
    ./util/benchmark_concurrency.sh --port $SERVER_PORT -c 1,2,4,8,16,32,64 -r 50 -l "dp${DP_SIZE}"
}

show_status() {
    echo "========================================"
    echo "DP Server Status"
    echo "========================================"
    
    if curl -s "http://127.0.0.1:$SERVER_PORT/health" > /dev/null 2>&1; then
        echo "Server (port $SERVER_PORT): ✓ HEALTHY"
        echo ""
        echo "Server info:"
        curl -s "http://127.0.0.1:$SERVER_PORT/get_model_info" | python -m json.tool 2>/dev/null || true
    else
        echo "Server (port $SERVER_PORT): ✗ NOT RUNNING"
    fi
}

case "${1:-start}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    status)
        show_status
        ;;
    benchmark)
        run_benchmark
        ;;
    -h|--help)
        echo "Data Parallel Test Script for 2x B200 GPUs"
        echo ""
        echo "Each GPU runs a full model copy. Requests are load-balanced."
        echo "No KV transfer needed - simpler and more robust than PD disaggregation."
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  start      Start DP server (default)"
        echo "  stop       Stop server"
        echo "  status     Check server health"
        echo "  benchmark  Run concurrency benchmark"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_PRESET       Model to use (qwen3-moe, klear-20b)"
        echo "  DP_SIZE            Number of GPU replicas (default: 2)"
        echo "  LOAD_BALANCE       Load balancing method (round_robin, shortest_queue, minimum_tokens)"
        echo "  ENABLE_SPECULATIVE Enable NGRAM speculative decoding (default: 1)"
        echo "  REQUEST_GPU_MEMORY GPU memory fraction (default: 0.90)"
        echo ""
        echo "Examples:"
        echo "  $0                          # Start with defaults"
        echo "  DP_SIZE=2 $0                # Explicit 2 GPU replicas"
        echo "  LOAD_BALANCE=shortest_queue $0  # Use shortest queue LB"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use --help for usage"
        exit 1
        ;;
esac
