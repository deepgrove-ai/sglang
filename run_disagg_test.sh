#!/bin/bash
# PD Disaggregation Test for 2x B200 GPUs
#
# This script sets up prefill/decode disaggregation:
# - GPU 0: Prefill server (computation-intensive)
# - GPU 1: Decode server (memory-intensive)
# - Router: Coordinates requests between both
#
# Benefits:
# - Decode never interrupted by prefill batches
# - Each phase optimized independently
# - Better tail latency for decode
#
# Usage:
#   ./run_disagg_test.sh              # Start all servers
#   ./run_disagg_test.sh stop         # Stop all servers
#   ./run_disagg_test.sh prefill      # Start only prefill server
#   ./run_disagg_test.sh decode       # Start only decode server
#   ./run_disagg_test.sh router       # Start only router
#   ./run_disagg_test.sh benchmark    # Run benchmark against router

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

# Configuration
MODEL_PRESET="${MODEL_PRESET:-qwen3-moe}"
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"  # mooncake, nixl, or fake
ENABLE_SPECULATIVE="${ENABLE_SPECULATIVE:-1}"     # 1=enable NGRAM speculative decoding, 0=disable
PREFILL_PORT=30000
DECODE_PORT=30001
ROUTER_PORT=8000

# Model paths
case "$MODEL_PRESET" in
    qwen3-moe)
        MODEL_ID="/mnt/data/30ba3b"
        MODEL_NAME="qwen3-moe-disagg"
        ;;
    klear-20b)
        MODEL_ID="/home/ubuntu/raghav/20ba1b"
        MODEL_NAME="klear-20b-disagg"
        ;;
    *)
        echo "Error: Unknown model preset '$MODEL_PRESET'"
        exit 1
        ;;
esac

# Environment setup
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/python:$PYTHONPATH"
# NOTE: Do NOT use expandable_segments with disaggregation - torch.cuda.MemPool doesn't support it
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PYTORCH_CUDA_ALLOC_CONF

# NVLink transport for B200 (recommended for NVL72 / single-node NVLink setups)
# This enables direct GPU-to-GPU KV cache transfer over NVLink
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL="${SGLANG_MOONCAKE_CUSTOM_MEM_POOL:-True}"
export MC_FORCE_MNNVL="${MC_FORCE_MNNVL:-True}"

# ============== i2s-tuned FlashInfer optimizations ==============
# These match run_server_ternary.sh i2s-tuned mode
export SGLANG_FLASHINFER_USE_TENSOR_CORE="${SGLANG_FLASHINFER_USE_TENSOR_CORE:-true}"
export SGLANG_FLASHINFER_WORKSPACE_SIZE="${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((1024*1024*1024))}"  # 1GB
export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE="${SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE:-4096}"

# ============== Ternary quantization settings ==============
export SGLANG_TERNARY_LOG_LEVEL="${SGLANG_TERNARY_LOG_LEVEL:-INFO}"
export SGLANG_TERNARY_CACHE="${SGLANG_TERNARY_CACHE:-1}"
export SGLANG_TERNARY_CACHE_WRITE="${SGLANG_TERNARY_CACHE_WRITE:-1}"
export TERNARY_MOE_DECODE_ONLY="${TERNARY_MOE_DECODE_ONLY:-1}"
export TERNARY_QUANTIZE_LM_HEAD="${TERNARY_QUANTIZE_LM_HEAD:-1}"
export TERNARY_FUSE_RMSNORM_QKV="${TERNARY_FUSE_RMSNORM_QKV:-1}"
export TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE="${TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE:-1}"

# PDL (Programmatic Dependent Launch) for Blackwell/Hopper
export TERNARY_ENABLE_PDL="${TERNARY_ENABLE_PDL:-1}"

# JAX Pallas Tuning (Batched Decode)
export TERNARY_USE_JAX_PALLAS="${TERNARY_USE_JAX_PALLAS:-1}"
export TERNARY_JAX_MIN_BATCH="${TERNARY_JAX_MIN_BATCH:-32}"
export TERNARY_JAX_MAX_BATCH="${TERNARY_JAX_MAX_BATCH:-128}"

# Pinned output copy for async GPU->CPU transfers
export SGLANG_ENABLE_PINNED_OUTPUT_COPY="${SGLANG_ENABLE_PINNED_OUTPUT_COPY:-1}"
export SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES="${SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES:-1048576}"

# Memory allocation
REQUEST_GPU_MEMORY="${REQUEST_GPU_MEMORY:-0.90}"

# Log files
LOG_DIR="${LOG_DIR:-/tmp/sglang_disagg}"
mkdir -p "$LOG_DIR"

kill_existing() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Killing existing process on port $port..."
        lsof -Pi :$port -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

start_prefill() {
    echo "========================================"
    echo "Starting PREFILL server on GPU 0"
    echo "Model: $MODEL_ID"
    echo "Port: $PREFILL_PORT"
    echo "Transfer backend: $TRANSFER_BACKEND"
    echo "========================================"
    
    kill_existing $PREFILL_PORT
    
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
        --model-path "$MODEL_ID" \
        --served-model-name "$MODEL_NAME-prefill" \
        --port $PREFILL_PORT \
        --tp-size 1 \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --trust-remote-code \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $TRANSFER_BACKEND \
        --quantization ternary \
        --attention-backend flashinfer \
        --chunked-prefill-size 1024 \
        2>&1 | tee "$LOG_DIR/prefill.log" &
    
    echo "Prefill server starting in background (log: $LOG_DIR/prefill.log)"
}

start_decode() {
    echo "========================================"
    echo "Starting DECODE server on GPU 1"
    echo "Model: $MODEL_ID"
    echo "Port: $DECODE_PORT"
    echo "Transfer backend: $TRANSFER_BACKEND"
    echo "========================================"
    
    kill_existing $DECODE_PORT
    
    # Build decode command with optional speculative decoding
    local DECODE_CMD="CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
        --model-path $MODEL_ID \
        --served-model-name $MODEL_NAME-decode \
        --port $DECODE_PORT \
        --tp-size 1 \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --trust-remote-code \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $TRANSFER_BACKEND \
        --quantization ternary \
        --attention-backend flashinfer \
        --chunked-prefill-size 1024"
    
    # Add speculative decoding if enabled (NGRAM doesn't need a draft model)
    if [[ "${ENABLE_SPECULATIVE:-1}" == "1" ]]; then
        DECODE_CMD="$DECODE_CMD --speculative-algorithm NGRAM --speculative-num-draft-tokens 4"
        echo "  Speculative decoding: NGRAM with 4 draft tokens"
    fi
    
    eval "$DECODE_CMD" 2>&1 | tee "$LOG_DIR/decode.log" &
    
    echo "Decode server starting in background (log: $LOG_DIR/decode.log)"
}

start_router() {
    echo "========================================"
    echo "Starting PD Router"
    echo "Prefill: http://127.0.0.1:$PREFILL_PORT"
    echo "Decode: http://127.0.0.1:$DECODE_PORT"
    echo "Router port: $ROUTER_PORT"
    echo "========================================"
    
    kill_existing $ROUTER_PORT
    
    python -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://127.0.0.1:$PREFILL_PORT" \
        --decode "http://127.0.0.1:$DECODE_PORT" \
        --host 0.0.0.0 \
        --port $ROUTER_PORT \
        2>&1 | tee "$LOG_DIR/router.log" &
    
    echo "Router starting in background (log: $LOG_DIR/router.log)"
}

wait_for_server() {
    local url=$1
    local name=$2
    local max_wait=${3:-300}
    
    echo "Waiting for $name to be ready at $url..."
    local start_time=$(date +%s)
    while true; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "$name is ready!"
            return 0
        fi
        
        local elapsed=$(($(date +%s) - start_time))
        if [ $elapsed -gt $max_wait ]; then
            echo "ERROR: $name failed to start within ${max_wait}s"
            return 1
        fi
        
        echo "  Still waiting... (${elapsed}s elapsed)"
        sleep 5
    done
}

stop_all() {
    echo "Stopping all disaggregation servers..."
    kill_existing $PREFILL_PORT
    kill_existing $DECODE_PORT
    kill_existing $ROUTER_PORT
    echo "All servers stopped."
}

run_benchmark() {
    echo "========================================"
    echo "Running benchmark against PD disaggregation"
    echo "Router endpoint: http://127.0.0.1:$ROUTER_PORT"
    echo "========================================"
    
    # Quick smoke test
    echo ""
    echo "=== Quick smoke test ==="
    curl -s -X POST "http://127.0.0.1:$ROUTER_PORT/generate" \
        -H "Content-Type: application/json" \
        -d '{
            "text": "Hello, how are you?",
            "sampling_params": {"temperature": 0, "max_new_tokens": 50}
        }' | python -m json.tool
    
    echo ""
    echo "=== Latency benchmark (single request) ==="
    
    # Single request latency test
    for i in 1 2 3; do
        echo ""
        echo "Request $i:"
        time curl -s -X POST "http://127.0.0.1:$ROUTER_PORT/generate" \
            -H "Content-Type: application/json" \
            -d '{
                "text": "Explain quantum computing in simple terms:",
                "sampling_params": {"temperature": 0, "max_new_tokens": 100}
            }' | python -c "import sys,json; d=json.load(sys.stdin); print(f'Output tokens: {d[\"meta_info\"][\"completion_tokens\"]}')"
    done
    
    echo ""
    echo "Benchmark complete!"
    echo "For full throughput testing, use:"
    echo "  python -m sglang.bench_serving --backend sglang --port $ROUTER_PORT --num-prompts 100"
}

show_status() {
    echo "========================================"
    echo "PD Disaggregation Status"
    echo "========================================"
    
    echo ""
    echo "Prefill server (port $PREFILL_PORT):"
    if curl -s "http://127.0.0.1:$PREFILL_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ HEALTHY"
    else
        echo "  ✗ NOT RUNNING"
    fi
    
    echo ""
    echo "Decode server (port $DECODE_PORT):"
    if curl -s "http://127.0.0.1:$DECODE_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ HEALTHY"
    else
        echo "  ✗ NOT RUNNING"
    fi
    
    echo ""
    echo "Router (port $ROUTER_PORT):"
    if curl -s "http://127.0.0.1:$ROUTER_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ HEALTHY"
    else
        echo "  ✗ NOT RUNNING"
    fi
}

# Main command handling
case "${1:-all}" in
    prefill)
        start_prefill
        ;;
    decode)
        start_decode
        ;;
    router)
        start_router
        ;;
    all)
        echo "Starting PD Disaggregation setup on 2x B200 GPUs"
        echo ""
        
        # Start prefill and decode in parallel
        start_prefill
        sleep 2
        start_decode
        
        # Wait for both servers to be ready
        echo ""
        echo "Waiting for servers to initialize..."
        wait_for_server "http://127.0.0.1:$PREFILL_PORT/health" "Prefill server" 600
        wait_for_server "http://127.0.0.1:$DECODE_PORT/health" "Decode server" 600
        
        # Start router
        echo ""
        start_router
        sleep 3
        
        echo ""
        echo "========================================"
        echo "PD Disaggregation is ready!"
        echo "========================================"
        echo ""
        echo "Endpoints:"
        echo "  Router (use this): http://127.0.0.1:$ROUTER_PORT"
        echo "  Prefill server:    http://127.0.0.1:$PREFILL_PORT"
        echo "  Decode server:     http://127.0.0.1:$DECODE_PORT"
        echo ""
        echo "Test with:"
        echo "  curl -X POST http://127.0.0.1:$ROUTER_PORT/generate \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"text\": \"Hello!\", \"sampling_params\": {\"max_new_tokens\": 50}}'"
        echo ""
        echo "Or run: $0 benchmark"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    benchmark)
        run_benchmark
        ;;
    logs)
        echo "Log files:"
        echo "  Prefill: $LOG_DIR/prefill.log"
        echo "  Decode:  $LOG_DIR/decode.log"
        echo "  Router:  $LOG_DIR/router.log"
        echo ""
        echo "Tail all logs with: tail -f $LOG_DIR/*.log"
        ;;
    -h|--help)
        echo "PD Disaggregation Test Script for 2x B200 GPUs"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  all        Start prefill, decode, and router (default)"
        echo "  prefill    Start only prefill server"
        echo "  decode     Start only decode server"
        echo "  router     Start only router"
        echo "  stop       Stop all servers"
        echo "  status     Check server health"
        echo "  benchmark  Run latency benchmark"
        echo "  logs       Show log file locations"
        echo ""
        echo "Environment variables:"
        echo "  MODEL_PRESET       Model to use (qwen3-moe, klear-20b)"
        echo "  TRANSFER_BACKEND   KV transfer backend (mooncake, nixl, fake)"
        echo "  ENABLE_SPECULATIVE Enable NGRAM speculative decoding (default: 1)"
        echo "  REQUEST_GPU_MEMORY GPU memory fraction (default: 0.90)"
        echo "  LOG_DIR            Directory for log files"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use --help for usage"
        exit 1
        ;;
esac
