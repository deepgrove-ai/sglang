#!/bin/bash
# Run SGLang server with different quantization modes:
#   - fp16: Pure FP16/BF16 (no quantization, baseline)
#   - i2s: Ternary + I2_S (8x memory reduction, FP16 inference)
#   - i2s-fp8: Ternary + I2_S + FP8 tensor cores (fastest)
#
# Usage:
#   ./run_server_ternary.sh              # Default: i2s mode, 1 GPU
#   ./run_server_ternary.sh fp16         # Pure FP16 baseline
#   ./run_server_ternary.sh i2s          # Ternary + I2_S (FP16 inference)
#   ./run_server_ternary.sh i2s --tp 4   # Use 4 GPUs with tensor parallelism
#   ./run_server_ternary.sh i2s-fp8 --tp 8  # FP8 mode on 8 GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

# Default values
QUANT_MODE="i2s"
TP_SIZE=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        fp16|i2s|i2s-tuned|i2s-kvfp8|i2s-fp8)
            QUANT_MODE="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [MODE] [--tp N]"
            echo ""
            echo "Modes:"
            echo "  fp16:      Pure FP16/BF16 (no quantization, baseline)"
            echo "  i2s:       Ternary + I2_S (8x memory reduction, FP16 inference)"
            echo "  i2s-tuned: Ternary + I2_S + Optimized FlashInfer (RECOMMENDED)"
            echo "  i2s-kvfp8: Ternary + I2_S + FP8 KV Cache (Lower memory, faster attention)"
            echo "  i2s-fp8:   Ternary + I2_S + FP8 tensor cores (fastest inference)"
            echo ""
            echo "Options:"
            echo "  --tp N     Tensor parallelism: shard model across N GPUs (default: 1)"
            echo ""
            echo "Examples:"
            echo "  $0 i2s --tp 2    # Run i2s mode on 2 GPUs"
            echo "  $0 fp16 --tp 8   # Run fp16 mode on 8 GPUs"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate TP size
if ! [[ "$TP_SIZE" =~ ^[1-8]$ ]]; then
    echo "Error: Invalid tensor parallelism size '$TP_SIZE' (must be 1-8)"
    exit 1
fi

# Set Python path
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/ternary_optimization/production:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable ternary quantization logging (set to DEBUG to see all layer checks)
# Options: DEBUG, INFO, WARNING, ERROR
# DEBUG: Shows all layer checks, skips, and quantization details
# INFO: Shows quantized layers and important events (default)
export SGLANG_TERNARY_LOG_LEVEL="${SGLANG_TERNARY_LOG_LEVEL:-INFO}"

# Startup ternary cache (speeds restarts massively; first run populates cache)
# - Enable/disable:
export SGLANG_TERNARY_CACHE="${SGLANG_TERNARY_CACHE:-1}"
# - Optional: override cache location (default: ~/.cache/sglang/ternary)
# export SGLANG_TERNARY_CACHE_DIR="/path/to/fast_nvme/sglang_ternary_cache"
# - Optional: disable cache writes (load-only)
export SGLANG_TERNARY_CACHE_WRITE="${SGLANG_TERNARY_CACHE_WRITE:-1}"

# MoE startup time can dominate; decode-only mode makes MoE caching practical.
# - 1: keep original BF16 MoE weights for prefill, use ternary kernel for decode when available
# - 0: quantize MoE BF16 weights too (slower startup)
export TERNARY_MOE_DECODE_ONLY="${TERNARY_MOE_DECODE_ONLY:-1}"

# Enable ternary profiling (set to 1 to enable for debugging)
# When enabled, tracks timing for ternary operations and saves to JSON file
export TERNARY_ENABLE_PROFILING="${TERNARY_ENABLE_PROFILING:-0}"
export TERNARY_PROFILE_OUTPUT="${TERNARY_PROFILE_OUTPUT:-/tmp/ternary_profile.json}"

# Op-level profiling mode (recommended when using /start_profile traces):
# CUDA graphs collapse many ops into graph replays, hiding per-op costs.
# Set SGLANG_PROFILE_OPLEVEL=1 to disable CUDA graphs + overlap + torch.compile for clearer traces.
# Set to 0 for normal production serving.
export SGLANG_PROFILE_OPLEVEL="${SGLANG_PROFILE_OPLEVEL:-0}"

# When op-level profiling is enabled, also disable torch.compile (conflicts with inference mode
# and hides per-op breakdown).  TORCHINDUCTOR_DISABLE=1 or TORCH_COMPILE_DISABLE=1 both work.
if [[ "${SGLANG_PROFILE_OPLEVEL:-0}" == "1" ]]; then
    export TORCHINDUCTOR_DISABLE=1
    export TORCH_COMPILE_DISABLE=1
    echo "Op-level profiling enabled: disabling CUDA graphs, overlap scheduling, and torch.compile"
fi

# Mode-specific configuration
case "$QUANT_MODE" in
    fp16)
        # Pure FP16 mode: no quantization
        QUANT_FLAG=""
        QUANT_DESC="Pure FP16/BF16 (baseline, no quantization)"
        ;;
    
    i2s)
        # Ternary + I2S mode (default)
        # The ternary quantization defaults to i2s mode, so we just need to specify --quantization ternary
        # Enable FlashInfer by default as it is generally faster for decode on H100
        QUANT_FLAG="--quantization ternary --attention-backend flashinfer"
        QUANT_DESC="Ternary + I2_S (8x memory reduction, FP16 inference)"
        ;;
    
    i2s-tuned)
        # Ternary + I2S + Tuned FlashInfer (RECOMMENDED for H100)
        # Optimizes attention computation with tensor cores and larger workspace
        export SGLANG_FLASHINFER_USE_TENSOR_CORE=true
        export SGLANG_FLASHINFER_WORKSPACE_SIZE=$((1024*1024*1024))  # 1GB
        export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=4096
        QUANT_FLAG="--quantization ternary --attention-backend flashinfer"
        QUANT_DESC="Ternary + I2_S + Tuned FlashInfer (RECOMMENDED for H100)"
        echo "Note: FlashInfer optimizations enabled (Tensor Cores, 1GB workspace, 4K tiles)"
        ;;
    
    i2s-kvfp8)
        # Ternary + I2S + FP8 KV Cache
        # Ternary weights for linear layers, FP8 for Attention KV cache
        QUANT_FLAG="--quantization ternary --kv-cache-dtype fp8_e5m2"
        QUANT_DESC="Ternary + I2_S + FP8 KV Cache (Lower memory, faster attention)"
        ;;

    i2s-fp8)
        # Ternary + I2S + FP8 mode (fastest inference)
        # Enable FP8 tensor core acceleration via environment variable
        export SGLANG_TERNARY_USE_FP8=1
        QUANT_FLAG="--quantization ternary"
        QUANT_DESC="Ternary + I2_S + FP8 tensor cores (fastest inference)"
        echo "Note: FP8 mode enabled (requires H100/Ada GPUs with torch._scaled_mm support)"
        ;;
esac

# Model and server configuration
MODEL_ID="RedMod/sft_dist_l_q"
MODEL_NAME="sft_dist_l_q-$QUANT_MODE"
SERVER_PORT=30080
DEFAULT_REQUEST_GPU_MEMORY="0.85"
REQUEST_GPU_MEMORY="${REQUEST_GPU_MEMORY:-$DEFAULT_REQUEST_GPU_MEMORY}"
# Performance-tuned defaults for ternary decode
# CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:--1}"
# CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-1}"
# DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
# DISABLE_OVERLAP="${DISABLE_OVERLAP:-1}"
# MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"

echo "========================================"
echo "Starting SGLang server"
echo "Mode: $QUANT_DESC"
echo "Model: $MODEL_ID"
echo "Port: $SERVER_PORT"
echo "Tensor Parallelism: $TP_SIZE GPU(s)"
echo "GPU memory allocation: $REQUEST_GPU_MEMORY"
echo "Ternary log level: $SGLANG_TERNARY_LOG_LEVEL"
if [[ "$TERNARY_ENABLE_PROFILING" == "1" ]]; then
    echo "Ternary profiling: ENABLED (output: $TERNARY_PROFILE_OUTPUT)"
else
    echo "Ternary profiling: DISABLED"
fi
echo "Chunked prefill size: $CHUNKED_PREFILL_SIZE"
echo "CUDA graph max batch size: $CUDA_GRAPH_MAX_BS"
echo "Disable radix cache: $DISABLE_RADIX_CACHE"
echo "Disable overlap: $DISABLE_OVERLAP"
echo "Max running requests: $MAX_RUNNING_REQUESTS"
echo "========================================"

# Check if server is already running on the port
if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Server already running on port $SERVER_PORT. Killing existing process..."
    lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Build the command
CMD="python -m sglang.launch_server \
    --model-path $MODEL_ID \
    --served-model-name $MODEL_NAME \
    --port $SERVER_PORT \
    --tp-size $TP_SIZE \
    --mem-fraction-static $REQUEST_GPU_MEMORY \
    --enable-p2p-check \
    --trust-remote-code \
    --attention-backend flashinfer"

#     --chunked-prefill-size $CHUNKED_PREFILL_SIZE \

#     # --max-running-requests $MAX_RUNNING_REQUESTS \
# Optional knobs
# if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
#     CMD="$CMD --disable-radix-cache"
# fi
# if [[ "$DISABLE_OVERLAP" == "1" ]]; then
#     CMD="$CMD --disable-overlap"
# fi

# Add quantization flag if set
if [ -n "$QUANT_FLAG" ]; then
    CMD="$CMD $QUANT_FLAG"
fi

# Profiling-friendly flags (optional)
if [[ "$SGLANG_PROFILE_OPLEVEL" == "1" ]]; then
    CMD="$CMD --disable-cuda-graph --disable-overlap-schedule"
    echo "Note: SGLANG_PROFILE_OPLEVEL=1 -> disabling CUDA graphs + overlap scheduling for op-level profiling"
fi

# Log the command for debugging
echo "Running command:"
echo "$CMD"
echo "========================================"

# Run the server
eval "$CMD"
