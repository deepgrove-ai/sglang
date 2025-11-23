#!/bin/bash
# Run SGLang server with different quantization modes:
#   - fp16: Pure FP16/BF16 (no quantization, baseline)
#   - i2s: Ternary + I2_S (8x memory reduction, FP16 inference)
#   - i2s-fp8: Ternary + I2_S + FP8 tensor cores (fastest)
#
# Usage:
#   ./run_server_ternary.sh          # Default: i2s mode
#   ./run_server_ternary.sh fp16     # Pure FP16 baseline
#   ./run_server_ternary.sh i2s      # Ternary + I2_S (FP16 inference)
#   ./run_server_ternary.sh i2s-fp8  # Ternary + I2_S + FP8 tensor cores

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate_env.sh"

# Parse quantization mode from command line
QUANT_MODE="${1:-i2s}"  # Default to i2s if not specified

# Validate mode
if [[ ! "$QUANT_MODE" =~ ^(fp16|i2s|i2s-tuned|i2s-kvfp8|i2s-fp8)$ ]]; then
    echo "Error: Invalid quantization mode '$QUANT_MODE'"
    echo "Usage: $0 [fp16|i2s|i2s-tuned|i2s-kvfp8|i2s-fp8]"
    echo "  fp16:      Pure FP16/BF16 (no quantization, baseline)"
    echo "  i2s:       Ternary + I2_S (8x memory reduction, FP16 inference)"
    echo "  i2s-tuned: Ternary + I2_S + Optimized FlashInfer (RECOMMENDED)"
    echo "  i2s-kvfp8: Ternary + I2_S + FP8 KV Cache (Lower memory, faster attention)"
    echo "  i2s-fp8:   Ternary + I2_S + FP8 tensor cores (fastest inference)"
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

# Enable ternary profiling (set to 1 to enable for debugging)
# When enabled, tracks timing for ternary operations and saves to JSON file
export TERNARY_ENABLE_PROFILING="${TERNARY_ENABLE_PROFILING:-0}"
export TERNARY_PROFILE_OUTPUT="${TERNARY_PROFILE_OUTPUT:-/tmp/ternary_profile.json}"

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

# Log the command for debugging
echo "Running command:"
echo "$CMD"
echo "========================================"

# Run the server
eval "$CMD"
