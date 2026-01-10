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
MODEL_PRESET="qwen3-moe"  # Default model preset

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL_PRESET="$2"
            shift 2
            ;;
        fp16|i2s|i2s-tuned|i2s-tuned-dp2|i2s-kvfp8|i2s-fp8|i2s-fp8-full)
            QUANT_MODE="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [MODE] [--tp N] [--model MODEL]"
            echo ""
            echo "Modes:"
            echo "  fp16:           Pure FP16/BF16 (no quantization, baseline)"
            echo "  i2s:            Ternary + I2_S (8x memory reduction, FP16 inference)"
            echo "  i2s-tuned:      Ternary + I2_S + Optimized FlashInfer (RECOMMENDED)"
            echo "  i2s-tuned-dp2:  i2s-tuned + Data Parallel on 2 GPUs (2x throughput)"
            echo "  i2s-kvfp8:      Ternary + I2_S + FP8 KV Cache (memory optimized, ~30% slower)"
            echo "  i2s-fp8:        Ternary + I2_S + FP8 policy (FP8 KV + FlashInfer tuning)"
            echo "  i2s-fp8-full:   EXPERIMENTAL: i2s-fp8 + FP8 bridge for prefill GEMM"
            echo ""
            echo "NOTE: FP8 KV cache has ~30% attention overhead (FlashInfer issue #1753)"
            echo "      Use 'i2s-tuned' for best throughput, FP8 modes only if memory-constrained"
            echo ""
            echo "Options:"
            echo "  --tp N       Tensor parallelism: shard model across N GPUs (default: 1)"
            echo "  --model M    Model preset: qwen3-moe (default), klear-20b"
            echo ""
            echo "Models:"
            echo "  qwen3-moe:  Qwen3 MoE 30B A3B (RedMod/sft_dist_l_q)"
            echo "  klear-20b:  Klear 20B MoE (/home/ubuntu/raghav/20ba1b)"
            echo ""
            echo "Examples:"
            echo "  $0 i2s --tp 2              # Run Qwen3 MoE with i2s mode on 2 GPUs (TP)"
            echo "  $0 i2s-tuned-dp2           # Run with DP=2 (2x throughput, recommended)"
            echo "  $0 fp16 --model klear-20b  # Run Klear 20B in fp16 mode"
            echo "  $0 i2s-tuned --model klear-20b  # Run Klear 20B with tuned i2s"
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

# Optional: quantize LM head (vocab projection) to ternary for decode speed.
# Can affect output quality; enable only for experiments.
export TERNARY_QUANTIZE_LM_HEAD="${TERNARY_QUANTIZE_LM_HEAD:-1}"

# Decode fusion knobs (safe defaults; guarded in code by tp=1 + decode + supported shapes)
# - 1: fuse input RMSNorm(+residual update) into ternary QKV for decode (removes per-layer RMSNorm kernel)
# - 0: disable and use standard RMSNorm path
export TERNARY_FUSE_RMSNORM_QKV="${TERNARY_FUSE_RMSNORM_QKV:-1}"
export TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE="${TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE:-1}"

# PDL (Programmatic Dependent Launch) for Blackwell/Hopper
# Enables hardware-accelerated kernel chaining for lower latency (M=1)
export TERNARY_ENABLE_PDL="${TERNARY_ENABLE_PDL:-1}"

# Overlap/scheduler knob: use pinned host memory for true async GPU->CPU copies of small tensors.
# This is critical for avoiding the per-step ~3ms stall seen as aten::to/copy_ + cudaMemcpyAsync.
export SGLANG_ENABLE_PINNED_OUTPUT_COPY="${SGLANG_ENABLE_PINNED_OUTPUT_COPY:-1}"
export SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES="${SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES:-1048576}"

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
    
    i2s-tuned-dp2)
        # Ternary + I2S + Tuned FlashInfer + Data Parallel on 2 GPUs
        # Each GPU runs full model, requests load-balanced between them
        # 2x throughput, same latency per request
        USE_DATA_PARALLEL=1
        DP_SIZE=2
        export SGLANG_FLASHINFER_USE_TENSOR_CORE=true
        export SGLANG_FLASHINFER_WORKSPACE_SIZE=$((1024*1024*1024))  # 1GB
        export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=4096
        QUANT_FLAG="--quantization ternary --attention-backend flashinfer"
        QUANT_DESC="Ternary + I2_S + Tuned FlashInfer + DP=2 (2x throughput)"
        echo "Note: Data Parallel mode - 2 GPUs, each with full model copy"
        echo "Note: FlashInfer optimizations enabled (Tensor Cores, 1GB workspace, 4K tiles)"
        ;;
    
    i2s-kvfp8)
        # Ternary + I2S + FP8 KV Cache
        # Ternary weights for linear layers, FP8 for Attention KV cache
        # NOTE: FP8 KV cache has ~30% attention overhead (FlashInfer issue #1753)
        # Use this mode ONLY if memory-constrained (long context / large batch)
        QUANT_FLAG="--quantization ternary --kv-cache-dtype fp8_e5m2"
        QUANT_DESC="Ternary + I2_S + FP8 KV Cache (Memory optimized, slower attention)"
        echo "WARNING: FP8 KV cache has ~30% attention overhead vs BF16 KV cache"
        echo "  - Use 'i2s-tuned' for best throughput"
        echo "  - Use this mode only if memory-constrained"
        ;;

    i2s-fp8)
        # Ternary + I2S + FP8 mode (fastest inference)
        # Enable FP8 tensor core acceleration via environment variable
        # This enables the FP8-first policy for ternary models:
        # - FP8 KV cache for lower memory + FP8 attention compute
        # - FlashInfer tuning for FP8 attention kernels
        # - DP4A kernels for ternary GEMMs (FP8 bridge disabled by default)
        export SGLANG_TERNARY_USE_FP8=1
        export SGLANG_TERNARY_FP8_GRANULARITY="${SGLANG_TERNARY_FP8_GRANULARITY:-per_token_group_128}"
        export SGLANG_TERNARY_FP8_BRIDGE="${SGLANG_TERNARY_FP8_BRIDGE:-0}"
        
        # FlashInfer tuning for FP8 attention (CRITICAL for performance)
        # FP8 attention benefits from tensor cores and larger workspace
        export SGLANG_FLASHINFER_USE_TENSOR_CORE=true
        export SGLANG_FLASHINFER_WORKSPACE_SIZE=$((1024*1024*1024))  # 1GB
        
        QUANT_FLAG="--quantization ternary --kv-cache-dtype fp8_e4m3"
        QUANT_DESC="Ternary + I2_S + FP8 KV cache + Tuned FlashInfer"
        echo "Note: FP8 mode enabled:"
        echo "  - FP8 KV cache: enabled (fp8_e4m3)"
        echo "  - FlashInfer: Tensor cores + 1GB workspace"
        echo "  - FP8 bridge: $SGLANG_TERNARY_FP8_BRIDGE (0=DP4A, 1=FP8 GEMM)"
        ;;
    
    i2s-fp8-full)
        # EXPERIMENTAL: Full FP8 pipeline with bridge enabled
        # This is for testing the FP8 bridge before custom kernels are available
        # WARNING: Bridge only runs for PREFILL (M >= MIN_M threshold)
        export SGLANG_TERNARY_USE_FP8=1
        export SGLANG_TERNARY_FP8_GRANULARITY="${SGLANG_TERNARY_FP8_GRANULARITY:-per_token_group_128}"
        export SGLANG_TERNARY_FP8_BRIDGE=1
        export SGLANG_TERNARY_FP8_BRIDGE_MIN_M="${SGLANG_TERNARY_FP8_BRIDGE_MIN_M:-64}"
        
        # FlashInfer tuning for FP8 attention (CRITICAL for performance)
        export SGLANG_FLASHINFER_USE_TENSOR_CORE=true
        export SGLANG_FLASHINFER_WORKSPACE_SIZE=$((1024*1024*1024))  # 1GB
        
        QUANT_FLAG="--quantization ternary --kv-cache-dtype fp8_e4m3"
        QUANT_DESC="Ternary + I2_S + FP8 Full Pipeline (EXPERIMENTAL)"
        echo "Note: FP8-full mode enabled (EXPERIMENTAL):"
        echo "  - FP8 KV cache: enabled (fp8_e4m3)"
        echo "  - FlashInfer: Tensor cores + 1GB workspace"
        echo "  - FP8 bridge: ENABLED for PREFILL ONLY (M >= $SGLANG_TERNARY_FP8_BRIDGE_MIN_M)"
        ;;
esac

# Model and server configuration
case "$MODEL_PRESET" in
    qwen3-moe)
        MODEL_ID="/mnt/data/30ba3b"
        MODEL_NAME="qwen3-moe-$QUANT_MODE"
        ;;
    klear-20b)
        MODEL_ID="/home/ubuntu/raghav/20ba1b"
        MODEL_NAME="klear-20b-$QUANT_MODE"
        ;;
    *)
        echo "Error: Unknown model preset '$MODEL_PRESET'"
        echo "Valid presets: qwen3-moe, klear-20b"
        exit 1
        ;;
esac
SERVER_PORT=30080
DEFAULT_REQUEST_GPU_MEMORY="0.95"
REQUEST_GPU_MEMORY="${REQUEST_GPU_MEMORY:-$DEFAULT_REQUEST_GPU_MEMORY}"
# Optional performance knobs (defaults keep SGLang auto-tuning behavior).
# For batch=1 decode-latency experiments, good starting point:
#   CUDA_GRAPH_BS="1" MAX_RUNNING_REQUESTS="1" CHUNKED_PREFILL_SIZE="-1"
# Chunked prefill: smaller chunks = less blocking of decode, but more overhead
# Try 512-2048 for latency optimization (default is usually 8192)
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-1024}"

# Speculative decoding: NGRAM doesn't need a draft model
# Uses n-gram patterns from context to predict tokens
# Enabled by default for lower latency (set to "" to disable)
# SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-NGRAM}"
# SPECULATIVE_NUM_DRAFT="${SPECULATIVE_NUM_DRAFT:-4}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:1}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"  # Test: disable CUDA graphs entirely
# MoE runner backend: auto, deep_gemm, triton, flashinfer_cutlass, flashinfer_mxfp4, cutlass
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
DISABLE_OVERLAP="${DISABLE_OVERLAP:-0}"  # maps to --disable-overlap-schedule
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
# torch.compile: enables torch.compile optimization for reduced Python overhead
# Use TORCH_COMPILE_MAX_BS=1 for batch-1 decode latency optimization
TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-}"

# torch.compile + CustomOp behavior:
# SGLang's default torch.compile path swaps many CustomOps (e.g., RMSNorm, RoPE)
# to their PyTorch-native implementations for traceability, which can be *much*
# slower than fused CUDA kernels on H100. For our ternary stack, preserving the
# original CUDA implementations is usually better (torch.compile will graph-break).
SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE="${SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE:-}"

echo "========================================"
echo "Starting SGLang server"
echo "Mode: $QUANT_DESC"
echo "Model preset: $MODEL_PRESET"
echo "Model path: $MODEL_ID"
echo "Port: $SERVER_PORT"
if [[ "${USE_DATA_PARALLEL:-0}" == "1" ]]; then
    echo "Data Parallelism: ${DP_SIZE:-2} GPU(s) (load-balanced replicas)"
else
    echo "Tensor Parallelism: $TP_SIZE GPU(s)"
fi
echo "GPU memory allocation: $REQUEST_GPU_MEMORY"
echo "Ternary log level: $SGLANG_TERNARY_LOG_LEVEL"
echo "Fuse RMSNorm+QKV (decode,tp=1): $TERNARY_FUSE_RMSNORM_QKV"
echo "Fuse RMSNorm+QKV allow during CUDA graph capture: $TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE"
echo "MoE decode-only (ternary MoE only on decode): $TERNARY_MOE_DECODE_ONLY"
echo "Batched MoE max tokens: $TERNARY_BATCHED_MOE_MAX_TOKENS (0=disabled)"
echo "Batched MoE debug logging: $TERNARY_BATCHED_MOE_DEBUG"
echo "Quantize LM head (TERNARY_QUANTIZE_LM_HEAD): $TERNARY_QUANTIZE_LM_HEAD"
echo "SGLANG_PROFILE_OPLEVEL (1 disables cuda graphs/overlap/compile): $SGLANG_PROFILE_OPLEVEL"
echo "TORCH_COMPILE_DISABLE: ${TORCH_COMPILE_DISABLE:-0}  TORCHINDUCTOR_DISABLE: ${TORCHINDUCTOR_DISABLE:-0}"
echo "Pinned output copy: $SGLANG_ENABLE_PINNED_OUTPUT_COPY (max_bytes=$SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES)"
if [[ "$TERNARY_ENABLE_PROFILING" == "1" ]]; then
    echo "Ternary profiling: ENABLED (output: $TERNARY_PROFILE_OUTPUT)"
else
    echo "Ternary profiling: DISABLED"
fi
echo "Chunked prefill size: ${CHUNKED_PREFILL_SIZE:-auto}"
echo "Speculative algorithm: ${SPECULATIVE_ALGORITHM:-disabled}"
echo "Speculative draft tokens: ${SPECULATIVE_NUM_DRAFT:-N/A}"
echo "MoE runner backend: ${MOE_RUNNER_BACKEND:-auto}"
echo "CUDA graph batch sizes: $CUDA_GRAPH_BS"
echo "CUDA graph max batch size: $CUDA_GRAPH_MAX_BS"
echo "Disable CUDA graphs: $DISABLE_CUDA_GRAPH"
echo "Disable radix cache: $DISABLE_RADIX_CACHE"
echo "Disable overlap schedule: $DISABLE_OVERLAP"
echo "Max running requests: $MAX_RUNNING_REQUESTS"
echo "torch.compile max batch size: ${TORCH_COMPILE_MAX_BS:-disabled}"
echo "torch.compile CustomOp mode: ${SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE:-default(native)}"
echo "========================================"

# Check if server is already running on the port
if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Server already running on port $SERVER_PORT. Killing existing process..."
    lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Build the command
if [[ "${USE_DATA_PARALLEL:-0}" == "1" ]]; then
    # Data Parallel mode: use router to load-balance across GPUs
    CMD="python -m sglang_router.launch_server \
        --model-path $MODEL_ID \
        --served-model-name $MODEL_NAME \
        --port $SERVER_PORT \
        --dp ${DP_SIZE:-2} \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --trust-remote-code \
        --attention-backend flashinfer"
else
    # Standard mode: single process with optional tensor parallelism
    CMD="python -m sglang.launch_server \
        --model-path $MODEL_ID \
        --served-model-name $MODEL_NAME \
        --port $SERVER_PORT \
        --tp-size $TP_SIZE \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --enable-p2p-check \
        --trust-remote-code \
        --attention-backend flashinfer"
fi

# Optional knobs (mostly useful for batch=1 decode latency experiments)
if [[ -n "$CHUNKED_PREFILL_SIZE" ]]; then
    CMD="$CMD --chunked-prefill-size $CHUNKED_PREFILL_SIZE"
fi

# Prefer an explicit capture list; fall back to max-bs if provided.
if [[ -n "$CUDA_GRAPH_BS" ]]; then
    CMD="$CMD --cuda-graph-bs $CUDA_GRAPH_BS"
elif [[ -n "$CUDA_GRAPH_MAX_BS" ]]; then
    CMD="$CMD --cuda-graph-max-bs $CUDA_GRAPH_MAX_BS"
fi

if [[ -n "$MAX_RUNNING_REQUESTS" ]]; then
    CMD="$CMD --max-running-requests $MAX_RUNNING_REQUESTS"
fi

if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
    CMD="$CMD --disable-cuda-graph"
fi

if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
    CMD="$CMD --disable-radix-cache"
fi

if [[ "$DISABLE_OVERLAP" == "1" ]]; then
    CMD="$CMD --disable-overlap-schedule"
fi

# Speculative decoding (NGRAM doesn't need a draft model)
if [[ -n "$SPECULATIVE_ALGORITHM" ]]; then
    CMD="$CMD --speculative-algorithm $SPECULATIVE_ALGORITHM"
    CMD="$CMD --speculative-num-draft-tokens $SPECULATIVE_NUM_DRAFT"
    echo "Speculative decoding: $SPECULATIVE_ALGORITHM with $SPECULATIVE_NUM_DRAFT draft tokens"
fi

# MoE runner backend
if [[ -n "$MOE_RUNNER_BACKEND" ]]; then
    CMD="$CMD --moe-runner-backend $MOE_RUNNER_BACKEND"
fi

# torch.compile optimization (reduces Python overhead significantly for batch-1)
if [[ -n "$TORCH_COMPILE_MAX_BS" ]]; then
    # Default to preserving CUDA CustomOps unless explicitly overridden.
    if [[ -z "$SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE" ]]; then
        export SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE="preserve"
    else
        export SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE="$SGLANG_TORCH_COMPILE_CUSTOM_OP_MODE"
    fi
    CMD="$CMD --enable-torch-compile --torch-compile-max-bs $TORCH_COMPILE_MAX_BS"
fi

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
