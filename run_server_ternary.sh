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

if [[ -n "${SGLANG_PYTHON:-}" ]]; then
    PYTHON_BIN="${SGLANG_PYTHON}"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: no Python interpreter found (set SGLANG_PYTHON)"
    exit 1
fi

# Default values
QUANT_MODE="i2s"
TP_SIZE=1
DP_SIZE_OVERRIDE=""
SERVER_PORT_OVERRIDE=""
DEFAULT_MANGROVE_MODEL_PATH="${MANGROVE_MODEL_PATH:-/scratch/mangrove}"
if [[ -d "$DEFAULT_MANGROVE_MODEL_PATH" ]]; then
    MODEL_PRESET="mangrove"
else
    MODEL_PRESET="qwen3-moe"
fi
MODEL_PATH_OVERRIDE=""
MODEL_NAME_OVERRIDE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --dp)
            DP_SIZE_OVERRIDE="$2"
            shift 2
            ;;
        --port)
            SERVER_PORT_OVERRIDE="$2"
            shift 2
            ;;
        --model)
            MODEL_PRESET="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --served-model-name)
            MODEL_NAME_OVERRIDE="$2"
            shift 2
            ;;
    fp16|i2s|i2s-tuned|i2s-maxspeed|i2s-tuned-dp2|i2s-kvfp8|i2s-fp8|i2s-fp8-full|ternary)
            QUANT_MODE="$1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [MODE] [--tp N] [--dp N] [--port P] [--model MODEL] [--model-path PATH] [--served-model-name NAME]"
            echo ""
            echo "Modes:"
            echo "  fp16:           Pure FP16/BF16 (no quantization, baseline)"
            echo "  i2s:            Ternary + I2_S (8x memory reduction, FP16 inference)"
            echo "  ternary:        Ternary-only (no FP16 fallback; i2s/BitNet only)"
            echo "  i2s-tuned:      Ternary + I2_S + Optimized FlashInfer (RECOMMENDED)"
            echo "  i2s-maxspeed:   Ternary + I2_S + Speculative decode (maximize C=1 and mid-load TPS)"
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
            echo "  --dp N       Data parallel replicas across N GPUs (for max throughput)"
            echo "  --port P     Server port (default: 30080)"
            echo "  --model M    Model preset: qwen3-moe, klear-20b, mangrove"
            echo "  --model-path PATH    Explicit model path (overrides --model preset)"
            echo "  --served-model-name NAME  Explicit served model name"
            echo ""
            echo "Models:"
            echo "  qwen3-moe:  Qwen3 MoE 30B A3B (/mnt/data/30ba3b)"
            echo "  klear-20b:  Klear 20B MoE (/home/ubuntu/raghav/20ba1b)"
            echo "  mangrove:   Production model path ($DEFAULT_MANGROVE_MODEL_PATH)"
            echo ""
            echo "Examples:"
            echo "  $0 i2s --tp 2              # Run Qwen3 MoE with i2s mode on 2 GPUs (TP)"
            echo "  $0 i2s-tuned-dp2           # Run with DP=2 (2x throughput, recommended)"
            echo "  $0 i2s-tuned --dp 4        # Run 4 data-parallel replicas"
            echo "  $0 i2s-maxspeed            # High-speed ternary profile (speculative decode)"
            echo "  $0 fp16 --model klear-20b  # Run Klear 20B in fp16 mode"
            echo "  $0 i2s-tuned --model klear-20b  # Run Klear 20B with tuned i2s"
            echo "  $0 i2s-tuned --model-path /scratch/mangrove"
            echo "  $0 i2s-fp8 --model mangrove --served-model-name mangrove-prod-fp8"
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
if [[ -n "$DP_SIZE_OVERRIDE" ]] && ! [[ "$DP_SIZE_OVERRIDE" =~ ^[1-8]$ ]]; then
    echo "Error: Invalid data parallel size '$DP_SIZE_OVERRIDE' (must be 1-8)"
    exit 1
fi
if [[ -n "$SERVER_PORT_OVERRIDE" ]] && ! [[ "$SERVER_PORT_OVERRIDE" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid port '$SERVER_PORT_OVERRIDE'"
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

# Resolve the CUTLASS i2s library path.
# Preference order: explicit env > repo-local > monorepo third_party > in-tree > sibling.
if [[ "$QUANT_MODE" == i2s* || "$QUANT_MODE" == "ternary" ]]; then
    resolve_i2s_cutlass_lib() {
        local candidates=(
            "$SCRIPT_DIR/libternary_cutlass_sm100.so"
            "$SCRIPT_DIR/third_party/ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so"
            "$SCRIPT_DIR/ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so"
            "$SCRIPT_DIR/../ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so"
            "${TERNARY_ROOT:-}/mangrove-turbo/libternary_cutlass_sm100.so"
            "/home/ubuntu/ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so"
        )
        local lib_path
        for lib_path in "${candidates[@]}"; do
            if [[ -n "$lib_path" && -f "$lib_path" ]]; then
                echo "$lib_path"
                return 0
            fi
        done
        return 1
    }

    I2S_CUTLASS_LIB_DEFAULT="$(resolve_i2s_cutlass_lib || true)"
    if [[ -n "$I2S_CUTLASS_LIB_DEFAULT" ]]; then
        export SGLANG_I2S_CUTLASS_LIB="${SGLANG_I2S_CUTLASS_LIB:-$I2S_CUTLASS_LIB_DEFAULT}"
    fi
fi

get_primary_gpu_compute_major() {
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r' || true)"
    if [[ "$cc" =~ ^([0-9]+)\.[0-9]+$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    echo ""
}

check_dp_router_ready() {
    local check_output
    if ! check_output="$("$PYTHON_BIN" - <<'PY'
import sys

try:
    import sglang_router  # noqa: F401
except Exception as e:
    print(f"missing_sglang_router:{e}")
    raise SystemExit(2)

try:
    from sglang_router.router import Router  # noqa: F401
except Exception as e:
    print(f"missing_rust_router:{e}")
    raise SystemExit(3)

print("ok")
PY
)" ; then
        echo "Error: data-parallel (--dp) launch requires sgl-router with Rust router support."
        echo "Probe output: ${check_output}"
        echo "How to fix:"
        echo "  1) Install Rust toolchain (rustc/cargo)."
        echo "  2) Build/install sgl-router with Rust extension (wheel build from sgl-router/)."
        echo "  3) Re-run with --dp N."
        echo "Fallback:"
        echo "  - Run without --dp for single-instance benchmarks."
        exit 1
    fi
}

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

# Optional kernel-path/fallback runtime stats for ternary debugging.
# Enable with SGLANG_TERNARY_COLLECT_PATH_STATS=1.
if [[ "${SGLANG_TERNARY_COLLECT_PATH_STATS:-0}" == "1" && -z "${SGLANG_TERNARY_FALLBACK_REPORT:-}" ]]; then
    export SGLANG_TERNARY_FALLBACK_REPORT="$SCRIPT_DIR/logs/ternary_runtime_{pid}.json"
fi

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
        export SGLANG_TERNARY_USE_I2S_CUTLASS="${SGLANG_TERNARY_USE_I2S_CUTLASS:-1}"
        QUANT_FLAG="--quantization ternary --attention-backend flashinfer"
        QUANT_DESC="Ternary + I2_S (8x memory reduction, FP16 inference)"
        ;;
    
    ternary)
        # Ternary-only mode: allow int2/i2s kernels but forbid FP16 fallback.
        GPU_COMPUTE_MAJOR="$(get_primary_gpu_compute_major)"
        if [[ "${SGLANG_TERNARY_ALLOW_STRICT_NON_SM100:-0}" != "1" ]]; then
            if [[ -n "$GPU_COMPUTE_MAJOR" && "$GPU_COMPUTE_MAJOR" -lt 10 ]]; then
                echo "Error: strict ternary-only mode is not currently supported on this GPU (compute capability ${GPU_COMPUTE_MAJOR}.x)."
                echo "Reason: M>1 strict ternary kernels are unavailable here while FP16 fallback is disabled."
                echo "Use 'i2s' or 'i2s-tuned' for production, or set SGLANG_TERNARY_ALLOW_STRICT_NON_SM100=1 to bypass this guard for debugging."
                exit 1
            fi
        fi
        export SGLANG_TERNARY_I2S_ONLY=1
        export SGLANG_TERNARY_MOE_TRITON="${SGLANG_TERNARY_MOE_TRITON:-1}"
        export SGLANG_TERNARY_USE_I2S_CUTLASS="${SGLANG_TERNARY_USE_I2S_CUTLASS:-1}"
        export SGLANG_TERNARY_DROP_FP16_WEIGHTS="${SGLANG_TERNARY_DROP_FP16_WEIGHTS:-1}"
        export SGLANG_TERNARY_USE_FP8=0
        export SGLANG_TERNARY_FP8_BRIDGE=0
        QUANT_FLAG="--quantization ternary --attention-backend flashinfer"
        QUANT_DESC="Ternary-only (no FP16 fallback; i2s/BitNet only)"
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

    i2s-maxspeed)
        # Ternary profile optimized for very high effective decode throughput.
        # Uses:
        # - tuned FlashInfer attention
        # - full MoE quantization path (not decode-only)
        # - NGRAM speculative decoding
        export SGLANG_FLASHINFER_USE_TENSOR_CORE=true
        export SGLANG_FLASHINFER_WORKSPACE_SIZE=$((1024*1024*1024))  # 1GB
        export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE=4096
        export TERNARY_MOE_DECODE_ONLY="${TERNARY_MOE_DECODE_ONLY_MAXSPEED:-0}"
        export SGLANG_TERNARY_MOE_TRITON="${SGLANG_TERNARY_MOE_TRITON:-1}"
        SPECULATIVE_ALGORITHM="${SPECULATIVE_ALGORITHM:-NGRAM}"
        SPECULATIVE_NUM_DRAFT="${SPECULATIVE_NUM_DRAFT:-4}"
        QUANT_FLAG="--quantization ternary --attention-backend flashinfer"
        QUANT_DESC="Ternary + I2_S + MaxSpeed profile (speculative decode + full MoE quant)"
        echo "Note: MaxSpeed ternary profile enabled (NGRAM speculative decode, draft=$SPECULATIVE_NUM_DRAFT, MoE Triton=${SGLANG_TERNARY_MOE_TRITON})"
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
        # Split-KV tuning knobs (now effective even without deterministic inference)
        export SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE="${SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE:-8192}"
        export SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE="${SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE:-4096}"
        # Reduce FlashInfer decode begin_forward overhead
        export SGLANG_FLASHINFER_USE_FAST_DECODE_PLAN="${SGLANG_FLASHINFER_USE_FAST_DECODE_PLAN:-true}"
        # Heuristic: for small decode batches, split-KV overhead can dominate
        export SGLANG_FLASHINFER_DECODE_DISABLE_SPLIT_KV_BELOW_BS="${SGLANG_FLASHINFER_DECODE_DISABLE_SPLIT_KV_BELOW_BS:-8}"
        
        QUANT_FLAG="--quantization ternary --kv-cache-dtype fp8_e4m3"
        QUANT_DESC="Ternary + I2_S + FP8 KV cache + Tuned FlashInfer"
        echo "Note: FP8 mode enabled:"
        echo "  - FP8 KV cache: enabled (fp8_e4m3)"
        echo "  - FlashInfer: Tensor cores + 1GB workspace"
        echo "  - FlashInfer: split tiles prefill=$SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE decode=$SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE"
        echo "  - FlashInfer: fast_decode_plan=$SGLANG_FLASHINFER_USE_FAST_DECODE_PLAN"
        echo "  - FlashInfer: disable_split_kv_below_bs=$SGLANG_FLASHINFER_DECODE_DISABLE_SPLIT_KV_BELOW_BS"
        echo "  - FP8 bridge: $SGLANG_TERNARY_FP8_BRIDGE (0=DP4A, 1=FP8 GEMM)"
        ;;
    
    i2s-fp8-full)
        # EXPERIMENTAL: Full FP8 pipeline with trtllm_mha attention (2x faster on Blackwell)
        export SGLANG_TERNARY_USE_FP8=1
        export SGLANG_TERNARY_FP8_GRANULARITY="${SGLANG_TERNARY_FP8_GRANULARITY:-per_token_group_128}"
        export SGLANG_TERNARY_FP8_BRIDGE=1
        export SGLANG_TERNARY_FP8_BRIDGE_MIN_M="${SGLANG_TERNARY_FP8_BRIDGE_MIN_M:-64}"
        
        # FP8 sticky hidden state: keep hidden states in FP8 between layers
        # (last layer outputs BF16 for final norm compatibility)
        export SGLANG_TERNARY_FP8_STICKY="${SGLANG_TERNARY_FP8_STICKY:-true}"
        
        # Use trtllm_mha attention backend for 2x faster FP8 attention on Blackwell
        # This backend uses trtllm-gen kernels with FP8 Q + FP8 KV cache
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-trtllm_mha}"
        
        # Decode-only RMSNorm+QKV fusion currently forces BF16 path in qwen3_moe.py.
        # That defeats FP8-first (it prevents FP8 RMSNorm + FP8 ternary linears).
        # Keep it OFF by default for FP8-full unless explicitly enabled.
        export TERNARY_FUSE_RMSNORM_QKV="${TERNARY_FUSE_RMSNORM_QKV:-0}"
        
        QUANT_FLAG="--quantization ternary --kv-cache-dtype fp8_e4m3"
        QUANT_DESC="Ternary + I2_S + FP8 Full Pipeline (trtllm_mha)"
        echo "Note: FP8-full mode enabled with trtllm_mha attention:"
        echo "  - FP8 KV cache: enabled (fp8_e4m3)"
        echo "  - FP8 sticky hidden: $SGLANG_TERNARY_FP8_STICKY (last layer outputs BF16)"
        echo "  - Attention backend: trtllm_mha (2x faster FP8 Q + FP8 KV on Blackwell)"
        echo "  - RMSNorm+QKV fusion: $TERNARY_FUSE_RMSNORM_QKV (0=FP8-first, 1=force BF16 fused kernel)"
        echo "  - FP8 bridge: ENABLED for PREFILL ONLY (M >= $SGLANG_TERNARY_FP8_BRIDGE_MIN_M)"
        ;;
esac

# Allow explicit --dp override for any mode.
if [[ -n "$DP_SIZE_OVERRIDE" ]]; then
    if [[ "$DP_SIZE_OVERRIDE" -gt 1 ]]; then
        USE_DATA_PARALLEL=1
        DP_SIZE="$DP_SIZE_OVERRIDE"
    else
        USE_DATA_PARALLEL=0
        DP_SIZE=1
    fi
fi

if [[ "${USE_DATA_PARALLEL:-0}" == "1" && "${TP_SIZE:-1}" != "1" ]]; then
    echo "Warning: --dp and --tp were both set. In this launcher, data-parallel mode ignores --tp and runs full-model replicas."
fi

# Model and server configuration
if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
    MODEL_ID="$MODEL_PATH_OVERRIDE"
    if [[ -n "$MODEL_NAME_OVERRIDE" ]]; then
        MODEL_NAME="$MODEL_NAME_OVERRIDE"
    else
        MODEL_BASENAME="$(basename "$MODEL_ID")"
        MODEL_BASENAME="${MODEL_BASENAME// /-}"
        MODEL_NAME="${MODEL_BASENAME}-${QUANT_MODE}"
    fi
else
    case "$MODEL_PRESET" in
        qwen3-moe)
            MODEL_ID="/mnt/data/30ba3b"
            MODEL_NAME="qwen3-moe-$QUANT_MODE"
            ;;
        klear-20b)
            MODEL_ID="/home/ubuntu/raghav/20ba1b"
            MODEL_NAME="klear-20b-$QUANT_MODE"
            ;;
        mangrove)
            MODEL_ID="$DEFAULT_MANGROVE_MODEL_PATH"
            MODEL_NAME="mangrove-$QUANT_MODE"
            ;;
        *)
            echo "Error: Unknown model preset '$MODEL_PRESET'"
            echo "Valid presets: qwen3-moe, klear-20b, mangrove"
            exit 1
            ;;
    esac
    if [[ -n "$MODEL_NAME_OVERRIDE" ]]; then
        MODEL_NAME="$MODEL_NAME_OVERRIDE"
    fi
fi

if [[ ! -e "$MODEL_ID" ]]; then
    echo "Error: Model path does not exist: $MODEL_ID"
    exit 1
fi
if [[ -n "$MODEL_PATH_OVERRIDE" ]]; then
    MODEL_PRESET_DISPLAY="custom (--model-path)"
else
    MODEL_PRESET_DISPLAY="$MODEL_PRESET"
fi
# Escape user-provided model id/name safely because command is executed via eval.
MODEL_ID_ESCAPED="$(printf '%q' "$MODEL_ID")"
MODEL_NAME_ESCAPED="$(printf '%q' "$MODEL_NAME")"
SERVER_PORT=30080
if [[ -n "$SERVER_PORT_OVERRIDE" ]]; then
    SERVER_PORT="$SERVER_PORT_OVERRIDE"
fi
DEFAULT_REQUEST_GPU_MEMORY="0.85"
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
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-128}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"  # Test: disable CUDA graphs entirely
# MoE runner backend: auto, deep_gemm, triton, flashinfer_cutlass, flashinfer_mxfp4, cutlass
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-}"
# Attention backend: flashinfer, trtllm_mha (2x faster with FP8 on Blackwell)
# Default to flashinfer; i2s-fp8-full mode sets trtllm_mha
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
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
echo "Model preset: $MODEL_PRESET_DISPLAY"
echo "Model path: $MODEL_ID"
echo "Port: $SERVER_PORT"
if [[ "${USE_DATA_PARALLEL:-0}" == "1" ]]; then
    echo "Data Parallelism: ${DP_SIZE:-2} GPU(s) (load-balanced replicas)"
else
    echo "Tensor Parallelism: $TP_SIZE GPU(s)"
fi
echo "GPU memory allocation: $REQUEST_GPU_MEMORY"
echo "Ternary log level: $SGLANG_TERNARY_LOG_LEVEL"
echo "I2S CUTLASS lib: ${SGLANG_I2S_CUTLASS_LIB:-unset}"
echo "Fuse RMSNorm+QKV (decode,tp=1): $TERNARY_FUSE_RMSNORM_QKV"
echo "Fuse RMSNorm+QKV allow during CUDA graph capture: $TERNARY_FUSE_RMSNORM_QKV_ALLOW_CAPTURE"
echo "MoE decode-only (ternary MoE only on decode): $TERNARY_MOE_DECODE_ONLY"
echo "Batched MoE max tokens: $TERNARY_BATCHED_MOE_MAX_TOKENS (0=disabled)"
echo "Batched MoE debug logging: $TERNARY_BATCHED_MOE_DEBUG"
echo "Batched ternary MoE (Triton): ${SGLANG_TERNARY_MOE_TRITON:-0}"
if [[ "${SGLANG_TERNARY_I2S_ONLY:-0}" == "1" && "${SGLANG_TERNARY_DROP_FP16_WEIGHTS:-0}" == "1" && "${SGLANG_TERNARY_MOE_TRITON:-0}" != "1" ]]; then
    echo "WARNING: Ternary-only + drop-FP16 is enabled but SGLANG_TERNARY_MOE_TRITON=0."
    echo "         M>1 MoE will fall back to FP16 and fail during CUDA graph capture."
fi
echo "Quantize LM head (TERNARY_QUANTIZE_LM_HEAD): $TERNARY_QUANTIZE_LM_HEAD"
echo "SGLANG_PROFILE_OPLEVEL (1 disables cuda graphs/overlap/compile): $SGLANG_PROFILE_OPLEVEL"
echo "TORCH_COMPILE_DISABLE: ${TORCH_COMPILE_DISABLE:-0}  TORCHINDUCTOR_DISABLE: ${TORCHINDUCTOR_DISABLE:-0}"
echo "Pinned output copy: $SGLANG_ENABLE_PINNED_OUTPUT_COPY (max_bytes=$SGLANG_PINNED_OUTPUT_COPY_MAX_BYTES)"
echo "Ternary-only (no FP16 fallback): ${SGLANG_TERNARY_I2S_ONLY:-0}"
echo "Drop FP16 ternary weights: ${SGLANG_TERNARY_DROP_FP16_WEIGHTS:-0}"
if [[ "$TERNARY_ENABLE_PROFILING" == "1" ]]; then
    echo "Ternary profiling: ENABLED (output: $TERNARY_PROFILE_OUTPUT)"
else
    echo "Ternary profiling: DISABLED"
fi
if [[ "${SGLANG_TERNARY_COLLECT_PATH_STATS:-0}" == "1" ]]; then
    echo "Ternary path/fallback stats: ENABLED (report: ${SGLANG_TERNARY_FALLBACK_REPORT:-unset})"
else
    echo "Ternary path/fallback stats: DISABLED"
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
echo "Attention backend: $ATTENTION_BACKEND"
echo "========================================"

# Check if server is already running on the port
if lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Server already running on port $SERVER_PORT. Killing existing process..."
    lsof -Pi :$SERVER_PORT -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Build the command
if [[ "${USE_DATA_PARALLEL:-0}" == "1" ]]; then
    check_dp_router_ready
    # Data Parallel mode: use router to load-balance across GPUs
    CMD="$PYTHON_BIN -m sglang_router.launch_server \
        --model-path $MODEL_ID_ESCAPED \
        --served-model-name $MODEL_NAME_ESCAPED \
        --port $SERVER_PORT \
        --dp ${DP_SIZE:-2} \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --trust-remote-code \
        --attention-backend $ATTENTION_BACKEND"
else
    # Standard mode: single process with optional tensor parallelism
    CMD="$PYTHON_BIN -m sglang.launch_server \
        --model-path $MODEL_ID_ESCAPED \
        --served-model-name $MODEL_NAME_ESCAPED \
        --port $SERVER_PORT \
        --tp-size $TP_SIZE \
        --mem-fraction-static $REQUEST_GPU_MEMORY \
        --enable-p2p-check \
        --trust-remote-code \
        --attention-backend $ATTENTION_BACKEND"
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

# Note: attention backend is set in the base command using $ATTENTION_BACKEND

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
