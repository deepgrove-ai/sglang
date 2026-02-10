#!/bin/bash
# Activation script for SGLang with ternary quantization
# This script sets up the environment for running ternary-quantized models

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Add sglang to Python path (development mode)
export PYTHONPATH="$SCRIPT_DIR/python:${PYTHONPATH:-}"

# CUDA configuration
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DETECTED_GPU_IDS="$(
            nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
                | tr '\n' ',' | sed 's/,$//'
        )"
        if [[ -n "$DETECTED_GPU_IDS" ]]; then
            export CUDA_VISIBLE_DEVICES="$DETECTED_GPU_IDS"
        else
            export CUDA_VISIBLE_DEVICES="0"
        fi
    else
        export CUDA_VISIBLE_DEVICES="0"
    fi
fi

# Ensure nvcc is available for JIT paths (e.g., flashinfer kernel compilation).
if ! command -v nvcc >/dev/null 2>&1; then
    for CUDA_BIN in /usr/local/cuda/bin /usr/local/cuda-12/bin /usr/local/cuda-12.6/bin; do
        if [ -x "$CUDA_BIN/nvcc" ]; then
            export PATH="$CUDA_BIN:$PATH"
            export CUDA_HOME="$(dirname "$CUDA_BIN")"
            break
        fi
    done
fi

# Memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Ternary kernel configuration
export TERNARY_PREFILL_SKIP_M="${TERNARY_PREFILL_SKIP_M:-8}"

# Optional: Enable CUDA activation quantizer (faster)
# export TERNARY_USE_CUDA_ACT_QUANT=1

echo "SGLang environment activated"
echo "  PYTHONPATH includes: $SCRIPT_DIR/python"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

