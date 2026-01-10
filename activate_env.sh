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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Memory allocation configuration
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Ternary kernel configuration
export TERNARY_PREFILL_SKIP_M="${TERNARY_PREFILL_SKIP_M:-8}"

# Optional: Enable CUDA activation quantizer (faster)
# export TERNARY_USE_CUDA_ACT_QUANT=1

echo "SGLang environment activated"
echo "  PYTHONPATH includes: $SCRIPT_DIR/python"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

