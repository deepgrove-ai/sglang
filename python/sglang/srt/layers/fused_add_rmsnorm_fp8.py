# Copyright 2024-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
FP8-aware Fused Add + RMSNorm kernel for FP8-first inference pipelines.

This module provides a Triton kernel that:
1. Takes FP8 input with per-token scale
2. Dequantizes to FP32 for numerical accuracy
3. Adds to BF16 residual (residual is updated in-place)
4. Computes RMSNorm in FP32
5. Quantizes output back to FP8 with per-token scale

This is critical for M=1 decode performance in FP8-first ternary pipelines,
as it avoids the FP8→BF16→FP8 round-trip overhead at every layer boundary.

Contract:
    Input:  x (FP8), x_scale (per-token float32), residual (BF16), weight (BF16/FP32)
    Output: x_out (FP8), out_scale (per-token float32), residual (BF16, updated in-place)
"""

import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_hip = is_hip()

# FP8 constants
FP8_E4M3_MAX = 448.0  # Max value for float8_e4m3fn
FP8_DTYPE = torch.float8_e4m3fn

# Flag to track kernel availability
_FP8_RMSNORM_AVAILABLE = False


@triton.jit
def _fused_add_rmsnorm_fp8_kernel(
    # Input pointers
    x_fp8_ptr,          # FP8 input tensor
    x_scale_ptr,        # Per-token input scale (float32)
    residual_ptr,       # BF16 residual tensor (updated in-place)
    weight_ptr,         # RMSNorm weight
    # Output pointers
    out_fp8_ptr,        # FP8 output tensor
    out_scale_ptr,      # Per-token output scale (float32)
    # Dimensions
    hidden_dim,
    # Constants
    eps: tl.constexpr,
    FP8_MAX: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel: dequant(x_fp8) + residual → rmsnorm → quant to FP8
    
    All computation is done in FP32 for numerical accuracy.
    Residual is updated in-place: residual = residual + dequant(x_fp8)
    """
    # Each program handles one row (one token)
    row_idx = tl.program_id(axis=0)
    
    # Compute offsets for this row
    row_start = row_idx * hidden_dim
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_dim
    
    # Load input scale for this token
    x_scale = tl.load(x_scale_ptr + row_idx)
    
    # Load FP8 input and dequantize to FP32
    x_fp8 = tl.load(x_fp8_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x_f32 = x_fp8.to(tl.float32) * x_scale
    
    # Load residual (BF16 → FP32)
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    residual_f32 = residual.to(tl.float32)
    
    # Add: new_residual = residual + dequant(x)
    new_residual_f32 = residual_f32 + x_f32
    
    # Store updated residual (FP32 → BF16)
    tl.store(residual_ptr + row_start + col_offsets, new_residual_f32.to(tl.bfloat16), mask=mask)
    
    # RMSNorm computation in FP32
    # variance = mean(x^2)
    squared = new_residual_f32 * new_residual_f32
    variance = tl.sum(squared, axis=0) / hidden_dim
    
    # rsqrt(variance + eps)
    inv_rms = tl.rsqrt(variance + eps)
    
    # Normalize
    normalized = new_residual_f32 * inv_rms
    
    # Load weight and apply
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    weight_f32 = weight.to(tl.float32)
    output_f32 = normalized * weight_f32
    
    # Quantize to FP8 with per-token scale
    # Compute scale: max_abs / FP8_MAX
    abs_max = tl.max(tl.abs(output_f32), axis=0)
    # Clamp to avoid division by zero
    out_scale = tl.maximum(abs_max / FP8_MAX, 1e-12)
    
    # Scale and clamp
    output_scaled = output_f32 / out_scale
    output_clamped = tl.minimum(tl.maximum(output_scaled, -FP8_MAX), FP8_MAX)
    
    # Store FP8 output
    # Note: Triton will cast to the output tensor's dtype (FP8)
    tl.store(out_fp8_ptr + row_start + col_offsets, output_clamped.to(tl.float8e4nv), mask=mask)
    
    # Store output scale
    tl.store(out_scale_ptr + row_idx, out_scale)


@triton.jit
def _fused_add_rmsnorm_fp8_inplace_kernel(
    # Input/Output pointers (x is modified in-place)
    x_fp8_ptr,          # FP8 input/output tensor (modified in-place)
    x_scale_ptr,        # Per-token scale (input scale, then output scale)
    residual_ptr,       # BF16 residual tensor (updated in-place)
    weight_ptr,         # RMSNorm weight
    # Dimensions
    hidden_dim,
    # Constants
    eps: tl.constexpr,
    FP8_MAX: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    In-place variant: uses same buffer for input and output.
    More memory efficient, CUDA-graph friendly.
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * hidden_dim
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_dim
    
    # Load input scale
    x_scale = tl.load(x_scale_ptr + row_idx)
    
    # Load and dequantize FP8 input
    x_fp8 = tl.load(x_fp8_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x_f32 = x_fp8.to(tl.float32) * x_scale
    
    # Load and add residual
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    new_residual_f32 = residual.to(tl.float32) + x_f32
    
    # Store updated residual
    tl.store(residual_ptr + row_start + col_offsets, new_residual_f32.to(tl.bfloat16), mask=mask)
    
    # RMSNorm
    variance = tl.sum(new_residual_f32 * new_residual_f32, axis=0) / hidden_dim
    inv_rms = tl.rsqrt(variance + eps)
    normalized = new_residual_f32 * inv_rms
    
    # Apply weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output_f32 = normalized * weight.to(tl.float32)
    
    # Quantize to FP8
    abs_max = tl.max(tl.abs(output_f32), axis=0)
    out_scale = tl.maximum(abs_max / FP8_MAX, 1e-12)
    output_scaled = output_f32 / out_scale
    output_clamped = tl.minimum(tl.maximum(output_scaled, -FP8_MAX), FP8_MAX)
    
    # Store FP8 output (in-place) and scale
    tl.store(x_fp8_ptr + row_start + col_offsets, output_clamped.to(tl.float8e4nv), mask=mask)
    tl.store(x_scale_ptr + row_idx, out_scale)




def fused_add_rmsnorm_fp8(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8-aware fused add + RMSNorm.
    
    Semantics:
        1. residual = residual + dequant(x_fp8, x_scale)  (in-place)
        2. output = rmsnorm(residual) * weight
        3. output_fp8, out_scale = quant_fp8(output)
    
    Args:
        x_fp8: FP8 input tensor, shape (batch_size, hidden_dim)
        x_scale: Per-token input scale, shape (batch_size,) or (batch_size, 1)
        residual: BF16 residual tensor, shape (batch_size, hidden_dim) - modified in-place
        weight: RMSNorm weight, shape (hidden_dim,)
        eps: Epsilon for numerical stability
        inplace: If True, reuse x_fp8 buffer for output (more efficient)
    
    Returns:
        (output_fp8, out_scale): FP8 output and per-token scales
    """
    assert x_fp8.dtype == FP8_DTYPE, f"Expected FP8 input, got {x_fp8.dtype}"
    assert residual.dtype == torch.bfloat16, f"Expected BF16 residual, got {residual.dtype}"
    assert x_fp8.shape == residual.shape, f"Shape mismatch: {x_fp8.shape} vs {residual.shape}"
    
    batch_size, hidden_dim = x_fp8.shape
    
    # Ensure contiguous
    if not x_fp8.is_contiguous():
        x_fp8 = x_fp8.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    
    # Flatten scale if needed
    x_scale_flat = x_scale.view(-1) if x_scale.dim() > 1 else x_scale
    assert x_scale_flat.shape[0] == batch_size
    
    # Try optimized hybrid path: Triton dequant+add+rmsnorm → sgl_kernel quant
    # This is faster because sgl_kernel's CUDA quantization is 7x faster than Triton
    if _is_cuda and _USE_HYBRID_FP8_RMSNORM:
        return _fused_add_rmsnorm_fp8_hybrid(
            x_fp8, x_scale_flat, residual, weight, eps, inplace
        )
    
    # Fallback to pure Triton implementation
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4)
    
    if inplace:
        out_fp8 = x_fp8
        out_scale = x_scale_flat.contiguous()
        
        _fused_add_rmsnorm_fp8_inplace_kernel[(batch_size,)](
            out_fp8,
            out_scale,
            residual,
            weight,
            hidden_dim,
            eps=eps,
            FP8_MAX=FP8_E4M3_MAX,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        out_fp8 = torch.empty_like(x_fp8)
        out_scale = torch.empty(batch_size, device=x_fp8.device, dtype=torch.float32)
        
        _fused_add_rmsnorm_fp8_kernel[(batch_size,)](
            x_fp8,
            x_scale_flat,
            residual,
            weight,
            out_fp8,
            out_scale,
            hidden_dim,
            eps=eps,
            FP8_MAX=FP8_E4M3_MAX,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    
    return out_fp8, out_scale


# Hybrid implementation (Triton + sgl_kernel quant) is slower due to extra memory traffic.
# Pure Triton with fused quantization is faster (21µs vs 27µs).
# Keep hybrid code available for future optimization but disabled by default.
_USE_HYBRID_FP8_RMSNORM = False
_sgl_per_token_quant_fp8 = None

try:
    from sgl_kernel import sgl_per_token_quant_fp8 as _sgl_per_token_quant_fp8
    # Hybrid disabled - pure Triton is faster due to better memory locality
    # _USE_HYBRID_FP8_RMSNORM = True
except ImportError:
    pass


@triton.jit
def _fused_dequant_add_rmsnorm_kernel(
    # Input pointers
    x_fp8_ptr,          # FP8 input
    x_scale_ptr,        # Per-token input scale
    residual_ptr,       # BF16 residual (updated in-place)
    weight_ptr,         # RMSNorm weight
    # Output pointer
    out_bf16_ptr,       # BF16 output (NOT FP8 - quantization done separately)
    # Dimensions
    hidden_dim,
    # Constants
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused dequant + add + rmsnorm, outputting BF16.
    Quantization to FP8 is done separately using fast sgl_kernel op.
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * hidden_dim
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_dim
    
    # Load and dequantize FP8 input
    x_scale = tl.load(x_scale_ptr + row_idx)
    x_fp8 = tl.load(x_fp8_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x_f32 = x_fp8.to(tl.float32) * x_scale
    
    # Load residual and add
    residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
    new_residual_f32 = residual.to(tl.float32) + x_f32
    
    # Store updated residual
    tl.store(residual_ptr + row_start + col_offsets, new_residual_f32.to(tl.bfloat16), mask=mask)
    
    # RMSNorm in FP32
    variance = tl.sum(new_residual_f32 * new_residual_f32, axis=0) / hidden_dim
    inv_rms = tl.rsqrt(variance + eps)
    normalized = new_residual_f32 * inv_rms
    
    # Apply weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output_f32 = normalized * weight.to(tl.float32)
    
    # Store BF16 output (quantization done by fast sgl_kernel op)
    tl.store(out_bf16_ptr + row_start + col_offsets, output_f32.to(tl.bfloat16), mask=mask)


def _fused_add_rmsnorm_fp8_hybrid(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    inplace: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid implementation: Triton for compute, sgl_kernel for fast FP8 quantization.
    
    This is faster because sgl_kernel's CUDA per_token_quant_fp8 is ~7x faster
    than Triton's quantization (4µs vs 30µs).
    """
    batch_size, hidden_dim = x_fp8.shape
    
    # Kernel config
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4)
    
    # Allocate BF16 intermediate buffer
    out_bf16 = torch.empty(batch_size, hidden_dim, device=x_fp8.device, dtype=torch.bfloat16)
    
    # Step 1: Fused dequant + add + rmsnorm → BF16
    _fused_dequant_add_rmsnorm_kernel[(batch_size,)](
        x_fp8,
        x_scale,
        residual,
        weight,
        out_bf16,
        hidden_dim,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    # Step 2: Fast FP8 quantization using sgl_kernel
    if inplace:
        out_fp8 = x_fp8  # Reuse buffer
        out_scale = x_scale.contiguous()
    else:
        out_fp8 = torch.empty_like(x_fp8)
        out_scale = torch.empty(batch_size, device=x_fp8.device, dtype=torch.float32)
    
    _sgl_per_token_quant_fp8(out_bf16, out_fp8, out_scale)
    
    return out_fp8, out_scale


def fused_add_rmsnorm_fp8_with_bf16_output(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    FP8 input → dequant + residual add → rmsnorm → BF16 output.
    
    Use this when the next operation needs BF16 (e.g., final layer before lm_head).
    Residual is still updated in-place.
    
    Returns:
        output_bf16: BF16 output tensor
    """
    assert x_fp8.dtype == FP8_DTYPE, f"Expected FP8 input, got {x_fp8.dtype}"
    assert residual.dtype == torch.bfloat16, f"Expected BF16 residual, got {residual.dtype}"
    
    batch_size, hidden_dim = x_fp8.shape
    
    # Flatten scale
    x_scale_flat = x_scale.view(-1) if x_scale.dim() > 1 else x_scale
    
    # Dequantize
    x_f32 = x_fp8.to(torch.float32) * x_scale_flat.view(-1, 1)
    
    # Add to residual (in-place)
    residual.add_(x_f32.to(torch.bfloat16))
    
    # RMSNorm in FP32
    residual_f32 = residual.to(torch.float32)
    variance = residual_f32.pow(2).mean(dim=-1, keepdim=True)
    normalized = residual_f32 * torch.rsqrt(variance + eps)
    
    # Apply weight
    output = (normalized * weight.to(torch.float32)).to(torch.bfloat16)
    
    return output


@triton.jit
def _rmsnorm_bf16_to_fp8_kernel(
    # Input pointers
    x_ptr,              # BF16 input tensor
    residual_ptr,       # BF16 residual tensor (can be None via flag)
    weight_ptr,         # RMSNorm weight
    # Output pointers
    out_fp8_ptr,        # FP8 output tensor
    out_scale_ptr,      # Per-token output scale (float32)
    # Dimensions
    hidden_dim,
    # Flags
    has_residual: tl.constexpr,
    # Constants
    eps: tl.constexpr,
    FP8_MAX: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm with optional residual add, outputting FP8.
    
    If has_residual:
        residual = residual + x  (in-place)
        output = rmsnorm(residual)
    Else:
        output = rmsnorm(x)
    
    Output is quantized to FP8 with per-token scale.
    """
    row_idx = tl.program_id(axis=0)
    row_start = row_idx * hidden_dim
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < hidden_dim
    
    # Load input
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    
    if has_residual:
        # Load residual, add, store back
        residual = tl.load(residual_ptr + row_start + col_offsets, mask=mask, other=0.0)
        norm_input = residual.to(tl.float32) + x_f32
        tl.store(residual_ptr + row_start + col_offsets, norm_input.to(tl.bfloat16), mask=mask)
    else:
        norm_input = x_f32
    
    # RMSNorm
    variance = tl.sum(norm_input * norm_input, axis=0) / hidden_dim
    inv_rms = tl.rsqrt(variance + eps)
    normalized = norm_input * inv_rms
    
    # Apply weight
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
    output_f32 = normalized * weight.to(tl.float32)
    
    # Quantize to FP8
    abs_max = tl.max(tl.abs(output_f32), axis=0)
    out_scale = tl.maximum(abs_max / FP8_MAX, 1e-12)
    output_scaled = output_f32 / out_scale
    output_clamped = tl.minimum(tl.maximum(output_scaled, -FP8_MAX), FP8_MAX)
    
    # Store
    tl.store(out_fp8_ptr + row_start + col_offsets, output_clamped.to(tl.float8e4nv), mask=mask)
    tl.store(out_scale_ptr + row_idx, out_scale)


def rmsnorm_to_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    RMSNorm with BF16 input, outputting FP8.
    
    This is used when entering an FP8 pipeline from BF16 (e.g., after embedding).
    
    Args:
        x: BF16 input tensor, shape (batch_size, hidden_dim)
        weight: RMSNorm weight, shape (hidden_dim,)
        residual: Optional BF16 residual tensor - if provided, updated in-place
        eps: Epsilon for numerical stability
    
    Returns:
        (output_fp8, out_scale, residual): FP8 output, per-token scales, updated residual
    """
    assert x.dtype == torch.bfloat16, f"Expected BF16 input, got {x.dtype}"
    
    batch_size, hidden_dim = x.shape
    has_residual = residual is not None
    
    if has_residual:
        assert residual.dtype == torch.bfloat16
        assert residual.shape == x.shape
    
    # Ensure contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Allocate outputs
    out_fp8 = torch.empty(batch_size, hidden_dim, device=x.device, dtype=FP8_DTYPE)
    out_scale = torch.empty(batch_size, device=x.device, dtype=torch.float32)
    
    # Kernel config
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4)
    
    # Use a dummy pointer if no residual
    residual_ptr = residual if has_residual else x  # x is just a placeholder
    
    _rmsnorm_bf16_to_fp8_kernel[(batch_size,)](
        x,
        residual_ptr,
        weight,
        out_fp8,
        out_scale,
        hidden_dim,
        has_residual=has_residual,
        eps=eps,
        FP8_MAX=FP8_E4M3_MAX,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return out_fp8, out_scale, residual


def is_fp8_rmsnorm_available() -> bool:
    """Check if FP8 RMSNorm kernels are available."""
    global _FP8_RMSNORM_AVAILABLE
    
    if _FP8_RMSNORM_AVAILABLE:
        return True
    
    # Try to compile a small test kernel
    if not (_is_cuda or _is_hip):
        return False
    
    try:
        # Small test to verify Triton FP8 support
        test_x = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
        test_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        test_x_fp8 = test_x.to(FP8_DTYPE)
        test_residual = torch.randn(1, 64, device="cuda", dtype=torch.bfloat16)
        test_weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
        
        # Run a test
        out_fp8, out_scale = fused_add_rmsnorm_fp8(
            test_x_fp8, test_scale, test_residual, test_weight, inplace=False
        )
        
        # Verify output
        assert out_fp8.dtype == FP8_DTYPE
        assert out_scale.dtype == torch.float32
        
        _FP8_RMSNORM_AVAILABLE = True
        logger.info("[FP8 RMSNorm] Triton FP8 fused add+rmsnorm kernel available")
        return True
        
    except Exception as e:
        logger.warning(f"[FP8 RMSNorm] Kernel not available: {e}")
        return False


# Convenience wrapper that handles scale attachment
def fused_add_rmsnorm_fp8_with_scale_attach(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-level wrapper for FP8 fused add + rmsnorm.
    
    Handles:
    - Extracting _fp8_scale attribute from input
    - Attaching _fp8_scale attribute to output
    - Falling back to BF16 path if input is not FP8
    
    Args:
        x: Input tensor (FP8 with _fp8_scale attribute, or BF16)
        residual: BF16 residual tensor (modified in-place)
        weight: RMSNorm weight
        eps: Epsilon
    
    Returns:
        (output, residual): Output tensor (FP8 or BF16), updated residual
    """
    if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # FP8 path
        x_scale = getattr(x, '_fp8_scale', None)
        if x_scale is None:
            # No scale attached, fall back to BF16
            x_bf16 = x.to(torch.bfloat16)
            # Use standard path (would need to import from layernorm)
            from sgl_kernel import fused_add_rmsnorm
            fused_add_rmsnorm(x_bf16, residual, weight, eps)
            return x_bf16, residual
        
        # Run FP8 kernel
        out_fp8, out_scale = fused_add_rmsnorm_fp8(
            x, x_scale, residual, weight, eps, inplace=True
        )
        
        # Attach scale to output
        out_fp8._fp8_scale = out_scale
        
        return out_fp8, residual
    else:
        # BF16 path - use standard kernel
        from sgl_kernel import fused_add_rmsnorm
        fused_add_rmsnorm(x, residual, weight, eps)
        return x, residual
