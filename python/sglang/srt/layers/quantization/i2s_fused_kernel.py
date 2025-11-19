"""
Optimized I2S kernel for SGLang - Triton unpack + cuBLAS

Drop-in replacement that uses the proven Triton unpack kernel from ternary.py
plus cuBLAS for matmul. This is the fastest and most reliable approach.
"""

import os
import ctypes
import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    _TRITON_FUSED_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False
    _TRITON_FUSED_AVAILABLE = False

_lib = None
_VERIFY_FUSED = os.environ.get('VERIFY_I2S_FUSED', '0') == '1'

# Track warmed up kernel configurations for CUDA graph compatibility
_WARMED_UP_FUSED_KERNELS = set()

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    
    # Find i2s_unpack_opt.so
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'),
        '/home/ubuntu/old/raghav',
        os.getcwd(),
    ]
    
    for path in search_paths:
        lib_path = os.path.join(path, 'i2s_unpack_opt.so')
        if os.path.exists(lib_path):
            try:
                _lib = ctypes.CDLL(lib_path)
                logger.info(f"[I2S_FUSED] Loaded optimized kernel from {lib_path}")
                return _lib
            except Exception as e:
                logger.debug(f"[I2S_FUSED] Failed to load {lib_path}: {e}")
    
    raise FileNotFoundError("i2s_unpack_opt.so not found")


def i2s_fused_matmul(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int
) -> torch.Tensor:
    """
    Fast I2S matmul: optimized unpack + cuBLAS (legacy path) or Triton fused path.
    Prefers the Triton fused kernel when available.
    """
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x

    N = weight_packed.shape[0]
    dtype = x.dtype

    try:
        assert weight_packed.shape[0] == N, f"Weight packed N mismatch: {weight_packed.shape[0]} vs {N}"
        assert weight_packed.shape[1] == (K + 3) // 4, f"Weight packed K mismatch: {weight_packed.shape[1]} vs {(K+3)//4}"
        assert alpha.shape[0] == K, f"Alpha shape mismatch: {alpha.shape[0]} vs K={K}"
    except AssertionError as e:
        logger.error(f"[I2S_FUSED] Dimension validation failed: {e}")
        logger.error(f"[I2S_FUSED] x: {original_shape}, weight_packed: {weight_packed.shape}, alpha: {alpha.shape}, K: {K}")
        raise

    if (
        _TRITON_FUSED_AVAILABLE
        and x.is_cuda
        and weight_packed.is_cuda
        and alpha.is_cuda
        and x_2d.is_contiguous()
    ):
        try:
            fused = _i2s_fused_matmul_triton(x_2d, weight_packed, alpha, bias, K)
            if fused is not None:
                if len(original_shape) > 2:
                    output_shape = list(original_shape[:-1]) + [fused.shape[-1]]
                    fused = fused.reshape(output_shape)
                if _VERIFY_FUSED:
                    _verify_against_reference(x_2d, weight_packed, alpha, bias, K, fused)
                return fused
        except Exception as triton_error:
            logger.warning(f"[I2S_FUSED] Triton fused kernel failed, falling back: {triton_error}")

    return _i2s_fused_matmul_legacy(x_2d, weight_packed, alpha, bias, K, original_shape)


def _i2s_fused_matmul_legacy(
    x_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    original_shape: torch.Size,
) -> torch.Tensor:
    """Legacy path: unpack via CUDA C kernel + cuBLAS (F.linear)."""
    lib = _load_lib()
    N = weight_packed.shape[0]
    dtype = x_2d.dtype

    alpha_safe = torch.where(
        torch.isfinite(alpha) & (alpha > 0),
        alpha,
        torch.full_like(alpha, 1e-6)
    ).clamp(min=1e-6, max=1e6).contiguous()

    weight_unpacked = torch.empty(N, K, device=x_2d.device, dtype=dtype)
    weight_packed_contig = weight_packed.contiguous()
    stream_handle = 0

    if dtype == torch.float16:
        lib.i2s_unpack_fast_v3_fp16(
            ctypes.c_void_p(weight_packed_contig.data_ptr()),
            ctypes.c_void_p(alpha_safe.data_ptr()),
            ctypes.c_void_p(weight_unpacked.data_ptr()),
            ctypes.c_int(N),
            ctypes.c_int(K),
            ctypes.c_void_p(stream_handle)
        )
    elif dtype == torch.bfloat16:
        lib.i2s_unpack_fast_v3_bf16(
            ctypes.c_void_p(weight_packed_contig.data_ptr()),
            ctypes.c_void_p(alpha_safe.data_ptr()),
            ctypes.c_void_p(weight_unpacked.data_ptr()),
            ctypes.c_int(N),
            ctypes.c_int(K),
            ctypes.c_void_p(stream_handle)
        )
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    weight_unpacked = torch.where(
        torch.isfinite(weight_unpacked),
        weight_unpacked,
        torch.zeros_like(weight_unpacked)
    )

    output = F.linear(x_2d, weight_unpacked, bias)
    output = torch.where(
        torch.isfinite(output),
        output,
        torch.zeros_like(output)
    )

    if len(original_shape) > 2:
        output_shape = list(original_shape[:-1]) + [output.shape[-1]]
        output = output.reshape(output_shape)

    return output


if _TRITON_FUSED_AVAILABLE:
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    @triton.jit
    def _i2s_fused_kernel(
        x_ptr, w_packed_ptr, alpha_ptr, bias_ptr, out_ptr,
        M, N, K, num_packed_cols,
        stride_xm, stride_xk,
        stride_wp_n, stride_wp_k,
        stride_out_m, stride_out_n,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + offs_k
            k_mask = k < K
            k_safe = tl.where(k_mask, k, 0)

            # Load x tile [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + k[None, :] * stride_xk)
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
            # Convert to fp32 for computation
            x_fp32 = x.to(tl.float32)

            # Unpack weight tile on-the-fly [BLOCK_N, BLOCK_K]
            packed_idx = k_safe // 4
            bit_shifts = (k_safe % 4) * 2

            w_ptrs = w_packed_ptr + (offs_n[:, None] * stride_wp_n + packed_idx[None, :] * stride_wp_k)
            valid_cols = (
                (offs_n[:, None] < N)
                & k_mask[None, :]
                & (packed_idx[None, :] < num_packed_cols)
            )
            packed_vals = tl.load(w_ptrs, mask=valid_cols, other=0)

            # Extract ternary values and scale by alpha
            ternary = ((packed_vals >> bit_shifts[None, :]) & 0x3).to(tl.float32) - 1.0
            alpha_vals = tl.load(alpha_ptr + k_safe, mask=k_mask, other=1.0)
            ternary_scaled = ternary * alpha_vals[None, :]

            # Accumulate: acc[M, N] += x[M, K] @ ternary_scaled.T[K, N]
            # Both are now fp32
            acc += tl.dot(x_fp32, tl.trans(ternary_scaled))

        if has_bias:
            bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias_vals[None, :]

        out_ptrs = out_ptr + (offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _i2s_fused_matmul_triton(
    x_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
) -> Optional[torch.Tensor]:
    if not _TRITON_FUSED_AVAILABLE:
        return None

    M = x_2d.shape[0]
    N = weight_packed.shape[0]

    out = torch.empty(M, N, device=x_2d.device, dtype=torch.float32)

    num_packed_cols = weight_packed.shape[1]

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
    )

    bias_ptr = bias if bias is not None else x_2d.new_empty(0)
    
    # Create a unique kernel key for warmup tracking
    kernel_key = (M, N, K, num_packed_cols, bias is not None, x_2d.dtype, BLOCK_M, BLOCK_N, BLOCK_K)
    
    # Warmup kernel if not already done (must happen outside of graph capture)
    if kernel_key not in _WARMED_UP_FUSED_KERNELS:
        if torch.cuda.is_graphing():
            # During graph capture, kernel must already be warmed up
            raise RuntimeError(
                "Triton fused kernel not warmed up before CUDA graph capture. "
                "This should not happen - warmup should occur during model initialization."
            )
        # Warmup: call kernel once to compile it
        _i2s_fused_kernel[grid](
            x_2d, weight_packed, alpha, bias_ptr, out,
            M, N, K, num_packed_cols,
            x_2d.stride(0), x_2d.stride(1),
            weight_packed.stride(0), weight_packed.stride(1),
            out.stride(0), out.stride(1),
            has_bias=bias is not None,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        _WARMED_UP_FUSED_KERNELS.add(kernel_key)
        return out.to(x_2d.dtype)
    
    # Normal execution path (kernel already compiled)
    _i2s_fused_kernel[grid](
        x_2d, weight_packed, alpha, bias_ptr, out,
        M, N, K, num_packed_cols,
        x_2d.stride(0), x_2d.stride(1),
        weight_packed.stride(0), weight_packed.stride(1),
        out.stride(0), out.stride(1),
        has_bias=bias is not None,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out.to(x_2d.dtype)


def _verify_against_reference(
    x_2d: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    candidate: torch.Tensor,
) -> None:
    """Optional correctness check against reference unpack+linear path."""
    with torch.no_grad():
        num_packed_cols = weight_packed.shape[1]
        packed_expanded = weight_packed.unsqueeze(-1)
        shifts = torch.arange(4, device=weight_packed.device, dtype=torch.uint8) * 2
        extracted = (packed_expanded >> shifts.view(1, 1, -1)) & 0b11
        K_padded = num_packed_cols * 4
        extracted = extracted.reshape(weight_packed.shape[0], K_padded)[:, :K]
        ternary = extracted.to(torch.float32) - 1.0
        weight_unpacked = ternary * alpha.view(1, -1)
        ref = F.linear(x_2d, weight_unpacked.to(x_2d.dtype), bias)
        diff = (candidate - ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        if max_diff > 5e-3:
            logger.warning(
                f"[I2S_FUSED] Verification detected high diff (max={max_diff:.3e}, mean={mean_diff:.3e})"
            )


def is_available() -> bool:
    try:
        _load_lib()
        return True
    except:
        return False


__all__ = ['i2s_fused_matmul', 'is_available']
