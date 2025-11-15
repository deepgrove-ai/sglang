"""Fused I2S unpack+matmul kernels for ternary quantization.

This module provides optimized kernels that unpack I2S weights on-the-fly
during matrix multiplication, avoiding the need to materialize the full unpacked
weight matrix.

The primary implementation uses CUTLASS for optimal performance. Falls back
to unfused unpack + matmul if CUTLASS is not available.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Try to import CUTLASS kernel
CUTLASS_AVAILABLE = False
CUTLASS_IMPORT_ERROR = None
try:
    from sgl_kernel.quantization import i2s_cutlass_matmul_wrapper
    CUTLASS_AVAILABLE = True
    CUTLASS_IMPORT_ERROR = None
    logger.info("[I2S] CUTLASS kernel is available")
except ImportError as e:
    CUTLASS_AVAILABLE = False
    CUTLASS_IMPORT_ERROR = str(e)
    logger.warning(f"[I2S] CUTLASS kernel not available (ImportError): {e}")
    # Try to see if the C++ function exists
    try:
        import sgl_kernel
        if hasattr(sgl_kernel, 'i2s_cutlass_matmul'):
            logger.warning("[I2S] C++ function exists but wrapper not available - check sgl-kernel.quantization.i2s")
        else:
            logger.warning("[I2S] C++ function i2s_cutlass_matmul not found in sgl_kernel - kernel may need rebuilding")
    except Exception:
        pass
except Exception as e:
    CUTLASS_AVAILABLE = False
    CUTLASS_IMPORT_ERROR = str(e)
    logger.warning(f"[I2S] CUTLASS kernel import failed: {e}")


def i2s_fused_matmul(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
) -> torch.Tensor:
    """
    Fused I2S unpack + matmul with automatic kernel selection.
    
    Tries CUTLASS kernel first (if available), then falls back to unpack + matmul.
    The unfused approach uses optimized Triton unpacking followed by PyTorch's
    optimized GEMM, which is faster than a naive fused Triton kernel.
    """
    # Try CUTLASS kernel first
    if CUTLASS_AVAILABLE:
        try:
            logger.info("[I2S] Using CUTLASS kernel for fused unpack+matmul")
            result = i2s_cutlass_matmul_wrapper(x, weight_packed, alpha, bias, K)
            logger.debug("[I2S] CUTLASS kernel completed successfully")
            return result
        except Exception as e:
            logger.warning(f"[I2S] CUTLASS kernel failed, using fallback: {e}")
            import traceback
            logger.debug(f"[I2S] CUTLASS kernel traceback: {traceback.format_exc()}")
    
    # Fallback: unpack then matmul
    # This uses the optimized Triton unpack kernel + PyTorch's optimized GEMM
    logger.debug("[I2S] Using fallback: unpack + matmul (no fusion)")
    from sglang.srt.layers.quantization.ternary import unpack_i2s_weights
    weight_unpacked = unpack_i2s_weights(weight_packed, K, alpha, x.dtype)
    return F.linear(x, weight_unpacked, bias)


def check_cutlass_compatible(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    min_compute_capability: int = 80,
) -> bool:
    """Check if tensors are compatible with CUTLASS kernel."""
    if not torch.cuda.is_available():
        return False
    
    # Check compute capability
    device = x.device
    if device.type != 'cuda':
        return False
    
    compute_capability = torch.cuda.get_device_capability(device.index)
    compute_version = compute_capability[0] * 10 + compute_capability[1]
    if compute_version < min_compute_capability:
        return False
    
    # Check tensor shapes are reasonable for tensor cores
    M, K = x.shape
    N = weight_packed.shape[0]
    
    # Tensor cores work best with certain alignments
    # For now, accept all shapes (can add more restrictions later)
    return True

