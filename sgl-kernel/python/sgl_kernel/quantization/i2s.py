"""I2S (Int2 Super-packed) quantization kernels."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    from sgl_kernel import i2s_cutlass_matmul

    CUTLASS_AVAILABLE = True
    logger.info("[I2S] Successfully imported i2s_cutlass_matmul from sgl_kernel")
except ImportError as e:
    CUTLASS_AVAILABLE = False
    logger.warning(f"[I2S] Failed to import i2s_cutlass_matmul from sgl_kernel: {e}")
except Exception as e:
    CUTLASS_AVAILABLE = False
    logger.warning(f"[I2S] Error importing i2s_cutlass_matmul: {e}")


def i2s_cutlass_matmul_wrapper(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
) -> torch.Tensor:
    """
    CUTLASS-based fused I2S unpack + matmul kernel.
    
    Args:
        x: Input tensor (M, K) - float16/bfloat16
        weight_packed: Packed weights (N, num_packed_cols) - uint8
        alpha: Per-column alpha scales (K,) - float32
        bias: Optional bias (N,) - same dtype as x
        K: Original K dimension (before packing)
    
    Returns:
        Output tensor (M, N) - same dtype as x
    """
    if not CUTLASS_AVAILABLE:
        error_msg = "CUTLASS I2S kernel not available"
        logger.error(f"[I2S] {error_msg}")
        raise RuntimeError(error_msg)
    
    M = x.shape[0]
    N = weight_packed.shape[0]
    device = x.device
    
    logger.debug(f"[I2S] CUTLASS kernel: M={M}, N={N}, K={K}, dtype={x.dtype}")
    
    # Allocate output tensor
    out = torch.empty((M, N), dtype=x.dtype, device=device)
    
    # Handle optional bias
    if bias is None:
        # Create dummy bias tensor (will be ignored by kernel)
        bias = torch.zeros(N, dtype=x.dtype, device=device)
    
    # Call CUTLASS kernel
    try:
        i2s_cutlass_matmul(out, x, weight_packed, alpha, bias, K)
        logger.debug("[I2S] CUTLASS kernel execution successful")
    except Exception as e:
        logger.error(f"[I2S] CUTLASS kernel execution failed: {e}")
        raise
    
    return out

