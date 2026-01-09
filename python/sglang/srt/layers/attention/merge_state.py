from typing import Optional, Tuple

import torch
from sgl_kernel import merge_state_v2

from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


# Automatically fallback to the Triton kernel in some cases
# (e.g., for AMD GPUs, when the head dimension is not a multiple
# of 4 or 8, and in FP8 precision)
def _supported_dtypes(o: torch.Tensor) -> bool:
    return o.dtype in [torch.float32, torch.half, torch.bfloat16]


def _supported_headdim(o: torch.Tensor) -> bool:
    headdim = o.shape[2]  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    if o.dtype == torch.float32:
        return headdim % 4 == 0
    return headdim % 8 == 0


def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Merge two attention states (output, lse) into one.

    Parameters
    ----------
    prefix_output : torch.Tensor
        The prefix attention output, shape: ``(num_tokens, num_heads, head_dim)``.
    prefix_lse : torch.Tensor
        The prefix log-sum-exp values, shape: ``(num_tokens, num_heads)``.
    suffix_output : torch.Tensor
        The suffix attention output, shape: ``(num_tokens, num_heads, head_dim)``.
    suffix_lse : torch.Tensor
        The suffix log-sum-exp values, shape: ``(num_tokens, num_heads)``.
    output : Optional[torch.Tensor]
        The merged attention output, if specified, the kernel will update this tensor inplace.
    output_lse : Optional[torch.Tensor]
        The merged log-sum-exp values, if specified, the kernel will update this tensor inplace.
    enable_pdl : Optional[bool]
        Whether to enable programmatic dependent launch for kernel chaining.
        If None, will be automatically enabled on Hopper architecture.

    Returns
    -------
    output : torch.Tensor
        The merged attention output, shape: ``(num_tokens, num_heads, head_dim)``.
    output_lse : torch.Tensor
        The merged log-sum-exp values, shape: ``(num_tokens, num_heads)``.
    """
    if (
        _is_cuda
        and _supported_dtypes(prefix_output)
        and _supported_headdim(prefix_output)
    ):
        try:
            return merge_state_v2(
                prefix_output,
                prefix_lse,
                suffix_output,
                suffix_lse,
                output,
                output_lse,
                enable_pdl,
            )
        except TypeError:
            return merge_state_v2(
                prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
            )
    else:
        # Fallback to Triton kernel
        return merge_state_triton(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
