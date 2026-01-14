# Copyright 2023-2024 SGLang Team
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
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging.version import Version

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_npu,
    is_xpu,
    supports_custom_op,
)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

# FP8 RMSNorm support (lazy-loaded)
_fp8_rmsnorm_checked = False
_fp8_rmsnorm_available = False

def _check_fp8_rmsnorm_available():
    """Lazy check for FP8 RMSNorm availability."""
    global _fp8_rmsnorm_checked, _fp8_rmsnorm_available
    if _fp8_rmsnorm_checked:
        return _fp8_rmsnorm_available
    _fp8_rmsnorm_checked = True
    try:
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import is_fp8_rmsnorm_available
        _fp8_rmsnorm_available = is_fp8_rmsnorm_available()
    except Exception:
        _fp8_rmsnorm_available = False
    return _fp8_rmsnorm_available

if _is_cuda or _is_xpu:
    # if _is_flashinfer_available:
    #     from flashinfer.norm import fused_add_rmsnorm
    # else:
    from sgl_kernel import (
        fused_add_rmsnorm,
        gemma_fused_add_rmsnorm,
        gemma_rmsnorm,
        rmsnorm,
    )
if _use_aiter:
    from aiter import rmsnorm2d_fwd as rms_norm
    from aiter import rmsnorm2d_fwd_with_add as fused_add_rms_norm
elif _is_hip:
    import vllm
    from vllm._custom_ops import fused_add_rms_norm, rms_norm

    _vllm_version = Version(vllm.__version__)

logger = logging.getLogger(__name__)

if _is_npu:
    import torch_npu


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        if _use_aiter:
            self._forward_method = self.forward_aiter
        if get_bool_env_var("SGLANG_ENABLE_DETERMINISTIC_INFERENCE"):
            self._forward_method = self.forward_native

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        
        # FP8 path: handle FP8 inputs with attached scale
        if x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            return self._forward_cuda_fp8(x, residual)
        
        if residual is not None:
            if not x.is_contiguous():
                x = x.contiguous()
            if not residual.is_contiguous():
                residual = residual.contiguous()
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        if not x.is_contiguous():
            x = x.contiguous()
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out
    
    def _forward_cuda_fp8(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Handle FP8 input with per-token scale.
        
        When FP8 RMSNorm is available, keeps hidden states in FP8 throughout.
        This enables FP8-first pipeline where downstream kernels (ternary, attention)
        can directly consume FP8, avoiding repeated dtype conversions.
        
        Trade-off: RMSNorm kernel is ~2x slower, but total pipeline can be faster
        due to reduced conversions and memory bandwidth.
        """
        x_scale = getattr(x, '_fp8_scale', None)
        
        # Use FP8 kernel if available and we have proper scale
        use_fp8_kernel = (
            _check_fp8_rmsnorm_available()
            and x_scale is not None
            and residual is not None
            and residual.dtype == torch.bfloat16
        )
        
        if use_fp8_kernel:
            from sglang.srt.layers.fused_add_rmsnorm_fp8 import fused_add_rmsnorm_fp8
            if not x.is_contiguous():
                x = x.contiguous()
            if not residual.is_contiguous():
                residual = residual.contiguous()
            
            # Ensure 2D for kernel
            orig_shape = x.shape
            x_2d = x.view(-1, self.hidden_size)
            residual_2d = residual.view(-1, self.hidden_size)
            x_scale_flat = x_scale.view(-1)
            
            # Run FP8 kernel: FP8 in → FP32 accumulation → FP8 out
            out_fp8, out_scale = fused_add_rmsnorm_fp8(
                x_2d, x_scale_flat, residual_2d, self.weight.data,
                self.variance_epsilon, inplace=True
            )
            
            # Reshape back
            out_fp8 = out_fp8.view(orig_shape)
            
            # Attach scale to output for downstream FP8 consumers
            out_fp8._fp8_scale = out_scale
            
            return out_fp8, residual
        
        # Fallback: dequantize FP8 to BF16 and use standard kernel
        if x_scale is not None:
            x_bf16 = (x.to(torch.float32) * x_scale.view(-1, 1)).to(torch.bfloat16)
        else:
            x_bf16 = x.to(torch.bfloat16)
        
        if residual is not None:
            if not x_bf16.is_contiguous():
                x_bf16 = x_bf16.contiguous()
            if not residual.is_contiguous():
                residual = residual.contiguous()
            fused_add_rmsnorm(x_bf16, residual, self.weight.data, self.variance_epsilon)
            return x_bf16, residual
        
        if not x_bf16.is_contiguous():
            x_bf16 = x_bf16.contiguous()
        out = rmsnorm(x_bf16, self.weight.data, self.variance_epsilon)
        return out

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            out, _, residual_out = torch_npu.npu_add_rms_norm(
                residual, x, self.weight.data, self.variance_epsilon
            )
            return out, residual_out
        return torch_npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]

    def forward_aiter(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            residual_out = torch.empty_like(x)
            output = torch.empty_like(x)
            fused_add_rms_norm(
                output,
                x,
                residual,
                residual_out,
                self.weight.data,
                self.variance_epsilon,
            )
            return output, residual_out
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    def forward_hip(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            # NOTE: Remove this if aiter kernel supports discontinuous input
            x = x.contiguous()
        if residual is not None:
            if _vllm_version < Version("0.9"):
                fused_add_rms_norm(x, residual, self.weight.data, self.variance_epsilon)
                return x, residual
            else:
                residual_out = torch.empty_like(x)
                output = torch.empty_like(x)
                fused_add_rms_norm(
                    output,
                    x,
                    residual_out,
                    residual,
                    self.weight.data,
                    self.variance_epsilon,
                )
                return output, residual_out
        out = torch.empty_like(x)
        rms_norm(out, x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight).to(orig_dtype)
        if residual is None:
            return x
        else:
            return x, residual

    def forward_cpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if _is_cpu_amx_available:
            if residual is not None:
                torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                    x, residual, self.weight.data, self.variance_epsilon
                )
                return x, residual
            return torch.ops.sgl_kernel.rmsnorm_cpu(
                x, self.weight.data, self.variance_epsilon
            )
        else:
            return self.forward_native(x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        if residual is not None:
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """
        if residual is not None:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size
            from sglang.srt.layers.flashinfer_comm_fusion import (
                flashinfer_allreduce_residual_rmsnorm,
            )

            fused_op = (
                torch.ops.sglang.flashinfer_allreduce_residual_rmsnorm
                if supports_custom_op()
                else flashinfer_allreduce_residual_rmsnorm
            )

            if get_tensor_model_parallel_world_size() > 1:
                fused_result = fused_op(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    eps=self.variance_epsilon,
                )
                if fused_result[0] is not None:
                    return fused_result

        return self.forward(x, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

        # Re-dispatch
        if _is_hip:
            self._forward_method = self.forward_native

    def _forward_impl(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
            return x, residual
        out = gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual)

    def forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x = x + residual
            residual = x

        x, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)
        return x if residual is None else (x, residual)

    def forward_xpu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._forward_impl(x, residual)


class Gemma3RMSNorm(CustomOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def forward_cuda(self, x):
        return self.forward_native(x)

    def forward_npu(self, x):
        output, _ = torch_npu.npu_gemma_rms_norm(x, self.weight, self.eps)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


if not (
    _is_cuda or _is_hip or _is_npu or (_is_cpu and _is_cpu_amx_available) or _is_xpu
):
    logger.info(
        "sgl-kernel layernorm implementation is not available on current platform. Fallback to other kernel libraries."
    )
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm  # noqa: F401
