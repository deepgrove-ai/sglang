"""Ternary quantization method for SGLang.

This module implements ternary quantization (weights in {-1, 0, 1} × alpha).

Supports two storage modes:
- i2s: 2-bit packed format (8x memory reduction)
- fp16: Direct ternary storage (no compression, for debugging)
"""

import gc
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

# Triton kernel for I2S unpacking (faster than PyTorch operations)
if TRITON_AVAILABLE:
    @triton.jit
    def _i2s_unpack_kernel(
        packed_ptr,
        alpha_ptr,
        output_ptr,
        N,
        K,
        num_packed_cols,
        stride_packed_n,
        stride_packed_k,
        stride_output_n,
        stride_output_k,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """
        Unpack I2S weights using optimized Triton kernel.
        
        Optimizations:
        - Efficient memory access patterns
        - Reduced redundant mask computations
        - Optimized bit extraction
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        n_mask = n_offsets < N
        k_mask = k_offsets < K
        
        n_indices = n_offsets[:, None]
        k_indices = k_offsets[None, :]
        
        packed_k_idx = k_indices // 4
        bit_pos = (k_indices % 4) * 2
        
        packed_offsets = n_indices * stride_packed_n + packed_k_idx * stride_packed_k
        valid_mask = n_mask[:, None] & k_mask[None, :] & (packed_k_idx < num_packed_cols)
        
        packed_bytes = tl.load(
            packed_ptr + packed_offsets,
            mask=valid_mask,
            other=0
        )
        
        extracted = (packed_bytes >> bit_pos) & 0b11
        val_ternary = extracted.to(tl.float32) - 1.0
        
        alpha_vals = tl.load(alpha_ptr + k_offsets, mask=k_mask, other=1.0)
        output_values = val_ternary * alpha_vals[None, :]
        
        output_mask = n_mask[:, None] & k_mask[None, :]
        output_offsets = n_offsets[:, None] * stride_output_n + k_offsets[None, :] * stride_output_k
        
        tl.store(output_ptr + output_offsets, output_values, mask=output_mask)

def get_tensor_memory_bytes(t: torch.Tensor) -> int:
    """Get the memory usage of a tensor in bytes."""
    if t is None or not hasattr(t, 'element_size'):
        return 0
    return t.numel() * t.element_size()


def get_layer_memory_bytes(layer: torch.nn.Module) -> int:
    """Get total memory usage of a layer's tensors."""
    total = 0
    for name, param in layer.named_parameters():
        total += get_tensor_memory_bytes(param)
    for name, buffer in layer.named_buffers():
        total += get_tensor_memory_bytes(buffer)
    return total


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def get_actual_gpu_memory_bytes(device: Optional[torch.device] = None) -> int:
    """Get actual GPU memory allocated (not just tensor sizes).
    
    This measures the real GPU memory usage, accounting for:
    - Actual allocated memory (may be larger than tensor sizes due to alignment)
    - Memory fragmentation
    - CUDA memory pool overhead
    
    Args:
        device: CUDA device to measure. If None, uses current device.
    
    Returns:
        Memory allocated in bytes, or 0 if CUDA not available.
    """
    if not torch.cuda.is_available():
        return 0
    
    if device is None:
        device = torch.cuda.current_device()
    
    return torch.cuda.memory_allocated(device)


def force_cleanup_and_sync(device: Optional[torch.device] = None) -> None:
    """Force cleanup of Python objects and CUDA cache.
    
    This ensures that:
    1. Python garbage collector runs
    2. CUDA cache is cleared
    3. All CUDA operations are synchronized
    
    Args:
        device: CUDA device to sync. If None, uses current device.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def measure_layer_memory_accurate(
    layer: torch.nn.Module,
    device: Optional[torch.device] = None,
    include_gpu_actual: bool = True
) -> Dict[str, Any]:
    """Accurately measure layer memory usage using multiple methods.
    
    Measures both theoretical tensor sizes and actual GPU memory allocation.
    This helps verify that quantization actually reduces memory.
    
    Args:
        layer: The layer to measure
        device: CUDA device. If None, uses current device.
        include_gpu_actual: Whether to measure actual GPU memory (requires CUDA)
    
    Returns:
        Dictionary with memory measurements:
            - param_memory_bytes: Theoretical parameter memory
            - buffer_memory_bytes: Theoretical buffer memory
            - total_theoretical_bytes: Total theoretical memory
            - gpu_allocated_bytes: Actual GPU memory allocated (if CUDA available)
            - gpu_reserved_bytes: GPU memory reserved by CUDA (if CUDA available)
    """
    # Theoretical memory (tensor sizes)
    param_memory = sum(get_tensor_memory_bytes(p) for p in layer.parameters())
    buffer_memory = sum(get_tensor_memory_bytes(b) for b in layer.buffers())
    total_theoretical = param_memory + buffer_memory
    
    result = {
        'param_memory_bytes': param_memory,
        'buffer_memory_bytes': buffer_memory,
        'total_theoretical_bytes': total_theoretical,
    }
    
    # Actual GPU memory (if available)
    if include_gpu_actual and torch.cuda.is_available():
        if device is None:
            device = torch.cuda.current_device()
        
        # Synchronize to ensure accurate measurement
        torch.cuda.synchronize(device)
        
        result['gpu_allocated_bytes'] = torch.cuda.memory_allocated(device)
        result['gpu_reserved_bytes'] = torch.cuda.memory_reserved(device)
        result['gpu_max_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    
    return result


def get_memory_snapshot(device: Optional[torch.device] = None) -> Dict[str, int]:
    """Get a snapshot of current GPU memory state.
    
    Useful for tracking memory changes before/after operations.
    
    Args:
        device: CUDA device. If None, uses current device.
    
    Returns:
        Dictionary with memory statistics:
            - allocated: Currently allocated memory
            - reserved: Currently reserved memory
            - max_allocated: Peak allocated memory
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}
    
    if device is None:
        device = torch.cuda.current_device()
    
    torch.cuda.synchronize(device)
    
    return {
        'allocated': torch.cuda.memory_allocated(device),
        'reserved': torch.cuda.memory_reserved(device),
        'max_allocated': torch.cuda.max_memory_allocated(device),
    }


def pack_i2s_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights {-1, 0, 1} into 2-bit format.
    
    Packing: 4 values per byte
    - -1 -> 00 (0)
    - 0  -> 01 (1) 
    - 1  -> 10 (2)
    """
    N, K = weight_ternary.shape
    
    weight_mapped = (weight_ternary + 1).clamp(0, 2).to(torch.uint8)
    
    pad_K = (4 - (K % 4)) % 4
    if pad_K > 0:
        weight_mapped = torch.nn.functional.pad(weight_mapped, (0, pad_K), value=1)
    
    K_padded = K + pad_K
    num_packed_cols = K_padded // 4
    
    weight_reshaped = weight_mapped.view(N, num_packed_cols, 4)
    
    weight_packed = (
        weight_reshaped[:, :, 0] |
        (weight_reshaped[:, :, 1] << 2) |
        (weight_reshaped[:, :, 2] << 4) |
        (weight_reshaped[:, :, 3] << 6)
    ).to(torch.uint8)
    
    return weight_packed


def unpack_i2s_weights(weight_packed: torch.Tensor, K: int, alpha: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unpack I2S weights using Triton kernel (if available) or optimized PyTorch operations.
    
    Tries Triton kernel first for better performance, falls back to PyTorch if unavailable.
    """
    assert weight_packed.dtype == torch.uint8, f"Expected uint8 packed weights, got {weight_packed.dtype}"
    assert alpha.dim() == 1 and alpha.shape[0] == K, f"Alpha shape {alpha.shape} doesn't match K={K}"
    assert alpha.dtype == torch.float32, f"Alpha must be stored in FP32 for precision, got {alpha.dtype}"
    
    N, num_packed_cols = weight_packed.shape
    device = weight_packed.device
    
    if TRITON_AVAILABLE and device.type == 'cuda':
        try:
            weight_packed_contig = weight_packed.contiguous()
            alpha_contig = alpha.contiguous()
            weight_unpacked = torch.empty(N, K, device=device, dtype=dtype)
            
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
            _i2s_unpack_kernel[grid](
                weight_packed_contig, alpha_contig, weight_unpacked,
                N, K, num_packed_cols,
                weight_packed_contig.stride(0), weight_packed_contig.stride(1),
                weight_unpacked.stride(0), weight_unpacked.stride(1),
                BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
            )
            
            return weight_unpacked
        except Exception as e:
            logger.debug(f"[TERNARY] Triton unpack kernel failed, using PyTorch fallback: {e}")
    
    packed_expanded = weight_packed.unsqueeze(-1)
    shift_positions = torch.arange(4, device=device, dtype=torch.uint8) * 2
    extracted_all = (packed_expanded >> shift_positions.view(1, 1, -1)) & 0b11
    
    K_padded = num_packed_cols * 4
    if K_padded == K:
        extracted = extracted_all.reshape(N, K)
    else:
        extracted = extracted_all.reshape(N, K_padded)[:, :K]
    
    val_ternary = extracted.to(torch.float32) - 1.0
    
    if alpha.is_contiguous():
        weight_unpacked_fp32 = val_ternary * alpha.view(1, -1)
    else:
        weight_unpacked_fp32 = val_ternary * alpha.contiguous().view(1, -1)
    
    weight_unpacked = weight_unpacked_fp32.to(dtype)
    
    return weight_unpacked


def _unpack_i2s_and_linear_impl(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack I2S weights and perform linear layer computation.
    
    Core implementation that can be compiled with torch.compile.
    """
    weight_unpacked = unpack_i2s_weights(weight_packed, K, alpha, dtype)
    out = F.linear(x, weight_unpacked, bias)
    return out




def _unpack_i2s_and_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack I2S weights and perform linear layer computation.
    
    Tries fused kernel first (CUTLASS or Triton), then falls back to unpack+matmul.
    """
    try:
        from sglang.srt.layers.quantization.i2s_fused_kernel import i2s_fused_matmul
        return i2s_fused_matmul(x, weight_packed, alpha, bias, K)
    except (ImportError, RuntimeError) as e:
        logger.debug(f"Fused kernel not available, using fallback: {e}")
        return _unpack_i2s_and_linear_impl(x, weight_packed, alpha, bias, K, dtype)


def replace_parameter(layer: nn.Module, name: str, new_param: torch.Tensor) -> None:
    """Replace a parameter in a layer, preserving attributes."""
    if hasattr(layer, name):
        old_param = getattr(layer, name)
        param_cls = type(old_param)
        requires_grad = False
        
        if isinstance(old_param, Parameter):
            new_param_obj = Parameter(new_param, requires_grad=requires_grad)
            for key, value in vars(old_param).items():
                if key not in ('_cdata', '_backward_hooks'):
                    try:
                        setattr(new_param_obj, key, value)
                    except AttributeError:
                        pass
        else:
            new_param_obj = new_param
        
        delattr(layer, name)
        setattr(layer, name, new_param_obj)
        
        if isinstance(new_param_obj, Parameter):
            layer._parameters[name] = new_param_obj


@dataclass
class TernaryConfig(QuantizationConfig):
    """Config class for ternary quantization.
    
    Args:
        threshold_scale: Scale factor for ternary quantization threshold (0.0-1.0)
            Lower values result in more aggressive quantization and sparsity.
        storage_mode: Storage mode - "i2s" (8x compression) or "fp16" (no compression, debugging)
            Default is "i2s" for best memory efficiency.
    """

    threshold_scale: float = 0.7
    storage_mode: str = "i2s"  # "i2s" or "fp16"

    def __post_init__(self):
        if not (0.0 < self.threshold_scale < 1.0):
            raise ValueError("threshold_scale must be between 0 and 1.")
        self.storage_mode = self.storage_mode.lower()
        if self.storage_mode not in ("i2s", "fp16"):
            raise ValueError(f"storage_mode must be 'i2s' or 'fp16', got '{self.storage_mode}'")

    @staticmethod
    def get_name() -> str:
        return "ternary"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """Return config filenames to search for quantization params."""
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TernaryConfig":
        threshold_scale = config.get("threshold_scale", 0.7)
        storage_mode = config.get("storage_mode", "i2s")
        return cls(threshold_scale, storage_mode)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        pref = prefix or ""
        lower_pref = pref.lower()

        if ("embed" in lower_pref) or ("lm_head" in lower_pref):
            return None

        if ("gate" in lower_pref) or ("router" in lower_pref):
            return None

        if isinstance(layer, LinearBase):
            return TernaryLinearMethod(self)
        
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_min_capability(self) -> int:
        """Minimum GPU capability required (SM version)."""
        return 0
    
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Supported activation dtypes."""
        return [torch.float16, torch.bfloat16]


class TernaryLinearMethod(LinearMethodBase):
    """Linear method for ternary quantization."""
    
    def __init__(self, quant_config: TernaryConfig):
        self.quant_config = quant_config

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, shard_id: Optional[str] = None):
        param.data.copy_(loaded_weight)

    def create_weights(
        self,
        layer: LinearBase,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        
        weight = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": self.weight_loader,
        })

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Apply ternary quantization to layer weights after loading."""
        try:
            weight = layer.weight.data
            original_dtype = weight.dtype
            N, K = weight.shape
            device = weight.device
            
            logger.info(f"[TERNARY] Quantizing layer: {weight.shape}")
            
            force_cleanup_and_sync(device)
            mem_before = measure_layer_memory_accurate(layer, device)
            gpu_snapshot_before = get_memory_snapshot(device)
            
            weight_memory_before = get_tensor_memory_bytes(weight)
            layer_memory_before = mem_before['total_theoretical_bytes']
            
            weight_fp32 = weight.float()
            absW = weight_fp32.abs()
            dim = 0
            th = self.quant_config.threshold_scale * absW.mean(dim, keepdim=True)
            mask = absW > th
            mask_f = mask.to(weight_fp32.dtype)
            alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.full_like(alpha, 1e-6))
            weight_ternary = weight_fp32.sign() * alpha * mask_f
            
            if self.quant_config.storage_mode == "i2s":
                weight_ternary_sign = torch.where(
                    mask,
                    weight_fp32.sign(),
                    torch.zeros_like(weight_fp32)
                ).to(torch.int8)
                
                weight_packed = pack_i2s_weights(weight_ternary_sign.float())
                replace_parameter(layer, 'weight', weight_packed)
                alpha_stored = alpha.view(-1).contiguous().to(torch.float32)
                layer.register_buffer('ternary_alpha', alpha_stored, persistent=False)
                
                layer._ternary_original_dtype = original_dtype
                layer._ternary_i2s_enabled = True
                layer._ternary_fp16_enabled = False
                layer._ternary_weight_shape = (N, K)
                
                del weight_fp32, absW, mask, mask_f, weight_ternary, weight_ternary_sign
                del weight
                
                force_cleanup_and_sync(device)
                mem_after = measure_layer_memory_accurate(layer, device)
                gpu_snapshot_after = get_memory_snapshot(device)
                
                weight_memory_after = get_tensor_memory_bytes(weight_packed)
                alpha_memory = get_tensor_memory_bytes(layer.ternary_alpha)
                layer_memory_after = mem_after['total_theoretical_bytes']
                
                theoretical_reduction_bytes = weight_memory_before - (weight_memory_after + alpha_memory)
                
                if layer_memory_after >= layer_memory_before:
                    logger.error(f"[TERNARY] ✗ ERROR: Layer tensor memory did not decrease!")
                
                if gpu_snapshot_before['allocated'] > 0 and gpu_snapshot_after['allocated'] > 0:
                    gpu_allocated_delta = gpu_snapshot_after['allocated'] - gpu_snapshot_before['allocated']
                    if gpu_allocated_delta > theoretical_reduction_bytes * 2:
                        logger.warning(f"[TERNARY] ⚠️  GPU memory increased unexpectedly by "
                                     f"{format_bytes(gpu_allocated_delta)}")
                
            elif self.quant_config.storage_mode == "fp16":
                weight_quantized = weight_ternary.to(original_dtype)
                replace_parameter(layer, 'weight', weight_quantized)
                layer.register_buffer('ternary_alpha', torch.ones(K, device=device, dtype=original_dtype), persistent=False)
                
                layer._ternary_original_dtype = original_dtype
                layer._ternary_i2s_enabled = False
                layer._ternary_fp16_enabled = True
                
                del weight_fp32, absW, mask, mask_f, weight_ternary
                del weight
                
                force_cleanup_and_sync(device)
                mem_after = measure_layer_memory_accurate(layer, device)
                gpu_snapshot_after = get_memory_snapshot(device)
                
                weight_memory_after = get_tensor_memory_bytes(weight_quantized)
                layer_memory_after = mem_after['total_theoretical_bytes']
                
                if gpu_snapshot_before['allocated'] > 0 and gpu_snapshot_after['allocated'] > 0:
                    gpu_allocated_delta = gpu_snapshot_after['allocated'] - gpu_snapshot_before['allocated']
                    if gpu_allocated_delta > weight_memory_before * 0.2:
                        logger.warning(f"[TERNARY] ⚠️  GPU memory increased unexpectedly by "
                                     f"{format_bytes(gpu_allocated_delta)}")
                
            else:
                raise ValueError(f"Unknown storage mode: {self.quant_config.storage_mode}")
                
        except Exception as e:
            logger.error(f"Error during ternary quantization: {e}. Keeping original weights.")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                logger.debug(f"Quantization error traceback: {traceback.format_exc()}")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = layer.weight
        i2s_enabled = getattr(layer, '_ternary_i2s_enabled', False)
        fp16_enabled = getattr(layer, '_ternary_fp16_enabled', False)
        
        x_compute = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
        b_compute = None if bias is None else (bias if bias.dtype in (torch.float16, torch.bfloat16) else bias.to(x_compute.dtype))
        
        if fp16_enabled:
            out = F.linear(x_compute, weight, b_compute)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        elif i2s_enabled and weight.dtype == torch.uint8:
            N, K = layer._ternary_weight_shape
            out = _unpack_i2s_and_linear(x_compute, weight, layer.ternary_alpha, b_compute, K, x_compute.dtype)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        else:
            out = F.linear(x_compute, weight, b_compute)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        
        return out


__all__ = ["TernaryConfig", "TernaryLinearMethod"]
