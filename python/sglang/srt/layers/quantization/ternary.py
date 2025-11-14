"""Ternary quantization method for SGLang.

This module implements ternary quantization (weights in {-1, 0, 1} × alpha)
following the approach from qwen2_correct.py training.

Supports two storage modes:
- i2s: 2-bit packed format (4x memory reduction)
- fp16: Direct ternary storage (no compression, for debugging)
"""

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

# Try to import Triton for potential future optimizations
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


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


def pack_i2s_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights {-1, 0, 1} into 2-bit format.
    
    Packing: 4 values per byte
    - -1 -> 00 (0)
    - 0  -> 01 (1) 
    - 1  -> 10 (2)
    """
    N, K = weight_ternary.shape
    
    # Map {-1, 0, 1} to {0, 1, 2}
    weight_mapped = (weight_ternary + 1).clamp(0, 2).to(torch.uint8)
    
    # Pad K to multiple of 4
    pad_K = (4 - (K % 4)) % 4
    if pad_K > 0:
        weight_mapped = torch.nn.functional.pad(weight_mapped, (0, pad_K), value=1)
    
    K_padded = K + pad_K
    num_packed_cols = K_padded // 4
    
    # Reshape to group 4 values
    weight_reshaped = weight_mapped.view(N, num_packed_cols, 4)
    
    # Pack 4 values into 1 byte
    weight_packed = (
        weight_reshaped[:, :, 0] |
        (weight_reshaped[:, :, 1] << 2) |
        (weight_reshaped[:, :, 2] << 4) |
        (weight_reshaped[:, :, 3] << 6)
    ).to(torch.uint8)
    
    return weight_packed


def unpack_i2s_weights(weight_packed: torch.Tensor, K: int, alpha: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unpack I2S weights using optimized PyTorch operations (GPU)."""
    assert weight_packed.dtype == torch.uint8, f"Expected uint8 packed weights, got {weight_packed.dtype}"
    assert alpha.dim() == 1 and alpha.shape[0] == K, f"Alpha shape {alpha.shape} doesn't match K={K}"
    assert alpha.dtype == torch.float32, f"Alpha must be stored in FP32 for precision, got {alpha.dtype}"
    
    N, num_packed_cols = weight_packed.shape
    device = weight_packed.device
    
    # Reshape packed to [N, num_packed_cols, 1] for broadcasting
    packed_3d = weight_packed.unsqueeze(-1)  # [N, num_packed_cols, 1]
    
    # Extract all 4 values per byte using broadcasting
    shift_positions = torch.arange(4, device=device, dtype=torch.uint8) * 2  # [0, 2, 4, 6]
    
    # Extract bits: [N, num_packed_cols, 4]
    extracted_all = (packed_3d >> shift_positions.view(1, 1, -1)) & 0b11
    
    # Reshape to [N, K] (handle padding)
    K_padded = num_packed_cols * 4
    if K_padded == K:
        extracted = extracted_all.reshape(N, K)
    else:
        extracted = extracted_all.reshape(N, K_padded)[:, :K]
    
    # Map {0, 1, 2} -> {-1, 0, 1}
    # Keep in FP32 for precision during alpha multiplication
    extracted_fp32 = extracted.to(torch.float32)
    val_ternary = extracted_fp32 - 1.0
    
    # Apply alpha scaling in FP32 to preserve precision
    # Alpha is stored in FP32, so ensure it's contiguous
    if not alpha.is_contiguous():
        alpha_contiguous = alpha.contiguous()
    else:
        alpha_contiguous = alpha
    
    # Multiply in FP32, then convert to target dtype
    weight_unpacked_fp32 = val_ternary * alpha_contiguous.view(1, -1)
    weight_unpacked = weight_unpacked_fp32.to(dtype)
    
    return weight_unpacked


def _unpack_i2s_and_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Unpack I2S weights and perform linear layer computation."""
    weight_unpacked = unpack_i2s_weights(weight_packed, K, alpha, dtype)
    out = F.linear(x, weight_unpacked, bias)
    return out


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
            Lower values = more aggressive quantization = more sparsity
            Default 0.7 matches qwen2_correct.py training
        storage_mode: Storage mode - "i2s" (4x compression) or "fp16" (no compression, debugging)
            Default is "i2s" for best memory efficiency with exact accuracy
    """

    threshold_scale: float = 0.7
    storage_mode: str = "i2s"  # "i2s" or "fp16"

    def __post_init__(self):
        if not (0.0 < self.threshold_scale < 1.0):
            raise ValueError("threshold_scale must be between 0 and 1.")
        # Normalize storage_mode
        self.storage_mode = self.storage_mode.lower()
        if self.storage_mode not in ("i2s", "fp16"):
            raise ValueError(f"storage_mode must be 'i2s' or 'fp16', got '{self.storage_mode}'")

    @staticmethod
    def get_name() -> str:
        return "ternary"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        """Return config filenames to search for quantization params."""
        return []  # Ternary doesn't need config files, uses runtime quantization

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TernaryConfig":
        threshold_scale = config.get("threshold_scale", 0.7)
        storage_mode = config.get("storage_mode", "i2s")
        return cls(threshold_scale, storage_mode)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        # Skip known layers by name
        pref = prefix or ""
        lower_pref = pref.lower()

        # Always skip embeddings and lm_head (following qwen2_correct.py training)
        if ("embed" in lower_pref) or ("lm_head" in lower_pref):
            return None

        # Apply to standard linear layers only
        if isinstance(layer, LinearBase):
            return TernaryLinearMethod(self)
        
        # MoE layers: Return None to use default unquantized path
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
        
        # Log initialization
        logger.info("=" * 80)
        if self.quant_config.storage_mode == "i2s":
            logger.info("[TERNARY] Quantization initialized - I2_S (Int2 Super-packed) mode")
            logger.info("[TERNARY] Packing: {-1→00, 0→01, 1→10} → 4x memory reduction")
            logger.info("[TERNARY] Accuracy: Exact match to qwen2_correct.py (alpha applied explicitly)")
        else:
            logger.info("[TERNARY] Quantization initialized - FP16 mode (no compression)")
            logger.info("[TERNARY] For debugging: ternary weights stored directly")
        
        # Log GPU memory if available
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                logger.info(f"[TERNARY] GPU Memory on init: allocated={allocated:.2f} MB, "
                          f"reserved={reserved:.2f} MB, total={total_memory:.2f} MB")
            except Exception as e:
                logger.debug(f"[TERNARY] Could not log GPU memory: {e}")
        
        logger.info("=" * 80)

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

            weight_memory_before = get_tensor_memory_bytes(weight)
            layer_memory_before = get_layer_memory_bytes(layer)
            
            mode_str = {
                "i2s": "I2_S (4x compression)",
                "fp16": "FP16 ternary (no compression)"
            }.get(self.quant_config.storage_mode, "unknown")
            
            logger.info(f"Quantizing layer to ternary + {mode_str}: {weight.shape}")
            logger.info(f"[TERNARY] Memory BEFORE quantization: weight={format_bytes(weight_memory_before)}, "
                       f"layer_total={format_bytes(layer_memory_before)}")
            
            # Quantize weights following qwen2_correct.py
            weight_fp32 = weight.float()
            absW = weight_fp32.abs()
            # Per-column quantization (dim=0)
            dim = 0
            th = self.quant_config.threshold_scale * absW.mean(dim, keepdim=True)
            mask = absW > th
            mask_f = mask.to(weight_fp32.dtype)
            # Per-column alpha scale
            alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)
            # Safety check for NaN/Inf
            alpha = torch.where(torch.isfinite(alpha), alpha, torch.full_like(alpha, 1e-6))
            weight_ternary = weight_fp32.sign() * alpha * mask_f
            
            if self.quant_config.storage_mode == "i2s":
                # Extract ternary signs for packing
                weight_ternary_sign = torch.where(
                    mask,
                    weight_fp32.sign(),
                    torch.zeros_like(weight_fp32)
                ).to(torch.int8)
                
                # Pack to 2-bit format
                weight_packed = pack_i2s_weights(weight_ternary_sign.float())
                
                # Replace weight and store alpha
                replace_parameter(layer, 'weight', weight_packed)
                # Store alpha in FP32 to preserve precision (critical for correct scaling)
                alpha_stored = alpha.view(-1).contiguous().to(torch.float32)
                layer.register_buffer('ternary_alpha', alpha_stored, persistent=False)
                
                # Set flags
                layer._ternary_original_dtype = original_dtype
                layer._ternary_i2s_enabled = True
                layer._ternary_fp16_enabled = False
                layer._ternary_weight_shape = (N, K)
                
                # Clean up
                del weight_fp32, absW, mask, mask_f, weight_ternary, weight_ternary_sign
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log results
                weight_memory_after = get_tensor_memory_bytes(weight_packed)
                alpha_memory = get_tensor_memory_bytes(layer.ternary_alpha)
                reduction_bytes = weight_memory_before - (weight_memory_after + alpha_memory)
                reduction_pct = (1 - (weight_memory_after + alpha_memory) / weight_memory_before) * 100
                
                logger.info(f"[TERNARY] ✓ Quantized to I2_S: {layer.weight.shape}, "
                           f"original_shape=({N}, {K})")
                logger.info(f"[TERNARY] Memory reduction: {format_bytes(reduction_bytes)} ({reduction_pct:.1f}%)")
                
            elif self.quant_config.storage_mode == "fp16":
                # Store ternary weights directly (for debugging)
                weight_quantized = weight_ternary.to(original_dtype)
                replace_parameter(layer, 'weight', weight_quantized)
                
                # Store dummy alpha (already applied)
                layer.register_buffer('ternary_alpha', torch.ones(K, device=weight.device, dtype=original_dtype), persistent=False)
                
                # Set flags
                layer._ternary_original_dtype = original_dtype
                layer._ternary_i2s_enabled = False
                layer._ternary_fp16_enabled = True
                
                # Clean up
                del weight_fp32, absW, mask, mask_f, weight_ternary
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"[TERNARY] ✓ Quantized to FP16 ternary (no compression): {layer.weight.shape}")
                logger.info(f"[TERNARY] Note: Alpha is already applied, stored as 1.0 for compatibility")
                
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
            # FP16 ternary mode: weights already have alpha applied
            out = F.linear(x_compute, weight, b_compute)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        elif i2s_enabled and weight.dtype == torch.uint8:
            # I2S mode: unpack and compute
            N, K = layer._ternary_weight_shape
            out = _unpack_i2s_and_linear(x_compute, weight, layer.ternary_alpha, b_compute, K, x_compute.dtype)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        else:
            # Fallback to regular matmul
            out = F.linear(x_compute, weight, b_compute)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        
        return out


__all__ = ["TernaryConfig", "TernaryLinearMethod"]
