"""Ternary quantization with storage options for SGLang.

This implementation:
- Per-column ternary quantization with alpha scaling (matching qwen2_correct.py)
- Storage modes: I2_S (RECOMMENDED: 4x compression, exact accuracy), or blockwise (2x compression)
- Runtime weight-only quantization (applied after model loading)
- Forward pass: I2_S unpacks and applies alpha explicitly (exact), blockwise uses FP8 with explicit alpha

Key features:
- Per-column threshold-based ternary quantization
- Per-column alpha scales (matching qwen2_correct.py)
- I2_S mode (RECOMMENDED): 4x memory reduction, exact accuracy, fast Triton kernels
- FP8 blockwise mode: 2x memory reduction with explicit alpha application
- NO TVM dependency = NO conflicts with xgrammar
"""

import gc
import logging
import os
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


def validate_quantization_correctness(
    weight_fp16: torch.Tensor,
    weight_quantized: torch.Tensor,
    alpha: torch.Tensor,
    quantization_mode: str,
    fp8_block_scales: Optional[torch.Tensor] = None,
    fp8_block_size: Optional[tuple] = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    num_test_samples: int = 10,
) -> bool:
    device = weight_fp16.device
    dtype = weight_fp16.dtype
    N, K = weight_fp16.shape
    
    torch.manual_seed(42)
    test_inputs = [
        torch.randn(1, K, device=device, dtype=dtype) for _ in range(num_test_samples)
    ]
    
    reference_outputs = []
    for x in test_inputs:
        y_ref = torch.nn.functional.linear(x, weight_fp16, None)
        reference_outputs.append(y_ref)
    
    quantized_outputs = []
    
    if quantization_mode == "i2s":
        from sglang.srt.layers.quantization.ternary import unpack_i2s_weights
        
        for x in test_inputs:
            weight_unpacked = unpack_i2s_weights(weight_quantized, K, alpha, dtype)
            y_quant = torch.nn.functional.linear(x, weight_unpacked, None)
            quantized_outputs.append(y_quant)
    
    elif quantization_mode == "fp8":
        block_size_N, block_size_K = fp8_block_size
        num_blocks_N = fp8_block_scales.shape[0]
        num_blocks_K = fp8_block_scales.shape[1]
        
        pad_N = (block_size_N - (N % block_size_N)) % block_size_N
        pad_K = (block_size_K - (K % block_size_K)) % block_size_K
        
        if pad_N > 0 or pad_K > 0:
            weight_padded = torch.nn.functional.pad(weight_quantized, (0, pad_K, 0, pad_N))
        else:
            weight_padded = weight_quantized
        
        N_padded, K_padded = weight_padded.shape
        
        weight_blocks = weight_padded.view(
            num_blocks_N, block_size_N,
            num_blocks_K, block_size_K
        ).permute(0, 2, 1, 3).contiguous()
        
        FP8_MAX = 448.0
        scales_expanded = fp8_block_scales.unsqueeze(-1).unsqueeze(-1)
        weight_dequant = weight_blocks.float() * scales_expanded * FP8_MAX
        
        weight_dequant = weight_dequant.permute(0, 2, 1, 3).contiguous()
        weight_dequant = weight_dequant.view(N_padded, K_padded)
        
        if pad_N > 0 or pad_K > 0:
            weight_dequant = weight_dequant[:N, :K]
        
        weight_dequant = weight_dequant * alpha.unsqueeze(0).to(dtype=dtype)
        
        for x in test_inputs:
            y_quant = torch.nn.functional.linear(x, weight_dequant, None)
            quantized_outputs.append(y_quant)
    
    else:
        logger.error(f"Unknown quantization mode: {quantization_mode}")
        return False
    
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    all_close = True
    
    for y_ref, y_quant in zip(reference_outputs, quantized_outputs):
        abs_diff = (y_ref - y_quant).abs()
        rel_diff = abs_diff / (y_ref.abs() + 1e-8)
        
        max_abs_diff = max(max_abs_diff, abs_diff.max().item())
        max_rel_diff = max(max_rel_diff, rel_diff.max().item())
        
        if not torch.allclose(y_ref, y_quant, rtol=rtol, atol=atol):
            all_close = False
    
    logger.info(
        f"[TERNARY] Validation ({quantization_mode}): "
        f"max_abs_diff={max_abs_diff:.6f}, max_rel_diff={max_rel_diff:.6f}, "
        f"passed={'✓' if all_close else '✗'}"
    )
    
    return all_close


def format_bytes(bytes_value: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_tensor_memory_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def get_layer_memory_bytes(layer: torch.nn.Module) -> int:
    total = 0
    for param in layer.parameters():
        total += get_tensor_memory_bytes(param)
    for buffer in layer.buffers():
        total += get_tensor_memory_bytes(buffer)
    return total


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
        stride_alpha,
        stride_output_n,
        stride_output_k,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        
        n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        n_mask = n_offsets < N
        k_mask = k_offsets < K
        
        packed_k_idx = k_offsets // 4
        packed_byte_offsets = n_offsets[:, None] * stride_packed_n + packed_k_idx[None, :] * stride_packed_k
        packed_mask = n_mask[:, None] & (packed_k_idx[None, :] < num_packed_cols)
        packed_bytes = tl.load(packed_ptr + packed_byte_offsets, mask=packed_mask, other=0)
        
        bit_pos_in_byte = k_offsets % 4
        shift_amounts = bit_pos_in_byte * 2
        
        packed_expanded = tl.broadcast_to(packed_bytes, (BLOCK_SIZE_N, BLOCK_SIZE_K))
        shift_expanded = tl.broadcast_to(shift_amounts[None, :], (BLOCK_SIZE_N, BLOCK_SIZE_K))
        extracted_values = (packed_expanded >> shift_expanded) & 0b11
        
        extracted_indices = extracted_values.to(tl.int32)
        val_ternary = tl.where(
            extracted_indices == 0, -1.0,
            tl.where(extracted_indices == 2, 1.0, 0.0)
        )
        
        alpha_values = tl.load(alpha_ptr + k_offsets * stride_alpha, mask=k_mask, other=1.0)
        output_values = val_ternary * alpha_values[None, :]
        
        output_mask = n_mask[:, None] & k_mask[None, :]
        output_offsets = n_offsets[:, None] * stride_output_n + k_offsets[None, :] * stride_output_k
        tl.store(output_ptr + output_offsets, output_values, mask=output_mask)


def pack_i2s_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
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
    N, num_packed_cols = weight_packed.shape
    device = weight_packed.device
    weight_unpacked = torch.empty(N, K, device=device, dtype=dtype)
    
    _i2s_unpack_kernel[(triton.cdiv(N, 128), triton.cdiv(K, 128))](
        weight_packed, alpha, weight_unpacked, N, K, num_packed_cols,
        weight_packed.stride(0), weight_packed.stride(1), alpha.stride(0),
        weight_unpacked.stride(0), weight_unpacked.stride(1),
        BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
    )
    
    return weight_unpacked


def _unpack_i2s_and_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    alpha: torch.Tensor,
    bias: Optional[torch.Tensor],
    K: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight_unpacked = unpack_i2s_weights(weight_packed, K, alpha, dtype)
    out = torch.matmul(x, weight_unpacked.t())
    if bias is not None:
        out = out + bias
    return out


@dataclass
class TernaryConfig(QuantizationConfig):
    """Config class for ternary quantization with storage options.
    
    Args:
        threshold_scale: Scale factor for ternary quantization threshold (0.0-1.0)
            Lower values = more aggressive quantization = more sparsity
            Default 0.7 matches qwen2_correct.py
        max_output_features: Skip quantization for layers larger than this
            (e.g., lm_head with 100K+ vocab size)
        fp8_storage_mode: Storage mode - "i2s" (RECOMMENDED: 4x compression, exact accuracy),
            or "blockwise" (2x compression, FP8 with explicit alpha)
        fp8_block_size: Block size for FP8 quantization (default: 128x128)
            Only used when fp8_storage_mode="blockwise"
            Larger blocks = fewer scales but potentially less accuracy
        use_i2s: DEPRECATED - use fp8_storage_mode="i2s" instead
            Enable I2_S (Int2 Super-packed) mode for 4x memory reduction
    """

    threshold_scale: float = 0.7
    max_output_features: int = 100_000
    fp8_storage_mode: str = "i2s"  # "i2s" (RECOMMENDED: 4x compression, exact accuracy), or "blockwise" (2x compression)
    fp8_block_size: tuple = (128, 128)  # (N_block, K_block) for blockwise FP8 quantization
    use_i2s: bool = False  # DEPRECATED - use fp8_storage_mode="i2s" instead

    def __post_init__(self):
        if not (0.0 < self.threshold_scale < 1.0):
            raise ValueError("threshold_scale must be between 0 and 1.")
        if self.max_output_features <= 0:
            raise ValueError("max_output_features must be positive.")
        # Normalize fp8_storage_mode
        self.fp8_storage_mode = self.fp8_storage_mode.lower()
        if self.fp8_storage_mode not in ("blockwise", "i2s"):
            raise ValueError(f"fp8_storage_mode must be 'blockwise' or 'i2s', got '{self.fp8_storage_mode}'")
        # Handle deprecated use_i2s flag
        if self.use_i2s:
            if self.fp8_storage_mode != "i2s":
                logger.warning(f"[TERNARY] use_i2s=True is deprecated. Setting fp8_storage_mode='i2s'")
            self.fp8_storage_mode = "i2s"

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
        max_output_features = config.get("max_output_features", 100_000)
        fp8_storage_mode = config.get("fp8_storage_mode", "i2s")  # Default to I2_S for best accuracy + memory efficiency
        # Support deprecated use_i2s flag
        use_i2s = config.get("use_i2s", False) or os.environ.get("TERNARY_USE_I2S", "0") == "1"
        if use_i2s:
            fp8_storage_mode = "i2s"
        fp8_block_size = config.get("fp8_block_size", (128, 128))
        if isinstance(fp8_block_size, (list, tuple)) and len(fp8_block_size) == 2:
            fp8_block_size = tuple(fp8_block_size)
        else:
            fp8_block_size = (128, 128)
        return cls(threshold_scale, max_output_features, fp8_storage_mode, fp8_block_size, use_i2s)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["LinearMethodBase"]:
        # Skip known layers by name
        pref = prefix or ""
        lower_pref = pref.lower()

        # Always skip embeddings; allow opting into lm_head quant via env
        if ("embed" in lower_pref):
            return None
        if ("lm_head" in lower_pref) and (os.environ.get("TERNARY_INCLUDE_LM_HEAD", "0") != "1"):
            return None

        # Optional coarse-grained skips controlled by env
        import os as _os
        if _os.environ.get("TERNARY_SKIP_QKV", "0") == "1" and ("qkv" in lower_pref or "qkv_proj" in lower_pref):
            return None
        if _os.environ.get("TERNARY_SKIP_O_PROJ", "0") == "1" and ("o_proj" in lower_pref):
            return None
        if _os.environ.get("TERNARY_SKIP_MLP", "0") == "1" and (
            "mlp" in lower_pref or "gate_up_proj" in lower_pref or "down_proj" in lower_pref
        ):
            return None

        if isinstance(layer, LinearBase):
            return TernaryLinearMethod(self)
        # MoE layers: Return None to use default unquantized path (BF16)
        # Ternary quantization is only for standard linear layers, not MoE
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_min_capability(self) -> int:
        """Minimum GPU capability required (SM version)."""
        # FP16/BF16 operations work on all modern GPUs
        return 0
    
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Supported activation dtypes."""
        return [torch.float16, torch.bfloat16]


class TernaryLinearMethod(LinearMethodBase):
    """
    Linear method for ternary quantization with runtime quantization.
    
    This applies ternary quantization after weights are loaded but before
    the first forward pass, avoiding TVM conflicts.
    """
    
    # Class-level shared stats across all instances
    _shared_memory_stats = {
        'total_before_mb': 0.0,
        'total_after_mb': 0.0,
        'total_scales_mb': 0.0,
        'layers_quantized': 0,
    }
    _shared_quantization_stats = {
        'total_layers': 0,
        'ternary_quantized': 0,
        'fp8_blockwise': 0,
        'i2s_packed': 0,
        'bf16_fallback': 0,
    }
    _final_summary_logged = False  # Track if final summary has been logged

    def __init__(self, quant_config: TernaryConfig):
        self.quant_config = quant_config
        
        # Use shared stats across all instances
        self._ternary_quantization_stats = TernaryLinearMethod._shared_quantization_stats
        self._memory_stats = TernaryLinearMethod._shared_memory_stats
        
        logger.info("=" * 80)
        if quant_config.fp8_storage_mode == "i2s":
            logger.info("[TERNARY] Quantization initialized - I2_S (Int2 Super-packed) mode [RECOMMENDED]")
            logger.info("[TERNARY] Packing: {-1→00, 0→01, 1→10} → 4x memory reduction")
            logger.info("[TERNARY] Accuracy: Exact match to qwen2_correct.py (alpha applied explicitly)")
        else:
            logger.info("[TERNARY] Quantization initialized - Ternary + FP8 blockwise storage")
            logger.info(f"[TERNARY] FP8 block size: {quant_config.fp8_block_size}")
            logger.info("[TERNARY] Accuracy: Good (alpha applied explicitly, but FP8 precision loss)")
        
        # Log GPU memory on initialization
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                if torch.cuda.device_count() > 0:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
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
        
        layer._ternary_output_size = output_size
        layer._ternary_is_too_large = output_size > self.quant_config.max_output_features

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, '_ternary_is_too_large', False):
            logger.info(f"Skipping quantization for large layer: {layer.weight.shape}")
            return
        
        try:
            weight = layer.weight.data
            original_dtype = weight.dtype
            N, K = weight.shape

            weight_memory_before = get_tensor_memory_bytes(weight)
            layer_memory_before = get_layer_memory_bytes(layer)
            
            mode_str = {
                "i2s": "I2_S",
                "blockwise": "FP8 blockwise"
            }.get(self.quant_config.fp8_storage_mode, "FP8")
            logger.info(f"Quantizing layer to ternary + {mode_str}: {weight.shape}")
            logger.info(f"[TERNARY] Memory BEFORE quantization: weight={format_bytes(weight_memory_before)}, "
                       f"layer_total={format_bytes(layer_memory_before)}")
            
            weight_fp32 = weight.float()
            absW = weight_fp32.abs()
            dim = 0
            th = self.quant_config.threshold_scale * absW.mean(dim, keepdim=True)
            mask = absW > th
            mask_f = mask.to(weight_fp32.dtype)
            alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)
            weight_ternary = weight_fp32.sign() * alpha * mask_f
            
            if self.quant_config.fp8_storage_mode == "i2s":
                weight_ternary_int = weight_ternary.sign().to(torch.int8)
                
                weight_packed = pack_i2s_weights(weight_ternary_int.float())
                
                replace_parameter(layer, 'weight', weight_packed)
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                layer._ternary_original_dtype = original_dtype
                layer._ternary_fp8_enabled = False
                layer._ternary_i2s_enabled = True
                layer._ternary_weight_shape = (N, K)
                
                if os.environ.get("TERNARY_VALIDATE_CORRECTNESS", "0") == "1":
                    weight_fp16_ref = weight_ternary.to(original_dtype)
                    validation_passed = validate_quantization_correctness(
                        weight_fp16=weight_fp16_ref,
                        weight_quantized=weight_packed,
                        alpha=alpha.view(-1),
                        quantization_mode="i2s",
                        rtol=1e-2,
                        atol=1e-2,
                    )
                    if not validation_passed:
                        logger.warning(f"[TERNARY] I2_S validation failed for layer {type(layer).__name__}")
                
                del weight_fp32, absW, mask, mask_f, weight_ternary, weight_ternary_int
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                weight_memory_after = get_tensor_memory_bytes(weight_packed)
                alpha_memory = get_tensor_memory_bytes(layer.ternary_alpha)
                layer_memory_after = get_layer_memory_bytes(layer)
                
                self._ternary_quantization_stats['ternary_quantized'] += 1
                self._ternary_quantization_stats['i2s_packed'] += 1
                self._memory_stats['total_before_mb'] += weight_memory_before / (1024 ** 2)
                self._memory_stats['total_after_mb'] += (weight_memory_after + alpha_memory) / (1024 ** 2)
                self._memory_stats['total_scales_mb'] += alpha_memory / (1024 ** 2)
                self._memory_stats['layers_quantized'] += 1
                
                reduction_bytes = weight_memory_before - (weight_memory_after + alpha_memory)
                reduction_pct = (1 - (weight_memory_after + alpha_memory) / weight_memory_before) * 100
                
                logger.info(f"[TERNARY] ✓ Quantized to I2_S (packed): {layer.weight.shape}, "
                           f"dtype={layer.weight.dtype}, original_shape=({N}, {K})")
                logger.info(f"[TERNARY] Memory AFTER quantization: weight={format_bytes(weight_memory_after)}, "
                           f"alpha={format_bytes(alpha_memory)}, layer_total={format_bytes(layer_memory_after)}")
                logger.info(f"[TERNARY] Memory reduction: {format_bytes(reduction_bytes)} ({reduction_pct:.1f}%)")
                
                if self._memory_stats['layers_quantized'] % 10 == 0:
                    cumulative_reduction = (1 - self._memory_stats['total_after_mb'] / self._memory_stats['total_before_mb']) * 100
                    logger.info(f"[TERNARY] Cumulative memory stats: {self._memory_stats['layers_quantized']} layers, "
                              f"before={self._memory_stats['total_before_mb']:.2f} MB, "
                              f"after={self._memory_stats['total_after_mb']:.2f} MB, "
                              f"reduction={cumulative_reduction:.1f}%")
                
                return
            
            fp8_available = False
            if hasattr(torch, 'float8_e4m3fn'):
                try:
                    test_tensor = torch.tensor([1.0], dtype=torch.float32, device=weight_ternary.device)
                    test_fp8 = test_tensor.to(torch.float8_e4m3fn)
                    fp8_available = (test_fp8.dtype == torch.float8_e4m3fn)
                except (RuntimeError, TypeError, AttributeError):
                    fp8_available = False
            
            if not fp8_available:
                logger.warning(f"[TERNARY] FP8 not available for {type(layer).__name__}. Using BF16 storage.")
                weight_quantized = weight_ternary.to(torch.bfloat16)
                replace_parameter(layer, 'weight', weight_quantized)
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                layer._ternary_original_dtype = original_dtype
                layer._ternary_fp8_enabled = False
                self._ternary_quantization_stats['bf16_fallback'] += 1
                
                del weight_fp32, absW, mask, mask_f, weight_ternary
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                weight_memory_after = get_tensor_memory_bytes(weight_quantized)
                alpha_memory = get_tensor_memory_bytes(layer.ternary_alpha)
                layer_memory_after = get_layer_memory_bytes(layer)
                self._memory_stats['total_before_mb'] += weight_memory_before / (1024 ** 2)
                self._memory_stats['total_after_mb'] += (weight_memory_after + alpha_memory) / (1024 ** 2)
                self._memory_stats['total_scales_mb'] += alpha_memory / (1024 ** 2)
                self._memory_stats['layers_quantized'] += 1
                
                logger.info(f"[TERNARY] ✓ Quantized to ternary (BF16): {layer.weight.shape}")
                logger.info(f"[TERNARY] Memory AFTER quantization: weight={format_bytes(weight_memory_after)}, "
                           f"alpha={format_bytes(alpha_memory)}, layer_total={format_bytes(layer_memory_after)}")
                logger.info(f"[TERNARY] Memory reduction: {format_bytes(weight_memory_before - weight_memory_after)} "
                           f"({(1 - weight_memory_after/weight_memory_before)*100:.1f}%)")
                return
            
            block_size_N, block_size_K = self.quant_config.fp8_block_size
            FP8_MAX = 448.0
            
            pad_N = (block_size_N - (N % block_size_N)) % block_size_N
            pad_K = (block_size_K - (K % block_size_K)) % block_size_K
            
            if pad_N > 0 or pad_K > 0:
                weight_padded = torch.nn.functional.pad(weight_ternary, (0, pad_K, 0, pad_N))
            else:
                weight_padded = weight_ternary
            
            N_padded, K_padded = weight_padded.shape
            
            num_blocks_N = N_padded // block_size_N
            num_blocks_K = K_padded // block_size_K
            
            weight_blocks = weight_padded.view(
                num_blocks_N, block_size_N,
                num_blocks_K, block_size_K
            )
            
            weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()
            
            abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)
            scales = abs_max / FP8_MAX
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)
            
            weight_scaled = weight_blocks / scales
            weight_scaled = weight_scaled.clamp(-FP8_MAX, FP8_MAX)
            weight_fp8_blocks = weight_scaled.to(torch.float8_e4m3fn)
            
            weight_fp8_blocks = weight_fp8_blocks.permute(0, 2, 1, 3).contiguous()
            weight_fp8 = weight_fp8_blocks.view(N_padded, K_padded)
            
            if pad_N > 0 or pad_K > 0:
                weight_fp8 = weight_fp8[:N, :K].contiguous()
            
            scales_2d = scales.view(num_blocks_N, num_blocks_K)
            
            replace_parameter(layer, 'weight', weight_fp8)
            layer.register_buffer('fp8_block_scales', scales_2d.contiguous(), persistent=False)
            layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
            layer._ternary_original_dtype = original_dtype
            layer._ternary_fp8_enabled = True
            layer._ternary_block_size = (block_size_N, block_size_K)
            layer._ternary_weight_shape = (N, K)
            
            if os.environ.get("TERNARY_VALIDATE_CORRECTNESS", "0") == "1":
                weight_fp16_ref = weight_ternary.to(original_dtype)
                validation_passed = validate_quantization_correctness(
                    weight_fp16=weight_fp16_ref,
                    weight_quantized=weight_fp8,
                    alpha=alpha.view(-1),
                    quantization_mode="fp8",
                    fp8_block_scales=scales_2d,
                    fp8_block_size=(block_size_N, block_size_K),
                    rtol=1e-2,
                    atol=1e-2,
                )
                if not validation_passed:
                    logger.warning(f"[TERNARY] FP8 validation failed for layer {type(layer).__name__}")
            
            del weight_fp32, absW, mask, mask_f, weight_ternary, weight_padded, weight_blocks, abs_max, scales, weight_scaled, weight_fp8_blocks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            weight_memory_after = get_tensor_memory_bytes(weight_fp8)
            scales_memory = get_tensor_memory_bytes(layer.fp8_block_scales)
            alpha_memory = get_tensor_memory_bytes(layer.ternary_alpha)
            layer_memory_after = get_layer_memory_bytes(layer)
            
            self._ternary_quantization_stats['ternary_quantized'] += 1
            self._ternary_quantization_stats['fp8_blockwise'] += 1
            self._memory_stats['total_before_mb'] += weight_memory_before / (1024 ** 2)
            self._memory_stats['total_after_mb'] += (weight_memory_after + scales_memory + alpha_memory) / (1024 ** 2)
            self._memory_stats['total_scales_mb'] += (scales_memory + alpha_memory) / (1024 ** 2)
            self._memory_stats['layers_quantized'] += 1
            
            reduction_bytes = weight_memory_before - (weight_memory_after + scales_memory + alpha_memory)
            reduction_pct = (1 - (weight_memory_after + scales_memory + alpha_memory) / weight_memory_before) * 100
            
            logger.info(
                f"[TERNARY] ✓ Quantized to ternary + FP8 blockwise: {layer.weight.shape}, "
                f"dtype={layer.weight.dtype}, blocks={num_blocks_N}x{num_blocks_K}"
            )
            logger.info(f"[TERNARY] Memory AFTER quantization: weight={format_bytes(weight_memory_after)}, "
                       f"scales={format_bytes(scales_memory)}, alpha={format_bytes(alpha_memory)}, "
                       f"layer_total={format_bytes(layer_memory_after)}")
            logger.info(f"[TERNARY] Memory reduction: {format_bytes(reduction_bytes)} ({reduction_pct:.1f}%)")
            
            if self._memory_stats['layers_quantized'] % 10 == 0:
                cumulative_reduction = (1 - self._memory_stats['total_after_mb'] / self._memory_stats['total_before_mb']) * 100
                logger.info(f"[TERNARY] Cumulative memory stats: {self._memory_stats['layers_quantized']} layers, "
                          f"before={self._memory_stats['total_before_mb']:.2f} MB, "
                          f"after={self._memory_stats['total_after_mb']:.2f} MB, "
                          f"reduction={cumulative_reduction:.1f}%")
            
            if self._memory_stats['layers_quantized'] % 50 == 0:
                self.log_final_memory_summary()
        except Exception as e:
            logger.error(f"Error during ternary+FP8 quantization: {e}. Keeping original weights.")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Quantization error traceback: {traceback.format_exc()}")
    
    def log_final_memory_summary(self):
        """Log final memory usage summary after all quantization is complete."""
        # Only log once, even if called multiple times
        if TernaryLinearMethod._final_summary_logged:
            return
        
        if self._memory_stats['layers_quantized'] == 0:
            return
        
        TernaryLinearMethod._final_summary_logged = True
        
        logger.info("=" * 80)
        logger.info("[TERNARY] FINAL MEMORY USAGE SUMMARY")
        logger.info("=" * 80)
        
        total_before = self._memory_stats['total_before_mb']
        total_after = self._memory_stats['total_after_mb']
        total_scales = self._memory_stats['total_scales_mb']
        layers_quantized = self._memory_stats['layers_quantized']
        
        reduction_mb = total_before - total_after
        reduction_pct = (1 - total_after / total_before) * 100 if total_before > 0 else 0
        
        logger.info(f"[TERNARY] Total layers quantized: {layers_quantized}")
        logger.info(f"[TERNARY] Memory BEFORE quantization: {total_before:.2f} MB ({total_before/1024:.2f} GB)")
        logger.info(f"[TERNARY] Memory AFTER quantization: {total_after:.2f} MB ({total_after/1024:.2f} GB)")
        logger.info(f"[TERNARY]   - Weights: {total_after - total_scales:.2f} MB")
        logger.info(f"[TERNARY]   - Scales (FP8 + alpha): {total_scales:.2f} MB")
        logger.info(f"[TERNARY] Total memory saved: {reduction_mb:.2f} MB ({reduction_mb/1024:.2f} GB)")
        logger.info(f"[TERNARY] Memory reduction: {reduction_pct:.1f}%")
        
        # Log GPU memory stats
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
                if torch.cuda.device_count() > 0:
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # MB
                    logger.info(f"[TERNARY] GPU Memory after quantization: allocated={allocated:.2f} MB, "
                              f"reserved={reserved:.2f} MB, total={total_memory:.2f} MB")
                    logger.info(f"[TERNARY] GPU Memory usage: {allocated/total_memory*100:.1f}% allocated, "
                              f"{reserved/total_memory*100:.1f}% reserved")
                    logger.info(f"[TERNARY] Note: Reserved memory may remain high due to PyTorch's memory allocator "
                              f"holding chunks. Actual weight memory reduction: {reduction_mb:.2f} MB ({reduction_pct:.1f}%)")
            except Exception as e:
                logger.debug(f"[TERNARY] Could not log GPU memory: {e}")
        
        logger.info("=" * 80)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = layer.weight
        fp8_enabled = getattr(layer, '_ternary_fp8_enabled', False)
        i2s_enabled = getattr(layer, '_ternary_i2s_enabled', False)
        
        x_compute = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
        b_compute = None if bias is None else (bias if bias.dtype in (torch.float16, torch.bfloat16) else bias.to(x_compute.dtype))
        
        if i2s_enabled and weight.dtype == torch.uint8:
            N, K = layer._ternary_weight_shape
            out = _unpack_i2s_and_linear(x_compute, weight, layer.ternary_alpha, b_compute, K, x_compute.dtype)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        elif fp8_enabled and weight.dtype == torch.float8_e4m3fn:
            block_size_N, block_size_K = getattr(layer, '_ternary_block_size', (128, 128))
            N, K = getattr(layer, '_ternary_weight_shape', weight.shape)
            fp8_block_scales = layer.fp8_block_scales
            alpha = layer.ternary_alpha
            
            num_blocks_N = fp8_block_scales.shape[0]
            num_blocks_K = fp8_block_scales.shape[1]
            
            pad_N = (block_size_N - (N % block_size_N)) % block_size_N
            pad_K = (block_size_K - (K % block_size_K)) % block_size_K
            
            if pad_N > 0 or pad_K > 0:
                weight_padded = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))
            else:
                weight_padded = weight
            
            N_padded, K_padded = weight_padded.shape
            
            weight_blocks = weight_padded.view(
                num_blocks_N, block_size_N,
                num_blocks_K, block_size_K
            ).permute(0, 2, 1, 3).contiguous()
            
            FP8_MAX = 448.0
            scales_expanded = fp8_block_scales.unsqueeze(-1).unsqueeze(-1)
            weight_dequant = weight_blocks.float() * scales_expanded * FP8_MAX
            
            weight_dequant = weight_dequant.permute(0, 2, 1, 3).contiguous()
            weight_fp16 = weight_dequant.view(N_padded, K_padded)
            
            if pad_N > 0 or pad_K > 0:
                weight_fp16 = weight_fp16[:N, :K].contiguous()
            
            weight_fp16 = weight_fp16 * alpha.unsqueeze(0).to(dtype=x_compute.dtype)
            
            out = torch.matmul(x_compute, weight_fp16.t())
            if b_compute is not None:
                out = out + b_compute
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        else:
            out = torch.matmul(x_compute, weight.t())
            if b_compute is not None:
                out = out + b_compute
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
        
        return out
