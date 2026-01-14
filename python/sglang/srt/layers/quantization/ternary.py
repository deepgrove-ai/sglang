"""Ternary quantization method for SGLang.

Implements ternary quantization (weights in {-1, 0, 1} × alpha).

Features:
- 8× memory savings with 2-bit weight storage (i2s format)
- Per-column alpha scaling for accuracy
- Optimized CUDA kernels for decode and prefill
"""

import ctypes
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

# torch.compile compatibility
try:
    import torch._dynamo
    _dynamo_disable = torch._dynamo.disable
    torch._dynamo.config.suppress_errors = True
    def _is_dynamo_compiling():
        return torch._dynamo.is_compiling()
except (ImportError, AttributeError):
    def _dynamo_disable(fn):
        return fn
    def _is_dynamo_compiling():
        return False

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)

DEFAULT_PREFILL_SKIP_M = int(os.environ.get("TERNARY_PREFILL_SKIP_M", "8"))

SUPPORTED_V4_NK_SHAPES = {
    # Attention projection shapes
    (5120, 2048), (2048, 4096), (2048, 2048),
    (2560, 2048), (256, 2048),
    # MoE expert shapes (Qwen3: hidden=2048, intermediate=768)
    (768, 2048), (2048, 768), (1536, 2048),
    # MoE expert shapes (Klear 20B: hidden=2048, intermediate=896)
    (896, 2048), (2048, 896), (1792, 2048),
    # LM head
    (151936, 2048),
}


# ============================================================================
# CUDA Kernel Loading
# ============================================================================

_PTR = ctypes.c_void_p
_INT = ctypes.c_int
_FLOAT = ctypes.c_float
_SIZE_T = ctypes.c_size_t

def _setup_kernel(lib, name: str, argtypes: list) -> bool:
    """Setup ctypes function signature if kernel exists."""
    if hasattr(lib, name):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = _INT
        return True
    return False
    
# Global library handle
BITNET_LIB = None
BITNET_CUDA_AVAILABLE = False
_KERNEL_CAPS = {}
I2S_CUTLASS_LIB = None
I2S_CUTLASS_AVAILABLE = False
I2S_CUTLASS_HAS_ALPHA_PTR = False

def _load_bitnet_library():
    """Load the CUDA kernel library."""
    global BITNET_LIB, BITNET_CUDA_AVAILABLE, _KERNEL_CAPS
    
    lib_paths = [
        os.path.join(os.path.dirname(__file__), '../../../../../libternary_bitnet.so'),
        './libternary_bitnet.so',
        '/usr/local/lib/libternary_bitnet.so',
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            try:
                BITNET_LIB = ctypes.CDLL(lib_path)
                BITNET_CUDA_AVAILABLE = True
                logger.info(f"[TERNARY] Loaded CUDA kernels from {lib_path}")
                break
            except Exception as e:
                logger.debug(f"[TERNARY] Failed to load {lib_path}: {e}")
    
    if BITNET_LIB is None:
        logger.warning("[TERNARY] CUDA kernels not found - performance will be degraded")
        return
            
    # Setup kernel signatures
    # Linear kernels
    _setup_kernel(BITNET_LIB, 'bitlinear_int8xint2_v4_simple',
                  [_PTR]*6 + [_INT]*3 + [_PTR])
    _setup_kernel(BITNET_LIB, 'bitlinear_bf16xint2_v4_megafused',
                  [_PTR]*4 + [_INT]*3 + [_PTR])
    _setup_kernel(BITNET_LIB, 'bitlinear_rmsnorm_bf16xint2_v4_megafused',
                  [_PTR]*4 + [_FLOAT] + [_PTR]*3 + [_INT]*3 + [_PTR])
    _setup_kernel(BITNET_LIB, 'v4_batch_megafused_v2_launch',
                  [_PTR]*4 + [_INT]*3 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ladder_fp8xint2_v4_megafused',
                  [_PTR]*7 + [_INT]*3 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_quantize_activation_fast',
                  [_PTR]*4 + [_INT]*2 + [_PTR])
    
    # MoE kernels
    _setup_kernel(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_shared',
                  [_PTR]*5 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_batched',
                  [_PTR]*5 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_shared_silu',
                  [_PTR]*5 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_combine_parallel',
                  [_PTR]*6 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_combine_bf16x2',
                  [_PTR]*6 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_batched_combine_bf16_weights',
                  [_PTR]*7 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_fp8_silu',
                  [_PTR]*6 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'ternary_moe_fp8_combine',
                  [_PTR]*7 + [_INT]*4 + [_PTR])
    _setup_kernel(BITNET_LIB, 'moe_batched_gate_up_silu',
                  [_PTR]*5 + [_INT]*5 + [_PTR])
    _setup_kernel(BITNET_LIB, 'moe_batched_down_combine',
                  [_PTR]*7 + [_INT]*5 + [_PTR])
    
    # Build capability cache
    _KERNEL_CAPS = {
        'megafused': hasattr(BITNET_LIB, 'bitlinear_bf16xint2_v4_megafused'),
        'batch_megafused': hasattr(BITNET_LIB, 'v4_batch_megafused_v2_launch'),
        'fp8_megafused': hasattr(BITNET_LIB, 'ladder_fp8xint2_v4_megafused'),
        'rmsnorm_megafused': hasattr(BITNET_LIB, 'bitlinear_rmsnorm_bf16xint2_v4_megafused'),
        'act_quant': hasattr(BITNET_LIB, 'ternary_quantize_activation_fast'),
        'moe_megafused_shared': hasattr(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_shared'),
        'moe_megafused_batched': hasattr(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_batched'),
        'moe_shared_silu': hasattr(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_shared_silu'),
        'moe_combine_parallel': hasattr(BITNET_LIB, 'ternary_moe_combine_parallel'),
        'moe_combine_bf16x2': hasattr(BITNET_LIB, 'ternary_moe_combine_bf16x2'),
        'moe_combine_bf16_weights': hasattr(BITNET_LIB, 'ternary_moe_megafused_gemv_indexed_batched_combine_bf16_weights'),
        'moe_fp8_silu': hasattr(BITNET_LIB, 'ternary_moe_fp8_silu'),
        'moe_fp8_combine': hasattr(BITNET_LIB, 'ternary_moe_fp8_combine'),
        'moe_batched_gate_up': hasattr(BITNET_LIB, 'moe_batched_gate_up_silu'),
        'moe_batched_down': hasattr(BITNET_LIB, 'moe_batched_down_combine'),
    }
    _KERNEL_CAPS['has_moe_full_fusion'] = (
        _KERNEL_CAPS['moe_megafused_shared'] and 
        _KERNEL_CAPS['moe_shared_silu'] and
        (_KERNEL_CAPS['moe_combine_parallel'] or _KERNEL_CAPS['moe_combine_bf16x2'])
    )


def _load_i2s_cutlass_library():
    """Load the SM100 i2s CUTLASS fused kernel library."""
    global I2S_CUTLASS_LIB, I2S_CUTLASS_AVAILABLE, I2S_CUTLASS_HAS_ALPHA_PTR

    env_path = os.environ.get("SGLANG_I2S_CUTLASS_LIB", "").strip()
    lib_paths = []
    if env_path:
        lib_paths.append(env_path)
    lib_paths.extend([
        os.path.join(os.path.dirname(__file__), '../../../../../libternary_cutlass_sm100.so'),
        os.path.join(os.path.dirname(__file__), '../../../../../ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so'),
        './libternary_cutlass_sm100.so',
        '/usr/local/lib/libternary_cutlass_sm100.so',
    ])

    for lib_path in lib_paths:
        if not lib_path:
            continue
        if os.path.exists(lib_path):
            try:
                I2S_CUTLASS_LIB = ctypes.CDLL(lib_path)
                I2S_CUTLASS_AVAILABLE = True
                logger.info(f"[TERNARY] Loaded i2s CUTLASS kernels from {lib_path}")
                break
            except Exception as e:
                logger.debug(f"[TERNARY] Failed to load i2s CUTLASS {lib_path}: {e}")

    if I2S_CUTLASS_LIB is None:
        return

    # Setup signatures
    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_set_alpha_const"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_set_alpha_const
        fn.argtypes = [_PTR, _INT, _PTR]
        fn.restype = _INT

    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs
        fn.argtypes = [_PTR, _PTR, _PTR, _INT, _INT, _INT]
        fn.restype = _SIZE_T

    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_run_streamk"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_run_streamk
        fn.argtypes = [_PTR, _PTR, _PTR, _INT, _INT, _INT, _INT, _PTR, _SIZE_T, _PTR]
        fn.restype = _INT

    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_run_streamk_alpha"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_run_streamk_alpha
        fn.argtypes = [_PTR, _PTR, _PTR, _PTR, _INT, _INT, _INT, _INT, _PTR, _SIZE_T, _PTR]
        fn.restype = _INT
        I2S_CUTLASS_HAS_ALPHA_PTR = True

# Load library at module import
_load_bitnet_library()
_load_i2s_cutlass_library()

# Export for ternary_hook.py FP8 sticky mode detection
BITNET_CUDA_FP8_MEGA_FUSED_AVAILABLE = _KERNEL_CAPS.get('fp8_megafused', False)

# BitNet weight packer (optional)
BITNET_PACK_AVAILABLE = False
convert_weight_int8_to_int2 = None
try:
    import sys
    bitnet_gpu_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../../../../../BitNet/gpu'))
    if os.path.isdir(bitnet_gpu_path) and bitnet_gpu_path not in sys.path:
        sys.path.append(bitnet_gpu_path)
    from pack_weight import convert_weight_int8_to_int2 as _bitnet_pack_fn
    convert_weight_int8_to_int2 = _bitnet_pack_fn
    BITNET_PACK_AVAILABLE = True
except Exception:
    pass


# ============================================================================
# Quantization Utilities
# ============================================================================

def _fp8_bridge_available() -> bool:
    """Return whether the FP8 bridge path is enabled."""
    return os.environ.get("SGLANG_TERNARY_FP8_BRIDGE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _check_fp8_runtime() -> bool:
    """Return whether FP8 runtime support is available."""
    if not torch.cuda.is_available():
        return False
    try:
        torch.tensor([1.0], device="cuda").to(torch.float8_e4m3fn)
    except Exception:
        return False
    return True


def get_fp8_runtime_info() -> Dict[str, Optional[str]]:
    """Return a small dict describing FP8 runtime availability."""
    if not torch.cuda.is_available():
        return {"available": False, "sm_version": None}
    major, minor = torch.cuda.get_device_capability()
    return {
        "available": _check_fp8_runtime(),
        "sm_version": f"sm_{major}{minor}",
    }


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def _get_i2s_cutlass_splits(N: int) -> int:
    env = os.environ.get("SGLANG_TERNARY_I2S_SPLITS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 3 if N >= 4096 else 4


def quantize_alpha_int8(alpha: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize per-column alpha to int8 with global scaling."""
    alpha_max = alpha.abs().max().clamp(min=1e-8)
    alpha_scale = (alpha_max / 127.0).item()
    alpha_q = torch.round(alpha / alpha_scale).clamp(-128, 127).to(torch.int8)
    return alpha_q, alpha_scale


def pack_i2s_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """Pack ternary weights {-1, 0, 1} into 2-bit format (4 values per byte)."""
    N, K = weight_ternary.shape
    weight_mapped = (weight_ternary + 1).clamp(0, 2).to(torch.uint8)
    
    pad_K = (4 - (K % 4)) % 4
    if pad_K > 0:
        weight_mapped = F.pad(weight_mapped, (0, pad_K), value=1)
    
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
    """Unpack I2S weights to dense format."""
    N, num_packed_cols = weight_packed.shape
    device = weight_packed.device
    
    packed_expanded = weight_packed.unsqueeze(-1)
    shift_positions = torch.arange(4, device=device, dtype=torch.uint8) * 2
    extracted_all = (packed_expanded >> shift_positions.view(1, 1, -1)) & 0b11
    
    K_padded = num_packed_cols * 4
    if K_padded == K:
        extracted = extracted_all.reshape(N, K)
    else:
        extracted = extracted_all.reshape(N, K_padded)[:, :K]
    
    val_ternary = extracted.to(torch.float32) - 1.0
    weight_unpacked = (val_ternary * alpha.view(1, -1)).to(dtype)
    
    return weight_unpacked


def _get_fp16_fallback_weight(layer: nn.Module, dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
    """Get cached FP16 fallback weight."""
    base = getattr(layer, "_ternary_weight_fp16", None)
    if base is None:
        return None
    cache = getattr(layer, "_ternary_weight_fp16_cache", None)
    if cache is None:
        cache = {}
        setattr(layer, "_ternary_weight_fp16_cache", cache)
    key = (dtype, str(device))
    tensor = cache.get(key)
    if tensor is None:
        tensor = base.to(dtype) if base.dtype != dtype else base
        if tensor.device != device:
            tensor = tensor.to(device, non_blocking=True)
        cache[key] = tensor
    return tensor


def replace_parameter(layer: nn.Module, name: str, new_param: torch.Tensor) -> None:
    """Replace a parameter with a new tensor."""
    if isinstance(getattr(layer, name, None), Parameter):
        delattr(layer, name)
    layer.register_parameter(name, Parameter(new_param, requires_grad=False))


# ============================================================================
# TernaryConfig
# ============================================================================

@dataclass
class TernaryConfig(QuantizationConfig):
    """Config for ternary quantization."""

    threshold_scale: float = 0.7
    storage_mode: str = "i2s"
    use_fp8: bool = False
    use_bitnet_kernel: bool = True
    fp8_hidden_scale_granularity: str = "per_token_group_128"
    kv_cache_quant_algo: Optional[str] = None
    recommended_kv_cache_dtype: Optional[str] = None

    def __post_init__(self):
        if not (0.0 < self.threshold_scale < 1.0):
            raise ValueError("threshold_scale must be between 0 and 1.")
        self.storage_mode = self.storage_mode.lower()
        if self.storage_mode not in ("i2s", "fp16"):
            raise ValueError(f"storage_mode must be 'i2s' or 'fp16'")
        
        # Auto-detect FP8 from environment
        if not self.use_fp8:
            env_fp8 = os.environ.get("SGLANG_TERNARY_USE_FP8", "0")
            self.use_fp8 = env_fp8.strip().lower() in ("1", "true", "yes", "on")

        if self.use_fp8:
            self.kv_cache_quant_algo = "FP8"
            self.recommended_kv_cache_dtype = "fp8_e4m3"
        else:
            self.kv_cache_quant_algo = None
            self.recommended_kv_cache_dtype = None
    
    @property
    def fp8_group_size(self) -> int:
        return 128 if "group_128" in self.fp8_hidden_scale_granularity else -1

    @staticmethod
    def get_name() -> str:
        return "ternary"

    @staticmethod
    def get_min_capability() -> int:
        return 80  # Ampere and above

    @staticmethod
    def get_supported_act_dtypes() -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float8_e4m3fn]

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TernaryConfig":
        return cls(
            threshold_scale=config.get("threshold_scale", 0.7),
            storage_mode=config.get("storage_mode", "i2s"),
            use_fp8=config.get("use_fp8", False),
            use_bitnet_kernel=config.get("use_bitnet_kernel", True),
            fp8_hidden_scale_granularity=config.get("fp8_hidden_scale_granularity", "per_token_group_128"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional["LinearMethodBase"]:
        pref = (prefix or "").lower()
        
        # Skip embeddings
        if "embed" in pref:
            return None

        # Skip lm_head unless explicitly enabled
        if "lm_head" in pref and os.environ.get("TERNARY_QUANTIZE_LM_HEAD", "0") != "1":
            return None

        # Skip MoE gates but allow MLP gate_proj
        if "gate" in pref and "gate_proj" not in pref:
            return None

        layer_class_name = type(layer).__name__
        if layer_class_name in ("FusedMoE", "DeepEPMoE"):
            return TernaryFusedMoEMethod(self)
        
        return TernaryLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []


# ============================================================================
# TernaryLinearMethod
# ============================================================================

class TernaryLinearMethod(LinearMethodBase):
    """Linear method for ternary quantization."""

    def __init__(self, quant_config: TernaryConfig):
        self.quant_config = quant_config

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
            torch.empty(output_size_per_partition, input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, {
            "input_dim": 1,
            "output_dim": 0,
            "weight_loader": kwargs.get("weight_loader", self.weight_loader),
        })

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor, shard_id: Optional[str] = None):
        param.data.copy_(loaded_weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Apply ternary quantization to layer weights."""
        weight = layer.weight.data
        original_dtype = weight.dtype
        N, K = weight.shape
        device = weight.device

        logger.info(f"[TERNARY] Quantizing layer: {weight.shape}")

        # Compute ternary weights
        weight_fp32 = weight.float()
        absW = weight_fp32.abs()
        th = self.quant_config.threshold_scale * absW.mean(dim=0, keepdim=True)
        mask = absW > th
        mask_f = mask.float()
        alpha = (absW * mask_f).sum(dim=0, keepdim=True) / mask_f.sum(dim=0, keepdim=True).clamp(min=1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.full_like(alpha, 1e-6))

        if self.quant_config.storage_mode == "i2s":
            weight_ternary_sign = torch.where(mask, weight_fp32.sign(), torch.zeros_like(weight_fp32)).to(torch.int8)
            weight_packed = pack_i2s_weights(weight_ternary_sign.float())
            replace_parameter(layer, "weight", weight_packed)

            # Pack for BitNet kernel if available
            bitnet_packed = False
            if BITNET_PACK_AVAILABLE:
                try:
                    weight_bitnet = convert_weight_int8_to_int2(weight_ternary_sign).contiguous()
                    if device.type == "cuda":
                        weight_bitnet = weight_bitnet.to(device, non_blocking=True)
                    layer.register_buffer("ternary_weight_bitnet", weight_bitnet, persistent=False)
                    layer._ternary_weight_bitnet_ptr = weight_bitnet.data_ptr()
                    bitnet_packed = True
                except Exception as e:
                    logger.warning(f"[TERNARY] BitNet packing failed: {e}")

            alpha_flat = alpha.view(-1).to(torch.float32).contiguous()
            layer.register_buffer("ternary_alpha", alpha_flat, persistent=False)

            # Quantize alpha for V4 kernel
            if bitnet_packed and BITNET_CUDA_AVAILABLE and device.type == "cuda":
                alpha_q, alpha_scale = quantize_alpha_int8(alpha_flat)
                layer.register_buffer("ternary_alpha_q", alpha_q.contiguous(), persistent=False)
                layer.register_buffer(
                    "ternary_alpha_scale",
                    torch.tensor([alpha_scale], device=device, dtype=torch.float32),
                    persistent=False,
                )
                layer._ternary_alpha_q_ptr = layer.ternary_alpha_q.data_ptr()
                layer._ternary_alpha_scale_ptr = layer.ternary_alpha_scale.data_ptr()

            layer._ternary_bitnet_enabled = bitnet_packed
        else:
            # FP16 mode
            weight_ternary = weight_fp32.sign() * alpha * mask_f
            replace_parameter(layer, "weight", weight_ternary.to(original_dtype))
            layer.register_buffer("ternary_alpha", torch.ones(K, device=device, dtype=original_dtype), persistent=False)
            layer._ternary_bitnet_enabled = False

        layer._ternary_weight_shape = (N, K)
        layer._ternary_K = K
        layer._ternary_N = N
                
    @_dynamo_disable
    @torch.no_grad()
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply ternary linear transformation."""
        weight = layer.weight
        weight_shape = getattr(layer, "_ternary_weight_shape", None)
        
        if weight_shape is None:
            # Not quantized, use regular linear
            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
            return F.linear(x_bf16, weight, bias)
        
        N, K = weight_shape
        x_shape = x.shape
        M = x.reshape(-1, K).shape[0]
        
        # Check kernel eligibility
        bitnet_enabled = getattr(layer, '_ternary_bitnet_enabled', False)
        can_use_kernel = (
            not _is_dynamo_compiling()
            and bitnet_enabled
            and BITNET_CUDA_AVAILABLE
            and BITNET_LIB is not None
            and weight.dtype == torch.uint8
            and x.is_cuda
            and (N, K) in SUPPORTED_V4_NK_SHAPES
        )
        
        # Handle FP8 input
        x_is_fp8 = x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        x_fp8_scale = getattr(x, '_fp8_scale', None)
        
        # Convert to BF16 for compute
        if x_is_fp8:
            if x_fp8_scale is not None:
                x_bf16 = (x.to(torch.float32) * x_fp8_scale.view(-1, 1)).to(torch.bfloat16)
            else:
                x_bf16 = x.to(torch.bfloat16)
        else:
            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        
        bias_bf16 = bias.to(torch.bfloat16) if bias is not None and bias.dtype != torch.bfloat16 else bias
        stream = torch.cuda.current_stream().cuda_stream
        
        if can_use_kernel:
            # FP8 megafused M=1 path
            if M == 1 and x_is_fp8 and _KERNEL_CAPS.get('fp8_megafused'):
                result = self._apply_fp8_megafused(layer, x, x_fp8_scale, N, K, stream)
                if result is not None:
                    if not getattr(layer, "_ternary_fp8_megafused_logged", False):
                        layer._ternary_fp8_megafused_logged = True
                        logger.info(
                            f"[TERNARY FP8 LINEAR] ladder_fp8xint2_v4_megafused hit (M=1) N={N} K={K} dtype={x.dtype}"
                        )
                    if bias_bf16 is not None:
                        result = result + bias_bf16
                    return result.view(*x_shape[:-1], N)
            
            # BF16 megafused M=1 path
            if M == 1 and _KERNEL_CAPS.get('megafused'):
                result = self._apply_megafused(layer, x_bf16, N, K, stream)
                if result is not None:
                    if not getattr(layer, "_ternary_bf16_megafused_logged", False):
                        layer._ternary_bf16_megafused_logged = True
                        logger.info(
                            f"[TERNARY BF16 LINEAR] bitlinear_bf16xint2_v4_megafused hit (M=1) N={N} K={K} x_dtype={x.dtype}"
                        )
                    if bias_bf16 is not None:
                        result = result + bias_bf16
                    return result.view(*x_shape[:-1], N)
            
            # Batch megafused M>1 path
            if M > 1 and _KERNEL_CAPS.get('batch_megafused'):
                result = self._apply_batch_megafused(layer, x_bf16, M, N, K, stream)
                if result is not None:
                    if bias_bf16 is not None:
                        result = result + bias_bf16
                    return result.view(*x_shape[:-1], N)

        # CUTLASS fused i2s path (SM100, M>1 prefill)
        if weight.dtype == torch.uint8 and not x_is_fp8:
            x_bf16_2d = x_bf16.reshape(-1, K)
            result = self._apply_i2s_cutlass(layer, x_bf16_2d, bias_bf16, M, N, K, stream)
            if result is not None:
                return result.view(*x_shape[:-1], N)
        
        # FP16 fallback
        weight_fp16 = _get_fp16_fallback_weight(layer, torch.bfloat16, x.device)
        if weight_fp16 is None:
            if weight.dtype == torch.uint8:
                weight_fp16 = unpack_i2s_weights(weight, K, layer.ternary_alpha, torch.bfloat16)
            else:
                weight_fp16 = weight.to(torch.bfloat16)
            layer.register_buffer("_ternary_weight_fp16", weight_fp16, persistent=False)
        
        output = F.linear(x_bf16, weight_fp16, bias_bf16)
        return output.view(*x_shape[:-1], N)

    def _apply_fp8_megafused(self, layer, x, x_fp8_scale, N, K, stream):
        """FP8 megafused kernel for M=1."""
        x_in = x.reshape(-1, K).view(torch.uint8).contiguous()
        
        if x_fp8_scale is not None:
            scale_tensor = x_fp8_scale.view(-1)[:1].contiguous()
        else:
            if not hasattr(layer, '_fp8_scale_one'):
                layer._fp8_scale_one = torch.ones(1, device=x.device, dtype=torch.float32)
            scale_tensor = layer._fp8_scale_one
        
        # IMPORTANT: per-stream scratch/output buffer.
        # With concurrency, SGLang can execute different requests on different CUDA
        # streams. Caching a single tensor on the shared `layer` object causes
        # cross-request races and can corrupt outputs after a concurrency benchmark.
        stream_id = int(stream)
        cache = getattr(layer, "_ternary_stream_outputs", None)
        if cache is None:
            cache = {}
            setattr(layer, "_ternary_stream_outputs", cache)
        cache_key = ("fp8_megafused_out", stream_id, N)
        output = cache.get(cache_key)
        if output is None or output.numel() != N or output.device != x.device:
            output = torch.empty(1, N, device=x.device, dtype=torch.bfloat16)
            cache[cache_key] = output
        
        ret = BITNET_LIB.ladder_fp8xint2_v4_megafused(
            _PTR(x_in.data_ptr()),
            _PTR(scale_tensor.data_ptr()),
            _PTR(layer.ternary_alpha.data_ptr()),
            _PTR(layer._ternary_weight_bitnet_ptr),
            _PTR(output.data_ptr()),
            _PTR(0), _PTR(0),  # unused FP8 output
            _INT(1), _INT(N), _INT(K),
            _PTR(stream),
        )
        return output if ret == 0 else None

    def _apply_megafused(self, layer, x_bf16, N, K, stream):
        """BF16 megafused kernel for M=1."""
        x_in = x_bf16.reshape(-1, K).contiguous()
        
        # IMPORTANT: per-stream scratch/output buffer (see note in _apply_fp8_megafused).
        stream_id = int(stream)
        cache = getattr(layer, "_ternary_stream_outputs", None)
        if cache is None:
            cache = {}
            setattr(layer, "_ternary_stream_outputs", cache)
        cache_key = ("bf16_megafused_out", stream_id, N)
        output = cache.get(cache_key)
        if output is None or output.numel() != N or output.device != x_bf16.device:
            output = torch.empty(1, N, device=x_bf16.device, dtype=torch.bfloat16)
            cache[cache_key] = output
        
        ret = BITNET_LIB.bitlinear_bf16xint2_v4_megafused(
            _PTR(x_in.data_ptr()),
            _PTR(layer._ternary_weight_bitnet_ptr),
            _PTR(layer.ternary_alpha.data_ptr()),
            _PTR(output.data_ptr()),
            _INT(1), _INT(N), _INT(K),
            _PTR(stream),
        )
        return output if ret == 0 else None

    def _apply_batch_megafused(self, layer, x_bf16, M, N, K, stream):
        """Batch megafused kernel for M>1.
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! WARNING: THIS KERNEL HAS KNOWN CORRECTNESS ISSUES !!!
        !!! DO NOT USE IN PRODUCTION - RESULTS ARE INCORRECT FOR M>1 !!!
        !!! This is kept for reference only. Use FP16 fallback instead. !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        # DISABLED DUE TO CORRECTNESS ISSUES - always return None to use fallback
        return None
        
        x_in = x_bf16.reshape(-1, K).contiguous()
        
        # Use cached output buffer to avoid allocation during CUDA graph capture
        cache_key = f"_batch_output_{M}_{N}"
        output = getattr(layer, cache_key, None)
        if output is None:
            output = torch.empty(M, N, device=x_bf16.device, dtype=torch.bfloat16)
            setattr(layer, cache_key, output)
        
        ret = BITNET_LIB.v4_batch_megafused_v2_launch(
            _PTR(x_in.data_ptr()),
            _PTR(layer.ternary_alpha.data_ptr()),
            _PTR(layer._ternary_weight_bitnet_ptr),
            _PTR(output.data_ptr()),
            _INT(M), _INT(N), _INT(K),
            _PTR(stream),
        )
        return output if ret == 0 else None

    def _apply_i2s_cutlass(self, layer, x_bf16_2d, bias_bf16, M, N, K, stream):
        """SM100 CUTLASS fused i2s kernel (FP16 A, FP32 output)."""
        if not I2S_CUTLASS_AVAILABLE or I2S_CUTLASS_LIB is None:
            return None
        if not _is_sm100():
            return None
        if M <= DEFAULT_PREFILL_SKIP_M:
            return None
        if (N % 64) != 0 or (K % 4) != 0 or (K % 16) != 0 or K > 8192:
            return None
        if not _env_flag("SGLANG_TERNARY_USE_I2S_CUTLASS", "1"):
            return None

        # Convert input to FP16 and pad M to 16 for CUTLASS TMA stride constraints.
        x_fp16 = x_bf16_2d.to(torch.float16)
        if not x_fp16.is_contiguous():
            x_fp16 = x_fp16.contiguous()

        M_run = ((M + 15) // 16) * 16
        if M_run != M:
            cache = getattr(layer, "_ternary_i2s_cutlass_cache", None)
            if cache is None:
                cache = {}
                setattr(layer, "_ternary_i2s_cutlass_cache", cache)
            stream_id = int(stream)
            buf = cache.get(stream_id)
            if buf is None:
                buf = {}
                cache[stream_id] = buf
            x_pad = buf.get("x_pad")
            if (
                x_pad is None
                or x_pad.shape != (M_run, K)
                or x_pad.device != x_fp16.device
            ):
                x_pad = torch.empty((M_run, K), device=x_fp16.device, dtype=torch.float16)
                buf["x_pad"] = x_pad
            x_pad.zero_()
            x_pad[:M].copy_(x_fp16)
            x_fp16 = x_pad

        # Column-major output buffer (N, M_run) to match kernel layout.
        out_cm = torch.empty((N, M_run), device=x_fp16.device, dtype=torch.float32)

        ws_bytes = int(
            I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs(
                _PTR(x_fp16.data_ptr()),
                _PTR(layer.weight.data_ptr()),
                _PTR(out_cm.data_ptr()),
                _INT(M_run),
                _INT(N),
                _INT(K),
            )
        )
        workspace_ptr = _PTR(0)
        if ws_bytes > 0:
            cache = getattr(layer, "_ternary_i2s_cutlass_cache", None)
            if cache is None:
                cache = {}
                setattr(layer, "_ternary_i2s_cutlass_cache", cache)
            stream_id = int(stream)
            buf = cache.get(stream_id)
            if buf is None:
                buf = {}
                cache[stream_id] = buf
            ws_buf = buf.get("workspace")
            if (
                ws_buf is None
                or ws_buf.numel() < ws_bytes
                or ws_buf.device != x_fp16.device
            ):
                ws_buf = torch.empty(ws_bytes, device=x_fp16.device, dtype=torch.uint8)
                buf["workspace"] = ws_buf
            workspace_ptr = _PTR(ws_buf.data_ptr())

        splits = _get_i2s_cutlass_splits(N)
        if I2S_CUTLASS_HAS_ALPHA_PTR and hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_run_streamk_alpha"):
            rc = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_run_streamk_alpha(
                _PTR(x_fp16.data_ptr()),
                _PTR(layer.weight.data_ptr()),
                _PTR(layer.ternary_alpha.data_ptr()),
                _PTR(out_cm.data_ptr()),
                _INT(M_run),
                _INT(N),
                _INT(K),
                _INT(splits),
                workspace_ptr,
                _SIZE_T(ws_bytes),
                _PTR(stream),
            )
        else:
            if not hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_set_alpha_const"):
                return None
            rc = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_set_alpha_const(
                _PTR(layer.ternary_alpha.data_ptr()),
                _INT(K),
                _PTR(stream),
            )
            if rc == 0:
                rc = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_run_streamk(
                    _PTR(x_fp16.data_ptr()),
                    _PTR(layer.weight.data_ptr()),
                    _PTR(out_cm.data_ptr()),
                    _INT(M_run),
                    _INT(N),
                    _INT(K),
                    _INT(splits),
                    workspace_ptr,
                    _SIZE_T(ws_bytes),
                    _PTR(stream),
                )

        if rc != 0:
            return None

        if bias_bf16 is not None:
            out_cm.add_(bias_bf16.to(torch.float32).view(-1, 1))

        out = out_cm.t().contiguous()
        if M_run != M:
            out = out[:M]
        return out.to(torch.bfloat16)


# ============================================================================
# TernaryFusedMoEMethod
# ============================================================================

class TernaryFusedMoEMethod(FusedMoEMethodBase, nn.Module):
    """Fused MoE method using ternary quantization."""
    
    def __init__(self, quant_config: TernaryConfig):
        FusedMoEMethodBase.__init__(self)
        nn.Module.__init__(self)
        self.quant_config = quant_config
        
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # w13_weight: fused gate_up [num_experts, 2*intermediate, hidden]
        w13_weight = Parameter(
            torch.empty(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # w2_weight: down projection [num_experts, hidden, intermediate]
        w2_weight = Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        
        layer._ternary_moe_num_experts = num_experts
        layer._ternary_moe_hidden_size = hidden_size
        layer._ternary_moe_intermediate_size = intermediate_size_per_partition
        
    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config):
        layer._ternary_moe_runner_config = moe_runner_config
        
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Apply ternary quantization to MoE weights."""
        num_experts = layer.w13_weight.shape[0]
        device = layer.w13_weight.device
        dtype = layer.w13_weight.dtype
        hidden_size = layer.w13_weight.shape[2]
        intermediate_size = layer.w13_weight.shape[1] // 2
        
        logger.info(f"[TERNARY MoE] Quantizing {num_experts} experts, BITNET_CUDA_AVAILABLE={BITNET_CUDA_AVAILABLE}")
        
        # MoE uses our own pack_i2s_weights, not BitNet packer
        use_v4 = BITNET_CUDA_AVAILABLE
        logger.info(f"[TERNARY MoE] use_v4={use_v4}, hidden_size={hidden_size}, intermediate_size={intermediate_size}")
        
        # Process w13 weights
        w13 = layer.w13_weight.data.float()
        absW13 = w13.abs()
        th13 = self.quant_config.threshold_scale * absW13.mean(dim=1, keepdim=True)
        mask13 = absW13 > th13
        mask13_f = mask13.float()
        alpha13 = (absW13 * mask13_f).sum(dim=1, keepdim=True) / mask13_f.sum(dim=1, keepdim=True).clamp(min=1)
        alpha13 = torch.where(torch.isfinite(alpha13), alpha13, torch.full_like(alpha13, 1e-6))

        w13_ternary = w13.sign() * alpha13 * mask13_f
        layer.w13_weight.data.copy_(w13_ternary.to(dtype))
        
        # Process w2 weights
        w2 = layer.w2_weight.data.float()
        absW2 = w2.abs()
        th2 = self.quant_config.threshold_scale * absW2.mean(dim=1, keepdim=True)
        mask2 = absW2 > th2
        mask2_f = mask2.float()
        alpha2 = (absW2 * mask2_f).sum(dim=1, keepdim=True) / mask2_f.sum(dim=1, keepdim=True).clamp(min=1)
        alpha2 = torch.where(torch.isfinite(alpha2), alpha2, torch.full_like(alpha2, 1e-6))

        w2_ternary = w2.sign() * alpha2 * mask2_f
        layer.w2_weight.data.copy_(w2_ternary.to(dtype))
        
        if use_v4:
            # Pack weights for V4 kernel
            pad_w13 = (4 - (hidden_size % 4)) % 4
            num_packed_cols_w13 = (hidden_size + pad_w13) // 4
            pad_w2 = (4 - (intermediate_size % 4)) % 4
            num_packed_cols_w2 = (intermediate_size + pad_w2) // 4

            w13_packed = torch.empty(num_experts, 2 * intermediate_size, num_packed_cols_w13, device=device, dtype=torch.uint8)
            w2_packed = torch.empty(num_experts, hidden_size, num_packed_cols_w2, device=device, dtype=torch.uint8)
            
            for e in range(num_experts):
                w13_sign = torch.where(mask13[e], w13[e].sign(), torch.zeros_like(w13[e])).to(torch.int8)
                w13_packed[e] = pack_i2s_weights(w13_sign.float())
                
                w2_sign = torch.where(mask2[e], w2[e].sign(), torch.zeros_like(w2[e])).to(torch.int8)
                w2_packed[e] = pack_i2s_weights(w2_sign.float())
            
            layer.register_buffer('_ternary_w13_packed', w13_packed.contiguous(), persistent=False)
            layer.register_buffer('_ternary_w2_packed', w2_packed.contiguous(), persistent=False)
            layer.register_buffer('_ternary_moe_alpha_w13', alpha13.view(num_experts, hidden_size).to(torch.float32).contiguous(), persistent=False)
            layer.register_buffer('_ternary_moe_alpha_w2', alpha2.view(num_experts, intermediate_size).to(torch.float32).contiguous(), persistent=False)
            
            # IMPORTANT: Do NOT cache a single set of scratch buffers on the shared layer.
            # Under concurrency, multiple requests can run on different CUDA streams and
            # overwrite these buffers, corrupting outputs (seen as garbage tokens after a
            # concurrency benchmark).
            #
            # We instead create per-stream scratch buffers lazily in the decode call path.
            max_top_k = 8
            N_w13 = 2 * intermediate_size
            layer._ternary_moe_max_top_k = max_top_k
            layer._ternary_moe_intermediate_size = intermediate_size
            layer._ternary_moe_hidden_size = hidden_size
            layer._ternary_moe_scratch_cache = {}  # (stream_id) -> dict of buffers
            
            # Cache ctypes pointers for read-only weights/scales (safe across streams)
            layer._ctypes_w13_packed = _PTR(layer._ternary_w13_packed.data_ptr())
            layer._ctypes_w2_packed = _PTR(layer._ternary_w2_packed.data_ptr())
            layer._ctypes_alpha_w13 = _PTR(layer._ternary_moe_alpha_w13.data_ptr())
            layer._ctypes_alpha_w2 = _PTR(layer._ternary_moe_alpha_w2.data_ptr())
            layer._ctypes_N_w13 = _INT(N_w13)
            layer._ctypes_K_w13 = _INT(hidden_size)
            layer._ctypes_N_w2 = _INT(hidden_size)
            layer._ctypes_K_w2 = _INT(intermediate_size)
            layer._ctypes_num_experts = _INT(num_experts)
            
            # Check full fusion eligibility
            is_supported_shape = (
                (N_w13 == 1536 and hidden_size == 2048) or  # Qwen3 MoE
                (N_w13 == 1792 and hidden_size == 2048)     # Klear 20B
            )
            layer._use_full_fusion = _KERNEL_CAPS.get('has_moe_full_fusion', False) and is_supported_shape
            layer._ternary_moe_v4_enabled = True
            logger.info(f"[TERNARY MoE] V4 enabled, full_fusion={layer._use_full_fusion}")
        else:
            layer._ternary_moe_v4_enabled = False
            layer._use_full_fusion = False
        
        layer._ternary_moe_enabled = True
        
    @_dynamo_disable
    def apply(self, layer: torch.nn.Module, dispatch_output):
        """Apply ternary MoE forward pass."""
        # Import MoE utilities
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe, MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = getattr(layer, '_ternary_moe_runner_config', None) or MoeRunnerConfig()
        
        # Check format
        if not hasattr(topk_output, 'topk_weights') or not hasattr(topk_output, 'topk_ids'):
            return StandardCombineInput(hidden_states=fused_moe(
                hidden_states=hidden_states,
                w1=layer.w13_weight, w2=layer.w2_weight,
                topk_output=topk_output,
                moe_runner_config=moe_runner_config,
            ))
        
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids
        num_tokens = hidden_states.shape[0]
        
        # Debug: log first call
        if not getattr(layer, '_moe_debug_logged', False):
            layer._moe_debug_logged = True
            logger.info(f"[TERNARY MoE DEBUG] num_tokens={num_tokens}, dtype={hidden_states.dtype}, "
                       f"_use_full_fusion={getattr(layer, '_use_full_fusion', False)}, "
                       f"_ternary_moe_v4_enabled={getattr(layer, '_ternary_moe_v4_enabled', False)}, "
                       f"has_moe_full_fusion={_KERNEL_CAPS.get('has_moe_full_fusion', False)}, "
                       f"moe_fp8_silu={_KERNEL_CAPS.get('moe_fp8_silu', False)}")
        
        # Separate decode debug (cuda graph capture logs are usually batched/prefill)
        if num_tokens == 1 and not getattr(layer, "_moe_decode_debug_logged", False):
            layer._moe_decode_debug_logged = True
            logger.info(
                f"[TERNARY MoE DECODE DEBUG] dtype={hidden_states.dtype} "
                f"use_full_fusion={getattr(layer, '_use_full_fusion', False)} "
                f"moe_fp8_silu={_KERNEL_CAPS.get('moe_fp8_silu', False)}"
            )
        
        # M=1 decode with full fusion
        x_is_fp8 = hidden_states.dtype == torch.float8_e4m3fn
        if num_tokens == 1 and getattr(layer, '_use_full_fusion', False):
            if x_is_fp8 and _KERNEL_CAPS.get('moe_fp8_silu'):
                # FP8 decode path
                result = self._apply_decode_fused_fp8(layer, hidden_states, topk_ids, topk_weights)
                if result is not None:
                    return StandardCombineInput(hidden_states=result)
            elif hidden_states.dtype == torch.bfloat16:
                # BF16 decode path
                result = self._apply_decode_fused(layer, hidden_states, topk_ids, topk_weights)
                if result is not None:
                    return StandardCombineInput(hidden_states=result)
                else:
                    if not getattr(layer, '_decode_fused_fail_logged', False):
                        layer._decode_fused_fail_logged = True
                        logger.warning(f"[TERNARY MoE] _apply_decode_fused returned None, using fallback")
        
        # M>1 batched CUDA kernels - DISABLED due to correctness issues
        # Always use fused_moe fallback for M>1
        
        # Log first M>1 call for debugging
        if num_tokens > 1 and not getattr(layer, '_moe_batched_fallback_logged', False):
            layer._moe_batched_fallback_logged = True
            logger.info(f"[TERNARY MoE] M>1 fallback: num_tokens={num_tokens}, dtype={hidden_states.dtype}")
        
        # Fallback to fused_moe - always convert to BF16 for safety
        if hidden_states.dtype == torch.bfloat16:
            hidden_states_for_moe = hidden_states
        else:
            # For FP8 or other dtypes, convert to BF16
            # Use .clone() to ensure we don't have any aliasing issues
            hidden_states_for_moe = hidden_states.to(torch.bfloat16).clone()
        
        return StandardCombineInput(hidden_states=fused_moe(
            hidden_states=hidden_states_for_moe,
            w1=layer.w13_weight, w2=layer.w2_weight,
            topk_output=topk_output,
            moe_runner_config=moe_runner_config,
        ))

    def _apply_decode_fused(self, layer, hidden_states, topk_ids, topk_weights):
        """Fully fused decode path for M=1."""
        stream = _PTR(torch.cuda.current_stream().cuda_stream)
        stream_id = int(stream.value) if hasattr(stream, "value") else int(stream)
        top_k = topk_ids.shape[1]

        # Per-stream scratch buffers to avoid cross-request corruption
        scratch = getattr(layer, "_ternary_moe_scratch_cache", None)
        if scratch is None:
            scratch = {}
            setattr(layer, "_ternary_moe_scratch_cache", scratch)
        buf = scratch.get(stream_id)
        if buf is None:
            max_top_k = int(getattr(layer, "_ternary_moe_max_top_k", 8))
            intermediate_size = int(getattr(layer, "_ternary_moe_intermediate_size"))
            hidden_size = int(getattr(layer, "_ternary_moe_hidden_size"))
            buf = {
                "intermediate": torch.empty(max_top_k, intermediate_size, device=hidden_states.device, dtype=torch.bfloat16),
                "combined": torch.empty(hidden_size, device=hidden_states.device, dtype=torch.bfloat16),
                "topk_w": torch.empty(max_top_k, device=hidden_states.device, dtype=torch.bfloat16),
            }
            scratch[stream_id] = buf

        intermediate_ptr = _PTR(buf["intermediate"].data_ptr())
        combined_ptr = _PTR(buf["combined"].data_ptr())
        topk_w_ptr_bf16 = _PTR(buf["topk_w"].data_ptr())
        
        expert_ids = topk_ids[0].to(torch.int32).contiguous()
        expert_ids_ptr = _PTR(expert_ids.data_ptr())
        top_k_int = _INT(top_k)
        
        x_row = hidden_states[0:1].contiguous()
        x_row_ptr = _PTR(x_row.data_ptr())
        
        # gate_up + silu
        ret = BITNET_LIB.ternary_moe_megafused_gemv_indexed_shared_silu(
            x_row_ptr,
            layer._ctypes_w13_packed,
            expert_ids_ptr,
            layer._ctypes_alpha_w13,
            intermediate_ptr,
            top_k_int,
            layer._ctypes_N_w13,
            layer._ctypes_K_w13,
            layer._ctypes_num_experts,
            stream,
        )
        if ret != 0:
            if not getattr(layer, '_silu_fail_logged', False):
                layer._silu_fail_logged = True
                logger.warning(f"[TERNARY MoE] gate_up+silu kernel returned {ret}")
            return None
        
        # Prepare weights
        w = topk_weights[0]
        w_bf16_buf = buf["topk_w"][:top_k]
        if w.dtype != torch.bfloat16:
            w_bf16_buf.copy_(w)
            w_ptr = topk_w_ptr_bf16
        else:
            w_ptr = _PTR(w.data_ptr())
                    
        # down + combine
        if _KERNEL_CAPS.get('moe_combine_parallel') and top_k == 8:
            ret = BITNET_LIB.ternary_moe_combine_parallel(
                intermediate_ptr,
                layer._ctypes_w2_packed,
                expert_ids_ptr,
                layer._ctypes_alpha_w2,
                w_ptr,
                combined_ptr,
                top_k_int,
                layer._ctypes_N_w2,
                layer._ctypes_K_w2,
                layer._ctypes_num_experts,
                stream,
            )
        elif _KERNEL_CAPS.get('moe_combine_bf16x2'):
            ret = BITNET_LIB.ternary_moe_combine_bf16x2(
                intermediate_ptr,
                layer._ctypes_w2_packed,
                expert_ids_ptr,
                layer._ctypes_alpha_w2,
                w_ptr,
                combined_ptr,
                top_k_int,
                layer._ctypes_N_w2,
                layer._ctypes_K_w2,
                layer._ctypes_num_experts,
                stream,
            )
        else:
            if not getattr(layer, '_combine_none_logged', False):
                layer._combine_none_logged = True
                logger.warning(f"[TERNARY MoE] No combine kernel available, top_k={top_k}, "
                              f"combine_parallel={_KERNEL_CAPS.get('moe_combine_parallel')}, "
                              f"combine_bf16x2={_KERNEL_CAPS.get('moe_combine_bf16x2')}")
            return None
        
        if ret != 0:
            if not getattr(layer, '_combine_fail_logged', False):
                layer._combine_fail_logged = True
                logger.warning(f"[TERNARY MoE] combine kernel returned {ret}")
            return None
        
        return buf["combined"].view(1, -1)

    def _apply_decode_fused_fp8(self, layer, hidden_states, topk_ids, topk_weights):
        """FP8 decode path for M=1."""
        stream = _PTR(torch.cuda.current_stream().cuda_stream)
        stream_id = int(stream.value) if hasattr(stream, "value") else int(stream)
        top_k = topk_ids.shape[1]

        # Per-stream scratch buffers to avoid cross-request corruption
        scratch = getattr(layer, "_ternary_moe_scratch_cache", None)
        if scratch is None:
            scratch = {}
            setattr(layer, "_ternary_moe_scratch_cache", scratch)
        buf = scratch.get(stream_id)
        if buf is None:
            max_top_k = int(getattr(layer, "_ternary_moe_max_top_k", 8))
            intermediate_size = int(getattr(layer, "_ternary_moe_intermediate_size"))
            hidden_size = int(getattr(layer, "_ternary_moe_hidden_size"))
            buf = {
                "intermediate": torch.empty(max_top_k, intermediate_size, device=hidden_states.device, dtype=torch.bfloat16),
                "combined": torch.empty(hidden_size, device=hidden_states.device, dtype=torch.bfloat16),
                "topk_w": torch.empty(max_top_k, device=hidden_states.device, dtype=torch.bfloat16),
            }
            scratch[stream_id] = buf

        intermediate_ptr = _PTR(buf["intermediate"].data_ptr())
        combined_ptr = _PTR(buf["combined"].data_ptr())
        topk_w_ptr_bf16 = _PTR(buf["topk_w"].data_ptr())
        
        expert_ids = topk_ids[0].to(torch.int32).contiguous()
        expert_ids_ptr = _PTR(expert_ids.data_ptr())
        top_k_int = _INT(top_k)
        
        # FP8 input - reinterpret as uint8
        x_fp8 = hidden_states[0:1].view(torch.uint8).contiguous()
        x_fp8_ptr = _PTR(x_fp8.data_ptr())
        
        # Get FP8 scale (attached to tensor or default to 1.0)
        fp8_scale = getattr(hidden_states, '_fp8_scale', None)
        if fp8_scale is not None:
            scale_ptr = _PTR(fp8_scale.view(-1)[:1].contiguous().data_ptr())
        else:
            if not hasattr(layer, '_fp8_moe_scale_one'):
                layer._fp8_moe_scale_one = torch.ones(1, device=hidden_states.device, dtype=torch.float32)
            scale_ptr = _PTR(layer._fp8_moe_scale_one.data_ptr())
        
        # gate_up + silu with FP8 input -> BF16 intermediate
        ret = BITNET_LIB.ternary_moe_fp8_silu(
            x_fp8_ptr,
            scale_ptr,
            layer._ctypes_w13_packed,
            expert_ids_ptr,
            layer._ctypes_alpha_w13,
            intermediate_ptr,
            top_k_int,
            layer._ctypes_N_w13,
            layer._ctypes_K_w13,
            layer._ctypes_num_experts,
            stream,
        )
        if ret != 0:
            if not getattr(layer, '_fp8_silu_fail_logged', False):
                layer._fp8_silu_fail_logged = True
                logger.warning(f"[TERNARY MoE] FP8 silu kernel returned {ret}")
            return None
        
        # Prepare weights for combine
        w = topk_weights[0]
        w_bf16_buf = buf["topk_w"][:top_k]
        if w.dtype != torch.bfloat16:
            w_bf16_buf.copy_(w)
            w_ptr = topk_w_ptr_bf16
        else:
            w_ptr = _PTR(w.data_ptr())
        
        # down + combine (same as BF16 path - intermediate is BF16)
        if _KERNEL_CAPS.get('moe_combine_parallel') and top_k == 8:
            ret = BITNET_LIB.ternary_moe_combine_parallel(
                intermediate_ptr,
                layer._ctypes_w2_packed,
                expert_ids_ptr,
                layer._ctypes_alpha_w2,
                w_ptr,
                combined_ptr,
                top_k_int,
                layer._ctypes_N_w2,
                layer._ctypes_K_w2,
                layer._ctypes_num_experts,
                stream,
            )
        elif _KERNEL_CAPS.get('moe_combine_bf16x2'):
            ret = BITNET_LIB.ternary_moe_combine_bf16x2(
                intermediate_ptr,
                layer._ctypes_w2_packed,
                expert_ids_ptr,
                layer._ctypes_alpha_w2,
                w_ptr,
                combined_ptr,
                top_k_int,
                layer._ctypes_N_w2,
                layer._ctypes_K_w2,
                layer._ctypes_num_experts,
                stream,
                    )
        else:
            return None
                
        if ret != 0:
            return None
        
        return buf["combined"].view(1, -1)

    def _apply_batched(self, layer, hidden_states, topk_ids, topk_weights, num_tokens):
        """Batched MoE path for M>1 (prefill).
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! WARNING: THIS KERNEL HAS KNOWN CORRECTNESS ISSUES !!!
        !!! DO NOT USE IN PRODUCTION - RESULTS ARE INCORRECT FOR M>1 !!!
        !!! This is kept for reference only. Use fused_moe fallback instead. !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        # DISABLED DUE TO CORRECTNESS ISSUES - always return None to use fallback
        return None
        
        if not _KERNEL_CAPS.get('moe_batched_gate_up') or not _KERNEL_CAPS.get('moe_batched_down'):
            return None
        
        stream = _PTR(torch.cuda.current_stream().cuda_stream)
        top_k = topk_ids.shape[1]
        hidden_size = layer._ternary_moe_hidden_size
        intermediate_size = layer._ternary_moe_intermediate_size
        num_experts = layer._ternary_moe_num_experts
        
        # Allocate/get batched buffers
        buf_key = f"_batched_buf_{num_tokens}"
        if not hasattr(layer, buf_key):
            setattr(layer, buf_key, {
                'intermediate': torch.empty(num_tokens, top_k, intermediate_size, device=hidden_states.device, dtype=torch.bfloat16),
                'output': torch.empty(num_tokens, hidden_size, device=hidden_states.device, dtype=torch.bfloat16),
                'acc': torch.empty(num_tokens, hidden_size, device=hidden_states.device, dtype=torch.float32),
            })
        bufs = getattr(layer, buf_key)
        intermediate_buf = bufs['intermediate']
        output_buf = bufs['output']
        acc_buf = bufs['acc']
        
        # gate_up + silu for all tokens
        ret = BITNET_LIB.moe_batched_gate_up_silu(
            _PTR(hidden_states.data_ptr()),
            _PTR(layer._ternary_w13_packed.data_ptr()),
            _PTR(topk_ids.to(torch.int32).contiguous().data_ptr()),
            _PTR(layer._ternary_moe_alpha_w13.data_ptr()),
            _PTR(intermediate_buf.data_ptr()),
            _INT(num_tokens),
            _INT(top_k),
            _INT(2 * intermediate_size),
            _INT(hidden_size),
            _INT(num_experts),
            stream,
        )
        if ret != 0:
            return None
        
        # down + combine for all tokens
        ret = BITNET_LIB.moe_batched_down_combine(
            _PTR(intermediate_buf.data_ptr()),
            _PTR(layer._ternary_w2_packed.data_ptr()),
            _PTR(topk_ids.to(torch.int32).contiguous().data_ptr()),
            _PTR(layer._ternary_moe_alpha_w2.data_ptr()),
            _PTR(topk_weights.to(torch.bfloat16).contiguous().data_ptr()),
            _PTR(acc_buf.data_ptr()),
            _PTR(output_buf.data_ptr()),
            _INT(num_tokens),
            _INT(top_k),
            _INT(hidden_size),
            _INT(intermediate_size),
            _INT(num_experts),
            stream,
        )
        if ret != 0:
            return None
        
        return output_buf


__all__ = ["TernaryConfig", "TernaryLinearMethod", "TernaryFusedMoEMethod"]
