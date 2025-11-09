"""Ternary quantization with FP8 blockwise storage for SGLang.

This implementation:
- Per-column ternary quantization with alpha scaling (matching qwen2_correct.py)
- FP8 blockwise storage for 2x memory reduction
- Runtime weight-only quantization (applied after model loading)
- Forward pass: x (FP16) * W (FP8) * scale → equivalent to FP16 matmul

Key features:
- Per-column threshold-based ternary quantization
- Per-column alpha scales (matching qwen2_correct.py)
- Blockwise FP8 quantization (128x128 blocks) for memory efficiency
- Standard FP16 matmul with dequantized weights (or FP8 kernels if available)
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
    quantization_mode: str,  # "fp8" or "i2s"
    fp8_block_scales: Optional[torch.Tensor] = None,
    fp8_block_size: Optional[tuple] = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    num_test_samples: int = 10,
) -> bool:
    """
    Validate that quantized weights produce the same outputs as FP16 weights.
    
    Args:
        weight_fp16: Original FP16/BF16 weight tensor (N, K)
        weight_quantized: Quantized weight (FP8, uint8, or BF16)
        alpha: Per-column alpha scales (K,)
        quantization_mode: "fp8" or "i2s"
        fp8_block_scales: Blockwise scales for FP8 mode (num_blocks_N, num_blocks_K)
        fp8_block_size: Block size for FP8 mode (block_N, block_K)
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        num_test_samples: Number of random test inputs to use
    
    Returns:
        True if validation passes, False otherwise
    """
    device = weight_fp16.device
    dtype = weight_fp16.dtype
    N, K = weight_fp16.shape
    
    # Generate random test inputs
    torch.manual_seed(42)  # For reproducibility
    test_inputs = [
        torch.randn(1, K, device=device, dtype=dtype) for _ in range(num_test_samples)
    ]
    
    # Compute reference outputs using FP16 weights
    reference_outputs = []
    for x in test_inputs:
        y_ref = torch.nn.functional.linear(x, weight_fp16, None)
        reference_outputs.append(y_ref)
    
    # Compute outputs using quantized weights
    quantized_outputs = []
    
    if quantization_mode == "i2s":
        # I2_S path: unpack and apply alpha scaling
        from sglang.srt.layers.quantization.ternary import unpack_i2s_weights, _I2S_LOOKUP_TABLE
        
        # Get lookup table
        lookup_table = _I2S_LOOKUP_TABLE.to(device=device, dtype=dtype)
        
        for x in test_inputs:
            # Unpack I2_S weights
            weight_unpacked = unpack_i2s_weights(
                weight_packed=weight_quantized,
                K=K,
                alpha=alpha,
                device=device,
                dtype=dtype,
                lookup_table=lookup_table
            )
            y_quant = torch.nn.functional.linear(x, weight_unpacked, None)
            quantized_outputs.append(y_quant)
    
    elif quantization_mode == "fp8":
        # FP8 path: dequantize using blockwise scales
        block_size_N, block_size_K = fp8_block_size
        num_blocks_N = fp8_block_scales.shape[0]
        num_blocks_K = fp8_block_scales.shape[1]
        
        # Dequantize FP8 weights
        pad_N = (block_size_N - (N % block_size_N)) % block_size_N
        pad_K = (block_size_K - (K % block_size_K)) % block_size_K
        
        if pad_N > 0 or pad_K > 0:
            weight_padded = torch.nn.functional.pad(weight_quantized, (0, pad_K, 0, pad_N))
        else:
            weight_padded = weight_quantized
        
        N_padded, K_padded = weight_padded.shape
        
        # Reshape into blocks
        weight_blocks = weight_padded.view(
            num_blocks_N, block_size_N,
            num_blocks_K, block_size_K
        ).permute(0, 2, 1, 3).contiguous()
        
        # Dequantize: weight_block * scale_block
        FP8_MAX = 448.0
        scales_expanded = fp8_block_scales.unsqueeze(-1).unsqueeze(-1)  # (num_blocks_N, num_blocks_K, 1, 1)
        weight_dequant = weight_blocks.float() * scales_expanded * FP8_MAX
        
        # Reshape back
        weight_dequant = weight_dequant.permute(0, 2, 1, 3).contiguous()
        weight_dequant = weight_dequant.view(N_padded, K_padded)
        
        # Remove padding
        if pad_N > 0 or pad_K > 0:
            weight_dequant = weight_dequant[:N, :K]
        
        # Apply per-column alpha scaling
        weight_dequant = weight_dequant * alpha.unsqueeze(0).to(dtype=dtype)
        
        for x in test_inputs:
            y_quant = torch.nn.functional.linear(x, weight_dequant, None)
            quantized_outputs.append(y_quant)
    
    else:
        logger.error(f"Unknown quantization mode: {quantization_mode}")
        return False
    
    # Compare outputs
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
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_tensor_memory_bytes(tensor: torch.Tensor) -> int:
    """Get memory usage of a tensor in bytes."""
    return tensor.numel() * tensor.element_size()


def get_layer_memory_bytes(layer: torch.nn.Module) -> int:
    """Get total memory usage of layer parameters and buffers."""
    total = 0
    for param in layer.parameters():
        total += get_tensor_memory_bytes(param)
    for buffer in layer.buffers():
        total += get_tensor_memory_bytes(buffer)
    return total


# I2_S (Int2 Super-packed) lookup table for fast unpacking
# Maps 2-bit values {00, 01, 10, 11} → ternary values {-1, 0, 1, 0}
# Packing: -1→00, 0→01, 1→10, unused→11→0
_I2S_LOOKUP_TABLE = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float32)


if TRITON_AVAILABLE:
    @triton.jit
    def _i2s_unpack_kernel(
        packed_ptr,  # (N, ceil(K/4)) uint8 packed weights
        alpha_ptr,  # (K,) per-column alpha scales
        output_ptr,  # (N, K) output unpacked and scaled weights
        N,  # Number of output features
        K,  # Number of input features (original, before padding)
        num_packed_cols,  # ceil(K/4)
        stride_packed_n,
        stride_packed_k,
        stride_alpha,
        stride_output_n,
        stride_output_k,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  # Should be multiple of 4 for best performance
    ):
        """Optimized Triton kernel to unpack I2_S weights and apply alpha scaling.
        
        For each packed byte, extracts 4 2-bit values:
        - val0 = (packed >> 0) & 0b11
        - val1 = (packed >> 2) & 0b11
        - val2 = (packed >> 4) & 0b11
        - val3 = (packed >> 6) & 0b11
        
        Lookup table [-1, 0, 1, 0] is hardcoded:
        - 0 → -1
        - 1 → 0
        - 2 → 1
        - 3 → 0
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        
        # Compute offsets
        n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        k_offsets = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        
        # Masks
        n_mask = n_offsets < N
        k_mask = k_offsets < K
        
        # Load packed bytes (each byte contains 4 weights)
        packed_k_idx = k_offsets // 4  # Which packed byte
        packed_byte_offsets = n_offsets[:, None] * stride_packed_n + packed_k_idx[None, :] * stride_packed_k
        packed_mask = n_mask[:, None] & (packed_k_idx[None, :] < num_packed_cols)
        packed_bytes = tl.load(packed_ptr + packed_byte_offsets, mask=packed_mask, other=0)
        
        # Extract 4 values from each byte
        # For each k_offsets[i], determine which bit position in the byte
        bit_pos_in_byte = k_offsets % 4  # 0, 1, 2, or 3
        shift_amounts = bit_pos_in_byte * 2  # 0, 2, 4, or 6
        
        # Extract 2-bit values: (packed >> shift) & 0b11
        # Broadcast packed_bytes to match k_offsets
        packed_expanded = tl.broadcast_to(packed_bytes, (BLOCK_SIZE_N, BLOCK_SIZE_K))
        shift_expanded = tl.broadcast_to(shift_amounts[None, :], (BLOCK_SIZE_N, BLOCK_SIZE_K))
        extracted_values = (packed_expanded >> shift_expanded) & 0b11
        
        # Hardcoded lookup table: [-1, 0, 1, 0]
        # 0 → -1, 1 → 0, 2 → 1, 3 → 0
        extracted_indices = extracted_values.to(tl.int32)
        val_ternary = tl.where(
            extracted_indices == 0, -1.0,
            tl.where(
                extracted_indices == 2, 1.0,
                0.0  # extracted_indices == 1 or 3
            )
        )
        
        # Load alpha scales and apply directly (no redundant broadcast)
        alpha_values = tl.load(alpha_ptr + k_offsets * stride_alpha, mask=k_mask, other=1.0)
        
        # Apply alpha scaling: val_ternary * alpha (broadcasting happens automatically)
        output_values = val_ternary * alpha_values[None, :]
        
        # Write output
        output_mask = n_mask[:, None] & k_mask[None, :]
        output_offsets = n_offsets[:, None] * stride_output_n + k_offsets[None, :] * stride_output_k
        tl.store(output_ptr + output_offsets, output_values, mask=output_mask)


def pack_i2s_weights(weight_ternary: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary weights into I2_S format (2 bits per weight).
    
    Packing scheme:
    - -1 → 00 (0)
    -  0 → 01 (1)
    - +1 → 10 (2)
    - Unused → 11 (3) → mapped to 0
    
    Stores 4 weights per byte (2 bits each).
    
    Args:
        weight_ternary: (N, K) tensor with values in {-1, 0, 1}
    
    Returns:
        weight_packed: (N, ceil(K/4)) uint8 tensor with packed weights
    """
    N, K = weight_ternary.shape
    
    # Map {-1, 0, 1} to {0, 1, 2}
    weight_mapped = (weight_ternary + 1).clamp(0, 2).to(torch.uint8)
    
    # Pad K to be divisible by 4
    pad_K = (4 - (K % 4)) % 4
    if pad_K > 0:
        weight_mapped = torch.nn.functional.pad(weight_mapped, (0, pad_K), value=1)  # Pad with 0 (mapped from -1)
    
    K_padded = K + pad_K
    num_packed_cols = K_padded // 4
    
    # Reshape to (N, num_packed_cols, 4) for packing
    weight_reshaped = weight_mapped.view(N, num_packed_cols, 4)
    
    # Pack 4 values into 1 byte: val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)
    weight_packed = (
        weight_reshaped[:, :, 0] |
        (weight_reshaped[:, :, 1] << 2) |
        (weight_reshaped[:, :, 2] << 4) |
        (weight_reshaped[:, :, 3] << 6)
    ).to(torch.uint8)
    
    return weight_packed


def unpack_i2s_weights(weight_packed: torch.Tensor, K: int, alpha: torch.Tensor, 
                       device: torch.device, dtype: torch.dtype = torch.bfloat16,
                       lookup_table: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Unpack I2_S weights using fast lookup table and apply alpha scaling.
    
    Uses Triton kernel if available for better performance, otherwise falls back
    to vectorized PyTorch operations.
    
    Args:
        weight_packed: (N, ceil(K/4)) uint8 tensor with packed weights
        K: Original number of input features (before padding)
        alpha: (K,) per-column alpha scales
        device: Device to create output tensor on
        dtype: Output dtype
        lookup_table: Pre-allocated lookup table (for CUDA graph compatibility).
                      If None, will create one (not CUDA graph compatible).
    
    Returns:
        weight_unpacked: (N, K) tensor with unpacked and scaled weights
    """
    N, num_packed_cols = weight_packed.shape
    K_padded = num_packed_cols * 4
    
    # Use Triton kernel if available for better performance
    if TRITON_AVAILABLE and weight_packed.is_cuda:
        if lookup_table is None:
            lookup_table = _I2S_LOOKUP_TABLE.to(device=device, dtype=dtype)
        
        # Create output tensor
        weight_unpacked = torch.empty(N, K, device=device, dtype=dtype)
        
        # Launch Triton kernel with optimized block sizes
        # Larger blocks = better GPU occupancy
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 128  # Multiple of 4 for optimal unpacking
        
        grid = lambda META: (
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
            triton.cdiv(K, META["BLOCK_SIZE_K"]),
        )
        
        _i2s_unpack_kernel[grid](
            weight_packed,
            alpha,
            weight_unpacked,
            N,
            K,
            num_packed_cols,
            weight_packed.stride(0),
            weight_packed.stride(1),
            alpha.stride(0),
            weight_unpacked.stride(0),
            weight_unpacked.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        )
        
        return weight_unpacked
    
    # Fallback: Vectorized PyTorch implementation
    # Extract 4 values from each packed byte using vectorized bit operations
    # For each byte, extract: (packed >> (2*i)) & 0b11 for i in [0,1,2,3]
    weight_values = torch.zeros(N, K_padded, dtype=torch.uint8, device=device)
    
    # Vectorized extraction: extract all 4 values at once (optimized)
    # Expand packed to (N, num_packed_cols, 1) then extract bits
    weight_packed_expanded = weight_packed.unsqueeze(-1)  # (N, num_packed_cols, 1)
    
    # Create shift amounts for each of the 4 positions [0, 2, 4, 6]
    shifts = torch.tensor([0, 2, 4, 6], device=device, dtype=torch.uint8).view(1, 1, 4)
    
    # Extract all 4 values: (packed >> shift) & 0b11
    weight_values_4d = (weight_packed_expanded >> shifts) & 0b11  # (N, num_packed_cols, 4)
    weight_values = weight_values_4d.view(N, K_padded)  # (N, K_padded)
    
    # Use lookup table to convert {0,1,2,3} → {-1,0,1,0}
    # Lookup table: [-1, 0, 1, 0]
    if lookup_table is None:
        # Fallback: create lookup table (not CUDA graph compatible)
        lookup_table = _I2S_LOOKUP_TABLE.to(device=device, dtype=dtype)
    weight_ternary = lookup_table[weight_values.to(torch.long)]  # (N, K_padded)
    
    # Remove padding if needed
    if K_padded > K:
        weight_ternary = weight_ternary[:, :K]
    
    # Apply per-column alpha scaling: weight_ternary * alpha
    weight_scaled = weight_ternary * alpha.unsqueeze(0).to(dtype=dtype)  # (N, K)
    
    return weight_scaled


@dataclass
class TernaryConfig(QuantizationConfig):
    """Config class for ternary quantization with FP8 blockwise storage.
    
    Args:
        threshold_scale: Scale factor for ternary quantization threshold (0.0-1.0)
            Lower values = more aggressive quantization = more sparsity
            Default 0.7 matches qwen2_correct.py
        max_output_features: Skip quantization for layers larger than this
            (e.g., lm_head with 100K+ vocab size)
        fp8_block_size: Block size for FP8 quantization (default: 128x128)
            Larger blocks = fewer scales but potentially less accuracy
        use_i2s: Enable I2_S (Int2 Super-packed) mode for 4x memory reduction
            Stores ternary weights as 2-bit packed values: {-1→00, 0→01, 1→10}
            Uses fast lookup table for unpacking during forward pass
    """

    threshold_scale: float = 0.7
    max_output_features: int = 100_000
    fp8_block_size: tuple = (128, 128)  # (N_block, K_block) for blockwise FP8 quantization
    use_i2s: bool = False  # Enable I2_S super-packed mode

    def __post_init__(self):
        if not (0.0 < self.threshold_scale < 1.0):
            raise ValueError("threshold_scale must be between 0 and 1.")
        if self.max_output_features <= 0:
            raise ValueError("max_output_features must be positive.")

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
        fp8_block_size = config.get("fp8_block_size", (128, 128))
        if isinstance(fp8_block_size, (list, tuple)) and len(fp8_block_size) == 2:
            fp8_block_size = tuple(fp8_block_size)
        else:
            fp8_block_size = (128, 128)
        use_i2s = config.get("use_i2s", False) or os.environ.get("TERNARY_USE_I2S", "0") == "1"
        return cls(threshold_scale, max_output_features, fp8_block_size, use_i2s)

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
        
        # Pre-allocate I2_S lookup table for CUDA graph compatibility
        # Will be initialized lazily on first use with proper device/dtype
        self._i2s_lookup_table_cache = {}
        
        logger.info("=" * 80)
        if quant_config.use_i2s:
            logger.info("[TERNARY] Quantization initialized - I2_S (Int2 Super-packed) mode")
            logger.info("[TERNARY] Packing: {-1→00, 0→01, 1→10} → 4x memory reduction")
        else:
            logger.info("[TERNARY] Quantization initialized - Ternary + FP8 blockwise storage")
            logger.info(f"[TERNARY] FP8 block size: {quant_config.fp8_block_size}")
        
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
        """Load weights from checkpoint."""
        # Simple copy - weights will be quantized later in process_weights_after_loading
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
        """
        Create weight parameters.
        
        Initially creates FP16/BF16 weights. These will be quantized to ternary
        in process_weights_after_loading().
        """
        output_size_per_partition = sum(output_partition_sizes)
        
        # Create FP16/BF16 weight (will be quantized after loading)
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
        
        # Store metadata for quantization
        layer._ternary_output_size = output_size
        layer._ternary_is_too_large = output_size > self.quant_config.max_output_features
        
        logger.debug(
            f"Created ternary weight: {output_size_per_partition}x{input_size_per_partition}, "
            f"dtype={params_dtype}, will_quantize={not layer._ternary_is_too_large}"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Apply ternary quantization + storage after weights are loaded.
        
        Step 1 (matches qwen2_correct.py quantize()):
        1. Compute per-column threshold (0.7 * mean(abs(weights)))
        2. Create binary mask (abs(weights) > threshold)
        3. Compute per-column alpha scale
        4. Quantize: sign(weight) * alpha * mask
        
        Step 2 (Storage format - depends on config):
        - I2_S mode: Pack ternary weights as 2-bit values (4x memory reduction)
        - FP8 mode: Divide weights into blocks, quantize to FP8 (2x memory reduction)
        - BF16 fallback: Store quantized weights in BF16
        """
        # Skip if layer is too large
        if getattr(layer, '_ternary_is_too_large', False):
            logger.info(f"Skipping quantization for large layer: {layer.weight.shape}")
            return
        
        try:
            weight = layer.weight.data  # (out_features, in_features)
            original_dtype = weight.dtype
            N, K = weight.shape

            # Track memory BEFORE quantization
            weight_memory_before = get_tensor_memory_bytes(weight)
            layer_memory_before = get_layer_memory_bytes(layer)
            
            # Step 1: EXACTLY matching qwen2_correct.py quantize() function
            mode_str = "I2_S" if self.quant_config.use_i2s else "FP8 blockwise"
            logger.info(f"Quantizing layer to ternary + {mode_str}: {weight.shape}")
            logger.info(f"[TERNARY] Memory BEFORE quantization: weight={format_bytes(weight_memory_before)}, "
                       f"layer_total={format_bytes(layer_memory_before)}")
            
            weight_fp32 = weight.float()  # (N, K) where N=out_features, K=in_features
            absW = weight_fp32.abs()
            dim = 0  # per-column threshold (per input feature, matching qwen2_correct.py)
            th = self.quant_config.threshold_scale * absW.mean(dim, keepdim=True)  # (1, K)
            mask = absW > th  # binary mask
            mask_f = mask.to(weight_fp32.dtype)
            # per-column scale α - EXACTLY matching qwen2_correct.py line 66-67
            alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)  # (1, K)
            # ternary quantization - EXACTLY matching qwen2_correct.py line 69
            weight_ternary = weight_fp32.sign() * alpha * mask_f  # (N, K)
            
            # I2_S path: Pack ternary weights as 2-bit values (4x memory reduction)
            if self.quant_config.use_i2s:
                # Convert to integer ternary values {-1, 0, 1}
                # weight_ternary already has values in {-alpha, 0, +alpha}, we need just {-1, 0, 1}
                # Use sign() but preserve zeros: sign(0) = 0, sign(positive) = 1, sign(negative) = -1
                weight_ternary_int = weight_ternary.sign().to(torch.int8)  # (N, K) with values in {-1, 0, 1}
                
                # Pack into I2_S format: 4 weights per byte
                weight_packed = pack_i2s_weights(weight_ternary_int.float())  # (N, ceil(K/4)) uint8
                
                # Pre-allocate lookup tables for CUDA graph compatibility (before graph capture)
                # Create lookup tables for both common dtypes (float16 and bfloat16)
                device = weight_packed.device
                for dtype in [torch.float16, torch.bfloat16]:
                    # Use consistent cache key format matching apply() method
                    cache_key = f"{device}:{dtype}"
                    if cache_key not in self._i2s_lookup_table_cache:
                        # Pre-allocate lookup table (CUDA graph compatible)
                        # This happens BEFORE CUDA graph capture, so .to() is safe
                        self._i2s_lookup_table_cache[cache_key] = _I2S_LOOKUP_TABLE.to(
                            device=device, dtype=dtype
                        )
                
                # Store packed weights and alpha scales
                # Use replace_parameter to ensure old weight is properly freed
                replace_parameter(layer, 'weight', weight_packed)
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)  # (K,) per-column scales
                layer._ternary_original_dtype = original_dtype
                layer._ternary_fp8_enabled = False
                layer._ternary_i2s_enabled = True
                layer._ternary_weight_shape = (N, K)
                
                # Validate correctness if enabled
                if os.environ.get("TERNARY_VALIDATE_CORRECTNESS", "0") == "1":
                    # Reconstruct FP16 weight for comparison (weight_ternary with alpha applied)
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
                
                # Explicitly delete intermediate tensors to free memory
                del weight_fp32, absW, mask, mask_f, weight_ternary, weight_ternary_int
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Free cached memory
                
                # Track memory AFTER quantization (I2_S path)
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
                
                # Log cumulative stats every 10 layers
                if self._memory_stats['layers_quantized'] % 10 == 0:
                    cumulative_reduction = (1 - self._memory_stats['total_after_mb'] / self._memory_stats['total_before_mb']) * 100
                    logger.info(f"[TERNARY] Cumulative memory stats: {self._memory_stats['layers_quantized']} layers, "
                              f"before={self._memory_stats['total_before_mb']:.2f} MB, "
                              f"after={self._memory_stats['total_after_mb']:.2f} MB, "
                              f"reduction={cumulative_reduction:.1f}%")
                
                return
            
            # Step 2: FP8 blockwise quantization for memory reduction
            fp8_available = False
            if hasattr(torch, 'float8_e4m3fn'):
                try:
                    # Test FP8 conversion
                    test_tensor = torch.tensor([1.0], dtype=torch.float32, device=weight_ternary.device)
                    test_fp8 = test_tensor.to(torch.float8_e4m3fn)
                    fp8_available = (test_fp8.dtype == torch.float8_e4m3fn)
                except (RuntimeError, TypeError, AttributeError):
                    fp8_available = False
            
            if not fp8_available:
                # Fallback: store in BF16
                logger.warning(f"[TERNARY] FP8 not available for {type(layer).__name__}. Using BF16 storage.")
                weight_quantized = weight_ternary.to(torch.bfloat16)
                replace_parameter(layer, 'weight', weight_quantized)
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                layer._ternary_original_dtype = original_dtype
                layer._ternary_fp8_enabled = False
                self._ternary_quantization_stats['bf16_fallback'] += 1
                
                # Explicitly delete intermediate tensors
                del weight_fp32, absW, mask, mask_f, weight_ternary
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Track memory AFTER quantization (BF16 path)
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
            
            # FP8 blockwise quantization
            block_size_N, block_size_K = self.quant_config.fp8_block_size
            FP8_MAX = 448.0
            
            # Pad if necessary to make dimensions divisible by block size
            pad_N = (block_size_N - (N % block_size_N)) % block_size_N
            pad_K = (block_size_K - (K % block_size_K)) % block_size_K
            
            if pad_N > 0 or pad_K > 0:
                weight_padded = torch.nn.functional.pad(weight_ternary, (0, pad_K, 0, pad_N))
            else:
                weight_padded = weight_ternary
            
            N_padded, K_padded = weight_padded.shape
            
            # Reshape into blocks: (num_blocks_N, block_size_N, num_blocks_K, block_size_K)
            num_blocks_N = N_padded // block_size_N
            num_blocks_K = K_padded // block_size_K
            
            weight_blocks = weight_padded.view(
                num_blocks_N, block_size_N,
                num_blocks_K, block_size_K
            )  # (num_blocks_N, block_size_N, num_blocks_K, block_size_K)
            
            # Permute to group blocks together: (num_blocks_N, num_blocks_K, block_size_N, block_size_K)
            weight_blocks = weight_blocks.permute(0, 2, 1, 3).contiguous()
            
            # Compute per-block max absolute value → scale
            abs_max = weight_blocks.abs().amax(dim=(-2, -1), keepdim=True)  # (num_blocks_N, num_blocks_K, 1, 1)
            scales = abs_max / FP8_MAX
            scales = torch.where(scales == 0, torch.ones_like(scales), scales)  # avoid division by zero
            
            # Quantize each block: weight_block / scale_block → FP8
            weight_scaled = weight_blocks / scales  # (num_blocks_N, num_blocks_K, block_size_N, block_size_K)
            weight_scaled = weight_scaled.clamp(-FP8_MAX, FP8_MAX)
            weight_fp8_blocks = weight_scaled.to(torch.float8_e4m3fn)
            
            # Reshape back to (N_padded, K_padded)
            weight_fp8_blocks = weight_fp8_blocks.permute(0, 2, 1, 3).contiguous()  # (num_blocks_N, block_size_N, num_blocks_K, block_size_K)
            weight_fp8 = weight_fp8_blocks.view(N_padded, K_padded)
            
            # Remove padding if needed
            if pad_N > 0 or pad_K > 0:
                weight_fp8 = weight_fp8[:N, :K].contiguous()
            
            # Reshape scales to 2D: (num_blocks_N, num_blocks_K)
            scales_2d = scales.view(num_blocks_N, num_blocks_K)
            
            # Store FP8 weights and blockwise scales
            # Use replace_parameter to ensure old weight is properly freed
            replace_parameter(layer, 'weight', weight_fp8)
            layer.register_buffer('fp8_block_scales', scales_2d.contiguous(), persistent=False)  # (num_blocks_N, num_blocks_K)
            layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)  # (K,) per-column scales
            layer._ternary_original_dtype = original_dtype
            layer._ternary_fp8_enabled = True
            layer._ternary_block_size = (block_size_N, block_size_K)
            layer._ternary_weight_shape = (N, K)
            
            # Validate correctness if enabled
            if os.environ.get("TERNARY_VALIDATE_CORRECTNESS", "0") == "1":
                # Reconstruct FP16 weight for comparison (weight_ternary with alpha applied)
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
            
            # Explicitly delete intermediate tensors to free memory
            del weight_fp32, absW, mask, mask_f, weight_ternary, weight_padded, weight_blocks, abs_max, scales, weight_scaled, weight_fp8_blocks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Track memory AFTER quantization (FP8 path)
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
            
            # Log cumulative stats every 10 layers
            if self._memory_stats['layers_quantized'] % 10 == 0:
                cumulative_reduction = (1 - self._memory_stats['total_after_mb'] / self._memory_stats['total_before_mb']) * 100
                logger.info(f"[TERNARY] Cumulative memory stats: {self._memory_stats['layers_quantized']} layers, "
                          f"before={self._memory_stats['total_before_mb']:.2f} MB, "
                          f"after={self._memory_stats['total_after_mb']:.2f} MB, "
                          f"reduction={cumulative_reduction:.1f}%")
            
            # Log detailed summary every 50 layers (likely near end of model)
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
        """
        Apply the quantized weights to input.
        
        Forward pass strategies:
        - I2_S mode: x (FP16) * unpack(W_packed) * alpha → equivalent to x (FP16) * W (FP16)
        - FP8 mode: x (FP16) * W (FP8) * scale → equivalent to x (FP16) * W (FP16)
        - BF16 fallback: use weights directly
        
        Strategy:
        - If I2_S weights: unpack using lookup table, apply alpha scaling, then FP16 matmul
        - If FP8 weights: dequantize using blockwise scales, then standard FP16 matmul
        - If BF16 weights: use directly (fallback path)
        """
        weight = layer.weight  # FP8, BF16, or uint8 (I2_S)
        fp8_enabled = getattr(layer, '_ternary_fp8_enabled', False)
        i2s_enabled = getattr(layer, '_ternary_i2s_enabled', False)
        
        # Log path selection on first call per layer type
        if not hasattr(layer, '_ternary_path_logged'):
            layer._ternary_path_logged = True
            logger.info(
                f"[TERNARY] Forward pass for {type(layer).__name__}: "
                f"i2s_enabled={i2s_enabled}, fp8_enabled={fp8_enabled}, "
                f"weight.dtype={weight.dtype}, weight.shape={weight.shape}"
            )
        
        # Prepare input
        x_compute = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
        b_compute = None if bias is None else (bias if bias.dtype in (torch.float16, torch.bfloat16) else bias.to(x_compute.dtype))
        
        if i2s_enabled and weight.dtype == torch.uint8:
            # I2_S path: Unpack using lookup table, apply alpha scaling, then matmul
            N, K = getattr(layer, '_ternary_weight_shape', weight.shape)
            alpha = layer.ternary_alpha  # (K,) per-column scales
            
            # Get pre-allocated lookup table (created during weight processing)
            # This avoids tensor creation during CUDA graph capture
            device_key = str(x_compute.device)
            dtype_key = str(x_compute.dtype)
            cache_key = f"{device_key}:{dtype_key}"
            
            if cache_key not in self._i2s_lookup_table_cache:
                # This should not happen if weights were processed correctly
                # But provide fallback for safety (though not CUDA graph compatible)
                logger.warning(f"[TERNARY] Lookup table not pre-allocated for {cache_key}. "
                             f"Creating during forward (not CUDA graph compatible).")
                self._i2s_lookup_table_cache[cache_key] = _I2S_LOOKUP_TABLE.to(
                    device=x_compute.device, dtype=x_compute.dtype
                )
            lookup_table = self._i2s_lookup_table_cache[cache_key]
            
            # Unpack I2_S weights and apply alpha scaling
            weight_unpacked = unpack_i2s_weights(
                weight_packed=weight,
                K=K,
                alpha=alpha,
                device=x_compute.device,
                dtype=x_compute.dtype,
                lookup_table=lookup_table
            )  # (N, K)
            
            # Standard FP16 matmul with unpacked weights
            out = torch.nn.functional.linear(x_compute, weight_unpacked, b_compute)
        elif fp8_enabled and weight.dtype == torch.float8_e4m3fn:
            # FP8 path: Optimized blockwise dequantization + alpha scaling
            # Note: We use blockwise dequantization because we have blockwise scales + per-column alpha
            # This is more efficient than trying to convert to per-channel scales for torch._scaled_mm
            block_size_N, block_size_K = getattr(layer, '_ternary_block_size', (128, 128))
            N, K = getattr(layer, '_ternary_weight_shape', weight.shape)
            fp8_block_scales = layer.fp8_block_scales  # (num_blocks_N, num_blocks_K)
            alpha = layer.ternary_alpha  # (K,) per-column scales
            
            num_blocks_N = fp8_block_scales.shape[0]
            num_blocks_K = fp8_block_scales.shape[1]
            
            # Pad if necessary
            pad_N = (block_size_N - (N % block_size_N)) % block_size_N
            pad_K = (block_size_K - (K % block_size_K)) % block_size_K
            
            if pad_N > 0 or pad_K > 0:
                weight_padded = torch.nn.functional.pad(weight, (0, pad_K, 0, pad_N))
            else:
                weight_padded = weight
            
            N_padded, K_padded = weight_padded.shape
            
            # Reshape into blocks and dequantize efficiently
            weight_blocks = weight_padded.view(
                num_blocks_N, block_size_N,
                num_blocks_K, block_size_K
            ).permute(0, 2, 1, 3).contiguous()  # (num_blocks_N, num_blocks_K, block_size_N, block_size_K)
            
            # Dequantize: convert FP8 to BF16 and apply block scales
            weight_fp16_blocks = weight_blocks.to(torch.bfloat16)
            scales_expanded = fp8_block_scales.view(num_blocks_N, num_blocks_K, 1, 1).to(torch.bfloat16)
            weight_fp16_blocks = weight_fp16_blocks * scales_expanded
            
            # Reshape back
            weight_fp16_blocks = weight_fp16_blocks.permute(0, 2, 1, 3).contiguous()
            weight_fp16 = weight_fp16_blocks.view(N_padded, K_padded)
            
            # Remove padding
            if pad_N > 0 or pad_K > 0:
                weight_fp16 = weight_fp16[:N, :K].contiguous()
            
            # Apply per-column alpha scaling
            weight_fp16 = weight_fp16 * alpha.unsqueeze(0).to(x_compute.dtype)
            
            # Standard FP16 matmul
            out = torch.nn.functional.linear(x_compute, weight_fp16, b_compute)
        else:
            # BF16 fallback path: use weights directly
            out = torch.nn.functional.linear(x_compute, weight, b_compute)
        
        return out.to(x.dtype)
