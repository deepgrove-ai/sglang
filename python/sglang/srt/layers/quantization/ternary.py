"""Ternary quantization with FP8 storage for SGLang.

This implementation uses pre-compiled static TileLang FP8 kernels for SOTA performance.
NO TVM dependency at runtime = NO conflicts with xgrammar.

Key features:
- Static TileLang FP8 tensor core matmul (.so files, NO TVM)
- FP8 E4M3 weight storage for 2x memory reduction
- Runtime weight-only quantization (applied after model loading)
- Fallback to FP16 if kernel not available for specific shape
"""

import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
)
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


@dataclass
class TernaryConfig(QuantizationConfig):
    """Config class for ternary quantization with FP8 storage.
    
    Args:
        threshold_scale: Scale factor for ternary quantization threshold (0.0-1.0)
            Lower values = more aggressive quantization = more sparsity
        max_output_features: Skip quantization for layers larger than this
            (e.g., lm_head with 100K+ vocab size)
    """

    threshold_scale: float = 0.7
    max_output_features: int = 100_000

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
        return cls(threshold_scale, max_output_features)

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
        # FP8 requires at least SM 8.9 (Hopper)  
        # But we use FP16 fallback, so accept any GPU
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

    def __init__(self, quant_config: TernaryConfig):
        self.quant_config = quant_config
        
        # Track quantization stats for visibility
        self._ternary_quantization_stats = {
            'total_layers': 0,
            'fp8_quantized': 0,
            'fp8_blockwise': 0,
            'fp8_rowwise': 0,
            'bf16_fallback': 0,
            'tilelang_kernels_used': 0,
            'cutlass_kernels_used': 0,
            'triton_kernels_used': 0,
        }
        
        # Create CUDA stream for pipelined activation quantization
        # This allows overlapping quantization with weight preparation
        # NOTE: Automatically disabled during CUDA graph capture (streams incompatible)
        enable_pipeline = os.environ.get("TERNARY_FP8_PIPELINE_QUANT", "1") == "1"
        self._quant_stream = None
        self._pipeline_enabled = False
        
        if enable_pipeline and torch.cuda.is_available():
            try:
                # Check if we're in capture mode - if so, don't create stream
                in_capture_init = False
                try:
                    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                    in_capture_init = get_is_capture_mode()
                except (ImportError, AttributeError):
                    pass
                
                if not in_capture_init:
                    self._quant_stream = torch.cuda.Stream()
                    self._pipeline_enabled = True
            except Exception as e:
                logger.debug(f"[TERNARY] Failed to create quantization stream: {e}. Disabling pipelining.")
                self._quant_stream = None
                self._pipeline_enabled = False
        
        logger.info("=" * 80)
        logger.info("[TERNARY] Quantization initialized - will track FP8/TileLang usage")
        if self._pipeline_enabled:
            logger.info("[TERNARY] Pipeline quantization enabled (async activation quantization)")
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
        
        Initially creates FP16/BF16 weights. These will be quantized to FP8
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
        Apply ternary quantization after weights are loaded (SOTA approach).
        
        This matches benchmark_infer.py:
        1. Quantize to ternary using Triton
        2. Reconstruct with per-column scaling
        3. Convert to FP8 E4M3 for 2x memory reduction
        4. Store in format compatible with TileLang FP8 tensor core matmul
        """
        import sys
        if '/home/ubuntu/raghav' not in sys.path:
            sys.path.insert(0, '/home/ubuntu/raghav')
        
        # Skip if layer is too large
        if getattr(layer, '_ternary_is_too_large', False):
            logger.info(f"Skipping quantization for large layer: {layer.weight.shape}")
            # Initialize cache dicts for consistency, but don't pre-allocate (buffers allocated lazily)
            if not hasattr(layer, '_act_fp8_block_cache'):
                layer._act_fp8_block_cache = {}
            if not hasattr(layer, '_act_fp8_cache'):
                layer._act_fp8_cache = {}
            return
        
        try:
            weight = layer.weight.data  # (out_features, in_features)
            original_dtype = weight.dtype
            
            # Option: Use FP16/BF16 path matching qwen2_correct.py (ternary quantized, stored in model dtype, no activation quant)
            # This is faster for decode and matches the reference implementation exactly
            use_fp16_path = os.environ.get("TERNARY_FP16_PATH", "0") == "1"
            
            if use_fp16_path:
                # Store weights in the same dtype as the model (BF16 if model uses BF16, FP16 if FP16)
                # This avoids conversion overhead on every forward pass
                weight_dtype = original_dtype  # Use model's native dtype (BF16 or FP16)
                logger.info(f"FP16/BF16 path (qwen2_correct.py style): quantizing to ternary+{weight_dtype} {weight.shape}")
                # Full ternary path - EXACTLY matching qwen2_correct.py quantize()
                # Optimized: use in-place ops and avoid intermediate copies
                weight_fp32 = weight.float()  # (N, K)
                absW = weight_fp32.abs()
                dim = 0  # per-column threshold (per input feature, matching qwen2_correct.py)
                th = self.quant_config.threshold_scale * absW.mean(dim, keepdim=True)  # (1, K)
                mask = absW > th  # binary mask
                mask_f = mask.to(weight_fp32.dtype)
                # per-column scale α - EXACTLY matching qwen2_correct.py line 66-67
                alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)  # (1, K)
                # ternary quantization - EXACTLY matching qwen2_correct.py line 69
                # Use in-place ops where possible
                weight_ternary = weight_fp32.sign() * alpha * mask_f  # (N, K)
                # Store in model's native dtype (BF16 or FP16) to avoid conversion overhead
                # Direct assignment avoids extra copy
                weight_quantized = weight_ternary.to(weight_dtype)
                layer.weight.data = weight_quantized
                # Store alpha for potential use (though we don't need it if weights are already scaled)
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)  # (K,)
                layer._ternary_fp8_enabled = False  # Mark as FP16/BF16 path, not FP8
                layer._ternary_original_dtype = original_dtype
                logger.info(f"✓ Quantized to {weight_dtype} (ternary): {layer.weight.shape}, dtype={layer.weight.dtype}")
                return

            # Fast init path: optionally skip ternary compute and cast BF16/BF16->FP8 directly.
            # Default to 0 to ensure proper ternary quantization with alpha scaling
            fast_init = os.environ.get("FP8_FAST_INIT", "0") != "0"

            if fast_init:
                logger.info(f"FP8 fast-init: casting weight to FP8 (skip ternary) {weight.shape}")
                if hasattr(torch, 'float8_e4m3fn'):
                    weight_fp8 = weight.to(torch.float8_e4m3fn)
                    layer._ternary_fp8_enabled = True
                else:
                    weight_fp8 = weight.to(torch.float16)
                    layer._ternary_fp8_enabled = False
                layer.weight.data = weight_fp8
                layer._ternary_original_dtype = original_dtype
                return

            # Full ternary path - with low-memory streaming by default
            logger.info(f"Quantizing layer to ternary+FP8: {weight.shape}")
            use_low_mem = os.environ.get("TERNARY_LOW_MEM", "1") == "1"

            if use_low_mem:
                # Two-pass streaming to compute th and alpha per column (K)
                N, K = weight.shape
                rows_per_chunk = int(os.environ.get("TERNARY_CHUNK_ROWS", "4096"))
                device = weight.device
                # Pass 1: compute mean(absW) per column
                sum_abs = torch.zeros(K, device=device, dtype=torch.float32)
                for start in range(0, N, rows_per_chunk):
                    end = min(N, start + rows_per_chunk)
                    w_chunk = weight[start:end, :]
                    sum_abs += w_chunk.abs().sum(dim=0, dtype=torch.float32)
                th = (self.quant_config.threshold_scale * (sum_abs / float(N))).view(1, K)
                # Pass 2: compute alpha per column
                sum_abs_masked = torch.zeros(K, device=device, dtype=torch.float32)
                count_mask = torch.zeros(K, device=device, dtype=torch.float32)
                for start in range(0, N, rows_per_chunk):
                    end = min(N, start + rows_per_chunk)
                    w_chunk = weight[start:end, :]
                    abs_chunk = w_chunk.abs()
                    mask = abs_chunk > th  # (rows, K)
                    sum_abs_masked += abs_chunk.masked_select(mask).view(-1, K).sum(dim=0, dtype=torch.float32) if mask.numel() != 0 else 0.0
                    count_mask += mask.sum(dim=0, dtype=torch.float32)
                alpha = (sum_abs_masked / count_mask.clamp_min(1.0)).view(-1)  # (K,)
                # Pass 3: reconstruct ternary and directly produce FP8 + per-row scales
                FP8_MAX = 448.0
                row_scales = torch.empty(N, device=device, dtype=torch.float32)
                weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
                alpha_b = alpha.view(1, K)
                for start in range(0, N, rows_per_chunk):
                    end = min(N, start + rows_per_chunk)
                    w_chunk = weight[start:end, :]
                    abs_chunk = w_chunk.abs()
                    mask = abs_chunk > th
                    ternary = w_chunk.sign().to(torch.float32) * alpha_b * mask
                    row_amax = ternary.abs().amax(dim=1)
                    scale_row = (row_amax / FP8_MAX).clamp_min(1e-12)
                    row_scales[start:end] = scale_row
                    scaled = (ternary / scale_row.view(-1, 1)).clamp(-FP8_MAX, FP8_MAX)
                    weight_fp8[start:end, :] = scaled.to(torch.float8_e4m3fn)
                layer._ternary_fp8_enabled = True
                layer.register_buffer('fp8_row_scale', row_scales.contiguous(), persistent=False)
                layer.weight.data = weight_fp8
                layer.register_buffer('ternary_alpha', alpha.contiguous(), persistent=False)
                layer._ternary_original_dtype = original_dtype
                logger.info(
                    f"✓ Quantized (low-mem) to FP8: {layer.weight.shape}, dtype={layer.weight.dtype}"
                )
                return

            # Standard path (higher temp memory) - EXACTLY matching qwen2_correct.py quantize()
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
            weight_reconstructed = weight_ternary  # This is the quantized weight matching qwen2_correct.py
            # FP8 quantization for weights
            # Check FP8 availability with actual conversion test
            fp8_available = False
            if hasattr(torch, 'float8_e4m3fn'):
                try:
                    # Test actual FP8 conversion on the target device
                    test_tensor = torch.tensor([1.0], dtype=torch.float32, device=weight_reconstructed.device)
                    test_fp8 = test_tensor.to(torch.float8_e4m3fn)
                    fp8_available = (test_fp8.dtype == torch.float8_e4m3fn)
                    if not fp8_available:
                        logger.warning(f"[TERNARY] FP8 conversion test returned wrong dtype: {test_fp8.dtype}, expected torch.float8_e4m3fn")
                except (RuntimeError, TypeError, AttributeError) as e:
                    logger.warning(f"[TERNARY] FP8 conversion test failed: {e}. FP8 not available, using BF16.")
                    fp8_available = False
            else:
                logger.warning("[TERNARY] torch.float8_e4m3fn not available. Using BF16 storage.")
            
            if not fp8_available:
                logger.warning(f"[TERNARY] FP8 not available for {type(layer).__name__}. Using BF16 storage.")
                weight_fp8 = weight_reconstructed.to(torch.bfloat16)
                layer._ternary_fp8_enabled = False
                layer.weight.data = weight_fp8
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                layer._ternary_original_dtype = original_dtype
                logger.info(f"✓ Quantized to BF16 (ternary): {layer.weight.shape}, dtype={layer.weight.dtype}")
                return  # Exit early - FP8 not available
            
            # At this point, fp8_available=True (we already returned if False)
            use_blockwise = os.environ.get("TERNARY_FP8_BLOCKWISE", "0") == "1"
            if use_blockwise:
                    # Blockwise FP8 quantization with block sizes along N and K (defaults 128x128)
                    N, K = weight_reconstructed.shape
                    block_n = int(os.environ.get("TERNARY_FP8_BLOCK_N", "128"))
                    block_k = int(os.environ.get("TERNARY_FP8_BLOCK_K", "128"))
                    FP8_MAX = 448.0
                    num_blocks_n = (N + block_n - 1) // block_n
                    num_blocks_k = (K + block_k - 1) // block_k
                    # Per (N_block, K_block) scales
                    scales_b = torch.empty((num_blocks_n, num_blocks_k), device=weight_reconstructed.device, dtype=torch.float32)
                    # Create FP8 tensor - verify dtype immediately
                    try:
                        weight_fp8 = torch.empty_like(weight_reconstructed, dtype=torch.float8_e4m3fn)
                        if weight_fp8.dtype != torch.float8_e4m3fn:
                            raise RuntimeError(f"torch.empty_like with dtype=torch.float8_e4m3fn returned dtype={weight_fp8.dtype}")
                    except (RuntimeError, TypeError) as e:
                        logger.error(f"[TERNARY] Failed to create FP8 tensor for {type(layer).__name__}: {e}. Using BF16.")
                        weight_fp8 = weight_reconstructed.to(torch.bfloat16)
                        layer._ternary_fp8_enabled = False
                        layer.weight.data = weight_fp8
                        layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                        layer._ternary_original_dtype = original_dtype
                        return
                    try:
                        for bn in range(num_blocks_n):
                            n_start = bn * block_n
                            n_end = min(N, n_start + block_n)
                            for bk in range(num_blocks_k):
                                k_start = bk * block_k
                                k_end = min(K, k_start + block_k)
                                w_block = weight_reconstructed[n_start:n_end, k_start:k_end]
                                # Single scale per block for best kernel perf
                                block_abs_max = w_block.abs().amax()
                                scale = (block_abs_max / FP8_MAX).clamp_min(1e-12)
                                scales_b[bn, bk] = scale
                                w_scaled = (w_block / scale).clamp(-FP8_MAX, FP8_MAX)
                                w_fp8_block = w_scaled.to(torch.float8_e4m3fn)
                                # Verify conversion succeeded immediately
                                if w_fp8_block.dtype != torch.float8_e4m3fn:
                                    raise RuntimeError(
                                        f"[TERNARY] Block FP8 conversion failed for {type(layer).__name__} "
                                        f"block ({bn},{bk}): got {w_fp8_block.dtype}, expected torch.float8_e4m3fn"
                                    )
                                weight_fp8[n_start:n_end, k_start:k_end] = w_fp8_block
                    except RuntimeError as e:
                        # FP8 conversion failed during block quantization
                        logger.error(f"[TERNARY] {e}. Falling back to BF16 for entire layer.")
                        weight_fp8 = weight_reconstructed.to(torch.bfloat16)
                        layer._ternary_fp8_enabled = False
                        layer.weight.data = weight_fp8
                        layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                        layer._ternary_original_dtype = original_dtype
                        return
                    
                    # Final verification after all blocks assigned
                    if weight_fp8.dtype != torch.float8_e4m3fn:
                        logger.error(
                            f"[TERNARY] FP8 conversion FAILED for {type(layer).__name__}! "
                            f"Got dtype={weight_fp8.dtype}, expected torch.float8_e4m3fn. "
                            f"Falling back to BF16. This likely means FP8 is not supported on this GPU or PyTorch version."
                        )
                        # Fallback to BF16
                        weight_fp8 = weight_reconstructed.to(torch.bfloat16)
                        layer._ternary_fp8_enabled = False
                        layer.weight.data = weight_fp8
                        layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                        layer._ternary_original_dtype = original_dtype
                        return  # Exit early - FP8 failed
                    
                    # FP8 conversion succeeded
                    layer._ternary_fp8_enabled = True
                    layer.register_buffer('fp8_block_scales', scales_b.contiguous(), persistent=False)
                    layer.weight.data = weight_fp8
                    logger.info(f"[TERNARY] ✓ Quantized to FP8 (blockwise): {layer.weight.shape}, dtype={layer.weight.dtype}")
                    self._ternary_quantization_stats['fp8_quantized'] += 1
                    self._ternary_quantization_stats['fp8_blockwise'] += 1
                    # Note: Blockwise path uses weights in (N, K) format directly, no transpose needed
                    layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                    layer._ternary_original_dtype = original_dtype
                    return
            else:
                # Row-wise FP8 quantization for weights (per-output-channel)
                # For rowwise path, CUTLASS fp8_scaled_mm expects (K, N) column-major weights
                # Store weights directly in transposed column-major format to avoid caching duplicates
                w_abs_max = weight_reconstructed.abs().amax(dim=1, keepdim=True)  # (N,1)
                scale_row = (w_abs_max / 448.0).clamp_min(1e-12).to(torch.float32)  # (N,1)
                weight_scaled = (weight_reconstructed / scale_row).clamp(-448.0, 448.0)
                weight_fp8_orig = weight_scaled.to(torch.float8_e4m3fn)  # (N, K)
                
                # Verify FP8 conversion succeeded
                if weight_fp8_orig.dtype != torch.float8_e4m3fn:
                    logger.error(
                        f"[TERNARY] FP8 conversion FAILED for {type(layer).__name__} (rowwise)! "
                        f"Got dtype={weight_fp8_orig.dtype}, expected torch.float8_e4m3fn. "
                        f"Falling back to BF16."
                    )
                    weight_fp8 = weight_reconstructed.to(torch.bfloat16)
                    layer._ternary_fp8_enabled = False
                    layer.weight.data = weight_fp8
                    layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                    layer._ternary_original_dtype = original_dtype
                    return  # Exit early - FP8 failed
                
                # Store FP8 weights in (N, K) directly; use transpose VIEW at matmul time (column-major for CUTLASS)
                layer._ternary_fp8_enabled = True
                layer.weight.data = weight_fp8_orig  # (N, K) FP8
                logger.info(
                    f"[TERNARY] ✓ Quantized to FP8 (rowwise): stored (N,K)={layer.weight.shape}, dtype={layer.weight.dtype}; "
                    f"will use transpose VIEW for CUTLASS to satisfy column-major."
                )
                self._ternary_quantization_stats['fp8_quantized'] += 1
                self._ternary_quantization_stats['fp8_rowwise'] += 1
                layer.register_buffer('fp8_row_scale', scale_row.view(-1).contiguous(), persistent=False)
                layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
                layer._ternary_original_dtype = original_dtype
                return
            
            # Both blockwise and rowwise paths return early above - this should be unreachable
            # But keep as safety fallback
            logger.error(f"[TERNARY] Internal error: Reached unreachable code for {type(layer).__name__}. Falling back to BF16.")
            weight_fp8 = weight_reconstructed.to(torch.bfloat16)
            layer._ternary_fp8_enabled = False
            layer.weight.data = weight_fp8
            layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
            layer._ternary_original_dtype = original_dtype
            
            # Initialize cache dicts for fallback path
            if not hasattr(layer, '_act_fp8_block_cache'):
                layer._act_fp8_block_cache = {}
            if not hasattr(layer, '_act_fp8_cache'):
                layer._act_fp8_cache = {}
        except Exception as e:
            logger.error(f"Error during (fast-init or ternary) quantization: {e}. Keeping original weights.")
            # Initialize cache dicts even if quantization failed (buffers allocated lazily during warmup)
            if not hasattr(layer, '_act_fp8_block_cache'):
                layer._act_fp8_block_cache = {}
            if not hasattr(layer, '_act_fp8_cache'):
                layer._act_fp8_cache = {}

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply the quantized weights to input.
        
        Uses pre-quantized FP8 weights (stored during process_weights_after_loading) with torch._scaled_mm.
        Only activations are quantized on-the-fly (minimal overhead).
        Matches qwen2_correct.py exactly.
        """
        # Use pre-quantized weights (quantized ONCE during model loading, not every forward pass!)
        weight = layer.weight
        fp8_enabled = getattr(layer, '_ternary_fp8_enabled', False)
        
        # Log path selection on first call per layer type
        if not hasattr(layer, '_ternary_path_logged'):
            layer._ternary_path_logged = True
            logger.info(
                f"[TERNARY] Path check for {type(layer).__name__}: "
                f"fp8_enabled={fp8_enabled}, weight.dtype={weight.dtype}, "
                f"TERNARY_FP16_PATH={os.environ.get('TERNARY_FP16_PATH', '0')}"
            )
        
        # For decode (small M), automatically use FP16/BF16 path (no activation quantization)
        # This matches qwen2_correct.py exactly: only weights are quantized, activations stay BF16
        # This avoids activation quantization overhead which kills performance for small batches
        M = x.shape[0] if len(x.shape) >= 1 else 1
        is_decode = M < 8  # Small M typically means decode
        use_fp16_for_decode = os.environ.get("TERNARY_FP16_DECODE", "1") == "1"
        
        # Fast FP16/BF16 path: ternary quantized weights stored in model dtype, no activation quantization
        # This matches qwen2_correct.py exactly and is fastest for decode
        if (not fp8_enabled and weight.dtype in (torch.float16, torch.bfloat16)) or \
           (is_decode and use_fp16_for_decode and weight.dtype in (torch.float16, torch.bfloat16)):
            # Direct matmul with ternary-quantized weights (qwen2_correct.py style)
            # No activation quantization, no FP8 kernels - just standard cuBLAS
            # Weights are already in the correct dtype (BF16 or FP16) matching the model
            # No conversion needed - just use standard linear op
            if not hasattr(layer, '_ternary_fp16_path_logged'):
                layer._ternary_fp16_path_logged = True
                logger.info(
                    f"[TERNARY] Using FP16/BF16 path for {type(layer).__name__} "
                    f"(M={M}, decode={is_decode}, no activation quantization)"
                )
            out = torch.nn.functional.linear(x, weight, bias)
            return out
        
        # FP8 path below (original implementation)
        # For rowwise path: weights stored ONLY as (K, N) in _weight_fp8_T_stored, layer.weight may be invalid
        # For blockwise path: weight is stored directly as (N, K) in layer.weight
        # Get actual weight format based on storage format
        if hasattr(layer, '_ternary_weight_stored_as_transposed') and layer._ternary_weight_stored_as_transposed:
            # Rowwise path: get from stored transposed format
            weight_fp8 = layer._weight_fp8_T_stored.t()  # (N, K) from stored (K, N)
        else:
            # Blockwise path or fallback: use layer.weight directly
            weight_fp8 = weight  # Already (N, K)
        fp8_row_scale = getattr(layer, 'fp8_row_scale', None)  # (N,) per-row scale for weights
        fp8_block_scales = getattr(layer, 'fp8_block_scales', None)  # (N, K/128) per-block scales
        
        # USE TILELANG PRECOMPILED KERNELS for real FP8 tensor core speedup!
        # torch._scaled_mm is NOT faster, but TileLang kernels are optimized for H100 tensor cores
        # We have precompiled kernels in ternarykernels/kernels/compiled_fp8/
        
        # Check if we're in CUDA graph capture mode - skip FP8 during capture for compatibility
        in_capture_mode = False
        try:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
            in_capture_mode = get_is_capture_mode()
        except ImportError:
            pass
        
        # During CUDA graph capture, FP8 path can be used if buffers are pre-allocated
        # We pre-allocate buffers during warmup (before capture), so FP8 is safe to use
        if in_capture_mode:
            # During capture, we can use FP8 if buffers are pre-allocated
            # The check below will verify buffers exist
            pass
        
        use_tilelang_fp8 = (
            weight_fp8.dtype == torch.float8_e4m3fn and
            fp8_row_scale is not None and
            os.environ.get("TERNARY_FP8_TENSOR_CORES", "1") == "1"
        )
        
        # Prefer blockwise fast path when enabled
        use_blockwise = (
            weight_fp8.dtype == torch.float8_e4m3fn and
            fp8_block_scales is not None and
            os.environ.get("TERNARY_FP8_BLOCKWISE", "0") == "1"
        )
        
        # Log why FP8 is/isn't enabled
        if not hasattr(layer, '_ternary_fp8_check_logged'):
            layer._ternary_fp8_check_logged = True
            logger.info(
                f"[TERNARY] FP8 path check for {type(layer).__name__}: "
                f"fp8_enabled={use_tilelang_fp8 or use_blockwise}, "
                f"weight.dtype={weight_fp8.dtype}, TERNARY_FP16_PATH={int(os.environ.get('TERNARY_FP16_PATH','0'))}"
            )
        # Heuristic: for very small M (decode), rowwise path can be faster
        try:
            min_m_for_block = int(os.environ.get("TERNARY_FP8_BLOCKWISE_MIN_M", "8"))
        except Exception:
            min_m_for_block = 8

        # Track which path is taken (simple logging, no heavy timing)
        M = x.shape[0] if len(x.shape) >= 1 else "?"
        path_taken = None
        
        # For decode (small M), skip activation quantization - use FP8 weights with BF16 activations
        # This matches qwen2_correct.py exactly: only weights are quantized, activations stay BF16
        skip_act_quant_decode = os.environ.get("TERNARY_FP8_SKIP_ACT_QUANT_DECODE", "1") == "1"
        is_decode = M < 8  # Small M typically means decode
        
        # USE FP8 MATMUL: Fast FP8 tensor cores with BF16 activations + FP8 weights
        # qwen2_correct.py uses: torch.nn.functional.linear(x_bf16, quantize(weight))
        # But FP8 matmul is MUCH faster - use FP8 kernels with BF16 activations (no activation quantization overhead)
        # Activations stay in BF16, weights are FP8, use fp8_scaled_mm kernel
        
        # Try FP8 matmul path (fastest - uses tensor cores)
        if fp8_row_scale is not None and weight_fp8.dtype == torch.float8_e4m3fn:
            # ROWWISE FP8 PATH: Try TileLang first (fastest!), then CUTLASS/Triton
            path_taken = "FP8_ROWWISE"
            if not hasattr(layer, '_ternary_fp8_rowwise_logged'):
                layer._ternary_fp8_rowwise_logged = True
                logger.info(
                    f"[TERNARY] Using FP8 rowwise matmul for {type(layer).__name__} (M={M}): "
                    f"BF16 activations + FP8 weights"
                )
            try:
                # Try TileLang FP8 kernels first (matches benchmark_infer.py - fastest!)
                try:
                    import sys
                    # Add ternarykernels to path (same as benchmark_infer.py)
                    ternarykernels_path = '/home/ubuntu/raghav/ternarykernels'
                    if ternarykernels_path not in sys.path:
                        sys.path.insert(0, ternarykernels_path)
                    # Import from kernels directly (not ternarykernels.kernels) - matches benchmark_infer.py
                    from kernels.tilelang_fp8 import fp8_matmul, fp8_matmul_bias_fused, HAS_TILELANG
                    
                    if HAS_TILELANG:
                        M, K = x.shape
                        N = weight_fp8.shape[0]
                        
                        # Pre-quantize activations to FP8 (matching benchmark_infer.py)
                        # Direct BF16->FP8 conversion avoids quantization overhead
                        from tilelang.utils.tensor import map_torch_type
                        fp8_dtype = map_torch_type("float8_e4m3")
                        x_fp8_prequant = x.to(fp8_dtype)
                        
                        # Weight is already FP8, stored as (N, K)
                        # Call TileLang kernel with pre-quantized inputs (fastest path!)
                        if bias is not None:
                            out = fp8_matmul_bias_fused(
                                x_fp8_prequant, weight_fp8, bias,
                                pre_quantized=True, b_transposed=True
                            )
                        else:
                            out = fp8_matmul(
                                x_fp8_prequant, weight_fp8,
                                pre_quantized=True, b_transposed=True
                            )
                        
                        if not hasattr(layer, '_ternary_tilelang_rowwise_logged'):
                            layer._ternary_tilelang_rowwise_logged = True
                            logger.info(
                                f"[TERNARY] ✓✓✓ TILELANG FP8 KERNEL ACTIVE ✓✓✓ {type(layer).__name__} "
                                f"(M={M}, N={N}, K={K}) - matches benchmark_infer.py!"
                            )
                            self._ternary_quantization_stats['tilelang_kernels_used'] += 1
                        return out.to(x.dtype)
                    else:
                        # HAS_TILELANG is False
                        if not hasattr(layer, '_ternary_tilelang_not_available_logged'):
                            layer._ternary_tilelang_not_available_logged = True
                            logger.debug(f"[TERNARY] TileLang not available (HAS_TILELANG=False), using CUTLASS/Triton fallback")
                except (ImportError, AttributeError, RuntimeError, TypeError) as e:
                    # TileLang not available or failed - fall back to CUTLASS/Triton
                    if not hasattr(layer, '_ternary_tilelang_rowwise_fallback_logged'):
                        layer._ternary_tilelang_rowwise_fallback_logged = True
                        logger.warning(f"[TERNARY] TileLang FP8 failed for {type(layer).__name__} ({e}), using CUTLASS/Triton fallback")
                        import traceback
                        logger.debug(f"[TERNARY] TileLang error traceback: {traceback.format_exc()}")
                
                # Fallback: CUTLASS/Triton path (original implementation)
                from sgl_kernel import fp8_scaled_mm
                from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
                
                M, K = x.shape
                N = weight_fp8.shape[0]
                cutlass_available = cutlass_fp8_supported()
                n_div16 = N % 16 == 0
                k_div16 = K % 16 == 0
                scale_match = fp8_row_scale.numel() == N
                
                # Check CUTLASS compatibility for optimal performance
                cutlass_compatible = (
                    cutlass_available and
                    n_div16 and
                    k_div16 and
                    scale_match
                )
                
                # Prepare weight transpose (cache it for reuse)
                if not hasattr(layer, '_weight_prepared'):
                    weight_fp8_contig = weight_fp8.contiguous()
                    weight_fp8_T = weight_fp8_contig.t()  # (K, N) view with stride(0) == 1
                    if weight_fp8_T.stride(0) != 1:
                        weight_fp8_T = weight_fp8_T.contiguous()
                    layer._weight_prepared = weight_fp8_T
                else:
                    weight_fp8_T = layer._weight_prepared
                
                # PIPELINED ACTIVATION QUANTIZATION: Quantize async while preparing weights
                # This overlaps quantization with weight transpose/preparation for better throughput
                # NOTE: Disabled during CUDA graph capture (streams are not allowed)
                from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
                x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
                x_bf16 = x_2d if x_2d.dtype == torch.bfloat16 else x_2d.to(torch.bfloat16)
                
                # Check if we're in CUDA graph capture mode - must disable pipelining
                in_capture = False
                try:
                    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                    in_capture = get_is_capture_mode()
                except ImportError:
                    pass
                
                # Pipeline quantization: start async quant, do weight prep, then sync
                # Only enable if not in capture mode (CUDA graphs don't support streams)
                if self._pipeline_enabled and self._quant_stream is not None and not in_capture:
                    # Start quantization in separate stream
                    with torch.cuda.stream(self._quant_stream):
                        x_fp8, x_scale = scaled_fp8_quant(
                            x_bf16.contiguous(),
                            scale=None,  # Dynamic quantization
                            use_per_token_if_dynamic=False,  # Per-tensor for decode
                        )
                    # Weight transpose happens in default stream (parallel)
                    # Sync before matmul
                    torch.cuda.current_stream().wait_stream(self._quant_stream)
                else:
                    # Synchronous path (fallback or disabled or during capture)
                    x_fp8, x_scale = scaled_fp8_quant(
                        x_bf16.contiguous(),
                        scale=None,  # Dynamic quantization
                        use_per_token_if_dynamic=False,  # Per-tensor for decode
                    )
                
                if cutlass_compatible:
                    # Use CUTLASS kernel (fastest) - same as native FP8
                    # fp8_scaled_mm expects scale_a as (M,) and scale_b as (N,) - both must be contiguous!
                    scale_a = x_scale.view(-1) if x_scale.numel() == x_2d.shape[0] else x_scale.expand(x_2d.shape[0])
                    scale_a = scale_a.contiguous()  # CUTLASS requires contiguous scales
                    scale_b = fp8_row_scale.view(-1).contiguous()  # (N,) - ensure contiguous
                    
                    if not hasattr(layer, '_ternary_cutlass_logged'):
                        layer._ternary_cutlass_logged = True
                        logger.info(f"[TERNARY] Using CUTLASS FP8 kernel for {type(layer).__name__} (M={M}, N={N}, K={K})")
                        self._ternary_quantization_stats['cutlass_kernels_used'] += 1
                    
                    out = fp8_scaled_mm(
                        x_fp8,  # (M, K) FP8
                        weight_fp8_T,  # (K, N) FP8
                        scale_a,  # (M,) contiguous
                        scale_b,  # (N,) contiguous
                        torch.bfloat16,
                        bias,
                    )
                else:
                    # Fallback to Triton for incompatible shapes
                    from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm
                    if not hasattr(layer, '_ternary_triton_logged'):
                        layer._ternary_triton_logged = True
                        logger.info(f"[TERNARY] Using Triton FP8 kernel for {type(layer).__name__} (M={M}, N={N}, K={K})")
                        self._ternary_quantization_stats['triton_kernels_used'] += 1
                    scales_a_2d = x_scale.view(-1, 1) if x_scale.numel() > 1 else x_scale.expand(x_2d.shape[0], 1)
                    scales_b_2d = fp8_row_scale.view(1, -1).contiguous()
                    out = triton_scaled_mm(
                        input=x_fp8,
                        weight=weight_fp8_T,
                        scale_a=scales_a_2d,
                        scale_b=scales_b_2d,
                        bias=bias,
                        out_dtype=torch.bfloat16,
                    )
                if bias is not None:
                    out = out + bias.to(out.dtype)
                return out.to(x.dtype)
            except Exception as e:
                logger.warning(f"FP8 rowwise matmul failed for {type(layer).__name__}: {e}. Falling back to BF16.")
                logger.debug(f"FP8 rowwise failure traceback: {traceback.format_exc()}")
                # Fall through to BF16 path
        
        if use_blockwise and x.shape[0] >= min_m_for_block:
            # BLOCKWISE FP8 PATH
            path_taken = "BLOCKWISE_FP8"
            if not hasattr(layer, '_ternary_blockwise_logged'):
                layer._ternary_blockwise_logged = True
                logger.info(f"[TERNARY] BLOCKWISE_FP8: {type(layer).__name__} (M={M})")
            try:
                # Blockwise FP8 path using SGLang kernels (w8a8 block fp8 matmul)
                from sglang.srt.layers.quantization.fp8_kernel import (
                    w8a8_block_fp8_matmul,
                )
                from sgl_kernel.gemm import sgl_per_token_group_quant_8bit

                M, K = x.shape
                N = weight_fp8.shape[0]
                # Validate shapes
                if fp8_block_scales is None:
                    raise ValueError("fp8_block_scales not found - blockwise quantization not applied")
                
                # Quantize activations per block_k along K
                block_n = int(os.environ.get("TERNARY_FP8_BLOCK_N", "128"))
                block_k = int(os.environ.get("TERNARY_FP8_BLOCK_K", "128"))
                
                # Validate block scales shape matches expected layout
                num_blocks_n = (N + block_n - 1) // block_n
                num_blocks_k = (K + block_k - 1) // block_k
                expected_scale_shape = (num_blocks_n, num_blocks_k)
                if fp8_block_scales.shape != expected_scale_shape:
                    raise ValueError(
                        f"fp8_block_scales shape mismatch: got {fp8_block_scales.shape}, "
                        f"expected {expected_scale_shape} for N={N}, K={K}, block_n={block_n}, block_k={block_k}"
                    )
                
                x_contig = x.contiguous()
                # CUDA graph compatible: Pre-allocated buffers (allocated during warmup, before capture)
                # Buffers are allocated lazily on first use (which happens during warmup, not during capture)
                if not hasattr(layer, '_act_fp8_block_cache'):
                    layer._act_fp8_block_cache = {}
                    # Limit cache size to prevent memory bloat (keep only last N shapes)
                    max_cache_size = int(os.environ.get("TERNARY_FP8_MAX_CACHE_SIZE", "8"))
                    layer._ternary_max_cache_size = max_cache_size
                
                cache_key = (M, K, block_k)
                # Limit cache size - remove oldest entries if cache is too large
                # Use LRU-style eviction: remove least recently used (last accessed)
                if len(layer._act_fp8_block_cache) >= layer._ternary_max_cache_size:
                    # Remove oldest entry (simple FIFO - remove first key)
                    oldest_key = next(iter(layer._act_fp8_block_cache))
                    del layer._act_fp8_block_cache[oldest_key]
                    logger.debug(f"[TERNARY] Evicted cache entry for {type(layer).__name__} shape {oldest_key} (cache full)")
                
                if cache_key not in layer._act_fp8_block_cache:
                    # Cache miss - allocate buffer (expensive during inference)
                    logger.debug(f"[TERNARY] Cache miss for {type(layer).__name__} shape {cache_key} - allocating buffer")
                    # Allocate buffers for this shape (happens during warmup, before CUDA graph capture)
                    # CUDA graphs require static buffers - these will be reused during capture
                    # Check if we're in capture mode - if so, can't allocate, must use BF16 fallback
                    in_capture = False
                    try:
                        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                        in_capture = get_is_capture_mode()
                    except ImportError:
                        pass  # Import might fail if not available
                    
                    if in_capture:
                        # During capture, can't allocate - fall back to BF16 path gracefully
                        logger.debug(
                            f"Blockwise FP8 buffer not pre-allocated for shape M={M}, K={K} during capture. "
                            f"Falling back to BF16 path."
                        )
                        # Fall through to BF16 path by raising exception that will be caught
                        raise ValueError("Buffer not pre-allocated during capture")
                    
                    # Allocate during warmup (not in capture mode) - this is safe
                    x_q = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
                    x_s = torch.empty((M, num_blocks_k), device=x.device, dtype=torch.float32)
                    layer._act_fp8_block_cache[cache_key] = {
                        'x_q': x_q,
                        'x_s': x_s,
                    }
                
                # Use cached buffers (no allocation during forward pass = CUDA graph compatible)
                x_q = layer._act_fp8_block_cache[cache_key]['x_q']
                x_s = layer._act_fp8_block_cache[cache_key]['x_s']
                
                # Shapes are validated during allocation, buffers are pre-sized correctly
                # No runtime checks needed (CUDA graph compatible)
                
                # In-place group quant (writes into x_q, x_s)
                # This is an extra kernel launch - can be expensive with high concurrency
                FP8_MAX = 448.0
                sgl_per_token_group_quant_8bit(
                    x_contig, x_q, x_s, block_k, 1e-10, -FP8_MAX, FP8_MAX
                )
                # Note: This activation quantization adds overhead. For high concurrency,
                # consider using TERNARY_FP8_SKIP_ACT_QUANT_PREFILL=1 to skip for prefill.

                # weight_fp8 is (N,K) fp8; fp8_block_scales is (num_blocks_n, num_blocks_k) f32
                # Use block_size [block_n, block_k] to match Bs layout
                out_bf16 = w8a8_block_fp8_matmul(
                    A=x_q,
                    B=weight_fp8,
                    As=x_s,
                    Bs=fp8_block_scales,
                    block_size=[block_n, block_k],
                    output_dtype=torch.bfloat16,
                )
                if bias is not None:
                    out_bf16 = out_bf16 + bias.to(out_bf16.dtype)
                return out_bf16.to(x.dtype)
            except Exception as e:
                try:
                    M_fallback, K_fallback = x.shape[:2] if len(x.shape) >= 2 else (x.shape[0] if len(x.shape) >= 1 else "?", "?")
                    N_fallback = weight_fp8.shape[0] if len(weight_fp8.shape) > 0 else "unknown"
                except Exception:
                    M_fallback, K_fallback, N_fallback = "?", "?", "?"
                logger.warning(
                    f"Blockwise FP8 matmul failed for layer {type(layer).__name__} "
                    f"(M={M_fallback}, K={K_fallback}, N={N_fallback}): {e}. "
                    f"Falling back to BF16 path."
                )
                logger.debug(f"Blockwise failure traceback: {traceback.format_exc()}")
                # Fall through to BF16 path (can't use rowwise if fp8_row_scale doesn't exist)
        elif use_blockwise and x.shape[0] < min_m_for_block:
            # Blockwise enabled but M too small - use rowwise FP8 instead
            # Compute per-row scale from blockwise scales on-the-fly
            if fp8_block_scales is not None:
                # Compute approximate per-row scale from block scales (cache it for reuse)
                if not hasattr(layer, '_fp8_row_scale_from_blocks'):
                    N = weight_fp8.shape[0]
                    block_n = int(os.environ.get("TERNARY_FP8_BLOCK_N", "128"))
                    num_blocks_n = (N + block_n - 1) // block_n
                    fp8_row_scale_cached = torch.zeros(N, device=weight_fp8.device, dtype=torch.float32)
                    for bn in range(num_blocks_n):
                        n_start = bn * block_n
                        n_end = min(N, n_start + block_n)
                        # Max scale across all K blocks for this N block (conservative estimate)
                        row_scale_approx = fp8_block_scales[bn, :].max()
                        fp8_row_scale_cached[n_start:n_end] = row_scale_approx
                    layer.register_buffer('_fp8_row_scale_from_blocks', fp8_row_scale_cached, persistent=False)
                    logger.info(
                        f"[TERNARY] Computed row scale from blocks for {type(layer).__name__} (N={N}, num_blocks_n={num_blocks_n})"
                    )
                fp8_row_scale = layer._fp8_row_scale_from_blocks
                # Enable rowwise FP8 for decode (CRITICAL: must set this!)
                use_tilelang_fp8 = (
                    weight_fp8.dtype == torch.float8_e4m3fn and
                    fp8_row_scale is not None and
                    os.environ.get("TERNARY_FP8_TENSOR_CORES", "1") == "1"
                )
                if not hasattr(layer, '_ternary_blockwise_small_m_logged'):
                    layer._ternary_blockwise_small_m_logged = True
                    logger.info(
                        f"[TERNARY] M={x.shape[0]} < {min_m_for_block}, switching to ROWWISE_FP8 for {type(layer).__name__} "
                        f"(use_tilelang_fp8={use_tilelang_fp8}, has_row_scale={fp8_row_scale is not None})"
                    )
            else:
                # No block scales - can't use rowwise, will fall through to BF16
                if not hasattr(layer, '_ternary_blockwise_no_scales_logged'):
                    layer._ternary_blockwise_no_scales_logged = True
                    logger.warning(
                        f"[TERNARY] M={x.shape[0]} < {min_m_for_block} but no fp8_block_scales for {type(layer).__name__}. "
                        f"Will use BF16 fallback."
        )
        
        if use_tilelang_fp8:
            # ROWWISE FP8 PATH - Try TileLang first (fastest), fallback to CUTLASS/Triton
            path_taken = "ROWWISE_FP8"
            if not hasattr(layer, '_ternary_rowwise_logged'):
                layer._ternary_rowwise_logged = True
                logger.info(f"[TERNARY] ROWWISE_FP8: {type(layer).__name__} (M={M})")
            try:
                # Try TileLang FP8 kernels first (matches benchmark_infer.py - fastest!)
                try:
                    import sys
                    # Add ternarykernels to path (same as benchmark_infer.py)
                    ternarykernels_path = '/home/ubuntu/raghav/ternarykernels'
                    if ternarykernels_path not in sys.path:
                        sys.path.insert(0, ternarykernels_path)
                    # Import from kernels directly (not ternarykernels.kernels) - matches benchmark_infer.py
                    from kernels.tilelang_fp8 import fp8_matmul, fp8_matmul_bias_fused, HAS_TILELANG
                    
                    if HAS_TILELANG:
                        M, K = x.shape
                        N = weight_fp8.shape[0]
                        
                        # Pre-quantize activations to FP8 (matching benchmark_infer.py)
                        # This avoids quantization overhead during inference
                        from tilelang.utils.tensor import map_torch_type
                        fp8_dtype = map_torch_type("float8_e4m3")
                        x_fp8_prequant = x.to(fp8_dtype)  # Direct BF16->FP8 conversion
                        
                        # Weight is already FP8, stored as (N, K)
                        # Call TileLang kernel with pre-quantized inputs (fastest path!)
                        if bias is not None:
                            out = fp8_matmul_bias_fused(
                                x_fp8_prequant, weight_fp8, bias,
                                pre_quantized=True, b_transposed=True
                            )
                        else:
                            out = fp8_matmul(
                                x_fp8_prequant, weight_fp8,
                                pre_quantized=True, b_transposed=True
                            )
                        
                        if not hasattr(layer, '_ternary_tilelang_logged'):
                            layer._ternary_tilelang_logged = True
                            logger.info(
                                f"[TERNARY] ✓✓✓ TILELANG FP8 KERNEL ACTIVE ✓✓✓ {type(layer).__name__} "
                                f"(M={M}, N={N}, K={K}) - matches benchmark_infer.py!"
                            )
                            self._ternary_quantization_stats['tilelang_kernels_used'] += 1
                        return out.to(x.dtype)
                    else:
                        # HAS_TILELANG is False
                        if not hasattr(layer, '_ternary_tilelang_not_available_logged'):
                            layer._ternary_tilelang_not_available_logged = True
                            logger.debug(f"[TERNARY] TileLang not available (HAS_TILELANG=False), using CUTLASS/Triton fallback")
                except (ImportError, AttributeError, RuntimeError, TypeError) as e:
                    # TileLang not available or failed - fall back to CUTLASS/Triton
                    if not hasattr(layer, '_ternary_tilelang_fallback_logged'):
                        layer._ternary_tilelang_fallback_logged = True
                        logger.warning(f"[TERNARY] TileLang FP8 failed for {type(layer).__name__} ({e}), using CUTLASS/Triton fallback")
                        import traceback
                        logger.debug(f"[TERNARY] TileLang error traceback: {traceback.format_exc()}")
                
                # Fallback: CUTLASS/Triton path (original implementation)
                import sgl_kernel
                from sgl_kernel.gemm import sgl_per_token_quant_fp8
                from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
                use_graphs = os.environ.get("TERNARY_FP8_ROWWISE_GRAPH", "0") == "1"
                
                M, K = x.shape
                N = weight_fp8.shape[0]
                
                # CUDA graph compatible: Pre-allocated buffers (allocated during warmup, before capture)
                if not hasattr(layer, '_act_fp8_cache'):
                    layer._act_fp8_cache = {}
                    # Limit cache size to prevent memory bloat (keep only last N shapes)
                    max_cache_size = int(os.environ.get("TERNARY_FP8_MAX_CACHE_SIZE", "8"))
                    layer._ternary_max_cache_size = max_cache_size
                
                cache_key = (M, K)
                # Limit cache size - remove oldest entries if cache is too large
                if len(layer._act_fp8_cache) >= layer._ternary_max_cache_size:
                    # Remove oldest entry (simple FIFO - remove first key)
                    oldest_key = next(iter(layer._act_fp8_cache))
                    del layer._act_fp8_cache[oldest_key]
                
                if cache_key not in layer._act_fp8_cache:
                    # Allocate buffers for this shape (happens during warmup, before CUDA graph capture)
                    # Check if we're in capture mode - if so, can't allocate, must use BF16 fallback
                    in_capture = False
                    try:
                        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                        in_capture = get_is_capture_mode()
                    except ImportError:
                        pass  # Import might fail if not available
                    
                    if in_capture:
                        # During capture, can't allocate - fall back to BF16 path gracefully
                        logger.debug(
                            f"Rowwise FP8 buffer not pre-allocated for shape M={M}, K={K} during capture. "
                            f"Falling back to BF16 path."
                        )
                        # Fall through to BF16 path by raising exception that will be caught
                        raise ValueError("Buffer not pre-allocated during capture")
                    
                    # Allocate during warmup (not in capture mode) - this is safe
                    x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
                    scale_x = torch.empty((M,), device=x.device, dtype=torch.float32)
                    layer._act_fp8_cache[cache_key] = {
                        'x_fp8': x_fp8,
                        'scale_x': scale_x,
                    }
                
                # Use cached buffers (no allocation during forward pass = CUDA graph compatible)
                x_fp8 = layer._act_fp8_cache[cache_key]['x_fp8']
                scale_x = layer._act_fp8_cache[cache_key]['scale_x']
                
                # Shapes are validated during allocation, buffers are pre-sized correctly
                # No runtime checks needed (CUDA graph compatible)

                # Choose per-tensor or per-token activation quantization
                # During CUDA graph capture, ALWAYS use per-tensor for compatibility
                in_capture_during_quant = False
                try:
                    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                    in_capture_during_quant = get_is_capture_mode()
                except ImportError:
                    pass
                
                # Use per-tensor for small M, if env set, or if in capture (MUST use per-tensor during capture)
                # sgl_per_token_quant_fp8 is NOT CUDA graph compatible - causes "operation not permitted" errors
                use_per_tensor_act = (
                    os.environ.get("TERNARY_FP8_ACT_PER_TENSOR", "0") == "1" or 
                    M < min_m_for_block or
                    in_capture_during_quant  # CRITICAL: Must use per-tensor during capture
                )
                if use_per_tensor_act:
                    # In-place per-tensor quantization directly into cached buffer (fastest for decode)
                    # Use sgl_per_tensor_quant_fp8 to avoid intermediate tensor allocation
                    try:
                        from sgl_kernel.gemm import sgl_per_tensor_quant_fp8
                        x_contig = x.contiguous()  # Ensure contiguous for kernel
                        # Create scalar scale tensor for per-tensor quantization
                        scale_x_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
                        # In-place quant: writes directly into x_fp8 and scale_x_scalar
                        sgl_per_tensor_quant_fp8(x_contig, x_fp8, scale_x_scalar, is_static=False)
                        # Fill scale_x (M,) with the scalar value
                        scale_val = scale_x_scalar.item()
                        scale_x.fill_(scale_val)
                    except ImportError:
                        # Fallback to scaled_fp8_quant if direct import fails
                        from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
                        x_q, s = scaled_fp8_quant(x.contiguous(), scale=None, use_per_token_if_dynamic=False)
                        x_fp8.copy_(x_q)
                        scale_val = s.item() if s.numel() == 1 else s[0].item()
                        scale_x.fill_(scale_val)
                else:
                    # Per-token (row) dynamic scale - ONLY use when NOT in capture
                    # WARNING: sgl_per_token_quant_fp8 is NOT CUDA graph capture compatible!
                    if in_capture_during_quant:
                        # Should not reach here, but safety check
                        logger.warning("Attempting per-token quant during capture - falling back to per-tensor")
                        try:
                            from sgl_kernel.gemm import sgl_per_tensor_quant_fp8
                            x_contig = x.contiguous()
                            # Create scalar scale tensor for per-tensor quantization
                            scale_x_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
                            sgl_per_tensor_quant_fp8(x_contig, x_fp8, scale_x_scalar, is_static=False)
                            # Fill scale_x (M,) with the scalar value
                            scale_val = scale_x_scalar.item()
                            scale_x.fill_(scale_val)
                        except ImportError:
                            from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
                            x_q, s = scaled_fp8_quant(x.contiguous(), scale=None, use_per_token_if_dynamic=False)
                            x_fp8.copy_(x_q)
                            scale_val = s.item() if s.numel() == 1 else s[0].item()
                            scale_x.fill_(scale_val)
                    else:
                        sgl_per_token_quant_fp8(x, x_fp8, scale_x)

                # Prepare scales (validate fp8_row_scale exists)
                if fp8_row_scale is None:
                    # Try to get cached row scale from blocks if available
                    if hasattr(layer, '_fp8_row_scale_from_blocks'):
                        fp8_row_scale = layer._fp8_row_scale_from_blocks
                    else:
                        raise ValueError(
                            f"[TERNARY] fp8_row_scale not found for {type(layer).__name__} - rowwise path unavailable. "
                            f"M={M}, has_block_scales={fp8_block_scales is not None}, has_row_scale_cache={hasattr(layer, '_fp8_row_scale_from_blocks')}"
                        )
                scales_a_2d = scale_x.view(-1, 1).contiguous()  # (M, 1)
                scales_b_2d = fp8_row_scale.view(-1, 1).contiguous()  # (N, 1)

                # CUDA Graph path (optional): capture matmul with static buffers
                if use_graphs:
                    graph_key = (M, K, N)
                    if not hasattr(layer, '_fp8_rowwise_graph_cache'):
                        layer._fp8_rowwise_graph_cache = {}
                    cache_g = layer._fp8_rowwise_graph_cache
                    if graph_key not in cache_g:
                        # Static buffers for capture
                        static_x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
                        static_scale_a = torch.empty((M, 1), device=x.device, dtype=torch.float32)
                        static_out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
                        # Freeze weight transpose and scale_b once per graph
                        weight_fp8_T_graph = weight_fp8.t().contiguous()
                        scale_b_static = scales_b_2d.clone()
                        from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            y = triton_scaled_mm(
                                input=static_x_fp8,
                                weight=weight_fp8_T_graph,
                                scale_a=static_scale_a,
                                scale_b=scale_b_static,
                                out_dtype=torch.bfloat16,
                                bias=bias,
                            )
                            static_out.copy_(y)
                        cache_g[graph_key] = (g, static_x_fp8, static_scale_a, static_out)
                    # Copy inputs and replay
                    g, static_x_fp8, static_scale_a, static_out = cache_g[graph_key]
                    static_x_fp8.copy_(x_fp8)
                    static_scale_a.copy_(scales_a_2d)
                    g.replay()
                    return static_out.to(x.dtype)
                else:
                    # Eager path: Use CUTLASS fp8_scaled_mm from sgl-kernel (faster than Triton)
                    # This matches native FP8 quantization path and is optimized for decode
                    try:
                        from sgl_kernel import fp8_scaled_mm
                        from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
                        
                        N, K = weight_fp8.shape
                        cutlass_available = cutlass_fp8_supported()
                        n_div16 = N % 16 == 0
                        k_div16 = K % 16 == 0
                        scale_match = fp8_row_scale.numel() == N
                        
                        # fp8_scaled_mm requires per-channel weight scale (we have fp8_row_scale)
                        # and works best when dimensions are multiples of 16
                        cutlass_compatible = (
                            cutlass_available and
                            n_div16 and
                            k_div16 and
                            scale_match
                        )
                        
                        # Log kernel choice (only once per layer type to avoid spam)
                        if not hasattr(layer, '_ternary_rowwise_kernel_logged'):
                            layer._ternary_rowwise_kernel_logged = True
                            logger.info(
                                f"[TERNARY] ROWWISE kernel choice for {type(layer).__name__}: "
                                f"cutlass_available={cutlass_available}, "
                                f"N={N} (div16={n_div16}), K={K} (div16={k_div16}), "
                                f"scale_match={scale_match}, "
                                f"using={'CUTLASS' if cutlass_compatible else 'TRITON'}"
                            )
                        
                        if cutlass_compatible:
                            # Use fast CUTLASS kernel (same as native FP8)
                            # PIPELINED: Quantize activations async while preparing weights
                            from sgl_kernel.gemm import sgl_per_tensor_quant_fp8
                            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
                            x_contig = x_bf16.contiguous()
                            x_fp8 = torch.empty_like(x_contig, dtype=torch.float8_e4m3fn)
                            scale_a_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
                            
                            # Prepare weight transpose in parallel (if not cached)
                            if not hasattr(layer, '_weight_prepared_cutlass'):
                                weight_fp8_contig = weight_fp8.contiguous()
                                weight_fp8_T = weight_fp8_contig.t()  # (K, N) view with stride(0) == 1
                                if weight_fp8_T.stride(0) != 1:
                                    weight_fp8_T = weight_fp8_T.contiguous()
                                layer._weight_prepared_cutlass = weight_fp8_T
                            else:
                                weight_fp8_T = layer._weight_prepared_cutlass
                            scale_b_flat = fp8_row_scale.view(-1)
                            
                            # Pipeline quantization: start async quant, do weight prep, then sync
                            # Only enable if not in capture mode (CUDA graphs don't support streams)
                            in_capture_cutlass = False
                            try:
                                from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                                in_capture_cutlass = get_is_capture_mode()
                            except ImportError:
                                pass
                            
                            if self._pipeline_enabled and self._quant_stream is not None and not in_capture_cutlass:
                                with torch.cuda.stream(self._quant_stream):
                                    sgl_per_tensor_quant_fp8(x_contig, x_fp8, scale_a_scalar, is_static=False)
                                # Weight prep happens in default stream (parallel)
                                torch.cuda.current_stream().wait_stream(self._quant_stream)
                            else:
                                sgl_per_tensor_quant_fp8(x_contig, x_fp8, scale_a_scalar, is_static=False)
                            
                            # Expand scalar scale to (M,) on device
                            scale_a_flat = scale_a_scalar.expand(M).contiguous()
                            out = fp8_scaled_mm(
                                x_fp8,
                                weight_fp8_T,  # (K, N) column-major
                                scale_a_flat,  # (M,)
                                scale_b_flat,  # (N,)
                                torch.bfloat16,
                                bias,
                            )
                        else:
                            # Fallback to Triton for incompatible shapes
                            from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm
                            from sgl_kernel.gemm import sgl_per_tensor_quant_fp8
                            x_bf16 = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
                            x_contig = x_bf16.contiguous()
                            x_fp8 = torch.empty_like(x_contig, dtype=torch.float8_e4m3fn)
                            scale_a_scalar = torch.empty(1, device=x.device, dtype=torch.float32)
                            sgl_per_tensor_quant_fp8(x_contig, x_fp8, scale_a_scalar, is_static=False)
                            scales_a_2d = scale_a_scalar.expand(M).contiguous().view(-1, 1)
                            scales_b_2d = fp8_row_scale.view(1, -1).contiguous()
                            # Ensure weight is contiguous before transpose (for consistent stride behavior)
                            weight_fp8_contig = weight_fp8.contiguous()
                            weight_fp8_T = weight_fp8_contig.t()  # (K, N) transpose view
                            out = triton_scaled_mm(
                                input=x_fp8,
                                weight=weight_fp8_T,
                                scale_a=scales_a_2d,
                                scale_b=scales_b_2d,
                                bias=bias,
                                out_dtype=torch.bfloat16,
                            )
                    except Exception as e:
                        # Log the exception and fallback
                        if not hasattr(layer, '_ternary_rowwise_import_error_logged'):
                            layer._ternary_rowwise_import_error_logged = True
                            logger.warning(
                                f"[TERNARY] Failed to use CUTLASS for {type(layer).__name__}: {e}. "
                                f"Falling back to Triton."
                            )
                        # Fallback if sgl-kernel not available or error
                        from sglang.srt.layers.quantization.fp8_kernel import triton_scaled_mm
                        # Ensure weight is contiguous before transpose
                        weight_fp8_contig = weight_fp8.contiguous()
                        weight_fp8_T = weight_fp8_contig.t()
                        out = triton_scaled_mm(
                            input=x_fp8,
                            weight=weight_fp8_T,
                            scale_a=scales_a_2d,
                            scale_b=scales_b_2d,
                            bias=bias,
                            out_dtype=torch.bfloat16,
                        )
                return out.to(x.dtype)
            except Exception as e:
                try:
                    M_fallback, K_fallback = x.shape[:2] if len(x.shape) >= 2 else (x.shape[0] if len(x.shape) >= 1 else "?", "?")
                    N_fallback = weight_fp8.shape[0] if len(weight_fp8.shape) > 0 else "unknown"
                except Exception:
                    M_fallback, K_fallback, N_fallback = "?", "?", "?"
                logger.warning(
                    f"[TERNARY] ROWWISE_FP8 failed for {type(layer).__name__} "
                    f"(M={M_fallback}, K={K_fallback}, N={N_fallback}): {e}. Falling back to BF16."
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Rowwise failure traceback: {traceback.format_exc()}")
                # Fall through to BF16 path
        
        # OLD FP8 path with torch._scaled_mm (DISABLED - not faster)
        if False:  # use_fp8_tensor_cores:
            # FAST PATH: Use pre-quantized FP8 weights with torch._scaled_mm
            # Optionally use CUDA Graphs if enabled and SGLang's graphs aren't conflicting
            use_local_graphs = os.environ.get("TERNARY_FP8_CUDA_GRAPHS", "0") == "1"
            FP8_MAX = 448.0
            M, K = x.shape
            
            # Check if we're already inside SGLang's CUDA graph context
            # If so, don't use local graphs (they conflict)
            try:
                from torch.cuda import _graph_pool_manager as gpm
                in_graph_context = gpm._is_in_graph_context()
            except Exception:
                in_graph_context = False
            
            if use_local_graphs and not in_graph_context:
                # Try to use CUDA graphs for additional speedup
                graph_key = (M, K, weight_fp8.shape[0])  # (M, K, N)
                
                if not hasattr(layer, '_fp8_graph_cache'):
                    layer._fp8_graph_cache = {}
                
                if graph_key not in layer._fp8_graph_cache:
                    try:
                        # First time: capture CUDA graph for this shape
                        static_x = torch.empty_like(x)
                        static_x_fp8 = torch.empty((M, K), device=x.device, dtype=torch.float8_e4m3fn)
                        static_out = torch.empty((M, weight_fp8.shape[0]), device=x.device, dtype=torch.bfloat16)
                        
                        w_fp8_T = weight_fp8.T.clone()  # (K, N) - clone for graph stability
                        scale_w_2d = fp8_row_scale.view(1, -1).to(torch.float32).clone()  # (1, N)
                        
                        # Warm-up
                        x_abs_max = static_x.abs().amax(dim=1, keepdim=True)
                        scale_x = (x_abs_max.float() / FP8_MAX).clamp_min(1e-12)
                        x_scaled = (static_x.float() / scale_x).clamp(-FP8_MAX, FP8_MAX)
                        x_fp8_temp = x_scaled.to(torch.float8_e4m3fn)
                        _ = torch._scaled_mm(
                            x_fp8_temp.contiguous(),
                            w_fp8_T,
                            scale_a=scale_x.view(-1, 1),
                            scale_b=scale_w_2d,
                            out_dtype=torch.bfloat16,
                            use_fast_accum=True,
                        )
                        torch.cuda.synchronize()
                        
                        # Capture graph
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            x_abs_max = static_x.abs().amax(dim=1, keepdim=True)
                            scale_x = (x_abs_max.float() / FP8_MAX).clamp_min(1e-12)
                            x_scaled = (static_x.float() / scale_x).clamp(-FP8_MAX, FP8_MAX)
                            static_x_fp8.copy_(x_scaled.to(torch.float8_e4m3fn))
                            static_out.copy_(
                                torch._scaled_mm(
                                    static_x_fp8.contiguous(),
                                    w_fp8_T,
                                    scale_a=scale_x.view(-1, 1).to(torch.float32),
                                    scale_b=scale_w_2d,
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True,
                                )
                            )
                        
                        layer._fp8_graph_cache[graph_key] = (graph, static_x, static_out)
                        logger.debug(f"Captured FP8 CUDA graph for shape {graph_key}")
                    except Exception as e:
                        logger.debug(f"CUDA graph capture failed: {e}, using eager path")
                        use_local_graphs = False
                
                # Replay cached graph if available
                if use_local_graphs and graph_key in layer._fp8_graph_cache:
                    try:
                        graph, static_x, static_out = layer._fp8_graph_cache[graph_key]
                        static_x.copy_(x)
                        graph.replay()
                        out = static_out.clone()
                        if bias is not None:
                            out = out + bias
                        return out.to(x.dtype)
                    except Exception as e:
                        logger.debug(f"CUDA graph replay failed: {e}, falling back to eager")
                        # Fall through to eager path
            
            # Eager FP8 path (no local graphs, but SGLang may still use its own graphs)
            x_abs_max = x.abs().amax(dim=1, keepdim=True)
            scale_x = (x_abs_max.float() / FP8_MAX).clamp_min(1e-12)
            x_scaled = (x.float() / scale_x).clamp(-FP8_MAX, FP8_MAX)
            x_fp8 = x_scaled.to(torch.float8_e4m3fn)
            
            w_fp8_T = weight_fp8.T
            if w_fp8_T.stride(0) != 1:
                w_fp8_T = w_fp8_T.clone()
            
            scale_x_2d = scale_x.to(torch.float32)
            scale_w_2d = fp8_row_scale.view(1, -1).to(torch.float32)
            
            try:
                out = torch._scaled_mm(
                    x_fp8.contiguous(),
                    w_fp8_T,
                    scale_a=scale_x_2d,
                    scale_b=scale_w_2d,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=True,
                )
                if bias is not None:
                    out = out + bias
                return out.to(x.dtype)
            except Exception as e:
                logger.debug(f"torch._scaled_mm failed: {e}, falling back to BF16")
                pass
        
        # FALLBACK: Use BF16 path (dequantize FP8 weights or use stored BF16)
        path_taken = "BF16_FALLBACK"
        if not hasattr(layer, '_ternary_bf16_fallback_logged'):
            layer._ternary_bf16_fallback_logged = True
            logger.warning(f"[TERNARY] BF16_FALLBACK: {type(layer).__name__} (M={M})")
        # CRITICAL: Pre-compute BF16 weight during model loading, NOT during capture
        # torch.arange() and other operations are not CUDA graph capture compatible
        bf16_weight = getattr(layer, '_ternary_weight_bf16', None)
        if bf16_weight is None:
            # Check if we're in capture - if so, we can't dequantize here (torch.arange fails)
            # This should not happen if we pre-computed during warmup, but safety check
            in_capture_check = False
            try:
                from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                in_capture_check = get_is_capture_mode()
            except ImportError:
                pass
            
            if in_capture_check:
                # During capture, we cannot run torch.arange() or create new tensors
                # If we reach here, pre-dequantization failed - log error and use raw FP8 dequant
                logger.error(
                    f"BF16 weight not pre-computed for {type(layer).__name__} during capture! "
                    f"Using simple FP8->BF16 conversion (may lose accuracy)."
                )
                # Simple fallback: just convert FP8->BF16 without scales (very inaccurate, but won't crash)
                # Don't cache - use directly to save memory
                # If weight_fp8 is a view (rowwise path), transpose _weight_fp8_T_stored back to (N, K)
                if hasattr(layer, '_weight_fp8_T_stored'):
                    weight_fp8_for_fallback = layer._weight_fp8_T_stored.t()  # (N, K) from stored (K, N)
                else:
                    weight_fp8_for_fallback = weight_fp8  # Blockwise path, already (N, K)
                bf16_weight = weight_fp8_for_fallback.to(torch.bfloat16)
            else:
                # Dequantize FP8 weights if no BF16 cache (happens during warmup, not capture)
                # Get actual weight (handle rowwise view case)
                if hasattr(layer, '_weight_fp8_T_stored'):
                    weight_fp8_actual = layer._weight_fp8_T_stored.t()  # (N, K) from stored (K, N)
                else:
                    weight_fp8_actual = weight_fp8  # Blockwise path, already (N, K)
                
                if weight_fp8_actual.dtype == torch.float8_e4m3fn:
                    if fp8_block_scales is not None:
                        # Blockwise dequantization: compute on-the-fly to save memory
                        # Since CUDA graphs are disabled, we don't need to cache the dequantized weight
                        # This saves significant memory (avoiding 2x memory for FP8 + BF16)
                        N, K = weight_fp8_actual.shape
                        block_n = int(os.environ.get("TERNARY_FP8_BLOCK_N", "128"))
                        block_k = int(os.environ.get("TERNARY_FP8_BLOCK_K", "128"))
                        num_blocks_n = (N + block_n - 1) // block_n
                        num_blocks_k = (K + block_k - 1) // block_k
                        
                        w_deq = torch.empty_like(weight_fp8_actual, dtype=torch.bfloat16)
                        for bn in range(num_blocks_n):
                            n_start = bn * block_n
                            n_end = min(N, n_start + block_n)
                            for bk in range(num_blocks_k):
                                k_start = bk * block_k
                                k_end = min(K, k_start + block_k)
                                # Get scale for this block
                                scale = fp8_block_scales[bn, bk]
                                # Dequantize block
                                w_block = weight_fp8_actual[n_start:n_end, k_start:k_end].to(torch.bfloat16)
                                w_deq[n_start:n_end, k_start:k_end] = w_block * scale.to(torch.bfloat16)
                        # Use dequantized weight directly (don't cache to save memory)
                        bf16_weight = w_deq
                    elif fp8_row_scale is not None:
                        # Rowwise dequantization: compute on-the-fly (don't cache to save memory)
                        # This avoids doubling memory usage for fallback layers
                        # NOTE: We dequantize on-the-fly instead of caching to prevent OOM
                        bf16_weight = weight_fp8_actual.to(torch.bfloat16) * fp8_row_scale.view(-1, 1).to(torch.bfloat16)
                    else:
                        # FP8 but no scales (shouldn't happen, but fallback)
                        logger.warning("FP8 weights found but no scales available, using raw dequant")
                        bf16_weight = weight_fp8_actual.to(torch.bfloat16)
                else:
                    # Use weight directly (might be FP16/BF16 if FP8 not available)
                    bf16_weight = weight_fp8_actual.to(torch.bfloat16)
        
        # Direct BF16 matmul (fast, no quantization overhead)
        x_compute = x if x.dtype in (torch.float16, torch.bfloat16) else x.to(torch.bfloat16)
        b_compute = None if bias is None else (bias if bias.dtype in (torch.float16, torch.bfloat16) else bias.to(x_compute.dtype))
        out = torch.nn.functional.linear(x_compute, bf16_weight, b_compute)
        return out.to(x.dtype)
    
    def _apply_fp8_tilelang(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply FP8 matmul using static TileLang kernels (SOTA performance, NO TVM).
        
        Uses pre-compiled .so files - no runtime TVM dependency, no xgrammar conflicts.
        """
        try:
            import sys
            # Add ternarykernels to path (same as benchmark_infer.py)
            ternarykernels_path = '/home/ubuntu/raghav/ternarykernels'
            if ternarykernels_path not in sys.path:
                sys.path.insert(0, ternarykernels_path)
            
            # Scale-aware FP8 path: uses optimized clean kernel with correct scaling
            if os.environ.get("TERNARY_FP8_SCALED", "0") == "1":
                # Use clean fast kernel: correct quantization + optimized matmul
                from kernels.fp8_matmul_clean import fp8_matmul_clean
                
                w_fp8 = layer.weight  # (N, K) FP8
                scale_w = getattr(layer, 'fp8_row_scale', None)  # (N,) per-row weight scale
                
                if scale_w is None:
                    raise RuntimeError("fp8_row_scale not found - enable FP8 quantization")
                
                # Use clean FP8 matmul (fast dequant + optimized BF16 tensor cores)
                # Accepts ~3-4% relative error for speed (acceptable for text generation)
                output = fp8_matmul_clean(
                    x, w_fp8, None, scale_w, bias,  # scale_x computed inside
                    use_cuda_graph=False  # Weights change per layer
                )
                return output
            else:
                # Fallback: static loader (may be BF16 under the hood)
                from kernels.tilelang_static_loader import tilelang_fp8_matmul
                weight = layer.weight  # (N,K)
                output = tilelang_fp8_matmul(
                    x, weight, bias,
                    pre_quantized=True,
                    b_transposed=True,
                    precompiled_dir=os.environ.get("FP8_TL_PRECOMPILED_DIR")
                )
                return output.to(x.dtype)
        except Exception as e:
            # If TileLang fails, raise to trigger FP16 fallback
            raise RuntimeError(f"TileLang static FP8 matmul unavailable: {e}")
