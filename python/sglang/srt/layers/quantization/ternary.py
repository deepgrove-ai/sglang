"""Ternary quantization with FP8 storage for SGLang.

This implementation uses CUTLASS/Triton FP8 kernels for SOTA performance.
NO TVM dependency = NO conflicts with xgrammar.

Key features:
- CUTLASS/Triton FP8 tensor core matmul
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
        logger.info("[TERNARY] Quantization initialized - will track FP8/CUTLASS usage")
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
        4. Store in format compatible with CUTLASS FP8 tensor core matmul
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

            # Standard FP8 rowwise quantization path - EXACTLY matching qwen2_correct.py quantize()
            logger.info(f"Quantizing layer to ternary+FP8: {weight.shape}")
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
                return
            
            # Row-wise FP8 quantization for weights (per-output-channel)
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
                return
            
            # Store FP8 weights in (N, K) directly; use transpose VIEW at matmul time (column-major for CUTLASS)
            layer._ternary_fp8_enabled = True
            layer.weight.data = weight_fp8_orig  # (N, K) FP8
            logger.info(
                f"[TERNARY] ✓ Quantized to FP8 (rowwise): stored (N,K)={layer.weight.shape}, dtype={layer.weight.dtype}"
            )
            self._ternary_quantization_stats['fp8_quantized'] += 1
            self._ternary_quantization_stats['fp8_rowwise'] += 1
            layer.register_buffer('fp8_row_scale', scale_row.view(-1).contiguous(), persistent=False)
            layer.register_buffer('ternary_alpha', alpha.view(-1).contiguous(), persistent=False)
            layer._ternary_original_dtype = original_dtype
            
            # Initialize cache dicts
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
        
        Uses pre-quantized FP8 rowwise weights (stored during process_weights_after_loading).
        Uses CUTLASS FP8 kernels (fastest compatible), falls back to Triton, then BF16.
        Activations are quantized on-the-fly with adaptive per-token/per-tensor strategy.
        """
        # Use pre-quantized weights (quantized ONCE during model loading, not every forward pass!)
        weight = layer.weight
        fp8_enabled = getattr(layer, '_ternary_fp8_enabled', False)
        
        # Log path selection on first call per layer type
        if not hasattr(layer, '_ternary_path_logged'):
            layer._ternary_path_logged = True
            logger.info(
                f"[TERNARY] Path check for {type(layer).__name__}: "
                f"fp8_enabled={fp8_enabled}, weight.dtype={weight.dtype}"
            )
        
        M = x.shape[0] if len(x.shape) >= 1 else 1
        
        # FP8 rowwise path: weights stored as (N, K)
        weight_fp8 = weight  # Already (N, K) FP8
        fp8_row_scale = getattr(layer, 'fp8_row_scale', None)  # (N,) per-row scale for weights
        
        # USE CUTLASS/Triton FP8 kernels for FP8 tensor core speedup
        
        # Check if we're in CUDA graph capture mode
        in_capture_mode = False
        try:
            from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
            in_capture_mode = get_is_capture_mode()
        except ImportError:
            pass
        
        # Log why FP8 is/isn't enabled
        if not hasattr(layer, '_ternary_fp8_check_logged'):
            layer._ternary_fp8_check_logged = True
            logger.info(
                f"[TERNARY] FP8 path check for {type(layer).__name__}: "
                f"fp8_enabled={fp8_enabled}, weight.dtype={weight_fp8.dtype}"
            )
        
        # FP8 rowwise matmul path using CUTLASS/Triton
        if fp8_row_scale is not None and weight_fp8.dtype == torch.float8_e4m3fn:
            # ROWWISE FP8 PATH: Use CUTLASS (fastest compatible), fallback to Triton
            if not hasattr(layer, '_ternary_fp8_rowwise_logged'):
                layer._ternary_fp8_rowwise_logged = True
                logger.info(
                    f"[TERNARY] Using FP8 rowwise matmul for {type(layer).__name__} (M={M}): "
                    f"BF16 activations + FP8 weights"
                )
            try:
                # CUTLASS/Triton path
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
                
                # OPTIMIZED ACTIVATION QUANTIZATION: Use best kernel for batch size
                # SGLang already batches requests, so x is (M, K) where M = sum of tokens across all requests
                # For large M (prefill): per-token quantization is more efficient
                # For small M (decode): per-tensor quantization is faster
                from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
                x_2d = x.view(-1, x.shape[-1]) if x.dim() > 2 else x
                x_bf16 = x_2d if x_2d.dtype == torch.bfloat16 else x_2d.to(torch.bfloat16)
                
                # Adaptive quantization strategy: per-token for large batches, per-tensor for small
                # Per-token is more accurate and efficient for large M (better GPU utilization)
                # Per-tensor is faster for small M (fewer kernel launches)
                quant_threshold = int(os.environ.get("TERNARY_FP8_PER_TOKEN_THRESHOLD", "64"))
                use_per_token = M >= quant_threshold
                
                # Check if we're in CUDA graph capture mode - must disable pipelining
                in_capture = False
                try:
                    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                    in_capture = get_is_capture_mode()
                except ImportError:
                    pass
                
                # Use scaled_fp8_quant directly - it handles allocation safely and efficiently
                # Buffer pre-allocation caused memory access issues, so use standard allocation path
                x_bf16_contig = x_bf16 if x_bf16.is_contiguous() else x_bf16.contiguous()
                from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
                
                if self._pipeline_enabled and self._quant_stream is not None and not in_capture:
                    # Pipeline quantization: quantize async while preparing weights
                    with torch.cuda.stream(self._quant_stream):
                        x_fp8, x_scale = scaled_fp8_quant(
                            x_bf16_contig,
                            scale=None,
                            use_per_token_if_dynamic=use_per_token,
                        )
                    torch.cuda.current_stream().wait_stream(self._quant_stream)
                else:
                    # Synchronous path: optimize for CUDA graph capture (no streams)
                    x_fp8, x_scale = scaled_fp8_quant(
                        x_bf16_contig,
                        scale=None,
                        use_per_token_if_dynamic=use_per_token,
                    )
                
                if cutlass_compatible:
                    # OPTIMIZED: Minimize scale tensor operations (major bottleneck at high concurrency)
                    # Pre-compute scale_b once (never changes - weight scales are static)
                    if not hasattr(layer, '_scale_b_precomputed'):
                        scale_b = fp8_row_scale.view(-1)
                        if not scale_b.is_contiguous():
                            scale_b = scale_b.contiguous()
                        layer._scale_b_precomputed = scale_b
                    else:
                        scale_b = layer._scale_b_precomputed
                    
                    # CUTLASS REQUIRES contiguous scales - ensure they are contiguous
                    # Optimize: only call contiguous() if needed (check first to avoid unnecessary copy)
                    if x_scale.numel() == 1:
                        # Per-tensor: expand scalar to (M,)
                        scale_a = x_scale.expand(M)
                        if not scale_a.is_contiguous():
                            scale_a = scale_a.contiguous()
                    else:
                        # Per-token: x_scale is already (M,) - flatten if needed
                        scale_a = x_scale.view(-1) if x_scale.dim() > 1 else x_scale
                        if not scale_a.is_contiguous():
                            scale_a = scale_a.contiguous()
                    
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
                return out.to(x.dtype)
            except Exception as e:
                logger.warning(f"FP8 rowwise matmul failed for {type(layer).__name__}: {e}. Falling back to BF16.")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"FP8 rowwise failure traceback: {traceback.format_exc()}")
                # Fall through to BF16 path
        
        # FALLBACK: Use BF16 path (dequantize FP8 weights or use stored BF16)
        if not hasattr(layer, '_ternary_bf16_fallback_logged'):
            layer._ternary_bf16_fallback_logged = True
            logger.warning(f"[TERNARY] BF16_FALLBACK: {type(layer).__name__} (M={M})")
        
        # Dequantize FP8 weights to BF16 for fallback
        bf16_weight = getattr(layer, '_ternary_weight_bf16', None)
        if bf16_weight is None:
            # Check if we're in capture - if so, we can't dequantize here
            in_capture_check = False
            try:
                from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
                in_capture_check = get_is_capture_mode()
            except ImportError:
                pass
            
            if in_capture_check:
                logger.error(
                    f"BF16 weight not pre-computed for {type(layer).__name__} during capture! "
                    f"Using simple FP8->BF16 conversion (may lose accuracy)."
                )
                bf16_weight = weight_fp8.to(torch.bfloat16)
            else:
                # Dequantize FP8 weights (rowwise path)
                weight_fp8_actual = weight_fp8  # Already (N, K)
                if weight_fp8_actual.dtype == torch.float8_e4m3fn:
                    if fp8_row_scale is not None:
                        # Rowwise dequantization: compute on-the-fly
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
