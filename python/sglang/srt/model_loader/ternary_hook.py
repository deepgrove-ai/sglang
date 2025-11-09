"""Runtime ternary+FP8 hook for SGLang.

This module is imported by the model loader when `--quantization ternary` is set.
It ensures linear layers have a Ternary quant method attached and proactively
quantizes weights after loading so serving can immediately use our FP8 kernels.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def _is_ternary_linear_method(obj) -> bool:
    return obj is not None and obj.__class__.__name__ == "TernaryLinearMethod"


def apply_ternary_quantization(
    model: torch.nn.Module, threshold_scale: float = 0.7, verbose: bool = False
) -> torch.nn.Module:
    """Attach ternary quant method to linear layers and quantize weights.

    Safe to call multiple times; skips layers already configured.
    """
    try:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.ternary import (
            TernaryConfig,
            TernaryLinearMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod, Fp8Config
    except Exception as e:
        logger.warning(f"Ternary quantization unavailable: {e}")
        return model

    import os
    th = float(os.environ.get("TERNARY_THRESHOLD_SCALE", str(threshold_scale)))
    max_out = int(os.environ.get("TERNARY_MAX_OUTPUT_FEATURES", "100000"))
    
    # Read I2_S mode from environment variable
    use_i2s = os.environ.get("TERNARY_USE_I2S", "0") == "1"
    
    # Read FP8 block size from environment (optional)
    fp8_block_k = int(os.environ.get("TERNARY_FP8_BLOCK_K", "128"))
    fp8_block_n = int(os.environ.get("TERNARY_FP8_BLOCK_N", "128"))
    fp8_block_size = (fp8_block_n, fp8_block_k)
    
    cfg = TernaryConfig(
        threshold_scale=th,
        max_output_features=max_out,
        fp8_block_size=fp8_block_size,
        use_i2s=use_i2s
    )
    
    if verbose or use_i2s:
        logger.info(
            f"TernaryConfig: threshold_scale={th}, max_output_features={max_out}, "
            f"fp8_block_size={fp8_block_size}, use_i2s={use_i2s}"
        )

    applied, quantized = 0, 0
    first_ternary_method = None  # Keep reference to first instance for final summary
    
    for m in model.modules():
        if isinstance(m, LinearBase):
            qm = getattr(m, "quant_method", None)
            # Always recreate quant_method to ensure correct config (especially for I2_S mode)
            # This ensures environment variables are properly applied
            m.quant_method = TernaryLinearMethod(cfg)
            if first_ternary_method is None:
                first_ternary_method = m.quant_method
            applied += 1
            # Proactively quantize weights now to avoid first-request latency
            try:
                m.quant_method.process_weights_after_loading(m)
                quantized += 1
            except Exception as e:  # best-effort; fallback will be used if needed
                logger.debug(f"Ternary quantization skipped for {m.__class__.__name__}: {e}")
        elif isinstance(m, FusedMoE):
            # Enable FP8 MoE by default unless explicitly disabled
            import os
            if os.environ.get("TERNARY_MOE_FP8", "1") != "0":
                if not hasattr(m, "quant_method") or m.quant_method is None:
                    m.quant_method = Fp8MoEMethod(Fp8Config(
                        is_checkpoint_fp8_serialized=False,
                        activation_scheme="dynamic",
                    ))
                    applied += 1
                # MoE weights are created by the MoE layer; best-effort post-processing
                try:
                    m.quant_method.process_weights_after_loading(m)
                    quantized += 1
                except Exception as e:
                    logger.debug(f"FP8 MoE post-process skipped: {e}")

    if verbose:
        logger.info(
            f"Ternary hook: attached {applied} methods; post-quantized {quantized} modules."
        )
    
    # Force garbage collection and clear CUDA cache after all quantization
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Log final memory summary after all layers are processed
    if first_ternary_method is not None:
        first_ternary_method.log_final_memory_summary()
    
    return model




