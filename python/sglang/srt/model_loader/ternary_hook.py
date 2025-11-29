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
            TernaryFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod, Fp8Config
    except Exception as e:
        logger.warning(f"Ternary quantization unavailable: {e}")
        return model

    import os
    th = float(os.environ.get("TERNARY_THRESHOLD_SCALE", str(threshold_scale)))
    
    # Default to i2s storage mode (can override with TERNARY_STORAGE_MODE env var if needed)
    storage_mode = os.environ.get("TERNARY_STORAGE_MODE", "i2s")
    
    
    cfg = TernaryConfig(
        threshold_scale=th,
        storage_mode=storage_mode
    )
    
    if verbose or storage_mode == "fp16":
        logger.info(
            f"TernaryConfig: threshold_scale={th}, "
            f"storage_mode={storage_mode}"
        )

    applied, quantized = 0, 0
    first_ternary_method = None  # Keep reference to first instance for final summary
    
    # IMPORTANT:
    # - If the model was constructed with quant_config=TernaryConfig (the usual
    #   `--quantization ternary` path), then LinearBase already attached a
    #   TernaryLinearMethod and `load_weights_and_postprocess` has *already*
    #   run `process_weights_after_loading` once. In that case we MUST NOT
    #   quantize again, or we would be quantizing already-quantized weights.
    # - If the model was built without quant_config, this hook will attach
    #   TernaryLinearMethod and perform quantization exactly once.
    for prefix, m in model.named_modules():
        if isinstance(m, LinearBase):
            existing_qm = getattr(m, "quant_method", None)

            if _is_ternary_linear_method(existing_qm):
                # Already using ternary; just track stats and reuse
                if first_ternary_method is None:
                    first_ternary_method = existing_qm
                applied += 1
                # Do NOT call process_weights_after_loading again.
                continue

            # Model was not built with a ternary quant_config; decide if this
            # layer should be quantized according to TernaryConfig rules.
            qm = cfg.get_quant_method(m, prefix=prefix)
            if qm is None:
                continue

            m.quant_method = qm
            if first_ternary_method is None and _is_ternary_linear_method(qm):
                first_ternary_method = qm
            applied += 1

            # First-time quantization for this layer
            try:
                qm.process_weights_after_loading(m)
                quantized += 1
            except Exception as e:  # best-effort; fallback will be used if needed
                logger.debug(f"Ternary quantization skipped for {m.__class__.__name__}: {e}")
        elif isinstance(m, FusedMoE):
            # Enable Ternary MoE quantization (uses V4 kernel)
            import os
            use_ternary_moe = os.environ.get("TERNARY_MOE", "1") == "1"
            
            existing_qm = getattr(m, "quant_method", None)
            is_ternary_moe = existing_qm is not None and existing_qm.__class__.__name__ == "TernaryFusedMoEMethod"
            
            if is_ternary_moe:
                # Already using TernaryFusedMoEMethod, just run post-processing
                try:
                    existing_qm.process_weights_after_loading(m)
                    quantized += 1
                    logger.info(f"[TERNARY MOE] Applied ternary quantization to {prefix}")
                except Exception as e:
                    logger.warning(f"[TERNARY MOE] Post-process failed for {prefix}: {e}")
                applied += 1
            elif use_ternary_moe:
                # Apply TernaryFusedMoEMethod
                if not hasattr(m, "quant_method") or m.quant_method is None or \
                   m.quant_method.__class__.__name__ == "UnquantizedFusedMoEMethod":
                    qm = TernaryFusedMoEMethod(cfg)
                    m.quant_method = qm
                    applied += 1
                    try:
                        qm.process_weights_after_loading(m)
                        quantized += 1
                        logger.info(f"[TERNARY MOE] Applied ternary quantization to {prefix}")
                    except Exception as e:
                        logger.warning(f"[TERNARY MOE] Quantization failed for {prefix}: {e}")
            else:
                # Fallback to FP8 MoE if TERNARY_MOE=0
                if not hasattr(m, "quant_method") or m.quant_method is None:
                    m.quant_method = Fp8MoEMethod(Fp8Config(
                        is_checkpoint_fp8_serialized=False,
                        activation_scheme="dynamic",
                    ))
                    applied += 1
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
    
    
    return model




