"""Runtime ternary+FP8 hook for SGLang.

This module is imported by the model loader when `--quantization ternary` is set.
It ensures linear layers have a Ternary quant method attached and proactively
quantizes weights after loading so serving can immediately use our FP8 kernels.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)

_TERNARY_CACHE_VERSION = "v1"


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _safe_fs_name(name: str) -> str:
    # Keep it stable + filesystem friendly.
    name = name.replace(".", "__")
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name)


def _try_get_tp_info() -> Tuple[int, int]:
    # Best-effort; works even when distributed isn't initialized.
    try:
        from sglang.srt.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return int(get_tensor_model_parallel_rank()), int(
            get_tensor_model_parallel_world_size()
        )
    except Exception:
        return 0, 1


def _get_cache_root(
    *,
    cache_dir: str | None,
    model_id: str | None,
    revision: str | None,
    threshold_scale: float,
    storage_mode: str,
) -> Path | None:
    if not _bool_env("SGLANG_TERNARY_CACHE", default=False):
        return None

    if not model_id:
        # No stable identifier -> skip caching.
        return None

    base = cache_dir or os.environ.get("SGLANG_TERNARY_CACHE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "sglang", "ternary")

    tp_rank, tp_size = _try_get_tp_info()
    payload = {
        "cache_version": _TERNARY_CACHE_VERSION,
        "model_id": model_id,
        "revision": revision or "",
        "threshold_scale": float(threshold_scale),
        "storage_mode": str(storage_mode),
        "tp_rank": int(tp_rank),
        "tp_size": int(tp_size),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return Path(base) / _TERNARY_CACHE_VERSION / digest / f"tp{tp_rank}-of-{tp_size}"


def _atomic_save_safetensors(path: Path, tensors: Dict[str, torch.Tensor], metadata: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except Exception as e:
        logger.debug(f"Ternary cache write skipped (safetensors unavailable): {e}")
        return

    # Atomic write to avoid partial/corrupt cache files on crash.
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    os.close(tmp_fd)
    try:
        save_file(tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, path)
    finally:
        with contextlib.suppress(Exception):
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _load_safetensors_with_meta(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]] | None:
    try:
        from safetensors import safe_open
    except Exception as e:
        logger.debug(f"Ternary cache load skipped (safetensors unavailable): {e}")
        return None

    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        for k in f.keys():  # noqa: SIM118
            tensors[k] = f.get_tensor(k)
    return tensors, meta


def _float_meta_eq(a: str | None, b: float, *, tol: float = 1e-8) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _try_load_linear_cache(
    *,
    layer: torch.nn.Module,
    prefix: str,
    cache_root: Path | None,
    threshold_scale: float,
    storage_mode: str,
    use_bitnet_kernel: bool,
) -> bool:
    if cache_root is None:
        return False
    if storage_mode != "i2s":
        return False

    cache_path = cache_root / "linear" / f"{_safe_fs_name(prefix)}.safetensors"
    if not cache_path.exists():
        return False

    loaded = _load_safetensors_with_meta(cache_path)
    if loaded is None:
        return False
    tensors, meta = loaded

    if meta.get("cache_version") != _TERNARY_CACHE_VERSION:
        return False
    if meta.get("kind") != "linear":
        return False
    if meta.get("storage_mode") != storage_mode:
        return False
    if not _float_meta_eq(meta.get("threshold_scale"), threshold_scale):
        return False

    # Validate shapes against the currently loaded weight.
    weight_orig = layer.weight.data
    N, K = weight_orig.shape
    if meta.get("N") and int(meta["N"]) != int(N):
        return False
    if meta.get("K") and int(meta["K"]) != int(K):
        return False

    weight_packed = tensors.get("weight_packed")
    alpha = tensors.get("alpha")
    if weight_packed is None or alpha is None:
        return False

    try:
        from sglang.srt.layers.quantization.ternary import (
            DEFAULT_PREFILL_SKIP_M,
            quantize_alpha_int8,
            replace_parameter,
        )
    except Exception:
        return False

    device = weight_orig.device
    original_dtype = weight_orig.dtype

    # Preserve original FP16/BF16 weight for fallback path (matches runtime quant behavior).
    weight_fp16 = weight_orig.to(
        torch.bfloat16 if original_dtype == torch.bfloat16 else torch.float16
    )
    layer.register_buffer("_ternary_weight_fp16", weight_fp16, persistent=False)
    setattr(layer, "_ternary_weight_fp16_cache", {})

    # Install packed weight + alpha
    replace_parameter(layer, "weight", weight_packed.contiguous().to(device, non_blocking=True))
    layer.register_buffer(
        "ternary_alpha",
        alpha.contiguous().to(device, dtype=torch.float32, non_blocking=True),
        persistent=False,
    )

    # Optional BitNet / V4 kernel buffers (safe to load even if kernel not available).
    bitnet_weight = tensors.get("weight_bitnet")
    if bitnet_weight is not None:
        layer.register_buffer(
            "ternary_weight_bitnet",
            bitnet_weight.contiguous().to(device, non_blocking=True),
            persistent=False,
        )
        layer._ternary_weight_bitnet_ptr = layer.ternary_weight_bitnet.data_ptr()

    alpha_q = tensors.get("alpha_q")
    alpha_scale = tensors.get("alpha_scale")
    if bitnet_weight is not None and use_bitnet_kernel and device.type == "cuda":
        if alpha_q is None or alpha_scale is None:
            # Compute once if not in cache.
            alpha_q_local, alpha_scale_local = quantize_alpha_int8(layer.ternary_alpha)
            alpha_q = alpha_q_local.detach().cpu()
            alpha_scale = torch.tensor([alpha_scale_local], dtype=torch.float32)

        layer.register_buffer(
            "ternary_alpha_q",
            alpha_q.contiguous().to(device, non_blocking=True),
            persistent=False,
        )
        layer.register_buffer(
            "ternary_alpha_scale",
            alpha_scale.contiguous().to(device, non_blocking=True, dtype=torch.float32),
            persistent=False,
        )
        layer._ternary_alpha_q_ptr = layer.ternary_alpha_q.data_ptr()
        layer._ternary_alpha_scale_ptr = layer.ternary_alpha_scale.data_ptr()

        # Match runtime behavior: preallocate common decode buffers.
        common_batch_sizes = [1, 2, 4, 8, 16, 32]
        for M_prealloc in common_batch_sizes:
            if M_prealloc <= DEFAULT_PREFILL_SKIP_M:
                setattr(
                    layer,
                    f"_ternary_act_int8_M{M_prealloc}",
                    torch.empty(M_prealloc, K, device=device, dtype=torch.int8),
                )
                setattr(
                    layer,
                    f"_ternary_act_scale_M{M_prealloc}",
                    torch.empty(M_prealloc, device=device, dtype=torch.bfloat16),
                )
                setattr(
                    layer,
                    f"_ternary_output_M{M_prealloc}",
                    torch.empty(M_prealloc, N, device=device, dtype=torch.bfloat16),
                )

    layer._ternary_original_dtype = original_dtype
    layer._ternary_i2s_enabled = True
    layer._ternary_fp16_enabled = False
    layer._ternary_bitnet_enabled = bitnet_weight is not None
    layer._ternary_weight_shape = (N, K)
    layer._ternary_K = K
    layer._ternary_N = N
    return True


def _try_save_linear_cache(
    *,
    layer: torch.nn.Module,
    prefix: str,
    cache_root: Path | None,
    threshold_scale: float,
    storage_mode: str,
) -> None:
    if cache_root is None:
        return
    if storage_mode != "i2s":
        return
    if not _bool_env("SGLANG_TERNARY_CACHE_WRITE", default=True):
        return

    # Only cache if quantization actually happened.
    if not getattr(layer, "_ternary_i2s_enabled", False):
        return
    if not hasattr(layer, "ternary_alpha"):
        return

    try:
        weight_packed = layer.weight.detach()
        alpha = layer.ternary_alpha.detach()
    except Exception:
        return

    N, K = getattr(layer, "_ternary_weight_shape", (None, None))

    tensors: Dict[str, torch.Tensor] = {
        "weight_packed": weight_packed.contiguous().cpu(),
        "alpha": alpha.contiguous().cpu(),
    }
    if hasattr(layer, "ternary_weight_bitnet"):
        tensors["weight_bitnet"] = (
            layer.ternary_weight_bitnet.detach().contiguous().cpu()
        )
    if hasattr(layer, "ternary_alpha_q"):
        tensors["alpha_q"] = layer.ternary_alpha_q.detach().contiguous().cpu()
    if hasattr(layer, "ternary_alpha_scale"):
        tensors["alpha_scale"] = layer.ternary_alpha_scale.detach().contiguous().cpu()

    meta: Dict[str, str] = {
        "cache_version": _TERNARY_CACHE_VERSION,
        "kind": "linear",
        "threshold_scale": repr(float(threshold_scale)),
        "storage_mode": str(storage_mode),
        "N": "" if N is None else str(int(N)),
        "K": "" if K is None else str(int(K)),
    }

    cache_path = cache_root / "linear" / f"{_safe_fs_name(prefix)}.safetensors"
    _atomic_save_safetensors(cache_path, tensors, meta)


def _try_load_moe_cache(
    *,
    layer: torch.nn.Module,
    prefix: str,
    cache_root: Path | None,
    threshold_scale: float,
    storage_mode: str,
) -> bool:
    if cache_root is None:
        return False
    if storage_mode != "i2s":
        return False

    cache_path = cache_root / "moe" / f"{_safe_fs_name(prefix)}.safetensors"
    if not cache_path.exists():
        return False

    loaded = _load_safetensors_with_meta(cache_path)
    if loaded is None:
        return False
    tensors, meta = loaded

    if meta.get("cache_version") != _TERNARY_CACHE_VERSION:
        return False
    if meta.get("kind") != "moe":
        return False
    if meta.get("storage_mode") != storage_mode:
        return False
    if not _float_meta_eq(meta.get("threshold_scale"), threshold_scale):
        return False

    # Validate shapes against live weights.
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        return False
    num_experts = int(layer.w13_weight.shape[0])
    hidden_size = int(layer.w13_weight.shape[2])
    intermediate_size = int(layer.w13_weight.shape[1] // 2)

    if meta.get("num_experts") and int(meta["num_experts"]) != num_experts:
        return False
    if meta.get("hidden_size") and int(meta["hidden_size"]) != hidden_size:
        return False
    if meta.get("intermediate_size") and int(meta["intermediate_size"]) != intermediate_size:
        return False

    w13_packed = tensors.get("w13_packed")
    w2_packed = tensors.get("w2_packed")
    alpha_w13 = tensors.get("alpha_w13")
    alpha_w2 = tensors.get("alpha_w2")
    if w13_packed is None or w2_packed is None or alpha_w13 is None or alpha_w2 is None:
        return False

    device = layer.w13_weight.device

    # Install decode-only artifacts for the V4 indexed MoE kernel.
    layer.register_buffer(
        "_ternary_w13_packed",
        w13_packed.contiguous().to(device, non_blocking=True),
        persistent=False,
    )
    layer.register_buffer(
        "_ternary_w2_packed",
        w2_packed.contiguous().to(device, non_blocking=True),
        persistent=False,
    )
    layer._ternary_w13_packed_ptr = layer._ternary_w13_packed.data_ptr()
    layer._ternary_w2_packed_ptr = layer._ternary_w2_packed.data_ptr()

    layer.register_buffer(
        "_ternary_moe_alpha_w13",
        alpha_w13.contiguous().to(device, dtype=torch.float32, non_blocking=True),
        persistent=False,
    )
    layer.register_buffer(
        "_ternary_moe_alpha_w2",
        alpha_w2.contiguous().to(device, dtype=torch.float32, non_blocking=True),
        persistent=False,
    )

    # Dimensions (used by forward)
    layer._ternary_moe_num_experts = num_experts
    layer._ternary_moe_hidden_size = hidden_size
    layer._ternary_moe_intermediate_size = intermediate_size

    # Enable V4 kernel only if the shared library is present.
    try:
        from sglang.srt.layers.quantization.ternary import (
            BITNET_CUDA_AVAILABLE,
            BITNET_LIB,
            BITNET_PACK_AVAILABLE,
        )

        use_v4 = (
            BITNET_PACK_AVAILABLE
            and BITNET_CUDA_AVAILABLE
            and BITNET_LIB is not None
            and hasattr(BITNET_LIB, "ternary_moe_gemv_indexed_batched")
        )
    except Exception:
        use_v4 = False

    if use_v4:
        max_top_k = 8
        N_w13 = 2 * intermediate_size
        N_w2 = hidden_size
        K_w13 = hidden_size
        K_w2 = intermediate_size

        layer.register_buffer(
            "_ternary_moe_gate_up_buf",
            torch.empty(max_top_k, N_w13, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_down_buf",
            torch.empty(max_top_k, N_w2, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_x_int8",
            torch.empty(max_top_k, K_w13, device=device, dtype=torch.int8),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_inter_int8",
            torch.empty(max_top_k, K_w2, device=device, dtype=torch.int8),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_x_scale",
            torch.empty(max_top_k, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_inter_scale",
            torch.empty(max_top_k, device=device, dtype=torch.bfloat16),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_topk_ids_buf",
            torch.empty(max_top_k, device=device, dtype=torch.int32),
            persistent=False,
        )
        layer.register_buffer(
            "_ternary_moe_intermediate_buf",
            torch.empty(max_top_k, intermediate_size, device=device, dtype=torch.bfloat16),
            persistent=False,
        )

        layer._ternary_moe_v4_enabled = True
    else:
        layer._ternary_moe_v4_enabled = False

    layer._ternary_moe_enabled = True
    return True


def _try_save_moe_cache(
    *,
    layer: torch.nn.Module,
    prefix: str,
    cache_root: Path | None,
    threshold_scale: float,
    storage_mode: str,
) -> None:
    if cache_root is None:
        return
    if storage_mode != "i2s":
        return
    if not _bool_env("SGLANG_TERNARY_CACHE_WRITE", default=True):
        return

    # Only cache if MoE decode artifacts exist (decode-only mode).
    if not getattr(layer, "_ternary_moe_enabled", False):
        return
    if not getattr(layer, "_ternary_moe_v4_enabled", False):
        return
    required = (
        hasattr(layer, "_ternary_w13_packed")
        and hasattr(layer, "_ternary_w2_packed")
        and hasattr(layer, "_ternary_moe_alpha_w13")
        and hasattr(layer, "_ternary_moe_alpha_w2")
    )
    if not required:
        return

    num_experts = int(getattr(layer, "_ternary_moe_num_experts", layer.w13_weight.shape[0]))
    hidden_size = int(getattr(layer, "_ternary_moe_hidden_size", layer.w13_weight.shape[2]))
    intermediate_size = int(getattr(layer, "_ternary_moe_intermediate_size", layer.w13_weight.shape[1] // 2))

    tensors: Dict[str, torch.Tensor] = {
        "w13_packed": layer._ternary_w13_packed.detach().contiguous().cpu(),
        "w2_packed": layer._ternary_w2_packed.detach().contiguous().cpu(),
        "alpha_w13": layer._ternary_moe_alpha_w13.detach().contiguous().cpu(),
        "alpha_w2": layer._ternary_moe_alpha_w2.detach().contiguous().cpu(),
    }
    meta: Dict[str, str] = {
        "cache_version": _TERNARY_CACHE_VERSION,
        "kind": "moe",
        "threshold_scale": repr(float(threshold_scale)),
        "storage_mode": str(storage_mode),
        "num_experts": str(num_experts),
        "hidden_size": str(hidden_size),
        "intermediate_size": str(intermediate_size),
    }

    cache_path = cache_root / "moe" / f"{_safe_fs_name(prefix)}.safetensors"
    _atomic_save_safetensors(cache_path, tensors, meta)


def _is_ternary_linear_method(obj) -> bool:
    return obj is not None and obj.__class__.__name__ == "TernaryLinearMethod"


def apply_ternary_quantization(
    model: torch.nn.Module,
    threshold_scale: float = 0.7,
    verbose: bool = False,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> torch.nn.Module:
    """Attach ternary quant method to linear layers and quantize weights.

    Safe to call multiple times; skips layers already configured.
    """
    try:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.ternary import (
            TernaryConfig,
            TernaryFusedMoEMethod,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod, Fp8Config
    except Exception as e:
        logger.warning(f"Ternary quantization unavailable: {e}")
        return model

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

    cache_root = _get_cache_root(
        cache_dir=cache_dir,
        model_id=model_id,
        revision=revision,
        threshold_scale=th,
        storage_mode=storage_mode,
    )
    if cache_root is not None and verbose:
        logger.info(f"Ternary cache enabled at: {str(cache_root)}")

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

            # First-time quantization for this layer (optionally from cache).
            loaded_from_cache = False
            if _is_ternary_linear_method(qm):
                try:
                    loaded_from_cache = _try_load_linear_cache(
                        layer=m,
                        prefix=prefix,
                        cache_root=cache_root,
                        threshold_scale=th,
                        storage_mode=storage_mode,
                        use_bitnet_kernel=getattr(cfg, "use_bitnet_kernel", True),
                    )
                except Exception as e:
                    logger.debug(f"Ternary cache load failed for {prefix}: {e}")
                    loaded_from_cache = False

            if loaded_from_cache:
                quantized += 1
            else:
                try:
                    qm.process_weights_after_loading(m)
                    quantized += 1
                    _try_save_linear_cache(
                        layer=m,
                        prefix=prefix,
                        cache_root=cache_root,
                        threshold_scale=th,
                        storage_mode=storage_mode,
                    )
                except Exception as e:  # best-effort; fallback will be used if needed
                    logger.debug(
                        f"Ternary quantization skipped for {m.__class__.__name__}: {e}"
                    )
        elif isinstance(m, FusedMoE):
            # Enable Ternary MoE quantization (uses V4 kernel)
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
                        moe_decode_only = _bool_env("TERNARY_MOE_DECODE_ONLY", default=False)
                        moe_loaded_from_cache = False
                        if moe_decode_only:
                            try:
                                moe_loaded_from_cache = _try_load_moe_cache(
                                    layer=m,
                                    prefix=prefix,
                                    cache_root=cache_root,
                                    threshold_scale=th,
                                    storage_mode=storage_mode,
                                )
                            except Exception as e:
                                logger.debug(f"[TERNARY MOE] Cache load failed for {prefix}: {e}")
                                moe_loaded_from_cache = False

                        if moe_loaded_from_cache:
                            quantized += 1
                            logger.info(f"[TERNARY MOE] Loaded ternary MoE cache for {prefix}")
                        else:
                            qm.process_weights_after_loading(m)
                            quantized += 1
                            logger.info(f"[TERNARY MOE] Applied ternary quantization to {prefix}")
                            if moe_decode_only:
                                _try_save_moe_cache(
                                    layer=m,
                                    prefix=prefix,
                                    cache_root=cache_root,
                                    threshold_scale=th,
                                    storage_mode=storage_mode,
                                )
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




