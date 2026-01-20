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
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_TERNARY_CACHE_VERSION = "v1"

# FP8 sticky quantization (BF16 -> FP8 + per-token scale) should be cheap.
# The naive PyTorch implementation (abs/amax/div/clamp/cast) launches many kernels
# and can dominate runtime. Use a fused Triton kernel when available.
_FP8_STICKY_TRITON_QUANT_AVAILABLE = False
_FP8_STICKY_SGL_KERNEL_QUANT_AVAILABLE = False
_sgl_per_token_quant_fp8 = None
try:
    # Prefer the CUDA extension quantizer when available (fastest and capture-safe
    # as long as outputs are preallocated).
    from sgl_kernel import sgl_per_token_quant_fp8 as _sgl_per_token_quant_fp8  # type: ignore

    _FP8_STICKY_SGL_KERNEL_QUANT_AVAILABLE = True
except Exception:
    _FP8_STICKY_SGL_KERNEL_QUANT_AVAILABLE = False

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore

    _FP8_STICKY_TRITON_QUANT_AVAILABLE = True

    @triton.jit
    def _bf16_to_fp8_per_token_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_ym: tl.constexpr,
        stride_yk: tl.constexpr,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)

        # Pass 1: absmax over K
        absmax = tl.zeros((), dtype=tl.float32)
        for k0 in tl.static_range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs * stride_xk,
                mask=offs < K,
                other=0.0,
            ).to(tl.float32)
            absmax = tl.maximum(absmax, tl.max(tl.abs(x), axis=0))

        # FP8 E4M3 max value
        fp8_max = 448.0
        scale = tl.maximum(absmax / fp8_max, 1e-12)
        tl.store(s_ptr + pid_m, scale)

        # Pass 2: quantize + store FP8
        for k0 in tl.static_range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(
                x_ptr + pid_m * stride_xm + offs * stride_xk,
                mask=offs < K,
                other=0.0,
            ).to(tl.float32)
            y = x / scale
            y = tl.minimum(tl.maximum(y, -fp8_max), fp8_max)
            tl.store(
                y_ptr + pid_m * stride_ym + offs * stride_yk,
                y.to(tl.float8e4nv),
                mask=offs < K,
            )
except Exception:
    _FP8_STICKY_TRITON_QUANT_AVAILABLE = False


def _bool_env(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _drop_fp16_weights() -> bool:
    v = os.environ.get("SGLANG_TERNARY_DROP_FP16_WEIGHTS")
    if v is None:
        return _bool_env("SGLANG_TERNARY_I2S_ONLY", default=False)
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _group_name(param_name: str) -> str:
    parts = param_name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return param_name


def _log_memory_report(model: torch.nn.Module) -> None:
    from collections import defaultdict

    totals_by_dtype = defaultdict(int)
    totals_by_group = defaultdict(int)
    totals_by_group_dtype = defaultdict(lambda: defaultdict(int))
    totals_by_kind = defaultdict(int)  # params vs buffers

    def _accumulate(named_tensors, kind: str) -> None:
        for name, tensor in named_tensors:
            if tensor is None:
                continue
            num_bytes = tensor.numel() * tensor.element_size()
            dtype_key = str(tensor.dtype)
            group = _group_name(name)
            totals_by_dtype[dtype_key] += num_bytes
            totals_by_group[group] += num_bytes
            totals_by_group_dtype[group][dtype_key] += num_bytes
            totals_by_kind[kind] += num_bytes

    _accumulate(model.named_parameters(recurse=True), "params")
    _accumulate(model.named_buffers(recurse=True), "buffers")

    lines = []
    lines.append(f"TERNARY MEMORY REPORT {datetime.now().isoformat()}")
    lines.append(f"Total params:  {_format_bytes(totals_by_kind['params'])}")
    lines.append(f"Total buffers: {_format_bytes(totals_by_kind['buffers'])}")
    lines.append("")
    lines.append("By dtype:")
    for dtype_key, num_bytes in sorted(totals_by_dtype.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {dtype_key}: {_format_bytes(num_bytes)}")
    lines.append("")
    lines.append("Top groups by memory (up to 40):")
    top_groups = sorted(totals_by_group.items(), key=lambda x: x[1], reverse=True)[:40]
    for group, num_bytes in top_groups:
        by_dtype = totals_by_group_dtype[group]
        dtype_parts = ", ".join(
            f"{k}={_format_bytes(v)}" for k, v in sorted(by_dtype.items(), key=lambda x: x[1], reverse=True)
        )
        lines.append(f"  {group} total={_format_bytes(num_bytes)} ({dtype_parts})")

    report_path = os.environ.get(
        "SGLANG_TERNARY_MEMORY_REPORT_FILE",
        "/root/raghav/ternary_memory_report.txt",
    )
    report_dir = os.path.dirname(report_path) or "."
    os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("[TERNARY MEM] report written to %s", report_path)


def _write_layer_type_report(model: torch.nn.Module, output_file: str) -> None:
    """Writes a breakdown of module types with counts and memory by dtype."""
    from collections import defaultdict

    type_counts = defaultdict(int)
    type_param_bytes = defaultdict(int)
    type_buffer_bytes = defaultdict(int)
    type_dtype_bytes = defaultdict(lambda: defaultdict(int))

    for _name, module in model.named_modules():
        cls_name = type(module).__name__
        type_counts[cls_name] += 1

        for _param_name, param in module.named_parameters(recurse=False):
            if param is None:
                continue
            mem_bytes = param.numel() * param.element_size()
            type_param_bytes[cls_name] += mem_bytes
            type_dtype_bytes[cls_name][str(param.dtype)] += mem_bytes

        for _buf_name, buf in module.named_buffers(recurse=False):
            if buf is None:
                continue
            mem_bytes = buf.numel() * buf.element_size()
            type_buffer_bytes[cls_name] += mem_bytes
            type_dtype_bytes[cls_name][str(buf.dtype)] += mem_bytes

    total_types = len(type_counts)
    total_modules = sum(type_counts.values())

    lines = []
    lines.append("=" * 80)
    lines.append("Ternary Layer Type Report")
    lines.append("=" * 80)
    lines.append(f"Total module instances: {total_modules}")
    lines.append(f"Unique module types:     {total_types}")
    lines.append("")
    lines.append("Top module types by memory (params + buffers):")
    lines.append("-" * 80)

    def _total_bytes(t: str) -> int:
        return type_param_bytes[t] + type_buffer_bytes[t]

    for cls_name in sorted(type_counts, key=_total_bytes, reverse=True):
        total_bytes = _total_bytes(cls_name)
        if total_bytes == 0 and type_counts[cls_name] == 0:
            continue
        lines.append(
            f"{cls_name:<36} count={type_counts[cls_name]:>6} "
            f"params={_format_bytes(type_param_bytes[cls_name]):>10} "
            f"buffers={_format_bytes(type_buffer_bytes[cls_name]):>10} "
            f"total={_format_bytes(total_bytes):>10}"
        )
        dtype_parts = ", ".join(
            f"{k}={_format_bytes(v)}"
            for k, v in sorted(type_dtype_bytes[cls_name].items(), key=lambda x: x[1], reverse=True)
        )
        if dtype_parts:
            lines.append(f"  dtypes: {dtype_parts}")

    report_dir = os.path.dirname(output_file) or "."
    os.makedirs(report_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_layer_quant_report(model: torch.nn.Module, output_file: str) -> None:
    """Writes per-layer quantization decisions and dtypes."""
    try:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    except Exception:
        LinearBase = ()
        FusedMoE = ()

    def _lm_head_enabled() -> bool:
        return os.environ.get("TERNARY_QUANTIZE_LM_HEAD", "0") == "1"

    lines = []
    lines.append("=" * 80)
    lines.append("Ternary Layer Quantization Report")
    lines.append("=" * 80)
    lines.append("prefix\tmodule\tquant_method\tweight_dtypes\treason")

    for prefix, module in model.named_modules():
        if not isinstance(module, (LinearBase, FusedMoE)):
            continue

        pref = (prefix or "").lower()
        reason = ""
        if "embed" in pref:
            reason = "skip_embed"
        elif "lm_head" in pref and not _lm_head_enabled():
            reason = "skip_lm_head"
        else:
            parts = pref.split(".")
            if any(p in ("router", "gate", "shared_expert_gate") for p in parts):
                reason = "skip_router_gate"

        qm = getattr(module, "quant_method", None)
        qm_name = type(qm).__name__ if qm is not None else "None"

        dtypes = []
        for name in ("weight", "w13_weight", "w2_weight"):
            t = getattr(module, name, None)
            if isinstance(t, torch.Tensor):
                dtypes.append(f"{name}:{str(t.dtype)}")
        for name in ("_ternary_w13_packed", "_ternary_w2_packed"):
            t = getattr(module, name, None)
            if isinstance(t, torch.Tensor):
                dtypes.append(f"{name}:{str(t.dtype)}")

        lines.append(
            f"{prefix}\t{type(module).__name__}\t{qm_name}\t{','.join(dtypes)}\t{reason}"
        )

    report_dir = os.path.dirname(output_file) or "."
    os.makedirs(report_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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

    if not _drop_fp16_weights():
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
    if _drop_fp16_weights():
        if hasattr(layer, "_ternary_weight_fp16"):
            layer._buffers.pop("_ternary_weight_fp16", None)
            delattr(layer, "_ternary_weight_fp16")
        if hasattr(layer, "_ternary_weight_fp16_cache"):
            delattr(layer, "_ternary_weight_fp16_cache")
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

        # MoE uses pack_i2s_weights from ternary.py, not BITNET_PACK_AVAILABLE
        use_v4 = (
            BITNET_CUDA_AVAILABLE
            and BITNET_LIB is not None
            and hasattr(BITNET_LIB, "ternary_moe_megafused_gemv_indexed_shared_silu")
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
        # Final combined output buffer for decode: [hidden] BF16.
        # Needed by fused ternary MoE combine kernel; must exist before CUDA graph capture.
        layer.register_buffer(
            "_ternary_moe_combined_buf",
            torch.empty(hidden_size, device=device, dtype=torch.bfloat16),
            persistent=False,
        )

        layer._ternary_moe_v4_enabled = True
        
        # Setup ctypes pointers for decode fused path
        import ctypes
        _PTR = ctypes.c_void_p
        _INT = ctypes.c_int
        layer._ctypes_w13_packed = _PTR(layer._ternary_w13_packed.data_ptr())
        layer._ctypes_w2_packed = _PTR(layer._ternary_w2_packed.data_ptr())
        layer._ctypes_alpha_w13 = _PTR(layer._ternary_moe_alpha_w13.data_ptr())
        layer._ctypes_alpha_w2 = _PTR(layer._ternary_moe_alpha_w2.data_ptr())
        layer._ctypes_intermediate_buf = _PTR(layer._ternary_moe_intermediate_buf.data_ptr())
        layer._ctypes_combined_buf = _PTR(layer._ternary_moe_combined_buf.data_ptr())
        layer._ctypes_topk_weights_bf16 = _PTR(layer._ternary_moe_topk_weights_bf16.data_ptr()) if hasattr(layer, '_ternary_moe_topk_weights_bf16') else None
        layer._ctypes_N_w13 = _INT(N_w13)
        layer._ctypes_K_w13 = _INT(hidden_size)
        layer._ctypes_N_w2 = _INT(hidden_size)
        layer._ctypes_K_w2 = _INT(intermediate_size)
        layer._ctypes_num_experts = _INT(num_experts)
        
        # Pre-allocate topk weights buffer if not exists
        if not hasattr(layer, '_ternary_moe_topk_weights_bf16'):
            layer.register_buffer(
                "_ternary_moe_topk_weights_bf16",
                torch.empty(max_top_k, device=device, dtype=torch.bfloat16),
                persistent=False,
            )
            layer._ctypes_topk_weights_bf16 = _PTR(layer._ternary_moe_topk_weights_bf16.data_ptr())
        
        from sglang.srt.layers.quantization.ternary import _KERNEL_CAPS
        # Check for full fusion: need megafused shared+silu + combine kernel
        layer._use_full_fusion = (
            _KERNEL_CAPS is not None 
            and _KERNEL_CAPS.get('moe_shared_silu', False) 
            and (_KERNEL_CAPS.get('moe_combine_parallel', False) or _KERNEL_CAPS.get('moe_combine_bf16x2', False))
        )
        logger.info(f"[TERNARY MOE] V4 enabled, full_fusion={layer._use_full_fusion}")
        
        # FP4 MoE init (disabled in cleaned up version)
        FP4_MAX = 0
        layer._fp4_moe_enabled = False
        if FP4_MAX > 0:
            try:
                from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_fp4 import unpack_ternary_to_fp4, create_fp4_blockscales_from_ternary_alpha
                from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
                from sglang.srt.layers.moe.cutlass_moe_fp4_optimized import FP4MoEBuffers
                topk = getattr(layer, 'top_k', 8)
                layer._fp4_w13 = unpack_ternary_to_fp4(layer._ternary_w13_packed, hidden_size)
                layer._fp4_w2 = unpack_ternary_to_fp4(layer._ternary_w2_packed, intermediate_size)
                layer._fp4_w13_blockscale, layer._fp4_w13_alphas = create_fp4_blockscales_from_ternary_alpha(layer._ternary_moe_alpha_w13, 2*intermediate_size, hidden_size, 16)
                layer._fp4_w2_blockscale, layer._fp4_w2_alphas = create_fp4_blockscales_from_ternary_alpha(layer._ternary_moe_alpha_w2, hidden_size, intermediate_size, 16)
                layer._fp4_a1_gscale = torch.ones(num_experts, device=device, dtype=torch.float32)
                layer._fp4_a2_gscale = torch.ones(num_experts, device=device, dtype=torch.float32)
                layer._fp4_moe_params = CutlassMoEParams(CutlassMoEType.BlockscaledFP4, device, num_experts, intermediate_size, hidden_size)
                layer._fp4_moe_buffers = FP4MoEBuffers.create(device, torch.bfloat16, FP4_MAX, topk, hidden_size, intermediate_size)
                layer._fp4_moe_enabled = True
                logger.info(f"[TERNARY MOE] FP4 enabled (max={FP4_MAX}, topk={topk})")
            except Exception as e:
                logger.warning(f"[TERNARY MOE] FP4 init failed: {e}")
        else:
            layer._fp4_moe_enabled = False
    else:
        layer._ternary_moe_v4_enabled = False
        layer._use_full_fusion = False
        layer._fp4_moe_enabled = False

    layer._ternary_moe_enabled = True
    try:
        from sglang.srt.layers.quantization.ternary import _drop_moe_fp16_weights
        _drop_moe_fp16_weights(layer)
    except Exception:
        pass
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
    
    # FP8 mode: read from SGLANG_TERNARY_USE_FP8 env var
    # When enabled, uses FP8 tensor cores and FP8 hidden state storage
    use_fp8 = _bool_env("SGLANG_TERNARY_USE_FP8", default=False)
    
    # FP8 hidden state scale granularity (default: per_token_group_128 for SM100)
    fp8_granularity = os.environ.get(
        "SGLANG_TERNARY_FP8_GRANULARITY", "per_token_group_128"
    )
    
    cfg = TernaryConfig(
        threshold_scale=th,
        storage_mode=storage_mode,
        use_fp8=use_fp8,
        fp8_hidden_scale_granularity=fp8_granularity,
    )
    
    if verbose or storage_mode == "fp16" or use_fp8:
        logger.info(
            f"TernaryConfig: threshold_scale={th}, "
            f"storage_mode={storage_mode}, "
            f"use_fp8={use_fp8}"
        )
    
    if use_fp8:
        logger.info(
            f"[TERNARY FP8] FP8-first mode enabled via SGLANG_TERNARY_USE_FP8=1\n"
            f"  - FP8 hidden state storage: {fp8_granularity}\n"
            f"  - FP8 tensor core compute for ternary GEMMs\n"
            f"  - Recommend: --kv-cache-dtype fp8_e4m3 for full FP8 pipeline"
        )
        # Log comprehensive FP8 status
        try:
            from sglang.srt.layers.quantization.ternary import log_fp8_status_summary
            log_fp8_status_summary(cfg)
        except ImportError:
            pass

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
        # Stash prefix for optional debug/validation logging.
        if prefix:
            setattr(m, "_ternary_prefix", prefix)
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
    
    # Store the ternary config on the model for later access (e.g., KV cache validation)
    model._ternary_config = cfg
    model._ternary_use_fp8 = use_fp8

    if _bool_env("SGLANG_TERNARY_MEMORY_REPORT", default=False):
        _log_memory_report(model)
    if _bool_env("SGLANG_TERNARY_LAYER_REPORT", default=False):
        report_file = os.environ.get(
            "SGLANG_TERNARY_LAYER_REPORT_FILE",
            os.path.expanduser("~/raghav/ternary_layer_report.txt"),
        )
        _write_layer_type_report(model, report_file)
    if _bool_env("SGLANG_TERNARY_LAYER_QUANT_REPORT", default=False):
        report_file = os.environ.get(
            "SGLANG_TERNARY_LAYER_QUANT_REPORT_FILE",
            os.path.expanduser("~/raghav/ternary_layer_quant_report.txt"),
        )
        _write_layer_quant_report(model, report_file)
    
    # Force garbage collection and clear CUDA cache after all quantization
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    
    return model


def get_ternary_config(model: torch.nn.Module):
    """
    Get the TernaryConfig from a model if it was applied via ternary quantization.
    
    Returns:
        TernaryConfig if ternary quantization was applied, None otherwise
    """
    return getattr(model, "_ternary_config", None)


def is_ternary_fp8_enabled(model: torch.nn.Module) -> bool:
    """
    Check if FP8-first mode is enabled for a ternary-quantized model.
    
    This can be used by ModelRunner or attention backends to validate
    that FP8 KV cache is being used when FP8-first mode is enabled.
    
    Returns:
        True if FP8-first mode is enabled, False otherwise
    """
    return getattr(model, "_ternary_use_fp8", False)


def validate_ternary_fp8_kv_cache(model: torch.nn.Module, kv_cache_dtype: torch.dtype) -> None:
    """
    Validate that FP8 KV cache is used when FP8-first ternary mode is enabled.
    
    Logs a warning if FP8-first is enabled but KV cache is not FP8.
    
    Args:
        model: The model with ternary quantization applied
        kv_cache_dtype: The actual KV cache dtype being used
    """
    if not is_ternary_fp8_enabled(model):
        return
    
    fp8_kv_dtypes = (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    # Also check for ROCm variants
    try:
        fp8_kv_dtypes = fp8_kv_dtypes + (
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        )
    except AttributeError:
        pass
    
    if kv_cache_dtype not in fp8_kv_dtypes:
        logger.warning(
            f"[TERNARY FP8] FP8-first mode is enabled but KV cache is using {kv_cache_dtype}.\n"
            f"  For optimal performance, use --kv-cache-dtype fp8_e4m3 or fp8_e5m2.\n"
            f"  This avoids FP8<->BF16 conversions at attention boundaries."
        )


# ============================================================================
# FP8 HIDDEN STATE STORAGE INFRASTRUCTURE
# ============================================================================
# These functions provide the building blocks for FP8 hidden state storage
# between transformer layers. The actual insertion of quant/dequant ops
# into the model forward pass is done separately.

class FP8HiddenStateManager:
    """
    Manager for FP8 hidden state storage between transformer layers.
    
    This class provides:
    1. Pre-allocated buffers for FP8 hidden states and scales (CUDA graph compatible)
    2. Methods to quantize/dequantize hidden states to/from FP8
    3. Configuration for scale granularity (per-token or per-token-group)
    
    Usage:
        manager = FP8HiddenStateManager(hidden_size=2048, max_tokens=4096)
        manager.allocate_buffers(device)
        
        # In model forward:
        hidden_fp8, scale = manager.quantize_hidden(hidden_bf16)
        # ... pass through layers ...
        hidden_bf16 = manager.dequantize_hidden(hidden_fp8, scale)
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_tokens: int = 4096,
        scale_granularity: str = "per_token_group_128",
        device: torch.device = None,
    ):
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.scale_granularity = scale_granularity
        self.device = device
        
        # Determine group size from granularity
        if scale_granularity == "per_token":
            self.group_size = hidden_size  # One scale per token
        elif scale_granularity == "per_token_group_128":
            self.group_size = 128
        else:
            self.group_size = 128  # Default
        
        # Pre-allocated buffers (None until allocate_buffers is called)
        self._hidden_fp8_buffer = None
        self._scale_buffer = None
        self._buffers_allocated = False
        
        # Track FP8 dtype
        self.fp8_dtype = torch.float8_e4m3fn
    
    def allocate_buffers(self, device: torch.device = None) -> None:
        """
        Pre-allocate buffers for FP8 hidden states and scales.
        
        Call this during model initialization to ensure buffers are available
        during CUDA graph capture.
        """
        if device is not None:
            self.device = device
        
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Allocate FP8 hidden state buffer
        self._hidden_fp8_buffer = torch.empty(
            self.max_tokens, self.hidden_size,
            device=self.device, dtype=self.fp8_dtype
        )
        
        # Allocate scale buffer
        num_groups = (self.hidden_size + self.group_size - 1) // self.group_size
        self._scale_buffer = torch.empty(
            self.max_tokens, num_groups,
            device=self.device, dtype=torch.float32
        )
        
        self._buffers_allocated = True
        logger.debug(
            f"[FP8 Hidden] Allocated buffers: "
            f"hidden={self._hidden_fp8_buffer.shape}, "
            f"scale={self._scale_buffer.shape}"
        )
    
    def quantize_hidden(
        self,
        hidden: torch.Tensor,
        use_preallocated: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize hidden states from BF16/FP16 to FP8.
        
        Args:
            hidden: Input tensor of shape (num_tokens, hidden_size)
            use_preallocated: If True, use pre-allocated buffers
        
        Returns:
            (hidden_fp8, scale): FP8 hidden states and scales
        """
        try:
            from sglang.srt.layers.quantization.fp8_kernel import (
                per_token_group_quant_fp8,
            )
        except ImportError:
            # Fallback to simple quantization
            return self._quantize_simple(hidden)
        
        num_tokens = hidden.shape[0]
        
        if use_preallocated and self._buffers_allocated and num_tokens <= self.max_tokens:
            # Use pre-allocated buffers
            hidden_fp8 = self._hidden_fp8_buffer[:num_tokens].view_as(hidden)
            scale = self._scale_buffer[:num_tokens]
            
            # Use sgl_kernel quant if available
            hidden_fp8_new, scale_new = per_token_group_quant_fp8(
                hidden.contiguous(), self.group_size
            )
            hidden_fp8.copy_(hidden_fp8_new)
            scale[:, :scale_new.shape[1]].copy_(scale_new)
            
            return hidden_fp8, scale[:, :scale_new.shape[1]]
        else:
            # Allocate new buffers
            return per_token_group_quant_fp8(hidden.contiguous(), self.group_size)
    
    def dequantize_hidden(
        self,
        hidden_fp8: torch.Tensor,
        scale: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize hidden states from FP8 to BF16/FP16.
        
        Args:
            hidden_fp8: FP8 hidden states of shape (num_tokens, hidden_size)
            scale: Scale factors of shape (num_tokens, num_groups)
            output_dtype: Output dtype (bfloat16 or float16)
        
        Returns:
            Dequantized hidden states in output_dtype
        """
        # Expand scale to match hidden dimensions
        num_tokens, hidden_size = hidden_fp8.shape
        num_groups = scale.shape[1]
        group_size = hidden_size // num_groups
        
        # Repeat scale for each element in group
        scale_expanded = scale.repeat_interleave(group_size, dim=1)
        if scale_expanded.shape[1] > hidden_size:
            scale_expanded = scale_expanded[:, :hidden_size]
        
        # Dequantize: hidden_bf16 = hidden_fp8 * scale
        hidden_dequant = hidden_fp8.to(torch.float32) * scale_expanded
        
        return hidden_dequant.to(output_dtype)
    
    def _quantize_simple(
        self,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple fallback quantization without sgl_kernel."""
        FP8_MAX = 448.0
        
        # Per-token quantization
        hidden_f32 = hidden.to(torch.float32)
        abs_max = hidden_f32.abs().amax(dim=1, keepdim=True)
        scale = (abs_max / FP8_MAX).clamp(min=1e-12)
        
        hidden_scaled = hidden_f32 / scale
        hidden_fp8 = hidden_scaled.clamp(-FP8_MAX, FP8_MAX).to(self.fp8_dtype)
        
        return hidden_fp8, scale


def setup_fp8_hidden_manager(
    model: torch.nn.Module,
    hidden_size: int,
    max_tokens: int = 4096,
    device: torch.device = None,
) -> None:
    """
    Setup FP8 hidden state manager on a model.
    
    This should be called after model loading to prepare for FP8 hidden state storage.
    
    Args:
        model: The model to attach the manager to
        hidden_size: Hidden dimension size
        max_tokens: Maximum number of tokens to support
        device: Device to allocate buffers on
    """
    config = get_ternary_config(model)
    if config is None or not config.use_fp8:
        return
    
    manager = FP8HiddenStateManager(
        hidden_size=hidden_size,
        max_tokens=max_tokens,
        scale_granularity=config.fp8_hidden_scale_granularity,
        device=device,
    )
    manager.allocate_buffers(device)
    
    model._fp8_hidden_manager = manager
    logger.info(
        f"[FP8 Hidden] Manager attached: hidden_size={hidden_size}, "
        f"max_tokens={max_tokens}, granularity={config.fp8_hidden_scale_granularity}"
    )


def get_fp8_hidden_manager(model: torch.nn.Module) -> FP8HiddenStateManager:
    """Get the FP8 hidden state manager from a model, if attached."""
    return getattr(model, "_fp8_hidden_manager", None)


# ============================================================================
# FP8 STICKY HIDDEN STATE HELPERS
# ============================================================================
# These functions wrap decoder layer boundaries to keep hidden states in FP8
# between layers, converting to BF16 only when needed for compute.

def _get_fp8_sticky_mode() -> str:
    """
    Sticky FP8 hidden state mode.

    Supported values:
      - "0"/"false": disabled
      - "1"/"true": force-enabled (will quantize at layer boundaries)
      - "auto": enable only when it is net-beneficial (default)
    """
    v = os.environ.get("SGLANG_TERNARY_FP8_STICKY", "auto").strip().lower()
    if v in ("0", "false", "off", "no"):
        return "0"
    if v in ("1", "true", "on", "yes"):
        return "1"
    return "auto"


def _fp8_between_layers_is_effective() -> bool:
    """
    Returns True only when keeping hidden/residual in FP8 *between decoder layers*
    avoids BF16 conversions inside the next layer.

    For FP8 sticky to be beneficial, we need BOTH:
    1. FP8 RMSNorm kernel - to consume FP8 hidden states at layer start
    2. FP8 ternary kernels - to avoid dequant at every linear layer

    Without both, FP8-between-layers adds conversion overhead without benefit.
    """
    # Check FP8 RMSNorm
    fp8_rmsnorm_ok = False
    try:
        from sglang.srt.layers.fused_add_rmsnorm_fp8 import is_fp8_rmsnorm_available
        fp8_rmsnorm_ok = is_fp8_rmsnorm_available()
    except Exception:
        pass
    
    if not fp8_rmsnorm_ok:
        return False
    
    # Check FP8 ternary kernels
    fp8_ternary_ok = False
    try:
        from sglang.srt.layers.quantization.ternary import BITNET_CUDA_FP8_MEGA_FUSED_AVAILABLE
        fp8_ternary_ok = BITNET_CUDA_FP8_MEGA_FUSED_AVAILABLE
    except Exception:
        pass
    
    # For now, FP8 sticky is only effective if we have FP8 ternary kernels
    # Without them, every linear layer does FP8→BF16→FP8 conversion
    # TODO: When FP8 ternary kernels are available, this will return True
    return fp8_ternary_ok


_FP8_STICKY_MODE = _get_fp8_sticky_mode()
_FP8_STICKY_ENABLED = _FP8_STICKY_MODE == "1" or (_FP8_STICKY_MODE == "auto" and _fp8_between_layers_is_effective())


def maybe_dequant_fp8_hidden(
    hidden_states: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize FP8 hidden states to BF16 at layer entry if needed.
    
    If hidden_states is already BF16/FP16/FP32, returns it unchanged.
    If hidden_states is FP8 with attached scale, dequantizes to output_dtype.
    
    This should be called at the START of a decoder layer forward().
    """
    if not _FP8_STICKY_ENABLED:
        return hidden_states
    
    # Check if input is FP8
    if hidden_states.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return hidden_states
    
    # Get attached scale
    fp8_scale = getattr(hidden_states, '_fp8_scale', None)
    
    if fp8_scale is not None:
        # Dequantize: hidden_bf16 = hidden_fp8 * scale
        # Scale is per-token, shape [num_tokens] or [num_tokens, 1]
        hidden_f32 = hidden_states.to(torch.float32)
        
        if fp8_scale.dim() == 1:
            scale_expanded = fp8_scale.unsqueeze(-1)  # [num_tokens, 1]
        else:
            scale_expanded = fp8_scale
        
        hidden_dequant = hidden_f32 * scale_expanded
        return hidden_dequant.to(output_dtype)
    else:
        # No scale attached, simple cast (lossy but better than crash)
        logger.warning(
            "[FP8 Sticky] FP8 hidden states without scale, falling back to simple cast"
        )
        return hidden_states.to(output_dtype)


def maybe_quant_fp8_hidden(
    hidden_states: torch.Tensor,
    force_fp8: bool = False,
    cache_owner: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Quantize BF16 hidden states to FP8 at layer exit if FP8-first mode is active.
    
    If hidden_states is already FP8, returns it unchanged.
    If FP8-first mode is disabled, returns hidden_states unchanged.
    
    This should be called at the END of a decoder layer forward().
    
    Args:
        hidden_states: Output hidden states (typically BF16)
        force_fp8: If True, always convert to FP8 (for explicit conversion points)
    
    Returns:
        FP8 hidden states with _fp8_scale attached, or original if not converting
    """
    if not _FP8_STICKY_ENABLED and not force_fp8:
        return hidden_states
    
    # Already FP8, return as-is
    if hidden_states.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return hidden_states

    # Fast path 1: sgl_kernel CUDA quant (fastest).
    if _FP8_STICKY_SGL_KERNEL_QUANT_AVAILABLE and hidden_states.dtype == torch.bfloat16:
        x2d = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()
        M, _K = x2d.shape

        if cache_owner is not None:
            # IMPORTANT: per-stream buffers.
            # Under concurrency, multiple requests can execute on different CUDA streams.
            # Reusing a single buffer on the shared module will cause races and corrupt
            # FP8 hidden states / scales, which then manifests as garbage generations.
            stream_id = int(torch.cuda.current_stream().cuda_stream)
            stream_cache = getattr(cache_owner, "_fp8_sticky_stream_cache", None)
            if stream_cache is None:
                stream_cache = {}
                setattr(cache_owner, "_fp8_sticky_stream_cache", stream_cache)
            buf = stream_cache.get(stream_id)
            if buf is None:
                buf = {}
                stream_cache[stream_id] = buf

            y = buf.get("y")
            s = buf.get("s")
            if y is None or y.shape != x2d.shape or y.device != x2d.device:
                y = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
                buf["y"] = y
            # sgl_kernel expects [M, 1]
            if s is None or tuple(s.shape) != (M, 1) or s.device != x2d.device:
                s = torch.empty((M, 1), device=x2d.device, dtype=torch.float32)
                buf["s"] = s
            if not getattr(cache_owner, "_fp8_sticky_quant_logged", False):
                cache_owner._fp8_sticky_quant_logged = True
                logger.info("[FP8 Sticky] Using sgl_kernel BF16->FP8 per-token quant")
        else:
            y = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
            s = torch.empty((M, 1), device=x2d.device, dtype=torch.float32)

        # Launch CUDA kernel
        _sgl_per_token_quant_fp8(x2d, y, s)

        out = y.view_as(hidden_states)
        out._fp8_scale = s.view(-1)
        return out

    # Fast path 2: fused Triton kernel (compile once at import).
    if _FP8_STICKY_TRITON_QUANT_AVAILABLE and hidden_states.dtype == torch.bfloat16:
        x2d = hidden_states.view(-1, hidden_states.shape[-1]).contiguous()
        M, K = x2d.shape

        if cache_owner is not None:
            # IMPORTANT: per-stream buffers (see sgl_kernel path above).
            stream_id = int(torch.cuda.current_stream().cuda_stream)
            stream_cache = getattr(cache_owner, "_fp8_sticky_stream_cache", None)
            if stream_cache is None:
                stream_cache = {}
                setattr(cache_owner, "_fp8_sticky_stream_cache", stream_cache)
            buf = stream_cache.get(stream_id)
            if buf is None:
                buf = {}
                stream_cache[stream_id] = buf

            y = buf.get("y")
            s = buf.get("s_triton")
            if y is None or y.shape != x2d.shape or y.device != x2d.device:
                y = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
                buf["y"] = y
            if s is None or s.numel() != M or s.device != x2d.device:
                s = torch.empty((M,), device=x2d.device, dtype=torch.float32)
                buf["s_triton"] = s
            if not getattr(cache_owner, "_fp8_sticky_quant_logged", False):
                cache_owner._fp8_sticky_quant_logged = True
                logger.info("[FP8 Sticky] Using Triton fused BF16->FP8 quant kernel")
        else:
            y = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
            s = torch.empty((M,), device=x2d.device, dtype=torch.float32)

        # Launch: use BLOCK_K=1024 so H=2048 takes 2 iterations.
        grid = (M,)
        _bf16_to_fp8_per_token_kernel[grid](
            x2d, y, s,
            x2d.stride(0), x2d.stride(1),
            y.stride(0), y.stride(1),
            K=K,
            BLOCK_K=1024,
            num_warps=8,
        )

        out = y.view_as(hidden_states)
        out._fp8_scale = s
        return out

    # Fallback: naive PyTorch implementation (slow; should be avoided for performance).
    if cache_owner is not None and not getattr(cache_owner, "_fp8_sticky_quant_fallback_logged", False):
        cache_owner._fp8_sticky_quant_fallback_logged = True
        logger.warning(
            f"[FP8 Sticky] Falling back to PyTorch BF16->FP8 quant (slow). "
            f"triton_ok={_FP8_STICKY_TRITON_QUANT_AVAILABLE} x_dtype={hidden_states.dtype}"
        )

    FP8_MAX = 448.0  # E4M3 max value
    hidden_f32 = hidden_states.to(torch.float32)
    abs_max = hidden_f32.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / FP8_MAX).clamp(min=1e-12).squeeze(-1)  # [num_tokens]
    hidden_scaled = hidden_f32 / scale.unsqueeze(-1)
    hidden_fp8 = hidden_scaled.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    hidden_fp8._fp8_scale = scale
    return hidden_fp8


def wrap_decoder_layer_forward_fp8(
    original_forward,
    use_fp8_first: bool = False,
    is_last_layer: bool = False,
    cache_owner: Optional[torch.nn.Module] = None,
):
    """
    Wrap a decoder layer's forward function to handle FP8 sticky hidden states.
    
    FP8 RMSNorm-aware wrapper:
    - hidden_states can flow through as FP8 (RMSNorm handles FP8 input natively)
    - residual must stay BF16 for numerical stability (dequant if FP8)
    - At exit: quantize hidden_states to FP8 (but keep residual BF16)
    - Last layer: output BF16 for final norm compatibility
    
    Usage:
        layer.forward = wrap_decoder_layer_forward_fp8(layer.forward, use_fp8_first=True)
    """
    from functools import wraps
    
    # Check once at wrap time if FP8 RMSNorm is available
    fp8_rmsnorm_available = _fp8_between_layers_is_effective()
    
    @wraps(original_forward)
    def wrapped_forward(
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Entry handling:
        # - If FP8 RMSNorm is available: let hidden_states flow as FP8 (RMSNorm handles it)
        # - Otherwise: dequant hidden_states to BF16 for compatibility
        # - Always ensure residual is BF16 for numerical stability
        
        if not fp8_rmsnorm_available:
            # No FP8 RMSNorm: dequant hidden_states to BF16
            hidden_states = maybe_dequant_fp8_hidden(hidden_states)
        # else: hidden_states can stay FP8, RMSNorm will handle it
        
        # Residual should always be BF16 for numerical stability
        if residual is not None and residual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            residual = maybe_dequant_fp8_hidden(residual)
        
        # Run original forward
        hidden_out, residual_out = original_forward(
            positions, hidden_states, forward_batch, residual
        )
        
        # Exit handling:
        # - Quantize hidden_states to FP8 (except for last layer)
        # - Keep residual as BF16 (don't quantize)
        if use_fp8_first and _FP8_STICKY_ENABLED and not is_last_layer:
            hidden_out = maybe_quant_fp8_hidden(hidden_out, cache_owner=cache_owner)
            # NOTE: residual stays BF16 for numerical stability
            # The FP8 RMSNorm kernel expects: FP8 hidden + BF16 residual
        
        return hidden_out, residual_out
    
    return wrapped_forward


def setup_fp8_sticky_layers(model: torch.nn.Module) -> None:
    """
    Setup FP8 sticky hidden state handling for all decoder layers.
    
    This wraps each decoder layer's forward function to handle FP8 <-> BF16
    conversion at layer boundaries.
    
    With FP8 RMSNorm:
    - Hidden states flow as FP8 between layers (no dequant at entry)
    - RMSNorm directly consumes FP8 hidden + BF16 residual
    - Last layer outputs BF16 for final norm compatibility
    
    Without FP8 RMSNorm:
    - FP8 sticky is disabled (would be pure overhead)
    
    Call this after model loading and ternary quantization is applied.
    """
    use_fp8_first = is_ternary_fp8_enabled(model)
    
    if not use_fp8_first:
        logger.debug("[FP8 Sticky] FP8-first mode not enabled, skipping layer wrapping")
        return
    
    # Re-check effectiveness (may have changed after model load)
    fp8_rmsnorm_available = _fp8_between_layers_is_effective()
    
    if not _FP8_STICKY_ENABLED:
        reason = "mode=disabled" if _FP8_STICKY_MODE == "0" else f"FP8 RMSNorm available={fp8_rmsnorm_available}"
        logger.info(
            f"[FP8 Sticky] FP8 sticky between layers is disabled ({reason}). "
            f"Using BF16 hidden states between layers."
        )
        return
    
    # First, collect all decoder layers to identify the last one
    decoder_layers = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "DecoderLayer" in class_name:
            if not hasattr(module, '_fp8_sticky_wrapped'):
                decoder_layers.append((name, module))
    
    if not decoder_layers:
        logger.debug("[FP8 Sticky] No decoder layers found")
        return
    
    num_layers = len(decoder_layers)
    wrapped_count = 0
    
    for idx, (name, module) in enumerate(decoder_layers):
        is_last_layer = (idx == num_layers - 1)
        
        # Wrap forward
        original_forward = module.forward
        module.forward = wrap_decoder_layer_forward_fp8(
            original_forward,
            use_fp8_first=True,
            is_last_layer=is_last_layer,
            cache_owner=module,
        )
        module._fp8_sticky_wrapped = True
        wrapped_count += 1

    # Warm up Triton FP8-sticky quant kernel BEFORE CUDA graph capture starts.
    # This avoids first-use JIT compilation during capture (which can be slow or unsafe).
    if wrapped_count > 0 and _FP8_STICKY_TRITON_QUANT_AVAILABLE:
        try:
            first_module = decoder_layers[0][1]
            hidden = getattr(first_module, "config", None)
            hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
            # Best-effort hidden size detection.
            if hidden_size is None:
                hidden_size = getattr(getattr(first_module, "self_attn", None), "hidden_size", None)
            if hidden_size is None:
                # Fallback: infer from RMSNorm weight
                hidden_size = int(first_module.input_layernorm.weight.numel())
            dummy = torch.zeros((1, int(hidden_size)), device=next(model.parameters()).device, dtype=torch.bfloat16)
            maybe_quant_fp8_hidden(dummy, force_fp8=True, cache_owner=first_module)
        except Exception:
            pass
    
    if wrapped_count > 0:
        logger.info(
            f"[FP8 Sticky] Wrapped {wrapped_count} decoder layers for FP8 sticky hidden states\n"
            f"  - FP8 RMSNorm: {'enabled' if fp8_rmsnorm_available else 'disabled (fallback to BF16)'}\n"
            f"  - Hidden states: FP8 between layers (except last → BF16)\n"
            f"  - Residual: BF16 throughout (numerical stability)"
        )
