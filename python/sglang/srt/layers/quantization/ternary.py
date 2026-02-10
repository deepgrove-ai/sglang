"""Ternary quantization method for SGLang.

Implements ternary quantization (weights in {-1, 0, 1} × alpha).

Features:
- 8× memory savings with 2-bit weight storage (i2s format)
- Per-column alpha scaling for accuracy
- Optimized CUDA kernels for decode and prefill
"""

import atexit
import ctypes
import json
import logging
import os
import signal
from collections import Counter
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

DEFAULT_PREFILL_SKIP_M = int(os.environ.get("TERNARY_PREFILL_SKIP_M", "1"))

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
        os.path.join(os.path.dirname(__file__), '../../../../../third_party/ternarykernels/mangrove-turbo/libternary_bitnet.so'),
        os.path.join(os.path.dirname(__file__), '../../../../../ternarykernels/mangrove-turbo/libternary_bitnet.so'),
        os.path.join(os.path.dirname(__file__), '../../../../../../ternarykernels/mangrove-turbo/libternary_bitnet.so'),
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
        os.path.join(os.path.dirname(__file__), '../../../../../third_party/ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so'),
        os.path.join(os.path.dirname(__file__), '../../../../../ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so'),
        os.path.join(os.path.dirname(__file__), '../../../../../../ternarykernels/mangrove-turbo/libternary_cutlass_sm100.so'),
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
    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs_streamk"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs_streamk
        fn.argtypes = [_PTR, _PTR, _PTR, _INT, _INT, _INT, _INT]
        fn.restype = _SIZE_T
    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size
        fn.argtypes = [_INT, _INT, _INT]
        fn.restype = _SIZE_T
    if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size_streamk"):
        fn = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_streamk
        fn.argtypes = [_INT, _INT, _INT, _INT]
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
    # Fallback to ladder packer from ternarykernels when BitNet/gpu packer
    # is not available in this monorepo setup.
    try:
        import sys

        ternary_roots = [
            os.path.join(os.path.dirname(__file__), '../../../../../third_party/ternarykernels/mangrove-turbo'),
            os.path.join(os.path.dirname(__file__), '../../../../../ternarykernels/mangrove-turbo'),
            os.path.join(os.path.dirname(__file__), '../../../../../../ternarykernels/mangrove-turbo'),
        ]
        env_ternary_root = os.environ.get("TERNARY_ROOT", "").strip()
        if env_ternary_root:
            ternary_roots.append(os.path.join(env_ternary_root, "mangrove-turbo"))

        for root in ternary_roots:
            src_dir = os.path.normpath(os.path.join(root, "src"))
            if os.path.isdir(src_dir) and src_dir not in sys.path:
                sys.path.append(src_dir)

        from ladder_pack_gpu import pack_ladder_gpu as _ladder_pack_gpu  # type: ignore

        def _ladder_pack_as_bitnet(weight_int8: torch.Tensor) -> torch.Tensor:
            if weight_int8.dtype != torch.int8:
                weight_int8 = weight_int8.to(torch.int8)
            if not weight_int8.is_cuda:
                raise RuntimeError("ladder_pack_gpu requires CUDA tensor input")
            N, K = weight_int8.shape
            if (N % 16) != 0 or (K % 32) != 0:
                raise RuntimeError(
                    f"ladder_pack_gpu shape unsupported for BitNet packing: N={N}, K={K}"
                )

            # Chunk packing to bound peak memory (important for large vocab/LM-head layers).
            max_rows = int(os.environ.get("SGLANG_TERNARY_LADDER_PACK_MAX_ROWS", "4096"))
            max_rows = max(16, (max_rows // 16) * 16)
            if N <= max_rows:
                return _ladder_pack_gpu(weight_int8).contiguous()

            out = torch.empty((N, K // 4), device=weight_int8.device, dtype=torch.uint8)
            for start in range(0, N, max_rows):
                end = min(start + max_rows, N)
                if (end - start) % 16 != 0:
                    raise RuntimeError(
                        f"ladder_pack_gpu chunk shape unsupported: rows={end-start}, K={K}"
                    )
                packed_chunk = _ladder_pack_gpu(weight_int8[start:end].contiguous())
                out[start:end].copy_(packed_chunk)
            return out.contiguous()

        convert_weight_int8_to_int2 = _ladder_pack_as_bitnet
        BITNET_PACK_AVAILABLE = True
        logger.info("[TERNARY] Using ladder_pack_gpu fallback for BitNet weight packing")
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


_TERNARY_GATE_SKIP_LOG: set[str] = set()
_TERNARY_MOE_VALIDATE_COUNT = 0
_TERNARY_MOE_VALIDATE_DONE: set[str] = set()
_TERNARY_PATH_COUNTER: Counter[str] = Counter()
_TERNARY_FALLBACK_COUNTER: Counter[str] = Counter()
_TERNARY_STATS_EVENTS = 0


def _collect_path_stats_enabled() -> bool:
    if _env_flag("SGLANG_TERNARY_COLLECT_PATH_STATS", "0"):
        return True
    return bool(os.environ.get("SGLANG_TERNARY_FALLBACK_REPORT", "").strip())


def _record_path_hit(name: str) -> None:
    global _TERNARY_STATS_EVENTS
    if not _collect_path_stats_enabled():
        return
    _TERNARY_PATH_COUNTER[name] += 1
    _TERNARY_STATS_EVENTS += 1
    _maybe_periodic_dump()


def _record_fallback_hit(name: str) -> None:
    global _TERNARY_STATS_EVENTS
    if not _collect_path_stats_enabled():
        return
    _TERNARY_FALLBACK_COUNTER[name] += 1
    _TERNARY_STATS_EVENTS += 1

    _maybe_periodic_dump()


def _maybe_periodic_dump() -> None:
    path_tmpl = os.environ.get("SGLANG_TERNARY_FALLBACK_REPORT", "").strip()
    if not path_tmpl:
        return
    interval = int(os.environ.get("SGLANG_TERNARY_FALLBACK_REPORT_EVERY", "5000"))
    if interval <= 0:
        return
    if _TERNARY_STATS_EVENTS % interval != 0:
        return
    _dump_runtime_stats()


def _dump_runtime_stats() -> None:
    path_tmpl = os.environ.get("SGLANG_TERNARY_FALLBACK_REPORT", "").strip()
    if not path_tmpl:
        return
    path = path_tmpl.replace("{pid}", str(os.getpid()))
    out = {
        "pid": os.getpid(),
        "path_counts": dict(sorted(_TERNARY_PATH_COUNTER.items())),
        "fallback_counts": dict(sorted(_TERNARY_FALLBACK_COUNTER.items())),
    }
    if _TERNARY_GATE_SKIP_LOG:
        out["gate_skip_layers"] = sorted(_TERNARY_GATE_SKIP_LOG)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _install_stats_signal_handlers() -> None:
    def _make_handler(prev_handler):
        def _handler(signum, frame):
            try:
                _dump_runtime_stats()
            except Exception:
                pass
            if callable(prev_handler):
                prev_handler(signum, frame)
                return
            if prev_handler == signal.SIG_IGN:
                return
            raise SystemExit(128 + int(signum))
        return _handler

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            prev = signal.getsignal(sig)
            signal.signal(sig, _make_handler(prev))
        except Exception:
            # Some runtimes may disallow signal registration in non-main threads.
            continue


atexit.register(_dump_runtime_stats)
_install_stats_signal_handlers()


def _record_gate_skip(prefix: str, reason: str) -> None:
    if not _env_flag("SGLANG_TERNARY_LIST_GATING_LAYERS", "0"):
        return
    name = prefix if prefix else "<unknown>"
    _TERNARY_GATE_SKIP_LOG.add(f"{name}\t{reason}")


def write_gate_skip_report(path: str) -> None:
    if not _TERNARY_GATE_SKIP_LOG:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in sorted(_TERNARY_GATE_SKIP_LOG):
            f.write(line + "\n")


def _should_validate_moe(layer: nn.Module) -> bool:
    if not _env_flag("SGLANG_TERNARY_MOE_VALIDATE", "0"):
        return False
    if _is_dynamo_compiling():
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    max_layers = int(os.environ.get("SGLANG_TERNARY_MOE_VALIDATE_MAX_LAYERS", "1"))
    global _TERNARY_MOE_VALIDATE_COUNT
    if _TERNARY_MOE_VALIDATE_COUNT >= max_layers:
        return False
    prefix = getattr(layer, "_ternary_prefix", "") or f"layer_{id(layer)}"
    if prefix in _TERNARY_MOE_VALIDATE_DONE:
        return False
    only_prefix = os.environ.get("SGLANG_TERNARY_MOE_VALIDATE_PREFIX")
    if only_prefix and only_prefix not in prefix:
        return False
    _TERNARY_MOE_VALIDATE_DONE.add(prefix)
    _TERNARY_MOE_VALIDATE_COUNT += 1
    return True


def _force_ternary_only() -> bool:
    return _env_flag("SGLANG_TERNARY_I2S_ONLY", "0") or _env_flag(
        "SGLANG_TERNARY_DROP_FP16_WEIGHTS", "0"
    )


def _drop_fp16_weights() -> bool:
    env = os.environ.get("SGLANG_TERNARY_DROP_FP16_WEIGHTS")
    if env is None:
        return _env_flag("SGLANG_TERNARY_I2S_ONLY", "0")
    return env.strip().lower() in ("1", "true", "yes", "on")


def _is_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def _get_i2s_cutlass_splits(N: int, M: Optional[int] = None, K: Optional[int] = None) -> int:
    env = os.environ.get("SGLANG_TERNARY_I2S_SPLITS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    if M is None or K is None:
        return 1 if (M is not None and M <= 16) else 4
    if M <= 16:
        return 1
    # Heuristic based on SM count and tile size.
    tile_m = 64
    tile_k = 32
    tile_n = 32 if N <= 1024 else 64
    tiles_m = (M + tile_m - 1) // tile_m
    tiles_n = (N + tile_n - 1) // tile_n
    total_tiles = tiles_m * tiles_n
    if not torch.cuda.is_available():
        return 4
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    target_tiles = max(1, num_sms * 3)
    if total_tiles >= target_tiles:
        return 1
    splits = (target_tiles + total_tiles - 1) // total_tiles
    # Prefer powers of two for reduction tree.
    splits = 1 << (splits - 1).bit_length()
    max_splits = max(1, (K + tile_k - 1) // tile_k)
    return max(1, min(splits, max_splits))


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


def _drop_moe_fp16_weights(layer: nn.Module) -> None:
    if not _drop_fp16_weights():
        return
    dropped = False
    for name in ("w13_weight", "w2_weight"):
        weight = getattr(layer, name, None)
        if not isinstance(weight, torch.Tensor):
            continue
        empty = torch.empty(0, device=weight.device, dtype=weight.dtype)
        replace_parameter(layer, name, empty)
        dropped = True
    if dropped:
        layer._ternary_moe_fp16_dropped = True


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

        # Keep router/gating layers in FP16/BF16 for stability.
        parts = pref.split(".")
        if any(p in ("router", "gate", "shared_expert_gate") for p in parts):
            _record_gate_skip(prefix, "router_or_gate")
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

            # Pack for BitNet kernel if available (still ternary/int2)
            use_bitnet_kernel = self.quant_config.use_bitnet_kernel
            bitnet_packed = False
            if use_bitnet_kernel and BITNET_PACK_AVAILABLE:
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
        if _drop_fp16_weights():
            if hasattr(layer, "_ternary_weight_fp16"):
                layer._buffers.pop("_ternary_weight_fp16", None)
                delattr(layer, "_ternary_weight_fp16")
            if hasattr(layer, "_ternary_weight_fp16_cache"):
                delattr(layer, "_ternary_weight_fp16_cache")
                
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
        force_ternary_only = _force_ternary_only()
        eligibility = {
            "not_dynamo_compiling": not _is_dynamo_compiling(),
            "use_bitnet_kernel": self.quant_config.use_bitnet_kernel,
            "bitnet_enabled": bitnet_enabled,
            "bitnet_cuda_available": BITNET_CUDA_AVAILABLE,
            "bitnet_lib_loaded": BITNET_LIB is not None,
            "weight_uint8": weight.dtype == torch.uint8,
            "x_cuda": x.is_cuda,
            "shape_supported": (N, K) in SUPPORTED_V4_NK_SHAPES,
        }
        can_use_kernel = all(eligibility.values())
        fallback_reasons: List[str] = []
        if not can_use_kernel:
            for k, v in eligibility.items():
                if not v:
                    reason = f"linear.bitnet_ineligible.{k}"
                    fallback_reasons.append(reason)
                    _record_fallback_hit(reason)
        
        # Handle FP8 input
        x_is_fp8 = x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        x_fp8_scale = getattr(x, '_fp8_scale', None)

        def _log_i2s_ineligible(reason: str) -> None:
            if not _env_flag("SGLANG_TERNARY_I2S_DEBUG", "0"):
                return
            cache = getattr(layer, "_ternary_i2s_ineligible_logged", None)
            if cache is None:
                cache = set()
                setattr(layer, "_ternary_i2s_ineligible_logged", cache)
            if reason in cache:
                return
            cache.add(reason)
            logger.info(
                "[TERNARY I2S CUTLASS] ineligible: %s (M=%d N=%d K=%d weight_dtype=%s x_dtype=%s)",
                reason,
                M,
                N,
                K,
                str(weight.dtype),
                str(x.dtype),
            )
        
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
                    _record_path_hit("linear.fp8_megafused_m1")
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
                    _record_path_hit("linear.bf16_megafused_m1")
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
                    _record_path_hit("linear.batch_megafused_m_gt_1")
                    if bias_bf16 is not None:
                        result = result + bias_bf16
                    return result.view(*x_shape[:-1], N)
                fallback_reasons.append("linear.batch_megafused_failed_or_disabled")
                _record_fallback_hit("linear.batch_megafused_failed_or_disabled")

        # CUTLASS fused i2s path (SM100, M>1 prefill)
        if weight.dtype == torch.uint8 and not x_is_fp8:
            x_bf16_2d = x_bf16.reshape(-1, K)
            result = self._apply_i2s_cutlass(layer, x_bf16_2d, bias_bf16, M, N, K, stream)
            if result is not None:
                _record_path_hit("linear.i2s_cutlass")
                return result.view(*x_shape[:-1], N)
            skip_reason = getattr(layer, "_ternary_i2s_cutlass_last_skip", None)
            if skip_reason:
                reason = f"linear.i2s_cutlass_skip.{skip_reason}"
                fallback_reasons.append(reason)
                _record_fallback_hit(reason)
        else:
            if weight.dtype != torch.uint8:
                _log_i2s_ineligible("weight_not_uint8")
                _record_fallback_hit("linear.i2s_cutlass_ineligible.weight_not_uint8")
            if x_is_fp8:
                _log_i2s_ineligible("fp8_input")
                _record_fallback_hit("linear.i2s_cutlass_ineligible.fp8_input")
        if force_ternary_only:
            _record_fallback_hit("linear.ternary_only_error.no_kernel_hit")
            raise RuntimeError(
                "Ternary-only mode enabled but no ternary kernel ran. "
                "Enable SGLANG_TERNARY_I2S_DEBUG=1 for skip reason."
            )
        
        # FP16 fallback
        if not fallback_reasons:
            fallback_reasons.append("linear.fp16_fallback.unknown")
            _record_fallback_hit("linear.fp16_fallback.unknown")
        _record_path_hit("linear.fp16_fallback")
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
        def _log_skip(reason: str) -> None:
            setattr(layer, "_ternary_i2s_cutlass_last_skip", reason)
            cache = getattr(layer, "_ternary_i2s_cutlass_skip_logged", None)
            if cache is None:
                cache = set()
                setattr(layer, "_ternary_i2s_cutlass_skip_logged", cache)
            if reason in cache:
                return
            cache.add(reason)
            logger.info(
                "[TERNARY I2S CUTLASS] skip: %s (M=%d N=%d K=%d dtype=%s fp8=%s)",
                reason,
                M,
                N,
                K,
                str(x_bf16_2d.dtype),
                "true" if x_bf16_2d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else "false",
            )

        if not I2S_CUTLASS_AVAILABLE or I2S_CUTLASS_LIB is None:
            _log_skip("lib_not_loaded")
            return None
        if not _is_sm100():
            _log_skip("not_sm100")
            return None
        if M <= DEFAULT_PREFILL_SKIP_M and not _force_ternary_only():
            _log_skip("M_too_small")
            return None
        if (N % 64) != 0 or (K % 4) != 0 or (K % 16) != 0 or K > 8192:
            _log_skip("shape_unsupported")
            return None
        if not _env_flag("SGLANG_TERNARY_USE_I2S_CUTLASS", "1"):
            _log_skip("disabled_by_env")
            return None

        if not x_bf16_2d.is_cuda:
            _log_skip("x_not_cuda")
            return None
        if not layer.weight.is_cuda:
            _log_skip("weight_not_cuda")
            return None
        if layer.weight.device != x_bf16_2d.device:
            _log_skip("weight_device_mismatch")
            return None
        alpha = getattr(layer, "ternary_alpha", None)
        if alpha is None:
            _log_skip("alpha_missing")
            return None
        if not alpha.is_cuda:
            _log_skip("alpha_not_cuda")
            return None
        if alpha.device != x_bf16_2d.device:
            _log_skip("alpha_device_mismatch")
            return None
        if not alpha.is_contiguous():
            alpha = alpha.contiguous()

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

        splits = _get_i2s_cutlass_splits(N, M_run, K)
        ws_bytes_ptr = 0
        if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs_streamk"):
            ws_bytes_ptr = int(
                I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs_streamk(
                    _PTR(x_fp16.data_ptr()),
                    _PTR(layer.weight.data_ptr()),
                    _PTR(out_cm.data_ptr()),
                    _INT(M_run),
                    _INT(N),
                    _INT(K),
                    _INT(splits),
                )
            )
        else:
            ws_bytes_ptr = int(
                I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_for_ptrs(
                    _PTR(x_fp16.data_ptr()),
                    _PTR(layer.weight.data_ptr()),
                    _PTR(out_cm.data_ptr()),
                    _INT(M_run),
                    _INT(N),
                    _INT(K),
                )
            )
        ws_bytes_shape = 0
        if hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size_streamk"):
            ws_bytes_shape = int(
                I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size_streamk(
                    _INT(M_run),
                    _INT(N),
                    _INT(K),
                    _INT(splits),
                )
            )
        elif hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_workspace_size"):
            ws_bytes_shape = int(
                I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_workspace_size(
                    _INT(M_run),
                    _INT(N),
                    _INT(K),
                )
            )
        ws_bytes = max(ws_bytes_ptr, ws_bytes_shape)
        if _env_flag("SGLANG_TERNARY_I2S_DEBUG", "0"):
            cache = getattr(layer, "_ternary_i2s_ws_logged", None)
            if cache is None:
                cache = set()
                setattr(layer, "_ternary_i2s_ws_logged", cache)
            key = (M_run, N, K, splits)
            if key not in cache:
                cache.add(key)
                logger.info(
                    "[TERNARY I2S CUTLASS] workspace=%d (M_run=%d N=%d K=%d splits=%d)",
                    ws_bytes,
                    M_run,
                    N,
                    K,
                    splits,
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

        def _run_i2s(workspace_ptr, ws_bytes) -> int:
            if I2S_CUTLASS_HAS_ALPHA_PTR and hasattr(
                I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_v8_run_streamk_alpha"
            ):
                return I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_run_streamk_alpha(
                    _PTR(x_fp16.data_ptr()),
                    _PTR(layer.weight.data_ptr()),
                    _PTR(alpha.data_ptr()),
                    _PTR(out_cm.data_ptr()),
                    _INT(M_run),
                    _INT(N),
                    _INT(K),
                    _INT(splits),
                    workspace_ptr,
                    _SIZE_T(ws_bytes),
                    _PTR(stream),
                )
            if not hasattr(I2S_CUTLASS_LIB, "i2s_fused_mixed_sm100_set_alpha_const"):
                _log_skip("alpha_const_unavailable")
                return -1
            rc_inner = I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_set_alpha_const(
                _PTR(alpha.data_ptr()),
                _INT(K),
                _PTR(stream),
            )
            if rc_inner != 0:
                return rc_inner
            return I2S_CUTLASS_LIB.i2s_fused_mixed_sm100_v8_run_streamk(
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

        rc = _run_i2s(workspace_ptr, ws_bytes)
        if rc == 11:
            # Retry with a growing workspace size to handle under-reported requirements.
            max_mb = int(os.environ.get("SGLANG_TERNARY_I2S_WS_GROW_MAX_MB", "256"))
            max_bytes = max_mb * 1024 * 1024
            grow_bytes = max(ws_bytes, 1 << 20)  # start with at least 1MB
            while rc == 11 and grow_bytes <= max_bytes:
                if _env_flag("SGLANG_TERNARY_I2S_DEBUG", "0"):
                    logger.info(
                        "[TERNARY I2S CUTLASS] retry rc=11: ws=%d -> %d (M=%d N=%d K=%d splits=%d)",
                        ws_bytes,
                        grow_bytes,
                        M_run,
                        N,
                        K,
                        splits,
                    )
                ws_bytes = grow_bytes
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
                rc = _run_i2s(workspace_ptr, ws_bytes)
                grow_bytes *= 2

        if rc != 0:
            _log_skip(f"kernel_rc_{rc}")
            return None

        if bias_bf16 is not None:
            out_cm.add_(bias_bf16.to(torch.float32).view(-1, 1))

        capture_active = torch.cuda.is_current_stream_capturing()
        if _env_flag("SGLANG_TERNARY_I2S_SYNC", "0") and not capture_active:
            torch.cuda.current_stream().synchronize()
        if _env_flag("SGLANG_TERNARY_I2S_VALIDATE", "0") and not capture_active:
            if not torch.isfinite(out_cm).all():
                if _env_flag("SGLANG_TERNARY_I2S_DEBUG", "0"):
                    if not getattr(layer, "_ternary_i2s_nonfinite_logged", False):
                        layer._ternary_i2s_nonfinite_logged = True
                        x_finite = torch.isfinite(x_fp16).all().item()
                        alpha_finite = torch.isfinite(alpha).all().item()
                        half_max = float(torch.finfo(torch.float16).max)
                        x_max = x_bf16_2d.abs().max().float().item()
                        alpha_max = alpha.abs().max().float().item()
                        x_over = (x_bf16_2d.abs() > half_max).sum().item()
                        alpha_over = (alpha.abs() > half_max).sum().item()
                        logger.warning(
                            "[TERNARY I2S CUTLASS] non_finite: x_finite=%s alpha_finite=%s "
                            "x_max=%.4e alpha_max=%.4e x_over_half=%d alpha_over_half=%d",
                            x_finite,
                            alpha_finite,
                            x_max,
                            alpha_max,
                            x_over,
                            alpha_over,
                        )
                _log_skip("output_non_finite")
                return None
        if _env_flag("SGLANG_TERNARY_I2S_DEBUG", "0"):
            if not getattr(layer, "_ternary_i2s_cutlass_hit_logged", False):
                layer._ternary_i2s_cutlass_hit_logged = True
                logger.info(
                    "[TERNARY I2S CUTLASS] hit (M=%d N=%d K=%d splits=%d ws=%d)",
                    M,
                    N,
                    K,
                    splits,
                    ws_bytes,
                )
        setattr(layer, "_ternary_i2s_cutlass_last_skip", "hit")

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
        _drop_moe_fp16_weights(layer)
        
    @_dynamo_disable
    def apply(self, layer: torch.nn.Module, dispatch_output):
        """Apply ternary MoE forward pass."""
        # Import MoE utilities
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe, MoeRunnerConfig
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        
        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        moe_runner_config = getattr(layer, '_ternary_moe_runner_config', None) or MoeRunnerConfig()
        force_ternary_only = _force_ternary_only() or getattr(layer, "_ternary_moe_fp16_dropped", False)
        
        # Check format
        if not hasattr(topk_output, 'topk_weights') or not hasattr(topk_output, 'topk_ids'):
            if force_ternary_only:
                _record_fallback_hit("moe.missing_topk.ternary_only_error")
                raise RuntimeError(
                    "Ternary-only mode enabled but MoE fallback requires FP16 weights "
                    "(missing topk info)."
                )
            _record_path_hit("moe.fp16_fallback.missing_topk")
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
                    _record_path_hit("moe.decode_fused_fp8")
                    return StandardCombineInput(hidden_states=result)
                _record_fallback_hit("moe.decode_fused_fp8_failed")
            elif hidden_states.dtype == torch.bfloat16:
                # BF16 decode path
                result = self._apply_decode_fused(layer, hidden_states, topk_ids, topk_weights)
                if result is not None:
                    _record_path_hit("moe.decode_fused_bf16")
                    return StandardCombineInput(hidden_states=result)
                else:
                    _record_fallback_hit("moe.decode_fused_bf16_failed")
                    if not getattr(layer, '_decode_fused_fail_logged', False):
                        layer._decode_fused_fail_logged = True
                        logger.warning(f"[TERNARY MoE] _apply_decode_fused returned None, using fallback")
        
        # M>1 batched ternary path (Triton) when available
        if num_tokens > 1:
            use_batched_ternary = _env_flag(
                "SGLANG_TERNARY_MOE_TRITON",
                "1" if force_ternary_only else "0",
            )
            if use_batched_ternary:
                result = self._apply_batched_triton(
                    layer,
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    moe_runner_config,
                )
                if result is not None:
                    _record_path_hit("moe.batched_triton")
                    return StandardCombineInput(hidden_states=result)
                _record_fallback_hit("moe.batched_triton_failed")
                if force_ternary_only:
                    _record_fallback_hit("moe.batched_triton_failed.ternary_only_error")
                    raise RuntimeError(
                        "Ternary-only mode enabled but batched ternary MoE path failed. "
                        "Set SGLANG_TERNARY_MOE_TRITON=0 to allow FP16 fallback."
                    )
        
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
        
        if force_ternary_only:
            _record_fallback_hit("moe.fp16_fallback_blocked.ternary_only_error")
            raise RuntimeError(
                "Ternary-only mode enabled but MoE fallback requires FP16 weights "
                "(M>1 ternary kernels are disabled). "
                "Set SGLANG_TERNARY_MOE_TRITON=1 to enable the batched ternary MoE path "
                "or disable SGLANG_TERNARY_I2S_ONLY/SGLANG_TERNARY_DROP_FP16_WEIGHTS to allow FP16 fallback."
            )
        _record_path_hit("moe.fp16_fallback")
        return StandardCombineInput(hidden_states=fused_moe(
            hidden_states=hidden_states_for_moe,
            w1=layer.w13_weight, w2=layer.w2_weight,
            topk_output=topk_output,
            moe_runner_config=moe_runner_config,
        ))

    def _apply_batched_triton(
        self,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        moe_runner_config,
    ) -> Optional[torch.Tensor]:
        try:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
                invoke_fused_moe_ternary_kernel,
                get_default_ternary_moe_config,
            )
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
                moe_sum_reduce_torch_compile,
                moe_sum_reduce_triton,
            )
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size,
            )
        except Exception as e:
            logger.debug(f"[TERNARY MoE] Triton ternary kernel unavailable: {e}")
            _record_fallback_hit("moe.batched_triton.unavailable")
            return None

        if moe_runner_config.activation != "silu":
            logger.warning(
                "[TERNARY MoE] Triton batched path only supports SiLU activation."
            )
            _record_fallback_hit("moe.batched_triton.unsupported_activation")
            return None
        if moe_runner_config.apply_router_weight_on_input:
            logger.warning(
                "[TERNARY MoE] apply_router_weight_on_input not supported in batched ternary path."
            )
            _record_fallback_hit("moe.batched_triton.unsupported_router_weight_on_input")
            return None

        if not getattr(layer, "_ternary_moe_v4_enabled", False):
            _record_fallback_hit("moe.batched_triton.v4_disabled")
            return None
        if not (
            hasattr(layer, "_ternary_w13_packed")
            and hasattr(layer, "_ternary_w2_packed")
            and hasattr(layer, "_ternary_moe_alpha_w13")
            and hasattr(layer, "_ternary_moe_alpha_w2")
        ):
            _record_fallback_hit("moe.batched_triton.weights_not_ready")
            return None

        # Some MoE module variants can reach this path without ternary metadata
        # being initialized on the specific layer object. Fall back safely.
        if not (
            hasattr(layer, "_ternary_moe_num_experts")
            and hasattr(layer, "_ternary_moe_hidden_size")
            and hasattr(layer, "_ternary_moe_intermediate_size")
        ):
            _record_fallback_hit("moe.batched_triton.meta_not_ready")
            return None

        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        M = hidden_states.shape[0]
        top_k = int(topk_ids.shape[1])
        try:
            num_experts = int(getattr(layer, "_ternary_moe_num_experts"))
            hidden_size = int(getattr(layer, "_ternary_moe_hidden_size"))
            intermediate_size = int(getattr(layer, "_ternary_moe_intermediate_size"))
            if num_experts <= 0 or hidden_size <= 0 or intermediate_size <= 0:
                raise ValueError("non-positive ternary MoE metadata")
        except Exception:
            _record_fallback_hit("moe.batched_triton.meta_invalid")
            return None
        N_w13 = 2 * intermediate_size
        gate_cfg = get_default_ternary_moe_config("gate_up", m=M)
        down_cfg = get_default_ternary_moe_config("down", m=M * top_k)

        gate_sorted_ids, gate_expert_ids, gate_num_post = moe_align_block_size(
            topk_ids, gate_cfg["BLOCK_SIZE_M"], num_experts
        )
        down_sorted_ids, down_expert_ids, down_num_post = moe_align_block_size(
            topk_ids, down_cfg["BLOCK_SIZE_M"], num_experts
        )

        stream_raw = torch.cuda.current_stream().cuda_stream
        if stream_raw is None:
            stream_id = 0
        else:
            try:
                stream_id = int(stream_raw)
            except TypeError:
                stream_id = int(stream_raw.value) if hasattr(stream_raw, "value") and stream_raw.value is not None else 0

        cache = getattr(layer, "_ternary_moe_batched_cache", None)
        if cache is None:
            cache = {}
            setattr(layer, "_ternary_moe_batched_cache", cache)
        key = (stream_id, M, top_k)
        buf = cache.get(key)
        if buf is None:
            buf = {
                "gateup_out": torch.empty((M, top_k, N_w13), device=hidden_states.device, dtype=torch.bfloat16),
                "intermediate": torch.empty((M * top_k, intermediate_size), device=hidden_states.device, dtype=torch.bfloat16),
                "down_out": torch.empty((M, top_k, hidden_size), device=hidden_states.device, dtype=torch.bfloat16),
                "out": torch.empty((M, hidden_size), device=hidden_states.device, dtype=torch.bfloat16),
            }
            cache[key] = buf

        gateup_out = buf["gateup_out"]
        intermediate = buf["intermediate"]
        down_out = buf["down_out"]
        out = buf["out"]

        invoke_fused_moe_ternary_kernel(
            hidden_states,
            layer._ternary_w13_packed,
            layer._ternary_moe_alpha_w13,
            gateup_out,
            topk_weights,
            topk_ids,
            gate_sorted_ids,
            gate_expert_ids,
            gate_num_post,
            False,
            top_k,
            config=gate_cfg,
        )

        try:
            from sgl_kernel import silu_and_mul  # type: ignore
            silu_and_mul(gateup_out.view(-1, N_w13), intermediate)
        except Exception:
            gate, up = gateup_out.view(-1, N_w13).chunk(2, dim=-1)
            intermediate.copy_(torch.nn.functional.silu(gate) * up)

        # Second GEMM: treat top_k as 1 so sorted_token_ids index directly into [M*topk, K]
        invoke_fused_moe_ternary_kernel(
            intermediate,
            layer._ternary_w2_packed,
            layer._ternary_moe_alpha_w2,
            down_out,
            topk_weights,
            topk_ids,
            down_sorted_ids,
            down_expert_ids,
            down_num_post,
            True,
            1,
            config=down_cfg,
        )

        if _should_validate_moe(layer):
            max_tokens = int(os.environ.get("SGLANG_TERNARY_MOE_VALIDATE_MAX_TOKENS", "1"))
            max_topk = int(os.environ.get("SGLANG_TERNARY_MOE_VALIDATE_MAX_TOPK", "1"))
            tokens_to_check = min(M, max_tokens)
            topk_to_check = min(top_k, max_topk)
            if tokens_to_check > 0 and topk_to_check > 0:
                with torch.no_grad():
                    w13_cache: Dict[int, torch.Tensor] = {}
                    w2_cache: Dict[int, torch.Tensor] = {}
                    gate_abs_max = 0.0
                    gate_rel_max = 0.0
                    down_abs_max = 0.0
                    down_rel_max = 0.0
                    for t in range(tokens_to_check):
                        x_t = hidden_states[t:t + 1]
                        for j in range(topk_to_check):
                            e = int(topk_ids[t, j])
                            w13 = w13_cache.get(e)
                            if w13 is None:
                                w13 = unpack_i2s_weights(
                                    layer._ternary_w13_packed[e],
                                    hidden_size,
                                    layer._ternary_moe_alpha_w13[e],
                                    torch.bfloat16,
                                )
                                w13_cache[e] = w13
                            gate_ref = (x_t @ w13.t()).squeeze(0).float()
                            gate_act = gateup_out[t, j].float()
                            gate_diff = (gate_act - gate_ref).abs()
                            gate_abs_max = max(gate_abs_max, gate_diff.max().item())
                            gate_rel_max = max(
                                gate_rel_max,
                                (gate_diff / (gate_ref.abs() + 1e-6)).max().item(),
                            )

                            gate = gateup_out[t, j, :intermediate_size].float()
                            up = gateup_out[t, j, intermediate_size:].float()
                            inter_ref = (F.silu(gate) * up).to(torch.bfloat16)
                            w2 = w2_cache.get(e)
                            if w2 is None:
                                w2 = unpack_i2s_weights(
                                    layer._ternary_w2_packed[e],
                                    intermediate_size,
                                    layer._ternary_moe_alpha_w2[e],
                                    torch.bfloat16,
                                )
                                w2_cache[e] = w2
                            down_ref = (inter_ref @ w2.t()).float()
                            down_ref = down_ref * topk_weights[t, j].float()
                            down_act = down_out[t, j].float()
                            down_diff = (down_act - down_ref).abs()
                            down_abs_max = max(down_abs_max, down_diff.max().item())
                            down_rel_max = max(
                                down_rel_max,
                                (down_diff / (down_ref.abs() + 1e-6)).max().item(),
                            )
                    prefix = getattr(layer, "_ternary_prefix", "<unknown>")
                    logger.warning(
                        "[TERNARY MoE VALIDATE] %s tokens=%d topk=%d "
                        "gate_abs=%.4e gate_rel=%.4e down_abs=%.4e down_rel=%.4e",
                        prefix,
                        tokens_to_check,
                        topk_to_check,
                        gate_abs_max,
                        gate_rel_max,
                        down_abs_max,
                        down_rel_max,
                    )

        routed_scaling_factor = moe_runner_config.routed_scaling_factor
        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if moe_runner_config.no_combine:
            return down_out.view(M, top_k, hidden_size)

        if top_k == 1 and routed_scaling_factor == 1.0:
            out.copy_(down_out.squeeze(1))
        elif top_k == 2 and routed_scaling_factor == 1.0:
            out.copy_(down_out[:, 0] + down_out[:, 1])
        else:
            if M <= 32:
                moe_sum_reduce_torch_compile(down_out, out, routed_scaling_factor)
            else:
                moe_sum_reduce_triton(down_out, out, routed_scaling_factor)

        return out

    def _apply_decode_fused(self, layer, hidden_states, topk_ids, topk_weights):
        """Fully fused decode path for M=1."""
        stream_raw = torch.cuda.current_stream().cuda_stream
        if stream_raw is None:
            stream = _PTR(0)
            stream_id = 0
        else:
            stream = _PTR(stream_raw)
            try:
                stream_id = int(stream_raw)
            except TypeError:
                stream_id = int(stream.value) if hasattr(stream, "value") and stream.value is not None else 0
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
        stream_raw = torch.cuda.current_stream().cuda_stream
        if stream_raw is None:
            stream = _PTR(0)
            stream_id = 0
        else:
            stream = _PTR(stream_raw)
            try:
                stream_id = int(stream_raw)
            except TypeError:
                stream_id = int(stream.value) if hasattr(stream, "value") and stream.value is not None else 0
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
