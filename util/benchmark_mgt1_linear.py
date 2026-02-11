#!/usr/bin/env python3
"""
Benchmark and correctness-check the experimental M>1 ternary linear CUDA kernel.

Kernel under test:
  v4_batch_megafused_v2_launch (libternary_bitnet.so)

This utility is the Phase-2/Phase-4 bridge:
- shape x M correctness grid against dense reference
- per-shape M latency stats for kernel path
- machine-readable JSON artifact for promotion gates
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.ternary import (
    BITNET_PACK_AVAILABLE,
    convert_weight_int8_to_int2,
)


DEFAULT_SHAPES = "5120x2048,2048x4096,2048x2048,1536x2048,768x2048,2048x768"
DEFAULT_M_VALUES = "2,4,8,16,32,64"


@dataclass
class ProbeResult:
    m: int
    n: int
    k: int
    status: str
    kernel_rc: int
    reference_mode: str
    max_abs: float
    max_rel: float
    mean_abs: float
    non_finite: int
    lat_ms_avg: float
    lat_ms_p50: float
    lat_ms_p95: float
    samples: int
    pass_thresholds: bool


def parse_shapes(raw: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"invalid shape entry '{item}', expected NxK")
        out.append((int(parts[0]), int(parts[1])))
    if not out:
        raise ValueError("no shapes specified")
    return out


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("empty integer list")
    return out


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def load_kernel(lib_path: Path):
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.v4_batch_megafused_v2_launch
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    m1_fn = None
    if hasattr(lib, "bitlinear_bf16xint2_v4_megafused"):
        m1_fn = lib.bitlinear_bf16xint2_v4_megafused
        m1_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        m1_fn.restype = ctypes.c_int
    return fn, m1_fn


def build_m1_kernel_reference(
    m1_kernel_fn,
    x: torch.Tensor,
    w_bitnet: torch.Tensor,
    alpha: torch.Tensor,
    n: int,
    k: int,
    stream_ptr,
) -> Tuple[torch.Tensor, int]:
    """Build reference by running the production M=1 kernel row-by-row."""
    if m1_kernel_fn is None:
        raise RuntimeError("M=1 kernel function is unavailable in libternary_bitnet.so")

    m = x.shape[0]
    out_ref = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    tmp_out = torch.empty((1, n), device=x.device, dtype=torch.bfloat16)
    for row in range(m):
        x_row = x[row : row + 1].contiguous()
        rc = m1_kernel_fn(
            ctypes.c_void_p(x_row.data_ptr()),
            ctypes.c_void_p(w_bitnet.data_ptr()),
            ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(tmp_out.data_ptr()),
            ctypes.c_int(1),
            ctypes.c_int(n),
            ctypes.c_int(k),
            stream_ptr,
        )
        if rc != 0:
            return out_ref, int(rc)
        out_ref[row : row + 1].copy_(tmp_out)
    return out_ref, 0


def make_weight_and_alpha(
    n: int, k: int, device: torch.device, seed: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    # Ternary signs in {-1,0,1}
    w_sign = torch.randint(low=-1, high=2, size=(n, k), generator=g, dtype=torch.int8).to(device)
    alpha = torch.empty(k, dtype=torch.float32).uniform_(0.25, 1.5, generator=g)
    alpha = alpha.to(device=device)
    if not BITNET_PACK_AVAILABLE or convert_weight_int8_to_int2 is None:
        raise RuntimeError(
            "BitNet packer is unavailable; cannot generate runtime-compatible packed weights."
        )
    w_bitnet = convert_weight_int8_to_int2(w_sign).contiguous()
    if not w_bitnet.is_cuda:
        w_bitnet = w_bitnet.to(device=device, non_blocking=True)
    # Dense reference corresponding to the same ternary sign and alpha semantics.
    w_ref = (w_sign.to(torch.float32) * alpha.view(1, -1)).to(torch.bfloat16)
    return w_bitnet, alpha, w_ref


def run_case(
    kernel_fn,
    m1_kernel_fn,
    m: int,
    n: int,
    k: int,
    reference_mode: str,
    iters: int,
    warmup: int,
    max_abs_threshold: float,
    max_rel_threshold: float,
    seed: int,
) -> ProbeResult:
    device = torch.device("cuda")
    stream = torch.cuda.current_stream(device=device)
    stream_ptr = ctypes.c_void_p(int(stream.cuda_stream))

    # Keep deterministic input per case for reproducibility.
    x_seed = seed + (m * 1000003 + n * 1009 + k * 17)
    g = torch.Generator(device="cpu")
    g.manual_seed(x_seed)
    x = torch.randn((m, k), generator=g, dtype=torch.float32).to(device=device, dtype=torch.bfloat16)
    w_bitnet, alpha, w_ref = make_weight_and_alpha(n, k, device, seed=x_seed + 19)

    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    rc = kernel_fn(
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(w_bitnet.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        stream_ptr,
    )
    torch.cuda.synchronize(device=device)
    if rc != 0:
        return ProbeResult(
            m=m,
            n=n,
            k=k,
            status="kernel_unsupported_or_failed",
            kernel_rc=int(rc),
            reference_mode=reference_mode,
            max_abs=0.0,
            max_rel=0.0,
            mean_abs=0.0,
            non_finite=0,
            lat_ms_avg=0.0,
            lat_ms_p50=0.0,
            lat_ms_p95=0.0,
            samples=0,
            pass_thresholds=False,
        )

    # Reference
    with torch.no_grad():
        if reference_mode == "m1_kernel":
            ref, ref_rc = build_m1_kernel_reference(
                m1_kernel_fn,
                x,
                w_bitnet,
                alpha,
                n,
                k,
                stream_ptr,
            )
            if ref_rc != 0:
                return ProbeResult(
                    m=m,
                    n=n,
                    k=k,
                    status="reference_kernel_failed",
                    kernel_rc=int(ref_rc),
                    reference_mode=reference_mode,
                    max_abs=0.0,
                    max_rel=0.0,
                    mean_abs=0.0,
                    non_finite=0,
                    lat_ms_avg=0.0,
                    lat_ms_p50=0.0,
                    lat_ms_p95=0.0,
                    samples=0,
                    pass_thresholds=False,
                )
        else:
            ref = F.linear(x, w_ref)
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    rel = diff / denom

    max_abs = float(diff.max().item())
    max_rel = float(rel.max().item())
    mean_abs = float(diff.mean().item())
    non_finite = int((~torch.isfinite(out)).sum().item() + (~torch.isfinite(ref)).sum().item())

    # Latency measurement
    times_ms: List[float] = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    total_iters = warmup + iters
    for i in range(total_iters):
        start_evt.record(stream)
        rc_i = kernel_fn(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(w_bitnet.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(m),
            ctypes.c_int(n),
            ctypes.c_int(k),
            stream_ptr,
        )
        end_evt.record(stream)
        end_evt.synchronize()
        if rc_i != 0:
            return ProbeResult(
                m=m,
                n=n,
                k=k,
                status="kernel_failed_during_timing",
                kernel_rc=int(rc_i),
                reference_mode=reference_mode,
                max_abs=max_abs,
                max_rel=max_rel,
                mean_abs=mean_abs,
                non_finite=non_finite,
                lat_ms_avg=0.0,
                lat_ms_p50=0.0,
                lat_ms_p95=0.0,
                samples=0,
                pass_thresholds=False,
            )
        if i >= warmup:
            times_ms.append(float(start_evt.elapsed_time(end_evt)))

    pass_thresholds = non_finite == 0 and max_abs <= max_abs_threshold and max_rel <= max_rel_threshold

    return ProbeResult(
        m=m,
        n=n,
        k=k,
        status="ok" if pass_thresholds else "threshold_failed",
        kernel_rc=0,
        reference_mode=reference_mode,
        max_abs=max_abs,
        max_rel=max_rel,
        mean_abs=mean_abs,
        non_finite=non_finite,
        lat_ms_avg=float(statistics.mean(times_ms)) if times_ms else 0.0,
        lat_ms_p50=percentile(times_ms, 50),
        lat_ms_p95=percentile(times_ms, 95),
        samples=len(times_ms),
        pass_thresholds=pass_thresholds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness/perf probe for v4_batch_megafused_v2_launch (M>1 ternary linear)."
    )
    parser.add_argument("--lib-path", default="/home/ubuntu/inference/sglang/libternary_bitnet.so")
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help=f"Comma list of NxK (default: {DEFAULT_SHAPES})")
    parser.add_argument("--m-values", default=DEFAULT_M_VALUES, help=f"Comma list of M values (default: {DEFAULT_M_VALUES})")
    parser.add_argument("--iters", type=int, default=80, help="Timed iterations per case.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations per case.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-abs-threshold", type=float, default=0.08)
    parser.add_argument("--max-rel-threshold", type=float, default=0.25)
    parser.add_argument(
        "--reference",
        choices=["m1_kernel", "dense"],
        default="m1_kernel",
        help="Reference mode for correctness checks.",
    )
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required.")
        return 2

    lib_path = Path(args.lib_path).expanduser().resolve()
    if not lib_path.exists():
        print(f"Error: kernel library not found: {lib_path}")
        return 1

    shapes = parse_shapes(args.shapes)
    m_values = parse_int_list(args.m_values)

    print("==============================================================")
    print("M>1 Ternary Linear Kernel Probe")
    print("==============================================================")
    print(f"Library:             {lib_path}")
    print(f"Shapes (N,K):        {shapes}")
    print(f"M values:            {m_values}")
    print(f"Iters/Warmup:        {args.iters}/{args.warmup}")
    print(f"Thresholds (abs/rel): {args.max_abs_threshold}/{args.max_rel_threshold}")
    print("==============================================================")

    kernel_fn, m1_kernel_fn = load_kernel(lib_path)
    results: List[ProbeResult] = []

    t0 = time.time()
    for n, k in shapes:
        for m in m_values:
            print(f"Running case M={m} N={n} K={k} ...")
            res = run_case(
                kernel_fn=kernel_fn,
                m1_kernel_fn=m1_kernel_fn,
                m=m,
                n=n,
                k=k,
                reference_mode=args.reference,
                iters=args.iters,
                warmup=args.warmup,
                max_abs_threshold=args.max_abs_threshold,
                max_rel_threshold=args.max_rel_threshold,
                seed=args.seed,
            )
            results.append(res)
            print(
                f"  status={res.status} rc={res.kernel_rc} "
                f"max_abs={res.max_abs:.5f} max_rel={res.max_rel:.5f} "
                f"lat_avg={res.lat_ms_avg:.4f}ms"
            )

    dt = time.time() - t0
    ok = [r for r in results if r.status == "ok"]
    threshold_failed = [r for r in results if r.status == "threshold_failed"]
    unsupported = [r for r in results if r.status not in ("ok", "threshold_failed")]

    summary: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "library": str(lib_path),
        "shapes": [{"n": n, "k": k} for n, k in shapes],
        "m_values": m_values,
        "iters": args.iters,
        "warmup": args.warmup,
        "reference_mode": args.reference,
        "thresholds": {
            "max_abs": args.max_abs_threshold,
            "max_rel": args.max_rel_threshold,
        },
        "elapsed_sec": dt,
        "counts": {
            "total_cases": len(results),
            "ok": len(ok),
            "threshold_failed": len(threshold_failed),
            "unsupported_or_failed": len(unsupported),
        },
        "results": [asdict(r) for r in results],
    }

    print("")
    print("==============================================================")
    print("Probe Summary")
    print("==============================================================")
    print(f"Total cases:          {summary['counts']['total_cases']}")
    print(f"Pass (thresholds):    {summary['counts']['ok']}")
    print(f"Threshold failed:     {summary['counts']['threshold_failed']}")
    print(f"Unsupported/failed:   {summary['counts']['unsupported_or_failed']}")
    if ok:
        fastest = min(ok, key=lambda x: x.lat_ms_avg)
        print(
            "Fastest passing case: "
            f"M={fastest.m} N={fastest.n} K={fastest.k} lat_avg={fastest.lat_ms_avg:.4f}ms"
        )
    print("==============================================================")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved: {out_path}")

    # Non-zero only if there are explicit threshold failures; unsupported cases can
    # happen during bring-up and should be interpreted from the JSON report.
    return 1 if threshold_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

