"""High-resolution analyzer for torch.profiler Chrome traces.

This module parses `*.trace.json` / `*.trace.json.gz` files exported by
`torch.profiler.profile(...).export_chrome_trace(...)` and produces an
operator/kernel-centric summary geared towards bottleneck hunting.

It is intentionally dependency-free (no pandas / tensorboard) so it can be used
on minimal serving environments.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class _Agg:
    total_us: float = 0.0
    count: int = 0
    min_us: float = float("inf")
    max_us: float = 0.0

    def add(self, dur_us: float) -> None:
        self.total_us += dur_us
        self.count += 1
        if dur_us < self.min_us:
            self.min_us = dur_us
        if dur_us > self.max_us:
            self.max_us = dur_us


def _read_json(path: Path) -> Dict[str, Any]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cat_tokens(cat: Any) -> Tuple[str, ...]:
    if not cat:
        return ()
    if not isinstance(cat, str):
        return ()
    # Chrome trace uses comma-separated categories.
    return tuple(x.strip() for x in cat.split(",") if x.strip())


def _classify_event(cat_tokens: Tuple[str, ...]) -> str:
    # Torch profiler convention:
    # - cpu_op: PyTorch op running on CPU
    # - Kernel/kernel: CUDA kernel execution (varies by PyTorch version)
    # - cuda_runtime: runtime API (launch, memcpy, sync, etc.)
    # - user_annotation: record_function / NVTX ranges
    lower = {t.lower() for t in cat_tokens}

    # GPU execution / transfers.
    if "kernel" in lower:
        return "cuda_kernel"
    if "gpu_memcpy" in lower or "gpu_memset" in lower:
        # Treat GPU memcpy/memset as GPU work for bottleneck ranking.
        return "cuda_kernel"

    # CPU operators.
    if "cpu_op" in lower:
        return "cpu_op"

    # CUDA runtime/driver APIs.
    if "cuda_runtime" in lower or "cuda_driver" in lower:
        return "cuda_runtime"

    # Annotations / record_function / NVTX.
    if "user_annotation" in lower:
        return "annotation"
    return "other"


def _iter_complete_events(trace_events: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for e in trace_events:
        # "X" = complete event (has duration).
        if e.get("ph") != "X":
            continue
        if "dur" not in e:
            continue
        yield e


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    # Linear interpolation between closest ranks.
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def analyze_trace_events(
    trace_events: Sequence[Dict[str, Any]],
    *,
    top_n: int = 50,
    micro_kernel_us: float = 10.0,
) -> Dict[str, Any]:
    """Analyze parsed chrome trace events (trace['traceEvents'])."""

    per_kind: Dict[str, Dict[str, _Agg]] = {
        "cpu_op": {},
        "cuda_kernel": {},
        "cuda_runtime": {},
        "annotation": {},
        "other": {},
    }
    micro_kernel: Dict[str, _Agg] = {}

    min_ts: Optional[float] = None
    max_ts_end: Optional[float] = None
    total_events = 0

    for e in _iter_complete_events(trace_events):
        total_events += 1
        name = str(e.get("name", ""))
        dur_us = float(e.get("dur", 0.0))
        ts = e.get("ts")
        if isinstance(ts, (int, float)):
            if min_ts is None or ts < min_ts:
                min_ts = float(ts)
            end_ts = float(ts) + dur_us
            if max_ts_end is None or end_ts > max_ts_end:
                max_ts_end = end_ts

        kind = _classify_event(_cat_tokens(e.get("cat")))
        bucket = per_kind.setdefault(kind, {})
        agg = bucket.get(name)
        if agg is None:
            agg = _Agg()
            bucket[name] = agg
        agg.add(dur_us)

        if kind == "cuda_kernel" and dur_us <= micro_kernel_us:
            mk = micro_kernel.get(name)
            if mk is None:
                mk = _Agg()
                micro_kernel[name] = mk
            mk.add(dur_us)

    wall_us = 0.0
    if min_ts is not None and max_ts_end is not None:
        wall_us = max(0.0, max_ts_end - min_ts)

    def top_table(kind: str) -> List[Dict[str, Any]]:
        items = list(per_kind.get(kind, {}).items())
        items.sort(key=lambda kv: kv[1].total_us, reverse=True)
        top_items = items[: max(0, int(top_n))]

        # Second pass: collect per-call durations only for top names to compute percentiles.
        wanted = {name for name, _ in top_items}
        samples: Dict[str, List[float]] = {name: [] for name in wanted}
        for e in _iter_complete_events(trace_events):
            if str(e.get("name", "")) not in wanted:
                continue
            if _classify_event(_cat_tokens(e.get("cat"))) != kind:
                continue
            samples[str(e.get("name", ""))].append(float(e.get("dur", 0.0)))

        out = []
        for name, agg in top_items:
            durs = samples.get(name) or []
            durs.sort()
            out.append(
                {
                    "name": name,
                    "calls": agg.count,
                    "total_ms": agg.total_us / 1000.0,
                    "avg_us": (agg.total_us / agg.count) if agg.count else 0.0,
                    "min_us": 0.0 if agg.min_us == float("inf") else agg.min_us,
                    "p50_us": _percentile(durs, 50.0),
                    "p90_us": _percentile(durs, 90.0),
                    "p99_us": _percentile(durs, 99.0),
                    "max_us": agg.max_us,
                }
            )
        return out

    def micro_kernel_table() -> List[Dict[str, Any]]:
        items = list(micro_kernel.items())
        items.sort(key=lambda kv: (kv[1].count, kv[1].total_us), reverse=True)
        out = []
        for name, agg in items[: max(0, int(top_n))]:
            out.append(
                {
                    "name": name,
                    "calls": agg.count,
                    "total_ms": agg.total_us / 1000.0,
                    "avg_us": (agg.total_us / agg.count) if agg.count else 0.0,
                    "max_us": agg.max_us,
                }
            )
        return out

    return {
        "meta": {
            "events_total": total_events,
            "wall_ms": wall_us / 1000.0,
            "micro_kernel_us_threshold": micro_kernel_us,
        },
        "top": {
            "cuda_kernels": top_table("cuda_kernel"),
            "cpu_ops": top_table("cpu_op"),
            "cuda_runtime": top_table("cuda_runtime"),
            "annotations": top_table("annotation"),
            "micro_kernels_by_count": micro_kernel_table(),
        },
    }


def analyze_trace_file(
    trace_path: str | os.PathLike[str],
    *,
    top_n: int = 50,
    micro_kernel_us: float = 10.0,
) -> Dict[str, Any]:
    path = Path(trace_path)
    trace = _read_json(path)
    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        raise ValueError(f"Unexpected traceEvents format in {path}")
    report = analyze_trace_events(events, top_n=top_n, micro_kernel_us=micro_kernel_us)
    report["meta"]["trace_path"] = str(path)
    return report


def _format_top_rows(rows: Sequence[Dict[str, Any]], *, limit: int) -> str:
    if not rows:
        return "  (none)\n"
    lines = []
    for r in rows[:limit]:
        lines.append(
            f"  {r['total_ms']:>10.3f} ms  "
            f"{r['calls']:>8}x  "
            f"avg={r.get('avg_us', 0.0):>8.2f} us  "
            f"p50={r.get('p50_us', 0.0):>8.2f} us  "
            f"p90={r.get('p90_us', 0.0):>8.2f} us  "
            f"p99={r.get('p99_us', 0.0):>8.2f} us  "
            f"{r['name']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Analyze torch.profiler chrome traces.")
    ap.add_argument("--trace", nargs="*", help="Trace file(s) to analyze (.json or .json.gz).")
    ap.add_argument(
        "--dir",
        help="Directory containing trace files. If set, analyzes all *.trace.json* files.",
    )
    ap.add_argument("--top", type=int, default=50, help="Top-N rows per table.")
    ap.add_argument(
        "--micro-us",
        type=float,
        default=10.0,
        help="Threshold (us) for 'micro-kernel swarm' classification.",
    )
    ap.add_argument(
        "--out-json",
        default=None,
        help="If set, write full analysis JSON to this path.",
    )
    ap.add_argument(
        "--print-limit",
        type=int,
        default=30,
        help="Limit printed rows per section (JSON still contains --top rows).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    trace_paths: List[Path] = []
    if args.dir:
        d = Path(args.dir)
        trace_paths.extend(sorted(d.glob("*.trace.json*"), key=lambda p: p.stat().st_mtime))
    if args.trace:
        trace_paths.extend(Path(p) for p in args.trace)

    # De-dup, keep stable order
    seen = set()
    unique_paths = []
    for p in trace_paths:
        if str(p) in seen:
            continue
        seen.add(str(p))
        unique_paths.append(p)
    trace_paths = unique_paths

    if not trace_paths:
        ap.error("Provide --trace FILE... or --dir DIR")

    all_reports = [analyze_trace_file(p, top_n=args.top, micro_kernel_us=args.micro_us) for p in trace_paths]

    for rep in all_reports:
        meta = rep["meta"]
        print("=" * 100)
        print(f"Trace: {meta.get('trace_path')}")
        print(f"Events: {meta.get('events_total')} | Wall: {meta.get('wall_ms'):.3f} ms")
        print("-" * 100)
        print("Top CUDA kernels (by total GPU time):")
        print(_format_top_rows(rep["top"]["cuda_kernels"], limit=args.print_limit), end="")
        print("-" * 100)
        print("Top CPU ops (by total CPU time):")
        print(_format_top_rows(rep["top"]["cpu_ops"], limit=args.print_limit), end="")
        print("-" * 100)
        print("Top CUDA runtime calls (memcpy/launch/sync) (by total CPU time):")
        print(_format_top_rows(rep["top"]["cuda_runtime"], limit=args.print_limit), end="")
        print("-" * 100)
        print(f"Micro-kernel swarm (CUDA kernels <= {args.micro_us} us) — ranked by call count:")
        mk_rows = rep["top"]["micro_kernels_by_count"]
        if not mk_rows:
            print("  (none)")
        else:
            for r in mk_rows[: args.print_limit]:
                print(
                    f"  {r['calls']:>10}x  {r['total_ms']:>10.3f} ms  "
                    f"avg={r['avg_us']:>7.2f} us  max={r['max_us']:>7.2f} us  {r['name']}"
                )
        print()

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"reports": all_reports}, f, indent=2)
        print(f"Wrote JSON report: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


