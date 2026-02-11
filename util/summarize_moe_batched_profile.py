#!/usr/bin/env python3
"""
Summarize [TERNARY MoE BATCHED PROFILE] stage timing lines from server logs.

Example:
  python util/summarize_moe_batched_profile.py \
    --log /tmp/server.log \
    --output-dir /tmp/moe_profile_summary
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median
from typing import Dict, List


PROFILE_RE = re.compile(
    r"TERNARY MoE BATCHED PROFILE\] calls=(\d+) "
    r"avg_ms\{gateup=([0-9.]+) silu_mul=([0-9.]+) down=([0-9.]+) combine=([0-9.]+) total=([0-9.]+)\}"
)


def stage_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    s = sorted(values)
    p95_idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "p95": float(s[p95_idx]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ternary batched MoE stage profile logs.")
    parser.add_argument("--log", required=True, help="Path to server log file.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for summary outputs (defaults to log directory).",
    )
    parser.add_argument(
        "--calls-filter",
        type=int,
        default=1,
        help="Only include profile lines with this calls value.",
    )
    parser.add_argument(
        "--max-total-ms",
        type=float,
        default=2.0,
        help="Ignore rows with total_ms above this threshold (outlier filter).",
    )
    args = parser.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"log file not found: {log_path}")

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(errors="ignore")
    rows = []
    for m in PROFILE_RE.finditer(text):
        rows.append(
            {
                "calls": int(m.group(1)),
                "gateup_ms": float(m.group(2)),
                "silu_mul_ms": float(m.group(3)),
                "down_ms": float(m.group(4)),
                "combine_ms": float(m.group(5)),
                "total_ms": float(m.group(6)),
            }
        )

    filtered = [r for r in rows if r["calls"] == args.calls_filter and r["total_ms"] <= args.max_total_ms]

    summary: Dict[str, object] = {
        "log_path": str(log_path),
        "profile_lines_total": len(rows),
        "calls_filter": args.calls_filter,
        "max_total_ms": args.max_total_ms,
        "filtered_lines": len(filtered),
    }

    for stage in ("gateup_ms", "silu_mul_ms", "down_ms", "combine_ms", "total_ms"):
        summary[stage] = stage_stats([r[stage] for r in filtered])

    total_mean = float(summary["total_ms"]["mean"]) if filtered else 0.0
    if total_mean > 0:
        for stage in ("gateup_ms", "silu_mul_ms", "down_ms", "combine_ms"):
            share = float(summary[stage]["mean"]) / total_mean * 100.0
            summary[stage.replace("_ms", "_share_pct")] = share

    json_path = out_dir / "moe_batched_profile_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# MoE Batched Stage Profile Summary",
        "",
        f"- Source log: `{log_path}`",
        f"- Total profile lines: {len(rows)}",
        f"- Filter: `calls={args.calls_filter}`, `total_ms <= {args.max_total_ms}`",
        f"- Filtered lines: {len(filtered)}",
        "",
        "| Stage | mean ms | median ms | p95 ms | share of total |",
        "|---|---:|---:|---:|---:|",
    ]

    if filtered:
        for stage, name in (
            ("gateup_ms", "gateup"),
            ("silu_mul_ms", "silu_mul"),
            ("down_ms", "down"),
            ("combine_ms", "combine"),
            ("total_ms", "total"),
        ):
            st = summary[stage]
            share = (
                f"{summary.get(stage.replace('_ms', '_share_pct'), 0.0):.1f}%"
                if stage != "total_ms"
                else "100.0%"
            )
            lines.append(
                f"| {name} | {st['mean']:.4f} | {st['median']:.4f} | {st['p95']:.4f} | {share} |"
            )
    else:
        lines.append("| n/a | 0.0000 | 0.0000 | 0.0000 | 0.0% |")

    md_path = out_dir / "moe_batched_profile_summary.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

