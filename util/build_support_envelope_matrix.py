#!/usr/bin/env python3
"""
Generate a corpus-backed support-envelope worksheet for ternary kernel strategy.

This tool is intentionally conservative:
- it records what is explicitly documented in local wafer corpora,
- it marks undocumented combinations as experimental,
- it writes citation file paths for every strategy row.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_CUDA_CORPUS_ROOT = Path("/home/ubuntu/.cache/wafer/corpora/cuda")
DEFAULT_CUTLASS_CORPUS_ROOT = Path("/home/ubuntu/.cache/wafer/corpora/cutlass")


@dataclass(frozen=True)
class ShapeFamily:
    name: str
    nk_shapes: str
    notes: str


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    kernel_path: str
    m_values: str
    arch_target: str
    dtype_route: str
    documented_support: str
    support_state: str
    citation_keys: Sequence[str]
    applies_to_families: Sequence[str]
    notes: str


SHAPE_FAMILIES: List[ShapeFamily] = [
    ShapeFamily(
        name="attention_proj",
        nk_shapes="5120x2048;2048x4096;2048x2048",
        notes="Dominant attention projection linear families.",
    ),
    ShapeFamily(
        name="moe_gate_up",
        nk_shapes="1536x2048;768x2048",
        notes="MoE gate/up-proj shape families seen in current model stack.",
    ),
    ShapeFamily(
        name="moe_down",
        nk_shapes="2048x768",
        notes="MoE down-proj family.",
    ),
]


CUTLASS_CITATIONS: Dict[str, str] = {
    "cutlass_functionality": "cutlass/latest/media/docs/cpp/functionality.md",
    "cutlass_fundamental_types": "cutlass/latest/media/docs/cpp/fundamental_types.md",
    "cutlass_gemm_api_3x": "cutlass/latest/media/docs/cpp/gemm_api_3x.md",
    "cutlass_blackwell_functionality": "cutlass/latest/media/docs/cpp/blackwell_functionality.md",
}

CUDA_CITATIONS: Dict[str, str] = {
    "cuda_intro_to_cpp": "cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.md",
    "cuda_advanced_kernel_programming": (
        "cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.md"
    ),
    "cuda_cluster_launch_control": (
        "cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.md"
    ),
}


STRATEGIES: List[Strategy] = [
    Strategy(
        strategy_id="h100_decode_m1_bitnet_megafused",
        kernel_path="src/kernels/gemv.cu (bitlinear_*_megafused, M=1)",
        m_values="1",
        arch_target="sm90 (H100)",
        dtype_route="i2s/int2 weights + bf16/fp8 activation path",
        documented_support=(
            "CUTLASS functionality tables explicitly list TensorOp int8/int4 operand routes and "
            "layout/alignment requirements; no int2 TensorOp type is listed."
        ),
        support_state="documented-primitives-needs-int2-unpack",
        citation_keys=("cutlass_functionality", "cutlass_fundamental_types"),
        applies_to_families=("attention_proj", "moe_gate_up", "moe_down"),
        notes=(
            "Treat direct int2 TensorOp assumptions as unsupported; prefer in-mainloop unpack to a "
            "documented operand type."
        ),
    ),
    Strategy(
        strategy_id="h100_mgt1_unpack_cutlass_mainloop",
        kernel_path="planned SM90 unpack+CUTLASS M>1 path (prefill/batched decode)",
        m_values="2,4,8,16,32,64",
        arch_target="sm90 (H100)",
        dtype_route="i2s/int2 unpack inside mainloop to int4/int8/fp16 fragments",
        documented_support=(
            "CUTLASS 3.x GEMM API documents Hopper warpspecialized/persistent scheduling and "
            "cluster-tiling abstractions suitable for architecture-aware M>1 kernels."
        ),
        support_state="documented-primitives-implementation-pending",
        citation_keys=("cutlass_gemm_api_3x", "cutlass_functionality"),
        applies_to_families=("attention_proj", "moe_gate_up", "moe_down"),
        notes="Gate enablement on per-shape correctness and long-run fallback ceilings.",
    ),
    Strategy(
        strategy_id="h100_moe_batched_triton_path",
        kernel_path="fused_moe_ternary_kernel.py (batched Triton path)",
        m_values="2,4,8,16,32,64",
        arch_target="sm90 (H100)",
        dtype_route="bf16/int8 mma variants in batched MoE kernels",
        documented_support=(
            "CUDA advanced kernel programming documents async execution/barrier rules; CUTLASS GEMM "
            "API documents persistent scheduler concepts relevant to batched work partitioning."
        ),
        support_state="documented-runtime-primitives-runtime-gating-pending",
        citation_keys=("cuda_advanced_kernel_programming", "cutlass_gemm_api_3x"),
        applies_to_families=("moe_gate_up", "moe_down"),
        notes=(
            "Primary blocker today is runtime readiness/gating (meta_not_ready), not lack of "
            "documented GPU execution primitives."
        ),
    ),
    Strategy(
        strategy_id="sm100_tcgen05_i2s_cutlass",
        kernel_path="src/kernels/i2s_cutlass_fused_mixed_sm100.cu",
        m_values="1,2,4,8,16,32,64",
        arch_target="sm100 (Blackwell)",
        dtype_route="fused tcgen05 path with split-k/stream-k variants",
        documented_support=(
            "CUTLASS Blackwell functionality docs cover SM100 capabilities; CUDA cluster launch "
            "control is explicitly introduced for compute capability 10.0 (Blackwell)."
        ),
        support_state="documented-sm100-only",
        citation_keys=("cutlass_blackwell_functionality", "cuda_cluster_launch_control"),
        applies_to_families=("attention_proj", "moe_gate_up", "moe_down"),
        notes="Do not assume parity on H100 without separate SM90 validation.",
    ),
]


def resolve_citations(
    cuda_root: Path, cutlass_root: Path
) -> tuple[Dict[str, Path], Dict[str, Path]]:
    resolved: Dict[str, Path] = {}
    missing: Dict[str, Path] = {}

    for key, rel in CUTLASS_CITATIONS.items():
        path = (cutlass_root / rel).resolve()
        resolved[key] = path
        if not path.exists():
            missing[key] = path

    for key, rel in CUDA_CITATIONS.items():
        path = (cuda_root / rel).resolve()
        resolved[key] = path
        if not path.exists():
            missing[key] = path

    return resolved, missing


def build_rows(citations: Dict[str, Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    family_by_name = {f.name: f for f in SHAPE_FAMILIES}

    for strategy in STRATEGIES:
        for family_name in strategy.applies_to_families:
            family = family_by_name[family_name]
            citation_files = ";".join(str(citations[k]) for k in strategy.citation_keys)
            rows.append(
                {
                    "shape_family": family.name,
                    "nk_shapes": family.nk_shapes,
                    "m_values": strategy.m_values,
                    "arch_target": strategy.arch_target,
                    "strategy_id": strategy.strategy_id,
                    "kernel_path": strategy.kernel_path,
                    "dtype_route": strategy.dtype_route,
                    "documented_support": strategy.documented_support,
                    "support_state": strategy.support_state,
                    "citation_files": citation_files,
                    "shape_notes": family.notes,
                    "strategy_notes": strategy.notes,
                }
            )
    return rows


def write_csv(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shape_family",
        "nk_shapes",
        "m_values",
        "arch_target",
        "strategy_id",
        "kernel_path",
        "dtype_route",
        "documented_support",
        "support_state",
        "citation_files",
        "shape_notes",
        "strategy_notes",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    rows: List[Dict[str, str]],
    output_md: Path,
    output_csv: Path,
    cuda_root: Path,
    cutlass_root: Path,
    missing_citations: Dict[str, Path],
) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).isoformat()

    lines: List[str] = []
    lines.append("# Ternary Support Envelope Worksheet")
    lines.append("")
    lines.append(f"- Generated: `{ts}`")
    lines.append(f"- CUDA corpus root: `{cuda_root}`")
    lines.append(f"- CUTLASS corpus root: `{cutlass_root}`")
    lines.append(f"- CSV artifact: `{output_csv}`")
    lines.append("")

    if missing_citations:
        lines.append("## Missing citations")
        lines.append("")
        for key, path in sorted(missing_citations.items()):
            lines.append(f"- `{key}` -> `{path}`")
        lines.append("")

    lines.append("## Matrix (condensed)")
    lines.append("")
    lines.append(
        "| shape_family | nk_shapes | m_values | arch_target | strategy_id | support_state |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['shape_family']} | {row['nk_shapes']} | {row['m_values']} | "
            f"{row['arch_target']} | {row['strategy_id']} | {row['support_state']} |"
        )
    lines.append("")
    lines.append(
        "Use the CSV for full notes/citations and keep each benchmark claim tied to one or more "
        "`citation_files` entries."
    )
    lines.append("")

    output_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a corpus-backed shape x M x arch strategy worksheet for ternary kernel planning."
        )
    )
    parser.add_argument(
        "--cuda-corpus-root",
        default=str(DEFAULT_CUDA_CORPUS_ROOT),
        help=f"CUDA corpus root (default: {DEFAULT_CUDA_CORPUS_ROOT})",
    )
    parser.add_argument(
        "--cutlass-corpus-root",
        default=str(DEFAULT_CUTLASS_CORPUS_ROOT),
        help=f"CUTLASS corpus root (default: {DEFAULT_CUTLASS_CORPUS_ROOT})",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path for the worksheet.",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional output markdown path (defaults to output-csv with .md suffix).",
    )
    parser.add_argument(
        "--allow-missing-citations",
        action="store_true",
        help="Continue even if one or more citation files are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cuda_root = Path(args.cuda_corpus_root).expanduser().resolve()
    cutlass_root = Path(args.cutlass_corpus_root).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_md = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md
        else output_csv.with_suffix(".md")
    )

    citations, missing = resolve_citations(cuda_root, cutlass_root)
    if missing and not args.allow_missing_citations:
        print("Error: missing citation files (rerun with --allow-missing-citations to continue).")
        for key, path in sorted(missing.items()):
            print(f"  - {key}: {path}")
        return 1

    rows = build_rows(citations)
    write_csv(rows, output_csv)
    write_markdown(rows, output_md, output_csv, cuda_root, cutlass_root, missing)

    print(f"Wrote support-envelope CSV: {output_csv}")
    print(f"Wrote support-envelope markdown: {output_md}")
    if missing:
        print(f"Warning: {len(missing)} citation file(s) were missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

