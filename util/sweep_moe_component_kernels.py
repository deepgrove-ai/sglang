#!/usr/bin/env python3
"""
Component-level sweep for ternary MoE Triton kernels (no server required).

This script benchmarks `invoke_fused_moe_ternary_kernel` for `gate_up` / `down`
across a broad config grid, ranks candidates, and validates top configs against
the default config output.

Corpus-grounded constraints used here:
- CUTLASS functionality docs (TensorOp/int8 tables + 128b alignment guidance):
  /home/ubuntu/.cache/wafer/corpora/cutlass/cutlass/latest/media/docs/cpp/functionality.md
- CUDA WMMA/Tensor Core warp-synchronous constraints:
  /home/ubuntu/.cache/wafer/corpora/cuda/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.md
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
    get_default_ternary_moe_config,
    invoke_fused_moe_ternary_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size


@dataclass
class CaseData:
    kind: str
    m: int
    top_k: int
    num_experts: int
    k_dim: int
    n_dim: int
    a: torch.Tensor
    b_packed: torch.Tensor
    alpha: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    out: torch.Tensor


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_case_data(kind: str, m: int, top_k: int, num_experts: int, seed: int, device: torch.device) -> CaseData:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + m * 17)

    hidden_size = 2048
    intermediate_size = 768
    if kind == "gate_up":
        k_dim = hidden_size
        n_dim = 2 * intermediate_size  # 1536
        a = torch.randn((m, k_dim), generator=g, dtype=torch.bfloat16).to(device=device)
    elif kind == "down":
        k_dim = intermediate_size
        n_dim = hidden_size
        a = torch.randn((m * top_k, k_dim), generator=g, dtype=torch.bfloat16).to(device=device)
    else:
        raise ValueError(f"unsupported kind: {kind}")

    # Packed ternary weights [E, N, K//4]. Random byte payload is sufficient for
    # relative kernel-speed ranking because all configs run identical data.
    b_packed = torch.randint(
        0,
        256,
        (num_experts, n_dim, k_dim // 4),
        generator=g,
        dtype=torch.uint8,
    ).to(device=device)
    alpha = torch.empty((num_experts, k_dim), dtype=torch.float32).uniform_(0.25, 1.5, generator=g).to(device=device)
    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(m, top_k),
        generator=g,
        dtype=torch.int32,
    ).to(device=device)
    topk_weights = torch.rand((m, top_k), generator=g, dtype=torch.float32)
    topk_weights = (topk_weights / topk_weights.sum(dim=1, keepdim=True)).to(device=device)
    out = torch.empty((m, top_k, n_dim), dtype=torch.bfloat16, device=device)
    return CaseData(
        kind=kind,
        m=m,
        top_k=top_k,
        num_experts=num_experts,
        k_dim=k_dim,
        n_dim=n_dim,
        a=a,
        b_packed=b_packed,
        alpha=alpha,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        out=out,
    )


def make_config_grid(case: CaseData, seed: int, max_configs: int) -> List[Dict[str, int]]:
    # Keep grid broad but constrained by known tensor-op friendly tile families.
    bs_m_vals = [8, 16, 32]
    bs_n_vals = [64, 128, 256]
    bs_k_vals = [64, 128, 256, 512]
    group_vals = [1, 8, 16, 32, 64]
    warp_vals = [2, 4, 8]
    stage_vals = [2, 3, 4, 5]

    base = get_default_ternary_moe_config(case.kind, case.m)
    fp32_alpha = int(base.get("FP32_ALPHA", 1))

    all_cfgs: List[Dict[str, int]] = []
    for bs_m, bs_n, bs_k, g, w, s in itertools.product(
        bs_m_vals, bs_n_vals, bs_k_vals, group_vals, warp_vals, stage_vals
    ):
        if bs_k % 4 != 0:
            continue
        if bs_k > case.k_dim:
            continue
        # Prefer configs aligned with TensorOp-friendly granularities.
        if bs_k % 32 != 0:
            continue
        if bs_n > case.n_dim * 2:
            continue
        cfg = {
            "BLOCK_SIZE_M": bs_m,
            "BLOCK_SIZE_N": bs_n,
            "BLOCK_SIZE_K": bs_k,
            "GROUP_SIZE_M": g,
            "num_warps": w,
            "num_stages": s,
            "FP32_ALPHA": fp32_alpha,
        }
        all_cfgs.append(cfg)

    # Deduplicate and deterministic sample if needed.
    dedup = {
        (
            c["BLOCK_SIZE_M"],
            c["BLOCK_SIZE_N"],
            c["BLOCK_SIZE_K"],
            c["GROUP_SIZE_M"],
            c["num_warps"],
            c["num_stages"],
            c["FP32_ALPHA"],
        ): c
        for c in all_cfgs
    }
    cfgs = list(dedup.values())
    if len(cfgs) <= max_configs:
        return cfgs

    rnd = random.Random(seed)
    idxs = list(range(len(cfgs)))
    rnd.shuffle(idxs)
    return [cfgs[i] for i in idxs[:max_configs]]


def run_kernel_once(case: CaseData, cfg: Dict[str, int], out: torch.Tensor) -> None:
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        case.topk_ids, cfg["BLOCK_SIZE_M"], case.num_experts
    )
    mul_routed_weight = case.kind == "down"
    runtime_top_k = 1 if case.kind == "down" else case.top_k
    invoke_fused_moe_ternary_kernel(
        case.a,
        case.b_packed,
        case.alpha,
        out,
        case.topk_weights,
        case.topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        runtime_top_k,
        config=cfg,
    )


def bench_case(case: CaseData, cfg: Dict[str, int], warmup: int, iters: int) -> float:
    out = case.out
    for _ in range(warmup):
        run_kernel_once(case, cfg, out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_kernel_once(case, cfg, out)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iters)


def max_abs_diff(case: CaseData, cfg_a: Dict[str, int], cfg_b: Dict[str, int]) -> float:
    out_a = torch.empty_like(case.out)
    out_b = torch.empty_like(case.out)
    run_kernel_once(case, cfg_a, out_a)
    run_kernel_once(case, cfg_b, out_b)
    torch.cuda.synchronize()
    return float((out_a.float() - out_b.float()).abs().max().item())


def cfg_to_name(cfg: Dict[str, int]) -> str:
    return (
        f"m{cfg['BLOCK_SIZE_M']}_n{cfg['BLOCK_SIZE_N']}_k{cfg['BLOCK_SIZE_K']}_"
        f"g{cfg['GROUP_SIZE_M']}_w{cfg['num_warps']}_s{cfg['num_stages']}_a{cfg['FP32_ALPHA']}"
    )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k in seen:
                continue
            seen.add(k)
            fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Component-level ternary MoE Triton kernel sweep.")
    parser.add_argument("--kind", choices=["gate_up", "down"], default="gate_up")
    parser.add_argument("--m-values", default="6,14,128", help="Comma-separated token counts.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage1-max-configs", type=int, default=256)
    parser.add_argument("--stage1-warmup", type=int, default=2)
    parser.add_argument("--stage1-iters", type=int, default=6)
    parser.add_argument("--topn-stage2", type=int, default=32)
    parser.add_argument("--stage2-warmup", type=int, default=6)
    parser.add_argument("--stage2-iters", type=int, default=24)
    parser.add_argument("--output-dir", default="", help="Output directory for artifacts.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    device = torch.device("cuda")
    m_values = parse_int_list(args.m_values)
    if not m_values:
        raise ValueError("m-values is empty")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (
        Path.cwd() / f"moe_component_sweep_{args.kind}_{torch.cuda.get_device_name(0).replace(' ', '_')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        m: make_case_data(
            kind=args.kind,
            m=m,
            top_k=args.top_k,
            num_experts=args.num_experts,
            seed=args.seed,
            device=device,
        )
        for m in m_values
    }
    m_stage1 = max(m_values)
    case_stage1 = cases[m_stage1]

    baseline_cfg = get_default_ternary_moe_config(args.kind, m_stage1)
    candidates = make_config_grid(case_stage1, seed=args.seed, max_configs=args.stage1_max_configs)

    print("==============================================================")
    print("MoE Component Kernel Sweep")
    print("==============================================================")
    print(f"Kind:            {args.kind}")
    print(f"M values:        {m_values}")
    print(f"Top-k:           {args.top_k}")
    print(f"Num experts:     {args.num_experts}")
    print(f"Stage1 configs:  {len(candidates)}")
    print(f"Output dir:      {out_dir}")
    print("==============================================================")

    stage1_rows: List[Dict[str, object]] = []
    for i, cfg in enumerate(candidates, start=1):
        name = cfg_to_name(cfg)
        try:
            ms = bench_case(case_stage1, cfg, warmup=args.stage1_warmup, iters=args.stage1_iters)
            stage1_rows.append(
                {
                    "rank_hint": i,
                    "name": name,
                    "m": m_stage1,
                    "latency_ms": ms,
                    **cfg,
                }
            )
        except Exception as e:
            stage1_rows.append(
                {
                    "rank_hint": i,
                    "name": name,
                    "m": m_stage1,
                    "latency_ms": "",
                    "error": str(e)[:300],
                    **cfg,
                }
            )

    write_csv(out_dir / "stage1_results.csv", stage1_rows)
    ok_stage1 = [r for r in stage1_rows if isinstance(r.get("latency_ms"), float)]
    ok_stage1.sort(key=lambda r: float(r["latency_ms"]))
    top_cfg_rows = ok_stage1[: args.topn_stage2]

    final_rows: List[Dict[str, object]] = []
    for rank, row in enumerate(top_cfg_rows, start=1):
        cfg = {
            "BLOCK_SIZE_M": int(row["BLOCK_SIZE_M"]),
            "BLOCK_SIZE_N": int(row["BLOCK_SIZE_N"]),
            "BLOCK_SIZE_K": int(row["BLOCK_SIZE_K"]),
            "GROUP_SIZE_M": int(row["GROUP_SIZE_M"]),
            "num_warps": int(row["num_warps"]),
            "num_stages": int(row["num_stages"]),
            "FP32_ALPHA": int(row["FP32_ALPHA"]),
        }
        for m in m_values:
            case = cases[m]
            base_cfg_m = get_default_ternary_moe_config(args.kind, m)
            try:
                ms = bench_case(case, cfg, warmup=args.stage2_warmup, iters=args.stage2_iters)
                base_ms = bench_case(case, base_cfg_m, warmup=args.stage2_warmup, iters=args.stage2_iters)
                diff = max_abs_diff(case, cfg, base_cfg_m)
                speedup = (base_ms / ms) if ms > 0 else 0.0
                final_rows.append(
                    {
                        "rank": rank,
                        "name": row["name"],
                        "m": m,
                        "latency_ms": ms,
                        "baseline_latency_ms": base_ms,
                        "speedup_vs_baseline": speedup,
                        "max_abs_vs_baseline": diff,
                        **cfg,
                    }
                )
            except Exception as e:
                final_rows.append(
                    {
                        "rank": rank,
                        "name": row["name"],
                        "m": m,
                        "latency_ms": "",
                        "baseline_latency_ms": "",
                        "speedup_vs_baseline": "",
                        "max_abs_vs_baseline": "",
                        "error": str(e)[:300],
                        **cfg,
                    }
                )

    write_csv(out_dir / "final_results.csv", final_rows)

    # Aggregate by config name over m-values.
    agg: Dict[str, Dict[str, object]] = {}
    for r in final_rows:
        if not isinstance(r.get("latency_ms"), float):
            continue
        name = str(r["name"])
        rec = agg.setdefault(
            name,
            {
                "name": name,
                "count": 0,
                "latency_ms_sum": 0.0,
                "speedup_sum": 0.0,
                "max_abs_max": 0.0,
                "cfg": {
                    "BLOCK_SIZE_M": r["BLOCK_SIZE_M"],
                    "BLOCK_SIZE_N": r["BLOCK_SIZE_N"],
                    "BLOCK_SIZE_K": r["BLOCK_SIZE_K"],
                    "GROUP_SIZE_M": r["GROUP_SIZE_M"],
                    "num_warps": r["num_warps"],
                    "num_stages": r["num_stages"],
                    "FP32_ALPHA": r["FP32_ALPHA"],
                },
            },
        )
        rec["count"] = int(rec["count"]) + 1
        rec["latency_ms_sum"] = float(rec["latency_ms_sum"]) + float(r["latency_ms"])
        rec["speedup_sum"] = float(rec["speedup_sum"]) + float(r["speedup_vs_baseline"])
        rec["max_abs_max"] = max(float(rec["max_abs_max"]), float(r["max_abs_vs_baseline"]))

    summary_rows = []
    for rec in agg.values():
        cnt = int(rec["count"])
        summary_rows.append(
            {
                "name": rec["name"],
                "mean_latency_ms": float(rec["latency_ms_sum"]) / cnt,
                "mean_speedup_vs_baseline": float(rec["speedup_sum"]) / cnt,
                "max_abs_vs_baseline": float(rec["max_abs_max"]),
                **rec["cfg"],
            }
        )
    summary_rows.sort(key=lambda r: float(r["mean_latency_ms"]))
    write_csv(out_dir / "summary.csv", summary_rows)

    meta = {
        "kind": args.kind,
        "m_values": m_values,
        "top_k": args.top_k,
        "num_experts": args.num_experts,
        "stage1_max_configs": args.stage1_max_configs,
        "stage1_warmup": args.stage1_warmup,
        "stage1_iters": args.stage1_iters,
        "topn_stage2": args.topn_stage2,
        "stage2_warmup": args.stage2_warmup,
        "stage2_iters": args.stage2_iters,
        "baseline_config_m_stage1": baseline_cfg,
        "citations": {
            "cutlass_functionality": "/home/ubuntu/.cache/wafer/corpora/cutlass/cutlass/latest/media/docs/cpp/functionality.md",
            "cuda_wmma_cpp_extensions": "/home/ubuntu/.cache/wafer/corpora/cuda/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.md",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    md_lines = [
        "# MoE Component Sweep Summary",
        "",
        f"- kind: `{args.kind}`",
        f"- m_values: `{m_values}`",
        f"- stage1 configs tested: `{len(candidates)}`",
        f"- stage2 configs retested: `{len(top_cfg_rows)}`",
        f"- artifacts: `{out_dir}`",
        "",
        "## Corpus citations",
        "",
        "- `/home/ubuntu/.cache/wafer/corpora/cutlass/cutlass/latest/media/docs/cpp/functionality.md`",
        "- `/home/ubuntu/.cache/wafer/corpora/cuda/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.md`",
        "",
        "## Top configs (mean across m_values)",
        "",
        "| name | mean latency ms | mean speedup vs baseline | max abs vs baseline |",
        "|---|---:|---:|---:|",
    ]
    for r in summary_rows[:10]:
        md_lines.append(
            f"| {r['name']} | {float(r['mean_latency_ms']):.4f} | "
            f"{float(r['mean_speedup_vs_baseline']):.4f} | {float(r['max_abs_vs_baseline']):.6f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")

    print("Done.")
    print(f"  {out_dir / 'stage1_results.csv'}")
    print(f"  {out_dir / 'final_results.csv'}")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

