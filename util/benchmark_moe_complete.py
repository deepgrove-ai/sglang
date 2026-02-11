#!/usr/bin/env python3
"""
Definitive MoE kernel component benchmark: Ternary vs FP16 Triton vs cuBLAS.

Tests the user's hypothesis: "Triton FP16 might be slower than cuBLAS FP16."

Strategies:
  1. bf16_ternary_triton   — unpack i2s→BF16 MMA (current ternary path)
  2. int8_ternary_triton   — quantize act→INT8, unpack i2s→INT8 MMA
  3. bf16_triton_h100      — standard fused_moe_kernel with H100-tuned configs
  4. cublas_bf16_permuted   — per-expert cuBLAS GEMM with explicit permutation

Model: Qwen3-30B-A3B MoE (hidden=2048, intermediate=768, E=128, top_k=8)

Key insight: Qwen3 MoE has NO shared expert. All MLP compute is in MoE.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

# ─── model constants ───────────────────────────────────────────────────────
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 768
NUM_EXPERTS = 128
TOP_K = 8


def _sync():
    torch.cuda.synchronize()


def _bench(fn, warmup=8, iters=30):
    """Benchmark a callable, return median ms."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def _load_h100_triton_config(N: int, M: int) -> dict:
    """Load H100-tuned config for standard FP16 Triton fused_moe kernel."""
    config_path = (
        Path(__file__).resolve().parent.parent
        / "python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0"
        / f"E={NUM_EXPERTS},N={N},device_name=NVIDIA_H100_80GB_HBM3.json"
    )
    if not config_path.exists():
        # Fallback to generic config
        return {
            "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3,
        }
    with open(config_path) as f:
        all_cfgs = json.load(f)
    
    # Find the best matching M key (pick the largest M key ≤ our M)
    m_keys = sorted(int(k) for k in all_cfgs.keys())
    idx = bisect.bisect_right(m_keys, M)
    if idx == 0:
        chosen_m = m_keys[0]
    else:
        chosen_m = m_keys[idx - 1]
    return all_cfgs[str(chosen_m)]


def make_moe_data(M: int, kind: str, seed: int = 42, device="cuda"):
    """Create test data for one MoE stage (gate_up or down)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + M * 17)

    H, I, E, K = HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS, TOP_K

    if kind == "gate_up":
        K_in = H
        N_out = 2 * I  # gate + up fused
        a = torch.randn(M, K_in, generator=g, dtype=torch.bfloat16).to(device)
    else:  # down
        K_in = I
        N_out = H
        # After gate_up + activation, each of M tokens has top_k expert results
        a = torch.randn(M * K, K_in, generator=g, dtype=torch.bfloat16).to(device)

    topk_ids = torch.randint(0, E, (M, K), generator=g, dtype=torch.int32).to(device)
    tw = torch.rand(M, K, generator=g, dtype=torch.float32)
    topk_weights = (tw / tw.sum(dim=1, keepdim=True)).to(device)

    # Ternary weights: random {-1, 0, 1}
    ternary = torch.randint(-1, 2, (E, N_out, K_in), generator=g, dtype=torch.float32)
    alpha = torch.empty(E, K_in, dtype=torch.float32).uniform_(0.3, 1.2, generator=g)

    # Pack ternary
    encoded = (ternary + 1).to(torch.uint8)
    packed = encoded.reshape(E, N_out, K_in // 4, 4)
    packed = packed[..., 0] | (packed[..., 1] << 2) | (packed[..., 2] << 4) | (packed[..., 3] << 6)
    w_packed = packed.to(device)

    # Dense BF16 weights for standard kernels [E, N, K]
    w_dense_bf16 = (ternary * alpha.unsqueeze(1)).to(torch.bfloat16).to(device)

    return {
        "a": a,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w_packed": w_packed,
        "alpha": alpha.to(device),
        "w_dense_bf16": w_dense_bf16,
        "K_in": K_in,
        "N_out": N_out,
        "kind": kind,
        "M": M,
    }


# ─── Strategy 1: BF16 Ternary Triton ───────────────────────────────────────

def bench_ternary_triton(data, warmup=8, iters=30):
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
        get_default_ternary_moe_config,
        invoke_fused_moe_ternary_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]

    if kind == "gate_up":
        mul_routed = False
        runtime_top_k = K_top
        out = torch.empty(M, K_top, data["N_out"], dtype=torch.bfloat16, device=a.device)
    else:
        mul_routed = True
        runtime_top_k = 1
        out = torch.empty(M * K_top, 1, data["N_out"], dtype=torch.bfloat16, device=a.device)

    cfg = get_default_ternary_moe_config(kind, M)
    sorted_ids, expert_ids, num_post = moe_align_block_size(data["topk_ids"], cfg["BLOCK_SIZE_M"], NUM_EXPERTS)

    def run():
        invoke_fused_moe_ternary_kernel(
            a, data["w_packed"], data["alpha"], out,
            data["topk_weights"], data["topk_ids"],
            sorted_ids, expert_ids, num_post,
            mul_routed, runtime_top_k, config=cfg,
        )
    return _bench(run, warmup, iters)


# ─── Strategy 2: INT8 Ternary Triton ───────────────────────────────────────

def bench_int8_ternary_triton(data, warmup=8, iters=30):
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
        get_default_ternary_moe_int8_config,
        invoke_fused_moe_ternary_int8_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size
    from sglang.srt.layers.quantization.ternary import quantize_activation_int8_global

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]

    if kind == "gate_up":
        mul_routed = False
        runtime_top_k = K_top
        out = torch.empty(M, K_top, data["N_out"], dtype=torch.bfloat16, device=a.device)
    else:
        mul_routed = True
        runtime_top_k = 1
        out = torch.empty(M * K_top, 1, data["N_out"], dtype=torch.bfloat16, device=a.device)

    cfg = get_default_ternary_moe_int8_config(kind, M)
    sorted_ids, expert_ids, num_post = moe_align_block_size(data["topk_ids"], cfg["BLOCK_SIZE_M"], NUM_EXPERTS)

    a_i8, act_scale = quantize_activation_int8_global(a)
    alpha_i8 = (data["alpha"] * act_scale).to(torch.float32).contiguous()

    def run():
        invoke_fused_moe_ternary_int8_kernel(
            a_i8, data["w_packed"], alpha_i8, out,
            data["topk_weights"], data["topk_ids"],
            sorted_ids, expert_ids, num_post,
            mul_routed, runtime_top_k, config=cfg,
        )
    return _bench(run, warmup, iters)


# ─── Strategy 3: FP16 Triton with H100-tuned configs ──────────────────────

def bench_standard_bf16_triton_h100(data, warmup=8, iters=30):
    """Standard BF16 Triton fused_moe with ACTUAL H100-tuned configs (what FP16 server uses)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w = data["w_dense_bf16"]
    N = data["N_out"]

    # Load H100-tuned config (per-N, per-M)
    config = _load_h100_triton_config(N, a.shape[0])
    BLOCK_M = config["BLOCK_SIZE_M"]

    if kind == "gate_up":
        mul_routed = False
        runtime_top_k = K_top
        out = torch.empty(M, K_top, N, dtype=torch.bfloat16, device=a.device)
    else:
        mul_routed = True
        runtime_top_k = 1
        out = torch.empty(M * K_top, 1, N, dtype=torch.bfloat16, device=a.device)

    sorted_ids, expert_ids, num_post = moe_align_block_size(data["topk_ids"], BLOCK_M, NUM_EXPERTS)

    def run():
        invoke_fused_moe_kernel(
            a, w, None, out,
            None, None, None,
            data["topk_weights"], data["topk_ids"],
            sorted_ids, expert_ids, num_post,
            mul_routed, runtime_top_k,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            no_combine=True,
        )
    return _bench(run, warmup, iters)


# ─── Strategy 4: cuBLAS BF16 per-expert permuted GEMM ─────────────────────

def bench_cublas_bf16_permuted(data, warmup=8, iters=30):
    """
    cuBLAS BF16 GEMM with explicit token permutation.
    
    This simulates what a cuBLAS-backed MoE would look like:
    1. Sort tokens by expert
    2. For each expert, call torch.mm (cuBLAS)
    3. Scatter results back
    
    This is the gold standard for dense GEMM performance.
    """
    M_orig, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w = data["w_dense_bf16"]  # [E, N, K]
    N = data["N_out"]
    K_in = data["K_in"]
    topk_ids = data["topk_ids"]
    topk_weights = data["topk_weights"]

    if kind == "gate_up":
        # Flatten: each token sends to top_k experts
        M_total = M_orig * K_top
        a_expanded = a.unsqueeze(1).expand(-1, K_top, -1).reshape(M_total, K_in)
        flat_expert_ids = topk_ids.reshape(-1)  # [M_total]
    else:
        M_total = M_orig * K_top
        a_expanded = a  # already [M*topk, K_in]
        flat_expert_ids = topk_ids.reshape(-1)

    out = torch.empty(M_total, N, dtype=torch.bfloat16, device=a.device)

    # Pre-compute sort order and per-expert slices
    sorted_order = flat_expert_ids.argsort()
    sorted_expert_ids = flat_expert_ids[sorted_order]
    a_sorted = a_expanded[sorted_order]

    # Find expert boundaries
    expert_starts = []
    expert_ends = []
    for e in range(NUM_EXPERTS):
        mask = (sorted_expert_ids == e)
        if mask.any():
            indices = mask.nonzero(as_tuple=True)[0]
            expert_starts.append((e, int(indices[0]), int(indices[-1]) + 1))
    
    # Pre-transpose weights: [E, K, N] for mm
    w_t = w.transpose(1, 2).contiguous()  # [E, K, N]

    def run():
        out_sorted = torch.empty_like(out)
        for e, start, end in expert_starts:
            if start < end:
                # torch.mm dispatches to cuBLAS
                torch.mm(a_sorted[start:end], w_t[e], out=out_sorted[start:end])
        # Scatter back
        out[sorted_order] = out_sorted

    return _bench(run, warmup, iters)


# ─── Strategy 5: cuBLAS BF16 batched (torch.bmm with padding) ─────────────

def bench_cublas_bf16_batched(data, warmup=8, iters=30):
    """
    cuBLAS batched GEMM: pad all experts to the same M and use torch.bmm.
    Higher GPU utilization than per-expert GEMM at cost of wasted compute.
    """
    M_orig, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w = data["w_dense_bf16"]  # [E, N, K]
    N = data["N_out"]
    K_in = data["K_in"]
    topk_ids = data["topk_ids"]

    if kind == "gate_up":
        M_total = M_orig * K_top
        a_expanded = a.unsqueeze(1).expand(-1, K_top, -1).reshape(M_total, K_in)
        flat_expert_ids = topk_ids.reshape(-1)
    else:
        M_total = M_orig * K_top
        a_expanded = a
        flat_expert_ids = topk_ids.reshape(-1)

    out = torch.empty(M_total, N, dtype=torch.bfloat16, device=a.device)

    # Pre-compute per-expert token counts
    expert_counts = torch.zeros(NUM_EXPERTS, dtype=torch.long, device=a.device)
    for e in range(NUM_EXPERTS):
        expert_counts[e] = (flat_expert_ids == e).sum()
    
    max_count = int(expert_counts.max().item())
    if max_count == 0:
        max_count = 1

    # Pre-allocate padded batches
    a_padded = torch.zeros(NUM_EXPERTS, max_count, K_in, dtype=torch.bfloat16, device=a.device)
    sorted_order = flat_expert_ids.argsort()
    sorted_expert_ids = flat_expert_ids[sorted_order]
    a_sorted = a_expanded[sorted_order]
    
    # Fill padded input
    offset = 0
    for e in range(NUM_EXPERTS):
        cnt = int(expert_counts[e].item())
        if cnt > 0:
            a_padded[e, :cnt] = a_sorted[offset:offset + cnt]
            offset += cnt

    # w_t: [E, K, N]
    w_t = w.transpose(1, 2).contiguous()

    def run():
        # Batched matmul: [E, max_count, K] @ [E, K, N] -> [E, max_count, N]
        out_batched = torch.bmm(a_padded, w_t)
        # Scatter back (simplified - just measure the bmm)
    
    return _bench(run, warmup, iters)


# ─── Strategy 6: FP8 Triton with pre-materialized weights ─────────────────

def bench_fp8_triton(data, warmup=8, iters=30):
    """FP8 pre-materialized ternary weights with FP8 MoE Triton kernel."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w_bf16 = data["w_dense_bf16"]
    N = data["N_out"]

    E = w_bf16.shape[0]
    # Per-expert FP8 quantization
    w_scale = w_bf16.float().abs().reshape(E, -1).amax(dim=1) / 448.0
    w_scale = w_scale.clamp(min=1e-12)
    w_fp8 = torch.zeros_like(w_bf16, dtype=torch.float8_e4m3fn)
    for e in range(E):
        w_fp8[e] = (w_bf16[e].float() / w_scale[e]).to(torch.float8_e4m3fn)
    b_scale = w_scale.reshape(E, 1, 1).expand(E, w_bf16.shape[1], 1).contiguous()

    config = _load_h100_triton_config(N, a.shape[0])
    BLOCK_M = config["BLOCK_SIZE_M"]

    if kind == "gate_up":
        mul_routed = False
        runtime_top_k = K_top
        out = torch.empty(M, K_top, N, dtype=torch.bfloat16, device=a.device)
    else:
        mul_routed = True
        runtime_top_k = 1
        out = torch.empty(M * K_top, 1, N, dtype=torch.bfloat16, device=a.device)

    sorted_ids, expert_ids, num_post = moe_align_block_size(data["topk_ids"], BLOCK_M, NUM_EXPERTS)

    def run():
        invoke_fused_moe_kernel(
            a, w_fp8, None, out,
            None, b_scale, None,
            data["topk_weights"], data["topk_ids"],
            sorted_ids, expert_ids, num_post,
            mul_routed, runtime_top_k,
            config=config,
            compute_type=tl.bfloat16,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=True,
            no_combine=True,
        )
    return _bench(run, warmup, iters)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Definitive MoE kernel benchmark")
    parser.add_argument("--m-values", default="1,4,8,16,32,64,128,256", help="Token counts")
    parser.add_argument("--kinds", default="gate_up,down", help="Stages")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output-dir", default="/tmp/moe_complete_benchmark")
    parser.add_argument("--skip", default="", help="Comma-separated strategies to skip")
    args = parser.parse_args()

    m_values = [int(x) for x in args.m_values.split(",")]
    kinds = args.kinds.split(",")
    skip = set(args.skip.split(",")) if args.skip else set()

    all_strategies = {
        "1_bf16_ternary_triton": bench_ternary_triton,
        "2_int8_ternary_triton": bench_int8_ternary_triton,
        "3_bf16_triton_h100": bench_standard_bf16_triton_h100,
        "4_cublas_bf16_permuted": bench_cublas_bf16_permuted,
        "5_cublas_bf16_batched": bench_cublas_bf16_batched,
        "6_fp8_triton_h100": bench_fp8_triton,
    }

    strategies = {k: v for k, v in all_strategies.items() if k not in skip}

    results = []
    print("=" * 90)
    print("DEFINITIVE MoE Kernel Comparison: Ternary vs FP16 Triton vs cuBLAS")
    print("=" * 90)
    print(f"Model: hidden={HIDDEN_SIZE}, inter={INTERMEDIATE_SIZE}, E={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"NOTE: Qwen3 MoE has NO shared expert. All MLP is MoE-only.")
    print(f"M values: {m_values}")
    print(f"Strategies: {list(strategies.keys())}")
    print("=" * 90)

    for M in m_values:
        for kind in kinds:
            data = make_moe_data(M, kind, seed=42)
            print(f"\n{'─'*70}")
            print(f"M={M:>4} | {kind:>10} | N={data['N_out']}, K={data['K_in']}")
            print(f"{'─'*70}")

            for name, fn in strategies.items():
                try:
                    ms = fn(data, warmup=args.warmup, iters=args.iters)
                    results.append({
                        "M": M, "kind": kind, "strategy": name,
                        "latency_ms": round(ms, 4), "status": "ok",
                    })
                    print(f"  {name:30s}: {ms:8.4f} ms")
                except Exception as e:
                    results.append({
                        "M": M, "kind": kind, "strategy": name,
                        "latency_ms": None, "status": f"error: {str(e)[:200]}",
                    })
                    print(f"  {name:30s}: ERROR - {str(e)[:120]}")

    # ─── Summary tables ────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("SUMMARY: Speedup vs bf16_triton_h100 (what FP16 server uses)")
    print("=" * 90)

    fp16_baseline = {}
    for r in results:
        if r["strategy"] == "3_bf16_triton_h100" and r["latency_ms"] is not None:
            fp16_baseline[(r["M"], r["kind"])] = r["latency_ms"]

    # Header
    strat_names = sorted(set(r["strategy"] for r in results))
    print(f"\n{'M':>4} {'kind':>10}", end="")
    for s in strat_names:
        short = s.split("_", 1)[1][:18]
        print(f"  {short:>18}", end="")
    print("   winner")
    print("-" * (16 + len(strat_names) * 20 + 10))

    for M in m_values:
        for kind in kinds:
            key = (M, kind)
            print(f"{M:>4} {kind:>10}", end="")

            best_name = None
            best_ms = float("inf")
            row_data = {}

            for s in strat_names:
                ms_val = None
                for r in results:
                    if r["M"] == M and r["kind"] == kind and r["strategy"] == s:
                        ms_val = r["latency_ms"]
                        break

                if ms_val is not None:
                    row_data[s] = ms_val
                    if ms_val < best_ms:
                        best_ms = ms_val
                        best_name = s

                    if key in fp16_baseline and fp16_baseline[key] > 0:
                        speedup = fp16_baseline[key] / ms_val
                        print(f"  {speedup:>17.3f}x", end="")
                    else:
                        print(f"  {ms_val:>16.4f}ms", end="")
                else:
                    print(f"  {'err':>18}", end="")

            if best_name:
                short = best_name.split("_", 1)[1][:20]
                print(f"   {short}")
            else:
                print()

    # ─── Key findings ──────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    # Compare ternary vs fp16 triton
    ternary_wins = 0
    fp16_wins = 0
    for M in m_values:
        for kind in kinds:
            key = (M, kind)
            ternary_ms = None
            fp16_ms = None
            cublas_ms = None
            for r in results:
                if r["M"] == M and r["kind"] == kind and r["latency_ms"] is not None:
                    if r["strategy"] == "1_bf16_ternary_triton":
                        ternary_ms = r["latency_ms"]
                    elif r["strategy"] == "3_bf16_triton_h100":
                        fp16_ms = r["latency_ms"]
                    elif r["strategy"] == "4_cublas_bf16_permuted":
                        cublas_ms = r["latency_ms"]

            if ternary_ms and fp16_ms:
                if ternary_ms < fp16_ms:
                    ternary_wins += 1
                else:
                    fp16_wins += 1

            if fp16_ms and cublas_ms:
                if cublas_ms < fp16_ms:
                    print(f"  M={M:>4} {kind:>10}: cuBLAS BEATS Triton FP16! ({cublas_ms:.4f} vs {fp16_ms:.4f} ms, {fp16_ms/cublas_ms:.2f}x)")
                else:
                    print(f"  M={M:>4} {kind:>10}: Triton FP16 beats cuBLAS ({fp16_ms:.4f} vs {cublas_ms:.4f} ms, {cublas_ms/fp16_ms:.2f}x)")

    print(f"\n  Ternary BF16 wins: {ternary_wins} / {ternary_wins + fp16_wins}")
    print(f"  FP16 Triton wins:  {fp16_wins} / {ternary_wins + fp16_wins}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
