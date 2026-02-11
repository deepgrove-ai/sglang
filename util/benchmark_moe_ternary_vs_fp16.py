#!/usr/bin/env python3
"""
Component-level benchmark: Ternary MoE vs Standard BF16 MoE kernel.

Both use the same Triton fused-expert pattern (single launch, all experts).
This gives us the true apples-to-apples comparison that matters for understanding
the DP8 throughput gap between ternary and FP16 serving.

Strategies tested:
1. bf16_triton_ternary:   unpack i2s → BF16 MMA (current default)
2. int8_triton_ternary:   quantize act → INT8, unpack i2s → INT8 MMA
3. bf16_triton_standard:  standard fused_moe_kernel with dense BF16 weights
4. fp8_triton_standard:   standard fused_moe_kernel with FP8 weight + activation

Model: Qwen3-30B-A3B MoE (hidden=2048, intermediate=768, E=128, top_k=8)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _bench(fn, warmup=5, iters=20):
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


def make_moe_data(M: int, kind: str, seed: int = 42, device="cuda"):
    """Create test data for one MoE stage (gate_up or down)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + M * 17)

    H, I, E, K = HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_EXPERTS, TOP_K

    if kind == "gate_up":
        K_in = H
        N_out = 2 * I
        a = torch.randn(M, K_in, generator=g, dtype=torch.bfloat16).to(device)
    else:
        K_in = I
        N_out = H
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

    # Dense BF16 weights (for standard kernel comparison)
    w_dense_bf16 = (ternary * alpha.unsqueeze(1)).to(torch.bfloat16).to(device)

    # Dense BF16 in layout [E, N, K] — standard kernel expects [E, K, N] or [E, N, K]
    # Standard fused_moe_kernel expects B shape [E, N, K] where the matmul is A[m,K] @ B[E,N,K].T -> [m, N]
    # Actually: invoke_fused_moe_kernel does: A @ B -> C, where B is [E, K, N] (so it computes A[m,K] @ B[K, N])
    # Let me check the standard kernel's weight convention...
    # Standard kernel: B = w1 = [E, 2*I, K]  (same as ternary: [E, N, K])
    # It accesses: b_ptr + expert * stride_be + (k // block) * stride_bk + n_offset * stride_bn

    return {
        "a": a,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w_packed": w_packed,
        "alpha": alpha.to(device),
        "w_dense_bf16": w_dense_bf16,  # [E, N, K]
        "K_in": K_in,
        "N_out": N_out,
        "kind": kind,
        "M": M,
    }


def bench_ternary_triton(data, warmup=5, iters=20):
    """BF16 Triton ternary kernel (current default)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
        get_default_ternary_moe_config,
        invoke_fused_moe_ternary_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w_packed = data["w_packed"]
    alpha = data["alpha"]

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
            a, w_packed, alpha, out,
            data["topk_weights"], data["topk_ids"],
            sorted_ids, expert_ids, num_post,
            mul_routed, runtime_top_k, config=cfg,
        )
    return _bench(run, warmup=warmup, iters=iters)


def bench_int8_triton_ternary(data, warmup=5, iters=20):
    """INT8 Triton ternary kernel."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_ternary_kernel import (
        get_default_ternary_moe_int8_config,
        invoke_fused_moe_ternary_int8_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size
    from sglang.srt.layers.quantization.ternary import quantize_activation_int8_global

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w_packed = data["w_packed"]
    alpha = data["alpha"]

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

    # Pre-quantize activation
    a_i8, act_scale = quantize_activation_int8_global(a)
    alpha_i8 = (alpha * act_scale).to(torch.float32).contiguous()

    def run():
        invoke_fused_moe_ternary_int8_kernel(
            a_i8, w_packed, alpha_i8, out,
            data["topk_weights"], data["topk_ids"],
            sorted_ids, expert_ids, num_post,
            mul_routed, runtime_top_k, config=cfg,
        )
    return _bench(run, warmup=warmup, iters=iters)


def bench_standard_bf16_triton(data, warmup=5, iters=20):
    """Standard BF16 Triton fused_moe_kernel with dense BF16 weights (what FP16 models use)."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    # Standard kernel expects weights [E, N, K] in BF16
    w = data["w_dense_bf16"]

    # Standard kernel: C is [M*topk, N] or similar
    if kind == "gate_up":
        mul_routed = False
        runtime_top_k = K_top
        out_shape = (M, K_top, data["N_out"])
    else:
        mul_routed = True
        runtime_top_k = 1
        out_shape = (M * K_top, 1, data["N_out"])
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=a.device)

    # Config for standard kernel
    BLOCK_SIZE_M = 64
    sorted_ids, expert_ids, num_post = moe_align_block_size(data["topk_ids"], BLOCK_SIZE_M, NUM_EXPERTS)

    config = {
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }

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
    return _bench(run, warmup=warmup, iters=iters)


def bench_standard_fp8_triton(data, warmup=5, iters=20):
    """Standard FP8 Triton fused_moe_kernel with pre-materialized FP8 weights."""
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import moe_align_block_size

    M, K_top = data["M"], TOP_K
    kind = data["kind"]
    a = data["a"]
    w_bf16 = data["w_dense_bf16"]  # [E, N, K]

    # Quantize weights to FP8 per-tensor
    E = w_bf16.shape[0]
    w_scale = w_bf16.float().abs().reshape(E, -1).amax(dim=1) / 448.0
    w_scale = w_scale.clamp(min=1e-12)
    w_fp8 = torch.zeros_like(w_bf16, dtype=torch.float8_e4m3fn)
    for e in range(E):
        w_fp8[e] = (w_bf16[e].float() / w_scale[e]).to(torch.float8_e4m3fn)
    # B_scale per-channel: [E, 1, 1] for per-tensor per-expert
    b_scale = w_scale.reshape(E, 1, 1).expand(E, w_bf16.shape[1], 1).contiguous()

    if kind == "gate_up":
        mul_routed = False
        runtime_top_k = K_top
        out_shape = (M, K_top, data["N_out"])
    else:
        mul_routed = True
        runtime_top_k = 1
        out_shape = (M * K_top, 1, data["N_out"])
    out = torch.empty(out_shape, dtype=torch.bfloat16, device=a.device)

    BLOCK_SIZE_M = 64
    sorted_ids, expert_ids, num_post = moe_align_block_size(data["topk_ids"], BLOCK_SIZE_M, NUM_EXPERTS)

    config = {
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }

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
    return _bench(run, warmup=warmup, iters=iters)


def main():
    parser = argparse.ArgumentParser(description="Ternary vs FP16 MoE kernel benchmark")
    parser.add_argument("--m-values", default="1,4,8,16,32,64,128,256", help="Token counts")
    parser.add_argument("--kinds", default="gate_up,down", help="Stages")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output-dir", default="/tmp/moe_ternary_vs_fp16")
    args = parser.parse_args()

    m_values = [int(x) for x in args.m_values.split(",")]
    kinds = args.kinds.split(",")

    strategies = {
        "bf16_ternary_triton": bench_ternary_triton,
        "int8_ternary_triton": bench_int8_triton_ternary,
        "bf16_standard_triton": bench_standard_bf16_triton,
        "fp8_standard_triton": bench_standard_fp8_triton,
    }

    results = []
    print("=" * 80)
    print("MoE Kernel Comparison: Ternary vs Standard (FP16 baseline)")
    print("=" * 80)
    print(f"Model: hidden={HIDDEN_SIZE}, inter={INTERMEDIATE_SIZE}, E={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"M values: {m_values}")
    print(f"Strategies: {list(strategies.keys())}")
    print("=" * 80)

    for M in m_values:
        for kind in kinds:
            data = make_moe_data(M, kind, seed=42)
            print(f"\nM={M:>4} {kind:>10}:")

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

    # Summary: speedup vs bf16_standard_triton (what FP16 models use)
    print("\n" + "=" * 80)
    print("SPEEDUP vs bf16_standard_triton (FP16 baseline)")
    print("=" * 80)
    fp16_baseline = {}
    for r in results:
        if r["strategy"] == "bf16_standard_triton" and r["latency_ms"] is not None:
            fp16_baseline[(r["M"], r["kind"])] = r["latency_ms"]

    for r in results:
        if r["latency_ms"] is None:
            continue
        key = (r["M"], r["kind"])
        if key in fp16_baseline and fp16_baseline[key] > 0:
            speedup = fp16_baseline[key] / r["latency_ms"]
            marker = " ← TERNARY WINS" if speedup > 1.0 and "ternary" in r["strategy"] else ""
            marker = " ← FP16 WINS" if speedup < 1.0 and "ternary" in r["strategy"] else marker
            print(f"  M={r['M']:>4} {r['kind']:>10} {r['strategy']:>30}: {speedup:.3f}x  ({r['latency_ms']:.4f} ms){marker}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
