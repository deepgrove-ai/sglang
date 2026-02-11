#!/usr/bin/env python3
"""
Component benchmark: Dense linear layers — Ternary CUDA kernel vs cuBLAS BF16.

These are the attention projections (QKV, O) that run on EVERY token in every layer.
If FP16 server beats ternary at high concurrency, these dense linears may be the gap.

Shapes (Qwen3-30B-A3B, hidden=2048):
  QKV fused:  N=5120, K=2048   (every layer)
  O proj:     N=2048, K=4096   (every layer, TP1)
  
48 layers × 2 dense linears = 96 dense GEMMs per forward pass.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SHAPES = [
    ("qkv_proj", 5120, 2048),
    ("o_proj", 2048, 4096),
    # Also include the MoE gate_up/down for reference
    ("moe_gate_up", 1536, 2048),
    ("moe_down", 2048, 768),
]


def _sync():
    torch.cuda.synchronize()


def _bench(fn, warmup=10, iters=50):
    """Benchmark, return median ms."""
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


def bench_cublas_bf16(M, N, K, warmup=10, iters=50):
    """cuBLAS BF16 GEMM via F.linear (what FP16 server uses for dense layers)."""
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    def run():
        F.linear(x, w)

    return _bench(run, warmup, iters)


def bench_ternary_m1_megafused(M, N, K, warmup=10, iters=50):
    """Ternary M=1 megafused CUDA kernel."""
    if M != 1:
        return None  # Only for M=1

    from sglang.srt.layers.quantization import ternary as tq

    if not tq._KERNEL_CAPS.get('megafused', False):
        return None

    lib = tq.BITNET_LIB
    if lib is None:
        return None

    if (N, K) not in tq.SUPPORTED_V4_NK_SHAPES:
        return None

    # Create ternary-packed weights
    ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    alpha = torch.rand(K, dtype=torch.float32).to("cuda") * 0.5 + 0.5

    # Pack weights using BitNet packing
    from sglang.srt.layers.quantization.ternary import convert_weight_int8_to_int2

    # convert_weight_int8_to_int2 takes (weight_int8,) -> packed tensor
    ternary_i8 = ternary.to(torch.int8).to("cuda")
    try:
        packed = convert_weight_int8_to_int2(ternary_i8).contiguous()
    except Exception:
        return None
    packed_ptr = packed.data_ptr()

    x = torch.randn(1, K, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(1, N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        # API: x_ptr, weight_ptr, alpha_ptr, output_ptr, M, N, K, stream
        lib.bitlinear_bf16xint2_v4_megafused(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed_ptr),
            ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(1), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_void_p(stream),
        )

    return _bench(run, warmup, iters)


def bench_ternary_mgt1_megafused(M, N, K, warmup=10, iters=50):
    """Ternary M>1 batch megafused CUDA kernel."""
    if M <= 1:
        return None

    from sglang.srt.layers.quantization import ternary as tq

    if not tq._KERNEL_CAPS.get('batch_megafused', False):
        return None

    lib = tq.BITNET_LIB
    if lib is None:
        return None

    if (N, K) not in tq.SUPPORTED_V4_NK_SHAPES:
        return None

    # Create ternary-packed weights
    from sglang.srt.layers.quantization.ternary import convert_weight_int8_to_int2

    ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    alpha = torch.rand(K, dtype=torch.float32).to("cuda") * 0.5 + 0.5

    ternary_i8 = ternary.to(torch.int8).to("cuda")
    try:
        packed = convert_weight_int8_to_int2(ternary_i8).contiguous()
    except Exception:
        return None
    packed_ptr = packed.data_ptr()

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream

    def run():
        # API: x_ptr, alpha_ptr, weight_ptr, output_ptr, M, N, K, stream
        lib.v4_batch_megafused_v2_launch(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(alpha.data_ptr()),
            ctypes.c_void_p(packed_ptr),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_void_p(stream),
        )

    return _bench(run, warmup, iters)


def bench_ternary_fp16_fallback(M, N, K, warmup=10, iters=50):
    """Ternary FP16 fallback — unpack weights then call F.linear (same as cuBLAS)."""
    # This simulates what happens when ternary kernel can't handle the shape
    ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    alpha = torch.rand(K, dtype=torch.float32).to("cuda") * 0.5 + 0.5
    w_bf16 = (ternary.to("cuda") * alpha.unsqueeze(0)).to(torch.bfloat16)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    def run():
        F.linear(x, w_bf16)

    return _bench(run, warmup, iters)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values", default="1,2,4,8,16,32,64,128,256", help="Batch sizes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output-dir", default="/tmp/dense_linear_benchmark")
    args = parser.parse_args()

    m_values = [int(x) for x in args.m_values.split(",")]

    print("=" * 90)
    print("Dense Linear Benchmark: Ternary CUDA Kernel vs cuBLAS BF16")
    print("=" * 90)
    print(f"These run 96 times per forward pass (48 layers × 2 dense linears)")
    print(f"M values: {m_values}")
    print()

    results = []

    for name, N, K in SHAPES:
        print(f"\n{'='*70}")
        print(f"Shape: {name} (N={N}, K={K})")
        print(f"{'='*70}")

        header = f"{'M':>4}  {'cuBLAS BF16':>12}  {'Ternary M=1':>12}  {'Ternary M>1':>12}  {'FP16 Fallback':>14}  {'Winner':>20}  {'Speedup':>8}"
        print(header)
        print("-" * len(header))

        for M in m_values:
            cublas_ms = bench_cublas_bf16(M, N, K, args.warmup, args.iters)

            # Ternary paths
            if M == 1:
                ternary_ms = bench_ternary_m1_megafused(M, N, K, args.warmup, args.iters)
            else:
                ternary_ms = bench_ternary_mgt1_megafused(M, N, K, args.warmup, args.iters)

            fallback_ms = bench_ternary_fp16_fallback(M, N, K, args.warmup, args.iters)

            # Determine winner
            candidates = {"cuBLAS BF16": cublas_ms}
            if ternary_ms is not None:
                candidates["Ternary CUDA"] = ternary_ms
            candidates["FP16 Fallback"] = fallback_ms

            best_name = min(candidates, key=lambda k: candidates[k])
            best_ms = candidates[best_name]
            speedup = cublas_ms / best_ms if best_ms > 0 else 0

            t_str = f"{ternary_ms:.4f}" if ternary_ms else "N/A"
            print(f"{M:>4}  {cublas_ms:>11.4f}ms  {t_str:>12}  ", end="")
            if M > 1 and ternary_ms:
                print(f"{ternary_ms:>11.4f}ms  ", end="")
            elif M > 1:
                print(f"{'N/A':>12}  ", end="")
            else:
                print(f"{'--':>12}  ", end="")
            print(f"{fallback_ms:>13.4f}ms  {best_name:>20}  {speedup:>7.2f}x")

            results.append({
                "shape": name, "N": N, "K": K, "M": M,
                "cublas_ms": round(cublas_ms, 4),
                "ternary_ms": round(ternary_ms, 4) if ternary_ms else None,
                "fallback_ms": round(fallback_ms, 4),
                "winner": best_name,
                "speedup_vs_cublas": round(speedup, 3),
            })

    # Summary: total per-layer latency at different M values
    print("\n" + "=" * 90)
    print("ESTIMATED PER-LAYER IMPACT (QKV + O_proj, 1 layer)")
    print("=" * 90)
    print(f"{'M':>4}  {'cuBLAS (µs)':>12}  {'Ternary (µs)':>13}  {'Fallback (µs)':>14}  {'Ternary/cuBLAS':>15}")
    print("-" * 65)

    for M in m_values:
        # QKV + O per layer
        cublas_total = 0
        ternary_total = 0
        fallback_total = 0
        all_ternary = True

        for r in results:
            if r["M"] == M and r["shape"] in ("qkv_proj", "o_proj"):
                cublas_total += r["cublas_ms"]
                if r["ternary_ms"] is not None:
                    ternary_total += r["ternary_ms"]
                else:
                    ternary_total += r["fallback_ms"]
                    all_ternary = False
                fallback_total += r["fallback_ms"]

        ratio = ternary_total / cublas_total if cublas_total > 0 else 0
        t_note = "" if all_ternary else " (w/fallback)"
        print(f"{M:>4}  {cublas_total*1000:>11.1f}  {ternary_total*1000:>12.1f}{t_note:12s}  {fallback_total*1000:>13.1f}  {ratio:>14.3f}x")

    # Total across 48 layers
    print(f"\n  × 48 layers = total attention linear latency impact")
    for M in m_values:
        cublas_48 = 0
        ternary_48 = 0
        for r in results:
            if r["M"] == M and r["shape"] in ("qkv_proj", "o_proj"):
                cublas_48 += r["cublas_ms"] * 48
                ternary_48 += (r["ternary_ms"] if r["ternary_ms"] else r["fallback_ms"]) * 48

        delta = (ternary_48 - cublas_48) * 1000  # in µs
        print(f"  M={M:>3}: cuBLAS total {cublas_48*1000:.0f} µs, ternary total {ternary_48*1000:.0f} µs, delta {delta:+.0f} µs ({delta/1000:+.2f} ms)")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(f"\nSaved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
