#!/usr/bin/env python3
"""
Performance benchmarks for FP8 ternary quantization.

This script provides:
1. Microbenchmarks for individual kernels (DP4A vs FP8 TC)
2. End-to-end prefill/decode latency comparison
3. Memory usage comparison
4. Nsight Compute profile generation

Usage:
    # Run all benchmarks
    python benchmark_ternary_fp8.py
    
    # Run specific benchmark
    python benchmark_ternary_fp8.py --benchmark linear
    python benchmark_ternary_fp8.py --benchmark moe
    
    # Generate NCU profile (requires root/sudo)
    python benchmark_ternary_fp8.py --ncu --kernel linear
    
    # Verbose output
    python benchmark_ternary_fp8.py -v
"""

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Set up path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    time_ms: float
    throughput_gflops: Optional[float] = None
    memory_mb: Optional[float] = None
    num_iterations: int = 100
    
    def __repr__(self):
        result = f"{self.name}: {self.time_ms:.3f} ms"
        if self.throughput_gflops:
            result += f" ({self.throughput_gflops:.1f} GFLOPS)"
        if self.memory_mb:
            result += f" ({self.memory_mb:.1f} MB)"
        return result


def warmup_cuda():
    """Warmup CUDA to stabilize measurements."""
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        for _ in range(10):
            _ = x @ x.T
        torch.cuda.synchronize()


def benchmark_kernel(func, *args, num_warmup=10, num_iterations=100, **kwargs) -> float:
    """
    Benchmark a kernel function using CUDA events.
    
    Returns:
        Average time in milliseconds
    """
    # Warmup
    for _ in range(num_warmup):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(num_iterations):
        func(*args, **kwargs)
    
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    return total_time_ms / num_iterations


class LinearBenchmark:
    """Benchmark linear layer operations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def run_bf16_baseline(self, M: int, N: int, K: int) -> BenchmarkResult:
        """Run BF16 F.linear baseline."""
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        
        def kernel():
            return F.linear(x, w)
        
        time_ms = benchmark_kernel(kernel)
        
        # Compute GFLOPS
        flops = 2 * M * N * K
        gflops = (flops / (time_ms * 1e-3)) / 1e9
        
        return BenchmarkResult(
            name=f"BF16 F.linear M={M} N={N} K={K}",
            time_ms=time_ms,
            throughput_gflops=gflops,
        )
    
    def run_fp8_bridge(self, M: int, N: int, K: int) -> Optional[BenchmarkResult]:
        """Run FP8 bridge linear."""
        try:
            from sgl_kernel import fp8_scaled_mm
            from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
        except ImportError:
            return None
        
        # Prepare inputs
        # x: (M, K) input activations
        # w: (N, K) original weight, will be transposed to (K, N) for fp8_scaled_mm
        x_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        
        # Quantize weight to FP8 and transpose (done once)
        # fp8_scaled_mm expects: (M, K) @ (K, N) with weight in column-major order
        FP8_MAX = 448.0
        # Per-channel (output dim) scaling on the original (N, K) weight
        w_abs_max = w_bf16.abs().amax(dim=1, keepdim=True)  # (N, 1)
        w_scale = (w_abs_max / FP8_MAX).clamp(min=1e-12).to(torch.float32)
        w_fp8 = (w_bf16 / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        w_scale = w_scale.squeeze(1)  # (N,)
        
        # Transpose weight from (N, K) to (K, N) for the GEMM
        # fp8_scaled_mm requires mat_b to be column-major
        # Column-major (K, N) = .T of row-major (N, K) without contiguous()
        w_fp8_t = w_fp8.T  # (K, N) with column-major strides (1, N)
        
        def kernel():
            # Quantize activation (per-token scaling)
            x_fp8, x_scale = per_token_group_quant_fp8(x_bf16.contiguous(), K)  # Full-row scale
            # FP8 GEMM: (M, K) @ (K, N) -> (M, N)
            return fp8_scaled_mm(
                x_fp8, w_fp8_t,
                x_scale, w_scale,
                out_dtype=torch.bfloat16,
                bias=None,
            )
        
        time_ms = benchmark_kernel(kernel)
        
        flops = 2 * M * N * K
        gflops = (flops / (time_ms * 1e-3)) / 1e9
        
        return BenchmarkResult(
            name=f"FP8 Bridge M={M} N={N} K={K}",
            time_ms=time_ms,
            throughput_gflops=gflops,
        )
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all linear benchmarks."""
        shapes = [
            # Decode shapes (M=1)
            (1, 5120, 2048),   # QKV fused
            (1, 2048, 2048),   # O proj
            (1, 1536, 2048),   # MoE gate_up fused
            (1, 2048, 768),    # MoE down
            
            # Prefill shapes (M>1)
            (32, 5120, 2048),
            (32, 2048, 2048),
            (128, 5120, 2048),
            (128, 2048, 2048),
            (512, 5120, 2048),
        ]
        
        results = []
        
        for M, N, K in shapes:
            if self.verbose:
                print(f"\nBenchmarking M={M}, N={N}, K={K}...")
            
            # BF16 baseline
            result_bf16 = self.run_bf16_baseline(M, N, K)
            results.append(result_bf16)
            
            # FP8 bridge
            result_fp8 = self.run_fp8_bridge(M, N, K)
            if result_fp8:
                results.append(result_fp8)
                
                # Compute speedup
                speedup = result_bf16.time_ms / result_fp8.time_ms
                if self.verbose:
                    print(f"  BF16: {result_bf16.time_ms:.3f} ms")
                    print(f"  FP8:  {result_fp8.time_ms:.3f} ms")
                    print(f"  Speedup: {speedup:.2f}x")
        
        self.results = results
        return results


class MoEBenchmark:
    """Benchmark MoE operations."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all MoE benchmarks."""
        # MoE benchmarks would require the actual MoE infrastructure
        # For now, return placeholder
        print("MoE benchmarks require model loading - skipping")
        return []


def generate_ncu_profile(kernel_type: str, output_dir: str = "./ncu_profiles"):
    """
    Generate Nsight Compute profile for a kernel.
    
    Args:
        kernel_type: Type of kernel to profile ("linear", "moe")
        output_dir: Directory to save profile output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a simple profiling script
    profile_script = f'''
import torch
import os
os.environ["SGLANG_TERNARY_FP8_BRIDGE"] = "1"

from sglang.srt.layers.quantization.fp8_kernel import per_token_group_quant_fp8
from sgl_kernel import fp8_scaled_mm

# Shapes for {kernel_type}
M, N, K = 128, 5120, 2048

# Create tensors
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
w = torch.randn(N, K, device="cuda", dtype=torch.float8_e4m3fn)
w_scale = torch.ones(N, device="cuda", dtype=torch.float32)

# Warmup
for _ in range(5):
    x_fp8, x_scale = per_token_group_quant_fp8(x.contiguous(), 128)
    out = fp8_scaled_mm(x_fp8, w.T.contiguous(), x_scale, w_scale, out_dtype=torch.bfloat16, bias=None)
torch.cuda.synchronize()

# Profile region
for _ in range(10):
    x_fp8, x_scale = per_token_group_quant_fp8(x.contiguous(), 128)
    out = fp8_scaled_mm(x_fp8, w.T.contiguous(), x_scale, w_scale, out_dtype=torch.bfloat16, bias=None)
torch.cuda.synchronize()
'''
    
    script_path = os.path.join(output_dir, f"profile_{kernel_type}.py")
    with open(script_path, "w") as f:
        f.write(profile_script)
    
    output_file = os.path.join(output_dir, f"{kernel_type}_profile.ncu-rep")
    
    ncu_cmd = [
        "ncu",
        "--set", "full",
        "--target-processes", "all",
        "-o", output_file.replace(".ncu-rep", ""),
        sys.executable, script_path,
    ]
    
    print(f"Running: {' '.join(ncu_cmd)}")
    print("Note: This requires sudo/root permissions")
    
    try:
        subprocess.run(ncu_cmd, check=True)
        print(f"Profile saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"NCU profiling failed: {e}")
    except FileNotFoundError:
        print("NCU not found. Please install NVIDIA Nsight Compute.")


def print_results_table(results: List[BenchmarkResult]):
    """Print benchmark results as a table."""
    if not results:
        print("No results to display")
        return
    
    # Group results by shape
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    
    print(f"{'Kernel':<50} {'Time (ms)':<12} {'GFLOPS':<10}")
    print("-" * 80)
    
    for result in results:
        gflops_str = f"{result.throughput_gflops:.1f}" if result.throughput_gflops else "N/A"
        print(f"{result.name:<50} {result.time_ms:>10.3f}  {gflops_str:>10}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="FP8 Ternary Performance Benchmarks")
    parser.add_argument("--benchmark", choices=["linear", "moe", "all"], default="all",
                        help="Which benchmark to run")
    parser.add_argument("--ncu", action="store_true", help="Generate NCU profile")
    parser.add_argument("--kernel", default="linear", help="Kernel to profile with NCU")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("CUDA not available, exiting")
        sys.exit(1)
    
    print(f"CUDA Device: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Warmup
    print("\nWarming up CUDA...")
    warmup_cuda()
    
    if args.ncu:
        generate_ncu_profile(args.kernel)
        return
    
    all_results = []
    
    if args.benchmark in ("linear", "all"):
        print("\n" + "=" * 80)
        print("Linear Benchmarks")
        print("=" * 80)
        
        linear_bench = LinearBenchmark(verbose=args.verbose)
        linear_results = linear_bench.run_all()
        all_results.extend(linear_results)
    
    if args.benchmark in ("moe", "all"):
        print("\n" + "=" * 80)
        print("MoE Benchmarks")
        print("=" * 80)
        
        moe_bench = MoEBenchmark(verbose=args.verbose)
        moe_results = moe_bench.run_all()
        all_results.extend(moe_results)
    
    # Print summary
    print_results_table(all_results)
    
    # Calculate average speedups
    bf16_results = [r for r in all_results if "BF16" in r.name]
    fp8_results = [r for r in all_results if "FP8" in r.name]
    
    if bf16_results and fp8_results:
        print("\nSpeedup Summary:")
        for bf16, fp8 in zip(bf16_results, fp8_results):
            if "BF16" in bf16.name and "FP8" in fp8.name:
                # Extract shape from name
                speedup = bf16.time_ms / fp8.time_ms
                print(f"  {fp8.name}: {speedup:.2f}x vs BF16")


if __name__ == "__main__":
    main()
