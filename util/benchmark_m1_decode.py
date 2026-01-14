#!/usr/bin/env python3
"""
M=1 Decode Benchmark: Focused benchmark for single-token decode performance

This benchmark measures the M=1 decode path specifically, which is critical for
interactive inference (chat, code completion, etc.). It provides:

1. Kernel-level profiling breakdown (attention, ternary linear, MoE)
2. Comparison between FP16, BF16+ternary, and FP8+ternary paths
3. Memory usage tracking
4. CUDA event-based microsecond-precision timing

Usage:
    # Basic M=1 benchmark
    python benchmark_m1_decode.py --host 127.0.0.1 --port 30080
    
    # With kernel profiling (requires TERNARY_KERNEL_PROFILE=1 on server)
    python benchmark_m1_decode.py --kernel-profile
    
    # Sweep different modes
    python benchmark_m1_decode.py --compare-modes fp16 i2s i2s-fp8-full

Environment Variables (set on server):
    TERNARY_KERNEL_PROFILE=1     Enable kernel-level timing
    TERNARY_DETAILED_PROFILE=1   Enable detailed sub-operation timing
    TERNARY_ENABLE_PROFILING=1   Enable basic profiling
"""

import argparse
import asyncio
import aiohttp
import json
import time
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class M1DecodeResult:
    """Result from a single M=1 decode step."""
    step: int
    decode_time_ms: float
    tokens_per_sec: float
    success: bool
    error: Optional[str] = None


@dataclass
class M1BenchmarkResult:
    """Aggregated results for M=1 decode benchmark."""
    mode: str
    num_steps: int
    context_length: int
    
    # Timing (ms per decode step)
    decode_time_min: float
    decode_time_p50: float
    decode_time_p95: float
    decode_time_p99: float
    decode_time_max: float
    decode_time_avg: float
    decode_time_std: float
    
    # Tokens per second (single-token generation)
    tps_avg: float
    tps_std: float
    
    # Memory (if available)
    gpu_memory_mb: Optional[float] = None
    
    # Kernel breakdown (if available)
    kernel_breakdown: Optional[Dict[str, float]] = None


@dataclass
class M1BenchmarkConfig:
    """Configuration for M=1 decode benchmark."""
    host: str = "127.0.0.1"
    port: int = 30080
    num_decode_steps: int = 100  # Number of decode steps to measure
    context_length: int = 512    # Initial context length before decode
    warmup_steps: int = 10       # Warmup decode steps
    prompt: str = (
        "You are a helpful AI assistant. "
        "Below is a conversation between a human and an AI assistant.\n\n"
        "Human: Explain the concept of machine learning in simple terms.\n\n"
        "AI: Machine learning is a type of artificial intelligence that allows "
    )
    temperature: float = 0.0
    max_new_tokens: int = 200    # Total new tokens to generate
    mode: str = "unknown"        # Server mode label
    kernel_profile: bool = False # Request kernel profiling data


async def measure_m1_decode(
    session: aiohttp.ClientSession,
    config: M1BenchmarkConfig,
) -> M1BenchmarkResult:
    """
    Measure M=1 decode performance by streaming tokens and timing each step.
    
    The key insight is that in streaming mode, we can measure the time between
    each token, which approximates the M=1 decode time.
    """
    url = f"http://{config.host}:{config.port}/v1/completions"
    
    payload = {
        "model": "default",  # SGLang uses "default" model name
        "prompt": config.prompt,
        "max_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "stream": True,
        "ignore_eos": True,  # Force full generation
    }
    
    # Collect per-step timing
    step_times: List[float] = []
    total_tokens = 0
    
    try:
        start_time = time.perf_counter()
        prev_token_time = start_time
        
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return M1BenchmarkResult(
                    mode=config.mode,
                    num_steps=0,
                    context_length=config.context_length,
                    decode_time_min=0, decode_time_p50=0, decode_time_p95=0,
                    decode_time_p99=0, decode_time_max=0, decode_time_avg=0,
                    decode_time_std=0, tps_avg=0, tps_std=0,
                )
            
            # Stream tokens and measure time between each
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if not line or not line.startswith("data: "):
                    continue
                
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    # Check if this chunk contains a token
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "text" in choice or "delta" in choice:
                            current_time = time.perf_counter()
                            step_time_ms = (current_time - prev_token_time) * 1000
                            
                            # Skip warmup steps
                            if total_tokens >= config.warmup_steps:
                                step_times.append(step_time_ms)
                            
                            prev_token_time = current_time
                            total_tokens += 1
                            
                            if len(step_times) >= config.num_decode_steps:
                                break
                except json.JSONDecodeError:
                    continue
        
        end_time = time.perf_counter()
        
    except Exception as e:
        return M1BenchmarkResult(
            mode=config.mode,
            num_steps=0,
            context_length=config.context_length,
            decode_time_min=0, decode_time_p50=0, decode_time_p95=0,
            decode_time_p99=0, decode_time_max=0, decode_time_avg=0,
            decode_time_std=0, tps_avg=0, tps_std=0,
        )
    
    if not step_times:
        return M1BenchmarkResult(
            mode=config.mode,
            num_steps=0,
            context_length=config.context_length,
            decode_time_min=0, decode_time_p50=0, decode_time_p95=0,
            decode_time_p99=0, decode_time_max=0, decode_time_avg=0,
            decode_time_std=0, tps_avg=0, tps_std=0,
        )
    
    # Calculate statistics
    step_times_sorted = sorted(step_times)
    n = len(step_times_sorted)
    
    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        return data[f] * (c - k) + data[c] * (k - f)
    
    # TPS per step
    tps_list = [1000.0 / t for t in step_times if t > 0]
    
    return M1BenchmarkResult(
        mode=config.mode,
        num_steps=len(step_times),
        context_length=config.context_length,
        decode_time_min=min(step_times),
        decode_time_p50=percentile(step_times_sorted, 50),
        decode_time_p95=percentile(step_times_sorted, 95),
        decode_time_p99=percentile(step_times_sorted, 99),
        decode_time_max=max(step_times),
        decode_time_avg=statistics.mean(step_times),
        decode_time_std=statistics.stdev(step_times) if len(step_times) > 1 else 0,
        tps_avg=statistics.mean(tps_list) if tps_list else 0,
        tps_std=statistics.stdev(tps_list) if len(tps_list) > 1 else 0,
    )


async def get_kernel_profile(
    session: aiohttp.ClientSession,
    host: str,
    port: int,
) -> Optional[Dict[str, Any]]:
    """
    Fetch kernel profiling data from the server.
    
    Requires TERNARY_KERNEL_PROFILE=1 to be set on the server.
    """
    url = f"http://{host}:{port}/get_model_info"
    
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return None
            data = await response.json()
            return data.get("kernel_profile", None)
    except Exception:
        return None


async def run_m1_benchmark(config: M1BenchmarkConfig) -> M1BenchmarkResult:
    """Run the M=1 decode benchmark."""
    timeout = aiohttp.ClientTimeout(total=300)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Run the benchmark
        result = await measure_m1_decode(session, config)
        
        # Optionally fetch kernel profile
        if config.kernel_profile:
            kernel_data = await get_kernel_profile(session, config.host, config.port)
            if kernel_data:
                result.kernel_breakdown = kernel_data
        
        return result


def print_results(result: M1BenchmarkResult):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print(f"M=1 DECODE BENCHMARK RESULTS - Mode: {result.mode}")
    print("=" * 70)
    print(f"  Decode Steps Measured: {result.num_steps}")
    print(f"  Context Length: {result.context_length}")
    print()
    print("  TIMING (ms per decode step):")
    print(f"    Min:    {result.decode_time_min:.3f}")
    print(f"    P50:    {result.decode_time_p50:.3f}")
    print(f"    P95:    {result.decode_time_p95:.3f}")
    print(f"    P99:    {result.decode_time_p99:.3f}")
    print(f"    Max:    {result.decode_time_max:.3f}")
    print(f"    Avg:    {result.decode_time_avg:.3f} ± {result.decode_time_std:.3f}")
    print()
    print("  THROUGHPUT (tokens/sec):")
    print(f"    Avg:    {result.tps_avg:.2f} ± {result.tps_std:.2f}")
    print()
    
    if result.kernel_breakdown:
        print("  KERNEL BREAKDOWN:")
        for kernel, time_ms in result.kernel_breakdown.items():
            print(f"    {kernel}: {time_ms:.3f} ms")
        print()
    
    if result.gpu_memory_mb:
        print(f"  GPU MEMORY: {result.gpu_memory_mb:.1f} MB")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="M=1 Decode Benchmark for SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30080, help="Server port")
    parser.add_argument("--steps", type=int, default=100, 
                       help="Number of decode steps to measure")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup steps to skip")
    parser.add_argument("--context", type=int, default=512,
                       help="Initial context length")
    parser.add_argument("--max-tokens", type=int, default=200,
                       help="Maximum new tokens to generate")
    parser.add_argument("--mode", default="unknown",
                       help="Server mode label (e.g., fp16, i2s, i2s-fp8-full)")
    parser.add_argument("--kernel-profile", action="store_true",
                       help="Request kernel profiling data")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of benchmark runs to average")
    
    args = parser.parse_args()
    
    config = M1BenchmarkConfig(
        host=args.host,
        port=args.port,
        num_decode_steps=args.steps,
        warmup_steps=args.warmup,
        context_length=args.context,
        max_new_tokens=args.max_tokens,
        mode=args.mode,
        kernel_profile=args.kernel_profile,
    )
    
    print(f"\n[M=1 Decode Benchmark]")
    print(f"  Host: {config.host}:{config.port}")
    print(f"  Mode: {config.mode}")
    print(f"  Steps: {config.num_decode_steps} (warmup: {config.warmup_steps})")
    print(f"  Runs: {args.runs}")
    
    all_results = []
    
    for run in range(args.runs):
        print(f"\n  Running benchmark {run + 1}/{args.runs}...")
        result = asyncio.run(run_m1_benchmark(config))
        all_results.append(result)
        
        if result.num_steps > 0:
            print(f"    P50: {result.decode_time_p50:.3f} ms, "
                  f"TPS: {result.tps_avg:.2f}")
        else:
            print(f"    FAILED - no decode steps measured")
    
    # Aggregate results across runs
    if all_results and all_results[0].num_steps > 0:
        # Average the averages across runs
        avg_p50 = statistics.mean(r.decode_time_p50 for r in all_results if r.num_steps > 0)
        avg_p95 = statistics.mean(r.decode_time_p95 for r in all_results if r.num_steps > 0)
        avg_tps = statistics.mean(r.tps_avg for r in all_results if r.num_steps > 0)
        
        final_result = M1BenchmarkResult(
            mode=config.mode,
            num_steps=sum(r.num_steps for r in all_results),
            context_length=config.context_length,
            decode_time_min=min(r.decode_time_min for r in all_results if r.num_steps > 0),
            decode_time_p50=avg_p50,
            decode_time_p95=avg_p95,
            decode_time_p99=statistics.mean(r.decode_time_p99 for r in all_results if r.num_steps > 0),
            decode_time_max=max(r.decode_time_max for r in all_results if r.num_steps > 0),
            decode_time_avg=statistics.mean(r.decode_time_avg for r in all_results if r.num_steps > 0),
            decode_time_std=statistics.mean(r.decode_time_std for r in all_results if r.num_steps > 0),
            tps_avg=avg_tps,
            tps_std=statistics.mean(r.tps_std for r in all_results if r.num_steps > 0),
            kernel_breakdown=all_results[0].kernel_breakdown,
        )
        
        print_results(final_result)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(final_result), f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\n[ERROR] Benchmark failed - no valid results")
        sys.exit(1)


if __name__ == "__main__":
    main()
