#!/usr/bin/env python3
"""
Concurrency Benchmark: Test throughput and latency at various concurrency levels

This benchmark sends concurrent requests to measure how ternary/FP16 servers
perform under load. Key metrics:
- Throughput (tokens/sec total)
- Latency percentiles (p50, p95, p99)
- Time to first token (TTFT)

Usage:
    python benchmark_concurrency.py --host 127.0.0.1 --port 30080
    python benchmark_concurrency.py --concurrency-levels 1,2,4,8,16,32
    python benchmark_concurrency.py --label "i2s-tuned" --output results.json
"""

import argparse
import asyncio
import aiohttp
import time
import json
import statistics
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class RequestResult:
    """Result from a single request."""
    success: bool
    latency_ms: float
    ttft_ms: float  # Time to first token (if streaming)
    tokens_generated: int
    tokens_per_sec: float
    error: Optional[str] = None


@dataclass
class ConcurrencyResult:
    """Aggregated results for a concurrency level."""
    concurrency: int
    num_requests: int
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    total_tokens: int
    
    # Throughput
    requests_per_sec: float
    tokens_per_sec_total: float  # Total throughput across all concurrent requests
    tokens_per_sec_per_request: float  # Average per-request generation speed
    
    # Latency (ms)
    latency_min: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    latency_avg: float
    
    # TTFT (ms) - Time to first token
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_avg: float = 0.0


@dataclass 
class BenchmarkConfig:
    """Configuration for the benchmark."""
    host: str = "127.0.0.1"
    port: int = 30080
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    requests_per_level: int = 50  # Total requests per concurrency level
    warmup_requests: int = 5
    max_tokens: int = 256  # Tokens to generate per request
    prompt: str = "Write a detailed explanation of how neural networks learn."
    temperature: float = 0.0
    ignore_eos: bool = True  # Force full generation for consistent measurement
    use_streaming: bool = False  # Enable for TTFT measurement
    timeout_sec: float = 120.0
    label: str = ""  # Label for this run (e.g., "i2s-tuned", "fp16")


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    timeout: float,
    use_streaming: bool = False
) -> RequestResult:
    """Send a single request and measure timing."""
    start_time = time.perf_counter()
    ttft = 0.0
    tokens = 0
    
    try:
        if use_streaming:
            # Streaming mode for accurate TTFT measurement
            stream_payload = payload.copy()
            stream_payload["stream"] = True
            
            async with session.post(
                url,
                json=stream_payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        success=False,
                        latency_ms=0,
                        ttft_ms=0,
                        tokens_generated=0,
                        tokens_per_sec=0,
                        error=f"HTTP {response.status}: {error_text[:100]}"
                    )
                
                first_chunk = True
                chunk_count = 0
                async for line in response.content:
                    if first_chunk and line.strip():
                        ttft = (time.perf_counter() - start_time) * 1000
                        first_chunk = False
                    if line.strip():
                        chunk_count += 1
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                # Estimate tokens from chunks (each chunk ~= 1 token in SSE)
                tokens = max(chunk_count - 1, payload.get("sampling_params", {}).get("max_new_tokens", 0))
        else:
            # Non-streaming mode - use server-reported timings if available
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                if response.status != 200:
                    error_text = await response.text()
                    return RequestResult(
                        success=False,
                        latency_ms=latency_ms,
                        ttft_ms=0,
                        tokens_generated=0,
                        tokens_per_sec=0,
                        error=f"HTTP {response.status}: {error_text[:100]}"
                    )
                
                data = await response.json()
                meta = data.get("meta_info", {})
                tokens = meta.get("completion_tokens", 0)
                if tokens == 0:
                    # Fallback: estimate from max_tokens
                    tokens = payload.get("sampling_params", {}).get("max_new_tokens", 0)
                
                # Try to get server-reported TTFT from meta_info
                # SGLang reports e2e_latency and other timings
                prompt_tokens = meta.get("prompt_tokens", len(payload.get("text", "").split()))
                
                # If server provides timing breakdown, use it
                # Otherwise estimate TTFT based on prompt length ratio
                if "prefill_token_logprobs" in meta or tokens > 0:
                    # Estimate: TTFT ≈ (prompt_tokens / total_tokens) * latency
                    # This is still an approximation but better than 5%
                    total_tokens_processed = prompt_tokens + tokens
                    if total_tokens_processed > 0:
                        ttft = latency_ms * (prompt_tokens / total_tokens_processed)
                    else:
                        ttft = latency_ms * 0.1  # 10% fallback
                else:
                    # No good estimate available - mark as N/A
                    ttft = -1  # Indicates "not measured"
        
        latency_sec = latency_ms / 1000.0
        tps = tokens / latency_sec if latency_sec > 0 else 0
        
        return RequestResult(
            success=True,
            latency_ms=latency_ms,
            ttft_ms=ttft,
            tokens_generated=tokens,
            tokens_per_sec=tps
        )
        
    except asyncio.TimeoutError:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            ttft_ms=0,
            tokens_generated=0,
            tokens_per_sec=0,
            error="Timeout"
        )
    except Exception as e:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            ttft_ms=0,
            tokens_generated=0,
            tokens_per_sec=0,
            error=str(e)[:100]
        )


async def run_concurrent_requests(
    config: BenchmarkConfig,
    concurrency: int,
    num_requests: int
) -> ConcurrencyResult:
    """Run requests at a specific concurrency level."""
    url = f"http://{config.host}:{config.port}/generate"
    payload = {
        "text": config.prompt,
        "sampling_params": {
            "max_new_tokens": config.max_tokens,
            "temperature": config.temperature,
            "ignore_eos": config.ignore_eos
        }
    }
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    async def bounded_request(session: aiohttp.ClientSession) -> RequestResult:
        async with semaphore:
            return await send_request(
                session, url, payload, config.timeout_sec, config.use_streaming
            )
    
    # Run all requests
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.perf_counter()
        
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
    
    # Aggregate results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        # All failed
        return ConcurrencyResult(
            concurrency=concurrency,
            num_requests=num_requests,
            successful_requests=0,
            failed_requests=len(failed),
            total_time_sec=total_time,
            total_tokens=0,
            requests_per_sec=0,
            tokens_per_sec_total=0,
            tokens_per_sec_per_request=0,
            latency_min=0,
            latency_p50=0,
            latency_p95=0,
            latency_p99=0,
            latency_max=0,
            latency_avg=0,
            ttft_p50=0,
            ttft_p95=0,
            ttft_avg=0
        )
    
    latencies = [r.latency_ms for r in successful]
    # Filter out invalid TTFT values (-1 means not measured)
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms >= 0]
    total_tokens = sum(r.tokens_generated for r in successful)
    
    latencies.sort()
    ttfts.sort()
    
    def percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f]) if c != f else data[f]
    
    return ConcurrencyResult(
        concurrency=concurrency,
        num_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_time_sec=total_time,
        total_tokens=total_tokens,
        requests_per_sec=len(successful) / total_time if total_time > 0 else 0,
        tokens_per_sec_total=total_tokens / total_time if total_time > 0 else 0,
        tokens_per_sec_per_request=statistics.mean([r.tokens_per_sec for r in successful]),
        latency_min=min(latencies),
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        latency_p99=percentile(latencies, 99),
        latency_max=max(latencies),
        latency_avg=statistics.mean(latencies),
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_avg=statistics.mean(ttfts) if ttfts else 0
    )


async def warmup(config: BenchmarkConfig, num_requests: int = 5):
    """Send warmup requests to ensure server is ready."""
    url = f"http://{config.host}:{config.port}/generate"
    payload = {
        "text": "Hello",
        "sampling_params": {"max_new_tokens": 10, "temperature": 0}
    }
    
    async with aiohttp.ClientSession() as session:
        for i in range(num_requests):
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    await response.text()
                    print(f"  Warmup {i+1}/{num_requests} complete")
            except Exception as e:
                print(f"  Warmup {i+1}/{num_requests} failed: {e}")


def print_results_table(results: List[ConcurrencyResult], label: str = ""):
    """Print results in a formatted table."""
    header_label = f" [{label}]" if label else ""
    
    print("\n" + "=" * 120)
    print(f"CONCURRENCY BENCHMARK RESULTS{header_label}")
    print("=" * 120)
    print(f"{'Conc':>6} | {'Reqs':>5} | {'OK':>4} | {'Fail':>4} | "
          f"{'Total TPS':>10} | {'Per-Req TPS':>11} | "
          f"{'Lat p50':>9} | {'Lat p95':>9} | {'Lat p99':>9} | {'Lat avg':>9}")
    print("-" * 120)
    
    for r in results:
        fail_str = f"{r.failed_requests}" if r.failed_requests == 0 else f"\033[91m{r.failed_requests}\033[0m"
        print(f"{r.concurrency:>6} | {r.num_requests:>5} | {r.successful_requests:>4} | {fail_str:>4} | "
              f"{r.tokens_per_sec_total:>10.1f} | {r.tokens_per_sec_per_request:>11.1f} | "
              f"{r.latency_p50:>9.1f} | {r.latency_p95:>9.1f} | {r.latency_p99:>9.1f} | {r.latency_avg:>9.1f}")
    
    print("=" * 120)
    print("\nLegend:")
    print("  Conc        = Concurrency level (simultaneous requests)")
    print("  Total TPS   = Total tokens/sec throughput (all concurrent requests combined)")
    print("  Per-Req TPS = Average tokens/sec per individual request")
    print("  Lat pXX     = Latency percentile in milliseconds (end-to-end request time)")
    print("\nNote: TTFT (Time to First Token) requires --streaming flag for accurate measurement.")
    print("      Without streaming, TTFT is estimated based on prompt/output ratio.")
    print()


def save_results(results: List[ConcurrencyResult], config: BenchmarkConfig, output_file: str):
    """Save results to JSON file."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "label": config.label,
        "config": {
            "host": config.host,
            "port": config.port,
            "max_tokens": config.max_tokens,
            "requests_per_level": config.requests_per_level,
            "prompt_length": len(config.prompt),
        },
        "results": [asdict(r) for r in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results saved to: {output_file}")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang server at various concurrency levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run against local server
  python benchmark_concurrency.py

  # Compare i2s-tuned vs FP16
  python benchmark_concurrency.py --port 30080 --label "i2s-tuned" --output i2s_results.json
  python benchmark_concurrency.py --port 30081 --label "fp16" --output fp16_results.json

  # Quick test with fewer requests
  python benchmark_concurrency.py --requests 10 --concurrency-levels 1,4,16

  # Heavy load test
  python benchmark_concurrency.py --requests 100 --concurrency-levels 1,2,4,8,16,32,64,128
        """
    )
    
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=30080, help="Server port")
    parser.add_argument(
        "--concurrency-levels", "-c",
        default="1,2,4,8,16,32,64",
        help="Comma-separated concurrency levels to test (default: 1,2,4,8,16,32,64)"
    )
    parser.add_argument(
        "--requests", "-r",
        type=int,
        default=50,
        help="Number of requests per concurrency level (default: 50)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=256,
        help="Max tokens to generate per request (default: 256)"
    )
    parser.add_argument(
        "--prompt",
        default="Write a detailed explanation of how neural networks learn through backpropagation.",
        help="Prompt to use for requests"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)"
    )
    parser.add_argument(
        "--label", "-l",
        default="",
        help="Label for this run (e.g., 'i2s-tuned', 'fp16')"
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=True,
        help="Force full token generation (ignore EOS)"
    )
    parser.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="Allow early stopping at EOS token"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for accurate TTFT measurement"
    )
    
    args = parser.parse_args()
    
    # Parse concurrency levels
    try:
        concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    except ValueError:
        print(f"Error: Invalid concurrency levels: {args.concurrency_levels}")
        sys.exit(1)
    
    config = BenchmarkConfig(
        host=args.host,
        port=args.port,
        concurrency_levels=concurrency_levels,
        requests_per_level=args.requests,
        warmup_requests=args.warmup,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        timeout_sec=args.timeout,
        ignore_eos=not args.no_ignore_eos,
        use_streaming=args.streaming,
        label=args.label
    )
    
    print("\n" + "=" * 80)
    print("CONCURRENCY BENCHMARK")
    print("=" * 80)
    print(f"Server:              http://{config.host}:{config.port}")
    print(f"Label:               {config.label or '(none)'}")
    print(f"Concurrency levels:  {config.concurrency_levels}")
    print(f"Requests per level:  {config.requests_per_level}")
    print(f"Max tokens:          {config.max_tokens}")
    print(f"Prompt length:       {len(config.prompt)} chars")
    print(f"Ignore EOS:          {config.ignore_eos}")
    print(f"Streaming (TTFT):    {config.use_streaming}")
    print("=" * 80)
    
    # Check server connectivity
    print("\nChecking server connectivity...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{config.host}:{config.port}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    print(f"  Server is healthy!")
                else:
                    print(f"  Warning: Health check returned {response.status}")
    except Exception as e:
        print(f"  Error connecting to server: {e}")
        print("  Proceeding anyway...")
    
    # Warmup
    print(f"\nRunning {config.warmup_requests} warmup requests...")
    await warmup(config, config.warmup_requests)
    
    # Run benchmarks for each concurrency level
    results = []
    for concurrency in config.concurrency_levels:
        print(f"\n>>> Testing concurrency={concurrency} with {config.requests_per_level} requests...")
        
        result = await run_concurrent_requests(
            config,
            concurrency,
            config.requests_per_level
        )
        results.append(result)
        
        # Quick summary
        if result.successful_requests > 0:
            print(f"    Total TPS: {result.tokens_per_sec_total:.1f}, "
                  f"Latency p50: {result.latency_p50:.0f}ms, "
                  f"p99: {result.latency_p99:.0f}ms, "
                  f"Failed: {result.failed_requests}")
        else:
            print(f"    All requests failed!")
    
    # Print summary table
    print_results_table(results, config.label)
    
    # Save results if output file specified
    if args.output:
        save_results(results, config, args.output)
    
    return results


if __name__ == "__main__":
    results = asyncio.run(main())




