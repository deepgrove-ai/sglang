#!/usr/bin/env python3
"""
Profile node load while driving N concurrent users against an SGLang endpoint.

This script combines:
  - request-level throughput/latency benchmarking
  - node-level CPU/RAM/GPU sampling over time

Typical usage:
  python util/profile_node_load.py --host 127.0.0.1 --port 30080 --users 32
  python util/profile_node_load.py --users-list 1,2,4,8,16,32 --requests-per-user 8
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:  # pragma: no cover - environment-dependent
    aiohttp = None

if TYPE_CHECKING:
    import aiohttp as aiohttp_typing


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    w = rank - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _read_cpu_totals() -> Tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        line = f.readline().strip()
    parts = line.split()
    nums = [int(x) for x in parts[1:]]
    total = sum(nums)
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    return total, idle


def _read_mem_usage() -> Tuple[float, float]:
    mem_total_kb = 0
    mem_available_kb = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available_kb = int(line.split()[1])
    if mem_total_kb <= 0:
        return 0.0, 0.0
    mem_used_kb = mem_total_kb - mem_available_kb
    mem_used_gb = mem_used_kb / (1024.0 * 1024.0)
    mem_pct = (mem_used_kb / mem_total_kb) * 100.0
    return mem_used_gb, mem_pct


def _read_gpu_metrics() -> Dict[str, float]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=1.5)
    except Exception:
        return {}

    util_vals: List[float] = []
    mem_util_vals: List[float] = []
    mem_used_vals: List[float] = []
    mem_total_vals: List[float] = []
    power_vals: List[float] = []
    for raw_line in out.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) != 5:
            continue
        try:
            util_vals.append(float(parts[0]))
            mem_util_vals.append(float(parts[1]))
            mem_used_vals.append(float(parts[2]))
            mem_total_vals.append(float(parts[3]))
            power_vals.append(float(parts[4]))
        except ValueError:
            continue

    if not util_vals:
        return {}

    total_mem_used = sum(mem_used_vals)
    total_mem_total = sum(mem_total_vals) if mem_total_vals else 0.0
    mem_pct = (total_mem_used / total_mem_total * 100.0) if total_mem_total > 0 else 0.0
    return {
        "gpu_util_avg_pct": _safe_mean(util_vals),
        "gpu_mem_util_avg_pct": _safe_mean(mem_util_vals),
        "gpu_mem_used_total_mb": total_mem_used,
        "gpu_mem_total_mb": total_mem_total,
        "gpu_mem_used_pct": mem_pct,
        "gpu_power_total_w": sum(power_vals),
        "gpu_count": float(len(util_vals)),
    }


@dataclass
class SystemSample:
    ts_unix: float
    rel_sec: float
    cpu_pct: float
    mem_used_gb: float
    mem_pct: float
    load1: float
    load5: float
    load15: float
    gpu_util_avg_pct: float = 0.0
    gpu_mem_util_avg_pct: float = 0.0
    gpu_mem_used_total_mb: float = 0.0
    gpu_mem_total_mb: float = 0.0
    gpu_mem_used_pct: float = 0.0
    gpu_power_total_w: float = 0.0
    gpu_count: float = 0.0


class SystemSampler:
    def __init__(self, interval_sec: float):
        self.interval_sec = interval_sec
        self.samples: List[SystemSample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts = 0.0

    def start(self) -> None:
        self._start_ts = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))

    def _run(self) -> None:
        prev_total, prev_idle = _read_cpu_totals()
        while not self._stop.is_set():
            time.sleep(self.interval_sec)
            now = time.time()
            total, idle = _read_cpu_totals()
            dt_total = max(1, total - prev_total)
            dt_idle = max(0, idle - prev_idle)
            cpu_pct = max(0.0, min(100.0, 100.0 * (1.0 - (dt_idle / dt_total))))
            prev_total, prev_idle = total, idle

            mem_used_gb, mem_pct = _read_mem_usage()
            load1, load5, load15 = os.getloadavg()
            gpu = _read_gpu_metrics()

            sample = SystemSample(
                ts_unix=now,
                rel_sec=now - self._start_ts,
                cpu_pct=cpu_pct,
                mem_used_gb=mem_used_gb,
                mem_pct=mem_pct,
                load1=load1,
                load5=load5,
                load15=load15,
                gpu_util_avg_pct=gpu.get("gpu_util_avg_pct", 0.0),
                gpu_mem_util_avg_pct=gpu.get("gpu_mem_util_avg_pct", 0.0),
                gpu_mem_used_total_mb=gpu.get("gpu_mem_used_total_mb", 0.0),
                gpu_mem_total_mb=gpu.get("gpu_mem_total_mb", 0.0),
                gpu_mem_used_pct=gpu.get("gpu_mem_used_pct", 0.0),
                gpu_power_total_w=gpu.get("gpu_power_total_w", 0.0),
                gpu_count=gpu.get("gpu_count", 0.0),
            )
            self.samples.append(sample)


@dataclass
class RequestResult:
    success: bool
    latency_ms: float
    tokens_generated: int
    error: str = ""


async def _send_generate_request(
    session: "aiohttp_typing.ClientSession",
    url: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> RequestResult:
    start = time.perf_counter()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            end = time.perf_counter()
            latency_ms = (end - start) * 1000.0
            if resp.status != 200:
                txt = await resp.text()
                return RequestResult(
                    success=False,
                    latency_ms=latency_ms,
                    tokens_generated=0,
                    error=f"HTTP {resp.status}: {txt[:120]}",
                )
            data = await resp.json()
            meta = data.get("meta_info", {})
            tokens = int(meta.get("completion_tokens", 0))
            if tokens <= 0:
                tokens = int(payload.get("sampling_params", {}).get("max_new_tokens", 0))
            return RequestResult(success=True, latency_ms=latency_ms, tokens_generated=tokens)
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(
            success=False,
            latency_ms=(end - start) * 1000.0,
            tokens_generated=0,
            error=str(e)[:120],
        )


async def _warmup(
    session: "aiohttp_typing.ClientSession",
    url: str,
    warmup_requests: int,
    timeout_sec: float,
) -> None:
    payload = {
        "text": "Warmup request",
        "sampling_params": {
            "max_new_tokens": 8,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    for _ in range(max(0, warmup_requests)):
        await _send_generate_request(session, url, payload, timeout_sec)


async def run_load(
    host: str,
    port: int,
    users: int,
    total_requests: int,
    max_tokens: int,
    prompt: str,
    temperature: float,
    timeout_sec: float,
    ignore_eos: bool,
    warmup_requests: int,
) -> Dict[str, Any]:
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "ignore_eos": ignore_eos,
        },
    }

    semaphore = asyncio.Semaphore(users)
    connector = aiohttp.TCPConnector(limit=max(users + 16, 32))
    async with aiohttp.ClientSession(connector=connector) as session:
        await _warmup(session, url, warmup_requests, timeout_sec)

        async def one_req() -> RequestResult:
            async with semaphore:
                return await _send_generate_request(session, url, payload, timeout_sec)

        t0 = time.perf_counter()
        tasks = [one_req() for _ in range(total_requests)]
        results: List[RequestResult] = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - t0

    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in ok]
    tokens = [r.tokens_generated for r in ok]
    total_tokens = sum(tokens)
    req_per_sec = len(ok) / elapsed if elapsed > 0 else 0.0
    tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0

    return {
        "users": users,
        "total_requests": total_requests,
        "successful_requests": len(ok),
        "failed_requests": len(failed),
        "elapsed_sec": elapsed,
        "requests_per_sec": req_per_sec,
        "tokens_per_sec": tok_per_sec,
        "total_tokens": total_tokens,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "p99": _percentile(latencies, 99),
            "avg": _safe_mean(latencies),
            "max": max(latencies) if latencies else 0.0,
        },
        "errors_preview": [r.error for r in failed[:5]],
    }


def summarize_samples(samples: List[SystemSample]) -> Dict[str, Any]:
    cpu = [s.cpu_pct for s in samples]
    mem = [s.mem_pct for s in samples]
    gpu_util = [s.gpu_util_avg_pct for s in samples if s.gpu_count > 0]
    gpu_mem = [s.gpu_mem_used_pct for s in samples if s.gpu_count > 0]
    gpu_power = [s.gpu_power_total_w for s in samples if s.gpu_count > 0]

    return {
        "num_samples": len(samples),
        "cpu_pct": {
            "avg": _safe_mean(cpu),
            "p95": _percentile(cpu, 95),
            "max": max(cpu) if cpu else 0.0,
        },
        "mem_pct": {
            "avg": _safe_mean(mem),
            "p95": _percentile(mem, 95),
            "max": max(mem) if mem else 0.0,
        },
        "gpu_util_pct": {
            "avg": _safe_mean(gpu_util),
            "p95": _percentile(gpu_util, 95),
            "max": max(gpu_util) if gpu_util else 0.0,
        },
        "gpu_mem_used_pct": {
            "avg": _safe_mean(gpu_mem),
            "p95": _percentile(gpu_mem, 95),
            "max": max(gpu_mem) if gpu_mem else 0.0,
        },
        "gpu_power_w": {
            "avg": _safe_mean(gpu_power),
            "p95": _percentile(gpu_power, 95),
            "max": max(gpu_power) if gpu_power else 0.0,
        },
    }


def write_samples_jsonl(samples: List[SystemSample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def parse_users_list(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("users list is empty")
    return out


def run_one_profile(
    args: argparse.Namespace,
    users: int,
    output_dir: Path,
) -> Dict[str, Any]:
    total_requests = args.total_requests
    if total_requests <= 0:
        total_requests = users * args.requests_per_user
    total_requests = max(total_requests, users)

    sampler = SystemSampler(interval_sec=args.sample_interval)
    sampler.start()
    load_result: Dict[str, Any]
    try:
        load_result = asyncio.run(
            run_load(
                host=args.host,
                port=args.port,
                users=users,
                total_requests=total_requests,
                max_tokens=args.max_tokens,
                prompt=args.prompt,
                temperature=args.temperature,
                timeout_sec=args.timeout,
                ignore_eos=args.ignore_eos,
                warmup_requests=args.warmup_requests,
            )
        )
    finally:
        sampler.stop()

    system_summary = summarize_samples(sampler.samples)
    profile_result = {
        "label": args.label,
        "host": args.host,
        "port": args.port,
        "users": users,
        "total_requests": total_requests,
        "sample_interval_sec": args.sample_interval,
        "load": load_result,
        "system": system_summary,
    }

    run_name = f"users_{users}"
    summary_path = output_dir / f"{run_name}.json"
    samples_path = output_dir / f"{run_name}_samples.jsonl"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(profile_result, f, indent=2)
    write_samples_jsonl(sampler.samples, samples_path)
    return profile_result


def write_sweep_csv(results: List[Dict[str, Any]], csv_path: Path) -> None:
    fields = [
        "users",
        "total_requests",
        "successful_requests",
        "failed_requests",
        "requests_per_sec",
        "tokens_per_sec",
        "latency_p50_ms",
        "latency_p95_ms",
        "cpu_avg_pct",
        "cpu_p95_pct",
        "gpu_util_avg_pct",
        "gpu_util_p95_pct",
        "gpu_mem_avg_pct",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "users": r["users"],
                    "total_requests": r["load"]["total_requests"],
                    "successful_requests": r["load"]["successful_requests"],
                    "failed_requests": r["load"]["failed_requests"],
                    "requests_per_sec": f"{r['load']['requests_per_sec']:.3f}",
                    "tokens_per_sec": f"{r['load']['tokens_per_sec']:.3f}",
                    "latency_p50_ms": f"{r['load']['latency_ms']['p50']:.3f}",
                    "latency_p95_ms": f"{r['load']['latency_ms']['p95']:.3f}",
                    "cpu_avg_pct": f"{r['system']['cpu_pct']['avg']:.3f}",
                    "cpu_p95_pct": f"{r['system']['cpu_pct']['p95']:.3f}",
                    "gpu_util_avg_pct": f"{r['system']['gpu_util_pct']['avg']:.3f}",
                    "gpu_util_p95_pct": f"{r['system']['gpu_util_pct']['p95']:.3f}",
                    "gpu_mem_avg_pct": f"{r['system']['gpu_mem_used_pct']['avg']:.3f}",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile node load for N concurrent users.")
    parser.add_argument("--host", default="127.0.0.1", help="SGLang server host")
    parser.add_argument("--port", type=int, default=30080, help="SGLang server port")
    parser.add_argument("--users", type=int, default=16, help="Concurrent users for a single run")
    parser.add_argument(
        "--users-list",
        default="",
        help="Comma-separated concurrency sweep (e.g. 1,2,4,8,16,32). Overrides --users.",
    )
    parser.add_argument(
        "--requests-per-user",
        type=int,
        default=8,
        help="Requests per user (used when --total-requests <= 0).",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=0,
        help="Total requests for each run. If 0, uses users * requests-per-user.",
    )
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens per request")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds")
    parser.add_argument("--warmup-requests", type=int, default=3, help="Warmup requests before load")
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        default=True,
        help="Force full token generation (default true).",
    )
    parser.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="Allow early stop at EOS.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
        help="System metric sampling interval in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: node_load_profile_<timestamp>).",
    )
    parser.add_argument("--label", default="mangrove-node-load", help="Run label")
    parser.add_argument(
        "--prompt",
        default="Write a concise explanation of why low-latency kernels matter for LLM serving.",
        help="Prompt text for benchmark requests",
    )
    args = parser.parse_args()

    if aiohttp is None:
        print("Error: missing dependency 'aiohttp'.")
        print("Install with: pip install aiohttp")
        return 2

    if args.no_ignore_eos:
        args.ignore_eos = False

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"node_load_profile_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    users_levels: List[int]
    if args.users_list:
        users_levels = parse_users_list(args.users_list)
    else:
        users_levels = [args.users]

    print("=" * 90)
    print("NODE LOAD PROFILER")
    print("=" * 90)
    print(f"Server:            http://{args.host}:{args.port}")
    print(f"Users sweep:       {users_levels}")
    print(f"Requests/user:     {args.requests_per_user} (unless --total-requests is set)")
    print(f"Max tokens:        {args.max_tokens}")
    print(f"Sampling interval: {args.sample_interval}s")
    print(f"Output dir:        {out_dir}")
    print("=" * 90)

    results: List[Dict[str, Any]] = []
    for users in users_levels:
        print(f"\n[Run] users={users}")
        run_result = run_one_profile(args, users, out_dir)
        results.append(run_result)
        print(
            "  load: ok={ok}/{tot}, tok/s={tps:.1f}, p50={p50:.1f}ms, p95={p95:.1f}ms".format(
                ok=run_result["load"]["successful_requests"],
                tot=run_result["load"]["total_requests"],
                tps=run_result["load"]["tokens_per_sec"],
                p50=run_result["load"]["latency_ms"]["p50"],
                p95=run_result["load"]["latency_ms"]["p95"],
            )
        )
        print(
            "  node: cpu_avg={cpu:.1f}%, gpu_avg={gpu:.1f}%, gpu_mem_avg={gpu_mem:.1f}%".format(
                cpu=run_result["system"]["cpu_pct"]["avg"],
                gpu=run_result["system"]["gpu_util_pct"]["avg"],
                gpu_mem=run_result["system"]["gpu_mem_used_pct"]["avg"],
            )
        )

    sweep_path = out_dir / "sweep_summary.json"
    with sweep_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_sweep_csv(results, out_dir / "sweep_summary.csv")

    # Best throughput / best latency quick pointers
    best_tps = max(results, key=lambda r: r["load"]["tokens_per_sec"])
    best_lat = min(results, key=lambda r: r["load"]["latency_ms"]["p50"])
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"Best throughput: users={best_tps['users']} tok/s={best_tps['load']['tokens_per_sec']:.1f} "
        f"(p50={best_tps['load']['latency_ms']['p50']:.1f}ms)"
    )
    print(
        f"Best p50 latency: users={best_lat['users']} p50={best_lat['load']['latency_ms']['p50']:.1f}ms "
        f"(tok/s={best_lat['load']['tokens_per_sec']:.1f})"
    )
    print(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
