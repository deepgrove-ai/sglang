#!/usr/bin/env python3
"""
Find max concurrent users under a TTFT target and estimate KV-cache budget per user.

This script is designed for production capacity questions such as:
  - "How many concurrent users can we serve with ~1s TTFT?"
  - "What is the per-user KV budget at each concurrency?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # pragma: no cover - environment dependent
    aiohttp = None


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


@dataclass
class RequestMetric:
    success: bool
    ttft_ms: float
    latency_ms: float
    tokens_generated: int
    error: Optional[str] = None


@dataclass
class LevelResult:
    concurrency: int
    requests: int
    successful_requests: int
    failed_requests: int
    total_time_sec: float
    total_tokens: int
    tokens_per_sec: float
    requests_per_sec: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_avg_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_avg_ms: float
    kv_tokens_per_user_budget: int
    kv_required_tokens_per_user: int
    kv_headroom_tokens_per_user: int
    kv_budget_ok: bool


def parse_levels(raw: str) -> List[int]:
    out: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("No concurrency levels provided")
    return out


async def wait_until_healthy(host: str, port: int, timeout_sec: float) -> bool:
    url = f"http://{host}:{port}/health"
    end = time.time() + timeout_sec
    async with aiohttp.ClientSession() as session:
        while time.time() < end:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(1.0)
    return False


async def get_server_info(host: str, port: int) -> Dict[str, Any]:
    url = f"http://{host}:{port}/get_server_info"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            resp.raise_for_status()
            return await resp.json()


async def get_capacity_context(host: str, port: int) -> Dict[str, Any]:
    """
    Resolve KV capacity context for both direct server mode and router/DP mode.

    Returns:
      {
        "server_info": Dict[str, Any],           # best-effort representative server info
        "capacity_mode": "direct"|"router_workers"|"unknown",
        "worker_count": int,
        "max_total_num_tokens": int,             # cluster-level in router mode
        "num_reserved_decode_tokens": int,       # cluster-level in router mode
      }
    """
    base = f"http://{host}:{port}"
    timeout = aiohttp.ClientTimeout(total=20)

    async with aiohttp.ClientSession() as session:
        # 1) Try direct /get_server_info (single-server mode).
        try:
            async with session.get(f"{base}/get_server_info", timeout=timeout) as resp:
                resp.raise_for_status()
                info = await resp.json()
                return {
                    "server_info": info,
                    "capacity_mode": "direct",
                    "worker_count": 1,
                    "max_total_num_tokens": int(info.get("max_total_num_tokens", 0)),
                    "num_reserved_decode_tokens": int(
                        info.get("num_reserved_decode_tokens", 0)
                    ),
                }
        except Exception:
            pass

        # 2) Router mode fallback: discover workers and aggregate their capacities.
        worker_urls: List[str] = []
        try:
            async with session.get(f"{base}/workers", timeout=timeout) as resp:
                resp.raise_for_status()
                workers_payload = await resp.json()
                for w in workers_payload.get("workers", []):
                    if not isinstance(w, dict):
                        continue
                    if not w.get("is_healthy", True):
                        continue
                    url = str(w.get("url", "")).strip()
                    if url.startswith("http://"):
                        worker_urls.append(url.rstrip("/"))
        except Exception:
            worker_urls = []

        worker_infos: List[Dict[str, Any]] = []
        for wurl in worker_urls:
            try:
                async with session.get(f"{wurl}/get_server_info", timeout=timeout) as resp:
                    if resp.status != 200:
                        continue
                    w_info = await resp.json()
                    if isinstance(w_info, dict):
                        worker_infos.append(w_info)
            except Exception:
                continue

        if worker_infos:
            total_tokens = sum(int(x.get("max_total_num_tokens", 0)) for x in worker_infos)
            total_reserved = sum(
                int(x.get("num_reserved_decode_tokens", 0)) for x in worker_infos
            )
            return {
                "server_info": worker_infos[0],
                "capacity_mode": "router_workers",
                "worker_count": len(worker_infos),
                "max_total_num_tokens": int(total_tokens),
                "num_reserved_decode_tokens": int(total_reserved),
            }

    # 3) Unknown mode: keep benchmark working with best-effort zeros.
    return {
        "server_info": {},
        "capacity_mode": "unknown",
        "worker_count": 1,
        "max_total_num_tokens": 0,
        "num_reserved_decode_tokens": 0,
    }


async def estimate_prompt_tokens(
    host: str,
    port: int,
    prompt: str,
    timeout_sec: float,
) -> int:
    """Get prompt token count from server-reported meta_info."""
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": False,
        },
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    meta = data.get("meta_info", {})
    prompt_tokens = int(meta.get("prompt_tokens", 0))
    if prompt_tokens <= 0:
        # Best-effort fallback.
        prompt_tokens = max(1, len(prompt.split()))
    return prompt_tokens


async def send_stream_request(
    session: "aiohttp.ClientSession",
    url: str,
    payload: Dict[str, Any],
    timeout_sec: float,
) -> RequestMetric:
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as resp:
            if resp.status != 200:
                txt = await resp.text()
                return RequestMetric(
                    success=False,
                    ttft_ms=0.0,
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    tokens_generated=0,
                    error=f"HTTP {resp.status}: {txt[:120]}",
                )

            async for raw in resp.content:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                else:
                    data = line
                if data == "[DONE]":
                    break
                if first_token_time is None:
                    first_token_time = time.perf_counter()

            end = time.perf_counter()
            ttft_ms = (
                (first_token_time - start) * 1000.0
                if first_token_time is not None
                else (end - start) * 1000.0
            )
            latency_ms = (end - start) * 1000.0

            # We force max token generation with ignore_eos by default.
            max_tokens = int(payload["sampling_params"]["max_new_tokens"])
            return RequestMetric(
                success=True,
                ttft_ms=ttft_ms,
                latency_ms=latency_ms,
                tokens_generated=max_tokens,
            )
    except asyncio.TimeoutError:
        return RequestMetric(
            success=False,
            ttft_ms=0.0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tokens_generated=0,
            error="timeout",
        )
    except Exception as e:
        return RequestMetric(
            success=False,
            ttft_ms=0.0,
            latency_ms=(time.perf_counter() - start) * 1000.0,
            tokens_generated=0,
            error=str(e)[:120],
        )


async def run_level(
    host: str,
    port: int,
    concurrency: int,
    requests_count: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    ignore_eos: bool,
    timeout_sec: float,
    kv_tokens_per_user_budget: int,
    kv_required_tokens_per_user: int,
) -> LevelResult:
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": prompt,
        "stream": True,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "ignore_eos": ignore_eos,
        },
    }
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max(concurrency + 16, 64))
    ) as session:

        async def one_req() -> RequestMetric:
            async with sem:
                return await send_stream_request(session, url, payload, timeout_sec)

        t0 = time.perf_counter()
        tasks = [one_req() for _ in range(requests_count)]
        out: List[RequestMetric] = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t0

    ok = [x for x in out if x.success]
    ttfts = [x.ttft_ms for x in ok]
    lats = [x.latency_ms for x in ok]
    total_tokens = sum(x.tokens_generated for x in ok)
    kv_headroom = kv_tokens_per_user_budget - kv_required_tokens_per_user

    return LevelResult(
        concurrency=concurrency,
        requests=requests_count,
        successful_requests=len(ok),
        failed_requests=len(out) - len(ok),
        total_time_sec=total_time,
        total_tokens=total_tokens,
        tokens_per_sec=(total_tokens / total_time) if total_time > 0 else 0.0,
        requests_per_sec=(len(ok) / total_time) if total_time > 0 else 0.0,
        ttft_p50_ms=percentile(ttfts, 50),
        ttft_p95_ms=percentile(ttfts, 95),
        ttft_p99_ms=percentile(ttfts, 99),
        ttft_avg_ms=mean(ttfts),
        latency_p50_ms=percentile(lats, 50),
        latency_p95_ms=percentile(lats, 95),
        latency_p99_ms=percentile(lats, 99),
        latency_avg_ms=mean(lats),
        kv_tokens_per_user_budget=kv_tokens_per_user_budget,
        kv_required_tokens_per_user=kv_required_tokens_per_user,
        kv_headroom_tokens_per_user=kv_headroom,
        kv_budget_ok=kv_headroom >= 0,
    )


def print_table(results: List[LevelResult], ttft_target_ms: float) -> None:
    print("\n" + "=" * 148)
    print("TTFT + KV CAPACITY")
    print("=" * 148)
    print(
        f"{'Conc':>5} | {'Req':>4} | {'OK':>4} | {'Fail':>4} | "
        f"{'TTFT p50':>9} | {'TTFT p95':>9} | {'TTFT p99':>9} | "
        f"{'Tok/s':>9} | {'Lat p50':>8} | {'KV/user':>8} | {'KV need':>8} | {'KV ok':>5}"
    )
    print("-" * 148)
    for r in results:
        print(
            f"{r.concurrency:>5} | {r.requests:>4} | {r.successful_requests:>4} | {r.failed_requests:>4} | "
            f"{r.ttft_p50_ms:>9.1f} | {r.ttft_p95_ms:>9.1f} | {r.ttft_p99_ms:>9.1f} | "
            f"{r.tokens_per_sec:>9.1f} | {r.latency_p50_ms:>8.1f} | "
            f"{r.kv_tokens_per_user_budget:>8} | {r.kv_required_tokens_per_user:>8} | "
            f"{str(r.kv_budget_ok):>5}"
        )
    print("-" * 148)
    good = [
        r
        for r in results
        if r.failed_requests == 0 and r.ttft_p95_ms <= ttft_target_ms and r.kv_budget_ok
    ]
    if good:
        best = max(good, key=lambda x: x.concurrency)
        print(
            f"Max users under TTFT<= {ttft_target_ms:.0f} ms and KV budget OK: {best.concurrency} "
            f"(TTFT p95={best.ttft_p95_ms:.1f} ms)"
        )
    else:
        print(f"No concurrency level met TTFT<= {ttft_target_ms:.0f} ms and KV-budget constraints.")
    print("=" * 148)


async def main_async() -> int:
    parser = argparse.ArgumentParser(
        description="Find max concurrency under TTFT target and report KV cache per-user budget."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30080)
    parser.add_argument("--concurrency-levels", default="1,2,4,8,16,24,32,40,48,64")
    parser.add_argument("--requests-per-level", type=int, default=24)
    parser.add_argument("--ttft-target-ms", type=float, default=1000.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--kv-safety-factor", type=float, default=1.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--prompt",
        default="Explain in detail how transformer attention works and why KV caching matters for low-latency inference.",
    )
    parser.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="Allow early EOS stopping (default forces full max_new_tokens generation).",
    )
    args = parser.parse_args()

    if aiohttp is None:
        print("Error: aiohttp is not installed. Install with `pip install aiohttp`.")
        return 2

    levels = parse_levels(args.concurrency_levels)
    ignore_eos = not args.no_ignore_eos

    print("\n" + "=" * 80)
    print("TTFT + KV Capacity Benchmark")
    print("=" * 80)
    print(f"Server:               http://{args.host}:{args.port}")
    print(f"Concurrency levels:   {levels}")
    print(f"Requests/level:       {args.requests_per_level}")
    print(f"TTFT target:          {args.ttft_target_ms} ms")
    print(f"Max new tokens:       {args.max_new_tokens}")
    print(f"KV safety factor:     {args.kv_safety_factor}")
    print("=" * 80)

    healthy = await wait_until_healthy(args.host, args.port, timeout_sec=60.0)
    if not healthy:
        print("Error: server is not healthy")
        return 1

    capacity_ctx = await get_capacity_context(args.host, args.port)
    server_info = capacity_ctx["server_info"]
    capacity_mode = capacity_ctx["capacity_mode"]
    worker_count = int(capacity_ctx["worker_count"])
    max_total_num_tokens = int(capacity_ctx["max_total_num_tokens"])
    num_reserved_decode_tokens = int(capacity_ctx["num_reserved_decode_tokens"])
    usable_kv_tokens = max(
        0, int((max_total_num_tokens - num_reserved_decode_tokens) * args.kv_safety_factor)
    )
    if capacity_mode == "router_workers":
        print(
            f"KV capacity (router workers={worker_count}): "
            f"cluster_max_total_num_tokens={max_total_num_tokens}, "
            f"cluster_reserved_decode={num_reserved_decode_tokens}, "
            f"cluster_usable={usable_kv_tokens}"
        )
    elif capacity_mode == "direct":
        print(
            f"KV capacity: max_total_num_tokens={max_total_num_tokens}, "
            f"reserved_decode={num_reserved_decode_tokens}, usable={usable_kv_tokens}"
        )
    else:
        print(
            "Warning: unable to resolve KV capacity from server/router metadata; "
            "KV budget will be reported as 0."
        )

    # Warmup requests.
    async with aiohttp.ClientSession() as session:
        warmup_payload = {
            "text": "Warmup request",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0, "ignore_eos": True},
        }
        for _ in range(max(args.warmup, 0)):
            try:
                await send_stream_request(
                    session,
                    f"http://{args.host}:{args.port}/generate",
                    warmup_payload,
                    timeout_sec=min(args.timeout, 30.0),
                )
            except Exception:
                pass

    prompt_tokens = await estimate_prompt_tokens(
        args.host, args.port, args.prompt, timeout_sec=min(args.timeout, 30.0)
    )
    required_per_user = prompt_tokens + args.max_new_tokens
    print(
        f"Prompt tokens={prompt_tokens}, required KV/user ~= prompt+gen={required_per_user}"
    )

    results: List[LevelResult] = []
    for conc in levels:
        # Ensure we actually reach the target concurrency at least once.
        # Otherwise, if requests_per_level < concurrency, measured load is capped
        # by request count and does not reflect the intended concurrency level.
        effective_requests = max(args.requests_per_level, conc)
        kv_per_user_budget = usable_kv_tokens // conc if conc > 0 else 0
        print(
            f"Running concurrency={conc} with requests={effective_requests} ..."
        )
        level = await run_level(
            host=args.host,
            port=args.port,
            concurrency=conc,
            requests_count=effective_requests,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            ignore_eos=ignore_eos,
            timeout_sec=args.timeout,
            kv_tokens_per_user_budget=kv_per_user_budget,
            kv_required_tokens_per_user=required_per_user,
        )
        results.append(level)
        print(
            f"  ttft_p95={level.ttft_p95_ms:.1f}ms tok/s={level.tokens_per_sec:.1f} "
            f"kv_user={level.kv_tokens_per_user_budget} kv_ok={level.kv_budget_ok} fail={level.failed_requests}"
        )

    print_table(results, args.ttft_target_ms)
    good = [
        r
        for r in results
        if r.failed_requests == 0 and r.ttft_p95_ms <= args.ttft_target_ms and r.kv_budget_ok
    ]
    max_users = max([r.concurrency for r in good], default=0)

    output = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "server": {
            "host": args.host,
            "port": args.port,
            "capacity_mode": capacity_mode,
            "worker_count": worker_count,
            "model_path": server_info.get("model_path"),
            "quantization": server_info.get("quantization"),
            "max_total_num_tokens": max_total_num_tokens,
            "num_reserved_decode_tokens": num_reserved_decode_tokens,
            "usable_kv_tokens": usable_kv_tokens,
        },
        "config": {
            "concurrency_levels": levels,
            "requests_per_level": args.requests_per_level,
            "ttft_target_ms": args.ttft_target_ms,
            "max_new_tokens": args.max_new_tokens,
            "prompt_tokens": prompt_tokens,
            "kv_required_tokens_per_user": required_per_user,
            "ignore_eos": ignore_eos,
            "kv_safety_factor": args.kv_safety_factor,
        },
        "max_users_under_target": max_users,
        "results": [asdict(r) for r in results],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Saved: {out_path}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
