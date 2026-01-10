import argparse
import asyncio
import glob
import os
import time
from pathlib import Path

import aiohttp
import requests


def _post_json(url: str, payload: dict | None = None, timeout_s: int = 600) -> requests.Response:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile an SGLang server via /start_profile + /stop_profile and a single /generate request."
    )
    parser.add_argument("--host", default=os.environ.get("SGLANG_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SGLANG_PORT", "30080")))
    parser.add_argument("-c", "--concurrency", type=int, default=4, help="Number of concurrent requests")
    parser.add_argument("-r", "--requests", type=int, default=20, help="Total number of requests to send")
    parser.add_argument("--out-dir", default=os.environ.get("SGLANG_PROFILE_OUT_DIR", "/tmp/long_profile"))
    parser.add_argument("--prompt", default="Write a very long story about optimization.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="Allow early stopping at EOS (default forces full generation).",
    )
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--with-stack", action="store_true", help="Enable stack traces (adds overhead).")
    parser.add_argument("--record-shapes", action="store_true", help="Enable shape recording (adds overhead).")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=0,
        help="If >0, auto-stop profiling after this many forward steps. "
        "If set, /stop_profile may error because profiling already stopped.",
    )
    parser.add_argument("--merge-profiles", action="store_true", help="Merge multi-rank traces on rank0.")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Server: {base_url}")
    print(f"Output dir: {out_dir}")

    # Warmup (helps avoid profiling CUDA graph capture + lazy init).
    for i in range(max(0, args.warmup_iters)):
        print(f"Warmup {i+1}/{args.warmup_iters} ...")
        _post_json(
            f"{base_url}/generate",
            {
                "text": "Warmup",
                "sampling_params": {
                    "max_new_tokens": int(args.warmup_tokens),
                    "temperature": 0.0,
                    "ignore_eos": (not args.no_ignore_eos),
                },
            },
        )

    # Start profiler.
    print("Starting profiler ...")
    start_payload: dict = {
        "output_dir": str(out_dir),
        "activities": ["CPU", "GPU"],
        "with_stack": bool(args.with_stack),
        "record_shapes": bool(args.record_shapes),
        "merge_profiles": bool(args.merge_profiles),
    }
    if args.num_steps and args.num_steps > 0:
        start_payload["num_steps"] = int(args.num_steps)

    resp = _post_json(f"{base_url}/start_profile", start_payload)
    if resp.status_code != 200:
        print(f"/start_profile failed: HTTP {resp.status_code}: {resp.text}")
        return 1

    # Generate concurrent requests to capture steady-state.
    print(f"Sending {args.requests} requests with concurrency {args.concurrency} ...")
    
    async def send_request(session, semaphore, url, payload, req_id):
        async with semaphore:
            try:
                async with session.post(url, json=payload) as resp:
                    await resp.text()
                    return {"id": req_id, "status": resp.status}
            except Exception as e:
                return {"id": req_id, "error": str(e)}
    
    async def run_concurrent():
        semaphore = asyncio.Semaphore(args.concurrency)
        connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(args.requests):
                payload = {
                    "text": args.prompt,
                    "sampling_params": {
                        "max_new_tokens": int(args.max_new_tokens),
                        "temperature": float(args.temperature),
                        "ignore_eos": (not args.no_ignore_eos),
                    },
                }
                tasks.append(send_request(session, semaphore, f"{base_url}/generate", payload, i))
            return await asyncio.gather(*tasks)
    
    t0 = time.time()
    results = asyncio.run(run_concurrent())
    dt = time.time() - t0
    
    successes = sum(1 for r in results if r.get("status") == 200)
    print(f"Completed {successes}/{args.requests} requests in {dt:.3f}s ({successes/dt:.1f} req/s)")
    # Stop profiler.
    print("Stopping profiler ...")
    stop = _post_json(f"{base_url}/stop_profile", {})
    if stop.status_code != 200:
        # This commonly happens if num_steps auto-stopped the profiler already.
        print(f"/stop_profile returned HTTP {stop.status_code}: {stop.text}")

    traces = sorted(glob.glob(str(out_dir / "*.trace.json.gz")))
    if traces:
        print("Trace files:")
        for p in traces[-10:]:
            print(" ", p)
    else:
        print(f"No *.trace.json.gz found in {out_dir} (yet). Check server logs and re-list the dir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
