#!/usr/bin/env python3
"""
Load test: Simulate fake chat users hitting the OpenAI-compatible API.

Each "user" sends a multi-turn chat conversation, with think time between turns.
This tests the full serving stack: tokenization, prefill, decode, KV cache management,
hierarchical cache eviction/reload.

Usage:
    python3 util/loadtest_chat_users.py --port 30080 --num-users 32 --turns 3 --think-time 1.0
"""
import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = "You are a helpful assistant. Keep your answers concise (under 100 words)."

USER_MESSAGES = [
    "What is the capital of France?",
    "Tell me a fun fact about the Eiffel Tower.",
    "How tall is the Eiffel Tower in meters?",
    "What is the best time of year to visit Paris?",
    "Can you recommend a good restaurant near the Eiffel Tower?",
    "What other landmarks should I visit in Paris?",
    "How do I get from the airport to the city center?",
    "What is the weather like in Paris in spring?",
    "Tell me about the history of the Louvre.",
    "What are some common French phrases for tourists?",
    "How much does a typical meal cost in Paris?",
    "What is the best way to get around Paris?",
    "Tell me about French cuisine.",
    "What souvenirs should I buy from Paris?",
    "Can you summarize our conversation so far?",
]


@dataclass
class UserResult:
    user_id: int
    turns_completed: int = 0
    ttft_ms: List[float] = field(default_factory=list)
    total_tokens: int = 0
    errors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


async def simulate_user(
    session: aiohttp.ClientSession,
    user_id: int,
    base_url: str,
    model_name: str,
    num_turns: int,
    think_time: float,
    max_tokens_per_turn: int,
) -> UserResult:
    """Simulate a single chat user with multi-turn conversation."""
    result = UserResult(user_id=user_id)
    result.start_time = time.monotonic()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for turn in range(num_turns):
        # Pick a user message (cycle through the list)
        user_msg = USER_MESSAGES[(user_id * 7 + turn) % len(USER_MESSAGES)]
        messages.append({"role": "user", "content": user_msg})
        
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens_per_turn,
            "temperature": 0.7,
            "stream": True,
        }
        
        try:
            t0 = time.monotonic()
            first_token_time = None
            assistant_text = ""
            token_count = 0
            
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    err_text = await resp.text()
                    logger.warning(f"User {user_id} turn {turn}: HTTP {resp.status}: {err_text[:100]}")
                    result.errors += 1
                    continue
                
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                                result.ttft_ms.append((first_token_time - t0) * 1000)
                            assistant_text += content
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
            
            result.total_tokens += token_count
            result.turns_completed += 1
            messages.append({"role": "assistant", "content": assistant_text})
            
        except asyncio.TimeoutError:
            logger.warning(f"User {user_id} turn {turn}: timeout")
            result.errors += 1
        except Exception as e:
            logger.warning(f"User {user_id} turn {turn}: {e}")
            result.errors += 1
        
        # Think time between turns (simulate real user)
        if turn < num_turns - 1:
            await asyncio.sleep(think_time)
    
    result.end_time = time.monotonic()
    return result


async def run_load_test(args):
    base_url = f"http://{args.host}:{args.port}"
    
    # Health check
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    logger.error(f"Server not healthy: HTTP {resp.status}")
                    return
        except Exception as e:
            logger.error(f"Cannot reach server: {e}")
            return
    
    logger.info(f"Starting load test: {args.num_users} users, {args.turns} turns each, "
                f"think_time={args.think_time}s, max_tokens={args.max_tokens}")
    
    # Stagger user starts to simulate realistic arrival
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(args.num_users):
            task = asyncio.create_task(
                simulate_user(
                    session=session,
                    user_id=i,
                    base_url=base_url,
                    model_name=args.model,
                    num_turns=args.turns,
                    think_time=args.think_time,
                    max_tokens_per_turn=args.max_tokens,
                )
            )
            tasks.append(task)
            # Stagger starts: spread users over ramp_time seconds
            if args.ramp_time > 0 and i < args.num_users - 1:
                await asyncio.sleep(args.ramp_time / args.num_users)
        
        results: List[UserResult] = await asyncio.gather(*tasks)
    
    # Analyze results
    all_ttft = []
    total_tokens = 0
    total_errors = 0
    total_turns = 0
    durations = []
    
    for r in results:
        all_ttft.extend(r.ttft_ms)
        total_tokens += r.total_tokens
        total_errors += r.errors
        total_turns += r.turns_completed
        if r.end_time > r.start_time:
            durations.append(r.end_time - r.start_time)
    
    wall_time = max(r.end_time for r in results) - min(r.start_time for r in results)
    
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Users:              {args.num_users}")
    print(f"Turns per user:     {args.turns}")
    print(f"Think time:         {args.think_time}s")
    print(f"Total turns:        {total_turns} / {args.num_users * args.turns}")
    print(f"Total tokens gen'd: {total_tokens}")
    print(f"Total errors:       {total_errors}")
    print(f"Wall clock time:    {wall_time:.1f}s")
    print(f"Aggregate tok/s:    {total_tokens / wall_time:.0f}")
    
    if all_ttft:
        print(f"\n--- TTFT (Time to First Token) ---")
        print(f"  Mean:   {statistics.mean(all_ttft):.0f}ms")
        print(f"  Median: {statistics.median(all_ttft):.0f}ms")
        print(f"  P95:    {sorted(all_ttft)[int(len(all_ttft) * 0.95)]:.0f}ms")
        print(f"  P99:    {sorted(all_ttft)[min(int(len(all_ttft) * 0.99), len(all_ttft)-1)]:.0f}ms")
        print(f"  Max:    {max(all_ttft):.0f}ms")
    
    if durations:
        avg_session = statistics.mean(durations)
        print(f"\n--- Session Duration ---")
        print(f"  Mean:   {avg_session:.1f}s")
        print(f"  Min:    {min(durations):.1f}s")
        print(f"  Max:    {max(durations):.1f}s")
    
    success_rate = (total_turns / (args.num_users * args.turns)) * 100 if args.num_users * args.turns > 0 else 0
    print(f"\n--- Reliability ---")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Error rate:   {total_errors / (args.num_users * args.turns) * 100:.1f}%")
    
    verdict = "✅ PASS" if success_rate >= 95 and (not all_ttft or statistics.median(all_ttft) < 500) else "❌ FAIL"
    print(f"\nVerdict: {verdict}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Chat load test with fake users")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30080)
    parser.add_argument("--model", default="mangrove-ternary")
    parser.add_argument("--num-users", type=int, default=32)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--think-time", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--ramp-time", type=float, default=5.0, help="Seconds to ramp up all users")
    args = parser.parse_args()
    asyncio.run(run_load_test(args))


if __name__ == "__main__":
    main()
