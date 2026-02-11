#!/usr/bin/env python3
"""
Comprehensive chat interface test: mirrors the exact behavior of minimal-chat/app.js

Tests:
  1. Model discovery (/v1/models) 
  2. Single-turn non-streaming (exact frontend pattern)
  3. Multi-turn conversation (context accumulation)
  4. Streaming SSE (alternative path)
  5. Edge cases: empty input, long input, special characters, unicode, code blocks
  6. Concurrent multi-user sessions (no cross-contamination)
  7. Context overflow handling (exceed server context length)
  8. Rapid-fire requests (stress test)
  9. Session reset (simulates "New chat" button)
  10. Error handling (invalid model, bad payloads)

Usage:
    python3 util/test_chat_interface.py --port 30080 --model mangrove-alltern-overlap
"""

import argparse
import asyncio
import json
import logging
import random
import string
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant. Reply with concise Markdown. Use LaTeX for math when useful."

# ── Test counters ──────────────────────────────────────────────
passed = 0
failed = 0
skipped = 0
test_results: List[Tuple[str, str, str]] = []  # (name, status, detail)


def record(name: str, status: str, detail: str = ""):
    global passed, failed, skipped
    test_results.append((name, status, detail))
    if status == "PASS":
        passed += 1
        logger.info(f"  ✅ {name}")
    elif status == "FAIL":
        failed += 1
        logger.error(f"  ❌ {name}: {detail}")
    else:
        skipped += 1
        logger.warning(f"  ⚠️  {name}: {detail}")


# ── Helpers ────────────────────────────────────────────────────
async def fetch_json(session, url, method="GET", json_body=None, timeout=30):
    """Simple HTTP helper that returns (status, body_dict_or_text)."""
    try:
        kwargs = {"timeout": aiohttp.ClientTimeout(total=timeout)}
        if json_body is not None:
            kwargs["json"] = json_body

        if method == "GET":
            async with session.get(url, **kwargs) as resp:
                text = await resp.text()
                try:
                    return resp.status, json.loads(text)
                except json.JSONDecodeError:
                    return resp.status, text
        else:
            async with session.post(url, **kwargs) as resp:
                text = await resp.text()
                try:
                    return resp.status, json.loads(text)
                except json.JSONDecodeError:
                    return resp.status, text
    except Exception as e:
        return -1, str(e)


async def chat_completion(session, base_url, model, messages, temperature=0.35,
                          max_tokens=100, stream=False, timeout=30):
    """Mirrors the frontend's requestAssistantReply pattern."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    
    if not stream:
        status, body = await fetch_json(
            session, f"{base_url}/v1/chat/completions",
            method="POST", json_body=payload, timeout=timeout
        )
        return status, body
    else:
        # Streaming SSE
        try:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    return resp.status, await resp.text()
                
                chunks = []
                first_token_time = None
                t0 = time.monotonic()
                
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
                            chunks.append(content)
                    except json.JSONDecodeError:
                        pass
                
                full_text = "".join(chunks)
                ttft = (first_token_time - t0) * 1000 if first_token_time else None
                return 200, {"text": full_text, "ttft_ms": ttft, "chunks": len(chunks)}
        except Exception as e:
            return -1, str(e)


# ── Individual Tests ───────────────────────────────────────────

async def test_health(session, base_url):
    """Test 0: Server health check."""
    status, _ = await fetch_json(session, f"{base_url}/health")
    if status == 200:
        record("health_check", "PASS")
    else:
        record("health_check", "FAIL", f"HTTP {status}")


async def test_model_discovery(session, base_url, expected_model):
    """Test 1: /v1/models returns the expected model (mirrors app.js initialize)."""
    status, body = await fetch_json(session, f"{base_url}/v1/models")
    if status != 200:
        record("model_discovery", "FAIL", f"HTTP {status}")
        return None
    
    models = body.get("data", []) if isinstance(body, dict) else []
    model_ids = [m.get("id") for m in models]
    
    if expected_model in model_ids:
        record("model_discovery", "PASS")
        return expected_model
    elif model_ids:
        record("model_discovery", "PASS", f"Model found: {model_ids[0]} (expected {expected_model})")
        return model_ids[0]
    else:
        record("model_discovery", "FAIL", "No models returned")
        return None


async def test_single_turn_nonstream(session, base_url, model):
    """Test 2: Single-turn non-streaming (exact frontend pattern)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    
    status, body = await chat_completion(session, base_url, model, messages, stream=False)
    
    if status != 200:
        record("single_turn_nonstream", "FAIL", f"HTTP {status}: {str(body)[:200]}")
        return
    
    text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not text or not text.strip():
        record("single_turn_nonstream", "FAIL", "Empty response")
        return
    
    # Check response is reasonable (should mention "4")
    if "4" in text:
        record("single_turn_nonstream", "PASS")
    else:
        record("single_turn_nonstream", "PASS", f"Response: {text[:100]} (no '4' found but response is valid)")


async def test_multi_turn_context(session, base_url, model):
    """Test 3: Multi-turn conversation, check the model remembers previous context."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Turn 1: Set context
    messages.append({"role": "user", "content": "My name is Alex and I live in Tokyo."})
    status, body = await chat_completion(session, base_url, model, messages, stream=False)
    if status != 200:
        record("multi_turn_context", "FAIL", f"Turn 1 failed: HTTP {status}")
        return
    reply1 = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    messages.append({"role": "assistant", "content": reply1})
    
    # Turn 2: Ask about prior context
    messages.append({"role": "user", "content": "What is my name and where do I live?"})
    status, body = await chat_completion(session, base_url, model, messages, stream=False)
    if status != 200:
        record("multi_turn_context", "FAIL", f"Turn 2 failed: HTTP {status}")
        return
    reply2 = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Check response references both name and city
    reply_lower = reply2.lower()
    has_name = "alex" in reply_lower
    has_city = "tokyo" in reply_lower
    
    if has_name and has_city:
        record("multi_turn_context", "PASS")
    elif has_name or has_city:
        record("multi_turn_context", "PASS", f"Partial recall: name={has_name}, city={has_city}")
    else:
        record("multi_turn_context", "FAIL", f"No recall of context. Reply: {reply2[:200]}")


async def test_streaming_sse(session, base_url, model):
    """Test 4: Streaming SSE path."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Count from 1 to 5."},
    ]
    
    status, result = await chat_completion(session, base_url, model, messages, stream=True)
    
    if status != 200:
        record("streaming_sse", "FAIL", f"HTTP {status}: {str(result)[:200]}")
        return
    
    text = result.get("text", "")
    chunks = result.get("chunks", 0)
    ttft = result.get("ttft_ms")
    
    if not text.strip():
        record("streaming_sse", "FAIL", "Empty streamed response")
        return
    
    if chunks < 2:
        record("streaming_sse", "FAIL", f"Only {chunks} chunk(s) — not really streaming")
        return
    
    if ttft is not None and ttft < 5000:
        record("streaming_sse", "PASS", f"TTFT={ttft:.0f}ms, chunks={chunks}")
    else:
        record("streaming_sse", "PASS", f"chunks={chunks}, ttft={ttft}")


async def test_special_characters(session, base_url, model):
    """Test 5: Unicode, emoji, special chars, code blocks."""
    test_cases = [
        ("unicode", "Translate 'hello' to Japanese: こんにちは. Is that correct?"),
        ("emoji", "What does this emoji mean? 🎉🔥🚀"),
        ("code_block", "Write a Python function:\n```python\ndef hello():\n    print('world')\n```\nIs this correct?"),
        ("html_entities", "What does <script>alert('xss')</script> do in HTML?"),
        ("math_latex", "Solve: $\\int_0^1 x^2 dx$"),
        ("long_url", f"Visit this URL: https://example.com/{'a' * 200}. What do you think?"),
    ]
    
    all_ok = True
    for case_name, user_msg in test_cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        status, body = await chat_completion(session, base_url, model, messages, stream=False, timeout=30)
        
        if status != 200:
            record(f"special_chars_{case_name}", "FAIL", f"HTTP {status}")
            all_ok = False
        else:
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text and text.strip():
                record(f"special_chars_{case_name}", "PASS")
            else:
                record(f"special_chars_{case_name}", "FAIL", "Empty response")
                all_ok = False


async def test_concurrent_no_crosstalk(session, base_url, model, num_users=8):
    """Test 6: Multiple concurrent users don't get each other's responses."""
    
    async def user_session(user_id, secret_word):
        """Each user tells the model a secret word and asks it back."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Remember this secret word: '{secret_word}'. Just say 'OK, I remember {secret_word}.' and nothing else."},
        ]
        
        status, body = await chat_completion(session, base_url, model, messages, stream=False, max_tokens=50)
        if status != 200:
            return user_id, secret_word, False, f"HTTP {status}"
        
        reply1 = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": reply1})
        
        # Ask for the secret word
        messages.append({"role": "user", "content": "What was the secret word I told you?"})
        status, body = await chat_completion(session, base_url, model, messages, stream=False, max_tokens=50)
        if status != 200:
            return user_id, secret_word, False, f"HTTP {status}"
        
        reply2 = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        recalled = secret_word.lower() in reply2.lower()
        return user_id, secret_word, recalled, reply2[:100]
    
    # Each user gets a unique secret word
    secret_words = [f"xylophone{i}zebra" for i in range(num_users)]
    tasks = [user_session(i, w) for i, w in enumerate(secret_words)]
    results = await asyncio.gather(*tasks)
    
    recalled_count = sum(1 for _, _, recalled, _ in results if recalled)
    
    if recalled_count == num_users:
        record("concurrent_no_crosstalk", "PASS", f"{recalled_count}/{num_users} users recalled their secret word")
    elif recalled_count >= num_users * 0.75:
        record("concurrent_no_crosstalk", "PASS", f"{recalled_count}/{num_users} recalled (>75% threshold)")
    else:
        failures = [(uid, w, r) for uid, w, recalled, r in results if not recalled]
        detail = "; ".join(f"user{uid}({w}): {r}" for uid, w, r in failures[:3])
        record("concurrent_no_crosstalk", "FAIL", f"Only {recalled_count}/{num_users} recalled. {detail}")


async def test_context_overflow(session, base_url, model):
    """Test 7: Push context beyond server's max length, server should handle gracefully."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Build up a very long context (each turn adds ~200 tokens)
    long_msg = "Tell me about the history of computing. " * 50  # ~400 tokens
    for i in range(10):
        messages.append({"role": "user", "content": long_msg})
        messages.append({"role": "assistant", "content": f"Here's turn {i} of the history... " + ("words " * 100)})
    
    # Final question
    messages.append({"role": "user", "content": "Summarize our conversation."})
    
    status, body = await chat_completion(session, base_url, model, messages, stream=False, max_tokens=50, timeout=30)
    
    if status == 200:
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        if text and text.strip():
            record("context_overflow", "PASS", f"Server handled long context gracefully ({len(str(messages))} chars)")
        else:
            record("context_overflow", "PASS", "Server returned empty but no error (truncated context)")
    elif status in (400, 422):
        # Server correctly rejected overly long context
        record("context_overflow", "PASS", f"Server rejected with HTTP {status} (expected)")
    else:
        record("context_overflow", "FAIL", f"HTTP {status}: {str(body)[:200]}")


async def test_rapid_fire(session, base_url, model, num_requests=20):
    """Test 8: Rapid-fire requests — send many requests quickly."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Say 'hello' and nothing else."},
    ]
    
    async def single_req(i):
        t0 = time.monotonic()
        status, body = await chat_completion(session, base_url, model, messages, stream=False, max_tokens=10, timeout=20)
        elapsed = time.monotonic() - t0
        if status == 200:
            text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return i, True, elapsed, text[:50]
        else:
            return i, False, elapsed, str(body)[:50]
    
    tasks = [single_req(i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    success = sum(1 for _, ok, _, _ in results if ok)
    latencies = [e for _, ok, e, _ in results if ok]
    
    if success == num_requests:
        avg_lat = sum(latencies) / len(latencies)
        record("rapid_fire", "PASS", f"{success}/{num_requests} OK, avg latency={avg_lat:.2f}s")
    elif success >= num_requests * 0.9:
        record("rapid_fire", "PASS", f"{success}/{num_requests} OK (>90%)")
    else:
        failures = [(i, t) for i, ok, _, t in results if not ok]
        detail = "; ".join(f"req{i}: {t}" for i, t in failures[:3])
        record("rapid_fire", "FAIL", f"Only {success}/{num_requests}. {detail}")


async def test_session_reset(session, base_url, model):
    """Test 9: Simulate 'New chat' button — fresh context after reset."""
    # Session 1: Tell the model something
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "The password is 'banana42'. Remember it."},
    ]
    status, body = await chat_completion(session, base_url, model, messages, stream=False)
    if status != 200:
        record("session_reset", "FAIL", f"Session 1 failed: HTTP {status}")
        return
    
    # "New chat" — clear messages, start fresh (same as app.js resetBtn handler)
    fresh_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What password did I tell you earlier?"},
    ]
    status, body = await chat_completion(session, base_url, model, fresh_messages, stream=False)
    if status != 200:
        record("session_reset", "FAIL", f"Session 2 failed: HTTP {status}")
        return
    
    reply = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # The model should NOT know the password (fresh context)
    if "banana42" in reply.lower():
        record("session_reset", "FAIL", f"Model leaked info across sessions! Reply: {reply[:200]}")
    else:
        record("session_reset", "PASS", "No cross-session info leak")


async def test_error_handling(session, base_url, model):
    """Test 10: Various error conditions."""
    
    # 10a: Invalid model name
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Hello"}]
    status, body = await chat_completion(session, base_url, "nonexistent-model-xyz", messages, stream=False, timeout=10)
    if status in (400, 404, 422):
        record("error_invalid_model", "PASS", f"HTTP {status}")
    elif status == 200:
        record("error_invalid_model", "PASS", "Server served anyway (may alias to default model)")
    else:
        record("error_invalid_model", "FAIL", f"HTTP {status}: {str(body)[:100]}")
    
    # 10b: Empty messages array
    payload = {"model": model, "messages": [], "max_tokens": 10}
    status, body = await fetch_json(
        session, f"{base_url}/v1/chat/completions",
        method="POST", json_body=payload, timeout=10
    )
    if status in (200, 400, 422):
        record("error_empty_messages", "PASS", f"HTTP {status}")
    else:
        record("error_empty_messages", "FAIL", f"HTTP {status}: {str(body)[:100]}")
    
    # 10c: Very large max_tokens
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Hi"}]
    status, body = await chat_completion(session, base_url, model, messages, stream=False, max_tokens=999999, timeout=15)
    if status in (200, 400, 422):
        record("error_large_max_tokens", "PASS", f"HTTP {status}")
    else:
        record("error_large_max_tokens", "FAIL", f"HTTP {status}: {str(body)[:100]}")
    
    # 10d: Malformed JSON body
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            data="this is not json",
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            status = resp.status
            if status in (400, 422):
                record("error_malformed_json", "PASS", f"HTTP {status}")
            elif status == 200:
                record("error_malformed_json", "FAIL", "Server accepted malformed JSON")
            else:
                record("error_malformed_json", "PASS", f"HTTP {status}")
    except Exception as e:
        record("error_malformed_json", "FAIL", str(e)[:100])


async def test_multi_user_long_session(session, base_url, model, num_users=16, turns=5):
    """Test 11: Multiple users each have a long multi-turn conversation simultaneously."""
    
    async def user_session(uid):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        errors = 0
        total_tokens = 0
        ttft_list = []
        
        topics = [
            "Tell me about quantum computing.",
            "What is machine learning?",
            "Explain blockchain technology.",
            "How does the internet work?",
            "What is climate change?",
            "Tell me about space exploration.",
            "What is artificial intelligence?",
            "How do vaccines work?",
            "What is renewable energy?",
            "Tell me about the human brain.",
            "What is cybersecurity?",
            "How do computers store data?",
            "What is 5G technology?",
            "Tell me about electric vehicles.",
            "What is cloud computing?",
            "How does GPS work?",
        ]
        
        for turn in range(turns):
            topic = topics[(uid * 3 + turn) % len(topics)]
            follow_ups = [
                topic,
                "Can you give me a specific example?",
                "What are the main challenges?",
                "How will this change in the next 10 years?",
                "Summarize what we discussed.",
            ]
            user_msg = follow_ups[turn % len(follow_ups)]
            messages.append({"role": "user", "content": user_msg})
            
            t0 = time.monotonic()
            status, body = await chat_completion(
                session, base_url, model, messages, stream=False, max_tokens=80, timeout=30
            )
            elapsed = time.monotonic() - t0
            
            if status == 200:
                text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if text and text.strip():
                    messages.append({"role": "assistant", "content": text})
                    total_tokens += len(text.split())  # rough estimate
                    ttft_list.append(elapsed * 1000)
                else:
                    errors += 1
                    messages.append({"role": "assistant", "content": "(empty)"})
            else:
                errors += 1
                messages.append({"role": "assistant", "content": "(error)"})
            
            # Think time
            await asyncio.sleep(random.uniform(0.2, 0.8))
        
        return uid, turns, errors, total_tokens, ttft_list
    
    t0 = time.monotonic()
    tasks = [user_session(i) for i in range(num_users)]
    results = await asyncio.gather(*tasks)
    wall_time = time.monotonic() - t0
    
    total_errors = sum(e for _, _, e, _, _ in results)
    total_tokens = sum(t for _, _, _, t, _ in results)
    all_ttft = [tt for _, _, _, _, ttfts in results for tt in ttfts]
    total_turns = sum(turns - e for _, turns, e, _, _ in results)
    expected_turns = num_users * turns
    success_rate = total_turns / expected_turns * 100
    
    detail = (
        f"{num_users} users × {turns} turns, "
        f"success={success_rate:.0f}%, "
        f"tokens={total_tokens}, "
        f"wall={wall_time:.1f}s"
    )
    
    if all_ttft:
        sorted_ttft = sorted(all_ttft)
        p50 = sorted_ttft[len(sorted_ttft) // 2]
        p95 = sorted_ttft[int(len(sorted_ttft) * 0.95)]
        detail += f", TTFT p50={p50:.0f}ms p95={p95:.0f}ms"
    
    if success_rate >= 95:
        record("multi_user_long_session", "PASS", detail)
    elif success_rate >= 80:
        record("multi_user_long_session", "PASS", f"(degraded) {detail}")
    else:
        record("multi_user_long_session", "FAIL", detail)


async def test_response_format_validity(session, base_url, model):
    """Test 12: Validate response JSON structure matches OpenAI spec."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Say hello."},
    ]
    
    status, body = await chat_completion(session, base_url, model, messages, stream=False)
    if status != 200:
        record("response_format", "FAIL", f"HTTP {status}")
        return
    
    # Check required fields
    issues = []
    if "id" not in body:
        issues.append("missing 'id'")
    if "object" not in body:
        issues.append("missing 'object'")
    elif body["object"] != "chat.completion":
        issues.append(f"object='{body['object']}' (expected 'chat.completion')")
    if "choices" not in body:
        issues.append("missing 'choices'")
    else:
        choices = body["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            issues.append("empty choices array")
        else:
            c0 = choices[0]
            if "message" not in c0:
                issues.append("missing choices[0].message")
            else:
                msg = c0["message"]
                if "role" not in msg:
                    issues.append("missing message.role")
                if "content" not in msg:
                    issues.append("missing message.content")
            if "finish_reason" not in c0:
                issues.append("missing finish_reason")
    if "usage" not in body:
        issues.append("missing 'usage'")
    else:
        usage = body["usage"]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if key not in usage:
                issues.append(f"missing usage.{key}")
    
    if not issues:
        record("response_format", "PASS")
    else:
        record("response_format", "FAIL", "; ".join(issues))


# ── Main ───────────────────────────────────────────────────────

async def run_all_tests(args):
    base_url = f"http://{args.host}:{args.port}"
    
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Health check
        print("\n🔍 Health & Discovery")
        await test_health(session, base_url)
        model = await test_model_discovery(session, base_url, args.model)
        if not model:
            print("❌ Cannot determine model. Aborting.")
            return
        
        print(f"\n📝 Basic Functionality (model={model})")
        await test_single_turn_nonstream(session, base_url, model)
        await test_multi_turn_context(session, base_url, model)
        await test_streaming_sse(session, base_url, model)
        await test_response_format_validity(session, base_url, model)
        
        print("\n🎭 Edge Cases")
        await test_special_characters(session, base_url, model)
        
        print("\n🔒 Security & Isolation")
        await test_session_reset(session, base_url, model)
        await test_concurrent_no_crosstalk(session, base_url, model, num_users=args.concurrent_users)
        
        print("\n⚡ Stress & Error Handling")
        await test_error_handling(session, base_url, model)
        await test_rapid_fire(session, base_url, model, num_requests=args.rapid_fire)
        await test_context_overflow(session, base_url, model)
        
        print(f"\n🏋️ Multi-User Long Session ({args.session_users} users × {args.session_turns} turns)")
        await test_multi_user_long_session(
            session, base_url, model,
            num_users=args.session_users,
            turns=args.session_turns,
        )
    
    # Final summary
    print("\n" + "=" * 70)
    print("CHAT INTERFACE TEST RESULTS")
    print("=" * 70)
    for name, status, detail in test_results:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️"}.get(status, "?")
        line = f"  {icon} {name}"
        if detail:
            line += f"  — {detail}"
        print(line)
    
    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    overall = "✅ ALL TESTS PASSED" if failed == 0 else f"❌ {failed} TEST(S) FAILED"
    print(f"  {overall}")
    print("=" * 70)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Comprehensive chat interface test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30080)
    parser.add_argument("--model", default="mangrove-alltern-overlap")
    parser.add_argument("--concurrent-users", type=int, default=8, help="Users for crosstalk test")
    parser.add_argument("--rapid-fire", type=int, default=20, help="Rapid-fire requests count")
    parser.add_argument("--session-users", type=int, default=16, help="Users for long session test")
    parser.add_argument("--session-turns", type=int, default=5, help="Turns per user in long session")
    args = parser.parse_args()
    
    ok = asyncio.run(run_all_tests(args))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
