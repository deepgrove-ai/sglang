#!/usr/bin/env python3
"""
Hybrid runtime router:
- Routes low in-flight load to ternary backend (best low-latency/tok-s at C=1).
- Routes higher in-flight load to fp16 backend (best high-concurrency capacity).
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse


@dataclass
class RouterConfig:
    ternary_url: str
    fp16_url: str
    switch_inflight_threshold: int
    force_backend: str  # auto|ternary|fp16
    timeout_sec: float


class HybridState:
    def __init__(self, cfg: RouterConfig):
        self.cfg = cfg
        self.inflight = 0
        self.lock = asyncio.Lock()
        self.request_counter = 0
        self.ternary_routed = 0
        self.fp16_routed = 0
        self.last_ternary_health: Optional[bool] = None
        self.last_fp16_health: Optional[bool] = None
        self.started_at = time.time()
        self.session: Optional[aiohttp.ClientSession] = None

    async def ensure_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_sec)
            connector = aiohttp.TCPConnector(limit=2000)
            self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self.session

    async def close(self) -> None:
        if self.session is not None and not self.session.closed:
            await self.session.close()


def choose_backend_name(state: HybridState, inflight_after_increment: int) -> str:
    force = state.cfg.force_backend
    if force in ("ternary", "fp16"):
        return force
    if inflight_after_increment <= state.cfg.switch_inflight_threshold:
        return "ternary"
    return "fp16"


def backend_url(state: HybridState, name: str) -> str:
    return state.cfg.ternary_url if name == "ternary" else state.cfg.fp16_url


async def probe_health(session: aiohttp.ClientSession, base_url: str) -> bool:
    try:
        async with session.get(f"{base_url}/health") as resp:
            return resp.status == 200
    except Exception:
        return False


def create_app(cfg: RouterConfig) -> FastAPI:
    app = FastAPI()
    state = HybridState(cfg)

    @app.on_event("startup")
    async def _startup() -> None:
        await state.ensure_session()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await state.close()

    @app.get("/health")
    async def health() -> Response:
        session = await state.ensure_session()
        ternary_ok, fp16_ok = await asyncio.gather(
            probe_health(session, state.cfg.ternary_url),
            probe_health(session, state.cfg.fp16_url),
        )
        state.last_ternary_health = ternary_ok
        state.last_fp16_health = fp16_ok
        status = 200 if (ternary_ok or fp16_ok) else 503
        body = {
            "ok": status == 200,
            "ternary_ok": ternary_ok,
            "fp16_ok": fp16_ok,
            "inflight": state.inflight,
            "switch_inflight_threshold": state.cfg.switch_inflight_threshold,
            "force_backend": state.cfg.force_backend,
        }
        return JSONResponse(status_code=status, content=body)

    @app.get("/routing_stats")
    async def routing_stats() -> Dict[str, Any]:
        uptime = time.time() - state.started_at
        return {
            "uptime_sec": uptime,
            "inflight": state.inflight,
            "switch_inflight_threshold": state.cfg.switch_inflight_threshold,
            "force_backend": state.cfg.force_backend,
            "request_counter": state.request_counter,
            "ternary_routed": state.ternary_routed,
            "fp16_routed": state.fp16_routed,
            "last_ternary_health": state.last_ternary_health,
            "last_fp16_health": state.last_fp16_health,
            "ternary_url": state.cfg.ternary_url,
            "fp16_url": state.cfg.fp16_url,
        }

    @app.get("/workers")
    async def workers() -> Dict[str, Any]:
        session = await state.ensure_session()
        ternary_ok, fp16_ok = await asyncio.gather(
            probe_health(session, state.cfg.ternary_url),
            probe_health(session, state.cfg.fp16_url),
        )
        return {
            "workers": [
                {
                    "id": state.cfg.ternary_url,
                    "url": state.cfg.ternary_url,
                    "model_id": "ternary",
                    "is_healthy": ternary_ok,
                },
                {
                    "id": state.cfg.fp16_url,
                    "url": state.cfg.fp16_url,
                    "model_id": "fp16",
                    "is_healthy": fp16_ok,
                },
            ],
            "total": 2,
            "stats": {
                "regular_count": 2,
                "healthy_count": int(ternary_ok) + int(fp16_ok),
            },
        }

    @app.get("/get_server_info")
    async def get_server_info() -> Dict[str, Any]:
        return {
            "mode": "hybrid_router",
            "switch_inflight_threshold": state.cfg.switch_inflight_threshold,
            "force_backend": state.cfg.force_backend,
            "inflight": state.inflight,
            "ternary_url": state.cfg.ternary_url,
            "fp16_url": state.cfg.fp16_url,
        }

    async def route_request(payload: Dict[str, Any]) -> Tuple[str, int, Dict[str, str], bytes]:
        async with state.lock:
            state.inflight += 1
            inflight_now = state.inflight
            state.request_counter += 1
            chosen = choose_backend_name(state, inflight_now)
            if chosen == "ternary":
                state.ternary_routed += 1
            else:
                state.fp16_routed += 1
        try:
            session = await state.ensure_session()
            base = backend_url(state, chosen)
            url = f"{base}/generate"
            async with session.post(url, json=payload) as resp:
                data = await resp.read()
                headers = {
                    "x-hybrid-backend": chosen,
                    "x-hybrid-inflight-at-route": str(inflight_now),
                }
                ctype = resp.headers.get("content-type")
                if ctype:
                    headers["content-type"] = ctype
                return chosen, resp.status, headers, data
        finally:
            async with state.lock:
                state.inflight -= 1

    @app.post("/generate")
    async def generate(request: Request) -> Response:
        payload = await request.json()
        stream = bool(payload.get("stream", False))

        if not stream:
            _, status, headers, data = await route_request(payload)
            ctype = headers.get("content-type", "")
            if "application/json" in ctype:
                return Response(content=data, status_code=status, headers=headers, media_type="application/json")
            return Response(content=data, status_code=status, headers=headers)

        async with state.lock:
            state.inflight += 1
            inflight_now = state.inflight
            state.request_counter += 1
            chosen = choose_backend_name(state, inflight_now)
            if chosen == "ternary":
                state.ternary_routed += 1
            else:
                state.fp16_routed += 1

        session = await state.ensure_session()
        base = backend_url(state, chosen)
        url = f"{base}/generate"
        backend_resp = await session.post(url, json=payload)

        if backend_resp.status != 200:
            try:
                err = await backend_resp.text()
            finally:
                backend_resp.release()
                async with state.lock:
                    state.inflight -= 1
            return PlainTextResponse(
                content=err[:2048],
                status_code=backend_resp.status,
                headers={
                    "x-hybrid-backend": chosen,
                    "x-hybrid-inflight-at-route": str(inflight_now),
                },
            )

        async def stream_gen():
            try:
                async for chunk in backend_resp.content.iter_chunked(65536):
                    yield chunk
            finally:
                backend_resp.release()
                async with state.lock:
                    state.inflight -= 1

        media_type = backend_resp.headers.get("content-type", "text/event-stream")
        return StreamingResponse(
            stream_gen(),
            status_code=backend_resp.status,
            media_type=media_type,
            headers={
                "x-hybrid-backend": chosen,
                "x-hybrid-inflight-at-route": str(inflight_now),
            },
        )

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid ternary/fp16 runtime router")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30080)
    p.add_argument("--ternary-url", required=True)
    p.add_argument("--fp16-url", required=True)
    p.add_argument("--switch-inflight-threshold", type=int, default=48)
    p.add_argument("--force-backend", choices=["auto", "ternary", "fp16"], default="auto")
    p.add_argument("--timeout-sec", type=float, default=180.0)
    return p.parse_args()


def main() -> int:
    import uvicorn

    args = parse_args()
    cfg = RouterConfig(
        ternary_url=args.ternary_url.rstrip("/"),
        fp16_url=args.fp16_url.rstrip("/"),
        switch_inflight_threshold=args.switch_inflight_threshold,
        force_backend=args.force_backend,
        timeout_sec=args.timeout_sec,
    )
    app = create_app(cfg)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
