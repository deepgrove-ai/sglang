#!/usr/bin/env python3
"""
Lightweight reverse-proxy server for the Minimal Chat UI.

Serves static files from this directory AND proxies /v1/*, /health, /generate
to the SGLang backend. Everything on one port — no CORS issues.

Usage:
    python3 minimal-chat/serve.py                         # defaults: port=8080, backend=127.0.0.1:30080
    python3 minimal-chat/serve.py --port 4173 --backend-port 30080
    python3 minimal-chat/serve.py --port 8080 --backend-port 30080 --host 0.0.0.0

Testers visit:  http://<public-ip>:8080
"""

import argparse
import asyncio
import logging
import mimetypes
import os
import sys
from pathlib import Path

try:
    import aiohttp
    from aiohttp import web
except ImportError:
    print("ERROR: aiohttp is required.  pip install aiohttp")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("chat-proxy")

STATIC_DIR = Path(__file__).resolve().parent  # minimal-chat/

# Paths to proxy through to the SGLang backend
PROXY_PREFIXES = ("/v1/", "/health", "/generate", "/get_model_info", "/get_server_info")


def create_app(backend_url: str) -> web.Application:
    app = web.Application(client_max_size=10 * 1024 * 1024)  # 10MB max request
    app["backend_url"] = backend_url
    app["session"] = None

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    # Proxy routes FIRST (order matters)
    for prefix in PROXY_PREFIXES:
        app.router.add_route("*", prefix + "{path:.*}", proxy_handler)
        if not prefix.endswith("/"):
            app.router.add_route("*", prefix, proxy_handler)

    # Static files LAST (catch-all)
    app.router.add_get("/{path:.*}", static_handler)

    return app


async def on_startup(app):
    timeout = aiohttp.ClientTimeout(total=120)
    app["session"] = aiohttp.ClientSession(timeout=timeout)
    logger.info(f"Backend: {app['backend_url']}")


async def on_cleanup(app):
    if app["session"]:
        await app["session"].close()


async def proxy_handler(request: web.Request) -> web.StreamResponse:
    """Forward request to SGLang backend and stream the response back."""
    backend_url = request.app["backend_url"]
    session: aiohttp.ClientSession = request.app["session"]

    # Build target URL
    target = backend_url + request.path
    if request.query_string:
        target += "?" + request.query_string

    # Read body
    body = await request.read() if request.can_read_body else None

    # Forward headers (skip host)
    headers = {}
    for key, val in request.headers.items():
        lower = key.lower()
        if lower in ("host", "content-length"):
            continue
        headers[key] = val

    try:
        async with session.request(
            method=request.method,
            url=target,
            headers=headers,
            data=body,
        ) as backend_resp:
            # Check if streaming (SSE)
            content_type = backend_resp.headers.get("Content-Type", "")
            is_sse = "text/event-stream" in content_type

            if is_sse:
                # Stream SSE events back to client
                response = web.StreamResponse(
                    status=backend_resp.status,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
                await response.prepare(request)

                async for chunk in backend_resp.content.iter_any():
                    await response.write(chunk)

                await response.write_eof()
                return response
            else:
                # Non-streaming: read full body and return
                resp_body = await backend_resp.read()
                response = web.Response(
                    status=backend_resp.status,
                    body=resp_body,
                    content_type=content_type or "application/json",
                )
                # Copy relevant headers
                for key in ("X-Request-Id",):
                    if key in backend_resp.headers:
                        response.headers[key] = backend_resp.headers[key]
                return response

    except aiohttp.ClientError as e:
        logger.error(f"Backend error: {e}")
        return web.json_response(
            {"error": f"Backend unavailable: {e}"},
            status=502,
        )


async def static_handler(request: web.Request) -> web.Response:
    """Serve static files from the minimal-chat directory."""
    path = request.match_info.get("path", "")
    if not path or path == "/":
        path = "index.html"

    file_path = STATIC_DIR / path

    # Security: prevent path traversal
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(STATIC_DIR)):
            return web.Response(status=403, text="Forbidden")
    except Exception:
        return web.Response(status=400, text="Bad path")

    if not file_path.is_file():
        # SPA fallback: serve index.html
        file_path = STATIC_DIR / "index.html"
        if not file_path.is_file():
            return web.Response(status=404, text="Not found")

    content_type, _ = mimetypes.guess_type(str(file_path))
    if content_type is None:
        content_type = "application/octet-stream"

    return web.FileResponse(file_path, headers={"Content-Type": content_type})


def main():
    parser = argparse.ArgumentParser(description="Chat UI reverse proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Listen port (default: 8080)")
    parser.add_argument("--backend-host", default="127.0.0.1", help="SGLang host")
    parser.add_argument("--backend-port", type=int, default=30080, help="SGLang port")
    args = parser.parse_args()

    backend_url = f"http://{args.backend_host}:{args.backend_port}"
    app = create_app(backend_url)

    logger.info(f"Serving chat UI on http://{args.host}:{args.port}")
    logger.info(f"Proxying API to {backend_url}")
    logger.info(f"Static files from {STATIC_DIR}")

    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
