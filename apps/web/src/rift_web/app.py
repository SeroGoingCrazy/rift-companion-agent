"""FastAPI application factory.

``create_app(config)`` builds real services in the lifespan (DeepSeek, MCP clients,
checkpointed agent); tests pass ready-made ``services`` instead.

Run::

    uv run python -m rift_web --port 8000                       # booking-mcp over HTTP
    uv run python -m rift_web --booking-db data/booking.db      # booking-mcp in-process
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rift_web.routes import auth as auth_routes
from rift_web.routes import bookings as booking_routes
from rift_web.routes import chat as chat_routes
from rift_web.services import WebConfig, WebServices, build_services

PACKAGE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")


def create_app(config: WebConfig | None = None, *, services: WebServices | None = None) -> FastAPI:
    config = config or WebConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        async with build_services(config) as built:
            app.state.services = built
            yield

    app = FastAPI(title="峡谷陪练", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.templates = templates
    if services is not None:
        app.state.services = services  # also available without running the lifespan
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    app.include_router(auth_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(booking_routes.router)
    return app


def main(argv: Sequence[str] | None = None) -> int:
    import uvicorn

    p = argparse.ArgumentParser(prog="rift-web", description="峡谷陪练 web front-end")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--booking-db", default=None, help="run booking-mcp in-process on this DB")
    p.add_argument("--now", type=datetime.fromisoformat, default=None, help="fixed clock")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr)
    origins = (f"http://localhost:{args.port}", f"http://127.0.0.1:{args.port}")
    config = WebConfig(booking_db=args.booking_db, now=args.now, allowed_origins=origins)
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")
    return 0
