"""booking-mcp server: the five booking tools over MCP (stdio or Streamable HTTP).

- Input is validated with the tools' Pydantic models (JSON Schema published via
  ``tools/list``); output is returned as structured content plus a JSON text block.
- Business errors come back as tool errors (``isError: true``) whose structured content is
  ``{"error": {"code": "SLOT_CONFLICT", "jsonrpc_code": -32001, "message": ...}}``, so
  both LLM clients and the agent can read them.
- ``_meta.trace_id`` (and optional ``_meta.parent_span_id``) continue the caller's trace:
  the tool call becomes a ``tool:<name>`` span under it.
- Logs go to stderr only; stdout belongs to the stdio transport.

Usage::

    python -m booking_mcp --transport stdio --db data/booking.db
    python -m booking_mcp --transport http --port 8101
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, inspect, select
from starlette.applications import Starlette
from starlette.routing import Route

from booking_mcp import __version__
from booking_mcp.db import Companion, create_db_engine, init_db, make_session_factory, read_session
from booking_mcp.errors import BookingError, ErrorCode, InvalidArgument, NotFound
from booking_mcp.seed import seed_database
from booking_mcp.service import BookingService
from booking_mcp.tools import (
    cancel_booking,
    create_booking,
    find_companions,
    list_my_bookings,
    quote_price,
)
from booking_mcp.tools.schemas import (
    BookingListOut,
    BookingOut,
    CancelBookingInput,
    CancelOut,
    CreateBookingInput,
    FindCompanionsInput,
    FindCompanionsOutput,
    ListMyBookingsInput,
    QuoteOut,
    QuotePriceInput,
)
from rift_common.trace import detached, span, start_turn
from rift_common.trace.sinks.base import TraceSink
from rift_domain.config import load_domain_config

logger = logging.getLogger("booking_mcp")

SERVER_NAME = "booking-mcp"
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    run: Callable[[BookingService, Any], BaseModel]

    def tool(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_model.model_json_schema(),
            outputSchema=self.output_model.model_json_schema(mode="serialization"),
        )


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        find_companions.NAME,
        find_companions.DESCRIPTION,
        FindCompanionsInput,
        FindCompanionsOutput,
        find_companions.find_companions,
    ),
    ToolSpec(
        quote_price.NAME,
        quote_price.DESCRIPTION,
        QuotePriceInput,
        QuoteOut,
        quote_price.quote_price,
    ),
    ToolSpec(
        create_booking.NAME,
        create_booking.DESCRIPTION,
        CreateBookingInput,
        BookingOut,
        create_booking.create_booking,
    ),
    ToolSpec(
        list_my_bookings.NAME,
        list_my_bookings.DESCRIPTION,
        ListMyBookingsInput,
        BookingListOut,
        list_my_bookings.list_my_bookings,
    ),
    ToolSpec(
        cancel_booking.NAME,
        cancel_booking.DESCRIPTION,
        CancelBookingInput,
        CancelOut,
        cancel_booking.cancel_booking,
    ),
)


def error_result(error: BookingError) -> types.CallToolResult:
    payload = {"error": error.to_dict()}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structuredContent=payload,
        isError=True,
    )


def _validation_error(exc: ValidationError) -> InvalidArgument:
    problems = [
        f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
    ]
    return InvalidArgument("invalid arguments: " + "; ".join(problems), errors=problems)


def execute(service: BookingService, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate, run and serialize one tool call; raises ``BookingError``."""
    spec = next((t for t in TOOLS if t.name == name), None)
    if spec is None:
        raise NotFound(f"unknown tool {name!r}", available=[t.name for t in TOOLS])
    try:
        args = spec.input_model.model_validate(arguments)
    except ValidationError as exc:
        raise _validation_error(exc) from exc
    result: dict[str, Any] = spec.run(service, args).model_dump(mode="json")
    return result


def _meta_value(meta: Any, key: str) -> str | None:
    if meta is None:
        return None
    value = getattr(meta, key, None)
    if value is None and hasattr(meta, "model_extra") and meta.model_extra:
        value = meta.model_extra.get(key)
    return str(value) if value else None


def build_server(service: BookingService, sinks: Sequence[TraceSink] = ()) -> Server[Any, Any]:
    server: Server[Any, Any] = Server(SERVER_NAME, version=__version__)

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[types.Tool]:
        return [t.tool() for t in TOOLS]

    @server.call_tool(validate_input=False)  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
        meta = server.request_context.meta
        trace_id = _meta_value(meta, "trace_id")
        with contextlib.ExitStack() as stack:
            stack.enter_context(detached())  # never nest into an in-process caller's trace
            if trace_id:
                stack.enter_context(
                    start_turn(
                        trace_id=trace_id,
                        parent_span_id=_meta_value(meta, "parent_span_id"),
                        session_id=_meta_value(meta, "session_id"),
                        sinks=sinks,
                        service=SERVER_NAME,
                    )
                )
            tool_span = stack.enter_context(span(f"tool:{name}", kind="tool", args=arguments))
            try:
                result = await anyio.to_thread.run_sync(execute, service, name, arguments)
            except BookingError as exc:
                logger.info("tool %s failed: %s %s", name, exc.code.value, exc.message)
                tool_span.set_attrs(error_code=exc.code.value, error=exc.message)
                return error_result(exc)
            except Exception:
                logger.exception("tool %s crashed", name)
                tool_span.set_attr("error_code", ErrorCode.INTERNAL.value)
                return error_result(BookingError("internal error"))
            tool_span.set_attr("result_summary", _summary(result))
            return result

    return server


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    """Small, span-friendly view of a result."""
    if "candidates" in result:
        return {
            "candidates": [c["companion_id"] for c in result["candidates"]],
            "relaxations": result["relaxations"],
        }
    if "bookings" in result:
        return {"bookings": len(result["bookings"])}
    return {k: v for k, v in result.items() if not isinstance(v, dict | list)}


def create_http_app(server: Server[Any, Any]) -> Starlette:
    """Stateless Streamable HTTP app serving MCP at ``/mcp``."""
    manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(
        routes=[Route("/mcp", endpoint=StreamableHTTPASGIApp(manager))], lifespan=lifespan
    )


# --- CLI -------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="booking-mcp", description="Booking MCP server")
    p.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--db", default=str(REPO_ROOT / "data" / "booking.db"))
    p.add_argument("--domain", default=str(REPO_ROOT / "config" / "domain.yaml"))
    p.add_argument(
        "--seed-if-empty", action="store_true", help="seed demo data when the DB is empty"
    )
    p.add_argument(
        "--now", type=datetime.fromisoformat, default=None, help="fixed clock (demos / eval)"
    )
    p.add_argument(
        "--embedding",
        choices=("none", "mock", "settings"),
        default="none",
        help="style similarity backend (settings = config/settings.yaml embedding)",
    )
    p.add_argument("--trace-db", default=None, help="write tool spans to this SQLite file")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def build_service(args: argparse.Namespace) -> BookingService:
    domain = load_domain_config(args.domain)
    engine = create_db_engine(args.db)
    factory = make_session_factory(engine)
    empty = not inspect(engine).has_table("companions")
    if not empty:
        with read_session(factory) as s:
            empty = not s.scalar(select(func.count()).select_from(Companion))
    if empty and args.seed_if_empty:
        start = (args.now or datetime.now()).date()
        summary = seed_database(engine, factory, domain, start_date=start)
        logger.info("seeded %s companions from %s", summary.companions, start)
    elif empty:
        init_db(engine)
        logger.warning("booking DB %s is empty; run scripts/seed_all.py", args.db)

    embedder = None
    if args.embedding != "none":
        from rift_common.embedding import EmbeddingFactory
        from rift_common.settings import EmbeddingConfig, load_settings

        cfg = (
            EmbeddingConfig(provider="mock")
            if args.embedding == "mock"
            else load_settings(REPO_ROOT / "config" / "settings.yaml").embedding
        )
        embedder = EmbeddingFactory.create(cfg)

    clock = datetime.now if args.now is None else fixed_clock(args.now)
    return BookingService(factory, domain, embedder=embedder, clock=clock)


def fixed_clock(now: datetime) -> Callable[[], datetime]:
    return lambda: now


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    service = build_service(args)
    sinks: list[TraceSink] = []
    if args.trace_db:
        from rift_common.trace.sinks.sqlite import SqliteSink

        sinks.append(SqliteSink(args.trace_db))
    server = build_server(service, sinks)

    if args.transport == "stdio":
        from mcp.server.stdio import stdio_server

        async def run_stdio() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        anyio.run(run_stdio)
    else:
        import uvicorn

        logger.info("booking-mcp listening on http://%s:%s/mcp", args.host, args.port)
        uvicorn.run(create_http_app(server), host=args.host, port=args.port, log_level="warning")
    return 0


__all__ = ["TOOLS", "build_server", "create_http_app", "execute", "main"]
