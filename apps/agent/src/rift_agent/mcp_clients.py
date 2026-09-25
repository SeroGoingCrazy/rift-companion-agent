"""MCP clients for booking-mcp and knowledge-mcp.

- One short-lived session per call (both servers are stateless over Streamable HTTP),
  so no connection state has to survive across LangGraph tasks.
- Every call runs in a ``tool:<name>`` span and sends ``_meta.trace_id`` /
  ``parent_span_id`` / ``session_id`` so the server continues the same trace.
- Tool errors (``{"error": {"code": ...}}``) become Python exceptions
  (``SLOT_CONFLICT`` -> ``SlotConflictError`` ...); transport failures become
  ``McpUnavailable``.
- Tests pass an in-process connector (``inprocess_connector(server)``) instead of HTTP.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_connected_server_and_client_session

from rift_common.trace import current_trace, span

Connector = Callable[[], AbstractAsyncContextManager[ClientSession]]


# --- errors ------------------------------------------------------------------------------


class ToolError(Exception):
    """A tool call failed; ``code`` is the server's error code."""

    code = "TOOL_ERROR"

    def __init__(
        self, message: str, *, code: str | None = None, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details or {}


class SlotConflictError(ToolError):
    code = "SLOT_CONFLICT"


class NotFoundError(ToolError):
    code = "NOT_FOUND"


class ForbiddenError(ToolError):
    code = "FORBIDDEN"


class InvalidArgumentError(ToolError):
    code = "INVALID_ARGUMENT"


class McpUnavailable(ToolError):
    """The server could not be reached or the connection broke."""

    code = "UNAVAILABLE"


_ERRORS: dict[str, type[ToolError]] = {
    cls.code: cls
    for cls in (SlotConflictError, NotFoundError, ForbiddenError, InvalidArgumentError)
}


def error_from_payload(payload: dict[str, Any]) -> ToolError:
    nested = payload.get("error")
    err: dict[str, Any] = nested if isinstance(nested, dict) else payload
    code = str(err.get("code", "TOOL_ERROR"))
    cls = _ERRORS.get(code, ToolError)
    return cls(str(err.get("message", code)), code=code, details=err.get("details") or {})


# --- connectors ----------------------------------------------------------------------------


def http_connector(url: str, *, timeout_s: float = 15.0) -> Connector:
    @asynccontextmanager
    async def connect() -> AsyncIterator[ClientSession]:
        async with (
            httpx.AsyncClient(timeout=timeout_s) as http,
            streamable_http_client(url, http_client=http) as (read, write, _),
            ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=timeout_s)
            ) as session,
        ):
            await session.initialize()
            yield session

    return connect


def inprocess_connector(server: Server[Any, Any]) -> Connector:
    """Talk to a server object in this process (tests, single-process demos)."""

    @asynccontextmanager
    async def connect() -> AsyncIterator[ClientSession]:
        async with create_connected_server_and_client_session(server) as session:
            yield session

    return connect


def _leaf(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


# --- generic client ------------------------------------------------------------------------


@dataclass
class McpToolClient:
    name: str
    connect: Connector

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call ``tool``; returns its structured result or raises ``ToolError``."""
        with span(f"tool:{tool}", kind="tool", server=self.name, args=arguments) as s:
            meta: dict[str, Any] = {}
            trace = current_trace()
            if trace is not None:
                meta = {"trace_id": trace.trace_id, "parent_span_id": s.span_id}
                if trace.session_id:
                    meta["session_id"] = trace.session_id
            try:
                async with self.connect() as session:
                    result = await session.call_tool(tool, arguments, meta=meta or None)
            except ToolError:
                raise
            except Exception as exc:  # transport errors arrive wrapped in exception groups
                leaf = _leaf(exc)
                if isinstance(leaf, ToolError):
                    raise leaf from exc
                s.set_attr("error_code", McpUnavailable.code)
                raise McpUnavailable(
                    f"{self.name} unavailable: {type(leaf).__name__}: {leaf}"
                ) from exc

            payload = result.structuredContent
            text = "\n".join(getattr(c, "text", "") for c in result.content).strip()
            if payload is None:
                try:
                    decoded = json.loads(text) if text else {}
                    payload = decoded if isinstance(decoded, dict) else {"result": decoded}
                except json.JSONDecodeError:
                    payload = {"text": text}
            if result.isError:
                error = error_from_payload(payload if "error" in payload else {"message": text})
                s.set_attrs(error_code=error.code, error=error.message)
                raise error
            return payload


# --- booking -------------------------------------------------------------------------------


def _iso(value: datetime | str) -> str:
    return value.isoformat() if isinstance(value, datetime) else value


@dataclass
class BookingClient:
    client: McpToolClient

    async def find_companions(self, **filters: Any) -> dict[str, Any]:
        args = {k: v for k, v in filters.items() if v is not None}
        args["start_time"] = _iso(args["start_time"])
        return await self.client.call("find_companions", args)

    async def quote_price(
        self, companion_id: int, service_type: str, duration_hours: float
    ) -> dict[str, Any]:
        return await self.client.call(
            "quote_price",
            {
                "companion_id": companion_id,
                "service_type": service_type,
                "duration_hours": duration_hours,
            },
        )

    async def create_booking(
        self,
        *,
        user_id: int,
        companion_id: int,
        start_time: datetime | str,
        duration_hours: float,
        game_mode: str,
        service_type: str | None = None,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "user_id": user_id,
            "companion_id": companion_id,
            "start_time": _iso(start_time),
            "duration_hours": duration_hours,
            "game_mode": game_mode,
        }
        if service_type:
            args["service_type"] = service_type
        return await self.client.call("create_booking", args)

    async def list_my_bookings(
        self, user_id: int, status: str | None = None
    ) -> list[dict[str, Any]]:
        args: dict[str, Any] = {"user_id": user_id}
        if status:
            args["status"] = status
        out = await self.client.call("list_my_bookings", args)
        bookings: list[dict[str, Any]] = out.get("bookings", [])
        return bookings

    async def cancel_booking(
        self, user_id: int, booking_id: int, *, dry_run: bool
    ) -> dict[str, Any]:
        return await self.client.call(
            "cancel_booking", {"user_id": user_id, "booking_id": booking_id, "dry_run": dry_run}
        )

    # --- web support ---------------------------------------------------------------------

    async def ensure_user(self, nickname: str) -> dict[str, Any]:
        return await self.client.call("ensure_user", {"nickname": nickname})

    async def browse_companions(self, **filters: Any) -> list[dict[str, Any]]:
        args = {k: v for k, v in filters.items() if v is not None}
        out = await self.client.call("browse_companions", args)
        companions: list[dict[str, Any]] = out.get("companions", [])
        return companions

    async def pay_booking(self, user_id: int, booking_id: int) -> dict[str, Any]:
        return await self.client.call("pay_booking", {"user_id": user_id, "booking_id": booking_id})


# --- knowledge -----------------------------------------------------------------------------

_REFERENCES = re.compile(r"\*\*References \(JSON\):\*\*\s*```json\s*(\{.*?\})\s*```", re.S)


@dataclass(frozen=True)
class KnowledgeResult:
    collection: str | None
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.text.strip()


def parse_knowledge(payload: dict[str, Any], collection: str | None) -> KnowledgeResult:
    """Split the RAG tool's Markdown answer from its trailing JSON references block."""
    text = str(payload.get("text") or payload.get("content") or "")
    citations = list(payload.get("citations") or [])
    m = _REFERENCES.search(text)
    if m:
        with contextlib.suppress(json.JSONDecodeError):
            citations = citations or list(json.loads(m.group(1)).get("citations", []))
        text = text[: m.start()].rstrip().removesuffix("---").rstrip()
    return KnowledgeResult(collection, text, citations)


@dataclass
class KnowledgeClient:
    client: McpToolClient
    collections: tuple[str, ...] = ()

    async def query(
        self, query: str, *, collection: str | None = None, top_k: int = 3
    ) -> KnowledgeResult:
        args: dict[str, Any] = {"query": query, "top_k": top_k}
        if collection:
            args["collection"] = collection
        return parse_knowledge(await self.client.call("query_knowledge_hub", args), collection)

    async def query_all(self, query: str, *, top_k: int = 3) -> list[KnowledgeResult]:
        """Search every configured collection (one call each), dropping empty results."""
        results = []
        for collection in self.collections or (None,):
            r = await self.query(query, collection=collection, top_k=top_k)
            if not r.empty:
                results.append(r)
        return results
