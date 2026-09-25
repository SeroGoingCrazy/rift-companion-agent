"""D6: booking-mcp over the MCP protocol — in-process, stdio subprocess and HTTP."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
import uvicorn
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.memory import create_connected_server_and_client_session

from booking_mcp.errors import JSONRPC_CODES, ErrorCode
from booking_mcp.seed import seed_database
from booking_mcp.server import build_server, create_http_app, execute
from booking_mcp.service import BookingService
from rift_common.trace import TraceContext
from rift_common.trace.sinks.base import TraceSink

if TYPE_CHECKING:
    from tests.integration.conftest import BookingDB

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 10, 1, 14, 0)
START = datetime(2026, 10, 3, 20, 0)
#: The five booking tools the agent uses (DEV_SPEC 3.5.1) ...
AGENT_TOOLS = {
    "find_companions",
    "quote_price",
    "create_booking",
    "list_my_bookings",
    "cancel_booking",
}
#: ... plus web-support tools (login, order page, companion list).
WEB_TOOLS = {"ensure_user"}
EXPECTED_TOOLS = AGENT_TOOLS | WEB_TOOLS


class RecordingSink(TraceSink):
    def __init__(self) -> None:
        self.traces: list[TraceContext] = []

    def on_turn_end(self, trace: TraceContext) -> None:
        self.traces.append(trace)


@pytest.fixture
def world(booking_db: BookingDB) -> dict[str, int]:
    cid = booking_db.add_companion("阿狸酱")
    booking_db.add_schedule(cid, START - timedelta(hours=2), START + timedelta(hours=4))
    return {
        "companion": cid,
        "alice": booking_db.add_user("alice"),
        "bob": booking_db.add_user("bob"),
    }


@pytest.fixture
def service(booking_db: BookingDB) -> BookingService:
    return BookingService(booking_db.factory, booking_db.domain, clock=lambda: NOW)


@asynccontextmanager
async def _client(
    service: BookingService, sink: TraceSink | None = None
) -> AsyncIterator[ClientSession]:
    server = build_server(service, [sink] if sink else [])
    async with create_connected_server_and_client_session(server) as session:
        yield session


def _booking_args(world: dict[str, int], user: str = "alice") -> dict[str, Any]:
    return {
        "user_id": world[user],
        "companion_id": world["companion"],
        "start_time": START.isoformat(),
        "duration_hours": 2,
        "game_mode": "ranked_solo_duo",
    }


async def test_tools_list_has_booking_tools_with_schemas(service: BookingService) -> None:
    async with _client(service) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} >= AGENT_TOOLS
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for t in tools:
        assert t.description
        assert t.inputSchema["type"] == "object"
        assert t.inputSchema.get("additionalProperties") is False
        assert t.outputSchema is not None
    find = next(t for t in tools if t.name == "find_companions")
    assert set(find.inputSchema["required"]) == {"game_mode", "start_time", "duration_hours"}


async def test_find_quote_book_over_protocol(
    service: BookingService, world: dict[str, int]
) -> None:
    async with _client(service) as client:
        found = await client.call_tool(
            "find_companions",
            {"game_mode": "ranked_solo_duo", "start_time": START.isoformat(), "duration_hours": 2},
        )
        assert not found.isError
        assert found.structuredContent is not None
        [cand] = found.structuredContent["candidates"]
        assert cand["name"] == "阿狸酱"
        assert json.loads(found.content[0].text)["candidates"][0]["name"] == "阿狸酱"  # type: ignore[union-attr]

        quote = await client.call_tool(
            "quote_price",
            {"companion_id": cand["companion_id"], "service_type": "climb", "duration_hours": 2},
        )
        assert quote.structuredContent is not None
        assert quote.structuredContent["total"] == "120.00"  # 50 x 1.2 x 2

        booked = await client.call_tool("create_booking", _booking_args(world))
        assert not booked.isError
        assert booked.structuredContent is not None
        assert booked.structuredContent["status"] == "pending_payment"
        assert booked.structuredContent["total"] == "120.00"

        listed = await client.call_tool("list_my_bookings", {"user_id": world["alice"]})
        assert listed.structuredContent is not None
        assert len(listed.structuredContent["bookings"]) == 1


async def test_conflict_maps_to_slot_conflict(
    service: BookingService, world: dict[str, int]
) -> None:
    async with _client(service) as client:
        await client.call_tool("create_booking", _booking_args(world))
        again = await client.call_tool("create_booking", _booking_args(world, "bob"))
    assert again.isError
    assert again.structuredContent is not None
    error = again.structuredContent["error"]
    assert error["code"] == "SLOT_CONFLICT"
    assert error["jsonrpc_code"] == JSONRPC_CODES[ErrorCode.SLOT_CONFLICT] == -32001
    assert "SLOT_CONFLICT" in again.content[0].text  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("tool", "args", "code"),
    [
        ("cancel_booking", {"user_id": 1, "booking_id": 99}, "NOT_FOUND"),
        ("quote_price", {"companion_id": 1, "service_type": "climb", "duration_hours": 0.3},
         "INVALID_ARGUMENT"),
        ("find_companions",
         {"game_mode": "tft", "start_time": "2026-10-03T20:00", "duration_hours": 2},
         "INVALID_ARGUMENT"),
        ("list_my_bookings", {"user_id": 1, "extra": True}, "INVALID_ARGUMENT"),
        ("no_such_tool", {}, "NOT_FOUND"),
    ],
)  # fmt: skip
async def test_errors_map_to_codes(
    service: BookingService, world: dict[str, int], tool: str, args: dict[str, Any], code: str
) -> None:
    async with _client(service) as client:
        result = await client.call_tool(tool, args)
    assert result.isError
    assert result.structuredContent is not None
    assert result.structuredContent["error"]["code"] == code


async def test_forbidden_cancel(service: BookingService, world: dict[str, int]) -> None:
    async with _client(service) as client:
        booked = await client.call_tool("create_booking", _booking_args(world))
        assert booked.structuredContent is not None
        bid = booked.structuredContent["booking_id"]
        result = await client.call_tool(
            "cancel_booking", {"user_id": world["bob"], "booking_id": bid, "dry_run": False}
        )
    assert result.structuredContent is not None
    assert result.structuredContent["error"]["code"] == "FORBIDDEN"


async def test_unexpected_exception_becomes_internal_error(
    service: BookingService, world: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any) -> Any:
        raise RuntimeError("db exploded")

    monkeypatch.setattr("booking_mcp.server.execute", boom)
    async with _client(service) as client:
        result = await client.call_tool("list_my_bookings", {"user_id": world["alice"]})
    assert result.structuredContent is not None
    assert result.structuredContent["error"] == {
        "code": "INTERNAL",
        "jsonrpc_code": -32603,
        "message": "internal error",
    }


async def test_meta_trace_id_creates_child_span(
    service: BookingService, world: dict[str, int]
) -> None:
    sink = RecordingSink()
    trace_id = "ab" * 16
    async with _client(service, sink) as client:
        await client.call_tool(
            "list_my_bookings",
            {"user_id": world["alice"]},
            meta={"trace_id": trace_id, "parent_span_id": "1234567890abcdef", "session_id": "s1"},
        )
        await client.call_tool("list_my_bookings", {"user_id": world["alice"]})  # no meta
    [trace] = sink.traces
    assert trace.trace_id == trace_id
    assert trace.session_id == "s1"
    by_name = {s.name: s for s in trace.spans}
    tool_span = by_name["tool:list_my_bookings"]
    assert tool_span.kind == "tool"
    assert tool_span.parent_id == by_name["turn"].span_id
    assert by_name["turn"].parent_id == "1234567890abcdef"
    assert tool_span.attrs["result_summary"] == {"bookings": 0}


async def test_error_span_records_code(service: BookingService, world: dict[str, int]) -> None:
    sink = RecordingSink()
    async with _client(service, sink) as client:
        await client.call_tool(
            "cancel_booking", {"user_id": 1, "booking_id": 5}, meta={"trace_id": "cd" * 16}
        )
    tool_span = next(s for s in sink.traces[0].spans if s.name == "tool:cancel_booking")
    assert tool_span.attrs["error_code"] == "NOT_FOUND"


async def test_ensure_user_finds_or_creates(service: BookingService, world: dict[str, int]) -> None:
    async with _client(service) as client:
        first = await client.call_tool("ensure_user", {"nickname": " 新玩家 "})
        again = await client.call_tool("ensure_user", {"nickname": "新玩家"})
        existing = await client.call_tool("ensure_user", {"nickname": "alice"})
        empty = await client.call_tool("ensure_user", {"nickname": "  "})
    assert first.structuredContent is not None and again.structuredContent is not None
    assert first.structuredContent["created"] is True
    assert first.structuredContent["nickname"] == "新玩家"
    assert again.structuredContent == {**first.structuredContent, "created": False}
    assert existing.structuredContent is not None
    assert existing.structuredContent["user_id"] == world["alice"]
    assert empty.structuredContent is not None
    assert empty.structuredContent["error"]["code"] == "INVALID_ARGUMENT"


def test_execute_validates_and_serializes(service: BookingService, world: dict[str, int]) -> None:
    out = execute(service, "list_my_bookings", {"user_id": world["alice"]})
    assert out == {"bookings": []}


# --- real transports -------------------------------------------------------------------------


@pytest.fixture
def seeded_db(tmp_path: Path, booking_db: BookingDB) -> Path:
    seed_database(
        booking_db.engine, booking_db.factory, booking_db.domain, start_date=date(2026, 10, 1)
    )
    return tmp_path / "booking.db"


@pytest.mark.slow
async def test_stdio_subprocess(seeded_db: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "booking_mcp", "--transport", "stdio", "--db", str(seeded_db),
              "--now", NOW.isoformat(), "--log-level", "WARNING"],
    )  # fmt: skip
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        found = await session.call_tool(
            "find_companions",
            {"game_mode": "aram", "start_time": START.isoformat(), "duration_hours": 1},
        )
    assert names == EXPECTED_TOOLS
    assert found.structuredContent is not None
    assert found.structuredContent["candidates"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def http_url(service: BookingService, world: dict[str, int]) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(create_http_app(build_server(service)), port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.mark.slow
async def test_streamable_http(http_url: str, world: dict[str, int]) -> None:
    async with (
        streamable_http_client(http_url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        booked = await session.call_tool("create_booking", _booking_args(world))
        again = await session.call_tool("create_booking", _booking_args(world, "bob"))
    assert names == EXPECTED_TOOLS
    assert not booked.isError
    assert again.structuredContent is not None
    assert again.structuredContent["error"]["code"] == "SLOT_CONFLICT"


def test_http_rejects_plain_get(http_url: str) -> None:
    resp = httpx.get(http_url, headers={"accept": "text/html"})
    assert resp.status_code >= 400
