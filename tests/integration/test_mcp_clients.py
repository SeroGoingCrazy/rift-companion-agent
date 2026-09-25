"""F2: agent-side MCP clients — typed booking calls, error mapping, trace meta, HTTP."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest
import uvicorn
from mcp.server.lowlevel import Server

from booking_mcp.server import build_server, create_http_app
from booking_mcp.service import BookingService
from rift_agent.mcp_clients import (
    BookingClient,
    ForbiddenError,
    InvalidArgumentError,
    KnowledgeClient,
    McpToolClient,
    McpUnavailable,
    NotFoundError,
    SlotConflictError,
    ToolError,
    error_from_payload,
    http_connector,
    inprocess_connector,
    parse_knowledge,
)
from rift_common.trace import TraceContext, start_turn
from rift_common.trace.sinks.base import TraceSink

SAT_20 = datetime(2026, 10, 3, 20, 0)


class Recorder(TraceSink):
    def __init__(self) -> None:
        self.traces: list[TraceContext] = []

    def on_turn_end(self, trace: TraceContext) -> None:
        self.traces.append(trace)


def _booking(server: Server[Any, Any]) -> BookingClient:
    return BookingClient(McpToolClient("booking", inprocess_connector(server)))


async def _first_candidate(client: BookingClient) -> dict[str, Any]:
    out = await client.find_companions(
        game_mode="aram", start_time=SAT_20, duration_hours=2, companion_gender=None
    )
    assert out["candidates"], out
    candidate: dict[str, Any] = out["candidates"][0]
    return candidate


async def test_find_quote_create_list_cancel(booking_server: Server[Any, Any]) -> None:
    client = _booking(booking_server)
    cand = await _first_candidate(client)
    quote = await client.quote_price(cand["companion_id"], "casual", 2)
    assert quote["total"] == f"{float(cand['hourly_price']) * 2:.2f}"
    booking = await client.create_booking(
        user_id=1,
        companion_id=cand["companion_id"],
        start_time=cand["start_time"],
        duration_hours=2,
        game_mode="aram",
    )
    assert booking["status"] == "pending_payment"
    assert [b["booking_id"] for b in await client.list_my_bookings(1)] == [booking["booking_id"]]
    assert await client.list_my_bookings(1, status="cancelled") == []
    dry = await client.cancel_booking(1, booking["booking_id"], dry_run=True)
    assert dry["dry_run"] and dry["booking"]["status"] == "pending_payment"


async def test_slot_conflict_becomes_exception(booking_server: Server[Any, Any]) -> None:
    client = _booking(booking_server)
    cand = await _first_candidate(client)
    args: dict[str, Any] = {
        "user_id": 1,
        "companion_id": cand["companion_id"],
        "start_time": cand["start_time"],
        "duration_hours": 2,
        "game_mode": "aram",
    }
    await client.create_booking(**args)
    with pytest.raises(SlotConflictError) as err:
        await client.create_booking(**args)
    assert err.value.code == "SLOT_CONFLICT"
    assert err.value.details["companion_id"] == cand["companion_id"]


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda c: c.cancel_booking(1, 999, dry_run=True), NotFoundError),
        (lambda c: c.quote_price(1, "climb", 0.3), InvalidArgumentError),
        (lambda c: c.list_my_bookings(12345), NotFoundError),
    ],
)
async def test_error_codes_map_to_exceptions(
    booking_server: Server[Any, Any], call: Any, error: type[ToolError]
) -> None:
    with pytest.raises(error):
        await call(_booking(booking_server))


async def test_forbidden(booking_server: Server[Any, Any], seeded_booking: BookingService) -> None:
    from booking_mcp.db import User, write_session

    with write_session(seeded_booking.factory) as s:
        s.add(User(nickname="other"))
    client = _booking(booking_server)
    cand = await _first_candidate(client)
    b = await client.create_booking(
        user_id=1,
        companion_id=cand["companion_id"],
        start_time=cand["start_time"],
        duration_hours=1,
        game_mode="aram",
    )
    with pytest.raises(ForbiddenError):
        await client.cancel_booking(2, b["booking_id"], dry_run=False)


async def test_trace_meta_reaches_server(seeded_booking: BookingService) -> None:
    server_sink, client_sink = Recorder(), Recorder()
    client = _booking(build_server(seeded_booking, [server_sink]))
    with start_turn(session_id="sess-9", sinks=[client_sink]):
        await client.list_my_bookings(1)
    [client_trace], [server_trace] = client_sink.traces, server_sink.traces
    assert server_trace.trace_id == client_trace.trace_id
    assert server_trace.session_id == "sess-9"
    client_tool = next(s for s in client_trace.spans if s.name == "tool:list_my_bookings")
    assert client_tool.kind == "tool"
    assert server_trace.root is not None
    assert server_trace.root.parent_id == client_tool.span_id


async def test_client_span_records_error_code(booking_server: Server[Any, Any]) -> None:
    sink = Recorder()
    with start_turn(sinks=[sink]), pytest.raises(NotFoundError):
        await _booking(booking_server).cancel_booking(1, 404, dry_run=True)
    tool = next(s for s in sink.traces[0].spans if s.name == "tool:cancel_booking")
    assert tool.attrs["error_code"] == "NOT_FOUND"
    assert tool.status == "error"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def http_url(booking_server: Server[Any, Any]) -> Iterator[str]:
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_http_app(booking_server), port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.slow
async def test_http_connector_against_real_server(http_url: str) -> None:
    client = BookingClient(McpToolClient("booking", http_connector(http_url)))
    cand = await _first_candidate(client)
    args: dict[str, Any] = {
        "user_id": 1,
        "companion_id": cand["companion_id"],
        "start_time": cand["start_time"],
        "duration_hours": 2,
        "game_mode": "aram",
    }
    await client.create_booking(**args)
    with pytest.raises(SlotConflictError):
        await client.create_booking(**args)


async def test_unreachable_server_raises_unavailable() -> None:
    client = McpToolClient(
        "booking", http_connector(f"http://127.0.0.1:{_free_port()}/mcp", timeout_s=2)
    )
    with pytest.raises(McpUnavailable, match="booking unavailable"):
        await client.call("list_my_bookings", {"user_id": 1})


# --- knowledge ---------------------------------------------------------------------------


async def test_knowledge_query_parses_citations(knowledge_server: Server[Any, Any]) -> None:
    kc = KnowledgeClient(McpToolClient("knowledge", inprocess_connector(knowledge_server)))
    r = await kc.query("开局前取消退多少", collection="platform_rules")
    assert r.collection == "platform_rules"
    assert "退款档位" in r.text
    assert "References" not in r.text
    assert r.citations == [{"index": 1, "source": "platform_rules/doc.md", "score": 0.9}]


async def test_knowledge_query_all_skips_empty(knowledge_server: Server[Any, Any]) -> None:
    kc = KnowledgeClient(
        McpToolClient("knowledge", inprocess_connector(knowledge_server)),
        collections=("platform_rules", "modes_and_ranks", "companion_profiles"),
    )
    results = await kc.query_all("斗魂竞技场几个人")
    assert [r.collection for r in results] == ["modes_and_ranks"]
    assert await kc.query_all("今天天气") == []


def test_parse_knowledge_variants() -> None:
    assert parse_knowledge({"text": "plain"}, None).text == "plain"
    bad = "x\n---\n**References (JSON):**\n```json\n{not json}\n```"
    assert parse_knowledge({"text": bad}, None).citations == []
    r = parse_knowledge({"text": "", "citations": [{"a": 1}]}, "c")
    assert r.empty and r.citations == [{"a": 1}]


def test_error_from_payload() -> None:
    assert isinstance(error_from_payload({"error": {"code": "SLOT_CONFLICT"}}), SlotConflictError)
    e = error_from_payload({"message": "raw failure"})
    assert type(e) is ToolError and e.message == "raw failure"
    assert error_from_payload({"error": {"code": "WEIRD", "message": "m"}}).code == "WEIRD"
