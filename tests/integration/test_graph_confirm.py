"""F7: select -> quote -> confirm (interrupt) -> book; re-quote on change; slot conflicts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from booking_mcp.server import build_server
from booking_mcp.service import BookingService
from booking_mcp.tools.create_booking import create_booking
from booking_mcp.tools.list_my_bookings import list_my_bookings
from booking_mcp.tools.schemas import CreateBookingInput, ListMyBookingsInput
from rift_agent.graph.state import booking_state
from rift_common.trace import TraceContext, start_turn
from rift_common.trace.sinks.base import TraceSink

if TYPE_CHECKING:
    from tests.integration.conftest import AgentHarness, BookingDB

AGENT_NOW = datetime(2026, 10, 1, 14, 0)  # same fixed clock as the conftest harness
FRI_20 = datetime(2026, 10, 2, 20, 0)


def turn(confirmation: str = "none", intent: str = "booking", **delta: Any) -> dict[str, Any]:
    return {"turn_intent": intent, "delta": delta, "confirmation": confirmation}


TABLE = {
    "约个大乱斗明晚八点两小时": turn(
        game_mode="aram", start_time_expr="明晚八点", duration_hours=2
    ),
    "就第二个": turn(companion_name="乙"),
    "就第一个": turn(companion_name="甲"),
    "改成三小时再下单": turn(confirmation="yes", duration_hours=3),
    "确认": turn(confirmation="yes"),
    "算了": turn(confirmation="no"),
    "我再想想": turn(),
}  # fmt: skip


class World:
    def __init__(self, db: BookingDB) -> None:
        self.db = db
        self.a = db.add_companion("甲", rating=4.9, hourly_price=Decimal("60"))
        self.b = db.add_companion("乙", rating=4.5, hourly_price=Decimal("50"))
        for cid in (self.a, self.b):
            db.add_schedule(cid, FRI_20 - timedelta(hours=2), FRI_20 + timedelta(hours=4))
        self.user = db.add_user("demo")
        self.rival = db.add_user("rival")
        self.service = BookingService(db.factory, db.domain, clock=lambda: AGENT_NOW)
        self.server = build_server(self.service)

    def bookings(self, user_id: int | None = None) -> list[dict[str, Any]]:
        out = list_my_bookings(self.service, ListMyBookingsInput(user_id=user_id or self.user))
        return [b.model_dump(mode="json") for b in out.bookings]


@pytest.fixture
def world(make_booking_db: Callable[[str], BookingDB]) -> World:
    return World(make_booking_db("world"))


@pytest.fixture
def agent(make_agent: Callable[..., AgentHarness], world: World) -> AgentHarness:
    return make_agent(TABLE, server=world.server, user_id=world.user)


async def _to_confirmation(agent: AgentHarness, pick: str = "就第二个") -> dict[str, Any]:
    s = await agent.say("约个大乱斗明晚八点两小时")
    assert [c["name"] for c in s["candidate_cards"]] == ["甲", "乙"]
    return await agent.say(pick)


async def test_confirm_then_book(agent: AgentHarness, world: World) -> None:
    s = await _to_confirmation(agent)
    assert s["reply_type"] == "confirm_booking"
    assert s["pending_action"] == "await_confirm_booking"
    assert "乙" in s["reply"] and "50.00 × 1 × 2 = 100.00 元" in s["reply"]
    assert s["ui"]["confirm"]["total"] == "100.00"
    assert await agent.paused()
    assert world.bookings() == []  # nothing written before the user says yes

    s = await agent.say("确认")
    assert s["reply_type"] == "booked"
    assert not await agent.paused()
    assert s["phase"] == "IDLE" and s["pending_action"] is None
    assert booking_state(s).game_mode is None  # archived
    [order] = world.bookings()
    assert order["companion_name"] == "乙"
    assert order["status"] == "pending_payment"
    assert order["start_time"] == "2026-10-02T20:00:00"
    assert order["total"] == "100.00"
    assert f"#{order['booking_id']}" in s["reply"] and "开局提醒" in s["reply"]


async def test_change_duration_requotes_same_companion(agent: AgentHarness, world: World) -> None:
    await _to_confirmation(agent)
    s = await agent.say("改成三小时再下单")
    assert s["reply_type"] == "confirm_booking"
    assert await agent.paused()
    assert s["ui"]["confirm"]["companion"] == "乙"
    assert s["ui"]["confirm"]["total"] == "150.00"
    assert world.bookings() == []

    s = await agent.say("确认")
    [order] = world.bookings()
    assert (order["companion_name"], order["hours"], order["total"]) == ("乙", "3.0", "150.00")


async def test_decline_goes_back_to_candidates(agent: AgentHarness, world: World) -> None:
    await _to_confirmation(agent)
    s = await agent.say("算了")
    assert s["reply_type"] == "candidates"
    assert s["reply"].startswith("好的，不约这位了")
    assert not await agent.paused()
    b = booking_state(s)
    assert b.selected_companion_id is None and b.quote is None and b.companion_name is None
    assert len(b.candidates) == 2
    # Picking another one works from here.
    s = await agent.say("就第一个")
    assert s["reply_type"] == "confirm_booking" and s["ui"]["confirm"]["companion"] == "甲"


async def test_unclear_answer_asks_again(agent: AgentHarness) -> None:
    await _to_confirmation(agent)
    s = await agent.say("我再想想")
    assert s["reply_type"] == "reconfirm"
    assert "还在等你确认" in s["reply"] and "100.00" in s["reply"]
    assert await agent.paused()
    s = await agent.say("确认")
    assert s["reply_type"] == "booked"


async def test_slot_taken_meanwhile_offers_alternatives(agent: AgentHarness, world: World) -> None:
    await _to_confirmation(agent)
    create_booking(
        world.service,
        CreateBookingInput(
            user_id=world.rival,
            companion_id=world.b,
            start_time=FRI_20,
            duration_hours=2,
            game_mode="aram",
        ),
    )
    s = await agent.say("确认")
    assert s["reply_type"] == "candidates"
    assert s["reply"].startswith("刚才那个时段被别人抢先约走了")
    assert [c["name"] for c in s["candidate_cards"]] == ["甲"]
    assert world.bookings() == []
    assert s["phase"] == "BOOKING" and s["pending_action"] is None


async def test_confirm_span_is_not_an_error(agent: AgentHarness) -> None:
    class Rec(TraceSink):
        def __init__(self) -> None:
            self.traces: list[TraceContext] = []

        def on_turn_end(self, trace: TraceContext) -> None:
            self.traces.append(trace)

    await agent.say("约个大乱斗明晚八点两小时")
    sink = Rec()
    with start_turn(sinks=[sink]):
        await agent.say("就第二个")
    spans = {s.name: s for s in sink.traces[0].spans}
    assert spans["node:confirm"].status == "ok"
    assert spans["node:confirm"].attrs["interrupted"] is True
    assert "tool:quote_price" in spans
