"""F9: order management — list, choose, refund quote, confirm, cancel, slot released."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from booking_mcp.db import BookingStatus, read_session, write_session
from booking_mcp.db.repo import BookingRepo, ScheduleRepo
from booking_mcp.server import build_server
from booking_mcp.service import BookingService
from booking_mcp.tools.create_booking import create_booking
from booking_mcp.tools.schemas import CreateBookingInput
from rift_agent.graph.nodes.manage import parse_yes_no, pick_booking
from rift_domain.refund import refund

if TYPE_CHECKING:
    from tests.integration.conftest import AgentHarness, BookingDB

NOW = datetime(2026, 10, 1, 14, 0)
FRI_20 = datetime(2026, 10, 2, 20, 0)  # 30h ahead: full refund
THU_15 = datetime(2026, 10, 1, 15, 0)  # 1h ahead: no refund


def turn(confirmation: str = "none", intent: str = "booking", **delta: Any) -> dict[str, Any]:
    return {"turn_intent": intent, "delta": delta, "confirmation": confirmation}


TABLE = {
    "约个大乱斗明晚八点两小时": turn(game_mode="aram", start_time_expr="明晚八点", duration_hours=2)
}


class Shop:
    def __init__(self, db: BookingDB) -> None:
        self.db = db
        self.a = db.add_companion("甲", hourly_price=Decimal("60"))
        self.b = db.add_companion("乙", hourly_price=Decimal("50"))
        for cid in (self.a, self.b):
            db.add_schedule(cid, NOW, NOW + timedelta(days=3))
        self.user = db.add_user("demo")
        self.service = BookingService(db.factory, db.domain, clock=lambda: NOW)
        self.server = build_server(self.service)

    def book(self, companion: int, start: datetime, *, paid: bool = False) -> int:
        out = create_booking(
            self.service,
            CreateBookingInput(
                user_id=self.user,
                companion_id=companion,
                start_time=start,
                duration_hours=2,
                game_mode="aram",
            ),
        )
        if paid:
            with write_session(self.db.factory) as s:
                repo = BookingRepo(s)
                repo.pay(repo.require(out.booking_id), NOW)
        return out.booking_id

    def status(self, booking_id: int) -> BookingStatus:
        with read_session(self.db.factory) as s:
            return BookingRepo(s).require(booking_id).status


@pytest.fixture
def shop(make_booking_db: Callable[[str], BookingDB]) -> Shop:
    return Shop(make_booking_db("shop"))


@pytest.fixture
def agent(make_agent: Callable[..., AgentHarness], shop: Shop) -> AgentHarness:
    return make_agent(TABLE, server=shop.server, user_id=shop.user)


async def test_list_choose_confirm_cancel(agent: AgentHarness, shop: Shop) -> None:
    late = shop.book(shop.a, FRI_20, paid=True)
    soon = shop.book(shop.b, THU_15)

    s = await agent.say("我有哪些订单")
    assert s["reply_type"] == "manage_list" and s["phase"] == "MANAGE"
    assert s["manage"]["shown"] == [soon, late]  # by start time
    assert f"1. #{soon} 乙" in s["reply"] and f"2. #{late} 甲" in s["reply"]
    assert "待支付" in s["reply"] and "已支付" in s["reply"]

    s = await agent.say("取消第二个")
    assert s["reply_type"] == "cancel_confirm"
    assert s["pending_action"] == "await_confirm_cancel"
    assert await agent.paused()
    # Refund equals the domain calculation (C8): paid 60 x 1.0 x 2 = 120.00, 30h ahead.
    with read_session(shop.db.factory) as sess:
        expected = refund(BookingRepo(sess).require(late), NOW, shop.db.domain)
    assert s["facts"]["refund_amount"] == f"{expected.amount:.2f}" == "120.00"
    assert "可退 120.00 元" in s["reply"]
    assert shop.status(late) is BookingStatus.CONFIRMED  # nothing cancelled yet

    s = await agent.say("确认")
    assert s["reply_type"] == "cancelled"
    assert f"已取消订单 #{late}" in s["reply"] and "退款 120.00 元" in s["reply"]
    assert s["phase"] == "IDLE" and s["pending_action"] is None
    assert not await agent.paused()
    assert shop.status(late) is BookingStatus.CANCELLED
    assert shop.status(soon) is BookingStatus.PENDING_PAYMENT
    with read_session(shop.db.factory) as sess:
        assert ScheduleRepo(sess).is_free(shop.a, FRI_20, FRI_20 + timedelta(hours=2))


async def test_single_active_booking_is_picked_directly(agent: AgentHarness, shop: Shop) -> None:
    bid = shop.book(shop.b, THU_15)
    s = await agent.say("帮我把订单取消了")
    assert s["reply_type"] == "cancel_confirm"
    assert "还没支付，取消不产生任何费用" in s["reply"]
    s = await agent.say("算了，不取消了")
    assert s["reply_type"] == "cancel_aborted"
    assert shop.status(bid) is BookingStatus.PENDING_PAYMENT


async def test_ambiguous_cancel_asks_which(agent: AgentHarness, shop: Shop) -> None:
    first = shop.book(shop.a, FRI_20)
    second = shop.book(shop.b, FRI_20)
    s = await agent.say("取消订单")
    assert s["reply_type"] == "manage_which"
    assert not await agent.paused()
    s = await agent.say("乙那单")
    assert s["reply_type"] == "cancel_confirm"
    assert s["facts"]["order"]["booking_id"] == second
    s = await agent.say("嗯？")
    assert s["reply_type"] == "cancel_unclear"
    assert shop.status(second) is BookingStatus.PENDING_PAYMENT
    s = await agent.say("好的")
    assert s["reply_type"] == "cancelled"
    assert shop.status(second) is BookingStatus.CANCELLED
    assert shop.status(first) is BookingStatus.PENDING_PAYMENT


async def test_really_unclear_answer(agent: AgentHarness, shop: Shop) -> None:
    bid = shop.book(shop.b, THU_15)
    await agent.say("取消订单")
    s = await agent.say("那个陪玩师叫什么来着")
    assert s["reply_type"] == "cancel_unclear"
    assert await agent.paused()
    assert shop.status(bid) is BookingStatus.PENDING_PAYMENT


async def test_no_orders(agent: AgentHarness) -> None:
    s = await agent.say("我有哪些订单")
    assert s["reply_type"] == "manage_list" and "还没有订单" in s["reply"]
    assert s["phase"] == "IDLE"
    s = await agent.say("取消订单")
    assert s["reply_type"] == "manage_empty"


async def test_leaving_manage_by_booking(agent: AgentHarness, shop: Shop) -> None:
    shop.book(shop.a, THU_15)
    await agent.say("我有哪些订单")
    s = await agent.say("约个大乱斗明晚八点两小时")
    assert s["phase"] == "BOOKING"
    assert s["reply_type"] == "candidates"


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("确认", True), ("好的", True), ("是的，取消吧", True), ("ok", True),
        ("算了", False), ("不取消了", False), ("不确定", False), ("先别", False),
        ("嗯", None), ("好？", None), ("确认取消？", True), ("嗯？陪玩师是谁", None),
        ("那个陪玩师叫什么来着", None),
    ],
)  # fmt: skip
def test_parse_yes_no(text: str, answer: bool | None) -> None:
    assert parse_yes_no(text) is answer


def test_pick_booking_variants() -> None:
    bookings = [
        {"booking_id": 7, "companion_name": "甲", "start_time": "2026-10-02T20:00:00"},
        {"booking_id": 9, "companion_name": "乙", "start_time": "2026-10-03T20:00:00"},
    ]
    shown = [9, 7]
    assert pick_booking("取消#7", bookings, shown, NOW)["booking_id"] == 7  # type: ignore[index]
    assert pick_booking("订单号 9", bookings, shown, NOW)["booking_id"] == 9  # type: ignore[index]
    assert pick_booking("第一个", bookings, shown, NOW)["booking_id"] == 9  # type: ignore[index]
    assert pick_booking("第2单", bookings, shown, NOW)["booking_id"] == 7  # type: ignore[index]
    assert pick_booking("第三个", bookings, shown, NOW) is None
    assert pick_booking("甲的", bookings, shown, NOW)["booking_id"] == 7  # type: ignore[index]
    assert pick_booking("明天那单", bookings, shown, NOW)["booking_id"] == 7  # type: ignore[index]
    assert pick_booking("随便", bookings, shown, NOW) is None
    assert pick_booking("#99", bookings, shown, NOW) is None
