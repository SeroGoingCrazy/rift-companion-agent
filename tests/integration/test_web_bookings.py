"""G3: my-orders API — list, simulated payment, refund preview, cancel; ownership."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient

from booking_mcp.db import read_session
from booking_mcp.db.repo import ScheduleRepo
from booking_mcp.server import build_server
from booking_mcp.service import BookingService
from booking_mcp.tools.create_booking import create_booking
from booking_mcp.tools.schemas import CreateBookingInput
from rift_web.services import WebServices

if TYPE_CHECKING:
    from tests.integration.conftest import BookingDB

Make = Callable[..., tuple[TestClient, WebServices]]
NOW = datetime(2026, 10, 1, 14, 0)
FRI_20 = datetime(2026, 10, 2, 20, 0)


class Shop:
    def __init__(self, db: BookingDB) -> None:
        self.db = db
        self.companion = db.add_companion("甲", hourly_price=Decimal("60"))
        db.add_schedule(self.companion, NOW, NOW + timedelta(days=3))
        self.demo = db.add_user("demo")
        self.other = db.add_user("other")
        self.service = BookingService(db.factory, db.domain, clock=lambda: NOW)
        self.server = build_server(self.service)

    def book(self, user: int, start: datetime = FRI_20) -> int:
        return create_booking(
            self.service,
            CreateBookingInput(
                user_id=user,
                companion_id=self.companion,
                start_time=start,
                duration_hours=2,
                game_mode="ranked_solo_duo",
            ),
        ).booking_id


@pytest.fixture
def shop(make_booking_db: Callable[[str], BookingDB]) -> Shop:
    return Shop(make_booking_db("web_shop"))


@pytest.fixture
def client(make_web: Make, shop: Shop, web_login: Callable[..., None]) -> TestClient:
    c, _ = make_web(server=shop.server)
    web_login(c, "demo")
    return c


def _bookings(client: TestClient) -> list[dict[str, Any]]:
    resp = client.get("/api/bookings")
    assert resp.status_code == 200
    rows: list[dict[str, Any]] = resp.json()["bookings"]
    return rows


def test_pay_then_cancel_with_refund(client: TestClient, shop: Shop) -> None:
    bid = shop.book(shop.demo)
    [row] = _bookings(client)
    assert (row["booking_id"], row["status"], row["total"]) == (bid, "pending_payment", "144.00")

    paid = client.post(f"/api/bookings/{bid}/pay")
    assert paid.status_code == 200 and paid.json()["booking"]["status"] == "confirmed"
    assert _bookings(client)[0]["status"] == "confirmed"

    preview = client.post(f"/api/bookings/{bid}/cancel", json={"dry_run": True}).json()
    assert preview["dry_run"] and preview["paid"]
    assert (preview["refund_tier"], preview["refund_amount"]) == ("full", "144.00")
    assert _bookings(client)[0]["status"] == "confirmed"

    done = client.post(f"/api/bookings/{bid}/cancel", json={"dry_run": False}).json()
    assert done["booking"]["status"] == "cancelled" and done["refund_amount"] == "144.00"
    [row] = _bookings(client)
    assert row["status"] == "cancelled" and row["refund_amount"] == "144.00"
    with read_session(shop.db.factory) as s:
        assert ScheduleRepo(s).is_free(shop.companion, FRI_20, FRI_20 + timedelta(hours=2))


def test_cancel_defaults_to_dry_run(client: TestClient, shop: Shop) -> None:
    bid = shop.book(shop.demo)
    out = client.post(f"/api/bookings/{bid}/cancel", json={}).json()
    assert out["dry_run"] is True
    assert _bookings(client)[0]["status"] == "pending_payment"


def test_active_orders_listed_first(client: TestClient, shop: Shop) -> None:
    later = shop.book(shop.demo, FRI_20 + timedelta(hours=3))
    old = shop.book(shop.demo, FRI_20)
    client.post(f"/api/bookings/{old}/cancel", json={"dry_run": False})
    assert [b["booking_id"] for b in _bookings(client)] == [later, old]


def test_other_users_orders_are_forbidden(client: TestClient, shop: Shop) -> None:
    theirs = shop.book(shop.other)
    assert _bookings(client) == []
    pay = client.post(f"/api/bookings/{theirs}/pay")
    assert pay.status_code == 403 and pay.json()["code"] == "FORBIDDEN"
    cancel = client.post(f"/api/bookings/{theirs}/cancel", json={"dry_run": False})
    assert cancel.status_code == 403


def test_invalid_transitions_and_unknown_ids(client: TestClient, shop: Shop) -> None:
    bid = shop.book(shop.demo)
    assert client.post(f"/api/bookings/{bid}/pay").status_code == 200
    again = client.post(f"/api/bookings/{bid}/pay")
    assert again.status_code == 400 and again.json()["code"] == "INVALID_ARGUMENT"
    assert client.post("/api/bookings/999/pay").status_code == 404
    assert client.post("/api/bookings/999/cancel", json={}).status_code == 404


def test_requires_login(make_web: Make, shop: Shop) -> None:
    c, _ = make_web(server=shop.server)
    assert c.get("/api/bookings").status_code == 401
    assert c.post("/api/bookings/1/pay").status_code == 401
    assert c.post("/api/bookings/1/cancel", json={}).status_code == 401
    page = c.get("/bookings")
    assert page.status_code == 303 and page.headers["location"] == "/login?next=/bookings"


def test_service_down_is_503(
    client: TestClient, make_web: Make, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rift_agent.mcp_clients import McpUnavailable

    services: WebServices = client.app.state.services  # type: ignore[attr-defined]

    async def down(*_: Any, **__: Any) -> Any:
        raise McpUnavailable("booking unavailable")

    monkeypatch.setattr(services.booking, "list_my_bookings", down)
    resp = client.get("/api/bookings")
    assert resp.status_code == 503 and resp.json()["code"] == "UNAVAILABLE"


def test_page_and_script(client: TestClient) -> None:
    html = client.get("/bookings").text
    assert 'id="bookings"' in html and 'id="cancel-dialog"' in html
    assert '<script src="/static/bookings.js"' in html
    js = client.get("/static/bookings.js").text
    for hook in ("/pay", "dry_run: true", "dry_run: false", "模拟支付", "load()"):
        assert hook in js
