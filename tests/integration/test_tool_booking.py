"""D5: create_booking -> list_my_bookings -> cancel_booking(dry_run) -> cancel -> slot freed."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from booking_mcp.db import BookingStatus, read_session, write_session
from booking_mcp.db.repo import BookingRepo, ScheduleRepo
from booking_mcp.errors import ErrorCode, Forbidden, InvalidArgument, NotFound, SlotConflict
from booking_mcp.service import BookingService
from booking_mcp.tools.cancel_booking import cancel_booking
from booking_mcp.tools.create_booking import create_booking
from booking_mcp.tools.find_companions import find_companions
from booking_mcp.tools.list_my_bookings import list_my_bookings
from booking_mcp.tools.schemas import (
    CancelBookingInput,
    CreateBookingInput,
    FindCompanionsInput,
    ListMyBookingsInput,
)
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, ServiceType
from rift_domain.refund import refund

if TYPE_CHECKING:
    from tests.integration.conftest import BookingDB

START = datetime(2026, 10, 3, 20, 0)
DAY_BEFORE = START - timedelta(days=2)


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock(DAY_BEFORE)


@pytest.fixture
def svc(booking_db: BookingDB, clock: Clock) -> BookingService:
    return BookingService(booking_db.factory, booking_db.domain, clock=clock)


@pytest.fixture
def world(booking_db: BookingDB) -> dict[str, int]:
    cid = booking_db.add_companion(
        "阿狸酱",
        modes=(GameMode.RANKED_SOLO_DUO, GameMode.ARAM),
        service_types=(ServiceType.CLIMB, ServiceType.CASUAL),
        hourly_price=Decimal("60"),
    )
    booking_db.add_schedule(cid, START - timedelta(hours=2), START + timedelta(hours=4))
    return {
        "companion": cid,
        "alice": booking_db.add_user("alice"),
        "bob": booking_db.add_user("bob"),
    }


def _create(svc: BookingService, world: dict[str, int], **overrides: Any) -> int:
    fields: dict[str, Any] = {
        "user_id": world["alice"],
        "companion_id": world["companion"],
        "start_time": START,
        "duration_hours": 2,
        "game_mode": GameMode.RANKED_SOLO_DUO,
    }
    fields.update(overrides)
    return create_booking(svc, CreateBookingInput(**fields)).booking_id


def _is_free(db: BookingDB, cid: int) -> bool:
    with read_session(db.factory) as s:
        return ScheduleRepo(s).is_free(cid, START, START + timedelta(hours=2))


def test_full_lifecycle(booking_db: BookingDB, svc: BookingService, world: dict[str, int]) -> None:
    out = create_booking(
        svc,
        CreateBookingInput(
            user_id=world["alice"],
            companion_id=world["companion"],
            start_time=START,
            duration_hours=2,
            game_mode=GameMode.RANKED_SOLO_DUO,
        ),
    )
    assert out.status is BookingStatus.PENDING_PAYMENT
    assert out.service_type is ServiceType.CLIMB  # inferred from the ranked mode
    assert (out.unit_price, out.multiplier, out.total) == (
        Decimal("60.00"),
        Decimal("1.2"),
        Decimal("144.00"),
    )
    assert out.companion_name == "阿狸酱"
    assert out.end_time == START + timedelta(hours=2)
    assert not _is_free(booking_db, world["companion"])

    listed = list_my_bookings(svc, ListMyBookingsInput(user_id=world["alice"])).bookings
    assert [b.booking_id for b in listed] == [out.booking_id]

    # Pay so the refund is real money.
    with write_session(booking_db.factory) as s:
        repo = BookingRepo(s)
        repo.pay(repo.require(out.booking_id), DAY_BEFORE)

    quote = cancel_booking(
        svc, CancelBookingInput(user_id=world["alice"], booking_id=out.booking_id)
    )
    assert quote.dry_run and quote.paid
    assert (quote.refund_tier, quote.refund_ratio, quote.refund_amount) == (
        "full",
        Decimal("1.0"),
        Decimal("144.00"),
    )
    assert quote.hours_before_start == 48.0
    assert quote.booking.status is BookingStatus.CONFIRMED  # dry run changed nothing
    assert not _is_free(booking_db, world["companion"])

    done = cancel_booking(
        svc,
        CancelBookingInput(user_id=world["alice"], booking_id=out.booking_id, dry_run=False),
    )
    assert done.booking.status is BookingStatus.CANCELLED
    assert done.booking.refund_amount == Decimal("144.00")
    assert done.booking.cancelled_at == DAY_BEFORE
    assert _is_free(booking_db, world["companion"])

    cancelled = list_my_bookings(
        svc, ListMyBookingsInput(user_id=world["alice"], status=BookingStatus.CANCELLED)
    ).bookings
    assert [b.booking_id for b in cancelled] == [out.booking_id]
    pending = ListMyBookingsInput(user_id=world["alice"], status=BookingStatus.PENDING_PAYMENT)
    assert list_my_bookings(svc, pending).bookings == []


@pytest.mark.parametrize(
    ("hours_before", "tier", "amount"),
    [(24, "full", "144.00"), (3, "half", "72.00"), (2, "half", "72.00"), (1.5, "none", "0.00")],
)
def test_refund_tiers_match_domain(
    booking_db: BookingDB,
    svc: BookingService,
    clock: Clock,
    world: dict[str, int],
    domain: DomainConfig,
    hours_before: float,
    tier: str,
    amount: str,
) -> None:
    bid = _create(svc, world)
    with write_session(booking_db.factory) as s:
        repo = BookingRepo(s)
        repo.pay(repo.require(bid), DAY_BEFORE)
    clock.now = START - timedelta(hours=hours_before)
    out = cancel_booking(svc, CancelBookingInput(user_id=world["alice"], booking_id=bid))
    with read_session(booking_db.factory) as s:
        expected = refund(BookingRepo(s).require(bid), clock.now, domain)
    assert (out.refund_tier, out.refund_amount) == (tier, Decimal(amount))
    assert (expected.tier, expected.amount) == (tier, Decimal(amount))


def test_unpaid_booking_cancels_for_free(svc: BookingService, world: dict[str, int]) -> None:
    bid = _create(svc, world)
    out = cancel_booking(
        svc, CancelBookingInput(user_id=world["alice"], booking_id=bid, dry_run=False)
    )
    assert not out.paid
    assert out.policy_refund_amount == Decimal("144.00")
    assert out.refund_amount == Decimal("0.00")
    assert out.booking.refund_amount == Decimal("0.00")


def test_cancel_other_users_booking_is_forbidden(
    svc: BookingService, world: dict[str, int]
) -> None:
    bid = _create(svc, world)
    with pytest.raises(Forbidden) as err:
        cancel_booking(svc, CancelBookingInput(user_id=world["bob"], booking_id=bid, dry_run=False))
    assert err.value.code is ErrorCode.FORBIDDEN
    listed = list_my_bookings(svc, ListMyBookingsInput(user_id=world["alice"])).bookings
    assert listed[0].status is BookingStatus.PENDING_PAYMENT
    assert list_my_bookings(svc, ListMyBookingsInput(user_id=world["bob"])).bookings == []


def test_cancel_twice_is_invalid(svc: BookingService, world: dict[str, int]) -> None:
    bid = _create(svc, world)
    args = CancelBookingInput(user_id=world["alice"], booking_id=bid, dry_run=False)
    cancel_booking(svc, args)
    with pytest.raises(InvalidArgument, match="cannot be cancelled"):
        cancel_booking(svc, args)
    with pytest.raises(InvalidArgument, match="cannot be cancelled"):
        cancel_booking(svc, CancelBookingInput(user_id=world["alice"], booking_id=bid))


def test_cancel_unknown_booking(svc: BookingService, world: dict[str, int]) -> None:
    with pytest.raises(NotFound):
        cancel_booking(svc, CancelBookingInput(user_id=world["alice"], booking_id=42))


def test_double_booking_conflicts_and_slot_reopens(
    svc: BookingService, world: dict[str, int]
) -> None:
    first = _create(svc, world)
    with pytest.raises(SlotConflict):
        _create(svc, world, user_id=world["bob"], start_time=START + timedelta(hours=1))
    cancel_booking(svc, CancelBookingInput(user_id=world["alice"], booking_id=first, dry_run=False))
    second = _create(svc, world, user_id=world["bob"])
    assert second != first


def test_booked_companion_disappears_from_search(
    svc: BookingService, world: dict[str, int]
) -> None:
    def search() -> list[str]:
        out = find_companions(
            svc,
            FindCompanionsInput(
                game_mode=GameMode.RANKED_SOLO_DUO, start_time=START, duration_hours=2
            ),
        )
        return [c.name for c in out.candidates if c.start_time == START]

    assert search() == ["阿狸酱"]
    _create(svc, world)
    assert search() == []


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"start_time": DAY_BEFORE - timedelta(hours=1)}, InvalidArgument, "past"),
        ({"game_mode": GameMode.ARENA}, InvalidArgument, "does not play arena"),
        ({"service_type": ServiceType.COACHING}, InvalidArgument, "does not offer coaching"),
        ({"duration_hours": 8.5}, InvalidArgument, "duration"),
        ({"companion_id": 999}, NotFound, "companion 999"),
        ({"user_id": 999}, NotFound, "user 999"),
    ],
)
def test_create_booking_validation(
    svc: BookingService,
    world: dict[str, int],
    overrides: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        _create(svc, world, **overrides)


def test_list_unknown_user(svc: BookingService) -> None:
    with pytest.raises(NotFound):
        list_my_bookings(svc, ListMyBookingsInput(user_id=404))


def test_casual_mode_uses_casual_multiplier(svc: BookingService, world: dict[str, int]) -> None:
    out = create_booking(
        svc,
        CreateBookingInput(
            user_id=world["alice"],
            companion_id=world["companion"],
            start_time=START,
            duration_hours=1.5,
            game_mode=GameMode.ARAM,
        ),
    )
    assert (out.service_type, out.total) == (ServiceType.CASUAL, Decimal("90.00"))


def test_bookings_listed_by_start_time(
    booking_db: BookingDB, svc: BookingService, world: dict[str, int]
) -> None:
    later = _create(svc, world, start_time=START + timedelta(hours=2))
    earlier = _create(svc, world, start_time=START - timedelta(hours=2))
    ids: Callable[[], list[int]] = lambda: [  # noqa: E731
        b.booking_id
        for b in list_my_bookings(svc, ListMyBookingsInput(user_id=world["alice"])).bookings
    ]
    assert ids() == [earlier, later]
