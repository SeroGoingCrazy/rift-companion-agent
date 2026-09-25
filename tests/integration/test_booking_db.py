"""D1: booking database models, sessions and the booking status machine."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import Engine, event, inspect, select
from sqlalchemy.exc import IntegrityError, StatementError

from booking_mcp.db import (
    Booking,
    BookingStatus,
    Companion,
    Schedule,
    ScheduleStatus,
    User,
    create_db_engine,
    init_db,
    make_session_factory,
    read_session,
    reset_db,
    write_session,
)
from booking_mcp.db.models import ScaledDecimal
from booking_mcp.db.session import database_url
from booking_mcp.errors import ErrorCode, InvalidTransition
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType

START = datetime(2026, 10, 2, 20, 0)


@pytest.fixture
def engine(tmp_path: Path) -> Engine:
    eng = create_db_engine(tmp_path / "booking.db")
    init_db(eng)
    return eng


def _companion(name: str = "阿狸酱") -> Companion:
    c = Companion(
        name=name,
        gender=Gender.FEMALE,
        rank=Rank.DIAMOND,
        rank_tier=6,
        hourly_price=Decimal("60"),
        voice=True,
        rating=4.9,
        tags=["温柔", "会聊天"],
        bio="中单法师专精",
    )
    c.set_skills([Role.MID, Role.SUPPORT], [GameMode.RANKED_SOLO_DUO], [ServiceType.CLIMB])
    return c


def _booking(user: User, companion: Companion, **overrides: object) -> Booking:
    fields: dict[str, object] = {
        "user_id": user.id,
        "companion_id": companion.id,
        "start": START,
        "end": START + timedelta(hours=2),
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "service_type": ServiceType.CLIMB,
        "hours": Decimal("2"),
        "unit_price": Decimal("60"),
        "multiplier": Decimal("1.2"),
        "total": Decimal("144"),
    }
    fields.update(overrides)
    return Booking(**fields)


def test_create_all_tables(engine: Engine) -> None:
    tables = set(inspect(engine).get_table_names())
    assert {"users", "companions", "schedules", "bookings"} <= tables
    assert {"companion_roles", "companion_modes", "companion_service_types"} <= tables


def test_reset_db_empties_tables(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        s.add(User(nickname="tester"))
    reset_db(engine)
    with read_session(factory) as s:
        assert s.scalars(select(User)).all() == []


def test_companion_round_trip_with_skills(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        s.add(_companion())
    with read_session(factory) as s:
        c = s.scalars(select(Companion)).one()
        assert c.roles == (Role.MID, Role.SUPPORT)
        assert c.modes == (GameMode.RANKED_SOLO_DUO,)
        assert c.service_types == (ServiceType.CLIMB,)
        assert c.hourly_price == Decimal("60.00")
        assert c.tags == ["温柔", "会聊天"]


def test_booking_defaults_and_decimals(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        user, comp = User(nickname="tester"), _companion()
        s.add_all([user, comp])
        s.flush()
        s.add(_booking(user, comp, hours=Decimal("1.5"), multiplier=Decimal("1.25")))
    with read_session(factory) as s:
        b = s.scalars(select(Booking)).one()
        assert b.status is BookingStatus.PENDING_PAYMENT
        assert b.hours == Decimal("1.5")
        assert b.multiplier == Decimal("1.2500")
        assert b.total == Decimal("144.00")
        assert b.created_at is not None
        assert b.companion.name == "阿狸酱"
        assert b.start_time == START


def test_scaled_decimal_compares_in_sql(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        s.add_all([_companion("a"), _companion("b")])
        s.flush()
        s.scalars(select(Companion).where(Companion.name == "b")).one().hourly_price = Decimal(
            "80.5"
        )
    with read_session(factory) as s:
        cheap = s.scalars(select(Companion).where(Companion.hourly_price <= Decimal("80.49")))
        assert [c.name for c in cheap] == ["a"]


def test_scaled_decimal_handles_none_and_rounding() -> None:
    t = ScaledDecimal(2)
    assert t.process_bind_param(None, None) is None  # type: ignore[arg-type]
    assert t.process_result_value(None, None) is None  # type: ignore[arg-type]
    assert t.process_bind_param(Decimal("1.005"), None) == 101  # type: ignore[arg-type]
    assert t.process_result_value(101, None) == Decimal("1.01")  # type: ignore[arg-type]


def test_nickname_is_unique(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        s.add(User(nickname="dup"))
    with pytest.raises(IntegrityError), write_session(factory) as s:
        s.add(User(nickname="dup"))


def test_foreign_keys_enforced(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with pytest.raises(IntegrityError), write_session(factory) as s:
        s.add(Schedule(companion_id=999, start=START, end=START, status=ScheduleStatus.OPEN))


def test_invalid_enum_value_rejected(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        s.add(_companion())
    with (
        pytest.raises(StatementError, match="not among the defined enum values"),
        write_session(factory) as s,
    ):
        c = s.scalars(select(Companion)).one()
        c.gender = "robot"  # type: ignore[assignment]
        s.flush()


@pytest.mark.parametrize(
    ("path", "to"),
    [
        ([], BookingStatus.CONFIRMED),
        ([BookingStatus.CONFIRMED], BookingStatus.COMPLETED),
        ([], BookingStatus.CANCELLED),
        ([BookingStatus.CONFIRMED], BookingStatus.CANCELLED),
    ],
)
def test_allowed_transitions(path: list[BookingStatus], to: BookingStatus) -> None:
    b = Booking(status=BookingStatus.PENDING_PAYMENT)
    for step in path:
        b.transition(step)
    b.transition(to)
    assert b.status is to


@pytest.mark.parametrize(
    ("status", "to"),
    [
        (BookingStatus.PENDING_PAYMENT, BookingStatus.COMPLETED),
        (BookingStatus.PENDING_PAYMENT, BookingStatus.PENDING_PAYMENT),
        (BookingStatus.CONFIRMED, BookingStatus.PENDING_PAYMENT),
        (BookingStatus.CANCELLED, BookingStatus.CONFIRMED),
        (BookingStatus.CANCELLED, BookingStatus.CANCELLED),
        (BookingStatus.COMPLETED, BookingStatus.CANCELLED),
    ],
)
def test_illegal_transitions_raise(status: BookingStatus, to: BookingStatus) -> None:
    b = Booking(id=7, status=status)
    assert not b.can_transition(to)
    with pytest.raises(InvalidTransition, match=f"from {status.value} to {to.value}") as err:
        b.transition(to)
    assert err.value.code is ErrorCode.INVALID_ARGUMENT
    assert err.value.to_dict()["details"] == {"booking_id": 7, "status": status.value}
    assert b.status is status


def test_write_session_uses_begin_immediate(engine: Engine) -> None:
    seen: list[str] = []
    event.listen(engine, "before_cursor_execute", lambda _c, _cur, stmt, *_: seen.append(stmt))
    factory = make_session_factory(engine)
    with write_session(factory) as s:
        s.add(User(nickname="w"))
    with read_session(factory) as s:
        s.scalars(select(User)).all()
    assert [x for x in seen if x.startswith("BEGIN")] == ["BEGIN IMMEDIATE", "BEGIN DEFERRED"]


def test_write_session_rolls_back_on_error(engine: Engine) -> None:
    factory = make_session_factory(engine)
    with pytest.raises(RuntimeError), write_session(factory) as s:
        s.add(User(nickname="ghost"))
        s.flush()
        raise RuntimeError("boom")
    with read_session(factory) as s:
        assert s.scalars(select(User)).all() == []


def test_in_memory_engine_shares_one_database() -> None:
    eng = create_db_engine(":memory:")
    init_db(eng)
    factory = make_session_factory(eng)
    with write_session(factory) as s:
        s.add(User(nickname="mem"))
    with read_session(factory) as s:
        assert s.scalars(select(User.nickname)).all() == ["mem"]


@pytest.mark.parametrize(
    ("target", "url"),
    [
        (":memory:", "sqlite://"),
        ("sqlite:///x.db", "sqlite:///x.db"),
        (Path("data") / "booking.db", "sqlite:///data/booking.db"),
    ],
)
def test_database_url(target: str | Path, url: str) -> None:
    assert database_url(target) == url
