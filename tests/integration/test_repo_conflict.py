"""D3: repositories — SQL hard filter, availability and atomic booking creation."""

from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import select

from booking_mcp.db import (
    Booking,
    BookingStatus,
    Schedule,
    ScheduleStatus,
    read_session,
    write_session,
)
from booking_mcp.db.repo import (
    BookingRepo,
    CompanionRepo,
    ScheduleRepo,
    UserRepo,
    candidate_starts,
    to_profile,
)
from booking_mcp.errors import InvalidTransition, NotFound, SlotConflict
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_domain.matching import CompanionFilter, matches
from rift_domain.pricing import quote

if TYPE_CHECKING:
    from tests.integration.conftest import BookingDB

T20 = datetime(2026, 10, 3, 20, 0)


def _filter(**overrides: object) -> CompanionFilter:
    fields: dict[str, object] = {
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "start_time": T20,
        "duration_hours": Decimal(2),
        "service_type": ServiceType.CLIMB,
    }
    fields.update(overrides)
    return CompanionFilter(**fields)  # type: ignore[arg-type]


def _open_evening(db: BookingDB, cid: int) -> None:
    db.add_schedule(cid, T20 - timedelta(hours=2), T20 + timedelta(hours=4))


# --- availability -----------------------------------------------------------------------


@pytest.fixture
def one_companion(booking_db: BookingDB) -> int:
    cid = booking_db.add_companion("甲")
    _open_evening(booking_db, cid)  # 18:00-24:00
    booking_db.add_schedule(
        cid, T20 + timedelta(hours=2), T20 + timedelta(hours=3), ScheduleStatus.BLOCKED
    )  # 22:00-23:00 busy
    return cid


@pytest.mark.parametrize(
    ("start_h", "hours", "free"),
    [
        (20, 2, True),  # ends exactly where the blocked hour starts
        (18, 2, True),  # starts exactly when the window opens
        (23, 1, True),  # starts exactly when the block ends, ends at close
        (21, 2, False),  # overlaps the block
        (22, 1, False),  # is the block
        (17, 2, False),  # starts before the window
        (23, 2, False),  # runs past the window
    ],
)
def test_is_free(
    booking_db: BookingDB, one_companion: int, start_h: int, hours: int, free: bool
) -> None:
    start = T20.replace(hour=0) + timedelta(hours=start_h)
    with read_session(booking_db.factory) as s:
        assert ScheduleRepo(s).is_free(one_companion, start, start + timedelta(hours=hours)) is free


def test_free_ranges_cut_out_busy_time(booking_db: BookingDB, one_companion: int) -> None:
    day = T20.replace(hour=0)
    with read_session(booking_db.factory) as s:
        ranges = ScheduleRepo(s).free_ranges(one_companion, day, day + timedelta(days=1))
    assert ranges == [
        (day.replace(hour=18), day.replace(hour=22)),
        (day.replace(hour=23), day + timedelta(days=1)),
    ]


def test_free_ranges_clip_to_query_window(booking_db: BookingDB, one_companion: int) -> None:
    with read_session(booking_db.factory) as s:
        ranges = ScheduleRepo(s).free_ranges(one_companion, T20, T20 + timedelta(hours=1))
    assert ranges == [(T20, T20 + timedelta(hours=1))]


def test_released_hold_no_longer_blocks(booking_db: BookingDB, one_companion: int) -> None:
    uid = booking_db.add_user()
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    with write_session(booking_db.factory) as s:
        b = BookingRepo(s).create_atomic(
            user_id=uid, companion_id=one_companion, start=T20, game_mode=GameMode.ARAM, quote=q
        )
        bid = b.id
    with read_session(booking_db.factory) as s:
        assert not ScheduleRepo(s).is_free(one_companion, T20, T20 + timedelta(hours=2))
    with write_session(booking_db.factory) as s:
        repo = BookingRepo(s)
        repo.cancel(repo.require(bid), T20 - timedelta(days=2), Decimal("120"))
    with read_session(booking_db.factory) as s:
        assert ScheduleRepo(s).is_free(one_companion, T20, T20 + timedelta(hours=2))
        row = s.scalars(select(Schedule).where(Schedule.booking_id == bid)).one()
        assert row.status is ScheduleStatus.RELEASED


# --- static SQL filter ------------------------------------------------------------------


@pytest.fixture
def roster(booking_db: BookingDB) -> dict[str, int]:
    ids = {
        "gold_f_mid": booking_db.add_companion(
            "gold_f_mid", rank=Rank.GOLD, roles=(Role.MID,), hourly_price=Decimal("40")
        ),
        "dia_m_jg": booking_db.add_companion(
            "dia_m_jg",
            gender=Gender.MALE,
            rank=Rank.DIAMOND,
            roles=(Role.JUNGLE, Role.TOP),
            hourly_price=Decimal("70"),
        ),
        "master_f_sup_mute": booking_db.add_companion(
            "master_f_sup_mute",
            rank=Rank.MASTER,
            roles=(Role.SUPPORT,),
            voice=False,
            hourly_price=Decimal("83.33"),
        ),
        "chall_casual_only": booking_db.add_companion(
            "chall_casual_only",
            rank=Rank.CHALLENGER,
            modes=(GameMode.ARAM, GameMode.NORMAL_DRAFT),
            service_types=(ServiceType.CASUAL,),
            hourly_price=Decimal("120"),
        ),
    }
    for cid in ids.values():
        _open_evening(booking_db, cid)
    return ids


FILTER_CASES: list[tuple[str, dict[str, object], set[str]]] = [
    ("mode and service only", {}, {"gold_f_mid", "dia_m_jg", "master_f_sup_mute"}),
    ("aram casual", {"game_mode": GameMode.ARAM, "service_type": ServiceType.CASUAL},
     {"gold_f_mid", "dia_m_jg", "master_f_sup_mute", "chall_casual_only"}),
    ("rank band", {"rank_min": Rank.PLATINUM, "rank_max": Rank.DIAMOND}, {"dia_m_jg"}),
    ("rank min only", {"rank_min": Rank.DIAMOND}, {"dia_m_jg", "master_f_sup_mute"}),
    ("any of roles", {"roles": (Role.TOP, Role.SUPPORT)}, {"dia_m_jg", "master_f_sup_mute"}),
    ("gender", {"gender": Gender.MALE}, {"dia_m_jg"}),
    ("voice", {"voice_required": True}, {"gold_f_mid", "dia_m_jg"}),
    ("budget exact after multiplier",
     {"max_price_per_hour": Decimal(100), "price_multiplier": Decimal("1.2")},
     {"gold_f_mid", "dia_m_jg", "master_f_sup_mute"}),
    ("budget just below", {"max_price_per_hour": Decimal("83.32")}, {"gold_f_mid", "dia_m_jg"}),
    ("name", {"companion_name": "dia_m_jg"}, {"dia_m_jg"}),
]  # fmt: skip


@pytest.mark.parametrize(("label", "overrides", "expected"), FILTER_CASES)
def test_sql_filter_matches_domain_predicate(
    booking_db: BookingDB,
    roster: dict[str, int],
    label: str,
    overrides: dict[str, object],
    expected: set[str],
) -> None:
    f = _filter(**overrides)
    with read_session(booking_db.factory) as s:
        repo = CompanionRepo(s, booking_db.domain)
        sql_names = {c.name for c in s.scalars(repo.static_query(f))}
        py_names = {c.name for c in repo.list_all() if matches(to_profile(c), f, booking_db.domain)}
        found = {a.profile.name for a in repo.search(f)}
    assert sql_names == expected, label
    assert py_names == sql_names, label  # SQL path and in-memory predicate agree
    assert found == expected, label  # everyone is free at 20:00


# --- time-window search -----------------------------------------------------------------


def test_candidate_starts_order_and_not_before() -> None:
    f = _filter(time_shift_hours=Decimal(1))
    assert candidate_starts(f) == [
        T20,
        T20 + timedelta(minutes=30),
        T20 - timedelta(minutes=30),
        T20 + timedelta(hours=1),
        T20 - timedelta(hours=1),
    ]
    assert candidate_starts(f, not_before=T20) == [
        T20,
        T20 + timedelta(minutes=30),
        T20 + timedelta(hours=1),
    ]
    assert candidate_starts(_filter(), not_before=T20 + timedelta(minutes=1)) == []


def test_search_finds_shifted_start_only_when_window_relaxed(booking_db: BookingDB) -> None:
    cid = booking_db.add_companion("晚班")
    booking_db.add_schedule(cid, T20 + timedelta(hours=1), T20 + timedelta(hours=5))
    with read_session(booking_db.factory) as s:
        repo = CompanionRepo(s, booking_db.domain)
        assert repo.search(_filter()) == []
        [hit] = repo.search(_filter(time_shift_hours=Decimal(1)))
        assert repo.search(_filter(), not_before=T20 + timedelta(hours=1)) == []
    assert hit.profile.id == cid
    assert hit.start_time == T20 + timedelta(hours=1)


def test_search_returns_requested_start_when_free(
    booking_db: BookingDB, roster: dict[str, int]
) -> None:
    with read_session(booking_db.factory) as s:
        hits = CompanionRepo(s, booking_db.domain).search(_filter(time_shift_hours=Decimal(1)))
    assert {h.start_time for h in hits} == {T20}
    assert [h.profile.id for h in hits] == sorted(h.profile.id for h in hits)


# --- users -------------------------------------------------------------------------------


def test_user_get_or_create_is_idempotent(booking_db: BookingDB) -> None:
    with write_session(booking_db.factory) as s:
        first = UserRepo(s).get_or_create("小明").id
    with write_session(booking_db.factory) as s:
        assert UserRepo(s).get_or_create("小明").id == first
        with pytest.raises(NotFound, match="user 999"):
            UserRepo(s).require(999)


# --- atomic booking ----------------------------------------------------------------------


def test_create_atomic_writes_booking_and_hold(booking_db: BookingDB, one_companion: int) -> None:
    uid = booking_db.add_user()
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    with write_session(booking_db.factory) as s:
        b = BookingRepo(s).create_atomic(
            user_id=uid,
            companion_id=one_companion,
            start=T20,
            game_mode=GameMode.RANKED_SOLO_DUO,
            quote=q,
        )
        bid = b.id
    with read_session(booking_db.factory) as s:
        b = BookingRepo(s).require(bid)
        assert b.status is BookingStatus.PENDING_PAYMENT
        assert (b.start, b.end) == (T20, T20 + timedelta(hours=2))
        assert (b.unit_price, b.multiplier, b.total) == (Decimal("50.00"), Decimal("1.2"), q.total)
        hold = s.scalars(select(Schedule).where(Schedule.booking_id == bid)).one()
        assert hold.status is ScheduleStatus.BOOKED
        assert BookingRepo(s).list_for_user(uid) == [b]
        assert BookingRepo(s).list_for_user(uid, [BookingStatus.CANCELLED]) == []
        assert BookingRepo(s).active_overlapping(
            uid, T20 + timedelta(hours=1), T20 + timedelta(hours=3)
        ) == [b]


def test_create_atomic_rejects_taken_slot(booking_db: BookingDB, one_companion: int) -> None:
    uid = booking_db.add_user()
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    with pytest.raises(SlotConflict) as err, write_session(booking_db.factory) as s:
        BookingRepo(s).create_atomic(
            user_id=uid,
            companion_id=one_companion,
            start=T20 + timedelta(hours=1),  # 21-23 overlaps the 22-23 block
            game_mode=GameMode.ARAM,
            quote=q,
        )
    assert err.value.to_dict()["code"] == "SLOT_CONFLICT"
    with read_session(booking_db.factory) as s:
        assert s.scalars(select(Booking)).all() == []


def test_create_atomic_requires_write_session(booking_db: BookingDB, one_companion: int) -> None:
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    with (
        pytest.raises(RuntimeError, match="write_session"),
        read_session(booking_db.factory) as s,
    ):
        BookingRepo(s).create_atomic(
            user_id=1, companion_id=one_companion, start=T20, game_mode=GameMode.ARAM, quote=q
        )


@pytest.mark.parametrize(
    ("user", "companion", "what"), [(999, None, "user"), (None, 999, "companion")]
)
def test_create_atomic_unknown_ids(
    booking_db: BookingDB, one_companion: int, user: int | None, companion: int | None, what: str
) -> None:
    uid = booking_db.add_user()
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    with pytest.raises(NotFound, match=what), write_session(booking_db.factory) as s:
        BookingRepo(s).create_atomic(
            user_id=user or uid,
            companion_id=companion or one_companion,
            start=T20,
            game_mode=GameMode.ARAM,
            quote=q,
        )


def test_concurrent_bookings_for_same_slot_only_one_wins(
    booking_db: BookingDB, one_companion: int
) -> None:
    users = [booking_db.add_user(f"u{i}") for i in range(2)]
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    barrier = threading.Barrier(len(users))
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(uid: int) -> None:
        barrier.wait()
        try:
            with write_session(booking_db.factory) as s:
                BookingRepo(s).create_atomic(
                    user_id=uid,
                    companion_id=one_companion,
                    start=T20,
                    game_mode=GameMode.ARAM,
                    quote=q,
                )
            result = "ok"
        except SlotConflict:
            result = "conflict"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(u,)) for u in users]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(outcomes) == ["conflict", "ok"]
    with read_session(booking_db.factory) as s:
        assert len(s.scalars(select(Booking)).all()) == 1
        holds = s.scalars(select(Schedule).where(Schedule.status == ScheduleStatus.BOOKED)).all()
        assert len(holds) == 1


def test_many_concurrent_bookings_for_different_slots_all_win(booking_db: BookingDB) -> None:
    cids = [booking_db.add_companion(f"c{i}") for i in range(4)]
    for cid in cids:
        _open_evening(booking_db, cid)
    uid = booking_db.add_user()
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    errors: list[BaseException] = []

    def attempt(cid: int) -> None:
        try:
            with write_session(booking_db.factory) as s:
                BookingRepo(s).create_atomic(
                    user_id=uid, companion_id=cid, start=T20, game_mode=GameMode.ARAM, quote=q
                )
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=attempt, args=(c,)) for c in cids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    with read_session(booking_db.factory) as s:
        assert len(s.scalars(select(Booking)).all()) == 4


def test_pay_and_cancel_transitions(booking_db: BookingDB, one_companion: int) -> None:
    uid = booking_db.add_user()
    q = quote(Decimal(50), ServiceType.CLIMB, 2, booking_db.domain)
    now = T20 - timedelta(days=1)
    with write_session(booking_db.factory) as s:
        repo = BookingRepo(s)
        b = repo.create_atomic(
            user_id=uid, companion_id=one_companion, start=T20, game_mode=GameMode.ARAM, quote=q
        )
        repo.pay(b, now)
        assert (b.status, b.paid_at) == (BookingStatus.CONFIRMED, now)
        repo.cancel(b, now, Decimal("60"))
        assert (b.status, b.cancelled_at, b.refund_amount) == (
            BookingStatus.CANCELLED,
            now,
            Decimal("60"),
        )
        with pytest.raises(InvalidTransition):
            repo.pay(b, now)
        with pytest.raises(NotFound, match="booking 999"):
            repo.require(999)


def test_relaxed_filter_does_not_change_static_constraints(
    booking_db: BookingDB, roster: dict[str, int]
) -> None:
    f = _filter(gender=Gender.MALE)
    with read_session(booking_db.factory) as s:
        repo = CompanionRepo(s, booking_db.domain)
        strict = {a.profile.name for a in repo.search(f)}
        relaxed = {a.profile.name for a in repo.search(replace(f, gender=None))}
    assert strict == {"dia_m_jg"}
    assert strict < relaxed
