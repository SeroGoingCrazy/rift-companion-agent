"""Repositories: companion search (``CompanionFilter`` -> SQL), availability, bookings.

Availability rule: ``[start, end)`` is free when one ``open`` window covers it and no
``blocked``/``booked`` row overlaps it (touching ends do not overlap).

``BookingRepo.create_atomic`` must run in a ``write_session``: the ``BEGIN IMMEDIATE``
there takes SQLite's write lock before the availability re-check, so of two concurrent
bookings for one slot the second waits, then sees the first and raises ``SlotConflict``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Decimal

from sqlalchemy import Select, and_, select
from sqlalchemy.orm import Session

from booking_mcp.db.models import (
    ACTIVE_STATUSES,
    BUSY_STATUSES,
    Booking,
    BookingStatus,
    Companion,
    CompanionMode,
    CompanionRole,
    CompanionService,
    Schedule,
    ScheduleStatus,
    User,
)
from booking_mcp.db.session import BEGIN_MODE_OPTION
from booking_mcp.errors import NotFound, SlotConflict
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode
from rift_domain.matching import CompanionFilter, CompanionProfile
from rift_domain.slots import Quote

CENT = Decimal("0.01")
#: Granularity of alternative start times tried inside a relaxed time window.
START_STEP = timedelta(minutes=30)


def to_profile(c: Companion) -> CompanionProfile:
    return CompanionProfile(
        id=c.id,
        name=c.name,
        gender=c.gender,
        rank=c.rank,
        roles=c.roles,
        modes=c.modes,
        service_types=c.service_types,
        hourly_price=c.hourly_price,
        voice=c.voice,
        rating=c.rating,
        tags=tuple(c.tags),
        bio=c.bio,
    )


@dataclass(frozen=True)
class AvailableCompanion:
    profile: CompanionProfile
    #: Start time that fits the companion's schedule; differs from the requested one only
    #: when the time window was relaxed.
    start_time: datetime


def candidate_starts(f: CompanionFilter, not_before: datetime | None = None) -> list[datetime]:
    """Requested start first, then alternatives by growing distance (later before earlier)."""
    starts = [f.start_time]
    steps = int(timedelta(hours=float(f.time_shift_hours)) / START_STEP)
    for k in range(1, steps + 1):
        starts += [f.start_time + k * START_STEP, f.start_time - k * START_STEP]
    return [s for s in starts if not_before is None or s >= not_before]


def _fits(rows: Sequence[Schedule], start: datetime, end: datetime) -> bool:
    covered = any(
        r.status is ScheduleStatus.OPEN and r.start <= start and r.end >= end for r in rows
    )
    if not covered:
        return False
    return not any(r.status in BUSY_STATUSES and r.start < end and r.end > start for r in rows)


class ScheduleRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def rows_between(self, companion_id: int, start: datetime, end: datetime) -> list[Schedule]:
        """Schedule rows of one companion that overlap ``[start, end)``."""
        stmt = select(Schedule).where(
            Schedule.companion_id == companion_id,
            Schedule.start < end,
            Schedule.end > start,
            Schedule.status != ScheduleStatus.RELEASED,
        )
        return list(self.session.scalars(stmt))

    def is_free(self, companion_id: int, start: datetime, end: datetime) -> bool:
        return _fits(self.rows_between(companion_id, start, end), start, end)

    def free_ranges(
        self, companion_id: int, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Open time inside ``[start, end)`` with busy rows cut out, in order."""
        rows = self.rows_between(companion_id, start, end)
        busy = sorted((r.start, r.end) for r in rows if r.status in BUSY_STATUSES)
        ranges: list[tuple[datetime, datetime]] = []
        for w in sorted(
            (r for r in rows if r.status is ScheduleStatus.OPEN), key=lambda r: r.start
        ):
            cursor, stop = max(w.start, start), min(w.end, end)
            for b_start, b_end in busy:
                if b_end <= cursor or b_start >= stop:
                    continue
                if b_start > cursor:
                    ranges.append((cursor, b_start))
                cursor = max(cursor, b_end)
            if cursor < stop:
                ranges.append((cursor, stop))
        return ranges

    def hold(self, companion_id: int, start: datetime, end: datetime, booking_id: int) -> Schedule:
        row = Schedule(
            companion_id=companion_id,
            start=start,
            end=end,
            status=ScheduleStatus.BOOKED,
            booking_id=booking_id,
        )
        self.session.add(row)
        return row

    def release(self, booking_id: int) -> int:
        rows = self.session.scalars(
            select(Schedule).where(
                Schedule.booking_id == booking_id, Schedule.status == ScheduleStatus.BOOKED
            )
        ).all()
        for r in rows:
            r.status = ScheduleStatus.RELEASED
        return len(rows)


class CompanionRepo:
    def __init__(self, session: Session, domain: DomainConfig) -> None:
        self.session = session
        self.domain = domain

    def get(self, companion_id: int) -> Companion | None:
        return self.session.get(Companion, companion_id)

    def require(self, companion_id: int) -> Companion:
        c = self.get(companion_id)
        if c is None:
            raise NotFound(f"companion {companion_id} not found", companion_id=companion_id)
        return c

    def get_by_name(self, name: str) -> Companion | None:
        return self.session.scalars(select(Companion).where(Companion.name == name)).first()

    def list_all(self) -> list[Companion]:
        return list(self.session.scalars(select(Companion).order_by(Companion.id)))

    def static_query(self, f: CompanionFilter) -> Select[Companion]:
        """Every non-schedule constraint of ``f`` as SQL (mirrors ``matching.matches``)."""
        conds = [
            Companion.mode_rows.any(CompanionMode.game_mode == f.game_mode),
            Companion.service_rows.any(CompanionService.service_type == f.service_type),
        ]
        if f.rank_min is not None:
            conds.append(Companion.rank_tier >= self.domain.rank_index(f.rank_min))
        if f.rank_max is not None:
            conds.append(Companion.rank_tier <= self.domain.rank_index(f.rank_max))
        if f.roles is not None:
            conds.append(Companion.role_rows.any(CompanionRole.role.in_(f.roles)))
        if f.gender is not None:
            conds.append(Companion.gender == f.gender)
        if f.voice_required:
            conds.append(Companion.voice.is_(True))
        budget = f.max_base_price
        if budget is not None:
            # Floor to cents: prices are stored in cents, so this keeps "price <= budget".
            conds.append(Companion.hourly_price <= budget.quantize(CENT, ROUND_FLOOR))
        if f.companion_name is not None:
            conds.append(Companion.name == f.companion_name)
        return select(Companion).where(and_(*conds)).order_by(Companion.id)

    def search(
        self, f: CompanionFilter, *, not_before: datetime | None = None
    ) -> list[AvailableCompanion]:
        """Companions passing the hard filter that are free for some allowed start time."""
        starts = candidate_starts(f, not_before)
        if not starts:
            return []
        span = timedelta(hours=float(f.duration_hours))
        lo, hi = min(starts), max(starts) + span
        schedules = ScheduleRepo(self.session)
        found: list[AvailableCompanion] = []
        for c in self.session.scalars(self.static_query(f)):
            rows = schedules.rows_between(c.id, lo, hi)
            start = next((s for s in starts if _fits(rows, s, s + span)), None)
            if start is not None:
                found.append(AvailableCompanion(to_profile(c), start))
        return found


class UserRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)

    def require(self, user_id: int) -> User:
        u = self.get(user_id)
        if u is None:
            raise NotFound(f"user {user_id} not found", user_id=user_id)
        return u

    def get_or_create(self, nickname: str) -> User:
        user = self.session.scalars(select(User).where(User.nickname == nickname)).first()
        if user is None:
            user = User(nickname=nickname)
            self.session.add(user)
            self.session.flush()
        return user


class BookingRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, booking_id: int) -> Booking | None:
        return self.session.get(Booking, booking_id)

    def require(self, booking_id: int) -> Booking:
        b = self.get(booking_id)
        if b is None:
            raise NotFound(f"booking {booking_id} not found", booking_id=booking_id)
        return b

    def list_for_user(
        self, user_id: int, statuses: Sequence[BookingStatus] | None = None
    ) -> list[Booking]:
        stmt = select(Booking).where(Booking.user_id == user_id)
        if statuses:
            stmt = stmt.where(Booking.status.in_(statuses))
        return list(self.session.scalars(stmt.order_by(Booking.start, Booking.id)))

    def active_overlapping(self, user_id: int, start: datetime, end: datetime) -> list[Booking]:
        stmt = select(Booking).where(
            Booking.user_id == user_id,
            Booking.status.in_(ACTIVE_STATUSES),
            Booking.start < end,
            Booking.end > start,
        )
        return list(self.session.scalars(stmt))

    def create_atomic(
        self,
        *,
        user_id: int,
        companion_id: int,
        start: datetime,
        game_mode: GameMode,
        quote: Quote,
    ) -> Booking:
        """Re-check availability and write the booking plus its schedule hold.

        Raises ``SlotConflict`` if the slot is taken, ``NotFound`` for unknown ids.
        """
        mode = self.session.connection().get_execution_options().get(BEGIN_MODE_OPTION)
        if mode != "IMMEDIATE":
            raise RuntimeError("create_atomic must run inside write_session (BEGIN IMMEDIATE)")
        UserRepo(self.session).require(user_id)
        if self.session.get(Companion, companion_id) is None:
            raise NotFound(f"companion {companion_id} not found", companion_id=companion_id)

        end = start + timedelta(hours=float(quote.hours))
        if not ScheduleRepo(self.session).is_free(companion_id, start, end):
            raise SlotConflict(
                f"companion {companion_id} is not available {start:%Y-%m-%d %H:%M}–{end:%H:%M}",
                companion_id=companion_id,
                start_time=start.isoformat(),
                end_time=end.isoformat(),
            )
        booking = Booking(
            user_id=user_id,
            companion_id=companion_id,
            start=start,
            end=end,
            game_mode=game_mode,
            service_type=quote.service_type,
            hours=quote.hours,
            unit_price=quote.unit_price,
            multiplier=quote.multiplier,
            total=quote.total,
            status=BookingStatus.PENDING_PAYMENT,
        )
        self.session.add(booking)
        self.session.flush()
        ScheduleRepo(self.session).hold(companion_id, start, end, booking.id)
        self.session.flush()
        return booking

    def pay(self, booking: Booking, now: datetime) -> Booking:
        booking.transition(BookingStatus.CONFIRMED)
        booking.paid_at = now
        return booking

    def cancel(self, booking: Booking, now: datetime, refund_amount: Decimal) -> Booking:
        booking.transition(BookingStatus.CANCELLED)
        booking.cancelled_at = now
        booking.refund_amount = refund_amount
        ScheduleRepo(self.session).release(booking.id)
        return booking


__all__ = [
    "AvailableCompanion",
    "BookingRepo",
    "CompanionRepo",
    "ScheduleRepo",
    "UserRepo",
    "candidate_starts",
    "to_profile",
]
