"""SQLAlchemy 2.x models: users, companions, schedules, bookings.

Conventions:
- Datetimes are naive and in platform local time (the same clock the time parser uses).
- Decimals (money, hours, multipliers) are stored as scaled integers so SQLite compares
  them exactly; see ``ScaledDecimal``.
- A companion's availability is a set of ``open`` schedule windows minus overlapping
  ``blocked`` (taken offline) and ``booked`` (held by a booking) rows.
"""

from __future__ import annotations

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Dialect,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from booking_mcp.errors import InvalidTransition
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType


class ScaledDecimal(TypeDecorator[Decimal]):
    """Decimal stored as ``round(value * 10**scale)`` in an INTEGER column."""

    impl = Integer
    cache_ok = True

    def __init__(self, scale: int = 2) -> None:
        super().__init__()
        self.scale = scale
        self._factor = Decimal(10) ** scale
        self._quantum = Decimal(1).scaleb(-scale)

    def process_bind_param(self, value: Any, dialect: Dialect) -> int | None:
        if value is None:
            return None
        scaled = (Decimal(str(value)) * self._factor).quantize(Decimal(1), ROUND_HALF_UP)
        return int(scaled)

    def process_result_value(self, value: Any, dialect: Dialect) -> Decimal | None:
        if value is None:
            return None
        return (Decimal(value) / self._factor).quantize(self._quantum)


def _enum(cls: type[StrEnum]) -> Enum:
    return Enum(
        cls,
        native_enum=False,
        length=32,
        validate_strings=True,
        values_callable=lambda members: [m.value for m in members],
    )


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    nickname: Mapped[str] = mapped_column(String(32), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class CompanionRole(Base):
    __tablename__ = "companion_roles"

    companion_id: Mapped[int] = mapped_column(
        ForeignKey("companions.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[Role] = mapped_column(_enum(Role), primary_key=True)


class CompanionMode(Base):
    __tablename__ = "companion_modes"

    companion_id: Mapped[int] = mapped_column(
        ForeignKey("companions.id", ondelete="CASCADE"), primary_key=True
    )
    game_mode: Mapped[GameMode] = mapped_column(_enum(GameMode), primary_key=True)


class CompanionService(Base):
    __tablename__ = "companion_service_types"

    companion_id: Mapped[int] = mapped_column(
        ForeignKey("companions.id", ondelete="CASCADE"), primary_key=True
    )
    service_type: Mapped[ServiceType] = mapped_column(_enum(ServiceType), primary_key=True)


class Companion(Base):
    __tablename__ = "companions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(32), unique=True)
    gender: Mapped[Gender] = mapped_column(_enum(Gender))
    #: Solo/duo rank (the only rank the platform tracks).
    rank: Mapped[Rank] = mapped_column(_enum(Rank))
    #: Position of ``rank`` on the ladder (0 = iron), so SQL can compare ranks.
    rank_tier: Mapped[int] = mapped_column(Integer, index=True)
    hourly_price: Mapped[Decimal] = mapped_column(ScaledDecimal(2))
    voice: Mapped[bool] = mapped_column(Boolean, default=True)
    rating: Mapped[float] = mapped_column(Float)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    bio: Mapped[str] = mapped_column(Text, default="")

    role_rows: Mapped[list[CompanionRole]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by=CompanionRole.role
    )
    mode_rows: Mapped[list[CompanionMode]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by=CompanionMode.game_mode
    )
    service_rows: Mapped[list[CompanionService]] = relationship(
        cascade="all, delete-orphan", lazy="selectin", order_by=CompanionService.service_type
    )

    @property
    def roles(self) -> tuple[Role, ...]:
        return tuple(r.role for r in self.role_rows)

    @property
    def modes(self) -> tuple[GameMode, ...]:
        return tuple(m.game_mode for m in self.mode_rows)

    @property
    def service_types(self) -> tuple[ServiceType, ...]:
        return tuple(s.service_type for s in self.service_rows)

    def set_skills(
        self,
        roles: tuple[Role, ...] | list[Role],
        modes: tuple[GameMode, ...] | list[GameMode],
        service_types: tuple[ServiceType, ...] | list[ServiceType],
    ) -> None:
        self.role_rows = [CompanionRole(role=r) for r in roles]
        self.mode_rows = [CompanionMode(game_mode=m) for m in modes]
        self.service_rows = [CompanionService(service_type=s) for s in service_types]


class ScheduleStatus(StrEnum):
    #: The companion takes orders in this window.
    OPEN = "open"
    #: Taken outside the platform (seeded "already busy" time).
    BLOCKED = "blocked"
    #: Held by a booking.
    BOOKED = "booked"
    #: A booking's hold that was released by cancellation (kept for history).
    RELEASED = "released"


#: Statuses that make a time range unavailable.
BUSY_STATUSES = (ScheduleStatus.BLOCKED, ScheduleStatus.BOOKED)


class Schedule(Base):
    __tablename__ = "schedules"
    __table_args__ = (Index("ix_schedules_companion_start", "companion_id", "start"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    companion_id: Mapped[int] = mapped_column(ForeignKey("companions.id", ondelete="CASCADE"))
    start: Mapped[datetime] = mapped_column(DateTime)
    end: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[ScheduleStatus] = mapped_column(_enum(ScheduleStatus))
    booking_id: Mapped[int | None] = mapped_column(ForeignKey("bookings.id"), default=None)


class BookingStatus(StrEnum):
    PENDING_PAYMENT = "pending_payment"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


#: pending_payment -> confirmed -> completed | cancelled (unpaid orders can be cancelled too).
ALLOWED_TRANSITIONS: dict[BookingStatus, frozenset[BookingStatus]] = {
    BookingStatus.PENDING_PAYMENT: frozenset({BookingStatus.CONFIRMED, BookingStatus.CANCELLED}),
    BookingStatus.CONFIRMED: frozenset({BookingStatus.COMPLETED, BookingStatus.CANCELLED}),
    BookingStatus.COMPLETED: frozenset(),
    BookingStatus.CANCELLED: frozenset(),
}

#: Bookings that still hold their schedule slot.
ACTIVE_STATUSES = (BookingStatus.PENDING_PAYMENT, BookingStatus.CONFIRMED)


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (Index("ix_bookings_user_status", "user_id", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    companion_id: Mapped[int] = mapped_column(ForeignKey("companions.id"))
    start: Mapped[datetime] = mapped_column(DateTime)
    end: Mapped[datetime] = mapped_column(DateTime)
    game_mode: Mapped[GameMode] = mapped_column(_enum(GameMode))
    service_type: Mapped[ServiceType] = mapped_column(_enum(ServiceType))
    hours: Mapped[Decimal] = mapped_column(ScaledDecimal(1))
    unit_price: Mapped[Decimal] = mapped_column(ScaledDecimal(2))
    multiplier: Mapped[Decimal] = mapped_column(ScaledDecimal(4))
    total: Mapped[Decimal] = mapped_column(ScaledDecimal(2))
    status: Mapped[BookingStatus] = mapped_column(
        _enum(BookingStatus), default=BookingStatus.PENDING_PAYMENT
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    refund_amount: Mapped[Decimal | None] = mapped_column(ScaledDecimal(2), default=None)

    companion: Mapped[Companion] = relationship(lazy="joined")

    @property
    def start_time(self) -> datetime:
        """``rift_domain.refund.BookingLike`` view."""
        return self.start

    def can_transition(self, to: BookingStatus) -> bool:
        return to in ALLOWED_TRANSITIONS[self.status]

    def transition(self, to: BookingStatus) -> None:
        if not self.can_transition(to):
            raise InvalidTransition(
                f"booking {self.id} cannot go from {self.status.value} to {to.value}",
                booking_id=self.id,
                status=self.status.value,
            )
        self.status = to
