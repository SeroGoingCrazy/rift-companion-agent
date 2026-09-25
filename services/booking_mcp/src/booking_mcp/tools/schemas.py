"""Tool input/output models. Their JSON Schemas are what MCP clients see.

Money is serialized as a decimal string (``"144.00"``) so no client ever rounds it.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, WithJsonSchema

from booking_mcp.db.models import Booking, BookingStatus
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType

Money = Annotated[Decimal, Field(description="decimal string with 2 places, e.g. '144.00'")]
HoursIn = Annotated[float, Field(gt=0, le=24, multiple_of=0.5, description="hours, 0.5 steps")]

# Booking times are local wall-clock values. JSON Schema's date-time format
# requires a UTC offset, so advertising it makes strict MCP clients reject
# otherwise valid responses containing our naive datetime serialization.
LocalDateTime = Annotated[
    datetime,
    WithJsonSchema(
        {
            "type": "string",
            "pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$",
            "description": "ISO 8601 local wall-clock time without a UTC offset",
        },
        mode="serialization",
    ),
]


class _In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- find_companions -----------------------------------------------------------------------


class FindCompanionsInput(_In):
    game_mode: GameMode
    start_time: datetime = Field(description="local time, ISO 8601 e.g. 2026-10-02T20:00")
    duration_hours: HoursIn
    rank_requirement: Rank | None = Field(
        default=None, description="minimum solo/duo rank (ranked modes only)"
    )
    role_preference: list[Role] | None = Field(
        default=None, description="companion must play at least one of these"
    )
    service_type: ServiceType | None = Field(
        default=None, description="defaults to the mode's usual service (ranked -> climb)"
    )
    companion_gender: Gender | None = None
    voice_required: bool | None = None
    budget_per_hour: float | None = Field(
        default=None, gt=0, description="max price per hour the user pays (never relaxed)"
    )
    style_preference: str | None = Field(
        default=None, max_length=100, description="free text, e.g. 温柔会聊天"
    )
    companion_name: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)


class CandidateOut(_Out):
    companion_id: int
    name: str
    gender: Gender
    rank: Rank
    roles: list[Role]
    hourly_price: Money
    effective_hourly_price: Money = Field(description="hourly_price x service multiplier")
    voice: bool
    rating: float
    tags: list[str]
    bio: str
    score: float
    reasons: list[str]
    start_time: LocalDateTime = Field(description="bookable start (shifted if time was relaxed)")


class FindCompanionsOutput(_Out):
    service_type: ServiceType
    candidates: list[CandidateOut]
    #: Relaxation steps applied to get these results, in order (empty = exact match).
    relaxations: list[str]
    relaxation_notes: list[str]
    #: Steps attempted when even the fully relaxed search found nobody.
    relaxations_tried: list[str] = []


# --- quote_price -----------------------------------------------------------------------------


class QuotePriceInput(_In):
    companion_id: int
    service_type: ServiceType
    duration_hours: HoursIn


class QuoteOut(_Out):
    companion_id: int
    companion_name: str
    service_type: ServiceType
    unit_price: Money
    multiplier: Decimal
    hours: Decimal
    effective_hourly: Money
    total: Money


# --- bookings --------------------------------------------------------------------------------


class BookingOut(_Out):
    booking_id: int
    user_id: int
    companion_id: int
    companion_name: str
    start_time: LocalDateTime
    end_time: LocalDateTime
    game_mode: GameMode
    service_type: ServiceType
    hours: Decimal
    unit_price: Money
    multiplier: Decimal
    total: Money
    status: BookingStatus
    created_at: LocalDateTime
    paid_at: LocalDateTime | None = None
    cancelled_at: LocalDateTime | None = None
    refund_amount: Money | None = None

    @classmethod
    def from_model(cls, b: Booking) -> BookingOut:
        return cls(
            booking_id=b.id,
            user_id=b.user_id,
            companion_id=b.companion_id,
            companion_name=b.companion.name,
            start_time=b.start,
            end_time=b.end,
            game_mode=b.game_mode,
            service_type=b.service_type,
            hours=b.hours,
            unit_price=b.unit_price,
            multiplier=b.multiplier,
            total=b.total,
            status=b.status,
            created_at=b.created_at,
            paid_at=b.paid_at,
            cancelled_at=b.cancelled_at,
            refund_amount=b.refund_amount,
        )


# --- create_booking / list_my_bookings / cancel_booking ------------------------------------


class CreateBookingInput(_In):
    user_id: int
    companion_id: int
    start_time: datetime = Field(description="local time, ISO 8601 e.g. 2026-10-02T20:00")
    duration_hours: HoursIn
    game_mode: GameMode
    service_type: ServiceType | None = Field(
        default=None, description="defaults to the mode's usual service (ranked -> climb)"
    )


class ListMyBookingsInput(_In):
    user_id: int
    status: BookingStatus | None = None


class BookingListOut(_Out):
    bookings: list[BookingOut]


class CancelBookingInput(_In):
    user_id: int
    booking_id: int
    dry_run: bool = Field(
        default=True, description="true: only report the refund; false: really cancel"
    )


class CancelOut(_Out):
    booking_id: int
    dry_run: bool
    #: Whether the booking had been paid; unpaid bookings refund nothing.
    paid: bool
    refund_tier: str
    refund_label: str
    refund_ratio: Decimal
    #: What the refund policy gives for a paid booking at this time.
    policy_refund_amount: Money
    #: What the user actually gets back (0 when unpaid).
    refund_amount: Money
    hours_before_start: float
    booking: BookingOut


# --- web support: users ----------------------------------------------------------------------

Nickname = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16)]


class EnsureUserInput(_In):
    nickname: Nickname


class UserOut(_Out):
    user_id: int
    nickname: str
    created: bool


class PayBookingInput(_In):
    user_id: int
    booking_id: int


# --- web support: companion list -------------------------------------------------------------


class BrowseCompanionsInput(_In):
    game_mode: GameMode | None = None
    rank_requirement: Rank | None = Field(
        default=None, description="same meaning as in find_companions (ranked modes only)"
    )
    role: Role | None = None
    companion_gender: Gender | None = None
    days: int = Field(default=3, ge=1, le=14, description="free-slot horizon from now")


class FreeSlot(_Out):
    start: LocalDateTime
    end: LocalDateTime


class CompanionCard(_Out):
    companion_id: int
    name: str
    gender: Gender
    rank: Rank
    roles: list[Role]
    modes: list[GameMode]
    service_types: list[ServiceType]
    hourly_price: Money
    #: service type -> effective hourly price
    prices: dict[str, Money]
    voice: bool
    rating: float
    level: str
    tags: list[str]
    bio: str
    free_slots: list[FreeSlot]


class BrowseCompanionsOutput(_Out):
    companions: list[CompanionCard]
