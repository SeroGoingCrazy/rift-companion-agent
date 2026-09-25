"""``create_booking``: quote, then write the order and hold the slot in one transaction."""

from __future__ import annotations

from booking_mcp.db import read_session, write_session
from booking_mcp.db.repo import BookingRepo, CompanionRepo
from booking_mcp.errors import InvalidArgument
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import BookingOut, CreateBookingInput
from rift_domain.pricing import PricingError, quote

NAME = "create_booking"
DESCRIPTION = (
    "Create a booking (status pending_payment) for a companion at a start time. The slot "
    "is re-checked inside the write transaction; if someone took it first the call fails "
    "with SLOT_CONFLICT. Price = hourly price x service multiplier x hours."
)


def create_booking(service: BookingService, args: CreateBookingInput) -> BookingOut:
    domain = service.domain
    if args.start_time < service.now():
        raise InvalidArgument("start_time is in the past", start_time=args.start_time.isoformat())
    service_type = args.service_type or domain.default_service_type(args.game_mode)

    with read_session(service.factory) as session:
        companion = CompanionRepo(session, domain).require(args.companion_id)
        if args.game_mode not in companion.modes:
            raise InvalidArgument(
                f"{companion.name} does not play {args.game_mode.value}",
                offered=[m.value for m in companion.modes],
            )
        if service_type not in companion.service_types:
            raise InvalidArgument(
                f"{companion.name} does not offer {service_type.value}",
                offered=[s.value for s in companion.service_types],
            )
        try:
            q = quote(companion.hourly_price, service_type, args.duration_hours, domain)
        except PricingError as exc:
            raise InvalidArgument(str(exc)) from exc

    with write_session(service.factory) as session:
        booking = BookingRepo(session).create_atomic(
            user_id=args.user_id,
            companion_id=args.companion_id,
            start=args.start_time,
            game_mode=args.game_mode,
            quote=q,
        )
        return BookingOut.from_model(booking)
