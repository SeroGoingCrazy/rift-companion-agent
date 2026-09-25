"""``cancel_booking``: report the refund (``dry_run``) or cancel and release the slot.

The refund follows ``rift_domain.refund`` tiers; an unpaid booking is cancelled for free
(``refund_amount`` 0) but the policy amount is still reported.
"""

from __future__ import annotations

from decimal import Decimal

from booking_mcp.db import BookingStatus, write_session
from booking_mcp.db.repo import BookingRepo
from booking_mcp.errors import Forbidden, InvalidTransition
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import BookingOut, CancelBookingInput, CancelOut
from rift_domain.refund import refund

NAME = "cancel_booking"
DESCRIPTION = (
    "Cancel one of the user's bookings. With dry_run=true (default) only reports the refund "
    "(>=24h before start: full, 2-24h: half, <2h: none); with dry_run=false cancels it and "
    "frees the companion's time. Other users' bookings are FORBIDDEN."
)


def cancel_booking(service: BookingService, args: CancelBookingInput) -> CancelOut:
    now = service.now()
    # A write session even for dry runs: the check and the cancel see the same state.
    with write_session(service.factory) as session:
        repo = BookingRepo(session)
        booking = repo.require(args.booking_id)
        if booking.user_id != args.user_id:
            raise Forbidden(f"booking {booking.id} belongs to another user", booking_id=booking.id)
        if not booking.can_transition(BookingStatus.CANCELLED):
            raise InvalidTransition(
                f"booking {booking.id} is {booking.status.value} and cannot be cancelled",
                booking_id=booking.id,
                status=booking.status.value,
            )
        paid = booking.status is BookingStatus.CONFIRMED
        r = refund(booking, now, service.domain)
        actual = r.amount if paid else Decimal("0.00")
        if not args.dry_run:
            repo.cancel(booking, now, actual)
        return CancelOut(
            booking_id=booking.id,
            dry_run=args.dry_run,
            paid=paid,
            refund_tier=r.tier,
            refund_label=r.label,
            refund_ratio=r.ratio,
            policy_refund_amount=r.amount,
            refund_amount=actual,
            hours_before_start=round(float(r.hours_before), 2),
            booking=BookingOut.from_model(booking),
        )
