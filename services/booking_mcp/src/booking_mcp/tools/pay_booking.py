"""``pay_booking``: simulated payment, ``pending_payment -> confirmed`` (web order page)."""

from __future__ import annotations

from booking_mcp.db import write_session
from booking_mcp.db.repo import BookingRepo
from booking_mcp.errors import Forbidden
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import BookingOut, PayBookingInput

NAME = "pay_booking"
DESCRIPTION = (
    "Simulated payment for one of the user's bookings: pending_payment -> confirmed. "
    "Other users' bookings are FORBIDDEN; paid or cancelled ones are INVALID_ARGUMENT."
)


def pay_booking(service: BookingService, args: PayBookingInput) -> BookingOut:
    with write_session(service.factory) as session:
        repo = BookingRepo(session)
        booking = repo.require(args.booking_id)
        if booking.user_id != args.user_id:
            raise Forbidden(f"booking {booking.id} belongs to another user", booking_id=booking.id)
        repo.pay(booking, service.now())
        return BookingOut.from_model(booking)
