"""``list_my_bookings``: a user's bookings by start time, optionally one status only."""

from __future__ import annotations

from booking_mcp.db import read_session
from booking_mcp.db.repo import BookingRepo, UserRepo
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import BookingListOut, BookingOut, ListMyBookingsInput

NAME = "list_my_bookings"
DESCRIPTION = (
    "List a user's bookings ordered by start time; pass status "
    "(pending_payment / confirmed / completed / cancelled) to filter."
)


def list_my_bookings(service: BookingService, args: ListMyBookingsInput) -> BookingListOut:
    with read_session(service.factory) as session:
        UserRepo(session).require(args.user_id)
        statuses = [args.status] if args.status is not None else None
        bookings = BookingRepo(session).list_for_user(args.user_id, statuses)
        return BookingListOut(bookings=[BookingOut.from_model(b) for b in bookings])
