"""``quote_price``: total = companion base price x service multiplier x hours."""

from __future__ import annotations

from booking_mcp.db import read_session
from booking_mcp.db.repo import CompanionRepo
from booking_mcp.errors import InvalidArgument
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import QuoteOut, QuotePriceInput
from rift_domain.pricing import PricingError, quote

NAME = "quote_price"
DESCRIPTION = (
    "Price a booking: companion hourly price x service multiplier (casual/climb/coaching) "
    "x hours. Pure calculation, nothing is reserved."
)


def quote_price(service: BookingService, args: QuotePriceInput) -> QuoteOut:
    with read_session(service.factory) as session:
        companion = CompanionRepo(session, service.domain).require(args.companion_id)
        if args.service_type not in companion.service_types:
            raise InvalidArgument(
                f"{companion.name} does not offer {args.service_type.value}",
                companion_id=companion.id,
                offered=[s.value for s in companion.service_types],
            )
        try:
            q = quote(
                companion.hourly_price, args.service_type, args.duration_hours, service.domain
            )
        except PricingError as exc:
            raise InvalidArgument(str(exc)) from exc
        return QuoteOut(
            companion_id=companion.id,
            companion_name=companion.name,
            **q.model_dump(),
        )
