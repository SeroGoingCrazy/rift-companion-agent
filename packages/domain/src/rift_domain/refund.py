"""Refunds by time left before the booking starts (tiers from ``domain.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from rift_domain.config import DomainConfig
from rift_domain.pricing import to_decimal, to_money


class BookingLike(Protocol):
    @property
    def start_time(self) -> datetime: ...

    @property
    def total(self) -> Decimal: ...


@dataclass(frozen=True)
class Refund:
    tier: str
    label: str
    ratio: Decimal
    amount: Decimal
    #: Hours between ``now`` and the start (negative once the booking has started).
    hours_before: Decimal


def refund(booking: BookingLike, now: datetime, domain: DomainConfig) -> Refund:
    """First tier whose ``min_hours_before`` <= hours left wins, so boundaries are inclusive:
    exactly 24h before -> full, exactly 2h -> half, anything later -> none. After the start
    the last tier applies.
    """
    hours_before = Decimal(str((booking.start_time - now).total_seconds())) / Decimal(3600)
    tiers = domain.refund.tiers
    tier = next((t for t in tiers if hours_before >= t.min_hours_before), tiers[-1])
    return Refund(
        tier=tier.name,
        label=tier.label,
        ratio=tier.ratio,
        amount=to_money(to_decimal(booking.total) * tier.ratio),
        hours_before=hours_before,
    )
