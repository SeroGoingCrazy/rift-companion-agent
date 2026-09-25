"""Price quotes: total = base hourly price x service multiplier x hours (Decimal, 2 dp)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from rift_domain.config import DomainConfig
from rift_domain.enums import ServiceType
from rift_domain.slots import Quote

CENT = Decimal("0.01")


class PricingError(ValueError):
    """Invalid price or duration."""


def to_decimal(value: Decimal | int | float | str) -> Decimal:
    """Exact Decimal from user/config numbers (floats go through ``str`` to avoid 0.1 noise)."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def to_money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def quote(
    hourly_price: Decimal | int | float | str,
    service_type: ServiceType,
    hours: Decimal | int | float,
    domain: DomainConfig,
) -> Quote:
    price = to_decimal(hourly_price)
    if price <= 0:
        raise PricingError(f"hourly price must be positive, got {price}")
    duration = to_decimal(hours)
    if not domain.duration.is_valid(duration):
        d = domain.duration
        raise PricingError(
            f"duration must be {d.min_hours}-{d.max_hours}h in {d.step_hours}h steps, "
            f"got {duration}"
        )
    multiplier = domain.multiplier(service_type)
    return Quote(
        unit_price=to_money(price),
        service_type=service_type,
        multiplier=multiplier,
        hours=duration,
        effective_hourly=to_money(price * multiplier),
        total=to_money(price * multiplier * duration),
    )
