from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from rift_domain.config import DomainConfig, parse_domain_config
from rift_domain.enums import ServiceType
from rift_domain.pricing import PricingError, quote, to_money
from rift_domain.refund import refund

# --- pricing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("price", "service", "hours", "effective", "total"),
    [
        ("60", ServiceType.CASUAL, 2, "60.00", "120.00"),
        ("60", ServiceType.CLIMB, 2, "72.00", "144.00"),
        ("60", ServiceType.COACHING, 1.5, "90.00", "135.00"),
        ("45.5", ServiceType.CLIMB, 2.5, "54.60", "136.50"),
        ("33.33", ServiceType.CLIMB, 3, "40.00", "119.99"),  # rounded once, on the total
        ("0.05", ServiceType.CLIMB, 1, "0.06", "0.06"),  # half-up
        (80, ServiceType.CASUAL, 8, "80.00", "640.00"),
    ],
)
def test_quote(
    domain: DomainConfig,
    price: str | int,
    service: ServiceType,
    hours: float,
    effective: str,
    total: str,
) -> None:
    q = quote(price, service, hours, domain)
    assert q.effective_hourly == Decimal(effective)
    assert q.total == Decimal(total)
    assert q.multiplier == domain.multiplier(service)
    assert q.service_type is service
    assert q.total.as_tuple().exponent == -2


def test_quote_float_price_has_no_binary_noise(domain: DomainConfig) -> None:
    q = quote(0.1 + 0.2, ServiceType.CASUAL, 1, domain)  # 0.30000000000000004
    assert q.unit_price == Decimal("0.30") and q.total == Decimal("0.30")


@pytest.mark.parametrize("hours", [0.5, 0.3, 8.5, 0, -1])
def test_quote_rejects_invalid_duration(domain: DomainConfig, hours: float) -> None:
    with pytest.raises(PricingError, match="duration"):
        quote(60, ServiceType.CASUAL, hours, domain)


@pytest.mark.parametrize("price", [0, -10, "0.00"])
def test_quote_rejects_non_positive_price(domain: DomainConfig, price: int | str) -> None:
    with pytest.raises(PricingError, match="positive"):
        quote(price, ServiceType.CASUAL, 1, domain)


def test_to_money_rounds_half_up() -> None:
    assert to_money(Decimal("1.005")) == Decimal("1.01")
    assert to_money(Decimal("1.004")) == Decimal("1.00")


# --- refund -------------------------------------------------------------------------------

START = datetime(2026, 10, 2, 20, 0)


@dataclass(frozen=True)
class Booking:
    start_time: datetime
    total: Decimal


BOOKING = Booking(start_time=START, total=Decimal("144.00"))


@pytest.mark.parametrize(
    ("before", "tier", "amount"),
    [
        (timedelta(days=3), "full", "144.00"),
        (timedelta(hours=24), "full", "144.00"),  # boundary: exactly 24h -> full
        (timedelta(hours=24) - timedelta(seconds=1), "half", "72.00"),
        (timedelta(hours=6), "half", "72.00"),
        (timedelta(hours=2), "half", "72.00"),  # boundary: exactly 2h -> half
        (timedelta(hours=2) - timedelta(seconds=1), "none", "0.00"),
        (timedelta(minutes=10), "none", "0.00"),
        (timedelta(0), "none", "0.00"),
        (-timedelta(hours=1), "none", "0.00"),  # already started
    ],
)
def test_refund_tiers_and_boundaries(
    domain: DomainConfig, before: timedelta, tier: str, amount: str
) -> None:
    result = refund(BOOKING, START - before, domain)
    assert result.tier == tier
    assert result.amount == Decimal(amount)
    assert result.hours_before == Decimal(str(before.total_seconds())) / 3600


def test_refund_labels_and_ratio(domain: DomainConfig) -> None:
    result = refund(BOOKING, START - timedelta(hours=3), domain)
    assert result.label == "退一半"
    assert result.ratio == Decimal("0.5")


def test_refund_amount_rounds_to_cents(domain: DomainConfig) -> None:
    odd = Booking(start_time=START, total=Decimal("99.99"))
    assert refund(odd, START - timedelta(hours=5), domain).amount == Decimal("50.00")


def test_refund_tiers_follow_config(domain: DomainConfig) -> None:
    raw = domain.model_dump(mode="json")
    raw["refund"]["tiers"] = [
        {"name": "full", "min_hours_before": 48, "ratio": 1, "label": "全额"},
        {"name": "most", "min_hours_before": 12, "ratio": 0.8, "label": "八成"},
        {"name": "none", "min_hours_before": 0, "ratio": 0, "label": "不退"},
    ]
    custom = parse_domain_config(raw)
    assert refund(BOOKING, START - timedelta(hours=24), custom).tier == "most"
    assert refund(BOOKING, START - timedelta(hours=24), custom).amount == Decimal("115.20")
    assert refund(BOOKING, START - timedelta(hours=48), custom).tier == "full"
