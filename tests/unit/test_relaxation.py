import dataclasses
from datetime import datetime
from decimal import Decimal
from typing import Any

import pytest

from rift_domain.config import DomainConfig, parse_domain_config
from rift_domain.enums import GameMode, Gender, Rank, RelaxStepName, Role, ServiceType
from rift_domain.matching import CompanionFilter, describe_relaxation, relaxation_plan

T, G, R = RelaxStepName.TIME_WINDOW, RelaxStepName.COMPANION_GENDER, RelaxStepName.ROLE_PREFERENCE


def make_filter(**kw: Any) -> CompanionFilter:
    base: dict[str, Any] = {
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "start_time": datetime(2026, 10, 2, 20),
        "duration_hours": Decimal(2),
        "service_type": ServiceType.CLIMB,
        "rank_min": Rank.DIAMOND,
        "rank_max": Rank.GRANDMASTER,
        "roles": (Role.JUNGLE,),
        "gender": Gender.FEMALE,
        "max_price_per_hour": Decimal(80),
        "price_multiplier": Decimal("1.2"),
    }
    return CompanionFilter(**(base | kw))


def test_full_plan_in_configured_order(domain: DomainConfig) -> None:
    plan = relaxation_plan(make_filter(), domain)
    assert [s.name for s in plan] == [T, G, R]
    assert [s.relaxed for s in plan] == [(T,), (T, G), (T, G, R)]


def test_plan_is_cumulative(domain: DomainConfig) -> None:
    first, second, third = relaxation_plan(make_filter(), domain)
    assert first.filter.time_shift_hours == Decimal(1)
    assert first.filter.gender is Gender.FEMALE and first.filter.roles == (Role.JUNGLE,)
    # step 2 relaxes time AND gender
    assert second.filter.time_shift_hours == Decimal(1) and second.filter.gender is None
    assert second.filter.roles == (Role.JUNGLE,)
    assert third.filter.gender is None and third.filter.roles is None


def test_budget_and_rank_never_relaxed(domain: DomainConfig) -> None:
    original = make_filter()
    for step in relaxation_plan(original, domain):
        assert step.filter.max_price_per_hour == original.max_price_per_hour
        assert step.filter.price_multiplier == original.price_multiplier
        assert (step.filter.rank_min, step.filter.rank_max) == (Rank.DIAMOND, Rank.GRANDMASTER)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"gender": None}, [T, R]),
        ({"roles": None}, [T, G]),
        ({"gender": None, "roles": None}, [T]),
        ({"time_shift_hours": Decimal(1)}, [G, R]),  # already widened
    ],
)
def test_unset_constraints_are_skipped(
    domain: DomainConfig, overrides: dict[str, Any], expected: list[RelaxStepName]
) -> None:
    plan = relaxation_plan(make_filter(**overrides), domain)
    assert [s.name for s in plan] == expected
    assert plan[-1].relaxed == tuple(expected)


def test_original_filter_untouched(domain: DomainConfig) -> None:
    original = make_filter()
    snapshot = dataclasses.replace(original)
    relaxation_plan(original, domain)
    assert original == snapshot


def test_order_and_window_come_from_config(domain: DomainConfig) -> None:
    raw = domain.model_dump(mode="json")
    raw["relaxation"]["order"] = ["role_preference", "time_window"]
    raw["relaxation"]["time_window_hours"] = 2
    custom = parse_domain_config(raw)
    plan = relaxation_plan(make_filter(), custom)
    assert [s.name for s in plan] == [R, T]
    assert plan[-1].filter.time_shift_hours == Decimal(2)
    assert plan[-1].filter.gender is Gender.FEMALE  # gender not in this order


def test_describe_relaxation(domain: DomainConfig) -> None:
    assert describe_relaxation(T, domain) == "开始时间放宽到前后 1 小时"
    assert describe_relaxation(G, domain) == "不再限定陪玩师性别"
    assert describe_relaxation(R, domain) == "不再限定位置"
