from datetime import datetime

import pytest

from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Rank, Role, ServiceType, SlotField
from rift_domain.rules import apply_rules, compute_rules
from rift_domain.slots import ANY, BookingState

WHEN = datetime(2026, 10, 2, 20)
F = SlotField


def test_empty_state_misses_base_fields_in_order(domain: DomainConfig) -> None:
    result = compute_rules(BookingState(), domain)
    assert result.missing_fields == (F.GAME_MODE, F.START_TIME, F.DURATION_HOURS)
    assert result.service_type_effective is None
    assert result.cleared_fields == ()


def test_ranked_without_rank_misses_rank_requirement(domain: DomainConfig) -> None:
    state = BookingState(game_mode=GameMode.RANKED_SOLO_DUO, start_time=WHEN, duration_hours=2)
    assert compute_rules(state, domain).missing_fields == (F.RANK_REQUIREMENT,)


def test_rank_order_follows_base_fields(domain: DomainConfig) -> None:
    state = BookingState(game_mode=GameMode.RANKED_FLEX)
    assert compute_rules(state, domain).missing_fields == (
        F.START_TIME,
        F.DURATION_HOURS,
        F.RANK_REQUIREMENT,
    )


def test_any_satisfies_required(domain: DomainConfig) -> None:
    state = BookingState(
        game_mode=GameMode.RANKED_SOLO_DUO, start_time=WHEN, duration_hours=2, rank_requirement=ANY
    )
    assert compute_rules(state, domain).missing_fields == ()


def test_complete_casual_booking(domain: DomainConfig) -> None:
    state = BookingState(game_mode=GameMode.NORMAL_DRAFT, start_time=WHEN, duration_hours=1)
    result = compute_rules(state, domain)
    assert result.missing_fields == ()
    assert result.service_type_effective is ServiceType.CASUAL


@pytest.mark.parametrize("mode", [GameMode.ARAM, GameMode.ARAM_MAYHEM, GameMode.ARENA])
def test_non_ranked_modes_clear_rank_and_role(domain: DomainConfig, mode: GameMode) -> None:
    state = BookingState(
        game_mode=mode,
        start_time=WHEN,
        duration_hours=2,
        rank_requirement=Rank.DIAMOND,
        role_preference=(Role.MID,),
    )
    new_state, result = apply_rules(state, domain)
    assert result.cleared_fields == (F.RANK_REQUIREMENT, F.ROLE_PREFERENCE)
    assert new_state.rank_requirement is None and new_state.role_preference is None
    assert result.missing_fields == ()


def test_only_set_fields_are_reported_as_cleared(domain: DomainConfig) -> None:
    state = BookingState(game_mode=GameMode.ARAM, rank_requirement=ANY)
    assert compute_rules(state, domain).cleared_fields == (F.RANK_REQUIREMENT,)


def test_normal_draft_keeps_role_preference(domain: DomainConfig) -> None:
    state = BookingState(game_mode=GameMode.NORMAL_DRAFT, role_preference=(Role.SUPPORT,))
    new_state, result = apply_rules(state, domain)
    assert result.cleared_fields == ()
    assert new_state.role_preference == (Role.SUPPORT,)


@pytest.mark.parametrize(
    ("mode", "explicit", "expected"),
    [
        (GameMode.RANKED_SOLO_DUO, None, ServiceType.CLIMB),
        (GameMode.RANKED_FLEX, None, ServiceType.CLIMB),
        (GameMode.ARAM, None, ServiceType.CASUAL),
        (GameMode.ARENA, None, ServiceType.CASUAL),
        (GameMode.RANKED_SOLO_DUO, ServiceType.COACHING, ServiceType.COACHING),
        (GameMode.ARAM, ServiceType.COACHING, ServiceType.COACHING),
        (GameMode.RANKED_FLEX, ServiceType.CASUAL, ServiceType.CASUAL),
        (None, ServiceType.COACHING, ServiceType.COACHING),
        (None, None, None),
    ],
)
def test_service_type_inference(
    domain: DomainConfig,
    mode: GameMode | None,
    explicit: ServiceType | None,
    expected: ServiceType | None,
) -> None:
    state = BookingState(game_mode=mode, service_type=explicit)
    assert compute_rules(state, domain).service_type_effective is expected


@pytest.mark.parametrize("hours", [0.5, 8.5, 12])
def test_out_of_range_duration_is_invalid_and_missing(domain: DomainConfig, hours: float) -> None:
    state = BookingState(game_mode=GameMode.ARAM, start_time=WHEN, duration_hours=hours)
    new_state, result = apply_rules(state, domain)
    assert F.DURATION_HOURS in result.invalid_fields
    assert "1–8 小时" in result.invalid_fields[F.DURATION_HOURS]
    assert result.missing_fields == (F.DURATION_HOURS,)
    assert new_state.duration_hours is None


def test_apply_rules_writes_derived_fields(domain: DomainConfig) -> None:
    state = BookingState(game_mode=GameMode.RANKED_SOLO_DUO, start_time=WHEN)
    new_state, result = apply_rules(state, domain)
    assert new_state.missing_fields == (F.DURATION_HOURS, F.RANK_REQUIREMENT)
    assert new_state.service_type_effective is ServiceType.CLIMB
    assert new_state.game_mode is GameMode.RANKED_SOLO_DUO
    # idempotent
    again, second = apply_rules(new_state, domain)
    assert again == new_state and second == result
