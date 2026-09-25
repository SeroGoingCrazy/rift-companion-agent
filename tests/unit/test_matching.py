from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_domain.matching import (
    CompanionFilter,
    CompanionProfile,
    FilterError,
    build_filters,
    matches,
    rank_candidates,
)
from rift_domain.slots import ANY, BookingState

WHEN = datetime(2026, 10, 2, 20)
ALL_MODES = tuple(GameMode)
ALL_SERVICES = tuple(ServiceType)


def companion(id: int, **kw: Any) -> CompanionProfile:
    defaults: dict[str, Any] = {
        "name": f"c{id}",
        "gender": Gender.FEMALE,
        "rank": Rank.DIAMOND,
        "roles": (Role.MID,),
        "modes": ALL_MODES,
        "service_types": ALL_SERVICES,
        "hourly_price": Decimal("60"),
        "voice": True,
        "rating": 4.5,
    }
    return CompanionProfile(id=id, **(defaults | kw))


def solo_duo(**kw: Any) -> BookingState:
    base: dict[str, Any] = {
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "start_time": WHEN,
        "duration_hours": 2,
        "rank_requirement": Rank.DIAMOND,
    }
    return BookingState(**(base | kw))


# --- build_filters ---


def test_solo_duo_rank_band(domain: DomainConfig) -> None:
    f = build_filters(solo_duo(), domain)
    assert (f.rank_min, f.rank_max) == (Rank.DIAMOND, Rank.GRANDMASTER)
    assert f.service_type is ServiceType.CLIMB
    assert f.price_multiplier == Decimal("1.2")
    assert f.duration_hours == Decimal("2")
    assert f.end_time == WHEN + timedelta(hours=2)


def test_rank_band_clamped_at_top(domain: DomainConfig) -> None:
    f = build_filters(solo_duo(rank_requirement=Rank.GRANDMASTER), domain)
    assert (f.rank_min, f.rank_max) == (Rank.GRANDMASTER, Rank.CHALLENGER)


def test_flex_has_floor_but_no_ceiling(domain: DomainConfig) -> None:
    f = build_filters(solo_duo(game_mode=GameMode.RANKED_FLEX), domain)
    assert (f.rank_min, f.rank_max) == (Rank.DIAMOND, None)


def test_any_fields_do_not_filter(domain: DomainConfig) -> None:
    state = solo_duo(
        rank_requirement=ANY,
        role_preference=ANY,
        companion_gender=ANY,
        voice_required=ANY,
        budget_per_hour=ANY,
        companion_name=ANY,
    )
    f = build_filters(state, domain)
    assert f.rank_min is None and f.rank_max is None
    assert f.roles is None and f.gender is None and f.voice_required is None
    assert f.max_price_per_hour is None and f.companion_name is None


def test_preferences_become_constraints(domain: DomainConfig) -> None:
    state = solo_duo(
        role_preference=(Role.JUNGLE, Role.SUPPORT),
        companion_gender=Gender.FEMALE,
        voice_required=True,
        budget_per_hour=90,
        companion_name="阿狸酱",
        service_type=ServiceType.COACHING,
    )
    f = build_filters(state, domain)
    assert f.roles == (Role.JUNGLE, Role.SUPPORT)
    assert f.gender is Gender.FEMALE
    assert f.voice_required is True
    assert f.max_price_per_hour == Decimal("90")
    assert f.service_type is ServiceType.COACHING
    assert f.max_base_price == Decimal("60")  # 90 / 1.5
    assert f.companion_name == "阿狸酱"


def test_voice_false_is_not_a_constraint(domain: DomainConfig) -> None:
    assert build_filters(solo_duo(voice_required=False), domain).voice_required is None


def test_rank_ignored_for_non_ranked_modes(domain: DomainConfig) -> None:
    f = build_filters(solo_duo(game_mode=GameMode.NORMAL_DRAFT), domain)
    assert f.rank_min is None and f.service_type is ServiceType.CASUAL


def test_missing_required_fields(domain: DomainConfig) -> None:
    with pytest.raises(FilterError):
        build_filters(BookingState(game_mode=GameMode.ARAM), domain)


# --- matches ---


@pytest.mark.parametrize(
    ("rank", "ok"),
    [
        (Rank.EMERALD, False),
        (Rank.DIAMOND, True),
        (Rank.MASTER, True),
        (Rank.GRANDMASTER, True),
        (Rank.CHALLENGER, False),
    ],
)
def test_solo_duo_rank_interval(domain: DomainConfig, rank: Rank, ok: bool) -> None:
    f = build_filters(solo_duo(), domain)
    assert matches(companion(1, rank=rank), f, domain) is ok


def test_flex_allows_any_higher_rank(domain: DomainConfig) -> None:
    f = build_filters(solo_duo(game_mode=GameMode.RANKED_FLEX), domain)
    assert matches(companion(1, rank=Rank.CHALLENGER), f, domain)
    assert not matches(companion(1, rank=Rank.PLATINUM), f, domain)


def _filter(**kw: Any) -> CompanionFilter:
    base: dict[str, Any] = {
        "game_mode": GameMode.ARAM,
        "start_time": WHEN,
        "duration_hours": Decimal(2),
        "service_type": ServiceType.CASUAL,
    }
    return CompanionFilter(**(base | kw))


@pytest.mark.parametrize(
    ("filter_kw", "companion_kw", "ok"),
    [
        ({}, {}, True),
        ({}, {"modes": (GameMode.RANKED_SOLO_DUO,)}, False),
        ({"service_type": ServiceType.COACHING}, {"service_types": (ServiceType.CASUAL,)}, False),
        ({"roles": (Role.JUNGLE, Role.TOP)}, {"roles": (Role.TOP, Role.MID)}, True),
        ({"roles": (Role.JUNGLE,)}, {"roles": (Role.MID,)}, False),
        ({"gender": Gender.MALE}, {}, False),
        ({"voice_required": True}, {"voice": False}, False),
        ({"max_price_per_hour": Decimal(60)}, {"hourly_price": Decimal(60)}, True),
        ({"max_price_per_hour": Decimal(59)}, {"hourly_price": Decimal(60)}, False),
        (
            {"max_price_per_hour": Decimal(72), "price_multiplier": Decimal("1.2")},
            {"hourly_price": Decimal(60)},
            True,
        ),
        (
            {"max_price_per_hour": Decimal(71), "price_multiplier": Decimal("1.2")},
            {"hourly_price": Decimal(60)},
            False,
        ),
        ({"companion_name": "c1"}, {}, True),
        ({"companion_name": "阿狸酱"}, {}, False),
    ],
)
def test_matches_static_constraints(
    domain: DomainConfig, filter_kw: dict[str, Any], companion_kw: dict[str, Any], ok: bool
) -> None:
    assert matches(companion(1, **companion_kw), _filter(**filter_kw), domain) is ok


# --- rank_candidates ---


def test_role_hits_dominate_and_reasons_explain(domain: DomainConfig) -> None:
    state = solo_duo(role_preference=(Role.JUNGLE, Role.SUPPORT))
    pool = [
        companion(1, roles=(Role.MID,), rating=5.0),
        companion(2, roles=(Role.JUNGLE,), rating=4.0),
        companion(3, roles=(Role.JUNGLE, Role.SUPPORT), rating=4.0),
    ]
    ranked = rank_candidates(pool, state, domain)
    assert [s.companion.id for s in ranked] == [3, 2, 1]
    assert ranked[0].reasons[0] == "擅长打野/辅助"
    assert ranked[0].components["role"] == 1.0
    assert ranked[1].components["role"] == 0.5
    assert "评分 4.0" in ranked[0].reasons


def test_style_similarity_counts_when_available(domain: DomainConfig) -> None:
    state = solo_duo(style_preference="温柔会聊天")
    pool = [companion(1, rating=4.8), companion(2, rating=4.2)]
    ranked = rank_candidates(pool, state, domain, style_scores={1: 0.1, 2: 0.95})
    assert [s.companion.id for s in ranked] == [2, 1]
    assert "风格贴合「温柔会聊天」" in ranked[0].reasons
    assert not any("风格" in r for r in ranked[1].reasons)


def test_weights_renormalize_over_applicable_components(domain: DomainConfig) -> None:
    (only,) = rank_candidates([companion(1, rating=5.0)], solo_duo(), domain)
    assert only.score == pytest.approx(1.0)
    assert set(only.components) == {"rating"}


def test_score_matches_configured_weights(domain: DomainConfig) -> None:
    state = solo_duo(role_preference=(Role.MID,), style_preference="暴躁")
    (s,) = rank_candidates([companion(1, rating=4.0)], state, domain, style_scores={1: 0.5})
    w = domain.matching.weights
    expected = (w.role * 1.0 + w.style * 0.5 + w.rating * 0.8) / (w.role + w.style + w.rating)
    assert s.score == pytest.approx(expected)


def test_style_ignored_when_any_or_no_scores(domain: DomainConfig) -> None:
    state = solo_duo(style_preference=ANY)
    (s,) = rank_candidates([companion(1)], state, domain, style_scores={1: 0.9})
    assert "style" not in s.components
    (s,) = rank_candidates([companion(1)], solo_duo(style_preference="温柔"), domain)
    assert "style" not in s.components


def test_ties_are_stable_by_rating_then_id(domain: DomainConfig) -> None:
    pool = [companion(5, rating=4.0), companion(2, rating=4.0), companion(9, rating=4.0)]
    ranked = rank_candidates(pool, solo_duo(), domain)
    assert [s.companion.id for s in ranked] == [2, 5, 9]
    assert rank_candidates(list(reversed(pool)), solo_duo(), domain) == ranked


def test_top_k_and_candidate_conversion(domain: DomainConfig) -> None:
    pool = [companion(i, rating=3.0 + i / 10) for i in range(1, 6)]
    ranked = rank_candidates(pool, solo_duo(voice_required=True), domain, top_k=2)
    assert [s.companion.id for s in ranked] == [5, 4]
    cand = ranked[0].to_candidate()
    assert cand.companion_id == 5 and cand.name == "c5"
    assert "可开麦" in cand.reasons


def test_named_companion_reason(domain: DomainConfig) -> None:
    (s,) = rank_candidates([companion(1, name="阿狸酱")], solo_duo(companion_name="阿狸酱"), domain)
    assert s.reasons[0] == "指定陪玩师"


def test_empty_pool(domain: DomainConfig) -> None:
    assert rank_candidates([], solo_duo(), domain) == []
