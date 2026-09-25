"""Companion matching: hard filters (pure data, turned into SQL by booking-mcp), soft
ranking with explanations, and the relaxation plan for zero-result searches.

Availability (schedules) is a storage concern and is not checked here; ``matches``
covers every static attribute so the in-memory and SQL paths agree.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal

from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_domain.slots import ANY, BookingState, Candidate


@dataclass(frozen=True)
class CompanionProfile:
    id: int
    name: str
    gender: Gender
    rank: Rank
    roles: tuple[Role, ...]
    modes: tuple[GameMode, ...]
    service_types: tuple[ServiceType, ...]
    hourly_price: Decimal
    voice: bool
    rating: float
    tags: tuple[str, ...] = ()
    bio: str = ""


@dataclass(frozen=True)
class CompanionFilter:
    """Hard constraints for a companion search. ``None`` means "no constraint"."""

    game_mode: GameMode
    start_time: datetime
    duration_hours: Decimal
    service_type: ServiceType
    rank_min: Rank | None = None
    rank_max: Rank | None = None
    #: Companion must play at least one of these roles.
    roles: tuple[Role, ...] | None = None
    gender: Gender | None = None
    #: Only ``True`` constrains; "mic optional" is no constraint.
    voice_required: bool | None = None
    #: User's budget for the *effective* hourly price (base price x service multiplier).
    max_price_per_hour: Decimal | None = None
    price_multiplier: Decimal = Decimal(1)
    companion_name: str | None = None
    #: Accept start times within +/- this many hours (set by the time-window relaxation).
    time_shift_hours: Decimal = Decimal(0)

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(hours=float(self.duration_hours))

    @property
    def max_base_price(self) -> Decimal | None:
        """Budget translated to the companion's base hourly price (what SQL compares)."""
        if self.max_price_per_hour is None:
            return None
        return self.max_price_per_hour / self.price_multiplier


class FilterError(ValueError):
    """The state lacks what a search needs (mode, time, duration)."""


def build_filters(state: BookingState, domain: DomainConfig) -> CompanionFilter:
    if state.game_mode is None or state.start_time is None or state.duration_hours is None:
        raise FilterError("game_mode, start_time and duration_hours are required to search")
    mode = state.game_mode
    service_type = state.service_type or domain.default_service_type(mode)

    rank_min = rank_max = None
    if isinstance(state.rank_requirement, Rank) and domain.is_ranked(mode):
        rank_min = state.rank_requirement
        gap = domain.max_tier_gap_for(mode)
        if gap is not None:
            rank_max = domain.rank_offset(rank_min, gap)

    budget = state.budget_per_hour
    name = state.companion_name
    return CompanionFilter(
        game_mode=mode,
        start_time=state.start_time,
        duration_hours=Decimal(str(state.duration_hours)),
        service_type=service_type,
        rank_min=rank_min,
        rank_max=rank_max,
        roles=state.role_preference if isinstance(state.role_preference, tuple) else None,
        gender=state.companion_gender if isinstance(state.companion_gender, Gender) else None,
        voice_required=True if state.voice_required is True else None,
        max_price_per_hour=Decimal(str(budget)) if isinstance(budget, int | float) else None,
        price_multiplier=domain.multiplier(service_type),
        companion_name=name if isinstance(name, str) and name != ANY else None,
    )


def matches(companion: CompanionProfile, f: CompanionFilter, domain: DomainConfig) -> bool:
    """Whether ``companion`` passes every static hard constraint of ``f``."""
    if f.game_mode not in companion.modes or f.service_type not in companion.service_types:
        return False
    idx = domain.rank_index(companion.rank)
    if f.rank_min is not None and idx < domain.rank_index(f.rank_min):
        return False
    if f.rank_max is not None and idx > domain.rank_index(f.rank_max):
        return False
    if f.roles is not None and not set(f.roles) & set(companion.roles):
        return False
    if f.gender is not None and companion.gender != f.gender:
        return False
    if f.voice_required and not companion.voice:
        return False
    budget = f.max_base_price
    if budget is not None and companion.hourly_price > budget:
        return False
    return f.companion_name is None or companion.name == f.companion_name


# --- soft ranking ------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredCandidate:
    companion: CompanionProfile
    score: float
    components: Mapping[str, float] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()

    def to_candidate(self) -> Candidate:
        return Candidate(
            companion_id=self.companion.id,
            name=self.companion.name,
            score=round(self.score, 4),
            reasons=self.reasons,
        )


#: Style similarity at or above this is worth mentioning to the user.
STYLE_REASON_THRESHOLD = 0.5


def rank_candidates(
    companions: Iterable[CompanionProfile],
    state: BookingState,
    domain: DomainConfig,
    style_scores: Mapping[int, float] | None = None,
    top_k: int | None = None,
) -> list[ScoredCandidate]:
    """Weighted score of role match, style similarity and rating, with reasons.

    Only applicable components count and their weights are renormalized: without a role
    preference, role weight is dropped instead of scoring everyone 0. Ties break by rating
    then id, so the order is stable.
    """
    weights = domain.matching.weights
    prefs = state.role_preference if isinstance(state.role_preference, tuple) else None
    style = state.style_preference
    wants_style = isinstance(style, str) and style != ANY and style_scores is not None

    scored: list[ScoredCandidate] = []
    for c in companions:
        components: dict[str, float] = {}
        reasons: list[str] = []
        if isinstance(state.companion_name, str) and state.companion_name == c.name:
            reasons.append("指定陪玩师")
        if prefs:
            hit = [r for r in prefs if r in c.roles]
            components["role"] = len(hit) / len(prefs)
            if hit:
                reasons.append("擅长" + "/".join(domain.roles[r].label for r in hit))
        if wants_style and style_scores is not None and c.id in style_scores:
            similarity = min(max(style_scores[c.id], 0.0), 1.0)
            components["style"] = similarity
            if similarity >= STYLE_REASON_THRESHOLD:
                reasons.append(f"风格贴合「{style}」")
        components["rating"] = min(c.rating / domain.matching.rating_max, 1.0)
        reasons.append(f"评分 {c.rating:.1f}")
        if state.voice_required is True and c.voice:
            reasons.append("可开麦")

        total_weight = sum(getattr(weights, k) for k in components)
        score = (
            sum(getattr(weights, k) * v for k, v in components.items()) / total_weight
            if total_weight
            else 0.0
        )
        scored.append(ScoredCandidate(c, score, components, tuple(reasons)))

    scored.sort(key=lambda s: (-s.score, -s.companion.rating, s.companion.id))
    return scored[:top_k] if top_k is not None else scored
