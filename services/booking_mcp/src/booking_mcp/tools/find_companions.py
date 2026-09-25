"""``find_companions``: hard filter -> availability -> style similarity -> soft ranking,
relaxing ``time -> gender -> role`` step by step when nobody matches.

Budget and rank are never relaxed: a budget below every price returns no candidates.
"""

from __future__ import annotations

from booking_mcp.db import read_session
from booking_mcp.db.repo import AvailableCompanion, CompanionRepo
from booking_mcp.errors import InvalidArgument
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import CandidateOut, FindCompanionsInput, FindCompanionsOutput
from rift_domain.enums import RelaxStepName
from rift_domain.matching import (
    build_filters,
    describe_relaxation,
    rank_candidates,
    relaxation_plan,
)
from rift_domain.pricing import to_money
from rift_domain.rules import apply_rules
from rift_domain.slots import BookingState

NAME = "find_companions"
DESCRIPTION = (
    "Search bookable companions for a game mode and time slot. Applies hard filters "
    "(mode, service type, rank band, roles, gender, mic, budget), keeps only companions free "
    "for the whole slot, ranks by role fit, style similarity and rating, and when nobody "
    "matches relaxes time (±1h), then gender, then role. Budget and rank are never relaxed."
)


def to_state(args: FindCompanionsInput) -> BookingState:
    return BookingState(
        game_mode=args.game_mode,
        start_time=args.start_time,
        duration_hours=args.duration_hours,
        rank_requirement=args.rank_requirement,
        role_preference=tuple(args.role_preference) if args.role_preference else None,
        service_type=args.service_type,
        companion_gender=args.companion_gender,
        voice_required=args.voice_required,
        budget_per_hour=args.budget_per_hour,
        style_preference=args.style_preference,
        companion_name=args.companion_name,
    )


def find_companions(service: BookingService, args: FindCompanionsInput) -> FindCompanionsOutput:
    domain = service.domain
    now = service.now()
    if not domain.duration.is_valid(args.duration_hours):
        d = domain.duration
        raise InvalidArgument(
            f"duration_hours must be {d.min_hours}-{d.max_hours} in {d.step_hours}h steps",
            duration_hours=args.duration_hours,
        )
    if args.start_time < now:
        raise InvalidArgument("start_time is in the past", start_time=args.start_time.isoformat())

    # Drop slots that do not apply to the mode (e.g. rank for ARAM) before filtering.
    state, _ = apply_rules(to_state(args), domain)
    base = build_filters(state, domain)

    relaxed: tuple[RelaxStepName, ...] = ()
    tried: tuple[RelaxStepName, ...] = ()
    with read_session(service.factory) as session:
        repo = CompanionRepo(session, domain)
        hits: list[AvailableCompanion] = repo.search(base, not_before=now)
        if not hits:
            for step in relaxation_plan(base, domain):
                tried = step.relaxed
                hits = repo.search(step.filter, not_before=now)
                if hits:
                    relaxed = step.relaxed
                    break

    profiles = [h.profile for h in hits]
    starts = {h.profile.id: h.start_time for h in hits}
    scores = service.style_scores(args.style_preference, profiles)
    ranked = rank_candidates(
        profiles, state, domain, scores, top_k=args.top_k or domain.matching.top_k
    )
    multiplier = base.price_multiplier
    return FindCompanionsOutput(
        service_type=base.service_type,
        candidates=[
            CandidateOut(
                companion_id=s.companion.id,
                name=s.companion.name,
                gender=s.companion.gender,
                rank=s.companion.rank,
                roles=list(s.companion.roles),
                hourly_price=to_money(s.companion.hourly_price),
                effective_hourly_price=to_money(s.companion.hourly_price * multiplier),
                voice=s.companion.voice,
                rating=s.companion.rating,
                tags=list(s.companion.tags),
                bio=s.companion.bio,
                score=round(s.score, 4),
                reasons=list(s.reasons),
                start_time=starts[s.companion.id],
            )
            for s in ranked
        ],
        relaxations=[r.value for r in relaxed],
        relaxation_notes=[describe_relaxation(r, domain) for r in relaxed],
        relaxations_tried=[] if hits else [r.value for r in tried],
    )
