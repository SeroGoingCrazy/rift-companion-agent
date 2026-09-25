"""``browse_companions``: the web companion list — filters plus upcoming free time.

With a game mode the filter is exactly ``find_companions``'s hard filter (same
``apply_rules`` + ``build_filters`` + SQL), minus availability; without one only the
role and gender filters apply. Each card lists the free ranges of the next ``days`` days
that are long enough for the shortest bookable session.
"""

from __future__ import annotations

from datetime import timedelta

from booking_mcp.db import Companion, read_session
from booking_mcp.db.repo import CompanionRepo, ScheduleRepo
from booking_mcp.service import BookingService
from booking_mcp.tools.schemas import (
    BrowseCompanionsInput,
    BrowseCompanionsOutput,
    CompanionCard,
    FreeSlot,
)
from rift_domain.matching import build_filters
from rift_domain.pricing import to_money
from rift_domain.rules import apply_rules
from rift_domain.slots import BookingState

NAME = "browse_companions"
DESCRIPTION = (
    "List companions for the web companion page: optional game mode / rank / role / "
    "gender filters (same hard filter as find_companions) and free time in the next days."
)


def _filtered(
    service: BookingService, repo: CompanionRepo, args: BrowseCompanionsInput
) -> list[Companion]:
    if args.game_mode is None:
        rows = repo.list_all()
        if args.role is not None:
            rows = [c for c in rows if args.role in c.roles]
        if args.companion_gender is not None:
            rows = [c for c in rows if c.gender == args.companion_gender]
        return rows
    now = service.now()
    state = BookingState(
        game_mode=args.game_mode,
        # Time and duration only matter for availability, which the static filter ignores.
        start_time=now,
        duration_hours=float(service.domain.duration.min_hours),
        rank_requirement=args.rank_requirement,
        role_preference=(args.role,) if args.role else None,
        companion_gender=args.companion_gender,
    )
    state, _ = apply_rules(state, service.domain)
    return list(repo.session.scalars(repo.static_query(build_filters(state, service.domain))))


def browse_companions(
    service: BookingService, args: BrowseCompanionsInput
) -> BrowseCompanionsOutput:
    domain = service.domain
    now = service.now()
    horizon = now + timedelta(days=args.days)
    min_len = timedelta(hours=float(domain.duration.min_hours))
    cards: list[CompanionCard] = []
    with read_session(service.factory) as session:
        repo = CompanionRepo(session, domain)
        schedules = ScheduleRepo(session)
        for c in _filtered(service, repo, args):
            free = [
                FreeSlot(start=start, end=end)
                for start, end in schedules.free_ranges(c.id, now, horizon)
                if end - start >= min_len
            ]
            cards.append(
                CompanionCard(
                    companion_id=c.id,
                    name=c.name,
                    gender=c.gender,
                    rank=c.rank,
                    roles=list(c.roles),
                    modes=list(c.modes),
                    service_types=list(c.service_types),
                    hourly_price=to_money(c.hourly_price),
                    prices={
                        st.value: to_money(c.hourly_price * domain.multiplier(st))
                        for st in c.service_types
                    },
                    voice=c.voice,
                    rating=c.rating,
                    level=domain.policies.level_for(c.rating).name,
                    tags=list(c.tags),
                    bio=c.bio,
                    free_slots=free,
                )
            )
    cards.sort(key=lambda card: (-card.rating, card.companion_id))
    return BrowseCompanionsOutput(companions=cards)
