"""``find_companions``: search via booking-mcp and remember the shown candidates.

Also re-selects a companion kept across a change (``diff.keep_companion_id``) or named
by the user, so "改成三小时" or "就要阿狸酱" go straight to a new quote when that
companion is still bookable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.nodes.decide import select_candidate
from rift_agent.graph.state import AgentState, booking_state, dump_booking
from rift_agent.graph.tracing import traced_node
from rift_agent.mcp_clients import McpUnavailable
from rift_domain.slots import ANY, BookingState, Candidate


def search_args(booking: BookingState) -> dict[str, Any]:
    """Tool arguments from the merged slots; ANY and unset slots are left out."""

    def concrete(value: Any) -> Any:
        return None if value is None or value == ANY else value

    roles = booking.role_preference
    return {
        "game_mode": booking.game_mode,
        "start_time": booking.start_time,
        "duration_hours": booking.duration_hours,
        "rank_requirement": concrete(booking.rank_requirement),
        "role_preference": list(roles) if isinstance(roles, tuple) else None,
        "service_type": booking.service_type,
        "companion_gender": concrete(booking.companion_gender),
        "voice_required": concrete(booking.voice_required),
        "budget_per_hour": concrete(booking.budget_per_hour),
        "style_preference": concrete(booking.style_preference),
        "companion_name": concrete(booking.companion_name),
    }


@traced_node("find_companions")
async def find_companions(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    booking = booking_state(state)
    try:
        out = await deps.require_booking().find_companions(**search_args(booking))
    except McpUnavailable:
        return {"action": "unavailable", "reply_type": "service_unavailable"}

    cards: list[dict[str, Any]] = out.get("candidates", [])
    candidates = tuple(
        Candidate(
            companion_id=c["companion_id"],
            name=c["name"],
            score=c.get("score", 0.0),
            reasons=tuple(c.get("reasons", ())),
        )
        for c in cards
    )
    booking = booking.model_copy(
        update={
            "candidates": candidates,
            "relaxations": tuple(out.get("relaxations", ())),
            "selected_companion_id": None,
            "quote": None,
        }
    )
    facts: dict[str, Any] = dict(state.get("facts") or {})
    facts.update(
        cards=cards,
        relaxations=out.get("relaxations", []),
        relaxation_notes=out.get("relaxation_notes", []),
    )
    update: dict[str, Any] = {"candidate_cards": cards}

    if not cards:
        update.update(
            booking=dump_booking(booking),
            action="no_candidates",
            reply_type="no_candidates",
            facts={
                "tried": bool(out.get("relaxations_tried")),
                "has_budget": isinstance(booking.budget_per_hour, int | float),
            },
            ui=None,
        )
        return update

    keep = (state.get("diff") or {}).get("keep_companion_id")
    named = booking.companion_name if isinstance(booking.companion_name, str) else None
    pick = next(
        (
            c
            for c in candidates
            if (keep is not None and c.companion_id == keep) or (named and c.name == named)
        ),
        None,
    )
    if pick is not None and (keep is not None or len(candidates) == 1):
        selected = select_candidate({**state, "candidate_cards": cards}, booking, pick.companion_id)
        update.update(booking=dump_booking(selected), action="quote", facts=facts)
        return update

    if keep is not None:
        kept_name = next(
            (
                c.get("name")
                for c in state.get("candidate_cards") or []
                if c.get("companion_id") == keep
            ),
            None,
        )
        facts["companion_unavailable"] = kept_name
    update.update(
        booking=dump_booking(booking),
        action="present",
        reply_type="candidates",
        facts=facts,
        ui={"candidates": cards},
    )
    return update


def route_found(state: AgentState) -> str:
    if state.get("error"):
        return "render"
    return "quote" if state.get("action") == "quote" else "render"
