"""``decide``: pure code picks the next step of a booking turn.

Order of checks (first match wins):

1. waiting for a booking confirmation and nothing changed:
   yes -> ``book``; no -> back to the candidates; anything else -> ask again
2. "不约了" (confirmation=no, no slot changed, nothing pending) -> ``abandon``
3. an unresolved time expression is pending -> ``ask_time``
4. a required slot is missing -> ``ask_missing``
5. a companion is selected but not quoted -> ``quote``
6. the user named one of the shown candidates -> select them, ``quote``
7. no candidates yet (or they were invalidated) -> ``find``
8. otherwise show the candidates again
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.nodes.extract import load_extraction
from rift_agent.graph.state import AgentState, booking_state, dump_booking
from rift_agent.graph.tracing import traced_node
from rift_domain.enums import Confirmation, PendingAction
from rift_domain.slots import BookingState
from rift_domain.timeparse import parse_time_expr

BOOKING_CONFIRM = PendingAction.AWAIT_CONFIRM_BOOKING.value


def _card(state: AgentState, companion_id: int) -> dict[str, Any] | None:
    return next(
        (c for c in state.get("candidate_cards") or [] if c.get("companion_id") == companion_id),
        None,
    )


def select_candidate(state: AgentState, booking: BookingState, companion_id: int) -> BookingState:
    """Select a shown candidate; adopt their bookable start if the time was relaxed."""
    updates: dict[str, Any] = {"selected_companion_id": companion_id, "quote": None}
    card = _card(state, companion_id)
    if card and card.get("start_time"):
        start = datetime.fromisoformat(str(card["start_time"]))
        if start != booking.start_time:
            updates["start_time"] = start
    return booking.model_copy(update=updates)


def clear_selection(booking: BookingState) -> BookingState:
    return booking.model_copy(
        update={
            "selected_companion_id": None,
            "quote": None,
            "pending_action": None,
            "companion_name": None,
        }
    )


@traced_node("decide")
async def decide(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    extraction = load_extraction(state)
    assert extraction is not None
    booking = booking_state(state)
    diff = state.get("diff") or {}
    changed = bool(diff.get("changed"))
    confirmation = extraction.confirmation

    if state.get("pending_action") == BOOKING_CONFIRM and not changed:
        if confirmation is Confirmation.YES:
            return {"action": "book"}
        if confirmation is Confirmation.NO:
            booking = clear_selection(booking)
            return {
                "action": "present",
                "booking": dump_booking(booking),
                "pending_action": None,
                "reply_type": "candidates",
                "facts": {"cards": state.get("candidate_cards") or [], "declined": True},
            }
        return {
            "action": "reconfirm",
            "reply_type": "reconfirm",
            "facts": dict(state.get("facts") or {}),
        }

    if confirmation is Confirmation.NO and not changed and not state.get("pending_action"):
        return {"action": "abandon"}

    if booking.start_time is None and booking.start_time_expr:
        # Resolved this turn or not, an expression we could not turn into a time is
        # asked about directly ("明晚具体几点？") instead of generically.
        reason = diff.get("time_reason") or (
            parse_time_expr(booking.start_time_expr, get_deps(config).now()).reason
            or "unrecognized"
        )
        return {
            "action": "ask_time",
            "reply_type": "ask_time",
            "facts": {"reason": reason, "expr": booking.start_time_expr},
        }

    if booking.missing_fields:
        return {
            "action": "ask_missing",
            "reply_type": "ask_missing",
            "facts": {
                "field": booking.missing_fields[0].value,
                "missing": [f.value for f in booking.missing_fields],
                "cleared": diff.get("cleared", []),
                "invalid": diff.get("invalid", {}),
            },
        }

    if booking.selected_companion_id is not None and booking.quote is None:
        return {"action": "quote"}

    name = booking.companion_name
    if isinstance(name, str):
        match = next((c for c in booking.candidates if c.name == name), None)
        if match is not None:
            booking = select_candidate(state, booking, match.companion_id)
            return {"action": "quote", "booking": dump_booking(booking)}

    if not booking.candidates:
        return {"action": "find"}

    return {
        "action": "present",
        "reply_type": "candidates",
        "facts": {"cards": state.get("candidate_cards") or []},
    }


def route_decision(state: AgentState) -> str:
    if state.get("error"):
        return "render"
    return {
        "book": "book",
        "abandon": "abandon",
        "quote": "quote",
        "find": "find_companions",
    }.get(state.get("action") or "", "render")
