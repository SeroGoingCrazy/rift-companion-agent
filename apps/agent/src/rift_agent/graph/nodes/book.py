"""``book``: create the order (booking-mcp re-checks the slot inside its transaction).

On ``SLOT_CONFLICT`` the selection is dropped and the graph searches again, so the
user immediately gets alternatives.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.state import AgentState, Phase, booking_state, dump_booking
from rift_agent.graph.tracing import traced_node
from rift_agent.mcp_clients import McpUnavailable, SlotConflictError, ToolError
from rift_domain.slots import BookingState


@traced_node("book")
async def book(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    booking = booking_state(state)
    assert booking.selected_companion_id is not None and booking.start_time is not None
    assert booking.game_mode is not None and booking.duration_hours is not None
    try:
        order = await deps.require_booking().create_booking(
            user_id=int(state.get("user_id") or 0),
            companion_id=booking.selected_companion_id,
            start_time=booking.start_time,
            duration_hours=booking.duration_hours,
            game_mode=booking.game_mode,
            service_type=booking.service_type_effective,
        )
    except SlotConflictError:
        booking = booking.model_copy(
            update={
                "selected_companion_id": None,
                "quote": None,
                "pending_action": None,
                "candidates": (),
                "companion_name": None,
            }
        )
        return {
            "action": "find",
            "booking": dump_booking(booking),
            "pending_action": None,
            "facts": {"conflict": True},
        }
    except McpUnavailable:
        return {"action": "unavailable", "reply_type": "service_unavailable"}
    except ToolError as exc:
        return {
            "action": "failed",
            "reply_type": "booking_failed",
            "facts": {"message": exc.message},
        }

    return {
        "action": "booked",
        "booking": dump_booking(BookingState()),
        "pending_action": None,
        "candidate_cards": [],
        "phase": Phase.IDLE.value,
        "reply_type": "booked",
        "facts": {"order": order},
        "ui": {"booked": order},
    }


def route_booked(state: AgentState) -> str:
    if state.get("error"):
        return "render"
    return "find_companions" if state.get("action") == "find" else "render"
