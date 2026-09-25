"""``quote``: price the selected companion (booking-mcp ``quote_price``) and ask to confirm."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.nodes.decide import clear_selection
from rift_agent.graph.state import AgentState, booking_state, dump_booking
from rift_agent.graph.tracing import traced_node
from rift_agent.mcp_clients import McpUnavailable, ToolError
from rift_domain.enums import PendingAction, ServiceType
from rift_domain.slots import BookingState, Quote


def confirm_facts(state: AgentState, booking: BookingState) -> dict[str, Any]:
    """What the confirmation card shows; rebuilt from state so a re-ask can reuse it."""
    card: dict[str, Any] = next(
        (
            c
            for c in state.get("candidate_cards") or []
            if c.get("companion_id") == booking.selected_companion_id
        ),
        {},
    )
    q = booking.quote
    service = q.service_type if q else booking.service_type_effective
    return {
        "kind": "booking",
        "companion_id": booking.selected_companion_id,
        "companion": card.get("name", booking.companion_name or ""),
        "rank": card.get("rank"),
        # .value: state must stay JSON-native (no enum objects in the checkpoint).
        "mode": booking.game_mode.value if booking.game_mode else None,
        "service": service.value if service else None,
        "start_time": booking.start_time.isoformat() if booking.start_time else None,
        "hours": str(q.hours) if q else str(booking.duration_hours),
        "unit_price": str(q.unit_price) if q else None,
        "multiplier": format(q.multiplier.normalize(), "f") if q else None,
        "total": str(q.total) if q else None,
        "time_adjusted": "time_window" in booking.relaxations,
    }


@traced_node("quote")
async def quote(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    booking = booking_state(state)
    assert booking.selected_companion_id is not None and booking.duration_hours is not None
    service = booking.service_type_effective or booking.service_type
    assert service is not None
    try:
        out = await deps.require_booking().quote_price(
            booking.selected_companion_id, service, booking.duration_hours
        )
    except McpUnavailable:
        return {"action": "unavailable", "reply_type": "service_unavailable"}
    except ToolError as exc:
        booking = clear_selection(booking)
        return {
            "action": "present",
            "booking": dump_booking(booking),
            "pending_action": None,
            "reply_type": "candidates",
            "facts": {"cards": state.get("candidate_cards") or [], "quote_error": exc.message},
        }

    q = Quote(
        unit_price=Decimal(str(out["unit_price"])),
        service_type=ServiceType(out["service_type"]),
        multiplier=Decimal(str(out["multiplier"])),
        hours=Decimal(str(out["hours"])),
        effective_hourly=Decimal(str(out["effective_hourly"])),
        total=Decimal(str(out["total"])),
    )
    booking = booking.model_copy(
        update={"quote": q, "pending_action": PendingAction.AWAIT_CONFIRM_BOOKING}
    )
    facts = confirm_facts(state, booking)
    return {
        "action": "confirm",
        "booking": dump_booking(booking),
        "pending_action": PendingAction.AWAIT_CONFIRM_BOOKING.value,
        "reply_type": "confirm_booking",
        "facts": {**(state.get("facts") or {}), **facts},
        "ui": {"confirm": facts},
    }
