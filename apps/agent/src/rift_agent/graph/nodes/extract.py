"""``extract_slots``: run the (routed) slot extractor on this turn's input."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.extractors.base import ExtractionContext
from rift_agent.graph.state import AgentState, booking_state
from rift_agent.graph.tracing import traced_node
from rift_common.llm import Message
from rift_domain.enums import PendingAction, TurnIntent
from rift_domain.slots import SlotExtraction, parse_extraction


def dump_extraction(extraction: SlotExtraction) -> dict[str, Any]:
    """JSON form that keeps "absent" and ``null`` apart (``exclude_unset``)."""
    data: dict[str, Any] = extraction.model_dump(mode="json", exclude_unset=True)
    return data


def load_extraction(state: AgentState) -> SlotExtraction | None:
    raw = state.get("extraction")
    return parse_extraction(raw) if raw else None


@traced_node("extract_slots")
async def extract_slots(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    booking = booking_state(state)
    history: tuple[Message, ...] = tuple(
        {"role": m["role"], "content": m["content"]} for m in (state.get("messages") or [])[:-1]
    )
    ctx = ExtractionContext(
        now=deps.now(),
        current_state=booking,
        user_input=state.get("user_input", ""),
        candidates=tuple(c.name for c in booking.candidates),
        history=history,
        pending_confirmation=state.get("pending_action")
        == PendingAction.AWAIT_CONFIRM_BOOKING.value,
    )
    result = await deps.require_extractor().extract(ctx)
    if result.extraction is None:
        return {"extraction": None, "reply_type": "extraction_failed", "action": "clarify"}
    return {"extraction": dump_extraction(result.extraction)}


def route_extraction(state: AgentState) -> str:
    if state.get("error") or not state.get("extraction"):
        return "render"
    intent = (state.get("extraction") or {}).get("turn_intent")
    if intent == TurnIntent.CONSULT.value:
        return "consult_interject"
    if intent == TurnIntent.UNRELATED.value:
        return "unrelated"
    return "merge_state"
