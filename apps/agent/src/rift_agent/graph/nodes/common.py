"""Turn entry (``route_phase``), help reply (``other``) and ``render``."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from rift_agent.deps import get_deps
from rift_agent.graph.state import TURN_RESET, AgentState, Phase, append_message, booking_state
from rift_agent.graph.tracing import traced_node

logger = logging.getLogger(__name__)

#: Replies that ask the user a yes/no question answered through ``confirm``.
CONFIRM_REPLIES = frozenset({"confirm_booking", "reconfirm", "cancel_confirm", "cancel_unclear"})


def start_of_turn(state: AgentState, text: str, keep_turns: int) -> dict[str, Any]:
    """Scratch reset + the user's message appended (used at entry and on resume)."""
    return {
        **TURN_RESET,
        "user_input": text,
        "turn_idx": int(state.get("turn_idx") or 0) + 1,
        "messages": append_message(
            list(state.get("messages") or []), "user", text, keep_turns=keep_turns
        ),
    }


@traced_node("route_phase")
async def route_phase(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    return start_of_turn(state, state.get("user_input", ""), deps.history_turns)


def route_from_phase(state: AgentState) -> str:
    if state.get("error"):
        return "render"
    phase = state.get("phase") or Phase.IDLE.value
    if phase == Phase.BOOKING.value:
        return "extract_slots"
    if phase == Phase.MANAGE.value:
        return "manage"
    return "classify"


@traced_node("other")
async def other(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    return {"reply_type": "help", "action": "help"}


@traced_node("render")
async def render(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    renderer = deps.require_replies()
    reply_type = state.get("reply_type") or "help"
    facts = dict(state.get("facts") or {})
    booking = booking_state(state)
    try:
        text = renderer.render(reply_type, facts, booking)
    except Exception:
        logger.exception("rendering %s failed", reply_type)
        reply_type, text = "error", renderer.render("error", {}, booking)
    messages = append_message(
        list(state.get("messages") or []), "assistant", text, keep_turns=deps.history_turns
    )
    return {"reply": text, "reply_type": reply_type, "messages": messages, "error": None}


def after_render(state: AgentState) -> str:
    """Pause for a yes/no when the reply just asked for one."""
    if state.get("pending_action") and state.get("reply_type") in CONFIRM_REPLIES:
        return "confirm"
    return END
