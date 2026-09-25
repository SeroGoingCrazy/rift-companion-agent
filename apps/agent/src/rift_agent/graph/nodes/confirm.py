"""``confirm``: pause the graph until the user answers the yes/no just asked.

The reply asking the question has already been rendered, so the turn ends here. The
next user message resumes the graph (``Command(resume=text)``): it becomes this turn's
input and goes to slot extraction (booking) or ``manage_confirm`` (cancellation).
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from rift_agent.deps import get_deps
from rift_agent.graph.nodes.common import start_of_turn
from rift_agent.graph.state import AgentState
from rift_agent.graph.tracing import traced_node
from rift_domain.enums import PendingAction


@traced_node("confirm")
async def confirm(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    answer = interrupt(
        {
            "pending_action": state.get("pending_action"),
            "reply": state.get("reply", ""),
            "ui": state.get("ui"),
        }
    )
    return start_of_turn(state, str(answer), deps.history_turns)


def route_confirmed(state: AgentState) -> str:
    if state.get("error"):
        return "render"
    if state.get("pending_action") == PendingAction.AWAIT_CONFIRM_CANCEL.value:
        return "manage_confirm"
    return "extract_slots"
