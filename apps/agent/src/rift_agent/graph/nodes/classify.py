"""``classify``: first-turn intent (booking / consult / manage / other) by the remote LLM.

Anything unparseable, and any LLM failure, falls back to ``other`` so the turn still
gets a (help) reply.
"""

from __future__ import annotations

import asyncio
import json
import logging
from enum import StrEnum
from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.state import AgentState
from rift_agent.graph.tracing import traced_node
from rift_agent.prompts import load_prompt
from rift_common.llm import LLMError, Message

logger = logging.getLogger(__name__)


class Intent(StrEnum):
    BOOKING = "booking"
    CONSULT = "consult"
    MANAGE = "manage"
    OTHER = "other"


def parse_intent(text: str) -> Intent:
    """``{"intent": "booking"}`` or a bare label; anything else is ``other``."""
    raw = text.strip()
    label: Any = raw
    try:
        data = json.loads(raw)
        label = data.get("intent") if isinstance(data, dict) else data
    except json.JSONDecodeError:
        pass
    try:
        return Intent(str(label).strip().lower())
    except ValueError:
        return Intent.OTHER


@traced_node("classify")
async def classify(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    messages: list[Message] = [
        {"role": "system", "content": load_prompt("classify.txt")},
        {"role": "user", "content": state.get("user_input", "")},
    ]
    try:
        result = await asyncio.to_thread(
            deps.llm.chat,
            messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=20,
        )
        intent = parse_intent(result.text)
    except LLMError:
        logger.warning("classify LLM call failed; treating as other", exc_info=True)
        intent = Intent.OTHER
    return {"intent": intent.value}


def route_intent(state: AgentState) -> str:
    """Next node after ``classify``."""
    if state.get("error"):
        return "render"
    return {
        Intent.BOOKING.value: "extract_slots",
        Intent.CONSULT.value: "consult",
        Intent.MANAGE.value: "manage",
    }.get(state.get("intent") or "", "other")
