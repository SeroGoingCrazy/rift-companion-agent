"""Questions and detours: ``consult`` (IDLE), ``consult_interject`` (during a booking),
``unrelated`` (pull the user back) and ``abandon`` ("不约了").

Consult answers come from knowledge-mcp (every configured collection is searched) and the
remote LLM, which must answer from the retrieved passages with citations. An interjection
never touches the booking slots; the reply ends with a hint about what the booking still
needs, so the next message ("那就两小时吧") continues where the user left off.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import AgentDeps, get_deps
from rift_agent.graph.nodes.quote import confirm_facts
from rift_agent.graph.state import AgentState, Phase, booking_state, dump_booking
from rift_agent.graph.tracing import traced_node
from rift_agent.mcp_clients import KnowledgeResult, McpUnavailable, ToolError
from rift_agent.prompts import load_prompt
from rift_common.llm import LLMError, Message
from rift_domain.enums import PendingAction
from rift_domain.slots import BookingState

logger = logging.getLogger(__name__)

COLLECTION_LABELS = {
    "platform_rules": "平台规则",
    "modes_and_ranks": "模式与段位",
    "companion_profiles": "陪玩师资料",
}


def build_context(results: list[KnowledgeResult]) -> str:
    return "\n\n".join(
        f"[{i}]（{COLLECTION_LABELS.get(r.collection or '', r.collection or '资料')}）\n{r.text}"
        for i, r in enumerate(results, start=1)
    )


async def answer_question(deps: AgentDeps, question: str) -> dict[str, Any]:
    """Facts for the ``consult`` reply: answer text and sources, or why there is none."""
    if deps.knowledge is None:
        return {"unavailable": True}
    try:
        results = await deps.knowledge.query_all(question)
    except McpUnavailable as exc:  # expected degradation: no traceback noise
        logger.warning("knowledge service unavailable: %s", exc)
        return {"unavailable": True}
    except ToolError:
        logger.warning("knowledge query failed", exc_info=True)
        return {"unavailable": True}
    if not results:
        return {"no_answer": True}

    sources = [COLLECTION_LABELS.get(r.collection or "", r.collection or "") for r in results]
    messages: list[Message] = [
        {"role": "system", "content": load_prompt("consult_answer.txt")},
        {"role": "user", "content": f"【资料】\n{build_context(results)}\n\n【问题】{question}"},
    ]
    try:
        result = await asyncio.to_thread(deps.llm.chat, messages, temperature=0.2, max_tokens=300)
        answer = result.text.strip()
    except LLMError:
        logger.warning("consult LLM failed; answering with the top passage", exc_info=True)
        answer = results[0].text.strip()
    return {"answer": answer, "sources": list(dict.fromkeys(sources))}


def resume_hint(state: AgentState, booking: BookingState) -> dict[str, Any]:
    """What to say to get back to the booking after a detour."""
    if state.get("pending_action") == PendingAction.AWAIT_CONFIRM_BOOKING.value:
        return {"confirm": True}
    if booking.missing_fields:
        return {"field": booking.missing_fields[0].value}
    if booking.candidates and booking.selected_companion_id is None:
        return {"candidates": True}
    return {}


def _has_booking(booking: BookingState) -> bool:
    return (
        booking.game_mode is not None or booking.start_time is not None or bool(booking.candidates)
    )


@traced_node("consult")
async def consult(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    facts = await answer_question(get_deps(config), state.get("user_input", ""))
    return {"action": "consult", "reply_type": "consult", "facts": facts}


@traced_node("consult_interject")
async def consult_interject(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    booking = booking_state(state)
    facts = await answer_question(get_deps(config), state.get("user_input", ""))
    update: dict[str, Any] = {"action": "consult", "reply_type": "consult"}
    if _has_booking(booking):
        facts["resume"] = resume_hint(state, booking)
        update["phase"] = Phase.BOOKING.value
        if facts["resume"].get("confirm"):
            update["ui"] = {"confirm": confirm_facts(state, booking)}
    else:
        update["phase"] = Phase.IDLE.value
    update["facts"] = facts
    return update


@traced_node("unrelated")
async def unrelated(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    booking = booking_state(state)
    return {
        "action": "unrelated",
        "reply_type": "unrelated",
        "facts": {"resume": resume_hint(state, booking) if _has_booking(booking) else {}},
    }


@traced_node("abandon")
async def abandon(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    dropped = get_deps(config).require_replies().summary(booking_state(state))
    return {
        "action": "abandon",
        "booking": dump_booking(BookingState()),
        "candidate_cards": [],
        "pending_action": None,
        "phase": Phase.IDLE.value,
        "reply_type": "abandon",
        "facts": {"dropped": dropped},
    }
