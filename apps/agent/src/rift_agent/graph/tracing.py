"""``@traced_node``: run a LangGraph node inside a ``node:<name>`` span.

A crashing node does not end the conversation: the exception is recorded on the span
(status ``error``), logged, and turned into ``{"error": ..., "reply_type": "error"}`` so
the graph routes to ``render`` and answers with a clarification while every other state
field keeps its previous value. LangGraph control-flow exceptions (``interrupt()``)
pass through untouched.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp

from rift_agent.graph.state import AgentState
from rift_common.trace import span

logger = logging.getLogger(__name__)

NodeFn = Callable[[AgentState, RunnableConfig], Awaitable[dict[str, Any]]]


def traced_node(name: str) -> Callable[[NodeFn], NodeFn]:
    def decorate(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            try:
                with span(f"node:{name}") as s:
                    update = await fn(state, config)
                    if update.get("action"):
                        s.set_attr("action", update["action"])
                    return update
            except GraphBubbleUp:
                raise
            except Exception as exc:
                logger.exception("node %s failed", name)
                return {"error": f"{name}: {type(exc).__name__}: {exc}", "reply_type": "error"}

        return wrapper

    return decorate


def unless_error(target: str) -> Callable[[AgentState], str]:
    """Edge router: go to ``target`` normally, straight to ``render`` after a crash."""

    def route(state: AgentState) -> str:
        return "render" if state.get("error") else target

    return route
