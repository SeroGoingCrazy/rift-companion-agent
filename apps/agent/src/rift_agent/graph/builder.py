"""Assemble the LangGraph state machine (DEV_SPEC 3.3.2).

::

    START -> route_phase -> classify (IDLE) / extract_slots (BOOKING) / manage (MANAGE)
    classify      -> extract_slots | consult | manage | other
    extract_slots -> merge_state | consult_interject | unrelated | render
    merge_state   -> decide -> ask (render) | find_companions | quote | book | abandon
    find_companions -> quote | render
    quote -> render ; book -> render | find_companions (slot conflict)
    render -> confirm (interrupt, when a yes/no was asked) | END
    confirm -> extract_slots (booking) | manage_confirm (cancellation)

Every edge leaving a node goes to ``render`` instead when that node crashed
(``traced_node`` sets ``error``). A route whose target is not registered also lands on
``render``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rift_agent.graph.nodes.book import book, route_booked
from rift_agent.graph.nodes.classify import classify, route_intent
from rift_agent.graph.nodes.common import after_render, other, render, route_from_phase, route_phase
from rift_agent.graph.nodes.confirm import confirm, route_confirmed
from rift_agent.graph.nodes.consult import abandon, consult, consult_interject, unrelated
from rift_agent.graph.nodes.decide import decide, route_decision
from rift_agent.graph.nodes.extract import extract_slots, route_extraction
from rift_agent.graph.nodes.find import find_companions, route_found
from rift_agent.graph.nodes.merge import merge_state
from rift_agent.graph.nodes.quote import quote
from rift_agent.graph.state import AgentState
from rift_agent.graph.tracing import NodeFn, unless_error

NODES: dict[str, NodeFn] = {
    "route_phase": route_phase,
    "classify": classify,
    "other": other,
    "extract_slots": extract_slots,
    "merge_state": merge_state,
    "decide": decide,
    "find_companions": find_companions,
    "quote": quote,
    "book": book,
    "confirm": confirm,
    "consult": consult,
    "consult_interject": consult_interject,
    "unrelated": unrelated,
    "abandon": abandon,
    "render": render,
}

#: node -> (router, possible targets)
ROUTES: dict[str, tuple[Callable[[AgentState], str], tuple[str, ...]]] = {
    "route_phase": (route_from_phase, ("classify", "extract_slots", "manage", "render")),
    "classify": (route_intent, ("extract_slots", "consult", "manage", "other", "render")),
    "extract_slots": (
        route_extraction,
        ("merge_state", "consult_interject", "unrelated", "render"),
    ),
    "decide": (route_decision, ("book", "abandon", "quote", "find_companions", "render")),
    "find_companions": (route_found, ("quote", "render")),
    "book": (route_booked, ("find_companions", "render")),
    "confirm": (route_confirmed, ("extract_slots", "manage_confirm", "render")),
    "render": (after_render, ("confirm", END)),
}

#: Plain "go on unless it crashed" edges.
EDGES: dict[str, str] = {
    "merge_state": "decide",
    "quote": "render",
    "other": "render",
    "consult": "render",
    "consult_interject": "render",
    "unrelated": "render",
    "abandon": "render",
}


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    *,
    nodes: Mapping[str, NodeFn] | None = None,
    routes: Mapping[str, tuple[Callable[[AgentState], str], tuple[str, ...]]] | None = None,
    edges: Mapping[str, str] | None = None,
) -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    nodes = NODES if nodes is None else nodes
    routes = ROUTES if routes is None else routes
    edges = EDGES if edges is None else edges

    g: StateGraph[AgentState, Any, AgentState, AgentState] = StateGraph(AgentState)
    for name, fn in nodes.items():
        g.add_node(name, fn)  # type: ignore[call-overload]
    g.add_edge(START, "route_phase")

    def resolve(target: str) -> str:
        return target if target == END or target in nodes else "render"

    for src, (router, targets) in routes.items():
        if src in nodes:
            g.add_conditional_edges(src, router, {t: resolve(t) for t in targets})
    for src, dst in edges.items():
        if src in nodes:
            target = resolve(dst)
            g.add_conditional_edges(src, unless_error(target), {target: target, "render": "render"})
    return g.compile(checkpointer=checkpointer)
