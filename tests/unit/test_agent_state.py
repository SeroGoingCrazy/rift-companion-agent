"""F1: agent state is checkpoint-safe; traced nodes record spans and absorb crashes."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from rift_agent.graph.state import (
    TURN_RESET,
    AgentState,
    Phase,
    append_message,
    booking_state,
    dump_booking,
    initial_state,
    phase_of,
)
from rift_agent.graph.tracing import traced_node, unless_error
from rift_common.trace import TraceContext, start_turn
from rift_common.trace.sinks.base import TraceSink
from rift_domain.enums import GameMode, PendingAction, Rank, Role, ServiceType
from rift_domain.slots import ANY, BookingState, Candidate, Quote


class Recorder(TraceSink):
    def __init__(self) -> None:
        self.traces: list[TraceContext] = []

    def on_turn_end(self, trace: TraceContext) -> None:
        self.traces.append(trace)


FULL = BookingState(
    game_mode=GameMode.RANKED_SOLO_DUO,
    start_time=datetime(2026, 10, 2, 20, 0),
    start_time_expr="明晚八点",
    duration_hours=2.0,
    rank_requirement=Rank.DIAMOND,
    role_preference=(Role.JUNGLE, Role.SUPPORT),
    companion_gender=ANY,
    budget_per_hour=80.0,
    candidates=(Candidate(companion_id=3, name="阿狸酱", score=0.9, reasons=("评分 4.9",)),),
    selected_companion_id=3,
    quote=Quote(
        unit_price=Decimal("60.00"),
        service_type=ServiceType.CLIMB,
        multiplier=Decimal("1.2"),
        hours=Decimal("2"),
        effective_hourly=Decimal("72.00"),
        total=Decimal("144.00"),
    ),
    pending_action=PendingAction.AWAIT_CONFIRM_BOOKING,
)


def test_booking_round_trips_through_json() -> None:
    dumped = dump_booking(FULL)
    assert dumped["start_time"] == "2026-10-02T20:00:00"
    assert dumped["quote"]["total"] == "144.00"
    assert booking_state({"booking": dumped}) == FULL


def test_initial_state_and_helpers() -> None:
    s = initial_state("s1", 7)
    assert phase_of(s) is Phase.IDLE
    assert booking_state(s) == BookingState()
    assert phase_of({}) is Phase.IDLE
    assert set(TURN_RESET) >= {"extraction", "reply", "ui", "error", "action"}


def test_append_message_keeps_window() -> None:
    msgs: list[Any] = []
    for i in range(5):
        msgs = append_message(msgs, "user", f"u{i}", keep_turns=2)
        msgs = append_message(msgs, "assistant", f"a{i}", keep_turns=2)
    assert [m["content"] for m in msgs] == ["u3", "a3", "u4", "a4"]
    assert append_message(msgs, "user", "x", keep_turns=0) == []


async def test_state_survives_sqlite_checkpoint(tmp_path: Path) -> None:
    async def write(state: AgentState) -> dict[str, Any]:
        return {"booking": dump_booking(FULL), "phase": Phase.BOOKING.value, "manage": {"x": [1]}}

    async def pause(state: AgentState) -> dict[str, Any]:
        answer = interrupt({"summary": "确认吗"})
        return {"user_input": answer}

    g: StateGraph[AgentState] = StateGraph(AgentState)
    g.add_node("write", write)
    g.add_node("pause", pause)
    g.add_edge(START, "write")
    g.add_edge("write", "pause")
    g.add_edge("pause", END)
    cfg: RunnableConfig = {"configurable": {"thread_id": "t"}}
    db = str(tmp_path / "cp.db")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await g.compile(checkpointer=saver).ainvoke(initial_state("t", 1), cfg)
    async with AsyncSqliteSaver.from_conn_string(db) as saver:  # a new process, same file
        snapshot = await g.compile(checkpointer=saver).aget_state(cfg)
    assert snapshot.next == ("pause",)
    assert booking_state(snapshot.values) == FULL
    assert snapshot.values["manage"] == {"x": [1]}


async def test_traced_node_records_span_and_action() -> None:
    @traced_node("decide")
    async def decide(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        return {"action": "find"}

    sink = Recorder()
    with start_turn(sinks=[sink]):
        assert await decide({}, {}) == {"action": "find"}
    node = next(s for s in sink.traces[0].spans if s.name == "node:decide")
    assert node.status == "ok"
    assert node.attrs["action"] == "find"


async def test_traced_node_turns_crash_into_error_reply() -> None:
    @traced_node("merge_state")
    async def merge(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        raise ValueError("bad delta")

    sink = Recorder()
    with start_turn(sinks=[sink]):
        update = await merge({}, {})
    assert update == {"error": "merge_state: ValueError: bad delta", "reply_type": "error"}
    node = next(s for s in sink.traces[0].spans if s.name == "node:merge_state")
    assert node.status == "error"
    assert node.error == "ValueError: bad delta"


async def test_traced_node_lets_interrupt_through(tmp_path: Path) -> None:
    @traced_node("confirm")
    async def confirm(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        return {"user_input": interrupt("ok?")}

    g: StateGraph[AgentState] = StateGraph(AgentState)
    g.add_node("confirm", confirm)
    g.add_edge(START, "confirm")
    g.add_edge("confirm", END)
    cfg: RunnableConfig = {"configurable": {"thread_id": "t"}}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        out = await g.compile(checkpointer=saver).ainvoke({}, cfg)
    assert "__interrupt__" in out
    assert "error" not in out


@pytest.mark.parametrize(("error", "target"), [(None, "decide"), ("boom", "render")])
def test_unless_error(error: str | None, target: str) -> None:
    assert unless_error("decide")({"error": error}) == target
