"""Public entry point: ``Agent`` runs one user turn at a time per session.

- The graph is checkpointed in SQLite (``thread_id = session_id``); a restarted process
  continues every session, including one paused on a confirmation.
- Each turn is one trace (``start_turn``): input, reply, phase before/after, turn index.
- ``run_turn_events`` streams structured events for the web UI (``token`` chunks, then
  ``candidates`` / ``confirm`` / ``booked`` payloads, then ``done``; ``error`` instead
  when the turn fails). ``run_turn`` yields only the text chunks (CLI).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from rift_agent.deps import AgentDeps
from rift_agent.graph.builder import build_graph
from rift_agent.graph.state import Phase, initial_state
from rift_common.trace import start_turn
from rift_common.trace.sinks.base import TraceSink

logger = logging.getLogger(__name__)

EventType = Literal["token", "candidates", "confirm", "booked", "done", "error"]
CHUNK_CHARS = 12


@dataclass(frozen=True)
class TurnResult:
    reply: str
    reply_type: str
    phase: str
    turn_idx: int
    trace_id: str
    ui: dict[str, Any] | None = None
    #: True when the agent asked a yes/no question and waits for the answer.
    awaiting_confirmation: bool = False


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    data: Any = None


def chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


@dataclass
class Agent:
    graph: Any
    deps: AgentDeps
    sinks: Sequence[TraceSink] = ()
    _locks: defaultdict[str, asyncio.Lock] = field(
        default_factory=lambda: defaultdict(asyncio.Lock), repr=False
    )

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        deps: AgentDeps,
        checkpoint_path: str | Path,
        *,
        sinks: Sequence[TraceSink] = (),
    ) -> AsyncIterator[Agent]:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            yield cls(build_graph(saver), deps, sinks)

    def _config(self, session_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": session_id, "deps": self.deps}}

    async def state(self, session_id: str) -> dict[str, Any]:
        return dict((await self.graph.aget_state(self._config(session_id))).values)

    async def history(self, session_id: str) -> list[dict[str, str]]:
        return list((await self.state(session_id)).get("messages") or [])

    async def run(self, session_id: str, user_id: int, text: str) -> TurnResult:
        async with self._locks[session_id]:
            return await self._run(session_id, user_id, text)

    async def _run(self, session_id: str, user_id: int, text: str) -> TurnResult:
        config = self._config(session_id)
        snapshot = await self.graph.aget_state(config)
        values = snapshot.values or {}
        turn_idx = int(values.get("turn_idx") or 0) + 1
        with start_turn(
            session_id=session_id,
            user_id=str(user_id),
            turn_idx=turn_idx,
            sinks=self.sinks,
            input=text,
            phase_before=values.get("phase", Phase.IDLE.value),
        ) as root:
            if snapshot.next:
                payload: Any = Command(resume=text)
            elif not values:
                payload = {**initial_state(session_id, user_id), "user_input": text}
            else:
                payload = {"user_input": text, "user_id": user_id}
            await self.graph.ainvoke(payload, config)
            after = await self.graph.aget_state(config)
            out = after.values
            result = TurnResult(
                reply=out.get("reply", ""),
                reply_type=out.get("reply_type", ""),
                phase=out.get("phase", Phase.IDLE.value),
                turn_idx=int(out.get("turn_idx") or turn_idx),
                trace_id=root.trace_id,
                ui=out.get("ui"),
                awaiting_confirmation=bool(after.next),
            )
            root.set_attrs(
                reply=result.reply, reply_type=result.reply_type, phase_after=result.phase
            )
            return result

    async def run_turn_events(
        self, session_id: str, user_id: int, text: str
    ) -> AsyncIterator[AgentEvent]:
        try:
            result = await self.run(session_id, user_id, text)
        except Exception as exc:
            logger.exception("turn failed for session %s", session_id)
            yield AgentEvent(
                "error", {"message": "抱歉，服务出了点问题，请稍后再试。", "detail": str(exc)}
            )
            return
        for piece in chunks(result.reply):
            yield AgentEvent("token", piece)
        ui = result.ui or {}
        for key in ("candidates", "confirm", "booked"):
            if ui.get(key):
                yield AgentEvent(key, ui[key])
        yield AgentEvent(
            "done",
            {
                "reply_type": result.reply_type,
                "phase": result.phase,
                "turn_idx": result.turn_idx,
                "trace_id": result.trace_id,
                "awaiting_confirmation": result.awaiting_confirmation,
            },
        )

    async def run_turn(self, session_id: str, user_id: int, text: str) -> AsyncIterator[str]:
        async for event in self.run_turn_events(session_id, user_id, text):
            if event.type == "token":
                yield str(event.data)
            elif event.type == "error":
                yield str(event.data["message"])
