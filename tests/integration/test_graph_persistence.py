"""F10: Agent API — SQLite persistence across restarts, turn traces, streamed events,
optional reply polishing with template fallback, settings-driven wiring."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from booking_mcp.server import build_server
from booking_mcp.service import BookingService
from booking_mcp.tools.list_my_bookings import list_my_bookings
from booking_mcp.tools.schemas import ListMyBookingsInput
from rift_agent.api import Agent, AgentEvent, chunks
from rift_agent.extractors.routed import RoutedSlotExtractor
from rift_agent.factory import build_deps, build_sinks, make_extractor
from rift_agent.graph.state import booking_state
from rift_agent.replies.polisher import ReplyPolisher, keeps_facts
from rift_common.llm.mock import MockLLM
from rift_common.settings import LLMConfig, SettingsError, load_settings
from rift_common.trace.sinks.sqlite import SqliteSink
from rift_domain.config import DomainConfig

if TYPE_CHECKING:
    from tests.integration.conftest import AgentHarness, BookingDB

NOW = datetime(2026, 10, 1, 14, 0)
FRI_20 = datetime(2026, 10, 2, 20, 0)
REPO_ROOT = Path(__file__).resolve().parents[2]


def turn(confirmation: str = "none", intent: str = "booking", **delta: Any) -> dict[str, Any]:
    return {"turn_intent": intent, "delta": delta, "confirmation": confirmation}


TABLE = {
    "约个大乱斗明晚八点两小时": turn(
        game_mode="aram", start_time_expr="明晚八点", duration_hours=2
    ),
    "就第一个": turn(companion_name="甲"),
    "确认": turn(confirmation="yes"),
}  # fmt: skip


class World:
    def __init__(self, db: BookingDB) -> None:
        self.a = db.add_companion("甲", hourly_price=Decimal("60"))
        db.add_schedule(self.a, FRI_20 - timedelta(hours=2), FRI_20 + timedelta(hours=4))
        self.user = db.add_user("demo")
        self.service = BookingService(db.factory, db.domain, clock=lambda: NOW)
        self.server = build_server(self.service)


@pytest.fixture
def world(make_booking_db: Callable[[str], BookingDB]) -> World:
    return World(make_booking_db("persist"))


@pytest.fixture
def harness(make_agent: Callable[..., AgentHarness], world: World) -> AgentHarness:
    return make_agent(TABLE, server=world.server, user_id=world.user)


async def test_session_survives_restart_while_awaiting_confirmation(
    tmp_path: Path, make_agent: Callable[..., AgentHarness], world: World
) -> None:
    db = tmp_path / "checkpoints.db"
    deps1 = make_agent(TABLE, server=world.server).deps
    async with Agent.open(deps1, db) as agent:
        r1 = await agent.run("sess", world.user, "约个大乱斗明晚八点两小时")
        r2 = await agent.run("sess", world.user, "就第一个")
    assert r1.reply_type == "candidates" and r1.turn_idx == 1
    assert r2.reply_type == "confirm_booking" and r2.awaiting_confirmation
    assert r2.ui is not None and r2.ui["confirm"]["total"] == "120.00"

    # A new process: new deps, new graph, same checkpoint file.
    deps2 = make_agent(TABLE, server=world.server).deps
    async with Agent.open(deps2, db) as agent:
        state = await agent.state("sess")
        assert state["pending_action"] == "await_confirm_booking"
        assert booking_state(state).selected_companion_id == world.a
        assert [m["role"] for m in await agent.history("sess")] == ["user", "assistant"] * 2
        r3 = await agent.run("sess", world.user, "确认")
    assert r3.reply_type == "booked" and not r3.awaiting_confirmation
    assert r3.turn_idx == 3 and r3.phase == "IDLE"
    orders = list_my_bookings(world.service, ListMyBookingsInput(user_id=world.user)).bookings
    assert [o.companion_name for o in orders] == ["甲"]


async def test_sessions_are_isolated(tmp_path: Path, harness: AgentHarness, world: World) -> None:
    async with Agent.open(harness.deps, tmp_path / "cp.db") as agent:
        await agent.run("a", world.user, "约个大乱斗明晚八点两小时")
        r = await agent.run("b", world.user, "就第一个")
        assert booking_state(await agent.state("a")).game_mode == "aram"
    assert r.reply_type != "confirm_booking"  # session b never saw candidates


async def test_turn_trace_written_to_sqlite(
    tmp_path: Path, harness: AgentHarness, world: World
) -> None:
    sink = SqliteSink(tmp_path / "traces.db")
    async with Agent.open(harness.deps, tmp_path / "cp.db", sinks=[sink]) as agent:
        r = await agent.run("traced", world.user, "约个大乱斗明晚八点两小时")
    [row] = sink.list_turns("traced")
    assert row["trace_id"] == r.trace_id
    assert row["input"] == "约个大乱斗明晚八点两小时" and row["reply"] == r.reply
    assert (row["phase_before"], row["phase_after"], row["turn_idx"]) == ("IDLE", "BOOKING", 1)
    names = {s["name"] for s in sink.list_spans(trace_id=r.trace_id)}
    assert {"node:classify", "node:extract_slots", "node:decide", "node:render"} <= names
    assert "tool:find_companions" in names
    assert any(n.startswith("llm:") for n in names)  # classify call is a generation span


async def test_run_turn_events_and_text_stream(
    tmp_path: Path, harness: AgentHarness, world: World
) -> None:
    async with Agent.open(harness.deps, tmp_path / "cp.db") as agent:
        events = [
            e async for e in agent.run_turn_events("ev", world.user, "约个大乱斗明晚八点两小时")
        ]
        kinds = [e.type for e in events]
        assert kinds[-1] == "done" and "candidates" in kinds and "error" not in kinds
        text = "".join(e.data for e in events if e.type == "token")
        assert text == (await agent.state("ev"))["reply"]
        assert events[-1].data["phase"] == "BOOKING"

        events = [e async for e in agent.run_turn_events("ev", world.user, "就第一个")]
        confirm = next(e for e in events if e.type == "confirm")
        assert confirm.data["companion"] == "甲"
        assert events[-1].data["awaiting_confirmation"] is True

        pieces = [p async for p in agent.run_turn("ev", world.user, "确认")]
        assert "下单成功" in "".join(pieces)


async def test_error_event_when_turn_crashes(
    tmp_path: Path, harness: AgentHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with Agent.open(harness.deps, tmp_path / "cp.db") as agent:

        async def boom(*_: Any, **__: Any) -> Any:
            raise RuntimeError("checkpoint disk full")

        monkeypatch.setattr(agent.graph, "ainvoke", boom)
        events = [e async for e in agent.run_turn_events("x", 1, "hi")]
        pieces = [p async for p in agent.run_turn("x", 1, "hi")]
    assert [e.type for e in events] == ["error"]
    assert events[0].data["detail"] == "checkpoint disk full"
    assert pieces == ["抱歉，服务出了点问题，请稍后再试。"]


def test_chunks() -> None:
    assert chunks("abcdef", 4) == ["abcd", "ef"]
    assert chunks("") == [""]
    assert AgentEvent("done").data is None


# --- polishing -------------------------------------------------------------------------------


def _polisher(**kw: Any) -> ReplyPolisher:
    return ReplyPolisher(MockLLM(LLMConfig(provider="mock", model="polish"), **kw))


async def test_polish_used_when_facts_survive(
    tmp_path: Path, harness: AgentHarness, world: World
) -> None:
    harness.deps.polisher = _polisher(handler=lambda m: "润色版：" + _template_of(m))
    async with Agent.open(harness.deps, tmp_path / "cp.db") as agent:
        r = await agent.run("p", world.user, "约个大乱斗明晚八点两小时")
    assert r.reply.startswith("润色版：")


async def test_polish_falls_back_when_numbers_change(
    tmp_path: Path, harness: AgentHarness, world: World
) -> None:
    harness.deps.polisher = _polisher(default="这里有几位不错的陪玩师，随便挑！")
    async with Agent.open(harness.deps, tmp_path / "cp.db") as agent:
        r = await agent.run("p", world.user, "约个大乱斗明晚八点两小时")
    assert r.reply.startswith("按「大乱斗")  # template kept: the prices were dropped


async def test_polish_falls_back_on_llm_error(harness: AgentHarness) -> None:
    llm = MockLLM(LLMConfig(provider="mock", model="p"))
    llm.add_rule(regex=".", error="timeout")
    text, polished = await ReplyPolisher(llm).polish("共 120.00 元", {})
    assert (text, polished) == ("共 120.00 元", False)


def test_keeps_facts() -> None:
    assert keeps_facts("共 120.00 元，#7", "一共是 120.00 元哦（订单 #7）")
    assert not keeps_facts("共 120.00 元", "一共 120 元")
    assert not keeps_facts("1 和 1", "只有 1")


def _template_of(messages: list[Any]) -> str:
    import json

    return str(json.loads(messages[-1]["content"])["template_reply"])


# --- settings wiring -------------------------------------------------------------------------


def _settings(tmp_path: Path, **slot: Any) -> Any:
    raw = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    raw["llm"]["default"] = {"provider": "mock", "model": "mock"}
    raw["slot_extractor"].update(slot)
    raw["trace"]["sqlite_path"] = str(tmp_path / "t.db")
    raw["trace"]["langfuse"]["enabled"] = False
    raw["reply"]["polish"] = True
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_settings(path, env={})


def test_build_deps_from_settings(tmp_path: Path, domain: DomainConfig) -> None:
    settings = _settings(tmp_path)
    deps = build_deps(settings, domain)
    assert isinstance(deps.extractor, RoutedSlotExtractor)
    assert deps.extractor.primary.name == "llm" and deps.extractor.fallback is None
    assert deps.polisher is not None and deps.replies is not None
    assert deps.knowledge is not None
    assert deps.knowledge.collections == ("platform_rules", "modes_and_ranks", "companion_profiles")
    assert deps.history_turns == settings.session.history_turns
    assert [type(s).__name__ for s in build_sinks(settings)] == ["SqliteSink"]


def test_local_extractor_not_available_yet(tmp_path: Path) -> None:
    settings = _settings(tmp_path, primary="llm", shadow="local")
    llm = MockLLM(LLMConfig(provider="mock", model="m"))
    with pytest.raises(SettingsError, match="'local' is not available yet"):
        make_extractor(settings, llm)
