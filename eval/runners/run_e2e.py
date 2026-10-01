"""L4 end-to-end evaluation: replay YAML scenarios against the full agent.

Every scenario gets a fresh world: a temporary booking database seeded for the
scenario's ``now`` (booking-mcp runs in-process), a fixed clock, a new checkpoint DB and
optional setup bookings. Turns are driven through ``rift_agent.api.Agent`` exactly as
the web app does; assertions look at the database and the trace spans, never at the
wording of replies.

``--config``:

- ``llm``: DeepSeek classifies, extracts and answers (M1);
- ``local``: llama-server extracts, DeepSeek for the rest;
- ``routed``: local extraction with DeepSeek fallback (M3);
- ``mock``: no model at all -- each turn's ``mock`` block (intent + extraction) is
  replayed; only scenarios that carry one run (CI smoke).

``--knowledge``: ``rag`` (knowledge-mcp in-process on its ingested KB), ``http``
(``mcp.knowledge.url``) or ``stub`` (keyword search over ``kb/`` Markdown, offline).

Metrics: task completion rate (all assertions held), average turns, per-turn latency
P50 / P95 and remote LLM calls per session (``llm:*`` generation spans, llama-server
excluded).

Scenario format (``eval/scenarios/*.yaml``)::

    id: interject_refund_then_continue
    now: "2026-10-01 14:00"
    user: { nickname: tester_01 }
    users: [tester_02]                      # optional extra users
    setup:                                  # optional bookings made before the chat
      bookings:
        - { game_mode: aram, start_time: "2026-10-04 20:00", duration_hours: 2, paid: true }
    turns:
      - say: "等下，取消要扣钱吗？"
        session: main                       # optional; other names = other sessions
        user: tester_01                     # optional; the session's user
        restart: false                      # rebuild the agent first (same checkpoints)
        expect_span: { tool: query_knowledge_hub }
        expect_reply_type: [consult]
        expect_slots: { rank_requirement: diamond }
        expect_relaxations: true
        mock: { intent: booking, extraction: {turn_intent: consult, delta: {}, ...} }
    expect:
      booking: { status: pending_payment, game_mode: ranked_solo_duo,
                 start_time: "2026-10-02 20:00", duration_hours: 2 }
      no_booking: true                      # instead of booking
      bookings: { tester_02: 0 }            # bookings created during the scenario
      setup_bookings: [{ status: cancelled, refund_ratio: 0.5 }]
      state_phase: IDLE

Usage::

    uv run --env-file .env python eval/runners/run_e2e.py --config llm
    uv run python eval/runners/run_e2e.py --config mock --knowledge stub
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import re
import sys
import tempfile
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import yaml

from booking_mcp.db import create_db_engine, init_db, make_session_factory
from booking_mcp.seed import seed_database
from booking_mcp.server import build_server
from booking_mcp.service import BookingService
from rift_agent.api import Agent
from rift_agent.deps import AgentDeps
from rift_agent.extractors.base import ExtractionContext, ExtractionResult, SlotExtractor
from rift_agent.graph.state import AgentState, booking_state
from rift_agent.mcp_clients import (
    BookingClient,
    KnowledgeClient,
    McpToolClient,
    http_connector,
    inprocess_connector,
)
from rift_agent.replies.templates import ReplyRenderer
from rift_common.embedding import BaseEmbedding
from rift_common.llm import BaseLLM
from rift_common.trace.context import TraceContext
from rift_common.trace.sinks.base import TraceSink
from rift_domain.config import DomainConfig, load_domain_config
from rift_domain.slots import parse_extraction

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_DIR = REPO_ROOT / "eval" / "scenarios"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"
DEFAULT_DOMAIN = REPO_ROOT / "config" / "domain.yaml"
KB_DIR = REPO_ROOT / "kb"
CONFIGS = ("llm", "local", "routed", "mock")
KNOWLEDGE_MODES = ("rag", "http", "stub")
COLLECTIONS = ("platform_rules", "modes_and_ranks", "companion_profiles")
MAIN = "main"


class ScenarioError(ValueError):
    """A scenario file is malformed."""


# --- scenarios -----------------------------------------------------------------------------

_TURN_KEYS = {
    "say",
    "session",
    "user",
    "restart",
    "expect_span",
    "expect_reply_type",
    "expect_slots",
    "expect_relaxations",
    "mock",
}
_EXPECT_KEYS = {"booking", "no_booking", "bookings", "setup_bookings", "state_phase"}
_TOP_KEYS = {"id", "description", "now", "user", "users", "setup", "turns", "expect"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


@dataclass(frozen=True)
class Turn:
    say: str
    session: str = MAIN
    user: str | None = None
    restart: bool = False
    expect_span: tuple[dict[str, Any], ...] = ()
    expect_reply_type: tuple[str, ...] = ()
    expect_slots: dict[str, Any] = field(default_factory=dict)
    expect_relaxations: bool | None = None
    mock: dict[str, Any] | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    now: datetime
    user: str
    turns: tuple[Turn, ...]
    expect: dict[str, Any]
    users: tuple[str, ...] = ()
    setup_bookings: tuple[dict[str, Any], ...] = ()
    description: str = ""
    path: str = ""

    @property
    def has_mock(self) -> bool:
        return all(t.mock is not None for t in self.turns)

    def nicknames(self) -> list[str]:
        names = [self.user, *self.users, *(t.user for t in self.turns if t.user)]
        names += [b["user"] for b in self.setup_bookings if b.get("user")]
        return list(dict.fromkeys(names))


def parse_scenario(data: dict[str, Any], *, path: str = "") -> Scenario:
    where = path or str(data.get("id", "<scenario>"))
    if not isinstance(data, dict):
        raise ScenarioError(f"{where}: not a mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise ScenarioError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("id", "now", "turns"):
        if key not in data:
            raise ScenarioError(f"{where}: missing {key!r}")
    turns: list[Turn] = []
    for i, raw in enumerate(data["turns"], 1):
        if not isinstance(raw, dict) or "say" not in raw:
            raise ScenarioError(f"{where}: turn {i} needs 'say'")
        bad = set(raw) - _TURN_KEYS
        if bad:
            raise ScenarioError(f"{where}: turn {i} has unknown keys {sorted(bad)}")
        mock = raw.get("mock")
        if mock is not None:
            if "intent" not in mock and "extraction" not in mock:
                raise ScenarioError(f"{where}: turn {i} mock needs intent and/or extraction")
            if mock.get("extraction") is not None:
                parse_extraction(mock["extraction"])
        turns.append(
            Turn(
                say=str(raw["say"]),
                session=str(raw.get("session", MAIN)),
                user=raw.get("user"),
                restart=bool(raw.get("restart", False)),
                expect_span=tuple(_as_list(raw.get("expect_span"))),
                expect_reply_type=tuple(_as_list(raw.get("expect_reply_type"))),
                expect_slots=dict(raw.get("expect_slots") or {}),
                expect_relaxations=raw.get("expect_relaxations"),
                mock=mock,
            )
        )
    expect = dict(data.get("expect") or {})
    bad = set(expect) - _EXPECT_KEYS
    if bad:
        raise ScenarioError(f"{where}: unknown expect keys {sorted(bad)}")
    setup = dict(data.get("setup") or {})
    user = (data.get("user") or {}).get("nickname", "tester_01")
    return Scenario(
        id=str(data["id"]),
        now=datetime.fromisoformat(str(data["now"])),
        user=user,
        users=tuple(data.get("users") or ()),
        turns=tuple(turns),
        expect=expect,
        setup_bookings=tuple(setup.get("bookings") or ()),
        description=str(data.get("description", "")),
        path=path,
    )


def load_scenarios(directory: Path, ids: Sequence[str] = ()) -> list[Scenario]:
    scenarios = []
    for p in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(p.read_text("utf-8"))
        scenario = parse_scenario(data, path=p.name)
        if not ids or scenario.id in ids:
            scenarios.append(scenario)
    seen = Counter(s.id for s in scenarios)
    dupes = [i for i, n in seen.items() if n > 1]
    if dupes:
        raise ScenarioError(f"duplicate scenario ids {dupes}")
    return scenarios


# --- fakes for the mock config and the offline knowledge stub -----------------------------


class ReplayExtractor(SlotExtractor):
    """Replays scripted extractions, in order, per user input."""

    name = "replay"

    def __init__(self, turns: Sequence[Turn]) -> None:
        self.queue: dict[str, deque[dict[str, Any]]] = {}
        for t in turns:
            extraction = (t.mock or {}).get("extraction")
            if extraction is not None:
                self.queue.setdefault(t.say, deque()).append(extraction)

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        scripted = self.queue.get(ctx.user_input)
        if not scripted:
            return ExtractionResult.failed(("no scripted extraction",), source=self.name)
        raw = scripted.popleft()
        return ExtractionResult(
            extraction=parse_extraction(raw),
            raw=json.dumps(raw, ensure_ascii=False),
            valid=True,
            model="replay",
            source=self.name,
        )


def mock_llm(turns: Sequence[Turn]) -> BaseLLM:
    """Classifier answers come from each turn's ``mock.intent``; other prompts get a stub."""
    from rift_common.llm.mock import MockLLM
    from rift_common.settings import LLMConfig

    llm = MockLLM(LLMConfig(provider="mock", model="mock-llm"))
    intents = {t.say: t.mock["intent"] for t in turns if t.mock and t.mock.get("intent")}

    def handler(messages: list[Any]) -> Any:
        if "意图分类器" in messages[0]["content"]:
            user = str(messages[-1]["content"])
            for say, intent in intents.items():
                if say in user:
                    return {"intent": intent}
            return {"intent": "other"}
        return "根据平台规则，详情以页面说明为准。"

    llm.handler = handler
    return llm


_SECTION = re.compile(r"^## ", re.M)


def load_kb_sections(kb_dir: Path = KB_DIR) -> dict[str, list[tuple[str, str]]]:
    """collection -> [(source, "## section" text)] from the generated Markdown KB."""
    sections: dict[str, list[tuple[str, str]]] = {c: [] for c in COLLECTIONS}
    if kb_dir.is_dir():
        for path in sorted(kb_dir.glob("*/*.md")):
            text = path.read_text("utf-8")
            parts = [p.strip() for p in _SECTION.split(text) if p.strip()]
            title = parts[0].splitlines()[0] if parts else path.stem
            for part in parts[1:] or parts:
                sections.setdefault(path.parent.name, []).append(
                    (f"{path.parent.name}/{path.name}", f"{title}\n## {part}")
                )
    if not any(sections.values()):  # no generated KB (e.g. CI): a few fixed passages
        sections["platform_rules"] = [
            ("platform_rules/refund.md", "## 退款档位\n开局前 24 小时以上取消全额退款；"
             "2–24 小时退一半；不足 2 小时不退款。"),
            ("platform_rules/billing.md", "## 计价公式\n总价 = 基础单价 × 服务系数 × 时长。"),
        ]  # fmt: skip
        sections["modes_and_ranks"] = [
            ("modes_and_ranks/game_modes.md", "## 斗魂竞技场\n2 人一队，共 8 队。"),
        ]
    return sections


def _bigrams(text: str) -> set[str]:
    chars = [c for c in text if not c.isspace()]
    return {a + b for a, b in pairwise(chars)}


def stub_knowledge_server(sections: dict[str, list[tuple[str, str]]]) -> Any:
    """Keyword (character bigram) search with knowledge-mcp's output format."""
    import mcp.types as mcp_types
    from mcp.server.lowlevel import Server

    server: Server[Any, Any] = Server("knowledge-stub")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[mcp_types.Tool]:
        schema = {"type": "object", "properties": {"query": {"type": "string"}}}
        return [mcp_types.Tool(name="query_knowledge_hub", description="", inputSchema=schema)]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        query = _bigrams(str(arguments.get("query", "")))
        pool = sections.get(arguments.get("collection") or "platform_rules", [])
        scored = sorted(((len(query & _bigrams(t)), s, t) for s, t in pool), reverse=True)
        hits = [(s, t) for score, s, t in scored[: int(arguments.get("top_k") or 3)] if score >= 2]
        if not hits:
            return [mcp_types.TextContent(type="text", text="")]
        body = "\n\n".join(f"[{i}] {t}" for i, (_, t) in enumerate(hits, 1))
        refs = {"citations": [{"index": i, "source": s} for i, (s, _) in enumerate(hits, 1)]}
        text = f"{body}\n\n---\n**References (JSON):**\n```json\n{json.dumps(refs)}\n```"
        return [mcp_types.TextContent(type="text", text=text)]

    return server


def make_knowledge(mode: str, settings: Any = None) -> KnowledgeClient:
    collections = tuple(settings.mcp.knowledge.collections) if settings else COLLECTIONS
    if mode == "stub":
        connector = inprocess_connector(stub_knowledge_server(load_kb_sections()))
    elif mode == "http":
        k = settings.mcp.knowledge
        connector = http_connector(k.url, timeout_s=k.timeout_s)
    elif mode == "rag":
        from src.mcp_server.protocol_handler import (  # type: ignore[import-untyped]
            create_mcp_server,
        )

        connector = inprocess_connector(create_mcp_server("knowledge-mcp", "0.1.0"))
    else:
        raise ValueError(f"unknown knowledge mode {mode!r}")
    return KnowledgeClient(McpToolClient("knowledge", connector), collections=collections)


# --- one scenario --------------------------------------------------------------------------


class SpanCollector(TraceSink):
    def __init__(self) -> None:
        self.traces: list[TraceContext] = []

    def on_turn_end(self, trace: TraceContext) -> None:
        self.traces.append(trace)


def remote_llm_calls(trace: TraceContext) -> int:
    return sum(
        1
        for s in trace.spans
        if s.kind == "generation" and s.name.startswith("llm:") and s.name != "llm:llama_server"
    )


def tool_names(trace: TraceContext) -> list[str]:
    return sorted({s.name.removeprefix("tool:") for s in trace.spans if s.name.startswith("tool:")})


@dataclass
class TurnRecord:
    session: str
    say: str
    reply: str
    reply_type: str
    phase: str
    latency_ms: float
    remote_llm_calls: int
    tools: list[str]
    failures: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    id: str
    passed: bool
    failures: list[str]
    turns: list[TurnRecord]
    bookings: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    error: str | None = None
    description: str = ""

    @property
    def remote_llm_calls(self) -> int:
        return sum(t.remote_llm_calls for t in self.turns)


@dataclass
class Runtime:
    """What every scenario shares: settings, domain, models, knowledge client."""

    config: str
    domain: DomainConfig
    knowledge: KnowledgeClient
    settings: Any = None
    llm: BaseLLM | None = None
    embedder: BaseEmbedding | None = None


def _norm_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_norm_value(v) for v in value]
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return getattr(value, "value", value)


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, int | float | Decimal) and not isinstance(expected, bool):
        try:
            return Decimal(str(actual)) == Decimal(str(expected))
        except (ArithmeticError, ValueError):
            return False
    if isinstance(expected, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}.*", expected):
        try:
            return datetime.fromisoformat(str(actual)) == datetime.fromisoformat(expected)
        except ValueError:
            return False
    if isinstance(expected, list) and isinstance(actual, list | tuple):
        return sorted(map(str, expected)) == sorted(map(str, actual))
    return bool(_norm_value(actual) == expected)


#: scenario key -> BookingOut key
_BOOKING_FIELDS = {"duration_hours": "hours"}


def check_booking(expected: dict[str, Any], booking: dict[str, Any], label: str) -> list[str]:
    failures = []
    for key, want in expected.items():
        if key == "refund_ratio":
            total = Decimal(str(booking.get("total") or 0))
            refund = Decimal(str(booking.get("refund_amount") or 0))
            got: Any = (refund / total).quantize(Decimal("0.01")) if total else None
            if got is None or got != Decimal(str(want)).quantize(Decimal("0.01")):
                failures.append(f"{label}.refund_ratio: expected {want}, got {got}")
            continue
        got = booking.get(_BOOKING_FIELDS.get(key, key))
        if not _same(want, got):
            failures.append(f"{label}.{key}: expected {want!r}, got {got!r}")
    return failures


@contextlib.asynccontextmanager
async def scenario_world(
    scenario: Scenario, runtime: Runtime, workdir: Path
) -> AsyncIterator[tuple[BookingClient, dict[str, int], list[int]]]:
    """Seeded booking DB + in-process booking-mcp, users, setup bookings."""
    engine = create_db_engine(workdir / "booking.db")
    try:
        init_db(engine)
        factory = make_session_factory(engine)
        seed_database(engine, factory, runtime.domain, start_date=scenario.now.date())
        service = BookingService(
            factory, runtime.domain, embedder=runtime.embedder, clock=lambda: scenario.now
        )
        booking = BookingClient(
            McpToolClient("booking", inprocess_connector(build_server(service)))
        )
        users = {}
        for nickname in scenario.nicknames():
            users[nickname] = int((await booking.ensure_user(nickname))["user_id"])
        setup_ids = []
        for spec in scenario.setup_bookings:
            setup_ids.append(
                await _setup_booking(booking, spec, users[spec.get("user", scenario.user)])
            )
        yield booking, users, setup_ids
    finally:
        engine.dispose()


async def _setup_booking(booking: BookingClient, spec: dict[str, Any], user_id: int) -> int:
    start = datetime.fromisoformat(str(spec["start_time"]))
    found = await booking.find_companions(
        game_mode=spec["game_mode"], start_time=start, duration_hours=spec["duration_hours"]
    )
    exact = [c for c in found["candidates"] if datetime.fromisoformat(c["start_time"]) == start]
    if not exact:
        raise ScenarioError(f"setup booking {spec}: no companion free at {start}")
    created = await booking.create_booking(
        user_id=user_id,
        companion_id=exact[0]["companion_id"],
        start_time=start,
        duration_hours=spec["duration_hours"],
        game_mode=spec["game_mode"],
    )
    booking_id = int(created["booking_id"])
    if spec.get("paid"):
        await booking.pay_booking(user_id, booking_id)
    return booking_id


def build_scenario_deps(scenario: Scenario, runtime: Runtime, booking: BookingClient) -> AgentDeps:
    def clock() -> datetime:
        return scenario.now

    if runtime.config == "mock":
        return AgentDeps(
            domain=runtime.domain,
            llm=mock_llm(scenario.turns),
            extractor=ReplayExtractor(scenario.turns),
            booking=booking,
            knowledge=runtime.knowledge,
            replies=ReplyRenderer(runtime.domain),
            clock=clock,
        )
    from rift_agent.factory import build_deps, with_extractor_config

    settings = with_extractor_config(runtime.settings, runtime.config)
    return build_deps(
        settings,
        runtime.domain,
        llm=runtime.llm,
        booking=booking,
        knowledge=runtime.knowledge,
        clock=clock,
    )


def check_turn(turn: Turn, record: TurnRecord, state: dict[str, Any] | None) -> list[str]:
    failures = []
    if turn.expect_reply_type and record.reply_type not in turn.expect_reply_type:
        failures.append(
            f"reply_type: expected {list(turn.expect_reply_type)}, got {record.reply_type!r}"
        )
    for want in turn.expect_span:
        if "tool" in want and want["tool"] not in record.tools:
            failures.append(f"span: expected tool:{want['tool']}, got {record.tools}")
    if state is not None and (turn.expect_slots or turn.expect_relaxations is not None):
        b = booking_state(cast(AgentState, state))
        for key, want in turn.expect_slots.items():
            got = _norm_value(getattr(b, key, None))
            if not _same(want, got):
                failures.append(f"slot {key}: expected {want!r}, got {got!r}")
        if turn.expect_relaxations is not None and bool(b.relaxations) != turn.expect_relaxations:
            failures.append(
                f"relaxations: expected {'some' if turn.expect_relaxations else 'none'}, "
                f"got {list(b.relaxations)}"
            )
    return failures


async def check_final(
    scenario: Scenario,
    agent: Agent,
    booking: BookingClient,
    users: dict[str, int],
    setup_ids: list[int],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    expect, failures = scenario.expect, []
    created: dict[str, list[dict[str, Any]]] = {}
    all_bookings: dict[int, dict[str, Any]] = {}
    for nickname, uid in users.items():
        rows = await booking.list_my_bookings(uid)
        all_bookings.update({b["booking_id"]: b for b in rows})
        created[nickname] = sorted(
            (b for b in rows if b["booking_id"] not in setup_ids), key=lambda b: b["booking_id"]
        )

    mine = created.get(scenario.user, [])
    if "booking" in expect:
        if not mine:
            failures.append("booking: expected a booking, none was created")
        else:
            failures += check_booking(expect["booking"], mine[-1], "booking")
    if expect.get("no_booking") and mine:
        failures.append(f"no_booking: {len(mine)} booking(s) were created")
    for nickname, count in (expect.get("bookings") or {}).items():
        got = len(created.get(nickname, []))
        if got != count:
            failures.append(f"bookings[{nickname}]: expected {count}, got {got}")
    for i, want in enumerate(expect.get("setup_bookings") or []):
        if i >= len(setup_ids):
            failures.append(f"setup_bookings[{i}]: no such setup booking")
            continue
        failures += check_booking(want, all_bookings[setup_ids[i]], f"setup_bookings[{i}]")
    if "state_phase" in expect:
        phase = (await agent.state(MAIN)).get("phase", "IDLE")
        if phase != expect["state_phase"]:
            failures.append(f"state_phase: expected {expect['state_phase']}, got {phase}")
    return failures, created


async def run_scenario(scenario: Scenario, runtime: Runtime) -> ScenarioResult:
    turns: list[TurnRecord] = []
    failures: list[str] = []
    created: dict[str, list[dict[str, Any]]] = {}
    collector = SpanCollector()
    with tempfile.TemporaryDirectory(prefix=f"l4_{scenario.id}_") as tmp:
        workdir = Path(tmp)
        try:
            async with scenario_world(scenario, runtime, workdir) as (booking, users, setup_ids):
                deps = build_scenario_deps(scenario, runtime, booking)
                checkpoints = workdir / "checkpoints.db"
                stack = contextlib.AsyncExitStack()
                agent = await stack.enter_async_context(
                    Agent.open(deps, checkpoints, sinks=[collector])
                )
                sessions: dict[str, str] = {}
                try:
                    for i, turn in enumerate(scenario.turns, 1):
                        if turn.restart:
                            await stack.aclose()
                            stack = contextlib.AsyncExitStack()
                            agent = await stack.enter_async_context(
                                Agent.open(deps, checkpoints, sinks=[collector])
                            )
                        nickname = turn.user or sessions.get(turn.session) or scenario.user
                        sessions.setdefault(turn.session, nickname)
                        t0 = time.perf_counter()
                        result = await agent.run(turn.session, users[nickname], turn.say)
                        latency = (time.perf_counter() - t0) * 1000
                        trace = collector.traces[-1]
                        record = TurnRecord(
                            session=turn.session,
                            say=turn.say,
                            reply=result.reply,
                            reply_type=result.reply_type,
                            phase=result.phase,
                            latency_ms=round(latency, 1),
                            remote_llm_calls=remote_llm_calls(trace),
                            tools=tool_names(trace),
                        )
                        needs_state = turn.expect_slots or turn.expect_relaxations is not None
                        state = await agent.state(turn.session) if needs_state else None
                        record.failures = check_turn(turn, record, state)
                        failures += [f"turn {i} ({turn.say}): {f}" for f in record.failures]
                        turns.append(record)
                    final, created = await check_final(scenario, agent, booking, users, setup_ids)
                    failures += final
                finally:
                    await stack.aclose()
        except Exception as exc:
            return ScenarioResult(
                scenario.id,
                False,
                [*failures, f"error: {type(exc).__name__}: {exc}"],
                turns,
                created,
                error=f"{type(exc).__name__}: {exc}",
                description=scenario.description,
            )
    return ScenarioResult(
        scenario.id, not failures, failures, turns, created, description=scenario.description
    )


async def run_all(
    scenarios: Sequence[Scenario],
    runtime: Runtime,
    *,
    log: Callable[[str], None] = print,
) -> list[ScenarioResult]:
    results = []
    for n, scenario in enumerate(scenarios, 1):
        result = await run_scenario(scenario, runtime)
        mark = "PASS" if result.passed else "FAIL"
        log(f"  [{n:>2}/{len(scenarios)}] {mark} {scenario.id} ({len(result.turns)} turns)")
        for f in result.failures:
            log(f"         - {f}")
        results.append(result)
    return results


# --- summary and report --------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    n = len(results)
    latencies = [t.latency_ms for r in results for t in r.turns]
    passed = [r for r in results if r.passed]
    return {
        "scenarios": n,
        "passed": len(passed),
        "completion_rate": round(len(passed) / n, 4) if n else 0.0,
        "avg_turns": round(sum(len(r.turns) for r in results) / n, 2) if n else 0.0,
        "turns": len(latencies),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies, default=0.0),
        },
        "remote_llm_calls_per_session": round(sum(r.remote_llm_calls for r in results) / n, 2)
        if n
        else 0.0,
        "remote_llm_calls_per_turn": round(
            sum(r.remote_llm_calls for r in results) / len(latencies), 2
        )
        if latencies
        else 0.0,
        "failed": [r.id for r in results if not r.passed],
    }


def render_markdown(report: dict[str, Any]) -> str:
    meta, s = report["meta"], report["summary"]
    lat = s["latency_ms"]
    lines = [
        f"# L4 端到端评测：{meta['config']}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 配置 | `{meta['config']}`（{meta['model'] or '-'}） |",
        f"| 知识库 | `{meta['knowledge']}` |",
        f"| 剧本 | {s['scenarios']} 个（`eval/scenarios/`） |",
        f"| 时间 | {meta['started_at']} |",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| **任务完成率** | **{s['completion_rate'] * 100:.1f}%**"
        f"（{s['passed']}/{s['scenarios']}） |",
        f"| 平均轮数 | {s['avg_turns']} |",
        f"| 单轮延迟 P50 / P95 / max（ms） | {lat['p50']:.0f} / {lat['p95']:.0f} / "
        f"{lat['max']:.0f} |",
        f"| 远程 LLM 调用 / 会话 | {s['remote_llm_calls_per_session']} |",
        f"| 远程 LLM 调用 / 轮 | {s['remote_llm_calls_per_turn']} |",
        "",
        "## 剧本结果",
        "",
        "| 剧本 | 结果 | 轮数 | LLM 调用 | 最慢一轮（ms） |",
        "|---|---|---:|---:|---:|",
    ]
    for r in report["scenarios"]:
        slowest = max((t["latency_ms"] for t in r["turns"]), default=0)
        calls = sum(t["remote_llm_calls"] for t in r["turns"])
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"| `{r['id']}` | {mark} | {len(r['turns'])} | {calls} | {slowest:.0f} |")
    failed = [r for r in report["scenarios"] if not r["passed"]]
    lines += ["", "## 失败详情", ""]
    if not failed:
        lines.append("无。")
    for r in failed:
        lines += [f"### {r['id']}", ""]
        lines += [f"- {f}" for f in r["failures"]]
        lines += ["", "| # | 会话 | 用户输入 | reply_type | 工具 |", "|---:|---|---|---|---|"]
        for i, t in enumerate(r["turns"], 1):
            tools = ", ".join(t["tools"]) or "-"
            lines.append(f"| {i} | {t['session']} | {t['say']} | `{t['reply_type']}` | {tools} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- CLI -----------------------------------------------------------------------------------


def make_runtime(args: argparse.Namespace) -> Runtime:
    domain = load_domain_config(args.domain)
    if args.config == "mock":
        return Runtime("mock", domain, make_knowledge(args.knowledge or "stub"))
    from rift_common.embedding import EmbeddingFactory
    from rift_common.llm import LLMFactory
    from rift_common.settings import load_settings

    settings = load_settings(args.settings)
    embedder = None if args.no_style_embedding else EmbeddingFactory.create(settings.embedding)
    return Runtime(
        args.config,
        domain,
        make_knowledge(args.knowledge or "rag", settings),
        settings=settings,
        llm=LLMFactory.create(settings.llm["default"]),
        embedder=embedder,
    )


def run(args: argparse.Namespace, *, runtime: Runtime | None = None) -> dict[str, Any]:
    scenarios = load_scenarios(args.scenarios, args.only or ())
    if args.config == "mock":
        skipped = [s.id for s in scenarios if not s.has_mock]
        scenarios = [s for s in scenarios if s.has_mock]
        if skipped:
            print(f"mock config: skipping {len(skipped)} scenario(s) without mock blocks")
    runtime = runtime or make_runtime(args)
    started = datetime.now()
    print(f"{len(scenarios)} scenarios; config {args.config}")
    results = asyncio.run(run_all(scenarios, runtime))
    model = runtime.llm.model if runtime.llm is not None else "mock"
    report: dict[str, Any] = {
        "meta": {
            "config": args.config,
            "model": model,
            "knowledge": args.knowledge or ("stub" if args.config == "mock" else "rag"),
            "started_at": f"{started:%Y-%m-%d %H:%M:%S}",
        },
        "summary": summarize(results),
        "scenarios": [asdict(r) for r in results],
    }
    stem: Path = args.out or REPORTS_DIR / f"e2e_{args.config}_{started:%Y%m%d_%H%M%S}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    report["paths"] = [str(json_path), str(md_path)]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=CONFIGS, default="llm")
    parser.add_argument("--knowledge", choices=KNOWLEDGE_MODES, default=None,
                        help="default: rag (stub for --config mock)")  # fmt: skip
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS_DIR)
    parser.add_argument("--only", nargs="*", help="scenario ids to run")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--domain", type=Path, default=DEFAULT_DOMAIN)
    parser.add_argument("--no-style-embedding", action="store_true",
                        help="rank companions without the bge style embedding")  # fmt: skip
    parser.add_argument("--out", type=Path, default=None, help="report path without suffix")
    args = parser.parse_args(argv)

    report = run(args)
    s = report["summary"]
    print(
        f"\ncompletion {s['passed']}/{s['scenarios']}; avg turns {s['avg_turns']}; "
        f"P50 {s['latency_ms']['p50']:.0f} ms, P95 {s['latency_ms']['p95']:.0f} ms; "
        f"remote LLM calls/session {s['remote_llm_calls_per_session']}"
    )
    for path in report["paths"]:
        print(f"report: {path}")
    return 0 if s["passed"] == s["scenarios"] else 1


if __name__ == "__main__":
    sys.exit(main())
