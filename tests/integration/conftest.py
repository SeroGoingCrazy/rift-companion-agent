"""Integration fixtures: a fresh file-backed booking SQLite per test, small builders for
hand-made companions and schedules, and in-process MCP servers for agent tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from mcp.server.lowlevel import Server
from sqlalchemy import Engine

from booking_mcp.db import (
    Companion,
    Schedule,
    ScheduleStatus,
    SessionFactory,
    User,
    create_db_engine,
    init_db,
    make_session_factory,
    write_session,
)
from booking_mcp.seed import seed_database
from booking_mcp.server import build_server
from booking_mcp.service import BookingService
from rift_agent.api import AgentEvent
from rift_agent.deps import AgentDeps
from rift_agent.extractors.base import ExtractionContext, ExtractionResult, SlotExtractor
from rift_agent.graph.builder import build_graph
from rift_agent.graph.state import initial_state
from rift_agent.mcp_clients import (
    BookingClient,
    KnowledgeClient,
    McpToolClient,
    inprocess_connector,
)
from rift_agent.replies.templates import ReplyRenderer
from rift_common.llm.mock import MockLLM
from rift_common.settings import LLMConfig
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_domain.slots import parse_extraction
from rift_web.app import create_app
from rift_web.auth import Auth
from rift_web.services import WebConfig, WebServices


@dataclass
class BookingDB:
    engine: Engine
    factory: SessionFactory
    domain: DomainConfig

    def add_user(self, nickname: str = "tester") -> int:
        with write_session(self.factory) as s:
            user = User(nickname=nickname)
            s.add(user)
            s.flush()
            return user.id

    def add_companion(self, name: str, **overrides: Any) -> int:
        fields: dict[str, Any] = {
            "gender": Gender.FEMALE,
            "rank": Rank.DIAMOND,
            "roles": (Role.MID,),
            "modes": tuple(GameMode),
            "service_types": tuple(ServiceType),
            "hourly_price": Decimal("50"),
            "voice": True,
            "rating": 4.5,
            "tags": ["温柔"],
            "bio": "",
        }
        fields.update(overrides)
        roles, modes, services = (
            fields.pop("roles"),
            fields.pop("modes"),
            fields.pop("service_types"),
        )
        with write_session(self.factory) as s:
            c = Companion(name=name, rank_tier=self.domain.rank_index(fields["rank"]), **fields)
            c.set_skills(roles, modes, services)
            s.add(c)
            s.flush()
            return c.id

    def add_schedule(
        self,
        companion_id: int,
        start: datetime,
        end: datetime,
        status: ScheduleStatus = ScheduleStatus.OPEN,
    ) -> None:
        with write_session(self.factory) as s:
            s.add(Schedule(companion_id=companion_id, start=start, end=end, status=status))


@pytest.fixture
def booking_db(tmp_path: Path, domain: DomainConfig) -> BookingDB:
    engine = create_db_engine(tmp_path / "booking.db")
    init_db(engine)
    return BookingDB(engine, make_session_factory(engine), domain)


@pytest.fixture
def make_booking_db(tmp_path: Path, domain: DomainConfig) -> Callable[[str], BookingDB]:
    def make(name: str) -> BookingDB:
        engine = create_db_engine(tmp_path / f"{name}.db")
        init_db(engine)
        return BookingDB(engine, make_session_factory(engine), domain)

    return make


# --- MCP servers for agent tests -----------------------------------------------------------

#: Fixed "now" for agent tests: Thursday 2026-10-01 14:00; seed starts that day.
AGENT_NOW = datetime(2026, 10, 1, 14, 0)


@pytest.fixture
def seeded_booking(booking_db: BookingDB) -> BookingService:
    seed_database(
        booking_db.engine, booking_db.factory, booking_db.domain, start_date=date(2026, 10, 1)
    )
    return BookingService(booking_db.factory, booking_db.domain, clock=lambda: AGENT_NOW)


@pytest.fixture
def booking_server(seeded_booking: BookingService) -> Server[Any, Any]:
    return build_server(seeded_booking)


#: collection -> [(keyword, markdown passage)]
KNOWLEDGE: dict[str, list[tuple[str, str]]] = {
    "platform_rules": [
        ("取消", "## 退款档位\n距开局 ≥ 24 小时全额退款；2–24 小时退一半；< 2 小时不退款。"),
        ("退", "## 退款档位\n距开局 ≥ 24 小时全额退款；2–24 小时退一半；< 2 小时不退款。"),
        ("收费", "## 计价公式\n总价 = 基础单价 × 服务系数 × 时长；教学系数 1.5。"),
    ],
    "modes_and_ranks": [
        ("斗魂", "## 斗魂竞技场\n人数：8 队 × 2 人。"),
    ],
    "companion_profiles": [],
}


def fake_knowledge_server(
    knowledge: dict[str, list[tuple[str, str]]] | None = None,
) -> Server[Any, Any]:
    """Mimics knowledge-mcp's query_knowledge_hub output (Markdown + JSON references)."""
    data = KNOWLEDGE if knowledge is None else knowledge
    server: Server[Any, Any] = Server("fake-knowledge")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="query_knowledge_hub",
                description="search",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        query, collection = arguments["query"], arguments.get("collection") or "platform_rules"
        hits = [text for kw, text in data.get(collection, []) if kw in query][:1]
        if not hits:
            return [mcp_types.TextContent(type="text", text="")]
        body = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(hits))
        refs = {"citations": [{"index": 1, "source": f"{collection}/doc.md", "score": 0.9}]}
        refs_json = json.dumps(refs, ensure_ascii=False)
        text = f"{body}\n\n---\n**References (JSON):**\n```json\n{refs_json}\n```"
        return [mcp_types.TextContent(type="text", text=text)]

    return server


@pytest.fixture
def knowledge_server() -> Server[Any, Any]:
    return fake_knowledge_server()


# --- agent graph harness -------------------------------------------------------------------


class TableExtractor(SlotExtractor):
    """Returns the extraction scripted for each exact user input (unknown -> failure)."""

    name = "table"

    def __init__(self, table: dict[str, dict[str, Any]]) -> None:
        self.table = table
        self.contexts: list[ExtractionContext] = []

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        self.contexts.append(ctx)
        raw = self.table.get(ctx.user_input)
        if raw is None:
            return ExtractionResult.failed(("no script",), source=self.name)
        return ExtractionResult(
            extraction=parse_extraction(raw), raw=json.dumps(raw), valid=True, source=self.name
        )


def booking_turn(**delta: Any) -> dict[str, Any]:
    return {"turn_intent": "booking", "delta": delta, "confirmation": "none"}


def classifier_llm(rules: list[tuple[str, str]] | None = None) -> MockLLM:
    """Mock remote LLM: classify by keyword, answer consults with a fixed sentence."""
    llm = MockLLM(LLMConfig(provider="mock", model="mock-llm"))

    def handler(messages: list[Any]) -> Any:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "意图分类器" in system:
            for kw, intent in rules or DEFAULT_INTENTS:
                if kw in user:
                    return {"intent": intent}
            return {"intent": "other"}
        return llm.default if llm.default is not None else "好的。"

    llm.handler = handler
    return llm


DEFAULT_INTENTS = [
    ("约", "booking"),
    ("订单", "manage"),
    ("取消", "manage"),
    ("扣钱", "consult"),
    ("退多少", "consult"),
    ("几个人", "consult"),
]


@pytest.fixture
def make_classifier() -> Callable[[], MockLLM]:
    return classifier_llm


@dataclass
class AgentHarness:
    graph: Any
    deps: AgentDeps
    session_id: str = "s1"
    user_id: int = 1

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.session_id, "deps": self.deps}}

    async def say(self, text: str) -> dict[str, Any]:
        snapshot = await self.graph.aget_state(self.config)
        if snapshot.next:
            payload: Any = Command(resume=text)
        elif not snapshot.values:
            payload = {**initial_state(self.session_id, self.user_id), "user_input": text}
        else:
            payload = {"user_input": text}
        await self.graph.ainvoke(payload, self.config)
        return dict((await self.graph.aget_state(self.config)).values)

    async def state(self) -> dict[str, Any]:
        return dict((await self.graph.aget_state(self.config)).values)

    async def paused(self) -> bool:
        return bool((await self.graph.aget_state(self.config)).next)


@pytest.fixture
def make_agent(
    booking_server: Server[Any, Any], knowledge_server: Server[Any, Any], domain: DomainConfig
) -> Callable[..., AgentHarness]:
    def make(
        table: dict[str, dict[str, Any]],
        *,
        llm: MockLLM | None = None,
        checkpointer: Any = None,
        session_id: str = "s1",
        user_id: int = 1,
        server: Server[Any, Any] | None = None,
    ) -> AgentHarness:
        deps = AgentDeps(
            domain=domain,
            llm=llm or classifier_llm(),
            extractor=TableExtractor(table),
            booking=BookingClient(
                McpToolClient("booking", inprocess_connector(server or booking_server))
            ),
            knowledge=KnowledgeClient(
                McpToolClient("knowledge", inprocess_connector(knowledge_server)),
                collections=("platform_rules", "modes_and_ranks", "companion_profiles"),
            ),
            replies=ReplyRenderer(domain),
            clock=lambda: AGENT_NOW,
        )
        graph = build_graph(checkpointer or MemorySaver())
        return AgentHarness(graph, deps, session_id=session_id, user_id=user_id)

    return make


# --- web app -------------------------------------------------------------------------------


class FakeChatAgent:
    """Stands in for ``rift_agent.api.Agent`` in web tests: scripted events per input."""

    def __init__(self, script: dict[str, list[AgentEvent]] | None = None) -> None:
        self.script = script or {}
        self.calls: list[tuple[str, int, str]] = []
        self.histories: dict[str, list[dict[str, str]]] = {}

    async def run_turn_events(
        self, session_id: str, user_id: int, text: str
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append((session_id, user_id, text))
        events = self.script.get(text) or [
            AgentEvent("token", f"echo:{text}"),
            AgentEvent("done", {}),
        ]
        for event in events:
            yield event

    async def history(self, session_id: str) -> list[dict[str, str]]:
        return self.histories.get(session_id, [])


@pytest.fixture
def fake_chat_agent() -> type[FakeChatAgent]:
    return FakeChatAgent


@pytest.fixture
def make_web(
    booking_server: Server[Any, Any], domain: DomainConfig
) -> Callable[..., tuple[TestClient, WebServices]]:
    def make(
        agent: Any = None,
        *,
        server: Server[Any, Any] | None = None,
        secret: str = "test-secret",
    ) -> tuple[TestClient, WebServices]:
        services = WebServices(
            agent=agent or FakeChatAgent(),
            booking=BookingClient(
                McpToolClient("booking", inprocess_connector(server or booking_server))
            ),
            domain=domain,
            auth=Auth(secret),
            clock=lambda: AGENT_NOW,
        )
        app = create_app(WebConfig(secret=secret), services=services)
        return TestClient(app, follow_redirects=False), services

    return make


@pytest.fixture
def web_login() -> Callable[..., None]:
    def login(client: TestClient, nickname: str = "tester") -> None:
        resp = client.post("/login", data={"nickname": nickname})
        assert resp.status_code == 303, resp.text

    return login
