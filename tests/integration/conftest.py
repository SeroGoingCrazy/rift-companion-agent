"""Integration fixtures: a fresh file-backed booking SQLite per test, small builders for
hand-made companions and schedules, and in-process MCP servers for agent tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import mcp.types as mcp_types
import pytest
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
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType


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
