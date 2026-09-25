"""What the routes need, built once per process (or injected by tests)."""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from rift_agent.api import AgentEvent
from rift_agent.mcp_clients import BookingClient
from rift_domain.config import DomainConfig
from rift_web.auth import Auth

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]


class ChatAgent(Protocol):
    def run_turn_events(
        self, session_id: str, user_id: int, text: str
    ) -> AsyncIterator[AgentEvent]: ...

    async def history(self, session_id: str) -> list[dict[str, str]]: ...


@dataclass(frozen=True)
class WebConfig:
    settings_path: Path = REPO_ROOT / "config" / "settings.yaml"
    domain_path: Path = REPO_ROOT / "config" / "domain.yaml"
    #: Cookie signing key; set RIFT_WEB_SECRET in production (random per process otherwise).
    secret: str = field(default_factory=lambda: os.environ.get("RIFT_WEB_SECRET", ""))
    allowed_origins: tuple[str, ...] = ("http://localhost:8000", "http://127.0.0.1:8000")
    #: Run booking-mcp in-process on this SQLite file instead of calling it over HTTP.
    booking_db: str | None = None
    #: Fixed clock for demos / evaluation.
    now: datetime | None = None
    cookie_secure: bool = False


@dataclass
class WebServices:
    agent: ChatAgent
    booking: BookingClient
    domain: DomainConfig
    auth: Auth
    clock: Callable[[], datetime] = datetime.now
    cookie_secure: bool = False
    extras: dict[str, Any] = field(default_factory=dict)


def make_clock(now: datetime | None) -> Callable[[], datetime]:
    if now is None:
        return datetime.now
    fixed = now
    return lambda: fixed


def resolve_secret(config: WebConfig) -> str:
    if config.secret:
        return config.secret
    logger.warning("RIFT_WEB_SECRET is not set; using a random key (logins reset on restart)")
    return secrets.token_urlsafe(32)


@asynccontextmanager
async def build_services(config: WebConfig) -> AsyncIterator[WebServices]:
    """Real services from settings: DeepSeek + MCP clients + checkpointed agent."""
    from rift_agent.api import Agent
    from rift_agent.factory import build_deps, build_sinks
    from rift_agent.mcp_clients import McpToolClient, inprocess_connector
    from rift_common.settings import load_settings
    from rift_domain.config import load_domain_config

    settings = load_settings(config.settings_path)
    domain = load_domain_config(config.domain_path)
    clock = make_clock(config.now)
    booking: BookingClient | None = None
    if config.booking_db:
        from booking_mcp.server import build_server, build_service, parse_args

        argv = ["--db", config.booking_db, "--domain", str(config.domain_path), "--seed-if-empty"]
        if config.now:
            argv += ["--now", config.now.isoformat()]
        server = build_server(build_service(parse_args(argv)))
        booking = BookingClient(McpToolClient("booking", inprocess_connector(server)))
    deps = build_deps(settings, domain, booking=booking, clock=clock)
    async with AsyncExitStack() as stack:
        agent = await stack.enter_async_context(
            Agent.open(deps, settings.session.checkpoint_path, sinks=build_sinks(settings))
        )
        assert deps.booking is not None
        yield WebServices(
            agent=agent,
            booking=deps.booking,
            domain=domain,
            auth=Auth(resolve_secret(config)),
            clock=clock,
            cookie_secure=config.cookie_secure,
        )
