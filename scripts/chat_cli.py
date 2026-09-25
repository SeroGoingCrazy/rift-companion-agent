"""Chat with the agent in a terminal (M1 demo).

Needs ``DEEPSEEK_API_KEY`` (remote LLM for classification, slot extraction and consult
answers). MCP servers come from ``config/settings.yaml`` unless overridden:

    uv run python scripts/chat_cli.py                               # HTTP MCP servers
    uv run python scripts/chat_cli.py --booking-db data/booking.db  # booking in-process
    uv run python scripts/chat_cli.py --session s1 --user-id 1      # continue a session

Commands: ``/new`` starts a new session, ``/state`` prints the booking slots, ``/quit``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from rift_agent.api import Agent
from rift_agent.factory import build_deps, build_sinks
from rift_agent.mcp_clients import BookingClient, McpToolClient, inprocess_connector
from rift_common.settings import load_settings
from rift_domain.config import load_domain_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _booking_inprocess(db: str, domain_path: str, now: datetime | None) -> BookingClient:
    from booking_mcp.server import build_server, build_service, parse_args

    args = parse_args(["--db", db, "--domain", domain_path, "--seed-if-empty"]
                       + (["--now", now.isoformat()] if now else []))  # fmt: skip
    return BookingClient(
        McpToolClient("booking", inprocess_connector(build_server(build_service(args))))
    )


async def chat(args: argparse.Namespace) -> int:
    settings = load_settings(args.settings)
    domain = load_domain_config(args.domain)
    clock = (lambda: args.now) if args.now else datetime.now
    booking = (
        _booking_inprocess(args.booking_db, args.domain, args.now) if args.booking_db else None
    )
    deps = build_deps(settings, domain, booking=booking, clock=clock)
    session = args.session or uuid.uuid4().hex[:8]
    print(f"会话 {session}（用户 {args.user_id}）。/new 新会话，/state 查看槽位，/quit 退出。")

    async with Agent.open(
        deps, settings.session.checkpoint_path, sinks=build_sinks(settings)
    ) as agent:
        while True:
            try:
                text = (await asyncio.to_thread(input, "你> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text == "/quit":
                return 0
            if text == "/new":
                session = uuid.uuid4().hex[:8]
                print(f"—— 新会话 {session} ——")
                continue
            if text == "/state":
                state = await agent.state(session)
                print({k: state.get(k) for k in ("phase", "pending_action", "booking")})
                continue
            print("助手> ", end="", flush=True)
            async for piece in agent.run_turn(session, args.user_id, text):
                print(piece, end="", flush=True)
            print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", default=str(REPO_ROOT / "config" / "settings.yaml"))
    parser.add_argument("--domain", default=str(REPO_ROOT / "config" / "domain.yaml"))
    parser.add_argument("--session", default=None)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--booking-db", default=None, help="run booking-mcp in-process on this DB")
    parser.add_argument("--now", type=datetime.fromisoformat, default=None, help="fixed clock")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr)
    return asyncio.run(chat(args))


if __name__ == "__main__":
    sys.exit(main())
