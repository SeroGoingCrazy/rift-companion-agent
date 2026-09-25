"""Drive booking-mcp over stdio the way Claude Desktop does: find -> quote -> book -> list.

Usage::

    uv run python scripts/mcp_smoke.py                    # temp DB seeded from --now's date
    uv run python scripts/mcp_smoke.py --db data/booking.db --now 2026-10-01T14:00

Exits non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _payload(result: Any) -> dict[str, Any]:
    data = result.structuredContent or json.loads(result.content[0].text)
    if result.isError:
        raise RuntimeError(f"tool error: {data['error']}")
    return dict(data)


async def run(db: Path, now: datetime, user_id: int) -> dict[str, Any]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "booking_mcp", "--transport", "stdio", "--db", str(db),
              "--seed-if-empty", "--now", now.isoformat(), "--log-level", "WARNING"],
    )  # fmt: skip
    start = (now + timedelta(days=1)).replace(hour=20, minute=0, second=0, microsecond=0)
    # errlog is passed explicitly: the default is bound to sys.stderr at import time.
    async with (
        stdio_client(params, errlog=sys.stderr) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = [t.name for t in (await session.list_tools()).tools]
        print("tools:", ", ".join(tools))

        found = _payload(
            await session.call_tool(
                "find_companions",
                {"game_mode": "aram", "start_time": start.isoformat(), "duration_hours": 2},
            )
        )
        if not found["candidates"]:
            raise RuntimeError("find_companions returned no candidates")
        pick = found["candidates"][0]
        print(f"found {len(found['candidates'])}: first = {pick['name']} at {pick['start_time']}")

        quote = _payload(
            await session.call_tool(
                "quote_price",
                {
                    "companion_id": pick["companion_id"],
                    "service_type": found["service_type"],
                    "duration_hours": 2,
                },
            )
        )
        print(f"quote: {quote['unit_price']} x {quote['multiplier']} x 2h = {quote['total']}")

        booking = _payload(
            await session.call_tool(
                "create_booking",
                {
                    "user_id": user_id,
                    "companion_id": pick["companion_id"],
                    "start_time": pick["start_time"],
                    "duration_hours": 2,
                    "game_mode": "aram",
                },
            )
        )
        print(f"booked #{booking['booking_id']}: {booking['status']} total {booking['total']}")

        listed = _payload(await session.call_tool("list_my_bookings", {"user_id": user_id}))
        refund = _payload(
            await session.call_tool(
                "cancel_booking", {"user_id": user_id, "booking_id": booking["booking_id"]}
            )
        )
        print(
            f"bookings: {len(listed['bookings'])}; cancel now -> {refund['refund_label']} "
            f"(policy {refund['policy_refund_amount']}, paid={refund['paid']})"
        )
        return {"tools": tools, "booking": booking, "bookings": listed["bookings"]}


def _root_cause(exc: BaseException) -> BaseException:
    """The transport wraps errors in exception groups; report the original one."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=None, help="default: a fresh temp DB")
    parser.add_argument("--now", type=datetime.fromisoformat, default=datetime(2026, 10, 1, 14))
    parser.add_argument("--user-id", type=int, default=1, help="seeded demo user is 1")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        db = args.db or Path(tmp) / "booking.db"
        try:
            asyncio.run(run(db, args.now, args.user_id))
        except Exception as exc:
            print(f"FAILED: {_root_cause(exc)}", file=sys.stderr)
            return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
