"""Create the booking database with deterministic seed data.

Usage::

    uv run python scripts/seed_all.py --reset                 # data/booking.db, from today
    uv run python scripts/seed_all.py --reset --start 2026-10-01 --db data/booking.db

Without ``--reset`` an existing, non-empty database is left untouched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from sqlalchemy import func, inspect, select

from booking_mcp.db import Companion, create_db_engine, make_session_factory, read_session
from booking_mcp.seed import DEFAULT_DAYS, DEFAULT_SEED, seed_database
from rift_domain.config import load_domain_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "booking.db"))
    parser.add_argument("--domain", default=str(REPO_ROOT / "config" / "domain.yaml"))
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reset", action="store_true", help="drop and recreate all tables")
    args = parser.parse_args(argv)

    engine = create_db_engine(args.db)
    factory = make_session_factory(engine)
    if not args.reset and inspect(engine).has_table("companions"):
        with read_session(factory) as s:
            count = s.scalar(select(func.count()).select_from(Companion)) or 0
        if count:
            print(f"{args.db} already has {count} companions; pass --reset to reseed.")
            return 0

    summary = seed_database(
        engine,
        factory,
        load_domain_config(args.domain),
        start_date=args.start or date.today(),
        days=args.days,
        seed=args.seed,
    )
    print(
        f"seeded {args.db}: {summary.companions} companions, {summary.schedules} schedule rows "
        f"({summary.blocked} blocked), {summary.users} users, "
        f"{summary.days} days from {summary.start_date}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
