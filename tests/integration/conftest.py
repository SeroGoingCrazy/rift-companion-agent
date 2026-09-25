"""Fixtures for booking-mcp integration tests: a fresh file-backed SQLite per test and
small builders for hand-made companions and schedules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
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
