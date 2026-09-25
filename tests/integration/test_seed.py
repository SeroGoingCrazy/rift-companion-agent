"""D2: deterministic seed data with coverage guarantees."""

from __future__ import annotations

import importlib.util
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import Engine, select

from booking_mcp.db import (
    Companion,
    Schedule,
    ScheduleStatus,
    User,
    create_db_engine,
    make_session_factory,
    read_session,
)
from booking_mcp.seed import (
    DEMO_USERS,
    MIN_PER_MODE,
    NAMES,
    CompanionSeed,
    generate_companions,
    generate_schedules,
    seed_database,
)
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Rank, ServiceType

REPO_ROOT = Path(__file__).resolve().parents[2]
START = date(2026, 10, 1)


@pytest.fixture(scope="module")
def companions(domain: DomainConfig) -> list[CompanionSeed]:
    return generate_companions(domain)


def test_about_thirty_unique_companions(companions: list[CompanionSeed]) -> None:
    assert len(companions) == len(NAMES) == 30
    assert len({c.name for c in companions}) == 30


@pytest.mark.parametrize("mode", list(GameMode))
def test_every_mode_has_enough_companions(companions: list[CompanionSeed], mode: GameMode) -> None:
    assert sum(mode in c.modes for c in companions) >= MIN_PER_MODE


@pytest.mark.parametrize("rank", list(Rank))
def test_every_solo_duo_rank_band_is_covered(
    companions: list[CompanionSeed], domain: DomainConfig, rank: Rank
) -> None:
    gap = domain.max_tier_gap_for(GameMode.RANKED_SOLO_DUO)
    assert gap is not None
    lo, hi = domain.rank_index(rank), domain.rank_index(domain.rank_offset(rank, gap))
    in_band = [
        c
        for c in companions
        if GameMode.RANKED_SOLO_DUO in c.modes
        and ServiceType.CLIMB in c.service_types
        and lo <= domain.rank_index(c.rank) <= hi
    ]
    # Solo/duo starts at silver, which still falls inside the iron and bronze bands.
    assert in_band, f"no solo/duo companion for band starting at {rank}"


def test_every_rank_is_present(companions: list[CompanionSeed]) -> None:
    assert {c.rank for c in companions} == set(Rank)


def test_modes_come_with_their_default_service_type(
    companions: list[CompanionSeed], domain: DomainConfig
) -> None:
    for c in companions:
        for mode in c.modes:
            assert domain.default_service_type(mode) in c.service_types, c.name


def test_coaching_only_for_high_ranks(
    companions: list[CompanionSeed], domain: DomainConfig
) -> None:
    coaches = [c for c in companions if ServiceType.COACHING in c.service_types]
    assert coaches
    assert all(domain.rank_index(c.rank) >= domain.rank_index(Rank.DIAMOND) for c in coaches)


def test_attributes_are_plausible(companions: list[CompanionSeed], domain: DomainConfig) -> None:
    for c in companions:
        assert 1 <= len(c.roles) <= 3
        assert 4.2 <= c.rating <= 5.0
        assert c.hourly_price > 0
        assert len(c.tags) == 2
        assert domain.rank_label(c.rank) in c.bio
    genders = Counter(c.gender for c in companions)
    assert len(genders) == 2
    assert sum(c.voice for c in companions) >= 20
    # Higher ranks cost more on average.
    low = [c.hourly_price for c in companions if domain.rank_index(c.rank) <= 2]
    high = [c.hourly_price for c in companions if domain.rank_index(c.rank) >= 7]
    assert sum(low) / len(low) < sum(high) / len(high)


def test_generation_is_deterministic(domain: DomainConfig) -> None:
    assert generate_companions(domain) == generate_companions(domain)
    assert generate_companions(domain, seed=1) != generate_companions(domain)
    assert generate_schedules(30, START) == generate_schedules(30, START)


def test_schedules_span_the_window_with_some_blocked() -> None:
    rows = generate_schedules(30, START, days=14)
    open_rows = [r for r in rows if r.status is ScheduleStatus.OPEN]
    blocked = [r for r in rows if r.status is ScheduleStatus.BLOCKED]
    assert blocked, "seed must include already-taken time"
    first = datetime.combine(START, datetime.min.time())
    assert min(r.start for r in rows) >= first
    assert max(r.start for r in rows) < first + timedelta(days=14)
    # Each blocked range lies inside an open window of the same companion.
    for b in blocked:
        assert any(
            o.companion_index == b.companion_index and o.start <= b.start and b.end <= o.end
            for o in open_rows
        )
    # At most one open window per companion per day.
    per_day = Counter((r.companion_index, r.start.date()) for r in open_rows)
    assert max(per_day.values()) == 1


def _dump(engine: Engine) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    factory = make_session_factory(engine)
    with read_session(factory) as s:
        comps = [
            (c.id, c.name, c.rank, c.hourly_price, c.roles, c.modes, c.service_types, c.bio)
            for c in s.scalars(select(Companion).order_by(Companion.id))
        ]
        scheds = [
            (r.companion_id, r.start, r.end, r.status)
            for r in s.scalars(select(Schedule).order_by(Schedule.id))
        ]
    return comps, scheds


def test_seed_database_is_repeatable(tmp_path: Path, domain: DomainConfig) -> None:
    engine = create_db_engine(tmp_path / "booking.db")
    factory = make_session_factory(engine)
    summary = seed_database(engine, factory, domain, start_date=START)
    first = _dump(engine)
    seed_database(engine, factory, domain, start_date=START)  # reseed drops old rows
    assert _dump(engine) == first
    assert summary.companions == 30
    assert summary.schedules == len(first[1])
    assert summary.blocked > 0
    assert [c[1] for c in first[0]] == list(NAMES)
    with read_session(factory) as s:
        assert s.scalars(select(User.nickname)).all() == list(DEMO_USERS)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seed_all", REPO_ROOT / "scripts" / "seed_all.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seed_script_reset_and_skip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    db = tmp_path / "script.db"
    assert script.main(["--db", str(db), "--reset", "--start", "2026-10-01"]) == 0
    assert "30 companions" in capsys.readouterr().out
    assert script.main(["--db", str(db), "--start", "2026-10-01"]) == 0
    assert "already has 30 companions" in capsys.readouterr().out
    engine = create_db_engine(db)
    assert len(_dump(engine)[0]) == 30


def test_seed_script_seeds_empty_database(tmp_path: Path) -> None:
    script = _load_script()
    db = tmp_path / "fresh.db"
    assert script.main(["--db", str(db), "--start", "2026-10-01", "--days", "3"]) == 0
    scheds = _dump(create_db_engine(db))[1]
    assert {row[1].date() for row in scheds} <= {START + timedelta(days=i) for i in range(4)}
