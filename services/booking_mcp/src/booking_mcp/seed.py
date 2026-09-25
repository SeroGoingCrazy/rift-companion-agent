"""Deterministic seed data: ~30 companions and 14 days of schedules.

Everything is drawn from ``random.Random(seed)`` in a fixed order, so the same
``(seed, start_date)`` always produces the same database (companion ids 1..N included).
Guarantees checked by the tests:

- every game mode is offered by at least ``MIN_PER_MODE`` companions;
- every solo/duo rank band ``[r, r + max_tier_gap]`` contains a companion;
- each companion's modes come with the service types those modes default to.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal

from sqlalchemy import Engine

from booking_mcp.db.models import Companion, Schedule, ScheduleStatus, User
from booking_mcp.db.session import SessionFactory, reset_db, write_session
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType

DEFAULT_SEED = 20261001
DEFAULT_DAYS = 14
MIN_PER_MODE = 5
DEMO_USERS = ("demo",)

# Fictional nicknames; order is part of the seed (ids follow it).
NAMES = (
    "阿狸酱", "夜雨声烦", "小鹿乱撞", "青柠汽水", "北城以北", "烤红薯",
    "月下独酌", "橘子味的猫", "星河滚烫", "奶茶三分糖", "峡谷清风", "不吃香菜",
    "云边小卖部", "晚风与你", "山海皆可平", "一只小熊软糖", "长安少年", "糖醋排骨",
    "白桃乌龙", "雪落无声", "南风知我意", "咕噜咕噜", "野区守望者", "薄荷微凉",
    "海盐芝士", "逆风翻盘王", "半糖主义", "日落飞行", "柚子茶", "林间有鹿",
)  # fmt: skip

# How many companions sit on each rank (low -> high); sums to len(NAMES).
RANK_COUNTS: dict[Rank, int] = {
    Rank.IRON: 1,
    Rank.BRONZE: 2,
    Rank.SILVER: 3,
    Rank.GOLD: 4,
    Rank.PLATINUM: 4,
    Rank.EMERALD: 4,
    Rank.DIAMOND: 4,
    Rank.MASTER: 3,
    Rank.GRANDMASTER: 3,
    Rank.CHALLENGER: 2,
}

STYLE_TAGS = (
    "温柔", "幽默", "话痨", "高冷", "技术流", "耐心", "会教学", "声音好听",
    "佛系", "细心", "指挥型", "稳健", "激进", "开朗", "安静", "会哄人",
)  # fmt: skip

# Probability a companion takes each optional mode (normal_draft is offered by everyone,
# solo/duo by everyone silver+).
MODE_ODDS: dict[GameMode, float] = {
    GameMode.RANKED_FLEX: 0.5,
    GameMode.ARAM: 0.5,
    GameMode.ARAM_MAYHEM: 0.4,
    GameMode.ARENA: 0.35,
}

# (start hour, length in hours) of a working window; evening shifts cross midnight.
SHIFTS = ((13, 6), (19, 6), (14, 10), (20, 5))


@dataclass(frozen=True)
class CompanionSeed:
    name: str
    gender: Gender
    rank: Rank
    roles: tuple[Role, ...]
    modes: tuple[GameMode, ...]
    service_types: tuple[ServiceType, ...]
    hourly_price: Decimal
    voice: bool
    rating: float
    tags: tuple[str, ...]
    bio: str


@dataclass(frozen=True)
class ScheduleSeed:
    #: 0-based index into the companion list (id - 1 once inserted).
    companion_index: int
    start: datetime
    end: datetime
    status: ScheduleStatus


@dataclass(frozen=True)
class SeedSummary:
    companions: int
    schedules: int
    blocked: int
    users: int
    start_date: date
    days: int


def _bio(
    rank: Rank,
    roles: Sequence[Role],
    modes: Sequence[GameMode],
    tags: Sequence[str],
    domain: DomainConfig,
) -> str:
    role_text = "/".join(domain.roles[r].label for r in roles)
    mode_text = "、".join(domain.game_modes[m].label for m in modes)
    return (
        f"{domain.rank_label(rank)}段位，主玩{role_text}。性格{'、'.join(tags)}，可接{mode_text}。"
    )


@dataclass
class _Draft:
    name: str
    gender: Gender
    rank: Rank
    roles: list[Role]
    modes: set[GameMode]
    coaching: bool
    hourly_price: Decimal
    voice: bool
    rating: float
    tags: list[str]


def generate_companions(domain: DomainConfig, seed: int = DEFAULT_SEED) -> list[CompanionSeed]:
    rng = random.Random(seed)
    ranks = [rank for rank, n in RANK_COUNTS.items() for _ in range(n)]
    rng.shuffle(ranks)
    silver = domain.rank_index(Rank.SILVER)
    diamond = domain.rank_index(Rank.DIAMOND)

    drafts: list[_Draft] = []
    for name, rank in zip(NAMES, ranks, strict=True):
        tier = domain.rank_index(rank)
        modes = {GameMode.NORMAL_DRAFT}
        if tier >= silver:
            modes.add(GameMode.RANKED_SOLO_DUO)
        modes.update(m for m, p in MODE_ODDS.items() if rng.random() < p)
        drafts.append(
            _Draft(
                name=name,
                gender=rng.choice(list(Gender)),
                rank=rank,
                roles=rng.sample(list(Role), k=rng.choice((1, 2, 2, 3))),
                modes=modes,
                coaching=tier >= diamond and rng.random() < 0.6,
                hourly_price=Decimal(30 + tier * 8 + rng.choice((-5, 0, 0, 5, 10))),
                voice=rng.random() < 0.85,
                rating=round(rng.uniform(4.2, 5.0), 1),
                tags=rng.sample(STYLE_TAGS, k=2),
            )
        )

    # Top up thin modes so every mode has enough companions to choose from.
    for mode in GameMode:
        takers = [d for d in drafts if mode in d.modes]
        others = [d for d in drafts if mode not in d.modes]
        for d in rng.sample(others, k=max(0, MIN_PER_MODE + 1 - len(takers))):
            d.modes.add(mode)

    companions: list[CompanionSeed] = []
    for d in drafts:
        modes_t = tuple(m for m in GameMode if m in d.modes)
        services = {domain.default_service_type(m) for m in modes_t}
        if d.coaching:
            services.add(ServiceType.COACHING)
        roles_t = tuple(r for r in Role if r in d.roles)
        companions.append(
            CompanionSeed(
                name=d.name,
                gender=d.gender,
                rank=d.rank,
                roles=roles_t,
                modes=modes_t,
                service_types=tuple(s for s in ServiceType if s in services),
                hourly_price=d.hourly_price,
                voice=d.voice,
                rating=d.rating,
                tags=tuple(d.tags),
                bio=_bio(d.rank, roles_t, modes_t, d.tags, domain),
            )
        )
    return companions


def generate_schedules(
    companion_count: int,
    start_date: date,
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
) -> list[ScheduleSeed]:
    """Open windows per working day, with some hours already blocked inside them."""
    rng = random.Random(seed + 1)
    rows: list[ScheduleSeed] = []
    for idx in range(companion_count):
        for offset in range(days):
            if rng.random() < 0.2:  # day off
                continue
            day = start_date + timedelta(days=offset)
            start_hour, length = rng.choice(SHIFTS)
            opens = datetime.combine(day, time(start_hour))
            closes = opens + timedelta(hours=length)
            rows.append(ScheduleSeed(idx, opens, closes, ScheduleStatus.OPEN))
            if rng.random() < 0.3:
                busy_from = opens + timedelta(hours=rng.randrange(0, length - 1))
                busy_hours = rng.choice((1, 2))
                busy_to = min(busy_from + timedelta(hours=busy_hours), closes)
                rows.append(ScheduleSeed(idx, busy_from, busy_to, ScheduleStatus.BLOCKED))
    return rows


def to_model(seed: CompanionSeed, domain: DomainConfig) -> Companion:
    c = Companion(
        name=seed.name,
        gender=seed.gender,
        rank=seed.rank,
        rank_tier=domain.rank_index(seed.rank),
        hourly_price=seed.hourly_price,
        voice=seed.voice,
        rating=seed.rating,
        tags=list(seed.tags),
        bio=seed.bio,
    )
    c.set_skills(seed.roles, seed.modes, seed.service_types)
    return c


def seed_database(
    engine: Engine,
    factory: SessionFactory,
    domain: DomainConfig,
    *,
    start_date: date,
    days: int = DEFAULT_DAYS,
    seed: int = DEFAULT_SEED,
) -> SeedSummary:
    """Drop everything and write a fresh, deterministic data set."""
    reset_db(engine)
    companions = generate_companions(domain, seed)
    schedules = generate_schedules(len(companions), start_date, days, seed)
    with write_session(factory) as s:
        models = [to_model(c, domain) for c in companions]
        s.add_all(models)
        s.add_all(User(nickname=n) for n in DEMO_USERS)
        s.flush()
        s.add_all(
            Schedule(
                companion_id=models[row.companion_index].id,
                start=row.start,
                end=row.end,
                status=row.status,
            )
            for row in schedules
        )
    return SeedSummary(
        companions=len(companions),
        schedules=len(schedules),
        blocked=sum(r.status is ScheduleStatus.BLOCKED for r in schedules),
        users=len(DEMO_USERS),
        start_date=start_date,
        days=days,
    )
