"""D4: find_companions (filter, availability, style ranking, relaxation) and quote_price."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from booking_mcp.db import ScheduleStatus
from booking_mcp.errors import InvalidArgument, NotFound
from booking_mcp.seed import seed_database
from booking_mcp.service import BookingService, style_text
from booking_mcp.tools.find_companions import find_companions
from booking_mcp.tools.quote_price import quote_price
from booking_mcp.tools.schemas import FindCompanionsInput, QuotePriceInput
from rift_common.embedding import EmbeddingError, MockEmbedding
from rift_common.settings import EmbeddingConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_domain.matching import CompanionProfile

if TYPE_CHECKING:
    from tests.integration.conftest import BookingDB

NOW = datetime(2026, 10, 1, 14, 0)  # Thursday
SAT_20 = datetime(2026, 10, 3, 20, 0)  # Saturday 20:00


def _service(db: BookingDB, **kw: Any) -> BookingService:
    return BookingService(db.factory, db.domain, clock=lambda: NOW, **kw)


def _args(**overrides: Any) -> FindCompanionsInput:
    fields: dict[str, Any] = {
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "start_time": SAT_20,
        "duration_hours": 2,
    }
    fields.update(overrides)
    return FindCompanionsInput(**fields)


@pytest.fixture
def saturday(booking_db: BookingDB) -> dict[str, int]:
    """Nobody female + diamond is free at Saturday 20:00; one is free from 21:00."""
    db = booking_db
    ids = {
        "female_dia_late": db.add_companion(
            "female_dia_late", gender=Gender.FEMALE, rank=Rank.DIAMOND, roles=(Role.SUPPORT,)
        ),
        "male_dia": db.add_companion(
            "male_dia", gender=Gender.MALE, rank=Rank.DIAMOND, roles=(Role.JUNGLE,)
        ),
        "female_gold": db.add_companion(
            "female_gold", gender=Gender.FEMALE, rank=Rank.GOLD, roles=(Role.SUPPORT,)
        ),
        "female_dia_busy": db.add_companion(
            "female_dia_busy", gender=Gender.FEMALE, rank=Rank.DIAMOND, roles=(Role.MID,)
        ),
    }
    day = SAT_20.replace(hour=0)
    db.add_schedule(ids["female_dia_late"], day.replace(hour=21), day + timedelta(hours=26))
    db.add_schedule(ids["male_dia"], day.replace(hour=18), day + timedelta(hours=24))
    db.add_schedule(ids["female_gold"], day.replace(hour=18), day + timedelta(hours=24))
    db.add_schedule(ids["female_dia_busy"], day.replace(hour=18), day + timedelta(hours=24))
    # Busy all evening: even a ±1h shift does not help.
    db.add_schedule(
        ids["female_dia_busy"], day.replace(hour=18), day + timedelta(hours=24),
        ScheduleStatus.BLOCKED,
    )  # fmt: skip
    return ids


def test_female_diamond_saturday_relaxes_time_window(
    booking_db: BookingDB, saturday: dict[str, int]
) -> None:
    out = find_companions(
        _service(booking_db),
        _args(companion_gender=Gender.FEMALE, rank_requirement=Rank.DIAMOND),
    )
    assert out.relaxations == ["time_window"]
    assert out.relaxation_notes == ["开始时间放宽到前后 1 小时"]
    assert out.relaxations_tried == []
    [cand] = out.candidates
    assert cand.name == "female_dia_late"
    assert cand.start_time == SAT_20 + timedelta(hours=1)
    assert out.service_type is ServiceType.CLIMB


def test_exact_match_needs_no_relaxation(booking_db: BookingDB, saturday: dict[str, int]) -> None:
    out = find_companions(
        _service(booking_db), _args(companion_gender=Gender.MALE, rank_requirement=Rank.DIAMOND)
    )
    assert out.relaxations == []
    assert [c.name for c in out.candidates] == ["male_dia"]
    assert out.candidates[0].start_time == SAT_20


def test_gender_relaxed_after_time(booking_db: BookingDB, saturday: dict[str, int]) -> None:
    out = find_companions(
        _service(booking_db),
        _args(
            companion_gender=Gender.FEMALE,
            rank_requirement=Rank.DIAMOND,
            start_time=SAT_20 - timedelta(hours=1),  # 19:00: late one is 2h away
        ),
    )
    assert out.relaxations == ["time_window", "companion_gender"]
    assert [c.name for c in out.candidates] == ["male_dia"]


def test_budget_is_never_relaxed(booking_db: BookingDB, saturday: dict[str, int]) -> None:
    out = find_companions(
        _service(booking_db),
        _args(companion_gender=Gender.FEMALE, rank_requirement=Rank.DIAMOND, budget_per_hour=10),
    )
    assert out.candidates == []
    assert out.relaxations == []
    assert out.relaxations_tried == ["time_window", "companion_gender"]


def test_rank_is_never_relaxed(booking_db: BookingDB, saturday: dict[str, int]) -> None:
    out = find_companions(_service(booking_db), _args(rank_requirement=Rank.MASTER))
    assert out.candidates == []
    assert out.relaxations_tried == ["time_window"]


def test_budget_compares_effective_price(booking_db: BookingDB, saturday: dict[str, int]) -> None:
    # Base price 50 x climb 1.2 = 60/h.
    svc = _service(booking_db)
    assert (
        find_companions(svc, _args(companion_name="male_dia", budget_per_hour=59)).candidates == []
    )
    [c] = find_companions(svc, _args(companion_name="male_dia", budget_per_hour=60)).candidates
    assert (c.hourly_price, c.effective_hourly_price) == (Decimal("50.00"), Decimal("60.00"))


def test_not_applicable_rank_ignored_for_aram(
    booking_db: BookingDB, saturday: dict[str, int]
) -> None:
    out = find_companions(
        _service(booking_db),
        _args(game_mode=GameMode.ARAM, rank_requirement=Rank.CHALLENGER, role_preference=["top"]),
    )
    assert out.service_type is ServiceType.CASUAL
    assert {c.name for c in out.candidates} == {"male_dia", "female_gold"}
    assert out.relaxations == []


def test_role_preference_ranks_and_explains(
    booking_db: BookingDB, saturday: dict[str, int]
) -> None:
    out = find_companions(
        _service(booking_db), _args(game_mode=GameMode.NORMAL_DRAFT, role_preference=["jungle"])
    )
    assert [c.name for c in out.candidates] == ["male_dia"]
    assert "擅长打野" in out.candidates[0].reasons


def test_style_similarity_reorders(booking_db: BookingDB) -> None:
    db = booking_db
    gentle = db.add_companion("gentle", tags=["温柔", "会聊天"], bio="温柔会聊天的辅助", rating=4.2)
    loud = db.add_companion("loud", tags=["暴躁", "激进"], bio="激进打法指挥", rating=5.0)
    for cid in (gentle, loud):
        db.add_schedule(cid, SAT_20 - timedelta(hours=2), SAT_20 + timedelta(hours=4))
    embedder = MockEmbedding(EmbeddingConfig(provider="mock"))
    svc = _service(db, embedder=embedder)
    base = find_companions(svc, _args(game_mode=GameMode.ARAM))
    styled = find_companions(svc, _args(game_mode=GameMode.ARAM, style_preference="温柔会聊天"))
    assert [c.name for c in base.candidates] == ["loud", "gentle"]  # rating only
    assert [c.name for c in styled.candidates] == ["gentle", "loud"]
    assert any("风格贴合" in r for r in styled.candidates[0].reasons)


def test_style_ignored_without_embedder_or_on_failure(booking_db: BookingDB) -> None:
    class Broken(MockEmbedding):
        def embed(self, texts: Any) -> Any:
            raise EmbeddingError("down")

    profile = CompanionProfile(
        id=1, name="x", gender=Gender.MALE, rank=Rank.GOLD, roles=(), modes=(),
        service_types=(), hourly_price=Decimal(1), voice=True, rating=5, tags=("a",), bio="b",
    )  # fmt: skip
    assert _service(booking_db).style_scores("温柔", [profile]) is None
    broken = _service(booking_db, embedder=Broken(EmbeddingConfig(provider="mock")))
    assert broken.style_scores("温柔", [profile]) is None
    assert style_text(profile) == "a b"


def test_style_vectors_are_cached(booking_db: BookingDB) -> None:
    calls: list[int] = []

    class Counting(MockEmbedding):
        def embed(self, texts: Any) -> Any:
            calls.append(len(texts))
            return super().embed(texts)

    profile = CompanionProfile(
        id=1, name="x", gender=Gender.MALE, rank=Rank.GOLD, roles=(), modes=(),
        service_types=(), hourly_price=Decimal(1), voice=True, rating=5, tags=("a",), bio="b",
    )  # fmt: skip
    svc = _service(booking_db, embedder=Counting(EmbeddingConfig(provider="mock")))
    svc.style_scores("温柔", [profile])
    svc.style_scores("幽默", [profile])
    assert calls == [1, 1, 1]  # companion once, then only the two queries


def test_top_k_limits_results(booking_db: BookingDB) -> None:
    for i in range(5):
        cid = booking_db.add_companion(f"c{i}", rating=4.0 + i / 10)
        booking_db.add_schedule(cid, SAT_20, SAT_20 + timedelta(hours=3))
    svc = _service(booking_db)
    default = find_companions(svc, _args(game_mode=GameMode.ARAM))
    assert len(default.candidates) == booking_db.domain.matching.top_k
    assert [c.name for c in default.candidates] == ["c4", "c3", "c2"]
    assert len(find_companions(svc, _args(game_mode=GameMode.ARAM, top_k=5)).candidates) == 5


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"start_time": NOW - timedelta(hours=1)}, "past"),
        ({"duration_hours": 9}, "duration_hours"),
    ],
)
def test_invalid_search_arguments(
    booking_db: BookingDB, overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(InvalidArgument, match=match):
        find_companions(_service(booking_db), _args(**overrides))


def test_input_model_is_strict() -> None:
    with pytest.raises(ValidationError):
        _args(game_mode="lol_chess")
    with pytest.raises(ValidationError):
        _args(duration_hours=1.3)
    with pytest.raises(ValidationError):
        _args(unknown=1)


def test_search_against_seed_data(booking_db: BookingDB) -> None:
    seed_database(
        booking_db.engine, booking_db.factory, booking_db.domain, start_date=date(2026, 10, 1)
    )
    out = find_companions(
        _service(booking_db), _args(rank_requirement=Rank.GOLD, start_time=SAT_20)
    )
    assert 1 <= len(out.candidates) <= booking_db.domain.matching.top_k
    for c in out.candidates:
        assert booking_db.domain.rank_index(Rank.GOLD) <= booking_db.domain.rank_index(c.rank)
        assert booking_db.domain.rank_index(c.rank) <= booking_db.domain.rank_index(Rank.EMERALD)
    scores = [c.score for c in out.candidates]
    assert scores == sorted(scores, reverse=True)


# --- quote_price ---------------------------------------------------------------------------


def test_quote_price(booking_db: BookingDB, saturday: dict[str, int]) -> None:
    out = quote_price(
        _service(booking_db),
        QuotePriceInput(
            companion_id=saturday["male_dia"], service_type=ServiceType.COACHING, duration_hours=2.5
        ),
    )
    assert out.companion_name == "male_dia"
    assert (out.unit_price, out.multiplier, out.hours) == (
        Decimal("50.00"),
        Decimal("1.5"),
        Decimal("2.5"),
    )
    assert (out.effective_hourly, out.total) == (Decimal("75.00"), Decimal("187.50"))
    assert out.model_dump(mode="json")["total"] == "187.50"


def test_quote_price_errors(booking_db: BookingDB) -> None:
    svc = _service(booking_db)
    cid = booking_db.add_companion("casual_only", service_types=(ServiceType.CASUAL,))
    with pytest.raises(NotFound):
        quote_price(svc, QuotePriceInput(companion_id=999, service_type="casual", duration_hours=1))
    with pytest.raises(InvalidArgument, match="does not offer climb") as err:
        quote_price(svc, QuotePriceInput(companion_id=cid, service_type="climb", duration_hours=1))
    assert err.value.details["offered"] == ["casual"]
    with pytest.raises(InvalidArgument, match="duration"):
        quote_price(
            svc, QuotePriceInput(companion_id=cid, service_type="casual", duration_hours=10)
        )
