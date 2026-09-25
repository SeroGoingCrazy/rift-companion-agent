"""G4: companion list — same hard filter as find_companions, free slots, web page."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from booking_mcp.db import read_session
from booking_mcp.db.repo import CompanionRepo, ScheduleRepo, to_profile
from booking_mcp.service import BookingService
from booking_mcp.tools.browse_companions import browse_companions
from booking_mcp.tools.find_companions import find_companions
from booking_mcp.tools.schemas import BrowseCompanionsInput, FindCompanionsInput
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_domain.matching import build_filters, matches
from rift_domain.pricing import to_money
from rift_domain.rules import apply_rules
from rift_domain.slots import BookingState
from rift_web.services import WebServices

Make = Callable[..., tuple[TestClient, WebServices]]
NOW = datetime(2026, 10, 1, 14, 0)

COMBOS: list[dict[str, Any]] = [
    {"game_mode": GameMode.RANKED_SOLO_DUO, "rank_requirement": Rank.DIAMOND},
    {"game_mode": GameMode.RANKED_SOLO_DUO, "rank_requirement": Rank.GOLD, "role": Role.JUNGLE},
    {"game_mode": GameMode.RANKED_FLEX, "rank_requirement": Rank.PLATINUM},
    {"game_mode": GameMode.ARAM, "companion_gender": Gender.FEMALE},
    {"game_mode": GameMode.ARENA, "rank_requirement": Rank.CHALLENGER},  # rank n/a -> ignored
    {"game_mode": GameMode.NORMAL_DRAFT, "role": Role.SUPPORT, "companion_gender": Gender.MALE},
]


def _expected_ids(service: BookingService, combo: dict[str, Any]) -> set[int]:
    state = BookingState(
        game_mode=combo["game_mode"],
        start_time=NOW,
        duration_hours=1,
        rank_requirement=combo.get("rank_requirement"),
        role_preference=(combo["role"],) if combo.get("role") else None,
        companion_gender=combo.get("companion_gender"),
    )
    state, _ = apply_rules(state, service.domain)
    f = build_filters(state, service.domain)
    with read_session(service.factory) as s:
        repo = CompanionRepo(s, service.domain)
        return {c.id for c in repo.list_all() if matches(to_profile(c), f, service.domain)}


@pytest.mark.parametrize("combo", COMBOS)
def test_browse_matches_find_hard_filter(
    seeded_booking: BookingService, combo: dict[str, Any]
) -> None:
    out = browse_companions(seeded_booking, BrowseCompanionsInput(**combo))
    ids = {c.companion_id for c in out.companions}
    assert ids == _expected_ids(seeded_booking, combo)
    assert ids, f"seed should have someone for {combo}"
    # Everyone find_companions returns (exact, no relaxation) is in the list too.
    found = find_companions(
        seeded_booking,
        FindCompanionsInput(
            game_mode=combo["game_mode"],
            start_time=datetime(2026, 10, 2, 20, 0),
            duration_hours=1,
            rank_requirement=combo.get("rank_requirement"),
            role_preference=[combo["role"]] if combo.get("role") else None,
            companion_gender=combo.get("companion_gender"),
            top_k=10,
        ),
    )
    if not found.relaxations:
        assert {c.companion_id for c in found.candidates} <= ids


def test_browse_without_mode_filters_role_and_gender(seeded_booking: BookingService) -> None:
    everyone = browse_companions(seeded_booking, BrowseCompanionsInput()).companions
    assert len(everyone) == 30
    ratings = [c.rating for c in everyone]
    assert ratings == sorted(ratings, reverse=True)
    mids = browse_companions(
        seeded_booking, BrowseCompanionsInput(role=Role.MID, companion_gender=Gender.FEMALE)
    ).companions
    assert mids and all(Role.MID in c.roles and c.gender is Gender.FEMALE for c in mids)


def test_free_slots_are_future_free_and_long_enough(seeded_booking: BookingService) -> None:
    cards = browse_companions(seeded_booking, BrowseCompanionsInput(days=3)).companions
    horizon = NOW + timedelta(days=3)
    with read_session(seeded_booking.factory) as s:
        schedules = ScheduleRepo(s)
        for card in cards:
            for slot in card.free_slots:
                assert NOW <= slot.start < slot.end <= horizon
                assert slot.end - slot.start >= timedelta(hours=1)
                assert schedules.is_free(card.companion_id, slot.start, slot.end)
    assert any(card.free_slots for card in cards)


def test_cards_carry_prices_and_level(seeded_booking: BookingService) -> None:
    [card, *_] = browse_companions(
        seeded_booking, BrowseCompanionsInput(game_mode=GameMode.RANKED_SOLO_DUO)
    ).companions
    assert "climb" in card.prices
    climb = seeded_booking.domain.multiplier(ServiceType.CLIMB)
    assert card.prices["climb"] == to_money(card.hourly_price * climb)
    assert card.level in {lv.name for lv in seeded_booking.domain.policies.companion_levels}


# --- web page --------------------------------------------------------------------------------


@pytest.fixture
def client(make_web: Make, web_login: Callable[..., None]) -> TestClient:
    c, _ = make_web()
    web_login(c, "demo")
    return c


def _names(html: str) -> list[str]:
    return re.findall(r'<span class="title">([^<]+)</span>', html)


def test_page_lists_filtered_companions(client: TestClient, seeded_booking: BookingService) -> None:
    html = client.get("/companions?mode=ranked_solo_duo&rank=diamond").text
    expected = browse_companions(
        seeded_booking,
        BrowseCompanionsInput(game_mode=GameMode.RANKED_SOLO_DUO, rank_requirement=Rank.DIAMOND),
    ).companions
    assert _names(html) == [c.name for c in expected]
    assert f"共 {len(expected)} 位" in html
    assert 'option value="diamond" selected' in html
    assert "约 TA" in html and "/chat?q=" in html


def test_page_without_filters_and_bad_values(client: TestClient) -> None:
    all_html = client.get("/companions").text
    assert len(_names(all_html)) == 30
    bad = client.get("/companions?mode=tft&rank=godlike&role=x&gender=y").text
    assert _names(bad) == _names(all_html)  # unknown values are treated as "不限"


def test_page_requires_login(make_web: Make) -> None:
    c, _ = make_web()
    resp = c.get("/companions?mode=aram")
    assert (
        resp.status_code == 303
        and resp.headers["location"] == "/login?next=/companions%3Fmode%3Daram"
    )


def test_page_shows_error_when_service_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rift_agent.mcp_clients import McpUnavailable

    services: WebServices = client.app.state.services  # type: ignore[attr-defined]

    async def down(**_: Any) -> Any:
        raise McpUnavailable("booking unavailable")

    monkeypatch.setattr(services.booking, "browse_companions", down)
    html = client.get("/companions").text
    assert "陪玩师列表暂时加载不了" in html


def test_chat_prefill_from_link(client: TestClient) -> None:
    assert 'get("q")' in client.get("/static/chat.js").text
