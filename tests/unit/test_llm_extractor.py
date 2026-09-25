"""F4: prompt rendering is deterministic; LLMSlotExtractor validates strictly."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pytest

from rift_agent.extractors import ExtractionContext, ExtractionResult, LLMSlotExtractor
from rift_agent.extractors.prompt import build_messages, render_context, state_view
from rift_agent.prompts import load_prompt
from rift_common.llm.mock import MockLLM
from rift_common.settings import LLMConfig
from rift_domain.enums import Confirmation, GameMode, Rank, Role, TurnIntent
from rift_domain.slots import ANY, BookingState

NOW = datetime(2026, 10, 1, 14, 0)
STATE = BookingState(
    game_mode=GameMode.RANKED_SOLO_DUO,
    start_time=datetime(2026, 10, 2, 20, 0),
    rank_requirement=Rank.DIAMOND,
    role_preference=(Role.JUNGLE,),
    companion_gender=ANY,
)
CTX = ExtractionContext(
    now=NOW,
    current_state=STATE,
    user_input="那就两小时吧",
    candidates=("阿狸酱", "夜雨声烦"),
    history=({"role": "user", "content": "帮我约个明晚八点的单双排"},),
)
GOOD = {"turn_intent": "booking", "delta": {"duration_hours": 2}, "confirmation": "none"}


def _mock(**kw: Any) -> MockLLM:
    return MockLLM(LLMConfig(provider="mock", model="mock-model"), **kw)


def test_render_context_is_deterministic_and_complete() -> None:
    a, b = render_context(CTX), render_context(CTX)
    assert a == b
    data = json.loads(a)
    assert data["now"] == "2026-10-01 14:00 星期四"
    assert data["current_state"] == {
        "game_mode": "ranked_solo_duo",
        "start_time": "2026-10-02 20:00 星期五",
        "rank_requirement": "diamond",
        "role_preference": ["jungle"],
        "companion_gender": "any",
    }
    assert data["candidates"] == ["阿狸酱", "夜雨声烦"]
    assert data["pending_confirmation"] is False
    assert data["user_input"] == "那就两小时吧"
    assert list(data) == sorted(data)


def test_state_view_skips_empty_and_derived_fields() -> None:
    assert state_view(BookingState()) == {}
    assert "missing_fields" not in state_view(STATE)


def test_system_prompt_is_the_shared_file() -> None:
    messages = build_messages(CTX)
    assert messages[0] == {"role": "system", "content": load_prompt("slot_extract.txt")}
    assert messages[1]["role"] == "user"
    # Same system prompt regardless of the turn: cacheable prefix.
    other = build_messages(ExtractionContext(now=NOW, current_state=BookingState(), user_input="x"))
    assert other[0] == messages[0]


async def test_valid_output() -> None:
    llm = _mock(default=GOOD)
    r = await LLMSlotExtractor(llm).extract(CTX)
    assert r.valid and r.error_kind is None and r.errors == ()
    assert r.extraction is not None
    assert r.extraction.turn_intent is TurnIntent.BOOKING
    assert r.extraction.confirmation is Confirmation.NONE
    assert r.extraction.delta.provided() == {"duration_hours": 2}
    assert r.model == "mock-model" and r.source == "llm"
    assert json.loads(r.raw) == GOOD


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("not json", "json"),
        ({**GOOD, "delta": {"duration_hours": 0.3}}, "delta.duration_hours"),
        ({**GOOD, "delta": {"game_mode": "tft"}}, "delta.game_mode"),
        ({**GOOD, "extra": 1}, "extra"),
        ({"turn_intent": "booking", "delta": {}}, "confirmation"),
    ],
)
async def test_invalid_output_is_reported(raw: Any, fragment: str) -> None:
    r = await LLMSlotExtractor(_mock(default=raw)).extract(CTX)
    assert not r.valid and r.extraction is None
    assert r.error_kind == "invalid"
    assert any(fragment in e.lower() for e in r.errors), r.errors


@pytest.mark.parametrize(
    ("error", "kind"),
    [("timeout", "timeout"), ("unavailable", "unavailable"), ("response_error", "unavailable")],
)
async def test_llm_errors_do_not_raise(error: str, kind: str) -> None:
    llm = _mock()
    llm.add_rule(contains="两小时", error=error)
    r = await LLMSlotExtractor(llm).extract(CTX)
    assert not r.valid and r.error_kind == kind
    assert r.errors and r.raw == ""


async def test_retry_feedback_is_sent() -> None:
    llm = _mock(default=GOOD)
    bad = ExtractionResult(
        extraction=None, raw='{"oops": 1}', valid=False, errors=("delta: Field required",)
    )
    await LLMSlotExtractor(llm).extract(CTX, feedback=bad)
    messages = llm.calls[0]
    assert messages[-2] == {"role": "assistant", "content": '{"oops": 1}'}
    assert "delta: Field required" in messages[-1]["content"]


def test_failed_helper() -> None:
    r = ExtractionResult.failed(("x",), source="routed")
    assert (r.valid, r.extraction, r.errors, r.source) == (False, None, ("x",), "routed")
