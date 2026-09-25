"""F6: booking main path — collect slots over several turns, ask for a precise time,
search companions (mock extractor + in-process booking-mcp)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from rift_agent.graph.state import booking_state

if TYPE_CHECKING:
    from tests.integration.conftest import AgentHarness


def turn(confirmation: str = "none", intent: str = "booking", **delta: Any) -> dict[str, Any]:
    return {"turn_intent": intent, "delta": delta, "confirmation": confirmation}


TABLE: dict[str, dict[str, Any]] = {
    "帮我约个单双排": turn(game_mode="ranked_solo_duo"),
    "明晚": turn(start_time_expr="明晚"),
    "八点，两个小时，钻石以上": turn(
        start_time_expr="八点", duration_hours=2, rank_requirement="diamond"
    ),
    "帮我约个大乱斗，明晚八点两小时，段位要钻石": turn(
        game_mode="aram", start_time_expr="明晚八点", duration_hours=2, rank_requirement="diamond"
    ),
    "约个大乱斗，昨天晚上八点": turn(game_mode="aram", start_time_expr="昨天晚上八点"),
    "约个大乱斗明晚八点，打九个小时": turn(
        game_mode="aram", start_time_expr="明晚八点", duration_hours=9
    ),
    "约个大乱斗明晚八点两小时，预算每小时5块": turn(
        game_mode="aram", start_time_expr="明晚八点", duration_hours=2, budget_per_hour=5
    ),
    "段位无所谓": turn(rank_requirement="any"),
}  # fmt: skip


async def test_three_turns_from_zero_to_candidates(
    make_agent: Callable[..., AgentHarness],
) -> None:
    agent = make_agent(TABLE)

    s1 = await agent.say("帮我约个单双排")
    assert s1["phase"] == "BOOKING"
    assert s1["reply_type"] == "ask_missing"
    assert s1["facts"]["field"] == "start_time"
    assert "单双排" in s1["reply"] and "什么时候" in s1["reply"]

    s2 = await agent.say("明晚")
    assert s2["reply_type"] == "ask_time"
    assert "具体是几点" in s2["reply"]
    assert booking_state(s2).start_time is None
    assert booking_state(s2).start_time_expr == "明晚"

    s3 = await agent.say("八点，两个小时，钻石以上")
    b = booking_state(s3)
    assert b.start_time is not None and b.start_time.isoformat() == "2026-10-02T20:00:00"
    assert s3["reply_type"] == "candidates"
    assert 1 <= len(b.candidates) <= 3
    assert s3["ui"]["candidates"] == s3["candidate_cards"]
    for card in s3["candidate_cards"]:
        assert card["rank"] in {"diamond", "master", "grandmaster"}
    assert "想约哪位" in s3["reply"]
    assert s3["candidate_cards"][0]["name"] in s3["reply"]
    # The extractor saw the collected state and conversation history.
    ctx = agent.deps.extractor.contexts[-1]  # type: ignore[union-attr]
    assert ctx.current_state.game_mode == "ranked_solo_duo"
    assert [m["content"] for m in ctx.history][:1] == ["帮我约个单双排"]


async def test_one_shot_request_and_not_applicable_rank(
    make_agent: Callable[..., AgentHarness],
) -> None:
    agent = make_agent(TABLE)
    s = await agent.say("帮我约个大乱斗，明晚八点两小时，段位要钻石")
    b = booking_state(s)
    assert b.rank_requirement is None  # not applicable to ARAM, cleared by rules
    assert b.service_type_effective == "casual"
    assert s["reply_type"] == "candidates"


async def test_any_satisfies_required_rank(make_agent: Callable[..., AgentHarness]) -> None:
    agent = make_agent(TABLE)
    await agent.say("帮我约个单双排")
    await agent.say("明晚")
    s = await agent.say("段位无所谓")
    assert s["reply_type"] == "ask_time"  # "明晚" is still pending; rank no longer missing
    assert "「明晚」具体是几点" in s["reply"]
    assert "rank_requirement" not in [f for f in booking_state(s).missing_fields]


async def test_past_time_is_rejected(make_agent: Callable[..., AgentHarness]) -> None:
    s = await make_agent(TABLE).say("约个大乱斗，昨天晚上八点")
    assert s["reply_type"] == "ask_time"
    assert s["facts"]["reason"] == "past"
    assert "已经过了" in s["reply"]


async def test_invalid_duration_is_asked_again(make_agent: Callable[..., AgentHarness]) -> None:
    s = await make_agent(TABLE).say("约个大乱斗明晚八点，打九个小时")
    assert s["reply_type"] == "ask_missing"
    assert s["facts"]["field"] == "duration_hours"
    assert "1–8 小时" in s["reply"]


async def test_budget_too_low_finds_nobody(make_agent: Callable[..., AgentHarness]) -> None:
    s = await make_agent(TABLE).say("约个大乱斗明晚八点两小时，预算每小时5块")
    assert s["reply_type"] == "no_candidates"
    assert "预算和段位要求不会自动放宽" in s["reply"]
    assert booking_state(s).candidates == ()


async def test_extraction_failure_keeps_state(make_agent: Callable[..., AgentHarness]) -> None:
    agent = make_agent(TABLE)
    await agent.say("帮我约个单双排")
    s = await agent.say("（无法解析的输入）")
    assert s["reply_type"] == "extraction_failed"
    assert booking_state(s).game_mode == "ranked_solo_duo"
    assert s["phase"] == "BOOKING"
    assert "单双排" in s["reply"]


async def test_node_crash_answers_and_keeps_state(
    make_agent: Callable[..., AgentHarness], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = make_agent(TABLE)
    await agent.say("帮我约个单双排")
    await agent.say("明晚")

    async def boom(**_: Any) -> Any:
        raise RuntimeError("search exploded")

    monkeypatch.setattr(agent.deps.booking, "find_companions", boom)
    s = await agent.say("八点，两个小时，钻石以上")
    assert s["reply_type"] == "error"
    assert "之前的信息我都还记着" in s["reply"]
    b = booking_state(s)
    assert b.start_time is not None and b.duration_hours == 2  # merged before the crash
    assert s["error"] is None  # cleared by render


async def test_idle_small_talk_gets_help(make_agent: Callable[..., AgentHarness]) -> None:
    s = await make_agent(TABLE).say("你好呀")
    assert s["reply_type"] == "help"
    assert s["phase"] == "IDLE"
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]
