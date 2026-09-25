"""F8: consults (IDLE and mid-booking), unrelated detours and explicit abandon."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from rift_agent.graph.state import booking_state
from rift_agent.mcp_clients import McpUnavailable
from rift_common.llm.mock import MockLLM
from rift_common.trace import TraceContext, start_turn
from rift_common.trace.sinks.base import TraceSink

if TYPE_CHECKING:
    from tests.integration.conftest import AgentHarness


def turn(confirmation: str = "none", intent: str = "booking", **delta: Any) -> dict[str, Any]:
    return {"turn_intent": intent, "delta": delta, "confirmation": confirmation}


TABLE = {
    "帮我约个明晚八点的单双排，钻石以上": turn(
        game_mode="ranked_solo_duo", start_time_expr="明晚八点", rank_requirement="diamond"
    ),
    "等下，取消要扣钱吗？": turn(intent="consult"),
    "那就两小时吧": turn(duration_hours=2),
    "今天天气怎么样": turn(intent="unrelated"),
    "不约了": turn(confirmation="no"),
}  # fmt: skip

ANSWER = "开局前 24 小时以上取消全额退款，2–24 小时退一半，不足 2 小时不退 [1]。"


class Recorder(TraceSink):
    def __init__(self) -> None:
        self.traces: list[TraceContext] = []

    def on_turn_end(self, trace: TraceContext) -> None:
        self.traces.append(trace)


@pytest.fixture
def agent(make_agent: Callable[..., AgentHarness], classifier: MockLLM) -> AgentHarness:
    return make_agent(TABLE, llm=classifier)


@pytest.fixture
def classifier(make_classifier: Callable[[], MockLLM]) -> MockLLM:
    llm = make_classifier()
    llm.default = ANSWER
    return llm


async def test_interjection_keeps_booking_and_continues(
    agent: AgentHarness, classifier: MockLLM
) -> None:
    s1 = await agent.say("帮我约个明晚八点的单双排，钻石以上")
    assert s1["facts"]["field"] == "duration_hours"
    before = booking_state(s1)

    sink = Recorder()
    with start_turn(sinks=[sink]):
        s2 = await agent.say("等下，取消要扣钱吗？")
    assert s2["reply_type"] == "consult"
    assert s2["phase"] == "BOOKING"
    assert ANSWER in s2["reply"]
    assert "（参考：平台规则）" in s2["reply"]
    assert "回到预约：打几个小时呢？" in s2["reply"]
    assert booking_state(s2) == before  # slots untouched
    names = [s.name for s in sink.traces[0].spans]
    assert "tool:query_knowledge_hub" in names and "node:consult_interject" in names
    # The LLM answered from the retrieved passage.
    consult_call = classifier.calls[-1]
    assert "退款档位" in consult_call[-1]["content"]
    assert "【问题】等下，取消要扣钱吗？" in consult_call[-1]["content"]

    s3 = await agent.say("那就两小时吧")
    b = booking_state(s3)
    assert b.duration_hours == 2
    assert b.game_mode == "ranked_solo_duo" and b.rank_requirement == "diamond"
    assert s3["reply_type"] in {"candidates", "no_candidates"}


async def test_idle_consult(agent: AgentHarness) -> None:
    s = await agent.say("斗魂竞技场几个人？")
    assert s["reply_type"] == "consult" and s["phase"] == "IDLE"
    assert "（参考：模式与段位）" in s["reply"]
    assert "回到预约" not in s["reply"]


async def test_no_answer_found(agent: AgentHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    async def nothing(question: str, **_: Any) -> list[Any]:
        return []

    monkeypatch.setattr(agent.deps.knowledge, "query_all", nothing)
    s = await agent.say("斗魂竞技场几个人？")
    assert "暂时没有查到相关规则" in s["reply"]


async def test_knowledge_down_keeps_booking(
    agent: AgentHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await agent.say("帮我约个明晚八点的单双排，钻石以上")

    async def down(question: str, **_: Any) -> list[Any]:
        raise McpUnavailable("knowledge unavailable")

    monkeypatch.setattr(agent.deps.knowledge, "query_all", down)
    s = await agent.say("等下，取消要扣钱吗？")
    assert "规则查询服务暂时连不上" in s["reply"]
    assert "打几个小时" in s["reply"]
    assert booking_state(s).game_mode == "ranked_solo_duo"


async def test_consult_llm_failure_falls_back_to_passage(
    agent: AgentHarness, classifier: MockLLM
) -> None:
    classifier.handler = None
    classifier.add_rule(contains="【问题】", error="unavailable")
    classifier.add_rule(contains="几个人", response={"intent": "consult"})
    s = await agent.say("斗魂竞技场几个人？")
    assert "8 队 × 2 人" in s["reply"]


async def test_unrelated_pulls_back(agent: AgentHarness) -> None:
    await agent.say("帮我约个明晚八点的单双排，钻石以上")
    s = await agent.say("今天天气怎么样")
    assert s["reply_type"] == "unrelated"
    assert "我们继续预约吧：还需要时长" in s["reply"]
    assert s["phase"] == "BOOKING"


async def test_abandon_clears_and_returns_to_idle(agent: AgentHarness) -> None:
    await agent.say("帮我约个明晚八点的单双排，钻石以上")
    s = await agent.say("不约了")
    assert s["reply_type"] == "abandon"
    assert s["phase"] == "IDLE"
    assert booking_state(s).game_mode is None
    assert "已取消：单双排" in s["reply"]
    # A new request starts from scratch through classification.
    s = await agent.say("帮我约个明晚八点的单双排，钻石以上")
    assert s["phase"] == "BOOKING" and s["facts"]["field"] == "duration_hours"
