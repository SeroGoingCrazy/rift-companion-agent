import json
from pathlib import Path

import pytest

from rift_common.llm import (
    LLMFactory,
    LLMTimeout,
    LLMUnavailable,
    Message,
    MockLLM,
    MockNoMatchError,
    MockRule,
)
from rift_common.settings import LLMConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "llm" / "mock_basic.yaml"


def _user(text: str) -> list[Message]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": text}]


def _mock(**kwargs: object) -> MockLLM:
    llm = LLMFactory.create(LLMConfig(provider="mock", model="mock-1"), **kwargs)
    assert isinstance(llm, MockLLM)
    return llm


def test_factory_creates_mock_from_config_fixture() -> None:
    llm = LLMFactory.create(LLMConfig(provider="mock", model="m", fixture=str(FIXTURE)))
    assert isinstance(llm, MockLLM)
    data = json.loads(llm.chat(_user("帮我约明晚八点单双排")).text)
    assert data["delta"]["game_mode"] == "ranked_solo_duo"
    assert data["delta"]["start_time_expr"] == "明晚八点"  # non-ASCII preserved


def test_regex_rule_and_default() -> None:
    llm = _mock(fixture=FIXTURE)
    assert llm.chat(_user("取消我的订单")).text == "manage"
    assert llm.chat(_user("我要取消")).text == "抱歉，我没听懂"


def test_first_matching_rule_wins_and_matches_last_user_message() -> None:
    llm = _mock(
        rules=[MockRule(contains="a", response="first"), MockRule(contains="a", response="2nd")]
    )
    messages: list[Message] = [
        {"role": "user", "content": "zzz"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "abc"},
    ]
    assert llm.chat(messages).text == "first"


def test_error_rules_raise_typed_errors() -> None:
    llm = _mock(fixture=FIXTURE)
    with pytest.raises(LLMTimeout):
        llm.chat(_user("这次会超时"))
    with pytest.raises(LLMUnavailable):
        llm.chat(_user("服务挂了"))


def test_no_match_without_default_raises() -> None:
    llm = _mock(rules=[MockRule(contains="x", response="y")])
    with pytest.raises(MockNoMatchError):
        llm.chat(_user("nothing"))


def test_handler_takes_precedence_and_calls_are_recorded() -> None:
    llm = _mock(fixture=FIXTURE, handler=lambda msgs: {"n": len(msgs)})
    result = llm.chat(_user("明晚"))
    assert json.loads(result.text) == {"n": 2}
    assert llm.calls == [_user("明晚")]
    assert result.model == "mock-1"
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens


def test_add_rule_at_runtime() -> None:
    llm = _mock(default="d")
    llm.add_rule(contains="hi", response="hello")
    assert llm.chat(_user("hi there")).text == "hello"


async def test_astream_reassembles_full_text() -> None:
    llm = _mock(default="一二三四五六七八九十")
    chunks = [c async for c in llm.astream(_user("x"))]
    assert len(chunks) > 1
    assert "".join(chunks) == "一二三四五六七八九十"


def test_invalid_rules_rejected() -> None:
    with pytest.raises(ValueError, match="contains"):
        MockRule(response="x")
    with pytest.raises(ValueError, match="error must be one of"):
        MockRule(contains="x", error="boom")
