"""F3: classify node with a mock LLM — four intents, robust parsing, LLM failures."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig

from rift_agent.deps import AgentDeps, get_deps
from rift_agent.graph.nodes.classify import Intent, classify, parse_intent, route_intent
from rift_agent.prompts import load_prompt, prompt_sha256
from rift_common.llm.mock import MockLLM
from rift_common.settings import LLMConfig
from rift_domain.config import DomainConfig

RULES = [
    {"contains": "约", "response": {"intent": "booking"}},
    {"contains": "扣钱", "response": {"intent": "consult"}},
    {"contains": "订单", "response": {"intent": "manage"}},
    {"contains": "天气", "response": {"intent": "other"}},
    {"contains": "乱码", "response": "not json at all"},
    {"contains": "超时", "error": "timeout"},
]


def _config(domain: DomainConfig, llm: MockLLM) -> RunnableConfig:
    return {"configurable": {"deps": AgentDeps(domain=domain, llm=llm)}}


@pytest.fixture
def llm() -> MockLLM:
    mock = MockLLM(LLMConfig(provider="mock", model="mock"))
    for rule in RULES:
        mock.add_rule(**rule)
    return mock


@pytest.mark.parametrize(
    ("text", "intent", "next_node"),
    [
        ("帮我约个明晚的双排", "booking", "extract_slots"),
        ("取消要扣钱吗", "consult", "consult"),
        ("我有哪些订单", "manage", "manage"),
        ("今天天气怎么样", "other", "other"),
        ("乱码回复", "other", "other"),
        ("模型超时了", "other", "other"),
    ],
)
async def test_classify_routes(
    domain: DomainConfig, llm: MockLLM, text: str, intent: str, next_node: str
) -> None:
    update = await classify({"user_input": text}, _config(domain, llm))
    assert update == {"intent": intent}
    assert route_intent({**update}) == next_node


async def test_classify_sends_prompt_and_json_mode(domain: DomainConfig, llm: MockLLM) -> None:
    await classify({"user_input": "帮我约"}, _config(domain, llm))
    [messages] = llm.calls
    assert messages[0] == {"role": "system", "content": load_prompt("classify.txt")}
    assert messages[1] == {"role": "user", "content": "帮我约"}


@pytest.mark.parametrize(
    ("raw", "intent"),
    [
        ('{"intent": "consult"}', Intent.CONSULT),
        ('{"intent": "BOOKING"}', Intent.BOOKING),
        ("manage", Intent.MANAGE),
        ('"manage"', Intent.MANAGE),
        ('{"intent": "refund"}', Intent.OTHER),
        ('{"label": "booking"}', Intent.OTHER),
        ("[1, 2]", Intent.OTHER),
        ("", Intent.OTHER),
    ],
)
def test_parse_intent(raw: str, intent: Intent) -> None:
    assert parse_intent(raw) is intent


def test_route_after_error() -> None:
    assert route_intent({"intent": "booking", "error": "x"}) == "render"


def test_get_deps_requires_deps() -> None:
    with pytest.raises(RuntimeError, match=r"configurable.deps"):
        get_deps({"configurable": {}})


def test_prompt_hash_is_stable() -> None:
    assert prompt_sha256("classify.txt") == prompt_sha256("classify.txt")
    assert len(prompt_sha256("classify.txt")) == 64


def test_deps_now_drops_microseconds(domain: DomainConfig, llm: MockLLM) -> None:
    from datetime import datetime

    deps = AgentDeps(domain=domain, llm=llm, clock=lambda: datetime(2026, 1, 1, 9, 0, 0, 123))
    assert deps.now() == datetime(2026, 1, 1, 9, 0)
