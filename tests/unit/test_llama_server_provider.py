import json
from collections.abc import Callable

import httpx
import pytest

from rift_common.llm import LLMFactory, LLMTimeout, LLMUnavailable
from rift_common.llm.llama_server import LlamaServerLLM
from rift_common.settings import LLMConfig

CFG = LLMConfig(
    provider="llama_server",
    base_url="http://llama-server:8080/v1",
    model="rift-slot-0.6b-q4km",
    timeout_s=5,
)
MSGS = [{"role": "user", "content": "明晚八点双排"}]


def _llm(
    handler: Callable[[httpx.Request], httpx.Response], cfg: LLMConfig = CFG
) -> LlamaServerLLM:
    llm = LLMFactory.create(cfg, transport=httpx.MockTransport(handler))
    assert isinstance(llm, LlamaServerLLM)
    return llm


def _ok(text: str = "{}") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_registered_under_llama_server() -> None:
    assert isinstance(_llm(lambda r: _ok()), LlamaServerLLM)


def test_deterministic_defaults_and_no_auth_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _ok('{"turn_intent": "booking"}')

    result = _llm(handler).chat(MSGS)
    body = json.loads(seen[0].content)
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 160
    assert "Authorization" not in seen[0].headers
    assert str(seen[0].url) == "http://llama-server:8080/v1/chat/completions"
    assert result.model == "rift-slot-0.6b-q4km"


def test_config_overrides_defaults() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ok()

    cfg = CFG.model_copy(update={"max_tokens": 80, "temperature": 0.2})
    _llm(handler, cfg).chat(MSGS)
    assert seen[0]["max_tokens"] == 80
    assert seen[0]["temperature"] == 0.2


def test_timeout_raises_llm_timeout_with_configured_timeout() -> None:
    timeouts: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeout, match="5"):
        _llm(handler).chat(MSGS)
    assert timeouts[0] == {"connect": 5, "read": 5, "write": 5, "pool": 5}


def test_unreachable_raises_llm_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(LLMUnavailable):
        _llm(handler).chat(MSGS)


def test_timeout_and_unavailable_are_distinct() -> None:
    assert not issubclass(LLMTimeout, LLMUnavailable)
    assert not issubclass(LLMUnavailable, LLMTimeout)


def test_ping_hits_root_health_endpoint() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"status": "ok"})

    assert _llm(handler).ping() is True
    assert seen == ["http://llama-server:8080/health"]


@pytest.mark.parametrize("status", [503, 500])
def test_ping_false_while_loading_or_broken(status: int) -> None:
    assert _llm(lambda r: httpx.Response(status)).ping() is False


def test_ping_false_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert _llm(handler).ping() is False
