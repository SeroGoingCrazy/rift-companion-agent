import json
from collections.abc import Callable

import httpx
import pytest

from rift_common.llm import (
    LLMFactory,
    LLMResponseError,
    LLMTimeout,
    LLMUnavailable,
    OpenAICompatibleLLM,
)
from rift_common.settings import LLMConfig

CFG = LLMConfig(
    provider="openai_compatible",
    base_url="https://api.example.com/v1",
    model="deepseek-chat",
    api_key="test-key",
    timeout_s=3,
)
MSGS = [{"role": "user", "content": "你好"}]


def _completion(text: str) -> dict[str, object]:
    return {
        "model": "deepseek-chat",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _llm(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAICompatibleLLM:
    llm = LLMFactory.create(CFG, transport=httpx.MockTransport(handler))
    assert isinstance(llm, OpenAICompatibleLLM)
    return llm


def test_chat_sends_openai_payload_and_parses_result() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_completion('{"ok": true}'))

    result = _llm(handler).chat(
        MSGS, response_format={"type": "json_object"}, temperature=0, max_tokens=50
    )
    req = seen[0]
    assert str(req.url) == "https://api.example.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer test-key"
    body = json.loads(req.content)
    assert body == {
        "model": "deepseek-chat",
        "messages": MSGS,
        "stream": False,
        "temperature": 0,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
    }
    assert json.loads(result.text) == {"ok": True}
    assert result.usage.total_tokens == 7
    assert result.latency_ms >= 0


def test_optional_params_omitted_when_unset() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("x"))

    _llm(handler).chat(MSGS)
    assert set(seen[0]) == {"model", "messages", "stream"}


def test_timeout_maps_to_llm_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMTimeout):
        _llm(handler).chat(MSGS)


def test_connect_error_maps_to_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailable, match="unreachable"):
        _llm(handler).chat(MSGS)


@pytest.mark.parametrize(
    ("status", "error"),
    [(500, LLMUnavailable), (503, LLMUnavailable), (429, LLMUnavailable), (400, LLMResponseError)],
)
def test_http_errors_are_classified(status: int, error: type[Exception]) -> None:
    with pytest.raises(error, match=f"HTTP {status}"):
        _llm(lambda r: httpx.Response(status, text="nope")).chat(MSGS)


def test_malformed_payload_raises_response_error() -> None:
    with pytest.raises(LLMResponseError, match="malformed"):
        _llm(lambda r: httpx.Response(200, json={"choices": []})).chat(MSGS)


def test_missing_base_url_rejected() -> None:
    with pytest.raises(ValueError, match="base_url"):
        LLMFactory.create(LLMConfig(provider="openai_compatible", model="m"))


def _sse(*lines: str) -> bytes:
    return "".join(f"{line}\n\n" for line in lines).encode()


async def test_astream_parses_sse_until_done() -> None:
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "你"}}]},
        {"choices": [{"delta": {"content": "好"}}]},
    ]
    body = _sse(*(f"data: {json.dumps(c)}" for c in chunks), "data: [DONE]", "data: ignored")

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    llm = LLMFactory.create(CFG, async_transport=httpx.MockTransport(handler))
    assert [c async for c in llm.astream(MSGS)] == ["你", "好"]


async def test_astream_http_error() -> None:
    llm = LLMFactory.create(
        CFG, async_transport=httpx.MockTransport(lambda r: httpx.Response(503, text="busy"))
    )
    with pytest.raises(LLMUnavailable, match="HTTP 503"):
        [c async for c in llm.astream(MSGS)]
