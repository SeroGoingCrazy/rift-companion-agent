"""OpenAI-compatible chat completions over plain httpx (DeepSeek, Qwen, OpenAI, llama.cpp)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from rift_common.llm.base import BaseLLM
from rift_common.llm.types import (
    ChatOptions,
    LLMResponseError,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Message,
    Usage,
)
from rift_common.settings import LLMConfig

_CHAT_PATH = "/chat/completions"


class OpenAICompatibleLLM(BaseLLM):
    provider_name = "openai_compatible"

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        if not config.base_url:
            raise ValueError(f"{self.provider_name} provider requires base_url")
        self.base_url = config.base_url.rstrip("/")
        self._async_transport = async_transport
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=config.timeout_s,
            transport=transport,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _payload(
        self, messages: list[Message], opts: ChatOptions, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": stream}
        if opts.temperature is not None:
            payload["temperature"] = opts.temperature
        if opts.max_tokens is not None:
            payload["max_tokens"] = opts.max_tokens
        if opts.response_format is not None:
            payload["response_format"] = opts.response_format
        return payload

    @staticmethod
    def _raise_for_status(status: int, body: str) -> None:
        if status < 400:
            return
        detail = body[:300]
        if status == 429 or status >= 500:
            raise LLMUnavailable(f"HTTP {status}: {detail}")
        raise LLMResponseError(f"HTTP {status}: {detail}")

    def _chat(self, messages: list[Message], opts: ChatOptions) -> LLMResult:
        start = time.perf_counter()
        try:
            resp = self._client.post(
                _CHAT_PATH, json=self._payload(messages, opts, stream=False), timeout=opts.timeout
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"{self.model} timed out after {opts.timeout}s") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"{self.base_url} unreachable: {exc}") from exc
        self._raise_for_status(resp.status_code, resp.text)
        latency_ms = (time.perf_counter() - start) * 1000
        return self._parse_completion(resp, latency_ms)

    def _parse_completion(self, resp: httpx.Response, latency_ms: float) -> LLMResult:
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError(f"malformed completion payload: {resp.text[:300]}") from exc
        raw_usage = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0)),
            completion_tokens=int(raw_usage.get("completion_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
        )
        return LLMResult(
            text=text, usage=usage, latency_ms=latency_ms, model=data.get("model") or self.model
        )

    async def _astream(self, messages: list[Message], opts: ChatOptions) -> AsyncIterator[str]:
        payload = self._payload(messages, opts, stream=True)
        try:
            async with (
                httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._headers(),
                    timeout=opts.timeout,
                    transport=self._async_transport,
                ) as client,
                client.stream("POST", _CHAT_PATH, json=payload) as resp,
            ):
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    self._raise_for_status(resp.status_code, body)
                async for line in resp.aiter_lines():
                    delta = _parse_sse_delta(line)
                    if delta is None:
                        continue
                    if delta is _DONE:
                        break
                    if delta:
                        yield delta
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"{self.model} stream timed out after {opts.timeout}s") from exc
        except httpx.TransportError as exc:
            raise LLMUnavailable(f"{self.base_url} unreachable: {exc}") from exc

    def close(self) -> None:
        self._client.close()


_DONE = "\x00[DONE]"


def _parse_sse_delta(line: str) -> str | None:
    """Return the content delta of one SSE line, ``_DONE`` at the end, ``None`` to skip."""
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return _DONE
    try:
        chunk = json.loads(data)
        content = chunk["choices"][0].get("delta", {}).get("content")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMResponseError(f"malformed stream chunk: {data[:200]}") from exc
    return content if isinstance(content, str) else ""
