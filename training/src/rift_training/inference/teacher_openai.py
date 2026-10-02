"""OpenAI Teacher: Chat Completions with Structured Outputs (``json_schema``, strict).

The API key is only ever read from the environment variable named by ``api_key_env``.
Reasoning models reject ``temperature`` and ``max_tokens``, so the payload only carries what
the config sets, and uses ``max_completion_tokens`` / ``reasoning_effort``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from rift_training.inference.base import Teacher, TeacherError, TeacherRequest, TeacherResponse


class TeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai", "mock"]
    model: str = Field(min_length=1)
    base_url: str = "https://api.openai.com/v1"
    #: Name of the environment variable holding the key (never the key itself).
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = Field(default=180.0, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    #: Parallel requests and retries of transient failures (429 / 5xx / timeouts).
    concurrency: int = Field(default=8, gt=0)
    max_retries: int = Field(default=4, ge=0)


class OpenAITeacher(Teacher):
    name = "openai"

    def __init__(
        self, config: TeacherConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        key = os.environ.get(config.api_key_env, "").strip()
        if not key:
            raise TeacherError(
                f"environment variable {config.api_key_env} is not set", retryable=False
            )
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=config.timeout_s,
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self.config.model

    def payload(self, request: TeacherRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request.messages(),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": request.schema,
                },
            },
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.config.max_completion_tokens
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort
        return payload

    async def generate(self, request: TeacherRequest) -> TeacherResponse:
        start = time.perf_counter()
        try:
            resp = await self._client.post("/chat/completions", json=self.payload(request))
        except httpx.TimeoutException as exc:
            raise TeacherError(f"timed out after {self.config.timeout_s}s", retryable=True) from exc
        except httpx.TransportError as exc:
            raise TeacherError(f"unreachable: {exc}", retryable=True) from exc
        if resp.status_code >= 400:
            retryable = resp.status_code in (408, 409, 429) or resp.status_code >= 500
            # 429 also means "out of credits"; 401 a bad key -- retrying cannot help.
            fatal = resp.status_code == 401 or "insufficient_quota" in resp.text
            raise TeacherError(
                f"HTTP {resp.status_code}: {resp.text[:300]}", retryable=retryable, fatal=fatal
            )
        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TeacherError(f"malformed response: {resp.text[:300]}", retryable=True) from exc
        if message.get("refusal"):
            raise TeacherError(f"refused: {message['refusal'][:200]}", retryable=False)
        if choice.get("finish_reason") == "length":
            raise TeacherError("answer truncated (finish_reason=length)", retryable=False)
        usage = data.get("usage") or {}
        return TeacherResponse(
            text=message.get("content") or "",
            model=data.get("model") or self.config.model,
            usage={
                k: int(usage.get(k, 0))
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def describe(config: TeacherConfig) -> dict[str, str]:
    """What a data card records about the Teacher (never the key)."""
    info = {"provider": config.provider, "model": config.model}
    if config.reasoning_effort:
        info["reasoning_effort"] = config.reasoning_effort
    if config.temperature is not None:
        info["temperature"] = json.dumps(config.temperature)
    return info
