"""Deterministic mock LLM for unit tests and offline CI.

Responses come from, in order: a ``handler`` callback, then rules (from ``rules=`` or a
fixture file), then ``default``. Rules match the last user message::

    # tests/fixtures/llm/example.yaml
    default: "好的"
    rules:
      - contains: "明晚"
        response: {"turn_intent": "booking", "delta": {}, "confirmation": "none"}
      - regex: "^取消"
        response: "manage"
      - contains: "超时"
        error: timeout          # timeout | unavailable | response_error

Mapping/list responses are serialized as JSON (``ensure_ascii=False``).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rift_common.llm.base import BaseLLM
from rift_common.llm.types import (
    ChatOptions,
    LLMError,
    LLMResponseError,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Message,
    Usage,
)
from rift_common.settings import LLMConfig

_ERRORS: dict[str, type[LLMError]] = {
    "timeout": LLMTimeout,
    "unavailable": LLMUnavailable,
    "response_error": LLMResponseError,
}

Handler = Callable[[list[Message]], Any]


class MockNoMatchError(LLMError):
    """No rule matched and no default was configured."""


@dataclass(frozen=True)
class MockRule:
    response: Any = None
    contains: str | None = None
    regex: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.contains is None and self.regex is None:
            raise ValueError("MockRule needs `contains` or `regex`")
        if self.error is not None and self.error not in _ERRORS:
            raise ValueError(f"MockRule.error must be one of {sorted(_ERRORS)}")

    def matches(self, text: str) -> bool:
        if self.contains is not None and self.contains not in text:
            return False
        return self.regex is None or re.search(self.regex, text) is not None


def _to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    return json.dumps(response, ensure_ascii=False)


def load_fixture(path: str | Path) -> tuple[list[MockRule], Any]:
    """Load ``(rules, default)`` from a YAML/JSON fixture file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"mock fixture {path} must be a mapping")
    rules = [MockRule(**r) for r in data.get("rules", [])]
    return rules, data.get("default")


class MockLLM(BaseLLM):
    provider_name = "mock"

    def __init__(
        self,
        config: LLMConfig,
        *,
        rules: Sequence[MockRule] = (),
        default: Any = None,
        handler: Handler | None = None,
        fixture: str | Path | None = None,
    ) -> None:
        super().__init__(config)
        self.rules: list[MockRule] = []
        self.default = default
        fixture = fixture if fixture is not None else config.fixture
        if fixture is not None:
            file_rules, file_default = load_fixture(fixture)
            self.rules.extend(file_rules)
            if self.default is None:
                self.default = file_default
        self.rules.extend(rules)
        self.handler = handler
        #: Every message list received, for assertions.
        self.calls: list[list[Message]] = []

    def add_rule(self, **kwargs: Any) -> None:
        self.rules.append(MockRule(**kwargs))

    def _respond(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        if self.handler is not None:
            return _to_text(self.handler(messages))
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        for rule in self.rules:
            if rule.matches(last_user):
                if rule.error is not None:
                    raise _ERRORS[rule.error](f"mock error for {last_user!r}")
                return _to_text(rule.response)
        if self.default is not None:
            return _to_text(self.default)
        raise MockNoMatchError(f"no mock rule matched {last_user!r}")

    def _chat(self, messages: list[Message], opts: ChatOptions) -> LLMResult:
        text = self._respond(messages)
        prompt_chars = sum(len(m["content"]) for m in messages)
        usage = Usage(prompt_chars, len(text), prompt_chars + len(text))
        return LLMResult(text=text, usage=usage, latency_ms=0.0, model=self.model)

    async def _astream(self, messages: list[Message], opts: ChatOptions) -> AsyncIterator[str]:
        text = self._respond(messages)
        for i in range(0, len(text), 4):
            yield text[i : i + 4]
