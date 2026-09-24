"""Value types and errors shared by all LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True)
class LLMResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    model: str = ""


@dataclass(frozen=True)
class ChatOptions:
    """Per-call options after merging call arguments with provider config defaults."""

    response_format: dict[str, Any] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float = 30.0


class LLMError(Exception):
    """Base class for provider failures."""


class LLMTimeout(LLMError):
    """The provider did not answer within the timeout."""


class LLMUnavailable(LLMError):
    """The provider could not be reached or is overloaded (connect error, 5xx)."""


class LLMResponseError(LLMError):
    """The provider answered, but the request was rejected or the payload was malformed."""
