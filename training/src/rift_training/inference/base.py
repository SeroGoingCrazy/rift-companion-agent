"""Pluggable Teacher backends: one structured (JSON Schema constrained) completion per call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TeacherRequest:
    system: str
    user: str
    #: JSON Schema the answer must follow (strict: every property required, no extras).
    schema: dict[str, Any]
    schema_name: str = "answer"
    #: Earlier turns of a retry: the previous answer and the validator's complaints.
    followups: tuple[dict[str, str], ...] = ()

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
            *self.followups,
        ]


@dataclass(frozen=True)
class TeacherResponse:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class TeacherError(Exception):
    """A failed Teacher call; ``retryable`` for rate limits, timeouts and 5xx."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class Teacher(ABC):
    #: Backend name recorded in data cards ("openai", "mock", ...).
    name: str = ""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def generate(self, request: TeacherRequest) -> TeacherResponse: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release network resources."""
