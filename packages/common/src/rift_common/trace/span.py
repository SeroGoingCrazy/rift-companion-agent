"""Span record."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

SpanStatus = Literal["ok", "error"]

#: Span kinds. ``generation`` marks an LLM call (model/input/output/usage attrs).
SpanKind = Literal["span", "generation", "tool"]


def new_trace_id() -> str:
    """32 hex chars: compatible with OpenTelemetry / Langfuse trace ids."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """16 hex chars: compatible with OpenTelemetry span ids."""
    return uuid.uuid4().hex[:16]


@dataclass
class Span:
    name: str
    trace_id: str
    parent_id: str | None = None
    kind: SpanKind = "span"
    attrs: dict[str, Any] = field(default_factory=dict)
    span_id: str = field(default_factory=new_span_id)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = "ok"
    error: str | None = None
    _t0: float = field(default_factory=time.perf_counter, repr=False)
    _duration_ms: float | None = field(default=None, repr=False)

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def set_attrs(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def finish(self, error: BaseException | None = None) -> None:
        if self.end_time is not None:
            return
        self._duration_ms = (time.perf_counter() - self._t0) * 1000
        self.end_time = self.start_time + self._duration_ms / 1000
        if error is not None:
            self.status = "error"
            self.error = f"{type(error).__name__}: {error}"

    @property
    def finished(self) -> bool:
        return self.end_time is not None

    @property
    def duration_ms(self) -> float | None:
        return self._duration_ms
