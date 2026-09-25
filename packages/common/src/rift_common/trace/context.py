"""Per-turn trace context built on ``contextvars``.

Usage::

    with start_turn(session_id="s1", user_id="u1", turn_idx=3, input=text, sinks=[sink]) as turn:
        with span("classify") as s:
            s.set_attr("label", "booking")
        turn.set_attr("reply", reply)
    # on exit every sink receives the finished TraceContext

The current span lives in a ContextVar, so the parent chain (the "span stack") stays correct
across nested calls, threads started with ``contextvars.copy_context`` and asyncio tasks.
Outside an active turn, ``span()`` yields a detached span that is never exported, so
instrumented code runs unchanged in tests and scripts.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rift_common.trace.span import Span, SpanKind, new_trace_id

if TYPE_CHECKING:
    from rift_common.trace.sinks.base import TraceSink

logger = logging.getLogger(__name__)

TURN_SPAN_NAME = "turn"

_current_trace: ContextVar[TraceContext | None] = ContextVar("rift_trace", default=None)
_current_span: ContextVar[Span | None] = ContextVar("rift_span", default=None)


@dataclass
class TraceContext:
    trace_id: str
    session_id: str | None = None
    user_id: str | None = None
    turn_idx: int | None = None
    #: Span id in a calling service (from ``_meta``) when this trace continues another one.
    parent_span_id: str | None = None
    sinks: Sequence[TraceSink] = ()
    #: Finished spans, in completion order (children before parents).
    spans: list[Span] = field(default_factory=list)
    root: Span | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    def dispatch(self, hook: str, *args: Any) -> None:
        """Call ``hook`` on every sink; a failing sink never breaks the conversation."""
        for sink in self.sinks:
            try:
                getattr(sink, hook)(*args)
            except Exception:
                logger.warning("trace sink %s.%s failed", type(sink).__name__, hook, exc_info=True)


def current_trace() -> TraceContext | None:
    return _current_trace.get()


def current_span() -> Span | None:
    return _current_span.get()


@contextmanager
def span(name: str, *, kind: SpanKind = "span", **attrs: Any) -> Iterator[Span]:
    """Open a child span of the current span. Exceptions are recorded and re-raised."""
    trace = _current_trace.get()
    parent = _current_span.get()
    if trace is None:
        detached = Span(name=name, trace_id="", kind=kind, attrs=dict(attrs))
        try:
            yield detached
        except BaseException as exc:
            detached.finish(exc)
            raise
        detached.finish()
        return

    s = Span(
        name=name,
        trace_id=trace.trace_id,
        parent_id=parent.span_id if parent is not None else trace.parent_span_id,
        kind=kind,
        attrs=dict(attrs),
    )
    trace.dispatch("on_span_start", s, trace)
    token = _current_span.set(s)
    try:
        yield s
    except BaseException as exc:
        s.finish(exc)
        raise
    finally:
        _current_span.reset(token)
        s.finish()
        trace.record(s)
        trace.dispatch("on_span_end", s, trace)


@contextmanager
def detached() -> Iterator[None]:
    """Run without the caller's trace: a server handling a request in the caller's process
    (in-process MCP transport) starts its own turn instead of nesting into the caller's."""
    trace_token = _current_trace.set(None)
    span_token = _current_span.set(None)
    try:
        yield
    finally:
        _current_span.reset(span_token)
        _current_trace.reset(trace_token)


@contextmanager
def start_turn(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    turn_idx: int | None = None,
    sinks: Sequence[TraceSink] = (),
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Start a trace for one conversation turn; yields the root ``turn`` span.

    Pass ``trace_id`` (and optionally ``parent_span_id``) to continue a trace started by
    another service. ``attrs`` land on the root span (``input``, ``phase_before``, ...).
    """
    if _current_trace.get() is not None:
        raise RuntimeError("start_turn() called while a turn is already active")
    trace = TraceContext(
        trace_id=trace_id or new_trace_id(),
        session_id=session_id,
        user_id=user_id,
        turn_idx=turn_idx,
        parent_span_id=parent_span_id,
        sinks=tuple(sinks),
    )
    token = _current_trace.set(trace)
    try:
        with span(TURN_SPAN_NAME, **attrs) as root:
            trace.root = root
            yield root
    finally:
        _current_trace.reset(token)
        trace.dispatch("on_turn_end", trace)
