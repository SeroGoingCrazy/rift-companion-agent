"""Trace sink interface.

Sinks are span processors: hooks fire when a span starts/ends and when the whole turn is
finished. Batch sinks (SQLite) only need ``on_turn_end``; live sinks (Langfuse) mirror
spans as they happen. Exceptions raised by a sink are caught and logged by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rift_common.trace.context import TraceContext
    from rift_common.trace.span import Span


class TraceSink:
    def on_span_start(self, span: Span, trace: TraceContext) -> None:
        """Called when a span opens (before its body runs)."""

    def on_span_end(self, span: Span, trace: TraceContext) -> None:
        """Called when a span finishes."""

    def on_turn_end(self, trace: TraceContext) -> None:
        """Called once after the root ``turn`` span finishes."""

    def close(self) -> None:
        """Flush and release resources."""
