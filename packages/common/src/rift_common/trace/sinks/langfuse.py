"""Langfuse Cloud sink (SDK v4, OpenTelemetry-based), mirroring spans live.

Mapping:
- turn (root span)   -> Langfuse trace (name, ``session_id``, ``user_id``; input, and the
  ``reply`` attr as output). Our 32-hex ``trace_id`` is reused so other services can
  attach spans to the same trace.
- ``kind=generation`` -> generation (model, input, output, usage_details, model_parameters)
- ``kind=tool`` / ``span`` -> tool / span observations
- remaining attrs    -> observation metadata; errors -> level ERROR + status message

Langfuse is optional: when disabled, missing keys, SDK not installed, or any SDK call
raising, the sink logs a warning and the conversation carries on untouched.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from rift_common.settings import LangfuseConfig
from rift_common.trace.context import TraceContext
from rift_common.trace.sinks.base import TraceSink
from rift_common.trace.span import Span

logger = logging.getLogger(__name__)

_AS_TYPE = {"generation": "generation", "tool": "tool", "span": "span"}
# Attributes mapped to dedicated Langfuse fields instead of metadata.
_RESERVED = frozenset({"input", "output", "reply", "model", "usage", "model_parameters"})


def _usage_details(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    mapping = {"prompt_tokens": "input", "completion_tokens": "output", "total_tokens": "total"}
    return {mapping[k]: int(v) for k, v in usage.items() if k in mapping and v is not None}


class LangfuseSink(TraceSink):
    def __init__(
        self,
        config: LangfuseConfig,
        *,
        client: Any | None = None,
        propagate_attributes: Callable[..., AbstractContextManager[Any]] | None = None,
    ) -> None:
        """``client`` / ``propagate_attributes`` default to the real SDK; tests inject fakes."""
        self.enabled = config.enabled
        self._client = client
        self._propagate = propagate_attributes
        self._observations: dict[str, Any] = {}
        self._propagations: dict[str, AbstractContextManager[Any]] = {}
        if not self.enabled or client is not None:
            return
        if not (config.public_key and config.secret_key):
            logger.warning("langfuse enabled but public/secret key missing; sink disabled")
            self.enabled = False
            return
        try:
            sdk = importlib.import_module("langfuse")
            self._client = sdk.Langfuse(
                public_key=config.public_key, secret_key=config.secret_key, host=config.host
            )
            self._propagate = sdk.propagate_attributes
        except Exception as exc:
            logger.warning("langfuse client unavailable (%s); sink disabled", exc)
            self.enabled = False

    def _safe(self, action: str, fn: Callable[[], None]) -> None:
        if not self.enabled:
            return
        try:
            fn()
        except Exception as exc:
            logger.warning("langfuse %s failed: %s", action, exc)

    # --- hooks -----------------------------------------------------------------------

    def on_span_start(self, span: Span, trace: TraceContext) -> None:
        self._safe("span start", lambda: self._start(span, trace))

    def on_span_end(self, span: Span, trace: TraceContext) -> None:
        self._safe("span end", lambda: self._end(span, trace))

    def on_turn_end(self, trace: TraceContext) -> None:
        def finish() -> None:
            cm = self._propagations.pop(trace.trace_id, None)
            if cm is not None:
                cm.__exit__(None, None, None)

        self._safe("turn end", finish)
        # Drop observations orphaned by an earlier failure so the map cannot grow.
        for span in trace.spans:
            self._observations.pop(span.span_id, None)

    def close(self) -> None:
        if self._client is not None:
            self._safe("flush", self._client.flush)

    # --- mapping ---------------------------------------------------------------------

    def _start(self, span: Span, trace: TraceContext) -> None:
        if self._client is None:
            return
        kwargs: dict[str, Any] = {
            "name": span.name,
            "as_type": _AS_TYPE.get(span.kind, "span"),
            "input": span.attrs.get("input"),
        }
        parent = self._observations.get(span.parent_id) if span.parent_id else None
        if parent is not None:
            obs = parent.start_observation(**kwargs)
        else:
            if self._propagate is not None and trace.trace_id not in self._propagations:
                cm = self._propagate(
                    session_id=trace.session_id, user_id=trace.user_id, trace_name=span.name
                )
                cm.__enter__()
                self._propagations[trace.trace_id] = cm
            trace_context = {"trace_id": trace.trace_id}
            if trace.parent_span_id:
                trace_context["parent_span_id"] = trace.parent_span_id
            obs = self._client.start_observation(trace_context=trace_context, **kwargs)
        self._observations[span.span_id] = obs

    def _end(self, span: Span, trace: TraceContext) -> None:
        obs = self._observations.pop(span.span_id, None)
        if obs is None:
            return
        attrs = span.attrs
        is_root = trace.root is span
        update: dict[str, Any] = {
            "output": attrs.get("reply") if is_root else attrs.get("output"),
            "metadata": {k: v for k, v in attrs.items() if k not in _RESERVED} or None,
        }
        if span.kind == "generation":
            update["model"] = attrs.get("model")
            update["model_parameters"] = attrs.get("model_parameters")
            update["usage_details"] = _usage_details(attrs.get("usage"))
        if span.status == "error":
            update["level"] = "ERROR"
            update["status_message"] = span.error
        # Trace-level input/output are derived from the root observation in SDK v4.
        obs.update(**{k: v for k, v in update.items() if v is not None})
        obs.end()
