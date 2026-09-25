"""F5: routing — retry once on invalid output, fall back on failure/timeout, shadow mode."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import pytest

from rift_agent.extractors import (
    ExtractionContext,
    ExtractionResult,
    RoutedSlotExtractor,
    SlotExtractor,
    compare_extractions,
)
from rift_common.trace import TraceContext, span, start_turn
from rift_common.trace.sinks.base import TraceSink
from rift_domain.slots import BookingState, SlotExtraction, parse_extraction

CTX = ExtractionContext(now=datetime(2026, 10, 1, 14), current_state=BookingState(), user_input="x")
GOOD = parse_extraction(
    {"turn_intent": "booking", "delta": {"duration_hours": 2}, "confirmation": "none"}
)
OTHER = parse_extraction(
    {"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "none"}
)


def ok(extraction: SlotExtraction = GOOD, source: str = "x") -> ExtractionResult:
    return ExtractionResult(
        extraction=extraction, raw=extraction.model_dump_json(), valid=True, source=source
    )


def bad(kind: str = "invalid", source: str = "x") -> ExtractionResult:
    return ExtractionResult(
        extraction=None, raw="{bad", valid=False, errors=(f"{kind} error",),
        error_kind=kind, source=source,  # type: ignore[arg-type]
    )  # fmt: skip


class Scripted(SlotExtractor):
    """Returns the scripted results in order; ``float`` = sleep that long, ``Exception`` = raise."""

    def __init__(self, name: str, script: Sequence[Any]) -> None:
        self.name = name
        self.script = list(script)
        self.feedback: list[ExtractionResult | None] = []

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        self.feedback.append(feedback)
        step = self.script.pop(0)
        if isinstance(step, float):
            await asyncio.sleep(step)
            step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        result: ExtractionResult = step
        return result


class Recorder(TraceSink):
    def __init__(self) -> None:
        self.traces: list[TraceContext] = []

    def on_turn_end(self, trace: TraceContext) -> None:
        self.traces.append(trace)


async def _run(router: RoutedSlotExtractor) -> tuple[ExtractionResult, dict[str, Any]]:
    sink = Recorder()
    with start_turn(sinks=[sink]), span("node:extract_slots"):
        result = await router.extract(CTX)
    attrs = next(s for s in sink.traces[0].spans if s.name == "node:extract_slots").attrs
    return result, attrs


async def test_primary_valid_first_time() -> None:
    primary = Scripted("llm", [ok(source="llm")])
    result, attrs = await _run(RoutedSlotExtractor(primary))
    assert result.valid and not result.retried and not result.fell_back
    assert result.attempts == ("llm:ok",)
    assert attrs["valid"] is True and attrs["extractor"] == "llm"


async def test_invalid_retries_once_with_feedback_then_succeeds() -> None:
    primary = Scripted("local", [bad(), ok(source="local")])
    result, attrs = await _run(RoutedSlotExtractor(primary, max_retries=1))
    assert result.valid and result.retried and not result.fell_back
    assert primary.feedback[0] is None
    assert primary.feedback[1] is not None and primary.feedback[1].errors == ("invalid error",)
    assert attrs["attempts"] == ["local:invalid", "local:ok"]


async def test_invalid_twice_falls_back() -> None:
    primary = Scripted("local", [bad(), bad()])
    fallback = Scripted("llm", [ok(source="llm")])
    result, attrs = await _run(RoutedSlotExtractor(primary, fallback))
    assert result.valid and result.retried and result.fell_back
    assert result.source == "llm"
    assert result.attempts == ("local:invalid", "local:invalid", "llm:ok")
    assert attrs["fell_back"] is True


async def test_timeout_skips_retry_and_falls_back() -> None:
    primary = Scripted("local", [0.5, ok()])
    fallback = Scripted("llm", [ok(source="llm")])
    result, _ = await _run(RoutedSlotExtractor(primary, fallback, timeout_s=0.05))
    assert result.fell_back and not result.retried
    assert result.attempts == ("local:timeout", "llm:ok")
    assert len(primary.feedback) == 1


async def test_unavailable_falls_back_without_retry() -> None:
    primary = Scripted("local", [bad("unavailable")])
    fallback = Scripted("llm", [ok(source="llm")])
    result, _ = await _run(RoutedSlotExtractor(primary, fallback))
    assert result.fell_back and result.attempts == ("local:unavailable", "llm:ok")


async def test_crashing_extractor_is_contained() -> None:
    primary = Scripted("local", [RuntimeError("segfault-ish")])
    fallback = Scripted("llm", [ok(source="llm")])
    result, _ = await _run(RoutedSlotExtractor(primary, fallback))
    assert result.fell_back and result.attempts == ("local:unavailable", "llm:ok")


async def test_everything_fails() -> None:
    primary = Scripted("local", [bad(), bad()])
    fallback = Scripted("llm", [bad("timeout")])
    result, attrs = await _run(RoutedSlotExtractor(primary, fallback))
    assert not result.valid and result.extraction is None
    assert result.errors == ("invalid error", "timeout error")
    assert attrs["valid"] is False and attrs["extract_errors"] == list(result.errors)


async def test_no_fallback_no_retry_budget() -> None:
    primary = Scripted("llm", [bad()])
    result, _ = await _run(RoutedSlotExtractor(primary, max_retries=0))
    assert not result.valid and result.attempts == ("llm:invalid",)


async def test_shadow_agreement_recorded_without_changing_result() -> None:
    primary = Scripted("llm", [ok(GOOD, "llm")])
    shadow = Scripted("local", [ok(OTHER, "local")])
    result, attrs = await _run(RoutedSlotExtractor(primary, shadow=shadow))
    assert result.extraction == GOOD and result.source == "llm"
    assert attrs["shadow_valid"] is True
    assert attrs["shadow_agree"] == {
        "turn_intent": True,
        "confirmation": True,
        "delta.duration_hours": False,
    }
    assert attrs["shadow_agree_all"] is False
    assert '"duration_hours":3' in attrs["shadow_output"].replace(" ", "")


@pytest.mark.parametrize(
    "script", [[bad()], [RuntimeError("boom")], [5.0, ok()]], ids=["invalid", "crash", "slow"]
)
async def test_failing_shadow_never_affects_main(script: list[Any]) -> None:
    primary = Scripted("llm", [ok(GOOD, "llm")])
    shadow = Scripted("local", script)
    router = RoutedSlotExtractor(primary, shadow=shadow, shadow_wait_s=0.05, timeout_s=10)
    result, attrs = await _run(router)
    assert result.valid and result.extraction == GOOD
    assert attrs["shadow_valid"] is False
    assert "shadow_error" in attrs


async def test_works_outside_a_trace() -> None:
    result = await RoutedSlotExtractor(Scripted("llm", [ok()])).extract(CTX)
    assert result.valid


def test_compare_extractions() -> None:
    a = parse_extraction(
        {"turn_intent": "booking", "delta": {"game_mode": "aram"}, "confirmation": "yes"}
    )
    b = parse_extraction({"turn_intent": "consult", "delta": {}, "confirmation": "yes"})
    assert compare_extractions(a, b) == {
        "turn_intent": False,
        "confirmation": True,
        "delta.game_mode": False,
    }
    assert all(compare_extractions(a, a).values())
