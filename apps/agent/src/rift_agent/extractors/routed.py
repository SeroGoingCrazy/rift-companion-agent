"""``RoutedSlotExtractor``: primary -> retry -> fallback -> give up, plus shadow mode.

1. ``primary.extract`` (bounded by ``timeout_s``) and strict validation;
2. invalid output -> retry with the validation errors as feedback (``max_retries``);
3. still invalid / timeout / unreachable -> ``fallback.extract`` if configured;
4. everything failed -> ``ExtractionResult.failed`` (the graph asks the user to rephrase
   and keeps its state);
5. ``shadow`` runs concurrently with the primary; its output and a per-field agreement
   map are written to the current span only (``shadow_output``, ``shadow_agree``). A
   failing or slow shadow never changes the returned result.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import replace
from typing import Any

from rift_agent.extractors.base import ExtractionContext, ExtractionResult, SlotExtractor
from rift_common.trace import current_span
from rift_domain.slots import SlotExtraction

logger = logging.getLogger(__name__)


def compare_extractions(a: SlotExtraction, b: SlotExtraction) -> dict[str, bool]:
    """Per-field agreement: intent, confirmation and every delta key either side emitted."""
    agree = {
        "turn_intent": a.turn_intent == b.turn_intent,
        "confirmation": a.confirmation == b.confirmation,
    }
    da, db = a.delta.provided(), b.delta.provided()
    for key in sorted(set(da) | set(db)):
        agree[f"delta.{key}"] = key in da and key in db and da[key] == db[key]
    return agree


class RoutedSlotExtractor(SlotExtractor):
    name = "routed"

    def __init__(
        self,
        primary: SlotExtractor,
        fallback: SlotExtractor | None = None,
        shadow: SlotExtractor | None = None,
        *,
        max_retries: int = 1,
        timeout_s: float = 5.0,
        shadow_wait_s: float = 2.0,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.shadow = shadow
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.shadow_wait_s = shadow_wait_s

    async def _attempt(
        self,
        extractor: SlotExtractor,
        ctx: ExtractionContext,
        feedback: ExtractionResult | None = None,
    ) -> ExtractionResult:
        try:
            return await asyncio.wait_for(
                extractor.extract(ctx, feedback=feedback), timeout=self.timeout_s
            )
        except TimeoutError:
            return replace(
                ExtractionResult.failed(
                    (f"timeout after {self.timeout_s}s",), source=extractor.name
                ),
                error_kind="timeout",
            )
        except Exception as exc:  # an extractor must not break the turn
            logger.exception("extractor %s crashed", extractor.name)
            return replace(
                ExtractionResult.failed((f"{type(exc).__name__}: {exc}",), source=extractor.name),
                error_kind="unavailable",
            )

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        shadow_task = asyncio.create_task(self._attempt(self.shadow, ctx)) if self.shadow else None
        attempts: list[str] = []

        result = await self._attempt(self.primary, ctx, feedback)
        attempts.append(f"{self.primary.name}:{'ok' if result.valid else result.error_kind}")
        retries = 0
        while not result.valid and result.error_kind == "invalid" and retries < self.max_retries:
            retries += 1
            result = await self._attempt(self.primary, ctx, result)
            attempts.append(f"{self.primary.name}:{'ok' if result.valid else result.error_kind}")

        fell_back = False
        if not result.valid and self.fallback is not None:
            fb = await self._attempt(self.fallback, ctx)
            attempts.append(f"{self.fallback.name}:{'ok' if fb.valid else fb.error_kind}")
            if fb.valid:
                result, fell_back = fb, True
            else:
                result = replace(result, errors=result.errors + fb.errors)

        final = replace(result, retried=retries > 0, fell_back=fell_back, attempts=tuple(attempts))
        self._record(final)
        if shadow_task is not None:
            await self._record_shadow(shadow_task, final)
        return final

    def _record(self, r: ExtractionResult) -> None:
        s = current_span()
        if s is None:
            return
        s.set_attrs(
            model=r.model,
            extractor=r.source,
            raw_output=r.raw,
            valid=r.valid,
            retried=r.retried,
            fell_back=r.fell_back,
            attempts=list(r.attempts),
            extract_latency_ms=round(r.latency_ms, 1),
        )
        if not r.valid:
            s.set_attr("extract_errors", list(r.errors))

    async def _record_shadow(
        self, task: asyncio.Task[ExtractionResult], main: ExtractionResult
    ) -> None:
        attrs: dict[str, Any] = {}
        try:
            shadow = await asyncio.wait_for(asyncio.shield(task), timeout=self.shadow_wait_s)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            attrs = {"shadow_valid": False, "shadow_error": "timeout"}
        except Exception as exc:  # pragma: no cover - _attempt already catches
            attrs = {"shadow_valid": False, "shadow_error": str(exc)}
        else:
            attrs = {
                "shadow_output": shadow.raw,
                "shadow_valid": shadow.valid,
                "shadow_latency_ms": round(shadow.latency_ms, 1),
            }
            if not shadow.valid:
                attrs["shadow_error"] = "; ".join(shadow.errors)
            if shadow.extraction is not None and main.extraction is not None:
                agree = compare_extractions(main.extraction, shadow.extraction)
                attrs["shadow_agree"] = agree
                attrs["shadow_agree_all"] = all(agree.values())
        s = current_span()
        if s is not None:
            s.set_attrs(**attrs)
