"""``LLMSlotExtractor``: remote LLM in JSON mode, validated against the strict C2 schema."""

from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from rift_agent.extractors.base import (
    ErrorKind,
    ExtractionContext,
    ExtractionResult,
    SlotExtractor,
)
from rift_agent.extractors.prompt import build_messages, retry_message
from rift_common.llm import BaseLLM, LLMError, LLMTimeout, LLMUnavailable, Message
from rift_domain.slots import parse_extraction


def validation_errors(exc: ValidationError) -> tuple[str, ...]:
    out = []
    for e in exc.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        out.append(f"{loc}: {e['msg']}")
    return tuple(out)


class LLMSlotExtractor(SlotExtractor):
    name = "llm"

    def __init__(self, llm: BaseLLM, *, max_tokens: int = 300, name: str | None = None) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        if name:
            self.name = name

    def messages(
        self, ctx: ExtractionContext, feedback: ExtractionResult | None = None
    ) -> list[Message]:
        messages = build_messages(ctx)
        if feedback is not None:
            messages += [{"role": "assistant", "content": feedback.raw}]
            messages.append(retry_message(feedback.errors))
        return messages

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        t0 = time.perf_counter()

        def elapsed() -> float:
            return (time.perf_counter() - t0) * 1000

        try:
            result = await asyncio.to_thread(
                self.llm.chat,
                self.messages(ctx, feedback),
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self.max_tokens,
            )
        except LLMTimeout as exc:
            return self._failure("timeout", f"timeout: {exc}", elapsed())
        except LLMUnavailable as exc:
            return self._failure("unavailable", f"unavailable: {exc}", elapsed())
        except LLMError as exc:
            return self._failure("unavailable", f"{type(exc).__name__}: {exc}", elapsed())

        raw = result.text.strip()
        try:
            extraction = parse_extraction(raw)
        except ValidationError as exc:
            return ExtractionResult(
                extraction=None,
                raw=raw,
                valid=False,
                errors=validation_errors(exc),
                model=self.llm.model,
                latency_ms=elapsed(),
                error_kind="invalid",
                source=self.name,
            )
        return ExtractionResult(
            extraction=extraction,
            raw=raw,
            valid=True,
            model=self.llm.model,
            latency_ms=elapsed(),
            source=self.name,
        )

    def _failure(self, kind: ErrorKind, error: str, latency_ms: float) -> ExtractionResult:
        return ExtractionResult(
            extraction=None,
            raw="",
            valid=False,
            errors=(error,),
            model=self.llm.model,
            latency_ms=latency_ms,
            error_kind=kind,
            source=self.name,
        )
