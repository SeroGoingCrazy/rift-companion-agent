"""Slot extractor interface: ``extract(ctx) -> ExtractionResult``.

An extractor never raises for bad model output or an unreachable model: it returns a
result with ``valid=False``, readable ``errors`` and an ``error_kind`` the router uses to
decide between retrying (``invalid``) and falling back (``timeout`` / ``unavailable``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from rift_common.llm.types import Message
from rift_domain.slots import BookingState, SlotExtraction

ErrorKind = Literal["invalid", "timeout", "unavailable"]


@dataclass(frozen=True)
class ExtractionContext:
    now: datetime
    current_state: BookingState
    user_input: str
    #: Names of the candidates shown to the user, in display order.
    candidates: tuple[str, ...] = ()
    history: tuple[Message, ...] = ()
    pending_confirmation: bool = False


@dataclass(frozen=True)
class ExtractionResult:
    extraction: SlotExtraction | None
    raw: str
    valid: bool
    errors: tuple[str, ...] = ()
    model: str = ""
    latency_ms: float = 0.0
    error_kind: ErrorKind | None = None
    #: Which extractor produced this result ("llm", "local", ...).
    source: str = ""
    #: Set by the router.
    retried: bool = False
    fell_back: bool = False
    attempts: tuple[str, ...] = field(default=())

    @classmethod
    def failed(cls, errors: tuple[str, ...] = (), *, source: str = "") -> ExtractionResult:
        return cls(extraction=None, raw="", valid=False, errors=errors, source=source)


class SlotExtractor(ABC):
    name: str = ""

    @abstractmethod
    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        """Extract this turn's slots; ``feedback`` is the previous invalid attempt."""
