"""Slot-extractor output contract and the code-side booking state.

Delta tri-state semantics (DEV_SPEC 3.2.2), per key of ``SlotDelta``:

=====================  ==================  ===============================
user said              JSON                meaning after merge
=====================  ==================  ===============================
nothing about it       key absent          keep the current value
"no preference"        ``"any"``           ANY: counts as filled, no filter
"forget that"          ``null``            cleared, missing again
a (new) value          value / new list    overwrite (lists replaced whole)
=====================  ==================  ===============================

"Absent" vs ``null`` is told apart with ``model_fields_set`` (see ``SlotDelta.provided``).
The extractor models are strict: extra keys, unknown enum values and wrong JSON types
(``"2"`` for a number, ``1`` for a bool) all fail validation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Final, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

from rift_domain.enums import (
    Confirmation,
    GameMode,
    Gender,
    PendingAction,
    Rank,
    Role,
    ServiceType,
    SlotField,
    TurnIntent,
)

ANY: Final = "any"
AnyValue = Literal["any"]


def _unique_roles(roles: list[Role]) -> list[Role]:
    if len(set(roles)) != len(roles):
        raise ValueError("role_preference must not repeat roles")
    return roles


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Hours = Annotated[float, Field(gt=0, le=24, multiple_of=0.5)]
Budget = Annotated[float, Field(gt=0, le=10000)]
RoleList = Annotated[list[Role], Field(min_length=1, max_length=5), AfterValidator(_unique_roles)]

_EXTRACTOR_CONFIG = ConfigDict(extra="forbid", strict=True, frozen=True)


class SlotDelta(BaseModel):
    """What the user said about each slot *this turn*. Absent key = not mentioned."""

    model_config = _EXTRACTOR_CONFIG

    game_mode: GameMode | None = None
    start_time_expr: Text | None = None
    duration_hours: Hours | None = None
    rank_requirement: Rank | AnyValue | None = None
    role_preference: RoleList | AnyValue | None = None
    service_type: ServiceType | None = None
    companion_gender: Gender | AnyValue | None = None
    voice_required: bool | AnyValue | None = None
    budget_per_hour: Budget | AnyValue | None = None
    style_preference: Text | AnyValue | None = None
    companion_name: Text | AnyValue | None = None

    def provided(self) -> dict[str, Any]:
        """Only the keys the extractor actually emitted (``None`` here means "clear")."""
        return {name: getattr(self, name) for name in self.model_fields_set}

    def is_empty(self) -> bool:
        return not self.model_fields_set


class SlotExtraction(BaseModel):
    """One turn of slot-extractor output."""

    model_config = _EXTRACTOR_CONFIG

    turn_intent: TurnIntent
    delta: SlotDelta
    confirmation: Confirmation


def parse_extraction(raw: str | bytes | Mapping[str, Any]) -> SlotExtraction:
    """Validate extractor output (JSON text or an already-decoded dict).

    Dicts go through JSON so enum strings validate under strict mode exactly like model
    output does. Raises ``pydantic.ValidationError`` on any protocol violation.
    """
    text = raw if isinstance(raw, str | bytes) else json.dumps(raw, ensure_ascii=False)
    return SlotExtraction.model_validate_json(text)


def extraction_json_schema() -> dict[str, Any]:
    """JSON Schema of ``SlotExtraction`` for LLM response constraints and training data."""
    return SlotExtraction.model_json_schema()


# --- code-side state ---------------------------------------------------------------------

_STATE_CONFIG = ConfigDict(extra="forbid", frozen=True)


class Candidate(BaseModel):
    """A companion shown to the user, so "the second one" can be resolved later."""

    model_config = _STATE_CONFIG

    companion_id: int
    name: str
    score: float = 0.0
    reasons: tuple[str, ...] = ()


class Quote(BaseModel):
    """Price computed by code (never by the model)."""

    model_config = _STATE_CONFIG

    unit_price: Decimal
    service_type: ServiceType
    multiplier: Decimal
    hours: Decimal
    effective_hourly: Decimal
    total: Decimal


class BookingState(BaseModel):
    """Merged slots plus fields derived by code."""

    model_config = _STATE_CONFIG

    # merged slots
    game_mode: GameMode | None = None
    start_time: datetime | None = None
    #: Raw expression behind ``start_time`` (kept when it could not be resolved yet).
    start_time_expr: str | None = None
    duration_hours: float | None = None
    rank_requirement: Rank | AnyValue | None = None
    role_preference: tuple[Role, ...] | AnyValue | None = None
    service_type: ServiceType | None = None
    companion_gender: Gender | AnyValue | None = None
    voice_required: bool | AnyValue | None = None
    budget_per_hour: float | AnyValue | None = None
    style_preference: str | AnyValue | None = None
    companion_name: str | AnyValue | None = None

    # derived by code
    service_type_effective: ServiceType | None = None
    missing_fields: tuple[SlotField, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    selected_companion_id: int | None = None
    quote: Quote | None = None
    relaxations: tuple[str, ...] = ()
    pending_action: PendingAction | None = None

    def slot(self, field: SlotField) -> Any:
        return getattr(self, field.value)

    def is_filled(self, field: SlotField) -> bool:
        """Set to a value or ANY (ANY satisfies required-field checks)."""
        return self.slot(field) is not None
