"""Task layer of the L2 score, and the combined per-sample verdict.

Fields compared: ``turn_intent``, ``confirmation`` and every delta key that *either* side
emitted. A key only one side emitted is wrong (``missing_key`` / ``extra_key``), so an
extra key for something the user never mentioned costs as much as a wrong value: in
tri-state semantics "absent" means "keep", and emitting it changes the merged state.

Values are compared by meaning, not by spelling:

- ``start_time_expr``: both expressions go through the C3 parser exactly as ``merge``
  would (with ``now`` and the current state), so "明晚8点" == "明天晚上八点"; when the
  expected one does not resolve to a time either, the normalized texts must match.
- ``role_preference``: as a set (order carries no meaning).
- ``style_preference``: a ``StyleMatcher`` (embedding similarity, see ``preferences``).
- everything else (enums, numbers, bools, names, ``"any"``, ``null``): equality.

Task score = correct fields / compared fields; a sample passes when its protocol check
passes and its task score is >= 0.95 (with 2-6 fields per sample: every field right).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Literal

from rift_domain.merge import merge
from rift_domain.slots import ANY, BookingState, SlotDelta, SlotExtraction
from rift_training.evaluation.preferences import ExactStyleMatcher, StyleMatcher
from rift_training.evaluation.protocol import ProtocolResult, check_protocol

PASS_THRESHOLD: Final = 0.95
ABSENT: Final = "<absent>"

FieldError = Literal["missing_key", "extra_key", "wrong_value"]


@dataclass(frozen=True)
class FieldResult:
    #: "turn_intent", "confirmation" or "delta.<key>"
    field: str
    expected: Any
    predicted: Any
    correct: bool
    error: FieldError | None = None
    detail: str = ""


@dataclass(frozen=True)
class TaskScore:
    fields: tuple[FieldResult, ...]

    @property
    def score(self) -> float:
        if not self.fields:
            return 1.0
        return sum(f.correct for f in self.fields) / len(self.fields)

    @property
    def errors(self) -> tuple[FieldResult, ...]:
        return tuple(f for f in self.fields if not f.correct)


@dataclass(frozen=True)
class SampleScore:
    protocol: ProtocolResult
    task: TaskScore | None = None
    threshold: float = PASS_THRESHOLD

    @property
    def task_score(self) -> float:
        return self.task.score if self.task is not None else 0.0

    @property
    def passed(self) -> bool:
        return self.protocol.ok and self.task is not None and self.task.score >= self.threshold


# --- value comparison ----------------------------------------------------------------------

_SPACES = re.compile(r"[\s，。,.!！?？]+")


def _norm(text: str) -> str:
    return _SPACES.sub("", text).lower()


def resolve_time(expr: str, state: BookingState, now: datetime) -> tuple[datetime | None, str]:
    """What ``merge`` makes of ``expr``: (resolved start time or None, stored expression)."""
    merged, _ = merge(state, SlotDelta(start_time_expr=expr), now)
    return merged.start_time, merged.start_time_expr or expr


def _compare_time(
    expected: str | None, predicted: str | None, state: BookingState, now: datetime
) -> tuple[bool, str]:
    if expected is None or predicted is None:
        return expected is None and predicted is None, ""
    exp_t, exp_expr = resolve_time(expected, state, now)
    pred_t, pred_expr = resolve_time(predicted, state, now)
    detail = f"{exp_t or 'unresolved'} vs {pred_t or 'unresolved'}"
    if exp_t is not None:
        return pred_t == exp_t, detail
    return pred_t is None and _norm(exp_expr) == _norm(pred_expr), detail


def _compare_roles(expected: Any, predicted: Any) -> bool:
    if isinstance(expected, list) and isinstance(predicted, list):
        return set(expected) == set(predicted)
    return bool(expected == predicted)


def _compare_style(expected: Any, predicted: Any, style: StyleMatcher) -> tuple[bool, str]:
    if (
        isinstance(expected, str)
        and isinstance(predicted, str)
        and ANY not in (expected, predicted)
    ):
        sim = style.similarity(expected, predicted)
        return sim >= style.threshold, f"similarity {sim:.3f} (threshold {style.threshold})"
    return expected == predicted, ""


def compare_value(
    key: str,
    expected: Any,
    predicted: Any,
    *,
    state: BookingState,
    now: datetime,
    style: StyleMatcher,
) -> tuple[bool, str]:
    """(correct, detail) for one delta key both sides emitted."""
    if key == "start_time_expr":
        return _compare_time(expected, predicted, state, now)
    if key == "role_preference":
        return _compare_roles(expected, predicted), ""
    if key == "style_preference":
        return _compare_style(expected, predicted, style)
    if key == "companion_name" and isinstance(expected, str) and isinstance(predicted, str):
        return expected.strip() == predicted.strip(), ""
    if isinstance(expected, bool) or isinstance(predicted, bool):
        return expected is predicted, ""
    if isinstance(expected, int | float) and isinstance(predicted, int | float):
        return abs(float(expected) - float(predicted)) < 1e-9, ""
    return bool(expected == predicted), ""


# --- scoring -------------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return getattr(value, "value", value)  # StrEnum -> str


def score_task(
    expected: SlotExtraction,
    predicted: SlotExtraction,
    *,
    state: BookingState,
    now: datetime,
    style: StyleMatcher | None = None,
) -> TaskScore:
    style = style or ExactStyleMatcher()
    results: list[FieldResult] = []
    for name in ("turn_intent", "confirmation"):
        exp, pred = getattr(expected, name).value, getattr(predicted, name).value
        results.append(
            FieldResult(name, exp, pred, exp == pred, None if exp == pred else "wrong_value")
        )

    exp_delta = {k: _jsonable(v) for k, v in expected.delta.provided().items()}
    pred_delta = {k: _jsonable(v) for k, v in predicted.delta.provided().items()}
    for key in sorted(exp_delta.keys() | pred_delta.keys()):
        name = f"delta.{key}"
        if key not in pred_delta:
            results.append(FieldResult(name, exp_delta[key], ABSENT, False, "missing_key"))
        elif key not in exp_delta:
            results.append(FieldResult(name, ABSENT, pred_delta[key], False, "extra_key"))
        else:
            ok, detail = compare_value(
                key, exp_delta[key], pred_delta[key], state=state, now=now, style=style
            )
            results.append(
                FieldResult(
                    name,
                    exp_delta[key],
                    pred_delta[key],
                    ok,
                    None if ok else "wrong_value",
                    detail,
                )
            )
    return TaskScore(tuple(results))


def score_output(
    raw: str,
    expected: SlotExtraction,
    *,
    state: BookingState,
    now: datetime,
    style: StyleMatcher | None = None,
    threshold: float = PASS_THRESHOLD,
) -> SampleScore:
    """Protocol check of the raw model output, then the task score if it parsed."""
    protocol = check_protocol(raw)
    if protocol.extraction is None:
        return SampleScore(protocol, None, threshold)
    task = score_task(expected, protocol.extraction, state=state, now=now, style=style)
    return SampleScore(protocol, task, threshold)
