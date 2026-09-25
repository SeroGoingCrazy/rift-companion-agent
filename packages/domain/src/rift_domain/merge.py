"""Merge a turn's ``SlotDelta`` into the ``BookingState`` (tri-state semantics).

Besides copying values, merging keeps derived state honest: anything computed from the
old slots (quote, shown candidates, a pending confirmation) is invalidated when the slots
it depended on change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rift_domain.slots import BookingState, SlotDelta
from rift_domain.timeparse import TimeParseResult, parse_time_expr

#: Slots whose change makes the current price quote stale.
QUOTE_FIELDS = frozenset({"companion_name", "duration_hours", "service_type", "game_mode"})
#: Slots whose change makes the shown candidate list stale.
FILTER_FIELDS = frozenset(
    {
        "game_mode",
        "start_time",
        "duration_hours",
        "rank_requirement",
        "role_preference",
        "service_type",
        "companion_gender",
        "voice_required",
        "budget_per_hour",
        "style_preference",
    }
)


@dataclass(frozen=True)
class StateDiff:
    #: slot name -> (old, new); only slots whose value actually changed.
    changed: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    #: derived fields reset because their inputs changed.
    invalidated: tuple[str, ...] = ()
    #: outcome of resolving ``start_time_expr`` this turn, if one was given.
    time_result: TimeParseResult | None = None

    @property
    def changed_fields(self) -> tuple[str, ...]:
        return tuple(self.changed)

    def __bool__(self) -> bool:
        return bool(self.changed or self.invalidated)


def _resolve_time(state: BookingState, expr: str, now: datetime) -> tuple[str, TimeParseResult]:
    """Parse ``expr``; if an earlier expression is still unresolved, try completing it.

    "明晚" (ambiguous) followed by "八点" resolves as "明晚八点".
    """
    if state.start_time is None and state.start_time_expr:
        combined = state.start_time_expr + expr
        result = parse_time_expr(combined, now)
        if result.ok:
            return combined, result
    return expr, parse_time_expr(expr, now, current=state.start_time)


def merge(state: BookingState, delta: SlotDelta, now: datetime) -> tuple[BookingState, StateDiff]:
    """Apply ``delta`` to ``state``; returns the new state and what changed.

    - key absent -> unchanged; ``"any"`` -> ANY; ``null`` -> cleared; value -> overwrite
      (lists are replaced whole).
    - ``start_time_expr`` is resolved with the time parser (relative to the current
      ``start_time``); an unresolved/past expression leaves ``start_time`` empty so the
      agent asks again, and ``diff.time_result`` says why.
    """
    updates: dict[str, Any] = {}
    time_result: TimeParseResult | None = None

    for key, value in delta.provided().items():
        if key == "start_time_expr":
            if value is None:
                updates["start_time_expr"] = None
                updates["start_time"] = None
            else:
                expr, time_result = _resolve_time(state, value, now)
                updates["start_time_expr"] = expr
                updates["start_time"] = time_result.value if time_result.ok else None
            continue
        if isinstance(value, list):
            value = tuple(value)
        updates[key] = value

    changed = {
        key: (getattr(state, key), new)
        for key, new in updates.items()
        if getattr(state, key) != new
    }
    slot_changes = set(changed) - {"start_time_expr"}

    resets: dict[str, Any] = {}
    if slot_changes & QUOTE_FIELDS and state.quote is not None:
        resets["quote"] = None
    if "companion_name" in slot_changes and state.selected_companion_id is not None:
        resets["selected_companion_id"] = None
    if slot_changes & FILTER_FIELDS:
        if state.candidates:
            resets["candidates"] = ()
        if state.relaxations:
            resets["relaxations"] = ()
        if state.selected_companion_id is not None:
            resets["selected_companion_id"] = None
    if slot_changes and state.pending_action is not None:
        resets["pending_action"] = None

    new_state = state.model_copy(update={k: new for k, (_, new) in changed.items()} | resets)
    diff = StateDiff(changed=changed, invalidated=tuple(sorted(resets)), time_result=time_result)
    return new_state, diff
