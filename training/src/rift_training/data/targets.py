"""Concrete target values for generation tasks (specs ``values`` section, v0.2+).

Left alone, the Teacher picks the most typical value every time (v0.1: diamond for almost
every rank, 2.5 h for most durations, "明晚八点" for a fifth of all times). With a
``values`` section, every slot a task needs gets a target drawn from a configured
distribution -- for the user's new values and for what is already in the state:

- enum / number / bool slots: the exact value (the answer schema pins it);
- ``role_preference``: a set of 1-3 roles;
- ``start_time_expr``: a target datetime plus a wording form (relative day, weekday,
  next week, digits, "X 小时后", or a shift of the current time); the validator parses
  the Teacher's wording with the online time parser and requires that exact datetime;
- ``companion_name``: the 1-based position of the chosen candidate;
- ``style_preference``: a theme to write about (not checked).

Targets use their own RNG seeded per task id, so adding them never changes the task shapes
drawn by ``specs.sample_tasks``.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rift_domain.enums import Gender, Rank, Role, ServiceType

TimeForm = Literal["relative_day", "weekday", "next_week", "digits", "after", "shift", "fuzzy"]
TIME_FORMAT = "%Y-%m-%d %H:%M"
#: Wording forms an absolute (non-shift) target may use.
ABSOLUTE_FORMS = ("relative_day", "weekday", "next_week", "digits", "after")
MIN_LEAD = timedelta(minutes=30)

_FROZEN = ConfigDict(extra="forbid", frozen=True)


def _positive(weights: Mapping[Any, float]) -> None:
    if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")


class TimeTargetSpec(BaseModel):
    model_config = _FROZEN

    forms: dict[str, float]
    hours: dict[int, float]
    minutes: dict[int, float] = Field(default_factory=lambda: {0: 0.7, 30: 0.3})
    day_offsets: dict[int, float] = Field(default_factory=lambda: {0: 0.4, 1: 0.45, 2: 0.15})
    after_hours: dict[float, float] = Field(default_factory=lambda: {0.5: 1.0, 1.0: 1.0, 2.0: 1.0})
    shift_hours: dict[float, float] = Field(default_factory=lambda: {1.0: 1.0, -0.5: 1.0})
    #: Share of modified times written as a shift ("晚一小时") rather than a new time.
    modify_shift: float = Field(default=0.5, ge=0, le=1)

    @field_validator("forms")
    @classmethod
    def _forms(cls, value: dict[str, float]) -> dict[str, float]:
        _positive(value)
        unknown = set(value) - set(ABSOLUTE_FORMS)
        if unknown:
            raise ValueError(f"unknown time forms {sorted(unknown)}")
        return value

    @field_validator("hours")
    @classmethod
    def _hours(cls, value: dict[int, float]) -> dict[int, float]:
        _positive(value)
        if not all(0 <= h <= 23 for h in value):
            raise ValueError("hours must lie in [0, 23]")
        return value


class ValueSpec(BaseModel):
    model_config = _FROZEN

    rank_requirement: dict[Rank, float]
    duration_hours: dict[float, float]
    roles: dict[Role, float]
    role_count: dict[int, float] = Field(default_factory=lambda: {1: 0.7, 2: 0.3})
    service_type: dict[ServiceType, float]
    companion_gender: dict[Gender, float]
    voice_required: dict[bool, float]
    budget_per_hour: dict[int, float]
    style_themes: list[str] = Field(min_length=1)
    time: TimeTargetSpec

    @field_validator(
        "rank_requirement",
        "duration_hours",
        "roles",
        "role_count",
        "service_type",
        "companion_gender",
        "voice_required",
        "budget_per_hour",
    )
    @classmethod
    def _weights(cls, value: dict[Any, float]) -> dict[Any, float]:
        _positive(value)
        return value


def task_rng(seed: int, task_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _choice(rng: random.Random, weights: Mapping[Any, float]) -> Any:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _value(v: Any) -> Any:
    return v.value if hasattr(v, "value") else v


class TargetDrawer:
    def __init__(self, spec: ValueSpec, rng: random.Random, now: datetime) -> None:
        self.spec = spec
        self.rng = rng
        self.now = now

    # --- times -------------------------------------------------------------------------

    def _clock(self, day: datetime) -> datetime:
        hour = _choice(self.rng, self.spec.time.hours)
        minute = _choice(self.rng, self.spec.time.minutes)
        return day.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def absolute_time(self, form: str) -> datetime:
        now, t = self.now, self.spec.time
        if form == "after":
            return now + timedelta(hours=_choice(self.rng, t.after_hours))
        if form == "weekday":
            day = now + timedelta(days=self.rng.randint(0, 6))
        elif form == "next_week":
            day = now + timedelta(days=7 - now.weekday() + self.rng.randint(0, 6))
        else:  # relative_day / digits
            day = now + timedelta(days=_choice(self.rng, t.day_offsets))
        target = self._clock(day)
        if target < now + MIN_LEAD:
            if form == "weekday":  # "周X" already today -> next week's day is "下周X"
                target += timedelta(days=7)
            elif form in ("relative_day", "digits"):
                target += timedelta(days=1)
            else:
                target = self._clock(now + timedelta(days=1))
        return target

    def time_target(self, *, base: datetime | None, shift: bool) -> dict[str, Any]:
        if shift and base is not None:
            for _ in range(10):
                hours = _choice(self.rng, self.spec.time.shift_hours)
                target = base + timedelta(hours=hours)
                if target >= self.now + MIN_LEAD:
                    return {
                        "form": "shift",
                        "shift_hours": hours,
                        "time": f"{target:{TIME_FORMAT}}",
                    }
        form = _choice(self.rng, self.spec.time.forms)
        target = self.absolute_time(form)
        if base is not None and target == base:
            target += timedelta(hours=1)
        return {"form": form, "time": f"{target:{TIME_FORMAT}}"}

    # --- other slots -------------------------------------------------------------------

    def value(self, name: str, num_candidates: int) -> Any:
        s = self.spec
        if name == "rank_requirement":
            return _value(_choice(self.rng, s.rank_requirement))
        if name == "duration_hours":
            return _choice(self.rng, s.duration_hours)
        if name == "role_preference":
            n = min(_choice(self.rng, s.role_count), len(s.roles))
            pool = dict(s.roles)
            roles = []
            for _ in range(n):
                role = _choice(self.rng, pool)
                roles.append(_value(role))
                del pool[role]
            return roles
        if name == "service_type":
            return _value(_choice(self.rng, s.service_type))
        if name == "companion_gender":
            return _value(_choice(self.rng, s.companion_gender))
        if name == "voice_required":
            return _choice(self.rng, s.voice_required)
        if name == "budget_per_hour":
            return _choice(self.rng, s.budget_per_hour)
        if name == "style_preference":
            return self.rng.choice(s.style_themes)
        if name == "companion_name":
            return self.rng.randint(1, max(num_candidates, 1))
        raise KeyError(name)


def _differs(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return set(a) != set(b)
    return bool(a != b)


def draw_targets(
    spec: ValueSpec,
    rng: random.Random,
    *,
    now: datetime,
    variant: str,
    state_fields: tuple[str, ...],
    delta: Mapping[str, str],
    num_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(state targets, delta value targets); keys are delta field names, no ``game_mode``."""
    drawer = TargetDrawer(spec, rng, now)
    state: dict[str, Any] = {}
    for name in state_fields:
        if name == "game_mode":
            continue
        if name == "start_time_expr":
            form = _choice(rng, {"relative_day": 0.6, "weekday": 0.4})
            state[name] = {"time": f"{drawer.absolute_time(form):{TIME_FORMAT}}"}
        elif name == "style_preference":
            state[name] = rng.choice(spec.style_themes)
        else:
            state[name] = drawer.value(name, num_candidates)

    values: dict[str, Any] = {}
    for name, kind in delta.items():
        if kind != "value" or name == "game_mode":
            continue
        if name == "style_preference" and variant.startswith("voice_"):
            continue  # the variant itself says what to write (a voice description)
        if name == "start_time_expr":
            if variant == "fuzzy":
                values[name] = {"form": "fuzzy"}
                continue
            base = datetime.strptime(state[name]["time"], TIME_FORMAT) if name in state else None
            shift = base is not None and (
                variant == "shift" or rng.random() < spec.time.modify_shift
            )
            values[name] = drawer.time_target(base=base, shift=shift)
            continue
        target = drawer.value(name, num_candidates)
        for _ in range(20):  # a modified slot must change
            if name not in state or _differs(target, state[name]):
                break
            target = drawer.value(name, num_candidates)
        values[name] = target
    return state, values
