"""Turn a Teacher answer into a training sample and check it (DEV_SPEC I3).

``assemble`` builds an L2-format sample (``SlotSample``) from the task and the answer: the
delta keys and their tri-state kinds come from the task, only "value" entries from the
Teacher. ``validate_answer`` then applies, in order:

1. the answer shape (JSON object, exact state / values keys, history and candidate counts);
2. the C2 contract and the L2 consistency rules (``SlotSample`` + ``sample_problems``);
3. business consistency with ``domain.yaml`` and the C3 time parser -- durations on the
   0.5 h grid within limits, the time expression copied verbatim and parseable, state
   times in the future, the conversation's game mode, names taken from the candidates,
   modified slots actually changed.

A sample is accepted only when the list of problems is empty.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from rift_domain.config import DomainConfig
from rift_domain.timeparse import Reason, parse_time_expr
from rift_training.data.prompting import state_key
from rift_training.data.specs import GenerationTask
from rift_training.data.targets import TIME_FORMAT
from rift_training.evaluation.dataset import SlotSample, sample_problems

#: Parser outcomes that mean the expression cannot become a booking time at all. Vague or
#: ambiguous expressions are kept: the agent asks again, and the extractor still copies them.
BAD_TIME_REASONS = frozenset(
    {Reason.EMPTY, Reason.UNRECOGNIZED, Reason.INVALID, Reason.NO_CURRENT, Reason.PAST}
)
ANSWER_KEYS = frozenset({"candidates", "state", "history", "user_input", "values"})


@dataclass
class Outcome:
    task_id: str
    sample: SlotSample | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.sample is not None and not self.problems


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def assemble(task: GenerationTask, answer: dict[str, Any]) -> dict[str, Any]:
    """The L2-format row for this task (``SlotSample`` fields)."""
    delta: dict[str, Any] = {}
    for name, kind in task.delta.items():
        delta[name] = (
            answer["values"][name] if kind == "value" else ("any" if kind == "any" else None)
        )
    state = {state_key(n): answer["state"][state_key(n)] for n in task.state_fields}
    return {
        "id": task.id,
        "category": task.category,
        "now": f"{task.now:%Y-%m-%d %H:%M}",
        "current_state": state,
        "candidates": answer["candidates"],
        "pending_confirmation": task.pending_confirmation,
        "history": answer["history"],
        "user_input": answer["user_input"].strip(),
        "expected": {
            "turn_intent": task.turn_intent.value,
            "delta": delta,
            "confirmation": task.confirmation.value,
        },
        "note": f"{task.variant}; {task.style}",
    }


def shape_problems(task: GenerationTask, answer: Any) -> list[str]:
    if not isinstance(answer, dict):
        return ["answer is not a JSON object"]
    problems: list[str] = []
    if set(answer) != ANSWER_KEYS:
        problems.append(f"answer keys {sorted(answer)} != {sorted(ANSWER_KEYS)}")
        return problems
    want_state = {state_key(n) for n in task.state_fields}
    want_values = {n for n, k in task.delta.items() if k == "value"}
    if not isinstance(answer["state"], dict) or set(answer["state"]) != want_state:
        problems.append(f"state keys must be {sorted(want_state)}")
    if not isinstance(answer["values"], dict) or set(answer["values"]) != want_values:
        problems.append(f"values keys must be {sorted(want_values)}")
    if not isinstance(answer["history"], list) or len(answer["history"]) != 2 * task.history_turns:
        problems.append(f"history must have {2 * task.history_turns} messages")
    candidates = answer["candidates"]
    if not isinstance(candidates, list) or len(candidates) != task.num_candidates:
        problems.append(f"candidates must have {task.num_candidates} names")
    elif any(not isinstance(c, str) or not c.strip() for c in candidates):
        problems.append("candidate names must be non-empty strings")
    elif len(set(candidates)) != len(candidates):
        problems.append("candidate names must be unique")
    if not isinstance(answer["user_input"], str) or not answer["user_input"].strip():
        problems.append("user_input is empty")
    return problems


def _history_problems(sample: SlotSample) -> list[str]:
    for i, message in enumerate(sample.history):
        expected = "user" if i % 2 == 0 else "assistant"
        if message.role != expected:
            return [f"history[{i}] should be a {expected} message"]
    return []


def business_problems(task: GenerationTask, sample: SlotSample, domain: DomainConfig) -> list[str]:
    problems = _history_problems(sample)
    state = sample.current_state
    delta = sample.expected["delta"]
    values = {n: delta[n] for n, k in task.delta.items() if k == "value"}
    current = sample.state()

    # durations on the configured grid
    for where, data in (("state", state), ("delta", values)):
        hours = data.get("duration_hours")
        if hours is not None and not domain.duration.is_valid(hours):
            problems.append(f"{where} duration_hours {hours} outside the allowed hours")

    # times: future state time, verbatim and parseable expression
    if current.start_time is not None and current.start_time <= sample.now:
        problems.append("state start_time is not after now")
    expr = values.get("start_time_expr")
    if isinstance(expr, str):
        if _squash(expr) not in _squash(sample.user_input):
            problems.append(f"start_time_expr {expr!r} is not copied from user_input")
        result = parse_time_expr(expr, sample.now, current=current.start_time)
        if result.reason in BAD_TIME_REASONS:
            problems.append(f"start_time_expr {expr!r} does not parse ({result.reason})")

    # the conversation's game mode
    if task.game_mode is not None:
        mode = values.get("game_mode", state.get("game_mode"))
        if mode != task.game_mode.value:
            problems.append(f"game_mode {mode!r} is not the task's {task.game_mode.value}")

    # names come from the candidates
    if sample.candidates:
        for where, data in (("state", state), ("delta", values)):
            name = data.get("companion_name")
            if isinstance(name, str) and name not in sample.candidates:
                problems.append(f"{where} companion_name {name!r} is not a candidate")

    # a modified slot must actually change
    for name, value in values.items():
        key = state_key(name)
        if key in state and name != "start_time_expr" and _same(state[key], value):
            problems.append(f"{name} is modified to its current value {value!r}")
    return problems


def target_problems(task: GenerationTask, sample: SlotSample) -> list[str]:
    """The answer must realize the task's target values (v0.2+ specs)."""
    problems: list[str] = []
    state = sample.current_state
    delta = sample.expected["delta"]
    current = sample.state()

    def pick(index: int) -> str | None:
        return sample.candidates[index - 1] if 0 < index <= len(sample.candidates) else None

    for name, target in task.state_targets.items():
        key = state_key(name)
        if name == "start_time_expr":
            ok = (
                current.start_time is not None
                and f"{current.start_time:{TIME_FORMAT}}" == target["time"]
            )
        elif name == "companion_name":
            ok = state.get(key) == pick(target)
        elif name == "style_preference":
            continue
        else:
            ok = _same(state.get(key), target)
        if not ok:
            problems.append(f"state {key} {state.get(key)!r} is not the target {target!r}")

    for name, target in task.targets.items():
        value = delta.get(name)
        if name == "start_time_expr":
            if "time" not in target:
                continue
            result = parse_time_expr(value, sample.now, current=current.start_time)
            got = f"{result.value:{TIME_FORMAT}}" if result.value else result.reason
            if got != target["time"]:
                problems.append(
                    f"start_time_expr {value!r} resolves to {got}, target {target['time']}"
                )
        elif name == "companion_name":
            if value != pick(target):
                problems.append(f"companion_name {value!r} is not candidate #{target}")
        elif name != "style_preference" and not _same(value, target):
            problems.append(f"{name} {value!r} is not the target {target!r}")
    return problems


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(map(str, a)) == sorted(map(str, b))
    return bool(a == b)


def validate_answer(task: GenerationTask, text: str, domain: DomainConfig) -> Outcome:
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as exc:
        return Outcome(task.id, None, [f"invalid JSON: {exc}"])
    problems = shape_problems(task, answer)
    if problems:
        return Outcome(task.id, None, problems)
    try:
        sample = SlotSample.model_validate(assemble(task, answer))
    except ValidationError as exc:
        return Outcome(
            task.id,
            None,
            [f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()],
        )
    problems = (
        sample_problems(sample)
        + business_problems(task, sample, domain)
        + target_problems(task, sample)
    )
    return Outcome(task.id, sample, problems)


def sample_row(sample: SlotSample) -> dict[str, Any]:
    """JSON row in the L2 file format (``load_samples`` reads it back)."""
    row: dict[str, Any] = sample.model_dump(mode="json", exclude_defaults=True)
    row["now"] = f"{sample.now:%Y-%m-%d %H:%M}"  # current_state keeps the Teacher's strings
    return row
