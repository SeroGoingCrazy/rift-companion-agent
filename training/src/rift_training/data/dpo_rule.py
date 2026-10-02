"""Rule-perturbation preference pairs (DEV_SPEC I8 ①).

``rejected`` is the canonical answer with exactly one targeted mistake, always protocol-valid
(so DPO learns the semantic boundary, not the JSON syntax) and always failing the L2 task
scorer. The mistakes mirror the errors that SFT r001-r003 kept making:

==================  ===================================================================
``any_to_null``      "没要求" written as a withdrawal (``"any"`` -> ``null``)
``null_to_any``      a withdrawal written as "no preference" (``null`` -> ``"any"``)
``any_to_absent``    "no preference" dropped
``extra_null``       a slot withdrawn although the user said nothing about it
``drop_key``         a mentioned slot missed (dense sentences)
``role_append``      "X 换成 Y" appended to the old roles instead of replacing them
``time_swallow``     the duration glued onto the time expression ("明晚八点两小时")
``name_literal``     a candidate reference copied as text ("2号吧") instead of the name
``value_swap``       a wrong value for one slot (rank, gender, duration ...)
``intent_flip``      consult <-> unrelated, or a bare booking turn read as consult
``confirm_flip``     a yes / no answer read as ``none``
==================  ===================================================================
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Sequence
from typing import Any

from rift_domain.enums import GameMode, Gender, Rank, ServiceType
from rift_training.contract import ANY, render_target
from rift_training.data.preference import PreferencePair
from rift_training.data.specs import ANY_FIELDS
from rift_training.evaluation.dataset import SlotSample
from rift_training.evaluation.preferences import ExactStyleMatcher
from rift_training.evaluation.task import score_output

Expected = dict[str, Any]
Perturb = Callable[[SlotSample, Expected, random.Random], Expected | None]

#: Sampling weights: the error families SFT kept making weigh more.
WEIGHTS: dict[str, float] = {
    "any_to_null": 3,
    "null_to_any": 2,
    "any_to_absent": 1,
    "extra_null": 0.7,  # applies to almost every multi-turn sample; keep it from dominating
    "drop_key": 1.5,
    "role_append": 3,
    "time_swallow": 3,
    "name_literal": 3,
    "value_swap": 0.7,
    "intent_flip": 2,
    "confirm_flip": 2,
}
#: Slots whose omission SFT made most often (dense sentences).
DROP_FIRST = ("service_type", "role_preference", "companion_gender", "style_preference")
_PUNCT = re.compile(r"[\s，。！？、,.!?~～;；]")


def _copy(expected: Expected, delta: dict[str, Any]) -> Expected:
    return {**expected, "delta": delta}


def any_to_null(_: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    keys = [k for k, v in e["delta"].items() if v == ANY]
    if not keys:
        return None
    return _copy(e, {**e["delta"], rng.choice(keys): None})


def null_to_any(_: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    keys = [k for k, v in e["delta"].items() if v is None and k in ANY_FIELDS]
    if not keys:
        return None
    return _copy(e, {**e["delta"], rng.choice(keys): ANY})


def any_to_absent(_: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    keys = [k for k, v in e["delta"].items() if v == ANY]
    if not keys:
        return None
    gone = rng.choice(keys)
    return _copy(e, {k: v for k, v in e["delta"].items() if k != gone})


def extra_null(s: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    if e["turn_intent"] != "booking":
        return None
    state_keys = [
        "start_time_expr" if k == "start_time" else k
        for k in s.current_state
        if k != "start_time_expr"
    ]
    keys = [k for k in state_keys if k not in e["delta"]]
    if not keys:
        return None
    key = "companion_name" if "companion_name" in keys else rng.choice(keys)
    return _copy(e, {**e["delta"], key: None})


def drop_key(_: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    keys = [k for k, v in e["delta"].items() if v is not None and v != ANY]
    if len(keys) < 2:
        return None
    first = [k for k in DROP_FIRST if k in keys]
    gone = first[0] if first and rng.random() < 0.7 else rng.choice(keys)
    return _copy(e, {k: v for k, v in e["delta"].items() if k != gone})


def role_append(s: SlotSample, e: Expected, _: random.Random) -> Expected | None:
    new = e["delta"].get("role_preference")
    old = s.current_state.get("role_preference")
    if not isinstance(new, list) or not isinstance(old, list):
        return None
    merged = list(dict.fromkeys([*old, *new]))
    if set(merged) == set(new) or len(merged) > 5:
        return None
    return _copy(e, {**e["delta"], "role_preference": merged})


def time_swallow(s: SlotSample, e: Expected, _: random.Random) -> Expected | None:
    expr = e["delta"].get("start_time_expr")
    if not isinstance(expr, str) or "duration_hours" not in e["delta"]:
        return None
    at = s.user_input.find(expr)
    if at < 0:
        return None
    rest = s.user_input[at + len(expr) :]
    tail = _PUNCT.split(rest, maxsplit=1)[0][:6]
    if not tail:
        return None
    delta = {k: v for k, v in e["delta"].items() if k != "duration_hours"}
    delta["start_time_expr"] = expr + tail
    return _copy(e, delta)


def name_literal(s: SlotSample, e: Expected, _: random.Random) -> Expected | None:
    name = e["delta"].get("companion_name")
    if not isinstance(name, str) or name == ANY or not s.candidates:
        return None
    literal = _PUNCT.sub("", s.user_input)
    if not literal or len(literal) > 8 or literal == name:
        delta = {k: v for k, v in e["delta"].items() if k != "companion_name"}
        return _copy(e, delta) if delta != e["delta"] else None
    return _copy(e, {**e["delta"], "companion_name": literal})


def _other(values: Sequence[str], current: str, rng: random.Random) -> str:
    return rng.choice([v for v in values if v != current])


def value_swap(_: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    d = e["delta"]
    options: list[tuple[str, Any]] = []
    ranks = [r.value for r in Rank]
    if d.get("rank_requirement") in ranks:
        i = ranks.index(d["rank_requirement"])
        options.append(("rank_requirement", ranks[i - 1] if i else ranks[i + 1]))
    if d.get("companion_gender") in [g.value for g in Gender]:
        options.append(
            ("companion_gender", _other([g.value for g in Gender], d["companion_gender"], rng))
        )
    if d.get("service_type") in [t.value for t in ServiceType]:
        options.append(
            ("service_type", _other([t.value for t in ServiceType], d["service_type"], rng))
        )
    if d.get("game_mode") in [m.value for m in GameMode]:
        options.append(("game_mode", _other([m.value for m in GameMode], d["game_mode"], rng)))
    if isinstance(d.get("voice_required"), bool):
        options.append(("voice_required", not d["voice_required"]))
    hours = d.get("duration_hours")
    if isinstance(hours, int | float) and not isinstance(hours, bool):
        options.append(("duration_hours", hours + 1 if hours <= 7 else hours - 1))
    if not options:
        return None
    key, value = rng.choice(options)
    return _copy(e, {**d, key: value})


def intent_flip(_: SlotSample, e: Expected, rng: random.Random) -> Expected | None:
    intent = e["turn_intent"]
    if intent == "consult":
        return {**e, "turn_intent": "unrelated"}
    if intent == "unrelated":
        return {**e, "turn_intent": "consult"}
    if not e["delta"] and e["confirmation"] == "none":
        return {**e, "turn_intent": rng.choice(["consult", "unrelated"])}
    return None


def confirm_flip(_: SlotSample, e: Expected, __: random.Random) -> Expected | None:
    if e["confirmation"] in ("yes", "no"):
        return {**e, "confirmation": "none"}
    return None


PERTURBATIONS: dict[str, Perturb] = {
    "any_to_null": any_to_null,
    "null_to_any": null_to_any,
    "any_to_absent": any_to_absent,
    "extra_null": extra_null,
    "drop_key": drop_key,
    "role_append": role_append,
    "time_swallow": time_swallow,
    "name_literal": name_literal,
    "value_swap": value_swap,
    "intent_flip": intent_flip,
    "confirm_flip": confirm_flip,
}
_MATCHER = ExactStyleMatcher()


def _rng(seed: int, sample_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def is_wrong(sample: SlotSample, rejected_text: str) -> bool:
    """True when the L2 scorer fails the rejected answer for this sample."""
    score = score_output(
        rejected_text, sample.extraction(), state=sample.state(), now=sample.now, style=_MATCHER
    )
    return not score.passed


def candidates_for(sample: SlotSample, seed: int = 0) -> list[tuple[str, str]]:
    """Every applicable (kind, rejected text) for a sample, wrong by the scorer."""
    rng = _rng(seed, sample.id)
    out: list[tuple[str, str]] = []
    for kind, perturb in PERTURBATIONS.items():
        rejected = perturb(sample, sample.expected, rng)
        if rejected is None:
            continue
        text = render_target(rejected)  # validates the protocol, canonical formatting
        if text != render_target(sample.expected) and is_wrong(sample, text):
            out.append((kind, text))
    return out


def build_rule_pairs(
    samples: Sequence[SlotSample], *, seed: int = 0, per_sample: int = 1
) -> list[PreferencePair]:
    """Up to ``per_sample`` pairs per sample, kinds drawn by ``WEIGHTS`` without repeats."""
    pairs: list[PreferencePair] = []
    for sample in samples:
        options = dict(candidates_for(sample, seed))
        rng = _rng(seed + 1, sample.id)
        for _ in range(min(per_sample, len(options))):
            kinds = list(options)
            kind = rng.choices(kinds, weights=[WEIGHTS[k] for k in kinds])[0]
            pairs.append(PreferencePair(sample, options.pop(kind), kind))
    return pairs
