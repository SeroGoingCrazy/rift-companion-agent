"""Scenario specs and the task sampler (DEV_SPEC I2).

A spec file (``training/configs/data/specs_v*.yaml``) describes *what* the SFT data should
cover; ``sample_tasks`` turns it into ``GenerationTask`` objects for the Teacher (I3):

- category counts follow the quotas exactly (largest remainder over ``total``), and so do the
  variant counts within a category;
- everything else -- game mode, ``now``, wording style, history length, which slots sit in
  ``current_state`` and which delta keys the user touches -- is drawn from a seeded RNG, so
  the same spec always yields the same tasks.

A task fixes the *shape* of the expected answer: ``turn_intent``, ``confirmation`` and, for
each delta key, whether the user gives a value, says "no preference" (``any``) or withdraws
it (``null``). The Teacher writes the dialogue and the concrete values; the validator checks
its answer against the task. Field names are ``SlotDelta`` keys; the slot applicability rules
(rank only in ranked modes, no rank / role in ARAM-like modes) come from ``domain.yaml``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence, Set
from datetime import date, datetime, time, timedelta
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from rift_domain.config import DomainConfig, load_domain_config
from rift_domain.enums import Confirmation, GameMode, SlotField, TurnIntent
from rift_domain.slots import AnyValue, SlotDelta
from rift_training.data.targets import ABSOLUTE_FORMS, ValueSpec, draw_targets, task_rng
from rift_training.evaluation.dataset import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[4]
SPECS_DIR = REPO_ROOT / "training" / "configs" / "data"
DOMAIN_PATH = REPO_ROOT / "config" / "domain.yaml"

ValueKind = Literal["value", "any", "null"]
#: Delta keys in schema order.
DELTA_FIELDS: tuple[str, ...] = tuple(SlotDelta.model_fields)
#: Delta key -> booking-state slot (only the time differs).
STATE_SLOT = {
    name: SlotField("start_time" if name == "start_time_expr" else name) for name in DELTA_FIELDS
}
QUOTA_TOLERANCE = 0.05


def _accepts_any(name: str) -> bool:
    """True when the field's type lists ``Literal["any"]`` (a free-text field would also
    *validate* "any", so the annotation is checked, not a trial validation)."""
    return AnyValue in get_args(SlotDelta.model_fields[name].annotation)


#: Delta keys whose contract allows ``"any"`` (derived from the schema, not hand-listed).
ANY_FIELDS = frozenset(name for name in DELTA_FIELDS if _accepts_any(name))


class SpecError(ValueError):
    """A spec file is malformed or asks for something the sampler cannot produce."""


# --- spec models ---------------------------------------------------------------------------

_SPEC_CONFIG = ConfigDict(extra="forbid", frozen=True)
IntRange = tuple[int, int]


def _check_range(value: IntRange) -> IntRange:
    lo, hi = value
    if lo < 0 or lo > hi:
        raise ValueError(f"bad range [{lo}, {hi}]")
    return value


def _check_fields(names: Iterable[str]) -> None:
    unknown = sorted(set(names) - set(DELTA_FIELDS))
    if unknown:
        raise ValueError(f"unknown delta fields {unknown}")


def _check_weights(weights: Mapping[Any, float], what: str) -> None:
    if not weights or any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError(f"{what} weights must be non-negative with a positive sum")


class FieldPick(BaseModel):
    """Pick ``count`` fields at random, optionally only from ``pool``."""

    model_config = _SPEC_CONFIG

    count: IntRange = (1, 1)
    pool: tuple[str, ...] | None = None

    @field_validator("count")
    @classmethod
    def _count(cls, value: IntRange) -> IntRange:
        return _check_range(value)

    @field_validator("pool")
    @classmethod
    def _pool(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None:
            _check_fields(value)
        return value


class StyleSpec(BaseModel):
    model_config = _SPEC_CONFIG

    weight: float = Field(ge=0)
    hint: str


class NowSpec(BaseModel):
    model_config = _SPEC_CONFIG

    start: date
    days: int = Field(ge=1)
    hours: IntRange = (9, 23)

    @field_validator("hours")
    @classmethod
    def _hours(cls, value: IntRange) -> IntRange:
        _check_range(value)
        if value[1] > 23:
            raise ValueError("hours must lie in [0, 23]")
        return value


class VariantSpec(BaseModel):
    """One sub-scenario of a category; ``hint`` is passed to the Teacher verbatim."""

    model_config = _SPEC_CONFIG

    weight: float = Field(gt=0)
    hint: str = Field(min_length=1)
    turn_intent: TurnIntent = TurnIntent.BOOKING
    confirmation: Confirmation = Confirmation.NONE
    pending_confirmation: bool = False
    #: Number of candidates already shown to the user.
    candidates: IntRange = (0, 0)
    #: Overrides the category's range (user/assistant exchanges before this turn).
    history_turns: IntRange | None = None
    #: Slots always / additionally in ``current_state``.
    require_state: tuple[str, ...] = ()
    state: FieldPick | None = None
    #: Delta keys the user always gives a (new) value for.
    require: tuple[str, ...] = ()
    #: New values for slots not yet in the state.
    value: FieldPick | None = None
    #: New values for slots already in the state.
    modify: FieldPick | None = None
    any: FieldPick | None = None
    #: Withdrawn slots (delta ``null``; always drawn from the state). Not called ``null``
    #: because YAML reads that key as None.
    withdraw: FieldPick | None = None
    #: Never picked at random in this variant.
    exclude: tuple[str, ...] = ()
    styles: dict[str, float] | None = None
    modes: dict[GameMode, float] | None = None
    #: Overrides ``values.time.forms`` for this variant's new times (e.g. only "after").
    time_forms: dict[str, float] | None = None

    @field_validator("candidates")
    @classmethod
    def _candidates(cls, value: IntRange) -> IntRange:
        return _check_range(value)

    @field_validator("history_turns")
    @classmethod
    def _history(cls, value: IntRange | None) -> IntRange | None:
        return None if value is None else _check_range(value)

    @model_validator(mode="after")
    def _consistent(self) -> VariantSpec:
        _check_fields((*self.require_state, *self.require, *self.exclude))
        if self.any and self.any.pool and not set(self.any.pool) <= ANY_FIELDS:
            bad = sorted(set(self.any.pool) - ANY_FIELDS)
            raise ValueError(f"fields {bad} do not accept 'any'")
        if self.confirmation == Confirmation.YES and not self.pending_confirmation:
            raise ValueError("confirmation 'yes' needs pending_confirmation")
        if self.turn_intent != TurnIntent.BOOKING and self.picks_delta():
            raise ValueError(f"a {self.turn_intent.value} turn carries no delta")
        if set(self.require) & set(self.require_state):
            raise ValueError("a field cannot be both required in the delta and in the state")
        if self.styles is not None:
            _check_weights(self.styles, "variant style")
        if self.modes is not None:
            _check_weights(self.modes, "variant mode")
        if self.time_forms is not None:
            _check_weights(self.time_forms, "variant time form")
            unknown = set(self.time_forms) - set(ABSOLUTE_FORMS)
            if unknown:
                raise ValueError(f"unknown time forms {sorted(unknown)}")
        return self

    def picks_delta(self) -> bool:
        picks = (self.value, self.modify, self.any, self.withdraw)
        return bool(self.require) or any(p is not None and p.count[1] > 0 for p in picks)


class CategorySpec(BaseModel):
    model_config = _SPEC_CONFIG

    quota: float = Field(gt=0, le=1)
    history_turns: IntRange = (0, 0)
    styles: dict[str, float] | None = None
    variants: dict[str, VariantSpec] = Field(min_length=1)

    @field_validator("history_turns")
    @classmethod
    def _history(cls, value: IntRange) -> IntRange:
        return _check_range(value)


class ScenarioSpec(BaseModel):
    model_config = _SPEC_CONFIG

    version: str = Field(min_length=1)
    #: "full": every category, quotas cover the whole set; "contrast": a targeted add-on (I7).
    kind: Literal["full", "contrast"] = "full"
    total: int = Field(gt=0)
    seed: int
    now: NowSpec
    styles: dict[str, StyleSpec] = Field(min_length=1)
    modes: dict[GameMode, float]
    field_weights: dict[str, float]
    categories: dict[str, CategorySpec]
    #: Target-value distributions (v0.2+); without them the Teacher picks the values.
    values: ValueSpec | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ScenarioSpec:
        _check_weights({k: s.weight for k, s in self.styles.items()}, "style")
        _check_weights(self.modes, "mode")
        _check_fields(self.field_weights)
        if any(w < 0 for w in self.field_weights.values()):
            raise ValueError("field_weights must be non-negative")
        missing = sorted(set(CATEGORIES) - set(self.categories))
        unknown = sorted(set(self.categories) - set(CATEGORIES))
        if self.kind == "contrast":
            missing = []  # a contrast set targets a few categories only
        if missing or unknown:
            raise ValueError(
                f"categories must be exactly {list(CATEGORIES)} "
                f"(missing {missing}, unknown {unknown})"
            )
        quota_sum = sum(c.quota for c in self.categories.values())
        if abs(quota_sum - 1) > 1e-6:
            raise ValueError(f"category quotas sum to {quota_sum:.4f}, expected 1")
        for cat_name, cat in self.categories.items():
            for overrides in (cat.styles, *(v.styles for v in cat.variants.values())):
                if overrides and not set(overrides) <= set(self.styles):
                    raise ValueError(
                        f"{cat_name}: unknown styles {sorted(set(overrides) - set(self.styles))}"
                    )
            for var_name, var in cat.variants.items():
                history = var.history_turns or cat.history_turns
                needs_history = var.pending_confirmation or var.candidates[1] > 0
                if needs_history and history[0] < 1:
                    raise ValueError(
                        f"{cat_name}.{var_name}: candidates / pending confirmation "
                        "need history >= 1"
                    )
        return self


# --- tasks ---------------------------------------------------------------------------------


class GenerationTask(BaseModel):
    """One sample the Teacher has to write; the shape of the answer is fixed here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    spec_version: str
    category: str
    variant: str
    hint: str
    style: str
    style_hint: str
    now: datetime
    #: The conversation's game mode, when it appears in the state or the delta.
    game_mode: GameMode | None
    #: Slots (delta key names) already in ``current_state``, in schema order.
    state_fields: tuple[str, ...]
    num_candidates: int
    pending_confirmation: bool
    history_turns: int
    turn_intent: TurnIntent
    confirmation: Confirmation
    #: Delta key -> how the user touches it this turn (schema order).
    delta: dict[str, ValueKind]
    #: Target values of the state slots / of the delta "value" keys (see ``targets``).
    state_targets: dict[str, Any] = Field(default_factory=dict)
    targets: dict[str, Any] = Field(default_factory=dict)


def load_spec(path_or_version: str | Path) -> ScenarioSpec:
    """Load ``specs_<version>.yaml`` from ``SPECS_DIR`` or an explicit path."""
    path = Path(path_or_version)
    if path.suffix not in (".yaml", ".yml"):  # a version such as "v0.1"
        path = SPECS_DIR / f"specs_{path_or_version}.yaml"
    if not path.exists():
        raise SpecError(f"{path}: not found")
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
        return ScenarioSpec.model_validate(data)
    except (yaml.YAMLError, ValidationError) as exc:
        raise SpecError(f"{path.name}: {exc}") from exc


@cache
def _default_domain() -> DomainConfig:
    return load_domain_config(DOMAIN_PATH)


def applicable_fields(mode: GameMode, domain: DomainConfig) -> frozenset[str]:
    """Delta keys that make sense in ``mode`` (rank only when ranked, nothing not-applicable)."""
    blocked: set[SlotField] = set()
    for rule in domain.rules.not_applicable:
        if mode in rule.when_modes:
            blocked.update(rule.fields)
    if not domain.game_modes[mode].ranked:
        blocked.add(SlotField.RANK_REQUIREMENT)
    return frozenset(name for name in DELTA_FIELDS if STATE_SLOT[name] not in blocked)


def allocate(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Split ``total`` proportionally (largest remainder; ties broken by declaration order)."""
    norm = sum(weights.values())
    exact = {k: total * w / norm for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    order = sorted(weights, key=lambda k: exact[k] - counts[k], reverse=True)
    for key in order[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def _weighted_choice(rng: random.Random, weights: Mapping[Any, float]) -> Any:
    keys = [k for k, w in weights.items() if w > 0]
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _pick(
    rng: random.Random,
    candidates: Iterable[str],
    count: IntRange,
    weights: Mapping[str, float],
    *,
    explicit: bool,
) -> list[str] | None:
    """Weighted sample without replacement; ``None`` when fewer than ``count[0]`` available.

    Fields with weight 0 are only eligible when the variant listed them in an explicit pool.
    """
    # Schema order, never set order: str hashes (and so set order) change per process.
    pool = {f: (weights.get(f, 1.0) or (1.0 if explicit else 0.0)) for f in _ordered(candidates)}
    pool = {f: w for f, w in pool.items() if w > 0}
    n = rng.randint(*count)
    if len(pool) < count[0]:
        return None
    picked: list[str] = []
    for _ in range(min(n, len(pool))):
        choice = _weighted_choice(rng, pool)
        picked.append(choice)
        del pool[choice]
    return picked


def _ordered(fields: Iterable[str]) -> tuple[str, ...]:
    chosen = set(fields)
    return tuple(name for name in DELTA_FIELDS if name in chosen)


class TaskSampler:
    """Draws the tasks of one variant; every draw is retried until the variant is satisfiable."""

    MAX_ATTEMPTS = 50

    def __init__(self, spec: ScenarioSpec, domain: DomainConfig, rng: random.Random) -> None:
        self.spec = spec
        self.domain = domain
        self.rng = rng

    def _mode_weights(self, var: VariantSpec) -> dict[GameMode, float]:
        weights = dict(var.modes or self.spec.modes)
        needed = {*var.require, *var.require_state}
        feasible = {
            m: w
            for m, w in weights.items()
            if w > 0 and needed <= applicable_fields(m, self.domain)
        }
        if not feasible:
            raise SpecError(f"no game mode allows all of {sorted(needed)}")
        return feasible

    def _now(self) -> datetime:
        cfg = self.spec.now
        day = cfg.start + timedelta(days=self.rng.randrange(cfg.days))
        hour = self.rng.randint(*cfg.hours)
        return datetime.combine(day, time(hour, self.rng.choice((0, 15, 30, 45))))

    def _style(self, cat: CategorySpec, var: VariantSpec) -> str:
        weights = var.styles or cat.styles or {k: s.weight for k, s in self.spec.styles.items()}
        style: str = _weighted_choice(self.rng, weights)
        return style

    def _try_fields(
        self, var: VariantSpec, mode: GameMode
    ) -> tuple[tuple[str, ...], dict[str, ValueKind]] | None:
        rng, weights = self.rng, self.spec.field_weights
        eligible = applicable_fields(mode, self.domain) - set(var.exclude)

        state = set(var.require_state)
        if var.state:
            extra = _pick(
                rng, eligible - state - set(var.require), var.state.count, weights, explicit=False
            )
            if extra is None:
                return None
            state.update(extra)

        delta: dict[str, ValueKind] = {name: "value" for name in var.require}

        def pick(spec: FieldPick | None, base: Set[str]) -> list[str] | None:
            if spec is None:
                return []
            pool = base if spec.pool is None else base & set(spec.pool)
            return _pick(rng, pool, spec.count, weights, explicit=spec.pool is not None)

        nulls = pick(var.withdraw, state - set(delta))
        if nulls is None:
            return None
        delta.update({n: "null" for n in nulls})
        modified = pick(var.modify, state - set(delta))
        if modified is None:
            return None
        delta.update({n: "value" for n in modified})
        anys = pick(var.any, (eligible & ANY_FIELDS) - set(delta))
        if anys is None:
            return None
        delta.update({n: "any" for n in anys})
        new = pick(var.value, eligible - state - set(delta))
        if new is None:
            return None
        delta.update({n: "value" for n in new})
        return _ordered(state), {name: delta[name] for name in _ordered(delta)}

    def draw(self, task_id: str, cat_name: str, var_name: str) -> GenerationTask:
        cat = self.spec.categories[cat_name]
        var = cat.variants[var_name]
        mode_weights = self._mode_weights(var)
        for _ in range(self.MAX_ATTEMPTS):
            mode: GameMode = _weighted_choice(self.rng, mode_weights)
            fields = self._try_fields(var, mode)
            if fields is not None:
                break
        else:
            raise SpecError(f"{cat_name}.{var_name}: cannot satisfy the field picks")
        state_fields, delta = fields
        style = self._style(cat, var)
        uses_mode = "game_mode" in state_fields or "game_mode" in delta
        task = GenerationTask(
            id=task_id,
            spec_version=self.spec.version,
            category=cat_name,
            variant=var_name,
            hint=var.hint,
            style=style,
            style_hint=self.spec.styles[style].hint,
            now=self._now(),
            game_mode=mode if uses_mode else None,
            state_fields=state_fields,
            num_candidates=self.rng.randint(*var.candidates),
            pending_confirmation=var.pending_confirmation,
            history_turns=self.rng.randint(*(var.history_turns or cat.history_turns)),
            turn_intent=var.turn_intent,
            confirmation=var.confirmation,
            delta=delta,
        )
        if self.spec.values is None:
            return task
        # A separate per-task RNG: targets never shift the task shapes drawn above.
        state_targets, targets = draw_targets(
            self.spec.values,
            task_rng(self.spec.seed, task_id),
            now=task.now,
            variant=var_name,
            state_fields=state_fields,
            delta=delta,
            num_candidates=task.num_candidates,
            time_forms=var.time_forms,
        )
        return task.model_copy(update={"state_targets": state_targets, "targets": targets})


def sample_tasks(
    spec: ScenarioSpec,
    *,
    total: int | None = None,
    seed: int | None = None,
    domain: DomainConfig | None = None,
) -> list[GenerationTask]:
    """All tasks of a spec, grouped by category in declaration order; deterministic per seed."""
    total = spec.total if total is None else total
    rng = random.Random(spec.seed if seed is None else seed)
    sampler = TaskSampler(spec, domain or _default_domain(), rng)
    per_category = allocate(total, {name: cat.quota for name, cat in spec.categories.items()})
    tasks: list[GenerationTask] = []
    for cat_name, cat in spec.categories.items():
        per_variant = allocate(
            per_category[cat_name], {name: v.weight for name, v in cat.variants.items()}
        )
        for var_name, n in per_variant.items():
            for _ in range(n):
                task_id = f"{spec.version}-{len(tasks) + 1:04d}"
                tasks.append(sampler.draw(task_id, cat_name, var_name))
    return tasks


# --- checks and summaries ------------------------------------------------------------------


def quota_errors(spec: ScenarioSpec, tasks: Sequence[GenerationTask]) -> dict[str, float]:
    """Relative error of each category's count against its quota share."""
    counts = Counter(t.category for t in tasks)
    errors: dict[str, float] = {}
    for name, cat in spec.categories.items():
        target = cat.quota * len(tasks)
        errors[name] = abs(counts[name] - target) / target if target else 0.0
    return errors


def task_problems(task: GenerationTask, domain: DomainConfig | None = None) -> list[str]:
    """Invariants every task must satisfy (the sampler guarantees them; tests re-check)."""
    domain = domain or _default_domain()
    problems: list[str] = []
    state = set(task.state_fields)
    for name, kind in task.delta.items():
        if kind == "null" and name not in state:
            problems.append(f"{name}: withdrawn but not in the state")
        if kind == "any" and name not in ANY_FIELDS:
            problems.append(f"{name}: does not accept 'any'")
    if task.game_mode is not None:
        allowed = applicable_fields(task.game_mode, domain)
        for name in sorted((state | set(task.delta)) - allowed):
            problems.append(f"{name}: not applicable in {task.game_mode.value}")
    if task.turn_intent != TurnIntent.BOOKING and task.delta:
        problems.append(f"{task.turn_intent.value} task with a delta")
    if task.confirmation == Confirmation.YES and not task.pending_confirmation:
        problems.append("confirmation yes without a pending confirmation")
    if (task.pending_confirmation or task.num_candidates) and task.history_turns < 1:
        problems.append("candidates / pending confirmation without history")
    if (
        task.category == "candidate_ref"
        and task.delta.get("companion_name") == "value"
        and task.num_candidates < 1
    ):
        problems.append("candidate reference without candidates")
    return problems


def summarize(tasks: Sequence[GenerationTask]) -> dict[str, Any]:
    kinds: dict[str, Counter[str]] = {}
    for task in tasks:
        for name, kind in task.delta.items():
            kinds.setdefault(name, Counter())[kind] += 1
    return {
        "total": len(tasks),
        "categories": dict(Counter(t.category for t in tasks)),
        "variants": dict(Counter(f"{t.category}.{t.variant}" for t in tasks)),
        "turn_intents": dict(Counter(t.turn_intent.value for t in tasks)),
        "confirmations": dict(Counter(t.confirmation.value for t in tasks)),
        "styles": dict(Counter(t.style for t in tasks)),
        "game_modes": dict(Counter(t.game_mode.value if t.game_mode else "-" for t in tasks)),
        "history_turns": dict(sorted(Counter(t.history_turns for t in tasks).items())),
        "delta_fields": {name: dict(kinds[name]) for name in _ordered(kinds)},
    }


def write_tasks(tasks: Sequence[GenerationTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for task in tasks:
            fh.write(json.dumps(task.model_dump(mode="json"), ensure_ascii=False) + "\n")


def read_tasks(path: Path) -> list[GenerationTask]:
    return [
        GenerationTask.model_validate_json(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]
