"""I2: scenario specs load strictly and the sampler hits the quotas with valid task shapes."""

from __future__ import annotations

import copy
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode
from rift_training.data.specs import (
    ANY_FIELDS,
    DELTA_FIELDS,
    QUOTA_TOLERANCE,
    SPECS_DIR,
    ScenarioSpec,
    SpecError,
    allocate,
    applicable_fields,
    load_spec,
    quota_errors,
    read_tasks,
    sample_tasks,
    summarize,
    task_problems,
    write_tasks,
)
from rift_training.evaluation.dataset import CATEGORIES

SPEC_PATH = SPECS_DIR / "specs_v0.1.yaml"
SCRIPT = Path(__file__).resolve().parents[2] / "training" / "scripts" / "data" / "sample_tasks.py"
OPTIONAL_FIELDS = {
    "style_preference",
    "companion_name",
    "companion_gender",
    "voice_required",
    "budget_per_hour",
}


@pytest.fixture(scope="module")
def spec() -> ScenarioSpec:
    return load_spec("v0.1")


@pytest.fixture(scope="module")
def tasks(spec: ScenarioSpec, domain: DomainConfig) -> list[Any]:
    return sample_tasks(spec, domain=domain)


def raw_spec() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(SPEC_PATH.read_text("utf-8"))
    return copy.deepcopy(data)


def write_spec(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "specs_test.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# --- acceptance ----------------------------------------------------------------------------


@pytest.mark.unit
def test_800_tasks_hit_every_category_quota_within_5_percent(
    spec: ScenarioSpec, tasks: list[Any]
) -> None:
    assert len(tasks) == 800
    errors = quota_errors(spec, tasks)
    assert set(errors) == set(CATEGORIES)
    assert max(errors.values()) < QUOTA_TOLERANCE, errors


@pytest.mark.unit
@pytest.mark.parametrize("total", [100, 333, 1500])
def test_quotas_hold_for_other_sizes(spec: ScenarioSpec, domain: DomainConfig, total: int) -> None:
    tasks = sample_tasks(spec, total=total, domain=domain)
    assert len(tasks) == total
    counts = Counter(t.category for t in tasks)
    for name, cat in spec.categories.items():
        assert abs(counts[name] - cat.quota * total) < 1, name


@pytest.mark.unit
def test_variant_counts_follow_their_weights(spec: ScenarioSpec, tasks: list[Any]) -> None:
    counts = Counter((t.category, t.variant) for t in tasks)
    per_category = Counter(t.category for t in tasks)
    for cat_name, cat in spec.categories.items():
        weight_sum = sum(v.weight for v in cat.variants.values())
        for var_name, var in cat.variants.items():
            target = per_category[cat_name] * var.weight / weight_sum
            assert abs(counts[cat_name, var_name] - target) < 1, (cat_name, var_name)


@pytest.mark.unit
def test_sampling_is_deterministic_per_seed(spec: ScenarioSpec, domain: DomainConfig) -> None:
    a = sample_tasks(spec, total=200, domain=domain)
    b = sample_tasks(spec, total=200, domain=domain)
    c = sample_tasks(spec, total=200, seed=spec.seed + 1, domain=domain)
    assert a == b
    assert a != c
    assert len({t.id for t in a}) == 200


@pytest.mark.unit
def test_every_task_is_consistent(tasks: list[Any], domain: DomainConfig) -> None:
    problems = [f"{t.id}: {p}" for t in tasks for p in task_problems(t, domain)]
    assert problems == []


# --- task shapes ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tasks_respect_mode_applicability(tasks: list[Any]) -> None:
    for t in tasks:
        touched = set(t.state_fields) | set(t.delta)
        if t.game_mode in (GameMode.ARAM, GameMode.ARAM_MAYHEM, GameMode.ARENA):
            assert not touched & {"rank_requirement", "role_preference"}, t.id
        if t.game_mode == GameMode.NORMAL_DRAFT:
            assert "rank_requirement" not in touched, t.id


@pytest.mark.unit
def test_fields_are_in_schema_order(tasks: list[Any]) -> None:
    order = {name: i for i, name in enumerate(DELTA_FIELDS)}
    for t in tasks:
        assert list(t.delta) == sorted(t.delta, key=order.__getitem__)
        assert list(t.state_fields) == sorted(t.state_fields, key=order.__getitem__)


@pytest.mark.unit
def test_intents_and_confirmations(tasks: list[Any]) -> None:
    for t in tasks:
        if t.category in ("consult", "unrelated"):
            assert t.turn_intent.value == t.category and t.delta == {}, t.id
        else:
            assert t.turn_intent.value == "booking", t.id
        if t.confirmation.value != "none":
            assert t.category == "confirmation" and t.pending_confirmation, t.id
    intents = summarize(tasks)["turn_intents"]
    assert intents["consult"] == 64 and intents["unrelated"] == 48


@pytest.mark.unit
def test_baseline_focus_points_are_covered(tasks: list[Any]) -> None:
    by_variant: dict[str, list[Any]] = {}
    for t in tasks:
        by_variant.setdefault(t.variant, []).append(t)

    # 1. "any" vs null on optional fields
    optional_any = [
        t for t in tasks for f, k in t.delta.items() if k == "any" and f in OPTIONAL_FIELDS
    ]
    withdrawn = [t for t in tasks for k in t.delta.values() if k == "null"]
    assert len(optional_any) >= 60
    assert len(withdrawn) >= 40
    assert all(t.delta.get("game_mode") != "null" for t in by_variant["null_withdraw"])

    # 2. declining to switch companions carries no delta
    for t in by_variant["decline_switch"]:
        assert t.confirmation.value == "no" and t.delta == {}
        assert "companion_name" in t.state_fields and t.num_candidates >= 2

    # 3. a nice voice is a style, not voice_required
    for t in by_variant["voice_as_style"]:
        assert t.delta.get("style_preference") == "value" and "voice_required" not in t.delta
    for t in by_variant["voice_and_style"]:
        assert t.delta["style_preference"] == "value" and t.delta["voice_required"] == "value"


@pytest.mark.unit
def test_context_needs_match_the_variant(tasks: list[Any]) -> None:
    for t in tasks:
        if t.category == "first_turn":
            assert t.history_turns == 0 and t.state_fields == ()
        if t.category == "candidate_ref":
            assert t.num_candidates >= 2 and t.delta["companion_name"] == "value"
        if t.variant == "shift":
            assert "start_time_expr" in t.state_fields and t.delta["start_time_expr"] == "value"


@pytest.mark.unit
def test_styles_and_modes_are_all_used(spec: ScenarioSpec, tasks: list[Any]) -> None:
    summary = summarize(tasks)
    assert set(summary["styles"]) == set(spec.styles)
    assert set(summary["game_modes"]) >= {m.value for m in GameMode}
    mode_slang = Counter(t.style for t in tasks if t.category == "mode_slang")
    assert mode_slang["slang"] > mode_slang["standard"]


@pytest.mark.unit
def test_now_stays_in_the_configured_window(spec: ScenarioSpec, tasks: list[Any]) -> None:
    lo, hi = spec.now.hours
    for t in tasks:
        assert 0 <= (t.now.date() - spec.now.start).days < spec.now.days
        assert lo <= t.now.hour <= hi and t.now.minute in (0, 15, 30, 45)


# --- helpers -------------------------------------------------------------------------------


@pytest.mark.unit
def test_any_fields_come_from_the_schema() -> None:
    assert {
        "rank_requirement",
        "role_preference",
        "companion_gender",
        "voice_required",
        "budget_per_hour",
        "style_preference",
        "companion_name",
    } == ANY_FIELDS


@pytest.mark.unit
def test_applicable_fields_follow_domain_rules(domain: DomainConfig) -> None:
    assert "rank_requirement" in applicable_fields(GameMode.RANKED_SOLO_DUO, domain)
    assert "rank_requirement" not in applicable_fields(GameMode.NORMAL_DRAFT, domain)
    assert "role_preference" in applicable_fields(GameMode.NORMAL_DRAFT, domain)
    arena = applicable_fields(GameMode.ARENA, domain)
    assert not arena & {"rank_requirement", "role_preference"}
    assert "style_preference" in arena


@pytest.mark.unit
def test_allocate_uses_largest_remainder() -> None:
    assert allocate(10, {"a": 1, "b": 1, "c": 1}) == {"a": 4, "b": 3, "c": 3}
    assert allocate(7, {"a": 0.5, "b": 0.3, "c": 0.2}) == {"a": 4, "b": 2, "c": 1}
    assert sum(allocate(800, {"x": 0.155, "y": 0.845}).values()) == 800


@pytest.mark.unit
def test_tasks_round_trip_through_jsonl(tasks: list[Any], tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    write_tasks(tasks[:50], path)
    assert read_tasks(path) == tasks[:50]
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.unit
def test_load_spec_by_version_or_path(tmp_path: Path) -> None:
    assert load_spec("v0.1") == load_spec(SPEC_PATH)
    with pytest.raises(SpecError, match="not found"):
        load_spec("v9.9")


@pytest.mark.unit
def test_script_samples_and_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module_spec = importlib.util.spec_from_file_location("sample_tasks", SCRIPT)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    out = tmp_path / "tasks.jsonl"
    assert module.main(["--spec", "v0.1", "--total", "120", "--out", str(out)]) == 0
    assert len(read_tasks(out)) == 120
    assert "quota error per category" in capsys.readouterr().out


# --- invalid specs -------------------------------------------------------------------------


def _set(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    node = data
    *parents, last = path.split(".")
    for key in parents:
        node = node[key]
    node[last] = value
    return data


INVALID = [
    ("categories.first_turn.quota", 0.5, "quotas sum"),
    ("categories.first_turn.variants.single_slot.require", ["start_time"], "unknown delta fields"),
    ("categories.tri_state.variants.any_required.any.pool", ["game_mode"], "do not accept 'any'"),
    (
        "categories.confirmation.variants.confirm_yes.pending_confirmation",
        False,
        "pending_confirmation",
    ),
    ("categories.consult.variants.platform_rules.require", ["duration_hours"], "carries no delta"),
    ("categories.candidate_ref.history_turns", [0, 2], "need history >= 1"),
    ("categories.mode_slang.styles", {"shouting": 1.0}, "unknown styles"),
    ("modes", {"aram": 0}, "mode weights"),
    ("categories.multi_turn.variants.add_fields.state.count", [3, 1], "bad range"),
]


@pytest.mark.unit
@pytest.mark.parametrize(("path", "value", "message"), INVALID)
def test_invalid_specs_are_rejected(tmp_path: Path, path: str, value: Any, message: str) -> None:
    data = _set(raw_spec(), path, value)
    with pytest.raises(SpecError, match=message):
        load_spec(write_spec(tmp_path, data))


@pytest.mark.unit
def test_missing_category_is_rejected(tmp_path: Path) -> None:
    data = raw_spec()
    del data["categories"]["unrelated"]
    data["categories"]["consult"]["quota"] += 0.06
    with pytest.raises(SpecError, match="missing \\['unrelated'\\]"):
        load_spec(write_spec(tmp_path, data))


@pytest.mark.unit
def test_unsatisfiable_variant_fails_loudly(tmp_path: Path, domain: DomainConfig) -> None:
    data = raw_spec()
    data["categories"]["mode_slang"]["variants"]["rank_slang"]["modes"] = {"aram": 1.0}
    spec = load_spec(write_spec(tmp_path, data))
    with pytest.raises(SpecError, match="no game mode allows"):
        sample_tasks(spec, domain=domain)
