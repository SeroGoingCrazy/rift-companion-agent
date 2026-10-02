"""v0.2 target values: drawn per task, pinned in the answer schema, enforced by the validator."""

from __future__ import annotations

import json
import random
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import pytest

from rift_domain.config import DomainConfig
from rift_domain.enums import Confirmation, GameMode, TurnIntent
from rift_training.data.prompting import answer_schema, task_prompt
from rift_training.data.specs import GenerationTask, load_spec, sample_tasks
from rift_training.data.targets import MIN_LEAD, TIME_FORMAT, TargetDrawer, draw_targets, task_rng
from rift_training.data.validate import validate_answer

NOW = datetime(2026, 10, 1, 14, 0)  # Thursday


@pytest.fixture(scope="module")
def v02(domain: DomainConfig) -> list[GenerationTask]:
    return sample_tasks(load_spec("v0.2"), domain=domain)


def _time(text: str) -> datetime:
    return datetime.strptime(text, TIME_FORMAT)


# --- drawing -------------------------------------------------------------------------------


@pytest.mark.unit
def test_v01_tasks_have_no_targets(domain: DomainConfig) -> None:
    tasks = sample_tasks(load_spec("v0.1"), total=50, domain=domain)
    assert all(t.targets == {} and t.state_targets == {} for t in tasks)


@pytest.mark.unit
def test_every_value_slot_gets_a_target(v02: list[GenerationTask]) -> None:
    for t in v02:
        expected = {
            n
            for n, k in t.delta.items()
            if k == "value"
            and n != "game_mode"
            and not (n == "style_preference" and t.variant.startswith("voice_"))
        }
        assert set(t.targets) == expected, t.id
        assert set(t.state_targets) == set(t.state_fields) - {"game_mode"}, t.id


@pytest.mark.unit
def test_targets_cover_the_value_space(v02: list[GenerationTask]) -> None:
    ranks = Counter(
        v
        for t in v02
        for d in (t.targets, t.state_targets)
        for n, v in d.items()
        if n == "rank_requirement"
    )
    durations = {
        v
        for t in v02
        for d in (t.targets, t.state_targets)
        for n, v in d.items()
        if n == "duration_hours"
    }
    forms = Counter(
        t.targets["start_time_expr"]["form"] for t in v02 if "start_time_expr" in t.targets
    )
    picks = Counter(t.targets["companion_name"] for t in v02 if "companion_name" in t.targets)
    assert len(ranks) == 10 and max(ranks.values()) < 0.25 * sum(ranks.values())
    assert {1.0, 4.0, 8.0} <= durations and len(durations) >= 10
    assert set(forms) == {
        "relative_day",
        "weekday",
        "next_week",
        "digits",
        "after",
        "shift",
        "fuzzy",
    }
    assert set(picks) == {1, 2, 3, 4}


@pytest.mark.unit
def test_times_are_in_the_future_and_shifts_add_up(v02: list[GenerationTask]) -> None:
    for t in v02:
        state_time = t.state_targets.get("start_time_expr")
        if state_time:
            assert _time(state_time["time"]) >= t.now + MIN_LEAD, t.id
        target = t.targets.get("start_time_expr")
        if not target or "time" not in target:
            continue
        when = _time(target["time"])
        assert when >= t.now + MIN_LEAD, t.id
        if target["form"] == "shift":
            base = _time(state_time["time"])
            assert when == base + timedelta(hours=target["shift_hours"]), t.id
        if t.variant == "shift":
            assert target["form"] == "shift", t.id


@pytest.mark.unit
def test_modified_slots_get_a_different_target(v02: list[GenerationTask]) -> None:
    for t in v02:
        for name, target in t.targets.items():
            if name in t.state_targets and name not in ("start_time_expr", "style_preference"):
                old = t.state_targets[name]
                if isinstance(target, list):
                    assert set(target) != set(old), t.id
                else:
                    assert target != old, t.id


@pytest.mark.unit
def test_candidate_targets_fit_the_candidates(v02: list[GenerationTask]) -> None:
    for t in v02:
        for d in (t.targets, t.state_targets):
            if "companion_name" in d:
                assert 1 <= d["companion_name"] <= t.num_candidates, t.id
        if t.variant == "switch_pick":
            assert t.targets["companion_name"] != t.state_targets["companion_name"], t.id


@pytest.mark.unit
def test_target_rng_is_per_task() -> None:
    a = task_rng(1, "x").random()
    assert a == task_rng(1, "x").random()
    assert a != task_rng(1, "y").random() and a != task_rng(2, "x").random()


@pytest.mark.unit
@pytest.mark.parametrize("form", ["relative_day", "weekday", "next_week", "digits", "after"])
def test_absolute_forms(form: str) -> None:
    spec = load_spec("v0.2").values
    assert spec is not None
    for seed in range(200):
        drawer = TargetDrawer(spec, random.Random(seed), NOW)
        when = drawer.absolute_time(form)
        assert when >= NOW + MIN_LEAD
        days = (when.date() - NOW.date()).days
        if form == "weekday":
            assert 0 <= days <= 7
        if form == "next_week":
            assert when.isocalendar().week == NOW.isocalendar().week + 1
        if form == "after":
            assert timedelta(minutes=30) <= when - NOW <= timedelta(hours=3)


@pytest.mark.unit
def test_fuzzy_times_have_no_target() -> None:
    spec = load_spec("v0.2").values
    assert spec is not None
    _, values = draw_targets(
        spec,
        random.Random(0),
        now=NOW,
        variant="fuzzy",
        state_fields=(),
        delta={"start_time_expr": "value"},
        num_candidates=0,
    )
    assert values == {"start_time_expr": {"form": "fuzzy"}}


# --- schema, prompt, validation ------------------------------------------------------------


def make_task(**kw: Any) -> GenerationTask:
    base: dict[str, Any] = {
        "id": "t-1",
        "spec_version": "v0.2",
        "category": "candidate_ref",
        "variant": "pick_and_change",
        "hint": "选候选并改一项",
        "style": "standard",
        "style_hint": "规范",
        "now": NOW,
        "game_mode": GameMode.RANKED_SOLO_DUO,
        "state_fields": ("game_mode", "start_time_expr", "duration_hours", "rank_requirement"),
        "num_candidates": 3,
        "pending_confirmation": False,
        "history_turns": 1,
        "turn_intent": TurnIntent.BOOKING,
        "confirmation": Confirmation.NONE,
        "delta": {"start_time_expr": "value", "duration_hours": "value", "companion_name": "value"},
        "state_targets": {
            "start_time_expr": {"time": "2026-10-02 20:00"},
            "duration_hours": 2.0,
            "rank_requirement": "master",
        },
        "targets": {
            "start_time_expr": {"form": "shift", "shift_hours": 1.5, "time": "2026-10-02 21:30"},
            "duration_hours": 4.0,
            "companion_name": 3,
        },
    }
    return GenerationTask.model_validate({**base, **kw})


def good_answer(**kw: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "candidates": ["阿狸酱", "夜雨声烦", "北极星"],
        "state": {
            "game_mode": "ranked_solo_duo",
            "start_time": "2026-10-02 20:00",
            "duration_hours": 2,
            "rank_requirement": "master",
        },
        "history": [
            {"role": "user", "content": "明晚八点双排两小时，大师以上"},
            {"role": "assistant", "content": "1. 阿狸酱 2. 夜雨声烦 3. 北极星，选哪位？"},
        ],
        "user_input": "就北极星吧，往后推一个半小时，打四个小时",
        "values": {
            "start_time_expr": "往后推一个半小时",
            "duration_hours": 4,
            "companion_name": "北极星",
        },
    }
    answer.update(kw)
    return answer


def problems(task: GenerationTask, answer: dict[str, Any], domain: DomainConfig) -> list[str]:
    return validate_answer(task, json.dumps(answer, ensure_ascii=False), domain).problems


@pytest.mark.unit
def test_schema_pins_targets(domain: DomainConfig) -> None:
    props = answer_schema(make_task(), domain)["properties"]
    assert props["state"]["properties"]["start_time"] == {
        "type": "string",
        "enum": ["2026-10-02 20:00"],
    }
    assert props["state"]["properties"]["duration_hours"] == {"type": "number", "enum": [2.0]}
    assert props["state"]["properties"]["rank_requirement"]["enum"] == ["master"]
    assert props["values"]["properties"]["duration_hours"]["enum"] == [4.0]
    assert props["values"]["properties"]["companion_name"] == {"type": "string"}
    assert props["values"]["properties"]["start_time_expr"] == {"type": "string"}
    roles = answer_schema(
        make_task(delta={"role_preference": "value"}, targets={"role_preference": ["mid", "top"]}),
        domain,
    )["properties"]["values"]["properties"]["role_preference"]
    assert roles["items"]["enum"] == ["mid", "top"] and roles["minItems"] == roles["maxItems"] == 2
    voice = answer_schema(
        make_task(delta={"voice_required": "value"}, targets={"voice_required": False}), domain
    )["properties"]["values"]["properties"]["voice_required"]
    assert voice == {"type": "boolean", "enum": [False]}


@pytest.mark.unit
def test_prompt_states_the_targets(domain: DomainConfig) -> None:
    text = task_prompt(make_task(), domain)
    assert "start_time=2026-10-02 20:00" in text and 'rank_requirement="master"' in text
    assert "往后推 1.5 小时" in text and "2026-10-02 21:30 星期五" in text
    assert "选第 3 位候选" in text and "目标值 4.0" in text


@pytest.mark.unit
def test_answer_on_target_passes(domain: DomainConfig) -> None:
    assert problems(make_task(), good_answer(), domain) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {
                "values": {
                    "start_time_expr": "往后推一小时",
                    "duration_hours": 4,
                    "companion_name": "北极星",
                }
            },
            "target 2026-10-02 21:30",
        ),
        (
            {
                "values": {
                    "start_time_expr": "往后推一个半小时",
                    "duration_hours": 3,
                    "companion_name": "北极星",
                }
            },
            "duration_hours 3 is not the target",
        ),
        (
            {
                "values": {
                    "start_time_expr": "往后推一个半小时",
                    "duration_hours": 4,
                    "companion_name": "阿狸酱",
                }
            },
            "not candidate #3",
        ),
        (
            {
                "state": {
                    "game_mode": "ranked_solo_duo",
                    "start_time": "2026-10-02 19:00",
                    "duration_hours": 2,
                    "rank_requirement": "master",
                }
            },
            "state start_time",
        ),
        (
            {
                "state": {
                    "game_mode": "ranked_solo_duo",
                    "start_time": "2026-10-02 20:00",
                    "duration_hours": 2,
                    "rank_requirement": "diamond",
                }
            },
            "state rank_requirement 'diamond'",
        ),
    ],
)
def test_answers_off_target_fail(
    domain: DomainConfig, change: dict[str, Any], message: str
) -> None:
    found = problems(make_task(), good_answer(**change), domain)
    assert any(message in p for p in found), found


@pytest.mark.unit
def test_absolute_time_target_is_parsed(domain: DomainConfig) -> None:
    task = make_task(
        state_fields=("game_mode",),
        state_targets={},
        delta={"start_time_expr": "value"},
        targets={"start_time_expr": {"form": "weekday", "time": "2026-10-03 21:30"}},
        num_candidates=0,
        category="relative_time",
        variant="weekday",
    )
    answer = good_answer(
        candidates=[],
        state={"game_mode": "ranked_solo_duo"},
        user_input="周六晚上九点半吧",
        values={"start_time_expr": "周六晚上九点半"},
    )
    assert problems(task, answer, domain) == []
    answer.update(user_input="周六晚上九点吧", values={"start_time_expr": "周六晚上九点"})
    assert any("resolves to 2026-10-03 21:00" in p for p in problems(task, answer, domain))
