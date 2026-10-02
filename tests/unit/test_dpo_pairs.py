"""I8: rule-perturbation and on-policy preference pairs, and their LLaMA-Factory rendering."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from rift_training.contract import parse_extraction, render_prompt, render_target
from rift_training.data import dpo_rule
from rift_training.data.dpo_onpolicy import onpolicy_pairs, read_samples
from rift_training.data.dpo_rule import build_rule_pairs, candidates_for, is_wrong
from rift_training.data.preference import (
    DATASET_INFO,
    PAIRS,
    PreferencePair,
    train_samples,
    write_pairs,
)
from rift_training.evaluation.dataset import SlotSample, parse_samples
from rift_training.evaluation.preferences import ExactStyleMatcher


def sample(user_input: str, delta: dict[str, Any], **kw: Any) -> SlotSample:
    expected = {
        "turn_intent": kw.pop("turn_intent", "booking"),
        "delta": delta,
        "confirmation": kw.pop("confirmation", "none"),
    }
    row = {
        "id": kw.pop("id", "s-1"),
        "category": kw.pop("category", "multi_turn"),
        "now": "2026-10-01 14:00",
        "user_input": user_input,
        "expected": expected,
        **kw,
    }
    return parse_samples([json.dumps(row, ensure_ascii=False)])[0]


RNG = random.Random(0)


def apply(name: str, s: SlotSample) -> dict[str, Any] | None:
    result: dict[str, Any] | None = dpo_rule.PERTURBATIONS[name](s, s.expected, RNG)
    return result


# --- perturbations -------------------------------------------------------------------------


@pytest.mark.unit
def test_tri_state_perturbations() -> None:
    s = sample("风格没要求，性别先不定", {"style_preference": "any", "companion_gender": None})
    assert apply("any_to_null", s)["delta"] == {"style_preference": None, "companion_gender": None}
    assert apply("null_to_any", s)["delta"] == {
        "style_preference": "any",
        "companion_gender": "any",
    }
    assert apply("any_to_absent", s)["delta"] == {"companion_gender": None}
    plain = sample("两小时", {"duration_hours": 2})
    assert apply("any_to_null", plain) is None and apply("null_to_any", plain) is None


@pytest.mark.unit
def test_extra_null_prefers_the_companion() -> None:
    s = sample(
        "不了，换一个人吧",
        {},
        confirmation="no",
        current_state={"game_mode": "aram", "companion_name": "阿狸酱"},
        candidates=["阿狸酱", "夜雨声烦"],
        pending_confirmation=True,
        history=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "确认吗"}],
    )
    assert apply("extra_null", s)["delta"] == {"companion_name": None}
    assert apply("extra_null", sample("讲个笑话", {}, turn_intent="unrelated")) is None


@pytest.mark.unit
def test_role_append_time_swallow_and_name_literal() -> None:
    swap = sample(
        "上单不要了改中路", {"role_preference": ["mid"]}, current_state={"role_preference": ["top"]}
    )
    assert apply("role_append", swap)["delta"] == {"role_preference": ["top", "mid"]}

    dense = sample(
        "后天下午三点三小时大乱斗",
        {"start_time_expr": "后天下午三点", "duration_hours": 3, "game_mode": "aram"},
    )
    assert apply("time_swallow", dense)["delta"] == {
        "start_time_expr": "后天下午三点三小时大乱斗",
        "game_mode": "aram",
    }
    spaced = sample(
        "后天下午三点，三小时", {"start_time_expr": "后天下午三点", "duration_hours": 3}
    )
    assert apply("time_swallow", spaced) is None  # nothing glued on

    pick = sample(
        "2号吧",
        {"companion_name": "夜雨声烦"},
        category="candidate_ref",
        candidates=["阿狸酱", "夜雨声烦"],
        history=[{"role": "user", "content": "x"}, {"role": "assistant", "content": "选哪位"}],
    )
    assert apply("name_literal", pick)["delta"] == {"companion_name": "2号吧"}


@pytest.mark.unit
def test_intent_and_confirmation_flips() -> None:
    assert (
        apply("intent_flip", sample("退款怎么算", {}, turn_intent="consult"))["turn_intent"]
        == "unrelated"
    )
    assert (
        apply("intent_flip", sample("订机票", {}, turn_intent="unrelated"))["turn_intent"]
        == "consult"
    )
    assert apply("intent_flip", sample("两小时", {"duration_hours": 2})) is None
    assert (
        apply(
            "confirm_flip",
            sample(
                "确认",
                {},
                confirmation="yes",
                pending_confirmation=True,
                history=[
                    {"role": "user", "content": "x"},
                    {"role": "assistant", "content": "确认吗"},
                ],
            ),
        )["confirmation"]
        == "none"
    )


@pytest.mark.unit
def test_value_swap_changes_exactly_one_slot() -> None:
    s = sample(
        "钻石，女生，三小时",
        {"rank_requirement": "diamond", "companion_gender": "female", "duration_hours": 3},
    )
    rejected = apply("value_swap", s)
    changed = [k for k, v in rejected["delta"].items() if v != s.expected["delta"][k]]
    assert len(changed) == 1


@pytest.mark.unit
def test_every_candidate_is_valid_and_wrong() -> None:
    s = sample(
        "风格没要求，上单不要了改中路，后天下午三点三小时",
        {
            "style_preference": "any",
            "role_preference": ["mid"],
            "start_time_expr": "后天下午三点",
            "duration_hours": 3,
        },
        current_state={"role_preference": ["top"], "game_mode": "normal_draft"},
    )
    found = candidates_for(s)
    kinds = {k for k, _ in found}
    assert {"any_to_null", "role_append", "time_swallow", "drop_key", "extra_null"} <= kinds
    for _, text in found:
        parse_extraction(text)  # protocol-valid
        assert is_wrong(s, text)
        assert text != render_target(s.expected)


@pytest.mark.unit
def test_build_rule_pairs_is_deterministic() -> None:
    samples = [
        sample(f"风格没要求{i}", {"style_preference": "any", "duration_hours": 2}, id=f"s-{i}")
        for i in range(20)
    ]
    a = build_rule_pairs(samples)
    assert [(p.sample.id, p.kind) for p in a] == [
        (p.sample.id, p.kind) for p in build_rule_pairs(samples)
    ]
    two = build_rule_pairs(samples, per_sample=2)
    assert len(two) == 40
    assert all(len({p.kind for p in two if p.sample.id == s.id}) == 2 for s in samples)


# --- on-policy -----------------------------------------------------------------------------


@pytest.mark.unit
def test_onpolicy_pairs_keep_distinct_failures() -> None:
    s = sample("两小时", {"duration_hours": 2})
    good = render_target(s.expected)
    extra = (
        '{"turn_intent": "booking", "delta": {"duration_hours": 2, "game_mode": null}, '
        '"confirmation": "none"}'
    )
    broken = "not json"
    pairs, stats = onpolicy_pairs(
        [s], {"s-1": [good, extra, extra, broken]}, ExactStyleMatcher(), max_per_sample=5
    )
    assert [p.rejected for p in pairs] == [extra, broken]
    assert pairs[0].kind == "onpolicy:field:delta.game_mode"
    assert pairs[1].kind.startswith("onpolicy:protocol:")
    assert stats.as_dict()["generation_pass_rate"] == 0.25 and stats.samples_with_failure == 1
    capped, _ = onpolicy_pairs([s], {"s-1": [extra, broken]}, ExactStyleMatcher(), max_per_sample=1)
    assert len(capped) == 1


@pytest.mark.unit
def test_onpolicy_all_correct_gives_no_pairs(tmp_path: Path) -> None:
    s = sample("两小时", {"duration_hours": 2})
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps({"id": "s-1", "samples": [render_target(s.expected)] * 4}) + "\n", "utf-8"
    )
    pairs, stats = onpolicy_pairs([s], read_samples(path), ExactStyleMatcher())
    assert pairs == [] and stats.passed == 4


# --- rendering -----------------------------------------------------------------------------


@pytest.mark.unit
def test_write_pairs_in_ranking_format(tmp_path: Path) -> None:
    s = sample("两小时", {"duration_hours": 2})
    pair = PreferencePair(
        s, '{"turn_intent": "booking", "delta": {}, "confirmation": "none"}', "drop_key"
    )
    card = write_pairs(tmp_path, [pair], name="rift_dpo_test", source={"x": 1})
    record = json.loads((tmp_path / PAIRS).read_text("utf-8"))
    assert record["messages"] == render_prompt(s)
    assert record["chosen"] == {"role": "assistant", "content": render_target(s.expected)}
    assert record["rejected"]["role"] == "assistant"
    info = json.loads((tmp_path / DATASET_INFO).read_text("utf-8"))["rift_dpo_test"]
    assert info["ranking"] is True and info["columns"]["rejected"] == "rejected"
    assert card["kinds"] == {"drop_key": 1} and len(card["prompt_sha256"]) == 64


@pytest.mark.unit
def test_train_samples_need_every_id(tmp_path: Path) -> None:
    sft, raw = tmp_path / "sft", tmp_path / "raw"
    sft.mkdir()
    raw.mkdir()
    s = sample("两小时", {"duration_hours": 2})
    (raw / "accepted.jsonl").write_text(
        json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n", "utf-8"
    )
    (sft / "train.jsonl").write_text(json.dumps({"id": "s-1"}) + "\n", "utf-8")
    assert [x.id for x in train_samples(sft, [raw])] == ["s-1"]
    (sft / "train.jsonl").write_text(json.dumps({"id": "nope"}) + "\n", "utf-8")
    with pytest.raises(ValueError, match="not found"):
        train_samples(sft, [raw])
