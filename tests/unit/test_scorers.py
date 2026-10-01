"""H2: L2 scorers — protocol layer, task layer (tri-state keys, time, roles, style)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar

import pytest

from rift_common.embedding import MockEmbedding, Vector
from rift_common.settings import EmbeddingConfig
from rift_domain.slots import BookingState, SlotExtraction, parse_extraction
from rift_training.evaluation.preferences import (
    EmbeddingStyleMatcher,
    ExactStyleMatcher,
    make_style_matcher,
    normalize_style,
)
from rift_training.evaluation.protocol import check_protocol
from rift_training.evaluation.task import ABSENT, score_output, score_task

NOW = datetime(2026, 10, 1, 14, 0)
EMPTY = BookingState()
CURRENT = BookingState(game_mode="ranked_solo_duo", start_time=datetime(2026, 10, 2, 20, 0))


def ext(confirmation: str = "none", intent: str = "booking", **delta: Any) -> SlotExtraction:
    return parse_extraction({"turn_intent": intent, "delta": delta, "confirmation": confirmation})


def raw(confirmation: str = "none", intent: str = "booking", **delta: Any) -> str:
    return json.dumps(
        {"turn_intent": intent, "delta": delta, "confirmation": confirmation}, ensure_ascii=False
    )


def task(expected: SlotExtraction, predicted: SlotExtraction, state: BookingState = EMPTY) -> Any:
    return score_task(expected, predicted, state=state, now=NOW)


# --- protocol ------------------------------------------------------------------------------


def test_protocol_accepts_valid_output() -> None:
    result = check_protocol('  {"turn_intent": "booking", "delta": {}, "confirmation": "none"} ')
    assert result.ok and result.error is None


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("", "empty"),
        ('```json\n{"turn_intent": "booking"}\n```', "json"),
        ("{'turn_intent': 'booking'}", "json"),
        ("[1, 2]", "not_object"),
        (raw(mood="happy"), "extra_key"),
        ('{"turn_intent": "booking", "delta": {}}', "missing_key"),
        (raw(game_mode="dota"), "enum"),
        (raw(intent="chitchat"), "enum"),
        (raw(duration_hours="2"), "type"),
        (raw(voice_required=1), "type"),
        (raw(duration_hours=2.3), "type"),
    ],
)
def test_protocol_errors(text: str, error: str) -> None:
    result = check_protocol(text)
    assert not result.ok
    assert result.error == error
    assert result.messages


def test_protocol_reports_extra_key_first_when_several_rules_break() -> None:
    result = check_protocol(raw(game_mode="dota", mood="x"))
    assert result.error == "extra_key"
    assert len(result.messages) == 2


# --- task: keys and tri-state --------------------------------------------------------------


def test_identical_output_scores_one() -> None:
    e = ext(game_mode="aram", start_time_expr="明晚八点", duration_hours=2)
    score = task(e, e)
    assert score.score == 1.0 and score.errors == ()
    assert [f.field for f in score.fields] == [
        "turn_intent",
        "confirmation",
        "delta.duration_hours",
        "delta.game_mode",
        "delta.start_time_expr",
    ]


def test_extra_key_is_wrong() -> None:
    """Acceptance: one key too many (should have been "not mentioned") fails the sample."""
    expected = ext(rank_requirement="diamond")
    predicted = ext(rank_requirement="diamond", service_type="climb")
    score = task(expected, predicted)
    (error,) = score.errors
    assert error.field == "delta.service_type"
    assert error.error == "extra_key" and error.expected == ABSENT
    assert score.score == pytest.approx(3 / 4)
    sample = score_output(raw(rank_requirement="diamond", service_type="climb"), expected,
                          state=EMPTY, now=NOW)  # fmt: skip
    assert sample.protocol.ok and not sample.passed


def test_missing_key_is_wrong() -> None:
    (error,) = task(ext(game_mode="aram", duration_hours=1), ext(game_mode="aram")).errors
    assert error.field == "delta.duration_hours"
    assert error.error == "missing_key" and error.predicted == ABSENT


def test_null_any_and_absent_are_three_different_answers() -> None:
    withdraw, no_pref, absent = ext(rank_requirement=None), ext(rank_requirement="any"), ext()
    assert task(withdraw, withdraw).score == 1.0
    assert task(withdraw, no_pref).errors[0].error == "wrong_value"
    assert task(withdraw, absent).errors[0].error == "missing_key"
    assert task(absent, no_pref).errors[0].error == "extra_key"


def test_intent_and_confirmation_are_scored() -> None:
    score = task(ext("yes"), ext("none", intent="consult"))
    assert {f.field for f in score.errors} == {"turn_intent", "confirmation"}
    assert score.score == 0.0


def test_empty_delta_on_both_sides() -> None:
    assert task(ext(intent="unrelated"), ext(intent="unrelated")).score == 1.0


# --- task: values --------------------------------------------------------------------------


def test_equivalent_time_expressions_match() -> None:
    """Acceptance: "明晚8点" and "明天晚上八点" resolve to the same time."""
    score = task(ext(start_time_expr="明晚8点"), ext(start_time_expr="明天晚上八点"))
    assert score.score == 1.0
    (field,) = [f for f in score.fields if f.field == "delta.start_time_expr"]
    assert "2026-10-02 20:00" in field.detail


def test_different_times_do_not_match() -> None:
    score = task(ext(start_time_expr="明晚八点"), ext(start_time_expr="明晚九点"))
    assert score.errors[0].field == "delta.start_time_expr"


def test_relative_time_resolves_against_current_state() -> None:
    expected = ext(start_time_expr="晚一小时")
    assert task(expected, ext(start_time_expr="改成晚一个小时"), CURRENT).score == 1.0
    assert task(expected, ext(start_time_expr="九点"), CURRENT).score == 1.0  # 21:00 too
    assert task(expected, ext(start_time_expr="提前一小时"), CURRENT).errors


def test_unresolved_expressions_compare_as_text() -> None:
    assert task(ext(start_time_expr="明晚"), ext(start_time_expr="明晚 ")).score == 1.0
    assert task(ext(start_time_expr="明晚"), ext(start_time_expr="周末")).errors
    # expected is vague but the prediction invented an hour
    assert task(ext(start_time_expr="明晚"), ext(start_time_expr="明晚八点")).errors


def test_unresolved_expression_completed_by_state_like_merge() -> None:
    state = BookingState(start_time_expr="明晚")
    assert task(ext(start_time_expr="八点"), ext(start_time_expr="8点"), state).score == 1.0


def test_withdrawn_time_vs_value() -> None:
    assert task(ext(start_time_expr=None), ext(start_time_expr=None)).score == 1.0
    assert task(ext(start_time_expr=None), ext(start_time_expr="明晚八点")).errors


def test_roles_compare_as_sets() -> None:
    assert task(ext(role_preference=["mid", "top"]), ext(role_preference=["top", "mid"])).score == 1
    assert task(ext(role_preference=["mid", "top"]), ext(role_preference=["mid"])).errors
    assert task(ext(role_preference="any"), ext(role_preference=["mid"])).errors


@pytest.mark.parametrize(
    ("key", "expected", "predicted", "ok"),
    [
        ("duration_hours", 2, 2.0, True),
        ("duration_hours", 2, 2.5, False),
        ("budget_per_hour", 60, 60.0, True),
        ("budget_per_hour", "any", 60, False),
        ("voice_required", True, True, True),
        ("voice_required", True, "any", False),
        ("game_mode", "ranked_flex", "ranked_solo_duo", False),
        ("companion_name", "夜雨声烦", "夜雨声烦", True),
        ("companion_name", "夜雨声烦", "阿狸酱", False),
    ],
)
def test_plain_values(key: str, expected: Any, predicted: Any, ok: bool) -> None:
    score = task(ext(**{key: expected}), ext(**{key: predicted}))
    assert (score.score == 1.0) is ok


# --- style preference ----------------------------------------------------------------------


class FixedEmbedding(MockEmbedding):
    """Hand-picked vectors so similarities are known; counts embedded texts."""

    VECTORS: ClassVar[dict[str, Vector]] = {
        "温柔会聊天": [1.0, 0.0],
        "性格温柔健谈": [0.8, 0.6],  # cos 0.8
        "高冷": [0.0, 1.0],  # cos 0.0
    }

    def __init__(self) -> None:
        super().__init__(EmbeddingConfig(provider="mock"))
        self.seen: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        self.seen.extend(texts)
        return [self.VECTORS[t] for t in texts]


def test_normalize_style() -> None:
    assert normalize_style("温柔一点、会聊天的！") == "温柔会聊天"
    assert normalize_style("一只小熊") == "一只小熊"


def test_exact_matcher() -> None:
    m = ExactStyleMatcher()
    assert m.matches("温柔会聊天", "温柔、会聊天的")
    assert not m.matches("温柔会聊天", "性格温柔健谈")


def test_embedding_matcher_threshold_and_cache() -> None:
    embedding = FixedEmbedding()
    m = EmbeddingStyleMatcher(embedding, threshold=0.7)
    assert m.similarity("温柔会聊天", "性格温柔健谈") == pytest.approx(0.8)
    assert m.matches("温柔会聊天", "性格温柔健谈")
    assert not m.matches("温柔会聊天", "高冷")
    assert m.similarity("温柔会聊天", "温柔一点会聊天") == 1.0  # normalized, no embedding
    assert embedding.seen.count("温柔会聊天") == 1


def test_style_scored_by_matcher_with_detail() -> None:
    m = EmbeddingStyleMatcher(FixedEmbedding(), threshold=0.7)
    expected = ext(style_preference="温柔会聊天")
    ok = score_task(expected, ext(style_preference="性格温柔健谈"), state=EMPTY, now=NOW, style=m)
    assert ok.score == 1.0
    bad = score_task(expected, ext(style_preference="高冷"), state=EMPTY, now=NOW, style=m)
    assert "similarity 0.000" in bad.errors[0].detail
    # "any" is never fuzzy-matched
    anyp = score_task(ext(style_preference="any"), ext(style_preference="温柔会聊天"),
                      state=EMPTY, now=NOW, style=m)  # fmt: skip
    assert anyp.errors


def test_make_style_matcher() -> None:
    assert isinstance(make_style_matcher(None), ExactStyleMatcher)
    m = make_style_matcher(EmbeddingConfig(provider="mock"), threshold=0.5)
    assert isinstance(m, EmbeddingStyleMatcher) and m.threshold == 0.5
    assert m.similarity("温柔", "温柔") == 1.0


# --- combined verdict ----------------------------------------------------------------------


def test_score_output_pass_and_protocol_failure() -> None:
    expected = ext(game_mode="aram", start_time_expr="今晚十点")
    good = score_output(raw(game_mode="aram", start_time_expr="今晚10点"), expected,
                        state=EMPTY, now=NOW)  # fmt: skip
    assert good.passed and good.task_score == 1.0

    broken = score_output("not json", expected, state=EMPTY, now=NOW)
    assert not broken.passed and broken.task is None and broken.task_score == 0.0
    assert broken.protocol.error == "json"


def test_score_output_threshold() -> None:
    expected = ext(game_mode="aram", start_time_expr="今晚十点", duration_hours=1)
    almost = raw(game_mode="aram", start_time_expr="今晚十点", duration_hours=2)
    assert not score_output(almost, expected, state=EMPTY, now=NOW).passed
    assert score_output(almost, expected, state=EMPTY, now=NOW, threshold=0.8).passed
