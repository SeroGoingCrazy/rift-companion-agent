"""H1: L2 slot datasets — schema, consistency rules, category coverage and checksums."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rift_training.evaluation.dataset import (
    CATEGORIES,
    DatasetError,
    SlotSample,
    check_datasets,
    checksum_problems,
    distribution_table,
    load_samples,
    parse_samples,
    sample_problems,
    write_checksums,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS = REPO_ROOT / "eval" / "datasets"


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validator = _load("validate_datasets", REPO_ROOT / "eval" / "runners" / "validate_datasets.py")


def row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "t-1",
        "category": "first_turn",
        "now": "2026-10-01 14:00",
        "user_input": "帮我约个明晚八点的单双排",
        "expected": {
            "turn_intent": "booking",
            "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明晚八点"},
            "confirmation": "none",
        },
    }
    base.update(overrides)
    return base


def sample(**overrides: Any) -> SlotSample:
    return SlotSample.model_validate(row(**overrides))


def lines(*rows: dict[str, Any]) -> list[str]:
    return [json.dumps(r, ensure_ascii=False) for r in rows]


# --- the committed datasets ----------------------------------------------------------------


def test_repo_datasets_are_valid_and_frozen() -> None:
    problems, table = validator.validate(DATASETS)
    assert problems == []
    assert "| 首轮抽取 | `first_turn` |" in table


def test_repo_datasets_sizes_and_categories() -> None:
    main = load_samples(DATASETS / "slot_main.jsonl")
    holdout = load_samples(DATASETS / "slot_holdout.jsonl")
    assert 55 <= len(main) <= 70
    assert 22 <= len(holdout) <= 30
    for samples in (main, holdout):
        assert {s.category for s in samples} == set(CATEGORIES)


def test_repo_datasets_keep_tri_state_literally() -> None:
    main = {s.id: s for s in load_samples(DATASETS / "slot_main.jsonl")}
    withdrawn = main["main-018"].extraction().delta
    assert withdrawn.provided() == {"start_time_expr": None}
    any_rank = main["main-016"].extraction().delta
    assert any_rank.provided() == {"rank_requirement": "any"}


# --- sample schema -------------------------------------------------------------------------


def test_sample_round_trip() -> None:
    s = sample(
        current_state={"game_mode": "aram", "start_time": "2026-10-02 20:00"},
        history=[
            {"role": "user", "content": "约个大乱斗"},
            {"role": "assistant", "content": "几点？"},
        ],
    )
    assert s.state().start_time is not None and s.state().start_time.hour == 20
    assert s.extraction().delta.provided()["game_mode"] == "ranked_solo_duo"
    assert s.history_messages()[0] == {"role": "user", "content": "约个大乱斗"}


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"category": "nope"}, "unknown category"),
        ({"current_state": {"quote": None}}, "unknown current_state keys"),
        ({"current_state": {"game_mode": "dota"}}, "current_state"),
        (
            {
                "expected": {
                    "turn_intent": "booking",
                    "delta": {"mood": "x"},
                    "confirmation": "none",
                }
            },
            "expected",
        ),
        (
            {
                "expected": {
                    "turn_intent": "booking",
                    "delta": {"duration_hours": "2"},
                    "confirmation": "none",
                }
            },
            "expected",
        ),
        ({"history": [{"role": "system", "content": "x"}]}, "history"),
        ({"extra": 1}, "extra"),
    ],
)
def test_parse_samples_reports_bad_lines(overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(DatasetError) as exc:
        parse_samples(["", *lines(row(**overrides))], source="x.jsonl")
    assert "x.jsonl:2" in str(exc.value)
    assert fragment in str(exc.value)


def test_parse_samples_collects_every_error() -> None:
    with pytest.raises(DatasetError) as exc:
        parse_samples(["{not json", *lines(row(category="bad"))])
    message = str(exc.value)
    assert "<memory>:1: invalid JSON" in message and "<memory>:2" in message


def test_missing_file() -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_samples(Path("does/not/exist.jsonl"))


# --- consistency rules ---------------------------------------------------------------------


def test_companion_name_must_be_a_candidate() -> None:
    s = sample(
        category="candidate_ref",
        candidates=["阿狸酱", "夜雨声烦"],
        expected={
            "turn_intent": "booking",
            "delta": {"companion_name": "小鹿"},
            "confirmation": "none",
        },
    )
    assert any("not one of the candidates" in p for p in sample_problems(s))


def test_candidate_ref_needs_candidates() -> None:
    s = sample(category="candidate_ref")
    assert "candidate_ref sample without candidates" in sample_problems(s)


def test_non_booking_turn_carries_no_slots() -> None:
    s = sample(
        expected={"turn_intent": "consult", "delta": {"game_mode": "aram"}, "confirmation": "none"}
    )
    assert any("should not carry slot values" in p for p in sample_problems(s))


def test_yes_requires_pending_confirmation() -> None:
    yes = {"turn_intent": "booking", "delta": {}, "confirmation": "yes"}
    assert sample_problems(sample(expected=yes))
    assert not sample_problems(sample(expected=yes, pending_confirmation=True))
    no = {"turn_intent": "booking", "delta": {}, "confirmation": "no"}
    assert not sample_problems(sample(expected=no))  # explicit abandon without a pending order


def test_history_must_end_with_assistant() -> None:
    s = sample(history=[{"role": "user", "content": "hi"}])
    assert "history must end with the assistant's message" in sample_problems(s)


def _full_set(prefix: str, text: str = "x") -> list[SlotSample]:
    return [
        sample(
            id=f"{prefix}-{cat}-{i}",
            category=cat,
            user_input=f"{prefix}{cat}{i}{text}",
            candidates=["阿狸酱"] if cat == "candidate_ref" else [],
        )
        for cat in CATEGORIES
        for i in range(3)
    ]


def test_check_datasets_passes_balanced_sets() -> None:
    assert check_datasets({"main": _full_set("m"), "holdout": _full_set("h")}) == []


def test_check_datasets_flags_thin_categories_duplicates_and_shared_inputs() -> None:
    main = _full_set("m")[1:]  # first_turn now has 2
    holdout = [*_full_set("h"), sample(id="h-dup", user_input="帮我约个 明晚八点的单双排！")]
    main.append(sample(id="m-dup", category="multi_turn"))
    main.append(sample(id="m-dup", category="multi_turn", user_input="别的"))
    problems = check_datasets({"main": main, "holdout": holdout})
    assert any("first_turn has 2 samples" in p for p in problems)
    assert any("duplicate id m-dup" in p for p in problems)
    assert any("user_input shared by main:m-dup and holdout:h-dup" in p for p in problems)


def test_distribution_table_counts() -> None:
    table = distribution_table({"main": _full_set("m"), "holdout": _full_set("h")[:3]})
    assert "| 首轮抽取 | `first_turn` | 3 | 3 |" in table
    assert "| 三态语义 | `tri_state` | 3 | 0 |" in table
    assert "**27** | **3**" in table


# --- checksums -----------------------------------------------------------------------------


def test_checksums_detect_changes(tmp_path: Path) -> None:
    data = tmp_path / "slot_main.jsonl"
    data.write_text("\n".join(lines(row())) + "\n", encoding="utf-8")
    sums = tmp_path / "CHECKSUMS"
    assert checksum_problems(sums, [data]) == ["slot_main.jsonl: no checksum recorded in CHECKSUMS"]
    write_checksums(sums, [data])
    assert b"\r\n" not in sums.read_bytes()
    assert checksum_problems(sums, [data]) == []
    data.write_text(data.read_text("utf-8") + "\n", encoding="utf-8")
    assert "checksum mismatch" in checksum_problems(sums, [data])[0]


def test_validator_cli_writes_checksums(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "slot_main.jsonl").write_text(
        "\n".join(
            json.dumps(s.model_dump(mode="json"), ensure_ascii=False) for s in _full_set("m")
        ),
        encoding="utf-8",
    )
    (tmp_path / "slot_holdout.jsonl").write_text(
        "\n".join(
            json.dumps(s.model_dump(mode="json"), ensure_ascii=False) for s in _full_set("h")
        ),
        encoding="utf-8",
    )
    assert validator.main(["--dir", str(tmp_path)]) == 1
    assert "no checksum recorded" in capsys.readouterr().out
    assert validator.main(["--dir", str(tmp_path), "--write-checksums"]) == 0
    assert validator.main(["--dir", str(tmp_path)]) == 0
