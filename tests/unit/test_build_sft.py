"""I4: dedup against the eval sets, coverage audit, LLaMA-Factory rendering and the data card."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rift_common.embedding import BaseEmbedding, Vector
from rift_common.settings import EmbeddingConfig
from rift_domain.config import DomainConfig
from rift_training.contract import prompt_hash, render_conversation
from rift_training.data.audit import alias_coverage, audit, missing_categories
from rift_training.data.card import load_card
from rift_training.data.dedup import (
    dedup_within,
    find_leaks,
    nearest_eval,
    similarity_histogram,
)
from rift_training.data.render_sft import DATASET_INFO, TRAIN, VAL, dataset_info, split, to_record
from rift_training.evaluation.dataset import CATEGORIES, SlotSample, parse_samples

SCRIPT = Path(__file__).resolve().parents[2] / "training" / "scripts" / "data" / "build_sft.py"


def row(id_: str, user_input: str, category: str = "first_turn", **kw: Any) -> dict[str, Any]:
    return {
        "id": id_,
        "category": category,
        "now": "2026-10-01 14:00",
        "user_input": user_input,
        "expected": {"turn_intent": "booking", "delta": {}, "confirmation": "none"},
        "note": "multi_slot; colloquial",
        **kw,
    }


def samples(*rows: dict[str, Any]) -> list[SlotSample]:
    return parse_samples([json.dumps(r, ensure_ascii=False) for r in rows])


class TableEmbedding(BaseEmbedding):
    """Fixed vectors per text; unknown texts are orthogonal to everything."""

    provider_name = "table"

    def __init__(self, table: dict[str, Vector]) -> None:
        super().__init__(EmbeddingConfig(provider="table", model="table"))
        self.table = table

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self.table.get(t, [0.0, 0.0, 1.0]) for t in texts]


EVAL = samples(row("main-1", "确认下单"), row("main-2", "今晚十点来两把大乱斗"))


# --- dedup ---------------------------------------------------------------------------------


@pytest.mark.unit
def test_exact_leaks_ignore_punctuation_and_spaces() -> None:
    train = samples(row("t1", "确认 下单！"), row("t2", "明晚八点双排"))
    leaks = find_leaks(train, EVAL, None)
    assert [(leak.sample_id, leak.eval_id, leak.reason) for leak in leaks] == [
        ("t1", "main-1", "exact")
    ]


@pytest.mark.unit
def test_embedding_leaks_use_the_threshold() -> None:
    train = samples(row("t1", "整大乱斗，今晚十点"), row("t2", "明晚八点双排"))
    embedder = TableEmbedding(
        {
            "今晚十点来两把大乱斗": [1.0, 0.0, 0.0],
            "整大乱斗，今晚十点": [0.95, 0.31, 0.0],  # cosine ~0.95
            "明晚八点双排": [0.6, 0.8, 0.0],  # cosine 0.6
        }
    )
    leaks = find_leaks(train, EVAL, embedder, threshold=0.92)
    assert [(leak.sample_id, leak.eval_id, leak.reason) for leak in leaks] == [
        ("t1", "main-2", "embedding")
    ]
    assert find_leaks(train, EVAL, embedder, threshold=0.99) == []
    nearest = {s.id: (e.id, round(sim, 2)) for s, e, sim, _ in nearest_eval(train, EVAL, embedder)}
    assert nearest == {"t1": ("main-2", 0.95), "t2": ("main-2", 0.6)}


@pytest.mark.unit
def test_in_set_duplicates_need_the_same_context() -> None:
    state = {"game_mode": "aram"}
    train = samples(
        row("t1", "两个小时"),
        row("t2", "两个小时。"),
        row("t3", "两个小时", current_state=state),
        row("t4", "三个小时"),
    )
    kept, removed = dedup_within(train)
    assert [s.id for s in kept] == ["t1", "t3", "t4"]
    assert removed == [("t2", "t1")]


@pytest.mark.unit
def test_similarity_histogram_buckets() -> None:
    hist = similarity_histogram([0.1, 0.65, 0.95, 0.99, 0.7], (0.0, 0.6, 0.7, 0.95))
    assert hist == {"0.0-0.6": 1, "0.6-0.7": 1, "0.7-0.95": 1, ">=0.95": 2}
    assert list(hist)[-1] == ">=0.95"


# --- audit ---------------------------------------------------------------------------------


@pytest.mark.unit
def test_audit_counts_coverage(domain: DomainConfig) -> None:
    train = samples(
        row(
            "t1",
            "整个海克斯，带飞",
            expected={
                "turn_intent": "booking",
                "delta": {"game_mode": "aram_mayhem", "service_type": "climb"},
                "confirmation": "none",
            },
        ),
        row(
            "t2",
            "风格无所谓",
            category="tri_state",
            note="any_optional; slang",
            expected={
                "turn_intent": "booking",
                "delta": {"style_preference": "any"},
                "confirmation": "none",
            },
        ),
    )
    report = audit(train, domain)
    assert report["total"] == 2
    assert report["variants"] == {"first_turn.multi_slot": 1, "tri_state.any_optional": 1}
    assert report["styles"] == {"colloquial": 1, "slang": 1}
    assert report["game_modes"] == {"aram_mayhem": 1, "-": 1}
    assert report["delta_fields"]["style_preference"] == {"value": 0, "any": 1, "null": 0}
    assert set(missing_categories(report)) == set(CATEGORIES) - {"first_turn", "tri_state"}
    coverage = alias_coverage(train, domain)
    assert coverage["counts"]["game_mode"]["aram_mayhem:海克斯"] == 1
    assert coverage["counts"]["service_type"]["climb:带飞"] == 1
    assert "game_mode.arena:斗魂" in coverage["unseen"]


# --- rendering -----------------------------------------------------------------------------


@pytest.mark.unit
def test_records_are_the_rendered_conversation() -> None:
    sample = samples(row("t1", "两个小时"))[0]
    record = to_record(sample)
    assert record == {"id": "t1", "messages": render_conversation(sample)}
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]


@pytest.mark.unit
def test_split_is_per_category_disjoint_and_deterministic() -> None:
    rows = [row(f"t{c}{i}", f"话{c}{i}", category=c) for c in CATEGORIES for i in range(10)]
    data = samples(*rows)
    train, val = split(data, val_ratio=0.1, seed=3)
    assert {s.category for s in val} == set(CATEGORIES)
    assert len(val) == 9 and len(train) == 81
    assert {s.id for s in train}.isdisjoint({s.id for s in val})
    assert split(data, val_ratio=0.1, seed=3) == (train, val)
    assert split(data, val_ratio=0.1, seed=4) != (train, val)
    assert split(data, val_ratio=0) == (sorted(data, key=lambda s: s.id), [])


@pytest.mark.unit
def test_dataset_info_registers_train_and_val() -> None:
    info = dataset_info("rift_sft_v0_1")
    assert set(info) == {"rift_sft_v0_1_train", "rift_sft_v0_1_val"}
    entry = info["rift_sft_v0_1_train"]
    assert entry["file_name"] == TRAIN and entry["formatting"] == "sharegpt"
    assert entry["tags"]["system_tag"] == "system" and entry["columns"] == {"messages": "messages"}


# --- end to end ----------------------------------------------------------------------------


def _load_script() -> Any:
    module_spec = importlib.util.spec_from_file_location("build_sft_script", SCRIPT)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_build_writes_dataset_and_card(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [row(f"v-{c}-{i}", f"{c} 第{i}句", category=c) for c in CATEGORIES for i in range(3)]
    rows.append(row("v-dup", "first_turn 第0句"))  # same input + context as v-first_turn-0
    rows.append(row("v-leak", "今晚十点来两把大乱斗，一个小时"))  # holdout-025 verbatim
    with (raw / "accepted.jsonl").open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    run = {"runs": [{"teacher": {"provider": "openai", "model": "m"}}], "status": {"accepted": 29}}
    (raw / "run.json").write_text(json.dumps(run), encoding="utf-8")

    out = tmp_path / "sft"
    summary = _load_script().build("v9.9", raw, out, embedder=None)
    assert summary["raw"] == 29 and summary["duplicates"] == 1 and summary["eval_leaks"] == 1
    assert summary["kept"] == 27 and summary["train"] + summary["val"] == 27
    assert summary["missing_categories"] == []

    lines = (out / TRAIN).read_text("utf-8").splitlines() + (out / VAL).read_text(
        "utf-8"
    ).splitlines()
    assert len(lines) == 27
    assert all(len(json.loads(line)["messages"]) == 3 for line in lines)
    assert "rift_sft_v9_9_train" in json.loads((out / DATASET_INFO).read_text("utf-8"))
    removed = [json.loads(line) for line in (out / "removed.jsonl").read_text("utf-8").splitlines()]
    assert {r["reason"] for r in removed} == {"duplicate", "eval_leak"}

    card = load_card(out)
    assert card.version == "v9.9" and card.num_samples == 27
    assert card.prompt_sha256 == prompt_hash()
    assert card.teacher == {"provider": "openai", "model": "m"}
    assert card.extra["dedup"]["eval_leaks"][0]["eval_id"] == "holdout-025"
    assert len(card.extra["teacher_prompt_sha256"]) == 64
    assert "audit" in card.extra


@pytest.mark.unit
def test_script_without_embedding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [row(f"v-{c}", f"{c} 一句", category=c) for c in CATEGORIES]
    (raw / "accepted.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    script = _load_script()
    out = tmp_path / "out"
    code = script.main(
        ["--version", "v9.9", "--raw", str(raw), "--out", str(out), "--no-embedding"]
    )
    assert code == 0
    assert load_card(out).teacher is None
    assert script.embedding_config().model == "BAAI/bge-small-zh-v1.5"
