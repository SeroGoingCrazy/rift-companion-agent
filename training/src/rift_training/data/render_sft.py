"""Render samples as a LLaMA-Factory SFT dataset (OpenAI-style ``messages``).

Each line is ``{"id": ..., "messages": [system, user, assistant]}`` produced by
``contract.render_conversation``, i.e. the exact online prompt plus the canonical target.
``dataset_info.json`` registers ``train`` / ``val`` in the sharegpt format with OpenAI tags.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rift_training.contract import render_conversation
from rift_training.evaluation.dataset import SlotSample

TRAIN = "train.jsonl"
VAL = "val.jsonl"
DATASET_INFO = "dataset_info.json"


def to_record(sample: SlotSample) -> dict[str, Any]:
    return {"id": sample.id, "messages": render_conversation(sample)}


def split(
    samples: Sequence[SlotSample], *, val_ratio: float = 0.05, seed: int = 0
) -> tuple[list[SlotSample], list[SlotSample]]:
    """Per-category random split, so every category is represented in ``val``."""
    rng = random.Random(seed)
    by_category: dict[str, list[SlotSample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)
    train: list[SlotSample] = []
    val: list[SlotSample] = []
    for group in by_category.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_ratio) if val_ratio > 0 else 0
        if val_ratio > 0 and len(shuffled) >= 2:
            n_val = max(n_val, 1)
        val += shuffled[:n_val]
        train += shuffled[n_val:]
    return sorted(train, key=lambda s: s.id), sorted(val, key=lambda s: s.id)


def dataset_info(prefix: str) -> dict[str, Any]:
    entry = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    return {
        f"{prefix}_train": {"file_name": TRAIN, **entry},
        f"{prefix}_val": {"file_name": VAL, **entry},
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sft(
    out_dir: Path, train: Sequence[SlotSample], val: Sequence[SlotSample], *, prefix: str
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / TRAIN, [to_record(s) for s in train])
    _write_jsonl(out_dir / VAL, [to_record(s) for s in val])
    with (out_dir / DATASET_INFO).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dataset_info(prefix), ensure_ascii=False, indent=2) + "\n")
    return [out_dir / TRAIN, out_dir / VAL, out_dir / DATASET_INFO]
