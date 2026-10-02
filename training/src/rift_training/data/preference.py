"""Preference pairs for DPO (DEV_SPEC I8): shared model, LLaMA-Factory rendering, data cards.

A pair is one training sample (its online prompt), the canonical answer as ``chosen`` and a
wrong answer as ``rejected``. Both DPO sets -- rule perturbations and on-policy samples --
are written in LLaMA-Factory's sharegpt *ranking* format with OpenAI-style tags::

    {"id": ..., "kind": ..., "messages": [system, user],
     "chosen": {"role": "assistant", "content": ...},
     "rejected": {"role": "assistant", "content": ...}}

Pairs are only built from the SFT *train* split, so the SFT val split stays unseen.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rift_training.contract import prompt_hash, render_prompt, render_target
from rift_training.data.render_sft import TRAIN
from rift_training.evaluation.dataset import SlotSample, load_samples

PAIRS = "pairs.jsonl"
DATASET_INFO = "dataset_info.json"
CARD = "data_card.json"


@dataclass(frozen=True)
class PreferencePair:
    sample: SlotSample
    rejected: str
    #: Why the rejected answer is wrong ("any_to_null", "drop_key", "onpolicy:extra_key" ...).
    kind: str

    @property
    def chosen(self) -> str:
        return render_target(self.sample.expected)

    def record(self) -> dict[str, Any]:
        return {
            "id": self.sample.id,
            "kind": self.kind,
            "messages": render_prompt(self.sample),
            "chosen": {"role": "assistant", "content": self.chosen},
            "rejected": {"role": "assistant", "content": self.rejected},
        }


def train_samples(sft_dir: Path, raw_dirs: Sequence[Path]) -> list[SlotSample]:
    """The SFT train split as full samples (ids from ``train.jsonl``, bodies from raw dirs)."""
    ids = [
        json.loads(line)["id"]
        for line in (sft_dir / TRAIN).read_text("utf-8").splitlines()
        if line.strip()
    ]
    pool = {s.id: s for d in raw_dirs for s in load_samples(d / "accepted.jsonl")}
    missing = [i for i in ids if i not in pool]
    if missing:
        raise ValueError(f"{len(missing)} train ids not found in the raw dirs, e.g. {missing[:3]}")
    return [pool[i] for i in ids]


def dataset_info(name: str) -> dict[str, Any]:
    return {
        name: {
            "file_name": PAIRS,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {"messages": "messages", "chosen": "chosen", "rejected": "rejected"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }


def write_pairs(
    out_dir: Path,
    pairs: Sequence[PreferencePair],
    *,
    name: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Write pairs, dataset_info and a data card (counts per kind and category)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / PAIRS).open("w", encoding="utf-8", newline="\n") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair.record(), ensure_ascii=False) + "\n")
    with (out_dir / DATASET_INFO).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(dataset_info(name), ensure_ascii=False, indent=2) + "\n")
    card = {
        "name": name,
        "num_pairs": len(pairs),
        "num_samples": len({p.sample.id for p in pairs}),
        "prompt_sha256": prompt_hash(),
        "kinds": dict(Counter(p.kind for p in pairs).most_common()),
        "categories": dict(Counter(p.sample.category for p in pairs).most_common()),
        "source": source,
    }
    with (out_dir / CARD).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(card, ensure_ascii=False, indent=2) + "\n")
    return card
