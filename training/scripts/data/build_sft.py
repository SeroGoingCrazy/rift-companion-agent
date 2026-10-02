"""Build a versioned SFT dataset from Teacher output (DEV_SPEC I4).

raw ``accepted.jsonl`` -> drop in-set duplicates -> drop near-duplicates of the L2 eval sets
-> per-category train / val split -> LLaMA-Factory files -> coverage audit -> data card::

    uv run python training/scripts/data/build_sft.py --version v0.1

Writes ``training/data/processed/sft/<version>/`` (train.jsonl, val.jsonl,
dataset_info.json, removed.jsonl, data_card.json, DATA_CARD.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from rift_common.embedding import BaseEmbedding, EmbeddingFactory
from rift_common.settings import EmbeddingConfig
from rift_domain.config import load_domain_config
from rift_training.data.audit import audit, missing_categories
from rift_training.data.card import build_card, write_card
from rift_training.data.dedup import (
    DEFAULT_THRESHOLD,
    dedup_within,
    find_leaks,
    nearest_eval,
    similarity_histogram,
)
from rift_training.data.generate import ACCEPTED, RUN
from rift_training.data.prompting import TEACHER_SYSTEM
from rift_training.data.render_sft import split, write_sft
from rift_training.data.specs import DOMAIN_PATH, REPO_ROOT
from rift_training.evaluation.dataset import load_samples

RAW_DIR = REPO_ROOT / "training" / "data" / "raw"
SFT_DIR = REPO_ROOT / "training" / "data" / "processed" / "sft"
EVAL_SETS = [REPO_ROOT / "eval" / "datasets" / f"slot_{n}.jsonl" for n in ("main", "holdout")]
SETTINGS = REPO_ROOT / "config" / "settings.yaml"
HIST_EDGES = (0.0, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95)


def embedding_config() -> EmbeddingConfig:
    """``settings.embedding`` read without expanding the rest of the settings (no keys)."""
    data = yaml.safe_load(SETTINGS.read_text("utf-8"))
    return EmbeddingConfig.model_validate(data["embedding"])


def teacher_from_run(raw_dir: Path) -> dict[str, str] | None:
    path = raw_dir / RUN
    if not path.exists():
        return None
    runs = json.loads(path.read_text("utf-8")).get("runs") or []
    teachers = {json.dumps(r["teacher"], sort_keys=True) for r in runs}
    if len(teachers) != 1:
        return {"provider": "mixed", "runs": str(len(runs))}
    teacher: dict[str, str] = runs[-1]["teacher"]
    return teacher


def build(
    version: str,
    raw_dir: Path,
    out_dir: Path,
    *,
    embedder: BaseEmbedding | None,
    threshold: float = DEFAULT_THRESHOLD,
    val_ratio: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    domain = load_domain_config(DOMAIN_PATH)
    raw = load_samples(raw_dir / ACCEPTED)
    eval_samples = [s for path in EVAL_SETS for s in load_samples(path)]

    unique, duplicates = dedup_within(sorted(raw, key=lambda s: s.id))
    nearest = nearest_eval(unique, eval_samples, embedder)
    leaks = find_leaks(unique, eval_samples, embedder, threshold=threshold)
    leaked = {leak.sample_id for leak in leaks}
    kept = [s for s in unique if s.id not in leaked]

    train, val = split(kept, val_ratio=val_ratio, seed=seed)
    write_sft(out_dir, train, val, prefix=f"rift_sft_{version.replace('.', '_')}")
    with (out_dir / "removed.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for dup, of in duplicates:
            fh.write(json.dumps({"id": dup, "reason": "duplicate", "of": of}) + "\n")
        for leak in leaks:
            fh.write(
                json.dumps(
                    {"id": leak.sample_id, "reason": "eval_leak", "leak": leak.as_dict()},
                    ensure_ascii=False,
                )
                + "\n"
            )

    report = audit(kept, domain)
    generation = (
        json.loads((raw_dir / RUN).read_text("utf-8"))["status"] if (raw_dir / RUN).exists() else {}
    )
    card = build_card(
        kept,
        name="sft",
        version=version,
        teacher=teacher_from_run(raw_dir),
        spec=f"specs_{version}",
        notes=[
            f"raw {len(raw)} -> {len(unique)} after in-set dedup -> {len(kept)} after removing "
            f"{len(leaks)} near-duplicates of the L2 eval sets",
            f"split: train {len(train)} / val {len(val)} (per category, seed {seed})",
        ],
        extra={
            "dedup": {
                "embedding": embedder.config.model if embedder else None,
                "threshold": threshold,
                "duplicates": len(duplicates),
                "eval_leaks": [leak.as_dict() for leak in leaks],
                "nearest_eval_similarity": similarity_histogram(
                    [sim for _, _, sim, _ in nearest], HIST_EDGES
                ),
            },
            "generation": generation,
            "teacher_prompt_sha256": hashlib.sha256(TEACHER_SYSTEM.read_bytes()).hexdigest(),
            "audit": report,
        },
    )
    write_card(card, out_dir)
    return {
        "raw": len(raw),
        "duplicates": len(duplicates),
        "eval_leaks": len(leaks),
        "kept": len(kept),
        "train": len(train),
        "val": len(val),
        "missing_categories": missing_categories(report),
        "unseen_aliases": report["aliases"]["unseen"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a versioned SFT dataset.")
    parser.add_argument("--version", default="v0.1")
    parser.add_argument("--raw", type=Path, help="default: training/data/raw/<version>")
    parser.add_argument("--out", type=Path, help="default: training/data/processed/sft/<version>")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--no-embedding", action="store_true", help="exact-match leak check only (no bge model)"
    )
    args = parser.parse_args(argv)

    embedder = None if args.no_embedding else EmbeddingFactory.create(embedding_config())
    summary = build(
        args.version,
        args.raw or RAW_DIR / args.version,
        args.out or SFT_DIR / args.version,
        embedder=embedder,
        threshold=args.threshold,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["missing_categories"] else 0


if __name__ == "__main__":
    sys.exit(main())
