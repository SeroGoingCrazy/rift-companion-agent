"""Build DPO preference sets (DEV_SPEC I8) from the SFT train split.

    uv run python training/scripts/data/build_dpo.py --mode rule
    uv run python training/scripts/data/build_dpo.py --mode onpolicy \
        --samples training/data/raw/onpolicy_r003_0.6b.jsonl

Writes ``training/data/processed/dpo/<mode>_v1/`` (pairs.jsonl, dataset_info.json,
data_card.json). Pairs come only from the SFT train split, never from val or the eval sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from rift_common.settings import EmbeddingConfig
from rift_training.data.dpo_onpolicy import onpolicy_pairs, read_samples
from rift_training.data.dpo_rule import WEIGHTS, build_rule_pairs
from rift_training.data.preference import train_samples, write_pairs
from rift_training.data.specs import REPO_ROOT
from rift_training.evaluation.preferences import make_style_matcher

SFT_DIR = REPO_ROOT / "training" / "data" / "processed" / "sft"
DPO_DIR = REPO_ROOT / "training" / "data" / "processed" / "dpo"
RAW_DIR = REPO_ROOT / "training" / "data" / "raw"
#: The raw generation dirs behind each SFT version.
SFT_SOURCES = {"v0.4": ("v0.2", "v0.3c", "v0.4c")}
SETTINGS = REPO_ROOT / "config" / "settings.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DPO preference pairs.")
    parser.add_argument("--mode", choices=("rule", "onpolicy"), required=True)
    parser.add_argument("--sft", default="v0.4", help="SFT version whose train split is used")
    parser.add_argument("--samples", type=Path, help="on-policy samples (sample_onpolicy.py)")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--per-sample", type=int, default=1, help="rule pairs per sample")
    parser.add_argument("--max-per-sample", type=int, default=2, help="on-policy pairs per sample")
    args = parser.parse_args(argv)

    raw_dirs = [RAW_DIR / d for d in SFT_SOURCES[args.sft]]
    samples = train_samples(SFT_DIR / args.sft, raw_dirs)
    name = f"rift_dpo_{args.mode}_{args.version}"
    out = args.out or DPO_DIR / f"{args.mode}_{args.version}"
    source: dict[str, object] = {"sft": args.sft, "train_samples": len(samples)}

    if args.mode == "rule":
        pairs = build_rule_pairs(samples, per_sample=args.per_sample)
        source.update(weights=WEIGHTS, per_sample=args.per_sample)
    else:
        if args.samples is None:
            parser.error("--mode onpolicy needs --samples")
        embedding = yaml.safe_load(SETTINGS.read_text("utf-8"))["embedding"]
        style = make_style_matcher(EmbeddingConfig.model_validate(embedding))
        pairs, stats = onpolicy_pairs(
            samples, read_samples(args.samples), style, max_per_sample=args.max_per_sample
        )
        source.update(samples_file=args.samples.name, scoring=stats.as_dict())

    card = write_pairs(out, pairs, name=name, source=source)
    print(json.dumps(card, ensure_ascii=False, indent=2))
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
