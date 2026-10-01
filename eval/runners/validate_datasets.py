"""Validate the L2 slot datasets (schema, consistency, categories, checksums).

Checks every sample against the C2 ``SlotExtraction`` schema and the ``BookingState``
model, the per-sample consistency rules, at least 3 samples per category in each set,
no user input shared between main and holdout, and the frozen sha256 sums in
``eval/datasets/CHECKSUMS``. Prints the category distribution (Markdown) on success.

Usage::

    uv run python eval/runners/validate_datasets.py
    uv run python eval/runners/validate_datasets.py --write-checksums   # freeze after review
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from rift_training.evaluation.dataset import (
    DatasetError,
    SlotSample,
    check_datasets,
    checksum_problems,
    distribution_table,
    load_samples,
    write_checksums,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"
DATASETS = {"main": "slot_main.jsonl", "holdout": "slot_holdout.jsonl"}
CHECKSUMS = "CHECKSUMS"


def validate(directory: Path, *, write: bool = False) -> tuple[list[str], str]:
    """Return (problems, distribution table) for the datasets in ``directory``."""
    problems: list[str] = []
    loaded: dict[str, Sequence[SlotSample]] = {}
    for name, filename in DATASETS.items():
        try:
            loaded[name] = load_samples(directory / filename)
        except DatasetError as exc:
            problems.extend(str(exc).splitlines())
    if problems:
        return problems, ""
    problems.extend(check_datasets(loaded))
    files = [directory / f for f in DATASETS.values()]
    if write and not problems:
        write_checksums(directory / CHECKSUMS, files)
    problems.extend(checksum_problems(directory / CHECKSUMS, files))
    return problems, distribution_table(loaded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DATASETS_DIR)
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="record the current sha256 sums (only after the sets were reviewed)",
    )
    args = parser.parse_args(argv)

    problems, table = validate(args.dir, write=args.write_checksums)
    if table:
        print(table)
        print()
    if problems:
        for p in problems:
            print(f"[fail] {p}")
        print(f"\n{len(problems)} problem(s)")
        return 1
    print("datasets OK" + (" (checksums written)" if args.write_checksums else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
