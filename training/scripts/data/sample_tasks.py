"""Sample the generation tasks of a scenario spec and print their distribution (DEV_SPEC I2).

Usage::

    uv run python training/scripts/data/sample_tasks.py --spec v0.1
    uv run python training/scripts/data/sample_tasks.py --spec v0.1 \
        --out training/data/raw/tasks_v0.1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rift_training.data.specs import (
    QUOTA_TOLERANCE,
    load_spec,
    quota_errors,
    sample_tasks,
    summarize,
    task_problems,
    write_tasks,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", default="v0.1", help="spec version or path to a YAML file")
    parser.add_argument("--total", type=int, help="override the spec's total")
    parser.add_argument("--seed", type=int, help="override the spec's seed")
    parser.add_argument("--out", type=Path, help="write the tasks as JSONL")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    tasks = sample_tasks(spec, total=args.total, seed=args.seed)
    print(json.dumps(summarize(tasks), ensure_ascii=False, indent=2))

    errors = quota_errors(spec, tasks)
    print("\nquota error per category:")
    for name, err in errors.items():
        print(f"  {name:<14} {err:6.2%}")
    problems = [f"{t.id}: {p}" for t in tasks for p in task_problems(t)]
    for line in problems:
        print(f"PROBLEM {line}", file=sys.stderr)

    if args.out:
        write_tasks(tasks, args.out)
        print(f"\nwrote {len(tasks)} tasks to {args.out}")
    worst = max(errors.values())
    return 1 if problems or worst >= QUOTA_TOLERANCE else 0


if __name__ == "__main__":
    sys.exit(main())
