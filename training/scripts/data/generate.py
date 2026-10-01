"""Generate SFT samples with a Teacher (DEV_SPEC I3).

Trial run on 20 tasks spread over all categories, then the full spec::

    uv run --env-file .env python training/scripts/data/generate.py --spec v0.1 --limit 20
    uv run --env-file .env python training/scripts/data/generate.py --spec v0.1

Re-running resumes: accepted tasks are skipped, and so are rejected ones unless
``--retry-rejected``. ``--teacher mock`` dry-runs the pipeline without an API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rift_domain.config import load_domain_config
from rift_training.data.generate import generate, stratified, summarize_dir
from rift_training.data.specs import DOMAIN_PATH, REPO_ROOT, load_spec, sample_tasks
from rift_training.inference.factory import load_teacher_config, make_teacher
from rift_training.inference.teacher_openai import describe

RAW_DIR = REPO_ROOT / "training" / "data" / "raw"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SFT samples with a Teacher.")
    parser.add_argument("--spec", default="v0.1")
    parser.add_argument("--teacher", default="openai", help="openai | mock | path to a YAML")
    parser.add_argument("--model", help="override the Teacher model")
    parser.add_argument("--limit", type=int, help="only N tasks, spread over categories")
    parser.add_argument("--out", type=Path, help="default: training/data/raw/<spec>[-<teacher>]")
    parser.add_argument("--attempts", type=int, default=2, help="tries per task incl. feedback")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--retry-rejected", action="store_true")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    config = load_teacher_config(args.teacher, model=args.model)
    tasks = sample_tasks(spec)
    if args.limit:
        tasks = stratified(tasks, args.limit)
    suffix = "" if config.provider == "openai" else f"-{config.provider}"
    out_dir = args.out or RAW_DIR / f"{spec.version}{suffix}"

    teacher = make_teacher(config)

    def progress(task_id: str, status: str) -> None:
        print(f"{task_id}  {status}", flush=True)

    async def run() -> None:
        try:
            stats = await generate(
                tasks,
                teacher,
                load_domain_config(DOMAIN_PATH),
                out_dir,
                teacher_info=describe(config),
                attempts=args.attempts,
                concurrency=args.concurrency or config.concurrency,
                max_retries=config.max_retries,
                retry_rejected=args.retry_rejected,
                progress=progress,
            )
        finally:
            await teacher.aclose()
        print(
            f"\nthis run: {stats.accepted} accepted ({stats.accepted_first_try} first try), "
            f"{stats.rejected_validation} rejected, {stats.rejected_api} api errors, "
            f"{stats.skipped} skipped; tokens {stats.usage}"
        )

    asyncio.run(run())
    status = summarize_dir(out_dir)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
