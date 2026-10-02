"""Generate SFT samples with a Teacher (DEV_SPEC I3).

For every task: build the request (``prompting``), call the Teacher (transient failures are
retried with exponential backoff), validate the answer (``validate``); an answer that fails
validation is sent back once more with the problems listed. Results are appended line by
line, so an interrupted run resumes where it stopped:

``<out_dir>/accepted.jsonl``            L2-format samples (``load_samples`` reads them)
``<out_dir>/rejected/rejected.jsonl``   task id, kind (validation / api), problems, raw answer
``<out_dir>/tasks.jsonl``               the task list of this spec
``<out_dir>/run.json``                  Teacher, counts, pass rate, token usage
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rift_domain.config import DomainConfig
from rift_training.data.prompting import build_request, retry_request
from rift_training.data.specs import GenerationTask, write_tasks
from rift_training.data.validate import Outcome, sample_row, validate_answer
from rift_training.inference.base import Teacher, TeacherError, TeacherRequest, TeacherResponse

ACCEPTED = "accepted.jsonl"
REJECTED = "rejected/rejected.jsonl"
TASKS = "tasks.jsonl"
RUN = "run.json"

Sleep = Callable[[float], Awaitable[None]]


@dataclass
class RunStats:
    started_at: str = ""
    finished_at: str = ""
    teacher: dict[str, str] = field(default_factory=dict)
    requested: int = 0
    skipped: int = 0
    accepted: int = 0
    accepted_first_try: int = 0
    rejected_validation: int = 0
    rejected_api: int = 0
    teacher_calls: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    problems: dict[str, int] = field(default_factory=dict)
    #: The fatal error that ended the run early ("" when it ran to the end).
    stopped: str = ""

    def add_usage(self, usage: dict[str, int]) -> None:
        for key, value in usage.items():
            self.usage[key] = self.usage.get(key, 0) + value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def done_ids(out_dir: Path) -> tuple[set[str], set[str]]:
    """(accepted ids, ids the validator rejected and that were never accepted).

    Tasks that only failed at the API (rate limits, no credits) are not "done": a rerun
    tries them again without ``--retry-rejected``.
    """
    accepted = {row["id"] for row in _read_jsonl(out_dir / ACCEPTED)}
    latest = {row["task_id"]: row["kind"] for row in _read_jsonl(out_dir / REJECTED)}
    rejected = {tid for tid, kind in latest.items() if kind == "validation"} - accepted
    return accepted, rejected


def problem_kind(problem: str) -> str:
    """Coarse bucket for the rejection table ("start_time_expr 'x' does not parse" -> ...)."""
    head = problem.split(":")[0] if ":" in problem[:40] else problem
    words = [w for w in head.split() if not w.startswith(("'", '"'))]
    return " ".join(words[:4])


async def call_teacher(
    teacher: Teacher,
    request: TeacherRequest,
    *,
    max_retries: int,
    sleep: Sleep = asyncio.sleep,
    backoff_s: float = 2.0,
) -> TeacherResponse:
    for attempt in range(max_retries + 1):
        try:
            return await teacher.generate(request)
        except TeacherError as exc:
            if not exc.retryable or attempt == max_retries:
                raise
            await sleep(backoff_s * 2**attempt)
    raise AssertionError("unreachable")


async def generate_one(
    task: GenerationTask,
    teacher: Teacher,
    domain: DomainConfig,
    *,
    attempts: int,
    max_retries: int,
    stats: RunStats,
    sleep: Sleep = asyncio.sleep,
) -> tuple[Outcome, str, int]:
    """Returns the last outcome, the last raw answer and the number of attempts used."""
    request = build_request(task, domain)
    outcome = Outcome(task.id, None, ["not attempted"])
    raw = ""
    for attempt in range(1, attempts + 1):
        response = await call_teacher(teacher, request, max_retries=max_retries, sleep=sleep)
        stats.teacher_calls += 1
        stats.add_usage(response.usage)
        raw = response.text
        outcome = validate_answer(task, raw, domain)
        if outcome.ok:
            return outcome, raw, attempt
        request = retry_request(request, raw, outcome.problems)
    return outcome, raw, attempts


async def generate(
    tasks: Sequence[GenerationTask],
    teacher: Teacher,
    domain: DomainConfig,
    out_dir: Path,
    *,
    teacher_info: dict[str, str] | None = None,
    attempts: int = 2,
    concurrency: int = 8,
    max_retries: int = 4,
    retry_rejected: bool = False,
    sleep: Sleep = asyncio.sleep,
    progress: Callable[[str, str], None] | None = None,
) -> RunStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tasks(tasks, out_dir / TASKS)
    accepted_ids, rejected_ids = done_ids(out_dir)
    skip = accepted_ids | (set() if retry_rejected else rejected_ids)
    pending = [t for t in tasks if t.id not in skip]
    stats = RunStats(
        started_at=datetime.now().isoformat(timespec="seconds"),
        teacher=teacher_info or {"provider": teacher.name, "model": teacher.model},
        requested=len(tasks),
        skipped=len(tasks) - len(pending),
    )
    problems: Counter[str] = Counter()
    gate = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    stop: list[str] = []  # set by a fatal Teacher error; queued tasks are left for a rerun

    async def run(task: GenerationTask) -> None:
        async with gate:
            if stop:
                return
            started = time.perf_counter()
            try:
                outcome, raw, used = await generate_one(
                    task,
                    teacher,
                    domain,
                    attempts=attempts,
                    max_retries=max_retries,
                    stats=stats,
                    sleep=sleep,
                )
            except TeacherError as exc:
                if exc.fatal:
                    stop.append(str(exc))
                    if progress:
                        progress(task.id, f"fatal, stopping the run: {exc}")
                    return
                async with lock:
                    stats.rejected_api += 1
                    _append(out_dir / REJECTED, _rejection(task, "api", [str(exc)], "", 0))
                if progress:
                    progress(task.id, f"api error: {exc}")
                return
            async with lock:
                if outcome.ok and outcome.sample is not None:
                    stats.accepted += 1
                    stats.accepted_first_try += used == 1
                    _append(out_dir / ACCEPTED, sample_row(outcome.sample))
                    status = f"ok ({used} tries, {time.perf_counter() - started:.1f}s)"
                else:
                    stats.rejected_validation += 1
                    problems.update(problem_kind(p) for p in outcome.problems)
                    _append(
                        out_dir / REJECTED,
                        _rejection(task, "validation", outcome.problems, raw, used),
                    )
                    status = "rejected: " + "; ".join(outcome.problems[:2])
            if progress:
                progress(task.id, status)

    await asyncio.gather(*(run(t) for t in pending))
    stats.stopped = stop[0] if stop else ""
    stats.problems = dict(problems.most_common())
    stats.finished_at = datetime.now().isoformat(timespec="seconds")
    write_manifest(out_dir, stats)
    return stats


def _rejection(
    task: GenerationTask, kind: str, problems: list[str], raw: str, attempts: int
) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "category": task.category,
        "variant": task.variant,
        "kind": kind,
        "attempts": attempts,
        "problems": problems,
        "raw": raw,
    }


def summarize_dir(out_dir: Path) -> dict[str, Any]:
    """Final status per task id over all runs into this directory."""
    accepted = _read_jsonl(out_dir / ACCEPTED)
    accepted_ids = {row["id"] for row in accepted}
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(out_dir / REJECTED):
        if row["task_id"] not in accepted_ids:
            latest[row["task_id"]] = row
    rejected_validation = sum(1 for r in latest.values() if r["kind"] == "validation")
    judged = len(accepted_ids) + rejected_validation
    return {
        "accepted": len(accepted_ids),
        "rejected_validation": rejected_validation,
        "rejected_api": len(latest) - rejected_validation,
        "pass_rate": round(len(accepted_ids) / judged, 4) if judged else None,
        "accepted_by_category": dict(Counter(row["category"] for row in accepted)),
        "rejected_by_variant": dict(
            Counter(f"{r['category']}.{r['variant']}" for r in latest.values())
        ),
    }


def write_manifest(out_dir: Path, stats: RunStats) -> Path:
    path = out_dir / RUN
    history = json.loads(path.read_text("utf-8")).get("runs", []) if path.exists() else []
    manifest = {"runs": [*history, asdict(stats)], "status": summarize_dir(out_dir)}
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


def stratified(tasks: Sequence[GenerationTask], limit: int) -> list[GenerationTask]:
    """``limit`` tasks spread over categories and variants (round-robin), for trial runs."""
    groups: dict[str, dict[str, list[GenerationTask]]] = {}
    for task in tasks:
        groups.setdefault(task.category, {}).setdefault(task.variant, []).append(task)
    # within a category alternate variants, across categories alternate categories
    queues = [_round_robin(list(variants.values())) for variants in groups.values()]
    return sorted(_round_robin(queues)[:limit], key=lambda t: t.id)


def _round_robin[T](lists: Sequence[Sequence[T]]) -> list[T]:
    out: list[T] = []
    for depth in range(max((len(x) for x in lists), default=0)):
        out.extend(x[depth] for x in lists if depth < len(x))
    return out
