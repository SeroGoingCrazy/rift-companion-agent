"""L2 slot-extraction evaluation: run an extractor over a dataset, score, report.

Each sample is turned into the same ``ExtractionContext`` the agent builds online, so the
prompt is byte-identical to production. ``--extractor``:

- ``llm``: DeepSeek, one shot (no retry) -- measures the model itself;
- ``local``: llama-server (``llm.local_slot``), one shot;
- ``routed``: the M3 routing -- local primary, retry on invalid output, DeepSeek fallback.

Scoring (``rift_training.evaluation``): protocol layer (JSON, schema, enums) + task layer
(intent, confirmation, every delta key either side emitted; times compared after parsing,
``style_preference`` by local bge embedding). A sample passes when the protocol holds and
the task score is >= 0.95. Writes a JSON report (every sample) and a Markdown summary.

Usage::

    uv run --env-file .env python eval/runners/run_slot_eval.py --extractor llm --dataset main
    uv run --env-file .env python eval/runners/run_slot_eval.py --extractor llm \
        --dataset holdout --out eval/reports/baseline_llm_slot_holdout
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rift_agent.extractors.base import ExtractionResult, SlotExtractor
from rift_training.contract import context_from_sample
from rift_training.evaluation.dataset import (
    CATEGORIES,
    SlotSample,
    checksum_problems,
    load_samples,
    sha256_file,
)
from rift_training.evaluation.preferences import (
    DEFAULT_STYLE_THRESHOLD,
    StyleMatcher,
    make_style_matcher,
)
from rift_training.evaluation.task import PASS_THRESHOLD, score_output

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"
DATASETS = {"main": "slot_main.jsonl", "holdout": "slot_holdout.jsonl"}
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"
EXTRACTORS = ("llm", "local", "routed")


@dataclass
class ItemResult:
    id: str
    category: str
    user_input: str
    expected: dict[str, Any]
    raw: str
    passed: bool
    protocol_ok: bool
    protocol_error: str | None
    protocol_messages: list[str]
    task_score: float
    field_errors: list[dict[str, Any]]
    latency_ms: float
    source: str = ""
    retried: bool = False
    fell_back: bool = False
    attempts: list[str] = field(default_factory=list)
    extractor_errors: list[str] = field(default_factory=list)


# --- running -------------------------------------------------------------------------------


#: The context the agent's ``extract_slots`` node would build for a sample (shared with training).
build_context = context_from_sample


async def run_extractor(
    samples: Sequence[SlotSample],
    extractor: SlotExtractor,
    *,
    concurrency: int = 4,
    progress: Callable[[int, SlotSample, ExtractionResult], None] | None = None,
) -> list[tuple[ExtractionResult, float]]:
    """(result, wall-clock ms) per sample, in dataset order."""
    gate = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(sample: SlotSample) -> tuple[ExtractionResult, float]:
        nonlocal done
        async with gate:
            t0 = time.perf_counter()
            try:
                result = await extractor.extract(build_context(sample))
            except Exception as exc:  # a crashing extractor scores as an empty output
                result = ExtractionResult.failed((f"{type(exc).__name__}: {exc}",))
            elapsed = (time.perf_counter() - t0) * 1000
        done += 1
        if progress is not None:
            progress(done, sample, result)
        return result, elapsed

    return list(await asyncio.gather(*(one(s) for s in samples)))


def score_item(
    sample: SlotSample,
    result: ExtractionResult,
    latency_ms: float,
    *,
    style: StyleMatcher,
    threshold: float = PASS_THRESHOLD,
) -> ItemResult:
    score = score_output(
        result.raw,
        sample.extraction(),
        state=sample.state(),
        now=sample.now,
        style=style,
        threshold=threshold,
    )
    errors = [
        {
            "field": f.field,
            "error": f.error,
            "expected": f.expected,
            "predicted": f.predicted,
            "detail": f.detail,
        }
        for f in (score.task.errors if score.task else ())
    ]
    return ItemResult(
        id=sample.id,
        category=sample.category,
        user_input=sample.user_input,
        expected=sample.expected,
        raw=result.raw,
        passed=score.passed,
        protocol_ok=score.protocol.ok,
        protocol_error=score.protocol.error,
        protocol_messages=list(score.protocol.messages),
        task_score=round(score.task_score, 4),
        field_errors=errors,
        latency_ms=round(latency_ms, 1),
        source=result.source,
        retried=result.retried,
        fell_back=result.fell_back,
        attempts=list(result.attempts),
        extractor_errors=list(result.errors),
    )


async def evaluate(
    samples: Sequence[SlotSample],
    extractor: SlotExtractor,
    *,
    style: StyleMatcher,
    concurrency: int = 4,
    threshold: float = PASS_THRESHOLD,
    log: Callable[[str], None] = print,
) -> list[ItemResult]:
    def progress(n: int, sample: SlotSample, result: ExtractionResult) -> None:
        mark = "ok " if result.valid else "ERR"
        log(f"  [{n:>3}/{len(samples)}] {mark} {sample.id}")

    runs = await run_extractor(samples, extractor, concurrency=concurrency, progress=progress)
    return [
        score_item(s, r, ms, style=style, threshold=threshold)
        for s, (r, ms) in zip(samples, runs, strict=True)
    ]


# --- summary -------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _rate(n: int, total: int) -> float:
    return round(n / total, 4) if total else 0.0


def summarize(items: Sequence[ItemResult]) -> dict[str, Any]:
    n = len(items)
    latencies = [i.latency_ms for i in items]
    by_category: dict[str, dict[str, Any]] = {}
    for key in CATEGORIES:
        group = [i for i in items if i.category == key]
        if group:
            by_category[key] = {
                "n": len(group),
                "passed": sum(i.passed for i in group),
                "pass_rate": _rate(sum(i.passed for i in group), len(group)),
            }
    field_errors: dict[str, Counter[str]] = defaultdict(Counter)
    for i in items:
        for e in i.field_errors:
            field_errors[e["field"]][e["error"]] += 1
    scored = [i for i in items if i.protocol_ok]
    return {
        "n": n,
        "passed": sum(i.passed for i in items),
        "pass_rate": _rate(sum(i.passed for i in items), n),
        "protocol_pass_rate": _rate(len(scored), n),
        "mean_task_score": round(sum(i.task_score for i in scored) / len(scored), 4)
        if scored
        else 0.0,
        "protocol_errors": dict(Counter(i.protocol_error for i in items if i.protocol_error)),
        "field_errors": {
            f: dict(c) for f, c in sorted(field_errors.items(), key=lambda kv: -sum(kv[1].values()))
        },
        "by_category": by_category,
        "retried": sum(i.retried for i in items),
        "fell_back": sum(i.fell_back for i in items),
        "latency_ms": {
            "mean": round(sum(latencies) / n, 1) if n else 0.0,
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "max": max(latencies, default=0.0),
        },
    }


# --- Markdown ------------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_markdown(report: dict[str, Any], *, max_examples: int = 15) -> str:
    meta, s = report["meta"], report["summary"]
    lat = s["latency_ms"]
    lines = [
        f"# L2 槽位抽取评测：{meta['extractor']} / {meta['dataset']}",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 抽取器 | `{meta['extractor']}`（{meta['model'] or '-'}） |",
        f"| 数据集 | `{meta['dataset_file']}`（{s['n']} 条，sha256 `{meta['sha256'][:12]}`"
        f"{'，**checksum 不符**' if meta['checksum_problems'] else ''}） |",
        f"| 时间 | {meta['started_at']} |",
        f"| 通过标准 | 协议通过且任务分 ≥ {meta['pass_threshold']} |",
        f"| 风格匹配 | {meta['style_matcher']}（阈值 {meta['style_threshold']}） |",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| **通过率** | **{_pct(s['pass_rate'])}**（{s['passed']}/{s['n']}） |",
        f"| 协议通过率 | {_pct(s['protocol_pass_rate'])} |",
        f"| 平均任务分（协议通过样本） | {s['mean_task_score']:.3f} |",
        f"| 重试 / 降级次数 | {s['retried']} / {s['fell_back']} |",
        f"| 延迟 P50 / P90 / P95 / max（ms） | {lat['p50']:.0f} / {lat['p90']:.0f} / "
        f"{lat['p95']:.0f} / {lat['max']:.0f} |",
        "",
        "## 分类别通过率",
        "",
        "| 类别 | 通过 | 通过率 |",
        "|---|---:|---:|",
    ]
    for key, c in s["by_category"].items():
        lines.append(
            f"| {CATEGORIES[key]} `{key}` | {c['passed']}/{c['n']} | {_pct(c['pass_rate'])} |"
        )

    lines += ["", "## 协议错误", ""]
    if s["protocol_errors"]:
        lines += ["| 类型 | 次数 |", "|---|---:|"]
        lines += [f"| `{k}` | {v} |" for k, v in s["protocol_errors"].items()]
    else:
        lines.append("无。")

    lines += ["", "## 逐字段错误分布", ""]
    if s["field_errors"]:
        lines += [
            "| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |",
            "|---|---:|---:|---:|---:|",
        ]
        for f, c in s["field_errors"].items():
            m, e, w = c.get("missing_key", 0), c.get("extra_key", 0), c.get("wrong_value", 0)
            lines.append(f"| `{f}` | {m} | {e} | {w} | {m + e + w} |")
    else:
        lines.append("无。")

    failed = [i for i in report["items"] if not i["passed"]]
    lines += ["", f"## 错误样例（{min(len(failed), max_examples)}/{len(failed)}）", ""]
    if not failed:
        lines.append("无。")
    for i in failed[:max_examples]:
        lines += [
            f"### {i['id']}（{CATEGORIES[i['category']]}）",
            "",
            f"- 输入：{i['user_input']}",
            f"- 期望：`{_json(i['expected'])}`",
            f"- 输出：`{i['raw'] or '(空)'}`",
        ]
        if not i["protocol_ok"]:
            lines.append(f"- 协议错误 `{i['protocol_error']}`：{'；'.join(i['protocol_messages'])}")
        for e in i["field_errors"]:
            detail = f"（{e['detail']}）" if e["detail"] else ""
            lines.append(
                f"- `{e['field']}` {e['error']}：期望 `{_json(e['expected'])}`，"
                f"实际 `{_json(e['predicted'])}`{detail}"
            )
        if i["extractor_errors"] and not i["raw"]:
            lines.append(f"- 抽取器错误：{'；'.join(i['extractor_errors'])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- CLI -----------------------------------------------------------------------------------


def make_eval_extractor(kind: str, settings: Any) -> SlotExtractor:
    from rift_agent.factory import build_single_extractor, make_extractor, with_extractor_config
    from rift_common.llm import LLMFactory

    llm = LLMFactory.create(settings.llm["default"])
    if kind == "routed":
        return make_extractor(with_extractor_config(settings, "routed"), llm)
    return build_single_extractor(kind, settings, llm)


def resolve_dataset(name: str) -> Path:
    return DATASETS_DIR / DATASETS[name] if name in DATASETS else Path(name)


def model_name(extractor: SlotExtractor) -> str:
    inner = getattr(extractor, "primary", extractor)
    llm = getattr(inner, "llm", None)
    return str(getattr(llm, "model", "") or "")


def run(
    args: argparse.Namespace,
    *,
    extractor: SlotExtractor | None = None,
    style: StyleMatcher | None = None,
) -> dict[str, Any]:
    dataset = resolve_dataset(args.dataset)
    samples = load_samples(dataset)[: args.limit]
    problems = checksum_problems(dataset.parent / "CHECKSUMS", [dataset])
    for p in problems:
        print(f"[warn] {p}")

    settings = None
    if extractor is None or style is None:
        from rift_common.settings import load_settings

        settings = load_settings(args.settings)
    if extractor is None:
        assert settings is not None
        extractor = make_eval_extractor(args.extractor, settings)
    if style is None:
        assert settings is not None
        style = make_style_matcher(
            None if args.exact_style else settings.embedding, threshold=args.style_threshold
        )

    started = datetime.now()
    print(f"{len(samples)} samples from {dataset.name}; extractor {args.extractor}")
    items = asyncio.run(
        evaluate(
            samples,
            extractor,
            style=style,
            concurrency=args.concurrency,
            threshold=args.threshold,
        )
    )
    dataset_name = args.dataset if args.dataset in DATASETS else dataset.stem
    report: dict[str, Any] = {
        "meta": {
            "extractor": args.extractor,
            "model": model_name(extractor),
            "dataset": dataset_name,
            "dataset_file": str(dataset.relative_to(REPO_ROOT))
            if dataset.is_relative_to(REPO_ROOT)
            else str(dataset),
            "sha256": sha256_file(dataset),
            "checksum_problems": problems,
            "started_at": f"{started:%Y-%m-%d %H:%M:%S}",
            "pass_threshold": args.threshold,
            "style_matcher": type(style).__name__,
            "style_threshold": style.threshold,
            "concurrency": args.concurrency,
        },
        "summary": summarize(items),
        "items": [asdict(i) for i in items],
    }
    stem: Path = args.out or REPORTS_DIR / (
        f"slot_eval_{args.extractor}_{dataset_name}_{started:%Y%m%d_%H%M%S}"
    )
    stem.parent.mkdir(parents=True, exist_ok=True)
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    md_path.write_text(
        render_markdown(report, max_examples=args.examples), encoding="utf-8", newline="\n"
    )
    report["paths"] = [str(json_path), str(md_path)]
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extractor", choices=EXTRACTORS, default="llm")
    parser.add_argument("--dataset", default="main", help="main | holdout | path to a JSONL")
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N samples")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=PASS_THRESHOLD)
    parser.add_argument("--style-threshold", type=float, default=DEFAULT_STYLE_THRESHOLD)
    parser.add_argument(
        "--exact-style", action="store_true", help="match style_preference as text (no model)"
    )
    parser.add_argument("--examples", type=int, default=15, help="failed samples in Markdown")
    parser.add_argument(
        "--out", type=Path, default=None, help="report path without suffix (.json + .md)"
    )
    args = parser.parse_args(argv)

    report = run(args)
    s = report["summary"]
    print(
        f"\npass {s['passed']}/{s['n']} = {_pct(s['pass_rate'])}; "
        f"protocol {_pct(s['protocol_pass_rate'])}; "
        f"P50 {s['latency_ms']['p50']:.0f} ms, P95 {s['latency_ms']['p95']:.0f} ms"
    )
    for path in report["paths"]:
        print(f"report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
