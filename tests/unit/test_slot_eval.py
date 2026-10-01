"""H3: L2 runner — context building, scoring, summary, Markdown and report files."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rift_agent.extractors.base import ExtractionContext, ExtractionResult, SlotExtractor
from rift_agent.extractors.prompt import render_context
from rift_training.evaluation.dataset import SlotSample, load_samples
from rift_training.evaluation.preferences import ExactStyleMatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / "eval" / "datasets" / "slot_main.jsonl"


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("run_slot_eval", REPO_ROOT / "eval" / "runners" / "run_slot_eval.py")
SAMPLES = load_samples(MAIN)
BY_ID = {s.id: s for s in SAMPLES}


class ScriptedExtractor(SlotExtractor):
    """Answers each input from a table: dict -> JSON output, str -> raw text, None -> crash."""

    name = "scripted"

    def __init__(self, table: dict[str, Any], *, default: Any = "oracle") -> None:
        self.table = table
        self.default = default
        self.expected = {s.user_input: s.expected for s in SAMPLES}
        self.contexts: list[ExtractionContext] = []

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        self.contexts.append(ctx)
        answer = self.table.get(ctx.user_input, self.default)
        if answer == "oracle":
            answer = self.expected[ctx.user_input]
        if answer is None:
            raise RuntimeError("boom")
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return ExtractionResult(extraction=None, raw=text, valid=True, source=self.name)


def evaluate(samples: list[SlotSample], extractor: SlotExtractor) -> list[Any]:
    import asyncio

    return list(
        asyncio.run(
            runner.evaluate(samples, extractor, style=ExactStyleMatcher(), log=lambda _: None)
        )
    )


def test_context_matches_the_agent_prompt_input() -> None:
    sample = BY_ID["main-031"]
    ctx = runner.build_context(sample)
    payload = json.loads(render_context(ctx))
    assert payload["user_input"] == "就第二个吧"
    assert payload["candidates"] == ["阿狸酱", "夜雨声烦", "小鹿乱撞"]
    assert payload["now"] == "2026-10-01 14:00 星期四"
    assert payload["current_state"]["start_time"] == "2026-10-02 20:00 星期五"
    assert payload["history"][-1]["role"] == "assistant"
    assert payload["pending_confirmation"] is False


def test_oracle_passes_every_sample() -> None:
    items = evaluate(SAMPLES, ScriptedExtractor({}))
    assert all(i.passed for i in items)
    summary = runner.summarize(items)
    assert summary["pass_rate"] == 1.0 and summary["field_errors"] == {}
    assert set(summary["by_category"]) == {s.category for s in SAMPLES}


def test_failures_are_classified() -> None:
    picked = [BY_ID[i] for i in ("main-001", "main-016", "main-031", "main-043", "main-048")]
    table: dict[str, Any] = {
        # extra key: service_type was never said
        "帮我约个明晚八点的单双排，钻石以上，打两个小时": {
            "turn_intent": "booking",
            "delta": {
                "game_mode": "ranked_solo_duo",
                "start_time_expr": "明天晚上8点",
                "rank_requirement": "diamond",
                "duration_hours": 2,
                "service_type": "climb",
            },
            "confirmation": "none",
        },
        "段位无所谓": "```json\n{}\n```",
        "就第二个吧": {"turn_intent": "booking", "delta": {}, "confirmation": "none"},
        "今天天气怎么样": None,
    }
    items = {i.id: i for i in evaluate(picked, ScriptedExtractor(table))}
    extra = items["main-001"]
    assert extra.protocol_ok and not extra.passed
    assert [e["field"] for e in extra.field_errors] == ["delta.service_type"]
    assert items["main-016"].protocol_error == "json"
    assert items["main-031"].field_errors[0]["error"] == "missing_key"
    crashed = items["main-043"]
    assert crashed.protocol_error == "empty" and "RuntimeError: boom" in crashed.extractor_errors[0]
    assert items["main-048"].passed

    summary = runner.summarize(list(items.values()))
    assert summary["passed"] == 1 and summary["protocol_pass_rate"] == 0.6
    assert summary["protocol_errors"] == {"json": 1, "empty": 1}
    assert summary["field_errors"]["delta.service_type"] == {"extra_key": 1}
    assert summary["mean_task_score"] == pytest.approx((6 / 7 + 2 / 3 + 1.0) / 3, abs=1e-3)


def test_percentile_and_latency_summary() -> None:
    assert runner.percentile([], 50) == 0.0
    values = [float(v) for v in range(1, 101)]
    assert runner.percentile(values, 50) == 50.0
    assert runner.percentile(values, 95) == 95.0
    assert runner.percentile([3.0, 1.0, 2.0], 100) == 3.0


def test_markdown_report_sections() -> None:
    picked = [BY_ID["main-001"], BY_ID["main-043"]]
    items = evaluate(picked, ScriptedExtractor({"今天天气怎么样": "oops"}))
    report = {
        "meta": {
            "extractor": "llm",
            "model": "deepseek-chat",
            "dataset": "main",
            "dataset_file": "eval/datasets/slot_main.jsonl",
            "sha256": "ab" * 32,
            "checksum_problems": [],
            "started_at": "2026-10-01 14:00:00",
            "pass_threshold": 0.95,
            "style_matcher": "ExactStyleMatcher",
            "style_threshold": 1.0,
        },
        "summary": runner.summarize(items),
        "items": [runner.asdict(i) for i in items],
    }
    md = runner.render_markdown(report)
    assert md.startswith("# L2 槽位抽取评测：llm / main")
    assert "**50.0%**（1/2）" in md
    assert "| `json` | 1 |" in md
    assert "### main-043（无关话题）" in md and "- 输出：`oops`" in md
    assert "## 逐字段错误分布\n\n无。" in md


def test_run_writes_json_and_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "slot_main.jsonl"
    shutil.copy(MAIN, dataset)  # no CHECKSUMS next to it -> warning
    args = argparse.Namespace(
        extractor="llm",
        dataset=str(dataset),
        settings=None,
        limit=4,
        concurrency=2,
        threshold=0.95,
        style_threshold=0.7,
        exact_style=True,
        examples=5,
        out=tmp_path / "reports" / "baseline",
    )
    report = runner.run(args, extractor=ScriptedExtractor({}), style=ExactStyleMatcher())
    assert "no checksum recorded" in capsys.readouterr().out
    data = json.loads((tmp_path / "reports" / "baseline.json").read_text("utf-8"))
    assert data["summary"]["n"] == 4 and data["summary"]["pass_rate"] == 1.0
    assert data["meta"]["dataset"] == "slot_main" and data["meta"]["checksum_problems"]
    assert len(data["items"]) == 4
    assert (tmp_path / "reports" / "baseline.md").read_text("utf-8").startswith("# L2")
    assert report["paths"][0].endswith("baseline.json")


def test_resolve_dataset_names() -> None:
    assert runner.resolve_dataset("main") == MAIN
    assert runner.resolve_dataset("holdout").name == "slot_holdout.jsonl"
    assert runner.resolve_dataset("x/y.jsonl") == Path("x/y.jsonl")
