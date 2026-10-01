"""E6: L3 RAG golden set (generated from domain.yaml + seed) and the Hit@3 / MRR runner."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from booking_mcp.seed import generate_companions
from rift_domain.config import DomainConfig, parse_domain_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET = REPO_ROOT / "eval" / "datasets" / "rag_golden.jsonl"
RAW: dict[str, Any] = yaml.safe_load((REPO_ROOT / "config" / "domain.yaml").read_text("utf-8"))


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gen = _load("gen_rag_golden", REPO_ROOT / "scripts" / "gen_rag_golden.py")
kb = _load("gen_kb_docs", REPO_ROOT / "scripts" / "gen_kb_docs.py")
runner = _load("run_rag_eval", REPO_ROOT / "eval" / "runners" / "run_rag_eval.py")


def _golden(domain: DomainConfig) -> list[Any]:
    return gen.build_golden(domain, generate_companions(domain))


def _by_id(items: list[Any]) -> dict[str, Any]:
    return {item.id: item for item in items}


# --- golden set -----------------------------------------------------------------------------


def test_golden_set_covers_three_collections(domain: DomainConfig) -> None:
    items = _golden(domain)
    counts = {c: sum(i.collection == c for i in items) for c in kb.COLLECTIONS}

    assert 28 <= len(items) <= 40
    assert all(n >= 5 for n in counts.values()), counts
    assert len({i.id for i in items}) == len(items)
    assert len({i.query for i in items}) == len(items)


def test_expected_sources_are_generated_kb_docs(domain: DomainConfig) -> None:
    docs = {
        f"{d.collection}/{d.name}.md" for d in kb.build_docs(domain, generate_companions(domain))
    }

    for item in _golden(domain):
        assert set(item.expected_sources) <= docs, item.id
        assert all(src.startswith(f"{item.collection}/") for src in item.expected_sources)


def test_reference_answers_are_computed_from_domain(domain: DomainConfig) -> None:
    items = _by_id(_golden(domain))

    assert "全额退款" in items["refund_3h"].reference_answer
    assert "200.00 元" in items["refund_amount"].reference_answer
    assert "15 分钟" in items["refund_full"].query
    assert "提前 10 分钟则不退款" in items["refund_boundary"].reference_answer
    assert "288.00 元" in items["billing_compute"].reference_answer
    assert "钻石–宗师" in items["rank_band_diamond"].reference_answer


def test_reference_answers_follow_domain_yaml_changes() -> None:
    raw = copy.deepcopy(RAW)
    raw["service_types"]["climb"]["multiplier"] = 1.3
    raw["refund"]["tiers"] = [
        {"name": "full", "min_hours_before": 24, "ratio": 1.0, "label": "全额退款"},
        {"name": "part", "min_hours_before": 2, "ratio": 0.6, "label": "退六成"},
        {"name": "none", "min_hours_before": 0, "ratio": 0.0, "label": "不退款"},
    ]
    raw["policies"]["late_arrival"]["grace_minutes"] = 8
    items = _by_id(_golden(parse_domain_config(raw)))

    assert "80 × 1.3 × 3 = 312.00 元" in items["billing_compute"].reference_answer
    assert "120.00 元" in items["refund_amount"].reference_answer
    assert "8 分钟内" in items["late_grace"].reference_answer


def test_committed_dataset_matches_domain_yaml() -> None:
    assert runner.verify_checksum(DATASET) is None
    assert runner.check_fresh(DATASET) is None, "run: uv run python scripts/gen_rag_golden.py"


def test_write_produces_jsonl_and_checksum(tmp_path: Path, domain: DomainConfig) -> None:
    out = tmp_path / "golden.jsonl"
    digest = gen.write(_golden(domain)[:3], out)

    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    assert runner.verify_checksum(out) is None
    assert [item.id for item in runner.load_dataset(out)] == [i.id for i in _golden(domain)[:3]]

    out.write_text(out.read_text("utf-8").replace("退", "还"), encoding="utf-8")
    assert "edited by hand" in (runner.verify_checksum(out) or "")
    (tmp_path / "golden.jsonl.sha256").unlink()
    assert "missing checksum" in (runner.verify_checksum(out) or "")


# --- runner ---------------------------------------------------------------------------------


def _item(**kw: Any) -> Any:
    base = {
        "id": "q",
        "collection": "platform_rules",
        "category": "refund",
        "query": "取消退多少",
        "reference_answer": "退一半",
        "expected_sources": ("platform_rules/refund.md",),
    }
    return runner.GoldenItem(**{**base, **kw})


def _chunk(path: str, text: str = "x", collection: str | None = None) -> SimpleNamespace:
    meta: dict[str, Any] = {"source_path": path}
    if collection:
        meta["collection"] = collection
    return SimpleNamespace(text=text, metadata=meta)


def test_source_of_uses_chunk_collection_and_file_name() -> None:
    chunk = _chunk(r"C:\kb\platform_rules\refund.md", collection="platform_rules")

    assert runner.source_of(chunk, "other") == "platform_rules/refund.md"
    assert runner.source_of(_chunk("/kb/ranks.md"), "modes_and_ranks") == "modes_and_ranks/ranks.md"


@pytest.mark.parametrize(
    "sources, hit, rr",
    [
        (["platform_rules/refund.md", "platform_rules/billing.md"], True, 1.0),
        (["billing.md", "matching.md", "platform_rules/refund.md"], True, 1 / 3),
        (["a", "b", "c", "platform_rules/refund.md"], False, 0.25),
        (["a", "b"], False, 0.0),
    ],
)
def test_score_item_hit_at_3_and_reciprocal_rank(sources: list[str], hit: bool, rr: float) -> None:
    assert runner.score_item(_item(), sources, hit_k=3) == (hit, pytest.approx(rr))


def test_evaluate_and_summarize_per_collection() -> None:
    items = [
        _item(id="a"),
        _item(id="b", query="计费"),
        _item(
            id="c",
            query="钻石能匹配什么段位",
            collection="modes_and_ranks",
            expected_sources=("modes_and_ranks/ranks.md",),
        ),
    ]
    ranking = {
        "a": ["/kb/refund.md"],
        "b": ["/kb/billing.md", "/kb/matching.md", "/kb/late.md", "/kb/refund.md"],
        "c": ["/kb/game_modes.md", "/kb/ranks.md"],
    }
    by_query = {i.query: ranking[i.id] for i in items}
    calls: list[tuple[str, str, int]] = []

    def retrieve(query: str, collection: str, top_k: int) -> list[Any]:
        calls.append((query, collection, top_k))
        return [_chunk(p, text=f"text of {p}") for p in by_query[query]]

    results = runner.evaluate(items, retrieve, hit_k=3, top_k=10, log=lambda _m: None)
    summary = runner.summarize(results)

    assert calls[2] == ("钻石能匹配什么段位", "modes_and_ranks", 10)
    assert [r.hit for r in results] == [True, False, True]
    assert summary["by_collection"]["platform_rules"]["hit_rate"] == pytest.approx(0.5)
    assert summary["by_collection"]["platform_rules"]["mrr"] == pytest.approx((1 + 0.25) / 2)
    assert summary["overall"]["hit_rate"] == pytest.approx(2 / 3)
    assert "faithfulness" not in summary["overall"]


def test_evaluate_answers_from_top_hit_k_and_survives_judge_errors() -> None:
    seen: list[list[str]] = []

    def answer(query: str, contexts: list[str]) -> str:
        seen.append(contexts)
        return "退一半 [1]"

    def judge(query: str, contexts: list[str], ans: str) -> dict[str, float]:
        if query == "boom":
            raise RuntimeError("judge down")
        return {"faithfulness": 1.0, "answer_relevancy": 0.8}

    items = [_item(id="ok"), _item(id="bad", query="boom")]
    chunks = [_chunk(f"/kb/{n}.md", text=n) for n in ("refund", "billing", "late", "levels")]
    logs: list[str] = []

    results = runner.evaluate(
        items, lambda *_: chunks, answer=answer, score_ragas=judge, log=logs.append
    )
    summary = runner.summarize(results)

    assert seen[0] == ["refund", "billing", "late"]
    assert results[0].answer == "退一半 [1]"
    assert results[1].ragas == {}
    assert any("judge down" in line for line in logs)
    assert summary["overall"]["faithfulness"] == pytest.approx(1.0)
    assert summary["overall"]["answer_relevancy"] == pytest.approx(0.8)


def test_report_items_are_json_serializable() -> None:
    results = runner.evaluate([_item()], lambda *_: [_chunk("/kb/refund.md")], log=lambda _m: None)
    json.dumps([runner.asdict(r) for r in results], ensure_ascii=False)
