"""L3 RAG evaluation: Hit@3 / MRR@10 over the golden set, optional ragas.

Each golden question is searched in its own collection, the same way the agent's
consult node queries knowledge-mcp (``QueryKnowledgeHubTool.retrieve``). A question
hits when a chunk of its expected KB document is ranked within the first ``hit_k``;
its reciprocal rank is ``1 / rank`` of the first such chunk within ``top_k``.

With ``--ragas`` each question is also answered by DeepSeek from the top ``hit_k``
chunks using the agent's ``consult_answer.txt`` prompt, then scored for ragas
faithfulness / answer relevancy (judge: DeepSeek; embeddings: the knowledge-mcp
embedding, i.e. local bge). Needs ``DEEPSEEK_API_KEY``.

Usage::

    uv run python eval/runners/run_rag_eval.py
    uv run --env-file .env python eval/runners/run_rag_eval.py --ragas
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "eval" / "datasets" / "rag_golden.jsonl"
DEFAULT_SETTINGS = REPO_ROOT / "services" / "knowledge_mcp" / "config" / "settings.yaml"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
RAGAS_METRICS = ("faithfulness", "answer_relevancy")

#: Acceptance gates from DEV_SPEC E6.
MIN_HIT_RATE = 0.9
MIN_MRR = 0.8

Retriever = Callable[[str, str, int], list[Any]]
Answerer = Callable[[str, list[str]], str]
Scorer = Callable[[str, list[str], str], dict[str, float]]


@dataclass(frozen=True)
class GoldenItem:
    id: str
    collection: str
    category: str
    query: str
    reference_answer: str
    expected_sources: tuple[str, ...]

    @classmethod
    def from_json(cls, line: str) -> GoldenItem:
        data = json.loads(line)
        data["expected_sources"] = tuple(data["expected_sources"])
        return cls(**data)


@dataclass
class ItemResult:
    id: str
    collection: str
    query: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit: bool
    reciprocal_rank: float
    answer: str | None = None
    ragas: dict[str, float] = field(default_factory=dict)


# --- dataset --------------------------------------------------------------------------------


def load_dataset(path: Path) -> list[GoldenItem]:
    return [GoldenItem.from_json(line) for line in path.read_text("utf-8").splitlines() if line]


def verify_checksum(path: Path) -> str | None:
    """Return a problem description, or None when the ``.sha256`` sidecar matches."""
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        return f"missing checksum file {sidecar.name}"
    expected = sidecar.read_text("utf-8").split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        return f"{path.name} was edited by hand (sha256 {actual[:12]} != {expected[:12]})"
    return None


def _load_generator() -> ModuleType:
    name = "gen_rag_golden"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_fresh(path: Path) -> str | None:
    """Rebuild the golden set from domain.yaml + seed; report if the file is stale."""
    from booking_mcp.seed import generate_companions
    from rift_domain.config import load_domain_config

    gen = _load_generator()
    domain = load_domain_config(REPO_ROOT / "config" / "domain.yaml")
    current = gen.render(gen.build_golden(domain, generate_companions(domain)))
    if current != path.read_text("utf-8"):
        return "golden set is stale; run: uv run python scripts/gen_rag_golden.py"
    return None


# --- scoring --------------------------------------------------------------------------------


def source_of(result: Any, default_collection: str) -> str:
    """``collection/file.md`` of a retrieved chunk."""
    meta = getattr(result, "metadata", None) or {}
    collection = meta.get("collection") or default_collection
    name = Path(str(meta.get("source_path") or meta.get("source") or "")).name
    return f"{collection}/{name}"


def score_item(
    item: GoldenItem, retrieved_sources: Sequence[str], hit_k: int
) -> tuple[bool, float]:
    ranks = [i for i, src in enumerate(retrieved_sources, 1) if src in item.expected_sources]
    first = ranks[0] if ranks else None
    return (first is not None and first <= hit_k), (1.0 / first if first else 0.0)


def _mean(values: Sequence[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def summarize(results: Sequence[ItemResult]) -> dict[str, Any]:
    def block(rs: Sequence[ItemResult]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "count": len(rs),
            "hit_rate": _mean([float(r.hit) for r in rs]),
            "mrr": _mean([r.reciprocal_rank for r in rs]),
        }
        for metric in RAGAS_METRICS:
            values = [r.ragas[metric] for r in rs if metric in r.ragas]
            if values:
                out[metric] = _mean(values)
        return out

    collections = sorted({r.collection for r in results})
    return {
        "overall": block(results),
        "by_collection": {c: block([r for r in results if r.collection == c]) for c in collections},
    }


def evaluate(
    items: Sequence[GoldenItem],
    retrieve: Retriever,
    *,
    hit_k: int = 3,
    top_k: int = 10,
    answer: Answerer | None = None,
    score_ragas: Scorer | None = None,
    log: Callable[[str], None] = print,
) -> list[ItemResult]:
    results = []
    for item in items:
        chunks = retrieve(item.query, item.collection, top_k)
        sources = [source_of(c, item.collection) for c in chunks]
        hit, rr = score_item(item, sources, hit_k)
        result = ItemResult(
            item.id,
            item.collection,
            item.query,
            list(item.expected_sources),
            sources,
            hit,
            rr,
        )
        if answer is not None:
            contexts = [str(getattr(c, "text", c)) for c in chunks[:hit_k]]
            result.answer = answer(item.query, contexts)
            if score_ragas is not None:
                try:
                    result.ragas = score_ragas(item.query, contexts, result.answer)
                except Exception as exc:  # one judge failure must not sink the run
                    log(f"  ! ragas failed for {item.id}: {exc}")
        mark = "✓" if hit else "✗"
        log(f"  {mark} {item.id:<28} rr={rr:.2f}  top={sources[0] if sources else '-'}")
        results.append(result)
    return results


# --- real backends --------------------------------------------------------------------------


def make_retriever(settings: Any) -> Retriever:
    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

    tool = QueryKnowledgeHubTool(settings=settings)
    return lambda query, collection, top_k: tool.retrieve(query, top_k, collection)


def make_answerer() -> Answerer:
    """DeepSeek with the agent's consult prompt (same as ``consult`` in the agent)."""
    from rift_agent.prompts import load_prompt
    from rift_common.llm import LLMFactory
    from rift_common.settings import load_settings

    llm = LLMFactory.create(load_settings(REPO_ROOT / "config" / "settings.yaml").llm["default"])
    system = load_prompt("consult_answer.txt")

    def answer(query: str, contexts: list[str]) -> str:
        context = "\n\n".join(f"[{i}]\n{text}" for i, text in enumerate(contexts, 1))
        result = llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"【资料】\n{context}\n\n【问题】{query}"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        return result.text.strip()

    return answer


def make_ragas_scorer(settings: Any) -> Scorer:
    from src.observability.evaluation.ragas_evaluator import RagasEvaluator

    evaluator = RagasEvaluator(settings=settings, metrics=list(RAGAS_METRICS))
    return lambda query, contexts, answer: evaluator.evaluate(
        query=query, retrieved_chunks=contexts, generated_answer=answer
    )


# --- CLI ------------------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "  -  " if value is None or not math.isfinite(value) else f"{value:.3f}"


def print_summary(summary: dict[str, Any]) -> None:
    header = f"{'collection':<20}{'n':>4}{'Hit@3':>8}{'MRR':>8}"
    header += "".join(f"{m[:12]:>14}" for m in RAGAS_METRICS)
    print(header)
    rows = [*summary["by_collection"].items(), ("overall", summary["overall"])]
    for name, block in rows:
        line = f"{name:<20}{block['count']:>4}{_fmt(block['hit_rate']):>8}{_fmt(block['mrr']):>8}"
        line += "".join(f"{_fmt(block.get(m)):>14}" for m in RAGAS_METRICS)
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--hit-k", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--ragas", action="store_true", help="also answer + score with ragas")
    parser.add_argument("--min-hit-rate", type=float, default=MIN_HIT_RATE)
    parser.add_argument("--min-mrr", type=float, default=MIN_MRR)
    parser.add_argument("--out", type=Path, default=None, help="report JSON path")
    args = parser.parse_args(argv)
    logging.getLogger("src").setLevel(logging.WARNING)

    for problem in (verify_checksum(args.dataset), check_fresh(args.dataset)):
        if problem:
            print(f"[warn] {problem}")
    items = load_dataset(args.dataset)[: args.limit]

    from src.core.settings import load_settings

    settings = load_settings(str(args.settings))
    print(f"{len(items)} questions; embedding {settings.embedding.model}; ragas={args.ragas}")
    results = evaluate(
        items,
        make_retriever(settings),
        hit_k=args.hit_k,
        top_k=args.top_k,
        answer=make_answerer() if args.ragas else None,
        score_ragas=make_ragas_scorer(settings) if args.ragas else None,
    )
    summary = summarize(results)
    print()
    print_summary(summary)

    out = args.out or REPORTS_DIR / f"rag_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(args.dataset.relative_to(REPO_ROOT))
        if args.dataset.is_relative_to(REPO_ROOT)
        else str(args.dataset),
        "embedding": settings.embedding.model,
        "hit_k": args.hit_k,
        "top_k": args.top_k,
        "summary": summary,
        "items": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")

    overall = summary["overall"]
    passed = overall["hit_rate"] >= args.min_hit_rate and overall["mrr"] >= args.min_mrr
    print(
        f"gate Hit@{args.hit_k} >= {args.min_hit_rate}, MRR >= {args.min_mrr}: "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
