"""Tests for E6 – ragas judge on DeepSeek with project (local) embeddings, and retrieve()."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubConfig, QueryKnowledgeHubTool
from src.observability.evaluation import ragas_evaluator
from src.observability.evaluation.ragas_evaluator import RagasEvaluator


class CountingEmbedding(BaseEmbedding):
    def __init__(self, settings, **kwargs):
        self.calls = []

    def embed(self, texts, trace=None, **kwargs):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


def _settings(llm_provider="deepseek", emb_provider="counting", api_key=""):
    return SimpleNamespace(
        llm=SimpleNamespace(provider=llm_provider, model="deepseek-chat", api_key=api_key),
        embedding=SimpleNamespace(provider=emb_provider, model="bge", dimensions=2),
        evaluation=None,
    )


@pytest.fixture(autouse=True)
def counting_provider(monkeypatch):
    monkeypatch.setitem(EmbeddingFactory._PROVIDERS, "counting", CountingEmbedding)


def test_deepseek_judge_uses_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    evaluator = RagasEvaluator(settings=_settings(), metrics=["faithfulness"])

    llm, embeddings = evaluator._build_wrappers()

    (client,) = [c for c in evaluator._open_clients]
    assert str(client.base_url).rstrip("/") == ragas_evaluator.DEEPSEEK_BASE_URL
    assert client.api_key == "sk-test"
    assert embeddings.embed_texts(["退款", "计费规则"]) == [[2.0, 1.0], [4.0, 1.0]]
    asyncio.run(client.close())


def test_project_embeddings_async_paths():
    embeddings = ragas_evaluator._project_embeddings(_settings())

    one = asyncio.run(embeddings.aembed_text("段位"))
    many = asyncio.run(embeddings.aembed_texts(["a", "bb"]))

    assert one == [2.0, 1.0]
    assert many == [[1.0, 1.0], [2.0, 1.0]]
    assert embeddings.embed_text("abc") == [3.0, 1.0]


def test_unknown_llm_provider_still_rejected():
    evaluator = RagasEvaluator(settings=_settings(llm_provider="ollama"), metrics=["faithfulness"])

    with pytest.raises(ValueError, match="azure, openai, deepseek"):
        evaluator._build_wrappers()


def test_clients_are_closed_inside_the_event_loop(monkeypatch):
    evaluator = RagasEvaluator(settings=_settings(), metrics=["faithfulness"])
    client = MagicMock()
    closed = []

    async def close():
        closed.append(asyncio.get_running_loop())

    client.close = close

    def build():
        evaluator._track(client)
        return MagicMock(), MagicMock()

    async def score_all(*args):
        return {"faithfulness": 1.0}

    monkeypatch.setattr(evaluator, "_build_wrappers", build)
    monkeypatch.setattr(evaluator, "_score_all", score_all)

    assert evaluator._run_ragas("q", ["ctx"], "answer") == {"faithfulness": 1.0}
    assert len(closed) == 1


def test_retrieve_runs_search_then_rerank_in_collection():
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection_name="platform_rules"))
    tool = QueryKnowledgeHubTool(settings=settings, config=QueryKnowledgeHubConfig(max_top_k=5))
    tool._ensure_initialized = MagicMock()
    tool._perform_search = MagicMock(return_value=["r1", "r2"])
    tool._apply_rerank = MagicMock(return_value=["r2"])

    assert tool.retrieve("取消退多少", top_k=10, collection="modes_and_ranks") == ["r2"]
    tool._ensure_initialized.assert_called_once_with("modes_and_ranks")
    tool._perform_search.assert_called_once_with("取消退多少", 5, None)

    tool.config.enable_rerank = False
    tool.retrieve("取消退多少")
    tool._ensure_initialized.assert_called_with("platform_rules")
    assert tool._apply_rerank.call_count == 1
