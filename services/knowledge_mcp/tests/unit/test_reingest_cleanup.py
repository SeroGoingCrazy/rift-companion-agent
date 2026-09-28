"""Tests for E4 fixes found while ingesting the knowledge base.

- Re-ingesting a changed file drops its old BM25 postings (postings are keyed by
  vector IDs ``{sha256(source_path)[:8]}_{index}_{content}``, not by doc id / file hash).
- DocumentManager.delete_document removes BM25 postings by those ID prefixes.
- MCP tools resolve a relative Chroma directory against the service root, not the cwd.
- query_knowledge_hub falls back to settings.vector_store.collection_name.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.core.settings as settings_module
from src.core.types import Chunk, Document
from src.ingestion.document_manager import DocumentManager
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.mcp_server.tools.get_document_summary import (
    GetDocumentSummaryConfig,
    GetDocumentSummaryTool,
)
from src.mcp_server.tools.list_collections import ListCollectionsConfig, ListCollectionsTool
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


def _vector_id(chunk: Chunk) -> str:
    upserter = VectorUpserter.__new__(VectorUpserter)
    return upserter._generate_chunk_id(chunk)


def test_source_prefix_is_prefix_of_every_vector_id():
    chunk = Chunk(id="c", text="退款 50%", metadata={"source_path": "/kb/refund.md", "chunk_index": 3})

    vector_id = _vector_id(chunk)

    assert vector_id.startswith(VectorUpserter.source_prefix("/kb/refund.md"))
    assert vector_id.split("_")[1] == "0003"
    assert VectorUpserter.source_prefix("/kb/refund.md") != VectorUpserter.source_prefix(
        "/kb/billing.md"
    )


# ── Pipeline re-ingest with a real BM25 index ────────────────────────


def _pipeline(tmp_path: Path, text: dict) -> object:
    class FP:
        collection = "platform_rules"
        force = True

    fp = FP()
    fp.integrity_checker = MagicMock()
    fp.integrity_checker.compute_sha256.return_value = "hash"
    fp.integrity_checker.should_skip.return_value = False

    fp.loader = MagicMock()
    fp.loader.load.side_effect = lambda path: Document(
        id="doc_x", text=text["value"], metadata={"source_path": path}
    )
    fp.chunker = MagicMock()
    fp.chunker.split_document.side_effect = lambda doc: [
        Chunk(id=f"c{i}", text=part, metadata={"source_path": doc.metadata["source_path"], "chunk_index": i})
        for i, part in enumerate(doc.text.split("|"))
    ]
    for name in ("chunk_refiner", "metadata_enricher", "image_captioner"):
        stage = MagicMock()
        stage.transform.side_effect = lambda chunks, trace=None: chunks
        setattr(fp, name, stage)

    def process(chunks, trace=None):
        return SimpleNamespace(
            dense_vectors=[[0.1]] * len(chunks),
            sparse_stats=[
                {"chunk_id": c.id, "term_frequencies": {w: 1 for w in c.text.split()}, "doc_length": len(c.text.split())}
                for c in chunks
            ],
        )

    fp.batch_processor = MagicMock()
    fp.batch_processor.process.side_effect = process
    fp.vector_upserter = MagicMock()
    fp.vector_upserter.upsert.side_effect = lambda chunks, vectors, trace=None: [
        _vector_id(c) for c in chunks
    ]
    fp.bm25_indexer = BM25Indexer(index_dir=str(tmp_path / "bm25"))
    fp.image_storage = MagicMock()
    return fp


def _posting_ids(indexer: BM25Indexer) -> set[str]:
    indexer.load("platform_rules")
    return {p["chunk_id"] for term in indexer._index.values() for p in term["postings"]}


def test_reingesting_changed_file_replaces_bm25_postings(tmp_path):
    refund, billing = str(tmp_path / "refund.md"), str(tmp_path / "billing.md")
    refund_prefix = VectorUpserter.source_prefix(refund)
    billing_prefix = VectorUpserter.source_prefix(billing)
    text = {"value": "退款 一半|全额 退款|不 退款"}
    IngestionPipeline.run(_pipeline(tmp_path, text), refund)
    IngestionPipeline.run(_pipeline(tmp_path, {"value": "计费 系数"}), billing)
    before = _posting_ids(BM25Indexer(str(tmp_path / "bm25")))
    assert sum(i.startswith(refund_prefix) for i in before) == 3

    text["value"] = "退款 六成|全额 退款"
    result = IngestionPipeline.run(_pipeline(tmp_path, text), refund)

    after = _posting_ids(BM25Indexer(str(tmp_path / "bm25")))
    assert result.success
    assert sorted(i.split("_")[1] for i in after if i.startswith(refund_prefix)) == ["0000", "0001"]
    # other documents untouched
    assert {i for i in after if i.startswith(billing_prefix)} == {
        i for i in before if i.startswith(billing_prefix)
    } != set()


# ── DocumentManager ──────────────────────────────────────────────────


def test_delete_document_removes_bm25_postings_by_chunk_prefix():
    chroma = MagicMock()
    chroma.collection.get.return_value = {"ids": ["ab12cd34_0000_x", "ab12cd34_0001_y"]}
    bm25 = MagicMock()
    bm25.remove_document.return_value = True
    images = MagicMock()
    images.list_images.return_value = []
    manager = DocumentManager(chroma, bm25, images, MagicMock())

    result = manager.delete_document("/kb/refund.md", "platform_rules", source_hash="f" * 64)

    bm25.remove_document.assert_called_once_with("ab12cd34_", "platform_rules")
    assert result.bm25_removed is True
    chroma.collection.get.assert_called_once_with(where={"doc_hash": "f" * 64}, include=[])


# ── Path resolution & defaults ───────────────────────────────────────


@pytest.fixture
def service_root(tmp_path, monkeypatch):
    root = tmp_path / "service"
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.setattr(settings_module, "REPO_ROOT", root)
    monkeypatch.chdir(elsewhere)
    return root


def test_list_collections_resolves_chroma_dir_against_service_root(service_root):
    tool = ListCollectionsTool(config=ListCollectionsConfig(persist_directory="./db/chroma"))

    tool._get_chroma_client()

    assert (service_root / "db" / "chroma").is_dir()
    assert not (Path.cwd() / "db").exists()


def test_get_document_summary_resolves_chroma_dir_against_service_root(service_root):
    tool = GetDocumentSummaryTool(config=GetDocumentSummaryConfig(persist_directory="./db/chroma"))

    tool._get_chroma_client()

    assert (service_root / "db" / "chroma").is_dir()
    assert not (Path.cwd() / "db").exists()


async def test_query_defaults_to_settings_collection():
    settings = SimpleNamespace(vector_store=SimpleNamespace(collection_name="platform_rules"))
    tool = QueryKnowledgeHubTool(settings=settings)
    tool._ensure_initialized = MagicMock(side_effect=RuntimeError("stop"))

    await tool.execute(query="开局前三小时取消退多少")

    tool._ensure_initialized.assert_called_once_with("platform_rules")
