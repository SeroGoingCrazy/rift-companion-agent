"""E5: knowledge-mcp over Streamable HTTP, with RAG traces linked to the caller's trace.

A small KB is generated here and ingested with a deterministic hash embedding, so the
real pipeline, Chroma, BM25 and all three tools run offline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import src.core.settings as rag_settings
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from src.ingestion.pipeline import IngestionPipeline
from src.libs.embedding.base_embedding import BaseEmbedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.mcp_server import server as knowledge_server
from src.mcp_server.protocol_handler import create_mcp_server
from src.mcp_server.tools import query_knowledge_hub
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

TRACE_ID = "ab" * 16
SPAN_ID = "1234567890abcdef"

KB = {
    "refund.md": """# 取消与退款规则

## 退款档位

按取消时距离开局的时间计算退款：开局前 24 小时以上取消全额退款；开局前 2–24 小时取消退一半；
开局前不足 2 小时取消不退款。

## 未支付订单

待支付订单取消不产生费用。
""",
    "billing.md": """# 计费规则

## 计价公式

总价 = 陪玩师基础单价 × 服务类型系数 × 时长。上分系数 1.2，娱乐系数 1，教学系数 1.5。
""",
    "late_arrival.md": """# 迟到补偿

## 宽限时间

陪玩师在约定开局时间后 5 分钟内上线视为准时，超过后每迟到 1 分钟顺延 2 分钟。
""",
}


class HashEmbedding(BaseEmbedding):
    """Deterministic character-bigram hashing embedding (offline stand-in for bge)."""

    DIM = 64

    def __init__(self, settings: Any, **_: Any) -> None:
        self.dimension = self.DIM

    def embed(self, texts: list[str], trace: Any = None, **_: Any) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.DIM
            for a, b in itertools.pairwise(text):
                digest = hashlib.md5(f"{a}{b}".encode()).digest()
                vec[digest[0] % self.DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def get_dimension(self) -> int:
        return self.DIM


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def service_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the knowledge service's data/log dirs at a temp root and ingest a small KB."""
    root = tmp_path / "service"
    monkeypatch.setattr(rag_settings, "REPO_ROOT", root)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setitem(EmbeddingFactory._PROVIDERS, "hash", HashEmbedding)

    settings = rag_settings.load_settings()
    settings = dataclasses.replace(
        settings,
        embedding=dataclasses.replace(settings.embedding, provider="hash", dimensions=64),
    )
    kb = tmp_path / "kb" / "platform_rules"
    kb.mkdir(parents=True)
    pipeline = IngestionPipeline(settings, collection="platform_rules")
    for name, text in KB.items():
        (kb / name).write_bytes(text.encode("utf-8"))
        assert pipeline.run(str(kb / name)).success
    pipeline.integrity_checker.close()

    monkeypatch.setattr(
        query_knowledge_hub, "_tool_instance", QueryKnowledgeHubTool(settings=settings)
    )
    return root


@pytest.fixture
def http_url(service_root: Path) -> Iterator[str]:
    port = _free_port()
    app = knowledge_server.create_http_app(create_mcp_server("knowledge-mcp-test", "0"))
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _traces(root: Path) -> list[dict[str, Any]]:
    path = root / "logs" / "traces.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


async def test_three_tools_over_http_with_parent_trace(http_url: str, service_root: Path) -> None:
    doc_id = "doc_" + hashlib.sha256(KB["refund.md"].encode()).hexdigest()[:16]
    meta = {"trace_id": TRACE_ID, "parent_span_id": SPAN_ID}

    async with (
        streamable_http_client(http_url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        names = {t.name for t in (await session.list_tools()).tools}
        found = await session.call_tool(
            "query_knowledge_hub", {"query": "开局前三小时取消退多少", "top_k": 3}, meta=meta
        )
        collections = await session.call_tool("list_collections", {}, meta=meta)
        summary = await session.call_tool(
            "get_document_summary", {"doc_id": doc_id, "collection": "platform_rules"}
        )

    assert names == {"query_knowledge_hub", "list_collections", "get_document_summary"}
    assert not found.isError
    first_hit = found.content[0].text.split("### [2]")[0]
    assert "refund.md" in first_hit
    assert "platform_rules" in collections.content[0].text
    assert not summary.isError
    assert "取消与退款规则" in summary.content[0].text

    (trace,) = [t for t in _traces(service_root) if t["trace_type"] == "query"]
    assert trace["parent_trace_id"] == TRACE_ID
    assert trace["parent_span_id"] == SPAN_ID
    assert trace["trace_id"] != TRACE_ID
    assert trace["metadata"]["collection"] == "platform_rules"


async def test_calls_without_meta_have_no_parent(http_url: str, service_root: Path) -> None:
    async with (
        streamable_http_client(http_url) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        await session.call_tool(
            "query_knowledge_hub", {"query": "上分系数"}, meta={"trace_id": TRACE_ID}
        )
        await session.call_tool("query_knowledge_hub", {"query": "迟到补偿"})

    linked, plain = _traces(service_root)
    assert linked["parent_trace_id"] == TRACE_ID
    assert linked["parent_span_id"] is None
    assert "parent_trace_id" not in plain


def test_cli_defaults_to_stdio_and_port_8102() -> None:
    args = knowledge_server.parse_args([])
    assert (args.transport, args.port) == ("stdio", 8102)
    assert knowledge_server.parse_args(["--transport", "http"]).transport == "http"
