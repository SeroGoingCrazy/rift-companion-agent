"""Tests for E5 – linking RAG traces to the caller's trace via ``_meta.trace_id``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from src.core.trace import TraceCollector, TraceContext, caller_trace
from src.mcp_server.protocol_handler import ProtocolHandler, _meta_value, create_mcp_server


def test_trace_outside_caller_has_no_parent():
    trace = TraceContext()

    assert trace.parent_trace_id is None
    assert "parent_trace_id" not in trace.to_dict()


def test_traces_inside_caller_record_parent():
    with caller_trace("ab" * 16, "1234567890abcdef"):
        trace = TraceContext()
    after = TraceContext()

    assert trace.to_dict()["parent_trace_id"] == "ab" * 16
    assert trace.to_dict()["parent_span_id"] == "1234567890abcdef"
    assert after.parent_trace_id is None


def test_empty_trace_id_clears_parent():
    with caller_trace("outer"), caller_trace(None, "span"):
        trace = TraceContext()

    assert (trace.parent_trace_id, trace.parent_span_id) == (None, None)


def test_collector_writes_parent_trace_id(tmp_path):
    collector = TraceCollector(traces_path=tmp_path / "traces.jsonl")
    with caller_trace("cd" * 16):
        collector.collect(TraceContext())

    assert '"parent_trace_id": "' + "cd" * 16 + '"' in (tmp_path / "traces.jsonl").read_text("utf-8")


def test_meta_value_reads_extra_fields():
    meta = SimpleNamespace(trace_id=None, model_extra={"trace_id": "t1", "parent_span_id": 7})

    assert _meta_value(meta, "trace_id") == "t1"
    assert _meta_value(meta, "parent_span_id") == "7"
    assert _meta_value(None, "trace_id") is None
    assert _meta_value(SimpleNamespace(model_extra=None), "trace_id") is None


@pytest.mark.asyncio
async def test_tool_call_meta_becomes_parent_of_traces():
    seen: list[TraceContext] = []

    async def probe() -> str:
        seen.append(TraceContext())
        return "ok"

    handler = ProtocolHandler(server_name="t", server_version="0")
    handler.register_tool("probe", "records a trace", {"type": "object", "properties": {}}, probe)
    server = create_mcp_server("t", "0", protocol_handler=handler, register_tools=False)

    async with create_connected_server_and_client_session(server) as client:
        await client.call_tool("probe", {}, meta={"trace_id": "ef" * 16, "parent_span_id": "s1"})
        await client.call_tool("probe", {})

    assert (seen[0].parent_trace_id, seen[0].parent_span_id) == ("ef" * 16, "s1")
    assert seen[1].parent_trace_id is None
