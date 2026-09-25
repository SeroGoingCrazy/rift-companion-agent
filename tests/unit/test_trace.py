import asyncio
import contextvars
import threading
from pathlib import Path

import pytest

from rift_common.llm import LLMFactory
from rift_common.settings import LLMConfig
from rift_common.trace import (
    Span,
    TraceContext,
    current_span,
    current_trace,
    span,
    start_turn,
    traced,
)
from rift_common.trace.sinks import SqliteSink, TraceSink


class RecordingSink(TraceSink):
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.turns: list[TraceContext] = []

    def on_span_start(self, span: Span, trace: TraceContext) -> None:
        self.events.append(("start", span.name))

    def on_span_end(self, span: Span, trace: TraceContext) -> None:
        self.events.append(("end", span.name))

    def on_turn_end(self, trace: TraceContext) -> None:
        self.turns.append(trace)


class ExplodingSink(TraceSink):
    def on_span_start(self, span: Span, trace: TraceContext) -> None:
        raise RuntimeError("sink down")

    def on_turn_end(self, trace: TraceContext) -> None:
        raise RuntimeError("sink down")


def _by_name(trace: TraceContext) -> dict[str, Span]:
    return {s.name: s for s in trace.spans}


# --- hierarchy -------------------------------------------------------------------


def test_nested_spans_have_correct_parents() -> None:
    sink = RecordingSink()
    with start_turn(session_id="s1", user_id="u1", turn_idx=0, sinks=[sink], input="hi") as root:
        with span("extract_slots") as outer:
            assert current_span() is outer
            with span("tool:find_companions", kind="tool", top_k=3) as inner:
                inner.set_attr("hits", 2)
        with span("render_reply"):
            pass
        root.set_attr("reply", "ok")

    (trace,) = sink.turns
    spans = _by_name(trace)
    assert spans["turn"].parent_id is None
    assert spans["extract_slots"].parent_id == root.span_id
    assert spans["tool:find_companions"].parent_id == spans["extract_slots"].span_id
    assert spans["render_reply"].parent_id == root.span_id
    assert spans["tool:find_companions"].attrs == {"top_k": 3, "hits": 2}
    assert spans["tool:find_companions"].kind == "tool"
    assert {s.trace_id for s in trace.spans} == {trace.trace_id}
    assert len(trace.trace_id) == 32
    assert all(s.finished and s.duration_ms is not None for s in trace.spans)
    assert sink.events == [
        ("start", "turn"),
        ("start", "extract_slots"),
        ("start", "tool:find_companions"),
        ("end", "tool:find_companions"),
        ("end", "extract_slots"),
        ("start", "render_reply"),
        ("end", "render_reply"),
        ("end", "turn"),
    ]
    assert current_trace() is None and current_span() is None


def test_exception_is_recorded_and_reraised() -> None:
    sink = RecordingSink()
    with (
        pytest.raises(ValueError, match="bad slot"),
        start_turn(session_id="s", sinks=[sink]),
        span("extract_slots"),
    ):
        raise ValueError("bad slot")

    spans = _by_name(sink.turns[0])
    assert spans["extract_slots"].status == "error"
    assert spans["extract_slots"].error == "ValueError: bad slot"
    assert spans["turn"].status == "error"


def test_caught_exception_leaves_parent_ok() -> None:
    sink = RecordingSink()
    with start_turn(sinks=[sink]):
        try:
            with span("tool:create_booking"):
                raise RuntimeError("SLOT_CONFLICT")
        except RuntimeError:
            pass
    spans = _by_name(sink.turns[0])
    assert spans["tool:create_booking"].status == "error"
    assert spans["turn"].status == "ok"


def test_span_outside_turn_is_detached() -> None:
    with span("orphan") as s:
        s.set_attr("x", 1)
    assert s.finished and s.trace_id == "" and s.attrs == {"x": 1}


def test_nested_turn_rejected() -> None:
    with start_turn(), pytest.raises(RuntimeError, match="already active"), start_turn():
        pass


def test_continue_external_trace() -> None:
    sink = RecordingSink()
    with start_turn(trace_id="a" * 32, parent_span_id="b" * 16, sinks=[sink]):
        pass
    root = sink.turns[0].root
    assert root is not None
    assert root.trace_id == "a" * 32
    assert root.parent_id == "b" * 16


def test_failing_sink_does_not_break_turn(caplog: pytest.LogCaptureFixture) -> None:
    good = RecordingSink()
    with start_turn(sinks=[ExplodingSink(), good]) as root:
        root.set_attr("reply", "still works")
    assert len(good.turns) == 1
    assert "trace sink ExplodingSink.on_span_start failed" in caplog.text


# --- decorator -------------------------------------------------------------------


@traced("classify", model="deepseek-chat")
def classify(text: str) -> str:
    return "booking" if "约" in text else "other"


@traced()
async def fetch(n: int) -> int:
    await asyncio.sleep(0)
    return n * 2


@traced("boom")
def boom() -> None:
    raise KeyError("missing")


def test_traced_sync_records_static_attrs() -> None:
    sink = RecordingSink()
    with start_turn(sinks=[sink]):
        assert classify("帮我约") == "booking"
    s = _by_name(sink.turns[0])["classify"]
    assert s.attrs == {"model": "deepseek-chat"}
    assert classify.__name__ == "classify"


async def test_traced_async_and_concurrent_tasks_keep_parents() -> None:
    sink = RecordingSink()
    with start_turn(sinks=[sink]) as root:
        results = await asyncio.gather(fetch(1), fetch(2))
    assert results == [2, 4]
    fetch_spans = [s for s in sink.turns[0].spans if s.name.endswith("fetch")]
    assert len(fetch_spans) == 2
    assert all(s.parent_id == root.span_id for s in fetch_spans)


def test_traced_records_error() -> None:
    sink = RecordingSink()
    with start_turn(sinks=[sink]), pytest.raises(KeyError):
        boom()
    assert _by_name(sink.turns[0])["boom"].error == "KeyError: 'missing'"


def test_spans_in_copied_context_thread_attach_to_parent() -> None:
    sink = RecordingSink()
    with start_turn(sinks=[sink]) as root:
        ctx = contextvars.copy_context()
        t = threading.Thread(target=ctx.run, args=(classify, "x"))
        t.start()
        t.join()
    assert _by_name(sink.turns[0])["classify"].parent_id == root.span_id


# --- LLM generation span ---------------------------------------------------------------


def test_llm_chat_emits_generation_span() -> None:
    sink = RecordingSink()
    llm = LLMFactory.create(LLMConfig(provider="mock", model="mock-1"), default="好的")
    with start_turn(sinks=[sink]):
        llm.chat([{"role": "user", "content": "你好"}], temperature=0)
    gen = _by_name(sink.turns[0])["llm:mock"]
    assert gen.kind == "generation"
    assert gen.attrs["model"] == "mock-1"
    assert gen.attrs["input"] == [{"role": "user", "content": "你好"}]
    assert gen.attrs["output"] == "好的"
    assert gen.attrs["usage"]["completion_tokens"] == 2
    assert gen.attrs["model_parameters"] == {"temperature": 0, "max_tokens": None}


# --- SQLite sink -----------------------------------------------------------------


def test_sqlite_sink_writes_turns_and_spans(tmp_path: Path) -> None:
    sink = SqliteSink(tmp_path / "nested" / "traces.db")
    for idx, text in enumerate(["约个双排", "取消要扣钱吗"]):
        with start_turn(
            session_id="sess-1",
            user_id="u1",
            turn_idx=idx,
            sinks=[sink],
            input=text,
            phase_before="IDLE",
        ) as root:
            with span("classify") as s:
                s.set_attr("label", "booking")
            root.set_attrs(reply=f"reply-{idx}", phase_after="BOOKING", custom={"k": 1})
    with start_turn(session_id="other", sinks=[sink]):
        pass

    turns = sink.list_turns("sess-1")
    assert [t["input"] for t in turns] == ["约个双排", "取消要扣钱吗"]
    first = turns[0]
    assert first["reply"] == "reply-0"
    assert first["phase_before"] == "IDLE" and first["phase_after"] == "BOOKING"
    assert first["user_id"] == "u1"
    assert first["status"] == "ok"
    assert first["latency_ms"] >= 0
    assert first["attrs"] == {"custom": {"k": 1}}

    spans = sink.list_spans(trace_id=first["trace_id"])
    assert {s["name"] for s in spans} == {"turn", "classify"}
    classify_row = next(s for s in spans if s["name"] == "classify")
    assert classify_row["attrs"] == {"label": "booking"}
    assert classify_row["session_id"] == "sess-1"

    assert len(sink.list_spans(session_id="sess-1")) == 4
    assert sink.list_turns("missing") == []


def test_sqlite_sink_records_errors(tmp_path: Path) -> None:
    sink = SqliteSink(tmp_path / "traces.db")
    with pytest.raises(RuntimeError), start_turn(session_id="s", sinks=[sink]):
        raise RuntimeError("node crashed")
    (turn,) = sink.list_turns("s")
    assert turn["status"] == "error"
    assert turn["error"] == "RuntimeError: node crashed"


def test_sqlite_list_spans_requires_one_filter(tmp_path: Path) -> None:
    sink = SqliteSink(tmp_path / "traces.db")
    with pytest.raises(ValueError, match="exactly one"):
        sink.list_spans()


def test_detached_hides_the_callers_turn() -> None:
    from rift_common.trace import current_span, current_trace, detached

    with start_turn(session_id="outer") as root:
        with detached():
            assert current_trace() is None and current_span() is None
            with start_turn(session_id="inner") as inner:
                assert inner.parent_id is None
                assert current_trace() is not None
                assert current_trace().session_id == "inner"  # type: ignore[union-attr]
        assert current_span() is root
