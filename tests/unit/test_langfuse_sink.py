import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from rift_common.llm import LLMFactory
from rift_common.settings import LangfuseConfig, LLMConfig, TraceConfig
from rift_common.trace import span, start_turn
from rift_common.trace.sinks import LangfuseSink, SqliteSink, create_sinks

ENABLED = LangfuseConfig(enabled=True, public_key="${PK}", secret_key="${SK}")


class FakeObservation:
    def __init__(
        self, client: "FakeClient", name: str, parent: "FakeObservation | None", **kw: Any
    ):
        self.client = client
        self.name = name
        self.parent = parent
        self.start_kwargs = kw
        self.updates: dict[str, Any] = {}
        self.ended = False
        client.observations.append(self)

    def start_observation(self, *, name: str, **kw: Any) -> "FakeObservation":
        if self.client.fail_on == "child":
            raise ConnectionError("langfuse unreachable")
        return FakeObservation(self.client, name, self, **kw)

    def update(self, **kw: Any) -> None:
        self.updates.update(kw)

    def end(self) -> None:
        self.ended = True


class FakeClient:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.observations: list[FakeObservation] = []
        self.propagated: list[dict[str, Any]] = []
        self.exited = 0
        self.flushed = False

    def start_observation(self, *, trace_context: dict[str, str], name: str, **kw: Any) -> Any:
        if self.fail_on == "root":
            raise ConnectionError("langfuse unreachable")
        return FakeObservation(self, name, None, trace_context=trace_context, **kw)

    @contextmanager
    def propagate(self, **kw: Any) -> Iterator[None]:
        self.propagated.append(kw)
        yield
        self.exited += 1

    def flush(self) -> None:
        self.flushed = True

    def by_name(self, name: str) -> FakeObservation:
        return next(o for o in self.observations if o.name == name)


def _sink(client: FakeClient) -> LangfuseSink:
    return LangfuseSink(ENABLED, client=client, propagate_attributes=client.propagate)


def _run_turn(sink: LangfuseSink, **turn_kwargs: Any) -> str:
    llm = LLMFactory.create(LLMConfig(provider="mock", model="deepseek-chat"), default="好的")
    with start_turn(
        session_id="sess-1", user_id="u1", turn_idx=0, sinks=[sink], input="约个双排", **turn_kwargs
    ) as root:
        with span("extract_slots", valid=True):
            llm.chat([{"role": "user", "content": "约个双排"}], temperature=0)
        with span("tool:find_companions", kind="tool", input={"top_k": 3}) as tool:
            tool.set_attr("output", ["阿狸酱"])
        root.set_attrs(reply="为你找到 1 位陪玩", phase_after="BOOKING")
        return root.trace_id


def test_turn_maps_to_trace_with_session_and_user() -> None:
    client = FakeClient()
    sink = _sink(client)
    trace_id = _run_turn(sink)

    assert client.propagated == [{"session_id": "sess-1", "user_id": "u1", "trace_name": "turn"}]
    assert client.exited == 1
    root = client.by_name("turn")
    assert root.parent is None
    assert root.start_kwargs["trace_context"] == {"trace_id": trace_id}
    assert root.start_kwargs["input"] == "约个双排"
    assert root.updates["output"] == "为你找到 1 位陪玩"
    assert root.updates["metadata"] == {"phase_after": "BOOKING"}
    assert all(o.ended for o in client.observations)


def test_llm_call_maps_to_generation_with_usage() -> None:
    client = FakeClient()
    _run_turn(_sink(client))

    gen = client.by_name("llm:mock")
    assert gen.parent is client.by_name("extract_slots")
    assert gen.start_kwargs["as_type"] == "generation"
    assert gen.start_kwargs["input"] == [{"role": "user", "content": "约个双排"}]
    assert gen.updates["output"] == "好的"
    assert gen.updates["model"] == "deepseek-chat"
    assert gen.updates["model_parameters"] == {"temperature": 0, "max_tokens": None}
    assert gen.updates["usage_details"] == {"input": 4, "output": 2, "total": 6}


def test_tool_span_and_metadata() -> None:
    client = FakeClient()
    _run_turn(_sink(client))
    tool = client.by_name("tool:find_companions")
    assert tool.start_kwargs["as_type"] == "tool"
    assert tool.start_kwargs["input"] == {"top_k": 3}
    assert tool.updates["output"] == ["阿狸酱"]
    assert client.by_name("extract_slots").updates["metadata"] == {"valid": True}


def test_error_span_marked_error_level() -> None:
    client = FakeClient()
    sink = _sink(client)
    with start_turn(sinks=[sink]):
        try:
            with span("tool:create_booking", kind="tool"):
                raise RuntimeError("SLOT_CONFLICT")
        except RuntimeError:
            pass
    obs = client.by_name("tool:create_booking")
    assert obs.updates["level"] == "ERROR"
    assert obs.updates["status_message"] == "RuntimeError: SLOT_CONFLICT"
    assert "level" not in client.by_name("turn").updates


def test_continued_trace_passes_parent_span_id() -> None:
    client = FakeClient()
    _run_turn(_sink(client), trace_id="c" * 32, parent_span_id="d" * 16)
    assert client.by_name("turn").start_kwargs["trace_context"] == {
        "trace_id": "c" * 32,
        "parent_span_id": "d" * 16,
    }


@pytest.mark.parametrize("fail_on", ["root", "child"])
def test_client_exceptions_do_not_affect_conversation(
    fail_on: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = FakeClient(fail_on=fail_on)
    sink = _sink(client)
    with caplog.at_level(logging.WARNING):
        trace_id = _run_turn(sink)  # must not raise
    assert len(trace_id) == 32
    assert "langfuse span start failed" in caplog.text
    assert sink._observations == {}  # nothing leaks across turns


def test_disabled_config_is_noop() -> None:
    client = FakeClient()
    sink = LangfuseSink(LangfuseConfig(enabled=False), client=client)
    _run_turn(sink)
    assert client.observations == []


def test_missing_keys_disable_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        sink = LangfuseSink(LangfuseConfig(enabled=True))
    assert sink.enabled is False
    assert "key missing" in caplog.text
    _run_turn(sink)


def test_close_flushes() -> None:
    client = FakeClient()
    sink = _sink(client)
    sink.close()
    assert client.flushed


def test_create_sinks_dual_write(tmp_path: Any) -> None:
    cfg = TraceConfig(sqlite_path=str(tmp_path / "t.db"), langfuse=LangfuseConfig(enabled=False))
    sinks = create_sinks(cfg)
    assert [type(s) for s in sinks] == [SqliteSink]
    cfg_on = cfg.model_copy(update={"langfuse": LangfuseConfig(enabled=True)})
    assert [type(s) for s in create_sinks(cfg_on)] == [SqliteSink, LangfuseSink]


# --- real SDK, exported to memory (no network) --------------------------------------


def test_real_sdk_exports_expected_otel_attributes() -> None:
    pytest.importorskip("langfuse")
    from langfuse import Langfuse, propagate_attributes
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-lf-test-{uuid.uuid4().hex}",  # unique: the SDK caches clients per key
        secret_key="sk-lf-test",
        host="http://127.0.0.1:9",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )
    sink = LangfuseSink(ENABLED, client=client, propagate_attributes=propagate_attributes)
    trace_id = _run_turn(sink)
    sink.close()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"turn", "extract_slots", "llm:mock", "tool:find_companions"}
    assert {format(s.context.trace_id, "032x") for s in spans.values()} == {trace_id}
    root_attrs = dict(spans["turn"].attributes or {})
    assert root_attrs["session.id"] == "sess-1"
    assert root_attrs["user.id"] == "u1"
    assert root_attrs["langfuse.observation.input"] == "约个双排"
    assert root_attrs["langfuse.observation.output"] == "为你找到 1 位陪玩"
    gen_attrs = dict(spans["llm:mock"].attributes or {})
    assert gen_attrs["langfuse.observation.type"] == "generation"
    assert gen_attrs["langfuse.observation.model.name"] == "deepseek-chat"
    assert spans["llm:mock"].parent is not None
    assert spans["llm:mock"].parent.span_id == spans["extract_slots"].context.span_id
