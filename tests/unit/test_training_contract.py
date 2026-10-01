"""I1: training renders the same prompt bytes as the online extractor; data cards pin the prompt."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.runnables import RunnableConfig

from rift_agent.deps import AgentDeps
from rift_agent.extractors.base import ExtractionContext, ExtractionResult, SlotExtractor
from rift_agent.extractors.prompt import build_messages
from rift_agent.graph.nodes.extract import dump_extraction, extract_slots
from rift_agent.graph.state import dump_booking
from rift_agent.prompts import load_prompt
from rift_common.llm.mock import MockLLM
from rift_common.settings import LLMConfig
from rift_domain.config import DomainConfig
from rift_domain.enums import PendingAction
from rift_domain.slots import Candidate, parse_extraction
from rift_training.contract import (
    PROMPT_PATH,
    context_from_sample,
    prompt_hash,
    render_conversation,
    render_prompt,
    render_target,
)
from rift_training.data.card import (
    CARD_JSON,
    CARD_MD,
    build_card,
    load_card,
    prompt_mismatch,
    write_card,
)
from rift_training.evaluation.dataset import SlotSample, load_samples, parse_samples

DATASETS = Path(__file__).resolve().parents[2] / "eval" / "datasets"
ALL_SAMPLES = [
    *load_samples(DATASETS / "slot_main.jsonl"),
    *load_samples(DATASETS / "slot_holdout.jsonl"),
]


class CapturingExtractor(SlotExtractor):
    name = "capture"

    def __init__(self) -> None:
        self.seen: list[ExtractionContext] = []

    async def extract(
        self, ctx: ExtractionContext, *, feedback: ExtractionResult | None = None
    ) -> ExtractionResult:
        self.seen.append(ctx)
        return ExtractionResult.failed(source=self.name)


def agent_state_for(sample: SlotSample) -> dict[str, Any]:
    """The graph state the agent holds right before ``extract_slots`` for this sample."""
    booking = sample.state().model_copy(
        update={
            "candidates": tuple(
                Candidate(companion_id=i, name=name) for i, name in enumerate(sample.candidates)
            )
        }
    )
    messages = [*sample.history_messages(), {"role": "user", "content": sample.user_input}]
    return {
        "booking": dump_booking(booking),
        "messages": messages,
        "user_input": sample.user_input,
        "pending_action": PendingAction.AWAIT_CONFIRM_BOOKING.value
        if sample.pending_confirmation
        else None,
    }


async def online_messages(sample: SlotSample, domain: DomainConfig) -> list[Any]:
    extractor = CapturingExtractor()
    deps = AgentDeps(
        domain=domain,
        llm=MockLLM(LLMConfig(provider="mock", model="mock")),
        extractor=extractor,
        clock=lambda: sample.now,
    )
    config: RunnableConfig = {"configurable": {"deps": deps}}
    await extract_slots(agent_state_for(sample), config)  # type: ignore[arg-type]
    assert len(extractor.seen) == 1
    return build_messages(extractor.seen[0])


@pytest.mark.unit
async def test_training_prompt_is_byte_identical_to_the_online_node(domain: DomainConfig) -> None:
    assert len(ALL_SAMPLES) >= 80
    for sample in ALL_SAMPLES:
        online = await online_messages(sample, domain)
        training = render_prompt(sample)
        assert training == online, sample.id
        assert json.dumps(training, ensure_ascii=False) == json.dumps(online, ensure_ascii=False)


@pytest.mark.unit
def test_context_covers_every_input_field() -> None:
    sample = next(s for s in ALL_SAMPLES if s.history and s.candidates)
    ctx = context_from_sample(sample)
    assert ctx.now == sample.now
    assert ctx.user_input == sample.user_input
    assert ctx.candidates == tuple(sample.candidates)
    assert [m["content"] for m in ctx.history] == [m.content for m in sample.history]
    assert ctx.current_state == sample.state()


@pytest.mark.unit
def test_system_prompt_is_the_shared_prompt_file() -> None:
    sample = ALL_SAMPLES[0]
    system = render_prompt(sample)[0]
    assert system["role"] == "system"
    assert system["content"] == load_prompt("slot_extract.txt")
    assert system["content"] == PROMPT_PATH.read_text("utf-8").strip()


def _sample(expected: dict[str, Any], **kw: Any) -> SlotSample:
    row = {
        "id": "t-1",
        "category": "tri_state",
        "now": "2026-10-01 14:00",
        "user_input": "x",
        "expected": expected,
        **kw,
    }
    return parse_samples([json.dumps(row, ensure_ascii=False)])[0]


@pytest.mark.unit
def test_target_keeps_tri_state_and_schema_order() -> None:
    expected = {
        "confirmation": "none",
        "delta": {"companion_name": None, "style_preference": "any", "duration_hours": 2},
        "turn_intent": "booking",
    }
    target = render_target(expected)
    assert target == (
        '{"turn_intent": "booking", "delta": {"duration_hours": 2, '
        '"style_preference": "any", "companion_name": null}, "confirmation": "none"}'
    )
    assert parse_extraction(target) == parse_extraction(expected)
    assert '"budget_per_hour"' not in target  # absent stays absent


@pytest.mark.unit
def test_target_keeps_fractional_numbers_and_matches_the_agent_dump() -> None:
    expected = {"turn_intent": "booking", "delta": {"duration_hours": 2.5}, "confirmation": "none"}
    assert '"duration_hours": 2.5' in render_target(expected)
    extraction = parse_extraction(render_target(expected))
    assert dump_extraction(extraction) == dump_extraction(parse_extraction(expected))


@pytest.mark.unit
def test_every_l2_target_round_trips() -> None:
    for sample in ALL_SAMPLES:
        assert parse_extraction(render_target(sample.expected)) == sample.extraction(), sample.id


@pytest.mark.unit
def test_conversation_is_prompt_plus_assistant_target() -> None:
    sample = _sample({"turn_intent": "booking", "delta": {}, "confirmation": "none"})
    conversation = render_conversation(sample)
    assert conversation[:2] == render_prompt(sample)
    assert conversation[2] == {"role": "assistant", "content": render_target(sample.expected)}


# --- data card -----------------------------------------------------------------------------


@pytest.mark.unit
def test_card_records_prompt_hash_and_distributions(tmp_path: Path) -> None:
    samples = [
        _sample(
            {"turn_intent": "booking", "delta": {"style_preference": "any"}, "confirmation": "none"}
        ),
        _sample(
            {"turn_intent": "booking", "delta": {"style_preference": None}, "confirmation": "none"},
            id="t-2",
        ),
        _sample({"turn_intent": "consult", "delta": {}, "confirmation": "none"}, id="t-3"),
    ]
    card = build_card(
        samples,
        name="sft",
        version="v0.0-test",
        teacher={"provider": "mock", "model": "m"},
        spec="specs_v0.1",
        created_at=datetime(2026, 10, 1, 12, 0),
    )
    assert card.prompt_sha256 == prompt_hash()
    assert len(card.prompt_sha256) == 64
    assert card.num_samples == 3
    assert card.categories == {"tri_state": 3}
    assert card.turn_intents == {"booking": 2, "consult": 1}
    assert card.delta_fields == {"style_preference": {"value": 0, "any": 1, "null": 1}}
    assert prompt_mismatch(card) is None

    json_path, md_path = write_card(card, tmp_path / "v0.0")
    assert json_path.name == CARD_JSON and md_path.name == CARD_MD
    assert load_card(tmp_path / "v0.0") == card
    md = md_path.read_text("utf-8")
    assert card.prompt_sha256 in md
    assert "| `style_preference` | 0 | 1 | 1 |" in md
    assert b"\r\n" not in md_path.read_bytes()


@pytest.mark.unit
def test_prompt_mismatch_is_reported() -> None:
    card = build_card([], name="sft", version="v0")
    stale = card.model_copy(update={"prompt_sha256": "0" * 64})
    message = prompt_mismatch(stale)
    assert message is not None
    assert "slot_extract.txt" in message and "000000000000" in message
