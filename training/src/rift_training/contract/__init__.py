"""Training-side view of the online slot contract (DEV_SPEC I1).

Nothing here re-implements the contract: the output schema comes from ``rift_domain.slots``
and the prompt from ``rift_agent.extractors.prompt``, so a training sample renders to the
same bytes the agent sends online. The only training-specific piece is ``render_target``,
the canonical assistant message a sample is trained on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from rift_agent.extractors.base import ExtractionContext
from rift_agent.extractors.prompt import PROMPT_FILE, build_messages
from rift_agent.prompts import PROMPTS_DIR, prompt_sha256
from rift_common.llm import Message
from rift_domain.slots import (
    ANY,
    SlotDelta,
    SlotExtraction,
    extraction_json_schema,
    parse_extraction,
)
from rift_training.evaluation.dataset import SlotSample

__all__ = [
    "ANY",
    "PROMPT_FILE",
    "PROMPT_PATH",
    "SlotDelta",
    "SlotExtraction",
    "context_from_sample",
    "extraction_json_schema",
    "parse_extraction",
    "prompt_hash",
    "render_conversation",
    "render_prompt",
    "render_target",
]

PROMPT_PATH = PROMPTS_DIR / PROMPT_FILE


def prompt_hash() -> str:
    """sha256 of ``config/prompts/slot_extract.txt``, recorded in every data card."""
    return prompt_sha256(PROMPT_FILE)


def context_from_sample(sample: SlotSample) -> ExtractionContext:
    """The context the agent's ``extract_slots`` node would build for this sample."""
    return ExtractionContext(
        now=sample.now,
        current_state=sample.state(),
        user_input=sample.user_input,
        candidates=tuple(sample.candidates),
        history=tuple({"role": m.role, "content": m.content} for m in sample.history),
        pending_confirmation=sample.pending_confirmation,
    )


def render_prompt(sample: SlotSample) -> list[Message]:
    """System + user messages, byte-identical to the online extractor's request."""
    return build_messages(context_from_sample(sample))


def _compact_number(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def render_target(expected: SlotExtraction | Mapping[str, Any]) -> str:
    """Canonical assistant output: schema field order, tri-state kept, ``2`` not ``2.0``.

    Absent delta keys stay absent and ``null`` stays ``null`` (``exclude_unset``), exactly
    like the agent's ``dump_extraction``.
    """
    extraction = expected if isinstance(expected, SlotExtraction) else parse_extraction(expected)
    data: dict[str, Any] = extraction.model_dump(mode="json", exclude_unset=True)
    data["delta"] = {k: _compact_number(v) for k, v in data["delta"].items()}
    return json.dumps(data, ensure_ascii=False)


def render_conversation(sample: SlotSample) -> list[Message]:
    """Prompt plus the expected answer as the final assistant message (one SFT example)."""
    return [
        *render_prompt(sample),
        {"role": "assistant", "content": render_target(sample.expected)},
    ]
