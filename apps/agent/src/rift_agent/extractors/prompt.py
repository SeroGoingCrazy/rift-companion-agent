"""Prompt rendering shared by the online extractors and training data rendering.

The system message is the static ``config/prompts/slot_extract.txt`` (identical for every
call, so llama-server can cache its prefix); the user message is the context serialized
as JSON with sorted keys, so the same context always renders the same bytes.
"""

from __future__ import annotations

import json
from typing import Any

from rift_agent.extractors.base import ExtractionContext
from rift_agent.prompts import load_prompt
from rift_common.llm import Message
from rift_domain.slots import BookingState

PROMPT_FILE = "slot_extract.txt"
WEEKDAYS = "一二三四五六日"

#: Slots the model sees as ``current_state`` (derived fields are code-only).
STATE_FIELDS = (
    "game_mode",
    "start_time",
    "duration_hours",
    "rank_requirement",
    "role_preference",
    "service_type",
    "companion_gender",
    "voice_required",
    "budget_per_hour",
    "style_preference",
    "companion_name",
)


def fmt_time(value: Any) -> str:
    return f"{value:%Y-%m-%d %H:%M} 星期{WEEKDAYS[value.weekday()]}"


def state_view(state: BookingState) -> dict[str, Any]:
    """Filled slots only, JSON-friendly (tuples -> lists, datetimes -> readable text)."""
    view: dict[str, Any] = {}
    for name in STATE_FIELDS:
        value = getattr(state, name)
        if value is None:
            continue
        if name == "start_time":
            value = fmt_time(value)
        elif isinstance(value, tuple):
            value = list(value)
        view[name] = value
    return view


def render_context(ctx: ExtractionContext) -> str:
    payload = {
        "now": fmt_time(ctx.now),
        "current_state": state_view(ctx.current_state),
        "candidates": list(ctx.candidates),
        "pending_confirmation": ctx.pending_confirmation,
        "history": [{"role": m["role"], "content": m["content"]} for m in ctx.history],
        "user_input": ctx.user_input,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_messages(ctx: ExtractionContext) -> list[Message]:
    return [
        {"role": "system", "content": load_prompt(PROMPT_FILE)},
        {"role": "user", "content": render_context(ctx)},
    ]


def retry_message(errors: tuple[str, ...]) -> Message:
    listed = "；".join(errors[:5])
    return {
        "role": "user",
        "content": f"上一次输出不符合协议：{listed}。请重新输出，只输出一个合法的 JSON 对象。",
    }
