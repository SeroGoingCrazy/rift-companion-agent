"""LangGraph state of one conversation (``thread_id = session_id``).

Everything in ``AgentState`` is JSON-native (str / int / float / bool / list / dict /
None) so the SQLite checkpointer can store it without custom serializers; the booking
slots live under ``booking`` as ``BookingState.model_dump(mode="json")`` and are rebuilt
with ``booking_state(state)``.

Fields fall in two groups:

- **conversation**: ``phase``, ``booking``, ``pending_action``, ``messages``, ``manage``,
  ``candidate_cards``, ``turn_idx`` — carried across turns;
- **turn scratch**: ``user_input``, ``intent``, ``extraction``, ``diff``, ``action``,
  ``reply_type``, ``facts``, ``reply``, ``ui``, ``error`` — reset at the start of every
  turn by ``route_phase`` (see ``TURN_RESET``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, TypedDict

from rift_domain.slots import BookingState


class Phase(StrEnum):
    IDLE = "IDLE"
    BOOKING = "BOOKING"
    MANAGE = "MANAGE"


class ChatMessage(TypedDict):
    role: str
    content: str


class AgentState(TypedDict, total=False):
    # --- identity / conversation --------------------------------------------------------
    session_id: str
    user_id: int
    turn_idx: int
    phase: str
    #: ``BookingState`` as JSON.
    booking: dict[str, Any]
    #: ``PendingAction`` value while waiting for a yes/no, else None.
    pending_action: str | None
    #: Recent user/assistant messages (window of ``history_turns`` turns).
    messages: list[ChatMessage]
    #: Candidate cards last shown (full tool output rows, for the UI and selection).
    candidate_cards: list[dict[str, Any]]
    #: Order management: listed bookings, the one being cancelled, its refund quote.
    manage: dict[str, Any]

    # --- turn scratch -------------------------------------------------------------------
    user_input: str
    #: classify result for IDLE turns: booking | consult | manage | other.
    intent: str | None
    #: ``SlotExtraction`` as JSON, or None when extraction failed.
    extraction: dict[str, Any] | None
    #: What merging changed: {"changed": [...], "time_reason": str | None, ...}.
    diff: dict[str, Any]
    #: Next step chosen by ``decide`` (ask_missing / find / quote / book / ...).
    action: str | None
    reply_type: str
    #: Facts the reply template interpolates.
    facts: dict[str, Any]
    reply: str
    #: Structured payload for rich clients: {"candidates": [...]} / {"confirm": {...}}.
    ui: dict[str, Any] | None
    #: Set by ``traced_node`` when a node crashed; routes straight to ``render``.
    error: str | None


#: Scratch values every turn starts from.
TURN_RESET: dict[str, Any] = {
    "intent": None,
    "extraction": None,
    "diff": {},
    "action": None,
    "reply_type": "",
    "facts": {},
    "reply": "",
    "ui": None,
    "error": None,
}


def initial_state(session_id: str, user_id: int) -> AgentState:
    return {
        "session_id": session_id,
        "user_id": user_id,
        "turn_idx": 0,
        "phase": Phase.IDLE.value,
        "booking": dump_booking(BookingState()),
        "pending_action": None,
        "messages": [],
        "candidate_cards": [],
        "manage": {},
    }


def booking_state(state: AgentState) -> BookingState:
    return BookingState.model_validate(state.get("booking") or {})


def dump_booking(booking: BookingState) -> dict[str, Any]:
    data: dict[str, Any] = booking.model_dump(mode="json")
    return data


def phase_of(state: AgentState) -> Phase:
    return Phase(state.get("phase") or Phase.IDLE.value)


def append_message(
    messages: list[ChatMessage], role: str, content: str, *, keep_turns: int
) -> list[ChatMessage]:
    """New list with the message appended, trimmed to the last ``keep_turns`` turns."""
    out = [*messages, ChatMessage(role=role, content=content)]
    limit = max(keep_turns, 0) * 2
    return out[-limit:] if limit else []
