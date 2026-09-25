import json
from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from rift_domain.enums import (
    Confirmation,
    GameMode,
    Rank,
    Role,
    SlotField,
    TurnIntent,
)
from rift_domain.slots import (
    ANY,
    BookingState,
    SlotDelta,
    SlotExtraction,
    extraction_json_schema,
    parse_extraction,
)


def _envelope(delta: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {"turn_intent": "booking", "delta": delta, "confirmation": "none", **kw}


FULL_DELTA = {
    "game_mode": "ranked_solo_duo",
    "start_time_expr": "明晚八点",
    "duration_hours": 2,
    "rank_requirement": "diamond",
    "role_preference": ["jungle"],
    "service_type": "climb",
    "companion_gender": "female",
    "voice_required": True,
    "budget_per_hour": 80,
    "style_preference": "温柔会聊天",
    "companion_name": "阿狸酱",
}


def test_spec_example_parses() -> None:
    ext = parse_extraction(_envelope(FULL_DELTA))
    assert ext.turn_intent is TurnIntent.BOOKING
    assert ext.confirmation is Confirmation.NONE
    assert ext.delta.game_mode is GameMode.RANKED_SOLO_DUO
    assert ext.delta.role_preference == [Role.JUNGLE]
    assert ext.delta.duration_hours == 2.0
    assert set(ext.delta.provided()) == set(FULL_DELTA)


def test_parse_from_json_text_and_bytes() -> None:
    text = json.dumps(_envelope({"game_mode": "aram"}), ensure_ascii=False)
    assert parse_extraction(text).delta.game_mode is GameMode.ARAM
    assert parse_extraction(text.encode()).delta.game_mode is GameMode.ARAM


# --- tri-state -----------------------------------------------------------------------


def test_absent_vs_null_vs_any_are_distinguishable() -> None:
    delta = parse_extraction(_envelope({"start_time_expr": None, "rank_requirement": "any"})).delta
    provided = delta.provided()
    assert provided == {"start_time_expr": None, "rank_requirement": ANY}
    assert "game_mode" not in delta.model_fields_set  # absent
    assert delta.game_mode is None  # ...even though the attribute reads None
    assert delta.start_time_expr is None and "start_time_expr" in delta.model_fields_set


def test_empty_delta() -> None:
    delta = parse_extraction(_envelope({})).delta
    assert delta.is_empty()
    assert delta.provided() == {}


@pytest.mark.parametrize(
    "field",
    [
        "rank_requirement",
        "role_preference",
        "companion_gender",
        "voice_required",
        "budget_per_hour",
        "style_preference",
        "companion_name",
    ],
)
def test_any_accepted_on_preference_fields(field: str) -> None:
    assert parse_extraction(_envelope({field: "any"})).delta.provided() == {field: ANY}


@pytest.mark.parametrize("field", ["game_mode", "duration_hours", "service_type"])
def test_any_rejected_where_meaningless(field: str) -> None:
    with pytest.raises(ValidationError):
        parse_extraction(_envelope({field: "any"}))


# --- strict validation -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "loc"),
    [
        ({"game_mode": "tft"}, "game_mode"),
        ({"rank_requirement": "bronze_1"}, "rank_requirement"),
        ({"role_preference": ["jungle", "roam"]}, "role_preference"),
        ({"companion_gender": "other"}, "companion_gender"),
        ({"duration_hours": 0.3}, "duration_hours"),
        ({"duration_hours": 0}, "duration_hours"),
        ({"duration_hours": -1}, "duration_hours"),
        ({"duration_hours": "2"}, "duration_hours"),
        ({"voice_required": 1}, "voice_required"),
        ({"voice_required": "true"}, "voice_required"),
        ({"budget_per_hour": 0}, "budget_per_hour"),
        ({"budget_per_hour": "80"}, "budget_per_hour"),
        ({"role_preference": []}, "role_preference"),
        ({"role_preference": ["mid", "mid"]}, "role_preference"),
        ({"role_preference": "jungle"}, "role_preference"),
        ({"style_preference": "   "}, "style_preference"),
        ({"favourite_champion": "Ahri"}, "favourite_champion"),
    ],
)
def test_invalid_delta_rejected(delta: dict[str, Any], loc: str) -> None:
    with pytest.raises(ValidationError) as info:
        parse_extraction(_envelope(delta))
    assert any(loc in map(str, e["loc"]) for e in info.value.errors())


@pytest.mark.parametrize(
    "envelope",
    [
        {"turn_intent": "booking", "delta": {}},  # missing confirmation
        {"turn_intent": "chitchat", "delta": {}, "confirmation": "none"},
        {"turn_intent": "booking", "delta": {}, "confirmation": "maybe"},
        {"turn_intent": "booking", "delta": {}, "confirmation": "none", "extra": 1},
        {"turn_intent": "booking", "confirmation": "none"},  # missing delta
    ],
)
def test_invalid_envelope_rejected(envelope: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        parse_extraction(envelope)


def test_valid_half_hour_durations() -> None:
    for value in (1, 1.5, 2.5, 8):
        assert parse_extraction(_envelope({"duration_hours": value})).delta.duration_hours == value


def test_models_are_frozen() -> None:
    ext = parse_extraction(_envelope({"game_mode": "aram"}))
    with pytest.raises(ValidationError):
        ext.delta.game_mode = GameMode.ARENA  # type: ignore[misc]


def test_python_construction_with_enums() -> None:
    delta = SlotDelta(game_mode=GameMode.ARENA, rank_requirement=ANY)
    ext = SlotExtraction(turn_intent=TurnIntent.CONSULT, delta=delta, confirmation=Confirmation.NO)
    assert ext.delta.provided() == {"game_mode": GameMode.ARENA, "rank_requirement": ANY}


# --- JSON schema -----------------------------------------------------------------------


def test_json_schema_is_strict_and_lists_enums() -> None:
    schema = extraction_json_schema()
    assert set(schema["required"]) == {"turn_intent", "delta", "confirmation"}
    assert schema["additionalProperties"] is False
    defs = schema["$defs"]
    assert defs["SlotDelta"]["additionalProperties"] is False
    assert "required" not in defs["SlotDelta"]  # every delta key is optional
    assert set(defs["GameMode"]["enum"]) == {m.value for m in GameMode}
    duration = defs["SlotDelta"]["properties"]["duration_hours"]["anyOf"][0]
    assert duration["multipleOf"] == 0.5
    json.dumps(schema)  # serializable


# --- booking state -------------------------------------------------------------------------


def test_booking_state_defaults_and_helpers() -> None:
    state = BookingState()
    assert not state.is_filled(SlotField.GAME_MODE)
    state = state.model_copy(
        update={"rank_requirement": ANY, "start_time": datetime(2026, 10, 2, 20)}
    )
    assert state.is_filled(SlotField.RANK_REQUIREMENT)
    assert state.slot(SlotField.START_TIME) == datetime(2026, 10, 2, 20)
    assert state.missing_fields == () and state.candidates == ()


def test_booking_state_round_trips_json() -> None:
    state = BookingState(
        game_mode=GameMode.RANKED_FLEX,
        rank_requirement=Rank.GOLD,
        role_preference=(Role.MID, Role.TOP),
        start_time=datetime(2026, 10, 2, 20),
    )
    assert BookingState.model_validate_json(state.model_dump_json()) == state
