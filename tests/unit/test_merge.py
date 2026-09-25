from datetime import datetime
from decimal import Decimal
from typing import Any

from rift_domain.enums import GameMode, Gender, PendingAction, Rank, Role, ServiceType
from rift_domain.merge import StateDiff, merge
from rift_domain.slots import ANY, BookingState, Candidate, Quote, SlotDelta, parse_extraction
from rift_domain.timeparse import Reason

NOW = datetime(2026, 10, 1, 14, 0)


def delta(**fields: Any) -> SlotDelta:
    """Build a delta exactly as the extractor would emit it (JSON, strict)."""
    raw = {"turn_intent": "booking", "delta": fields, "confirmation": "none"}
    return parse_extraction(raw).delta


QUOTE = Quote(
    unit_price=Decimal("60"),
    service_type=ServiceType.CLIMB,
    multiplier=Decimal("1.2"),
    hours=Decimal("2"),
    effective_hourly=Decimal("72.00"),
    total=Decimal("144.00"),
)

BASE = BookingState(
    game_mode=GameMode.RANKED_SOLO_DUO,
    start_time=datetime(2026, 10, 2, 20),
    start_time_expr="明晚八点",
    duration_hours=2,
    rank_requirement=Rank.DIAMOND,
    role_preference=(Role.JUNGLE,),
    companion_gender=Gender.FEMALE,
)


# --- tri-state ---------------------------------------------------------------------------


def test_absent_keys_leave_state_untouched() -> None:
    state, diff = merge(BASE, delta(), NOW)
    assert state == BASE
    assert not diff
    assert diff == StateDiff()


def test_any_sets_any() -> None:
    state, diff = merge(BASE, delta(rank_requirement="any", companion_gender="any"), NOW)
    assert state.rank_requirement == ANY
    assert state.companion_gender == ANY
    assert diff.changed == {
        "rank_requirement": (Rank.DIAMOND, ANY),
        "companion_gender": (Gender.FEMALE, ANY),
    }


def test_null_clears() -> None:
    state, diff = merge(BASE, delta(role_preference=None, duration_hours=None), NOW)
    assert state.role_preference is None
    assert state.duration_hours is None
    assert set(diff.changed_fields) == {"role_preference", "duration_hours"}


def test_list_is_replaced_whole() -> None:
    state, diff = merge(BASE, delta(role_preference=["support", "adc"]), NOW)
    assert state.role_preference == (Role.SUPPORT, Role.ADC)
    assert diff.changed["role_preference"] == ((Role.JUNGLE,), (Role.SUPPORT, Role.ADC))


def test_value_overwrites_and_unchanged_values_are_not_in_diff() -> None:
    state, diff = merge(BASE, delta(game_mode="ranked_solo_duo", duration_hours=3), NOW)
    assert state.duration_hours == 3
    assert diff.changed_fields == ("duration_hours",)


def test_original_state_is_not_mutated() -> None:
    merge(BASE, delta(duration_hours=5), NOW)
    assert BASE.duration_hours == 2


def test_first_turn_from_empty_state() -> None:
    state, diff = merge(
        BookingState(),
        delta(
            game_mode="aram", start_time_expr="今晚八点", duration_hours=1.5, voice_required=True
        ),
        NOW,
    )
    assert state.game_mode is GameMode.ARAM
    assert state.start_time == datetime(2026, 10, 1, 20)
    assert state.voice_required is True
    assert set(diff.changed_fields) == {
        "game_mode",
        "start_time_expr",
        "start_time",
        "duration_hours",
        "voice_required",
    }


# --- time -----------------------------------------------------------------------------------


def test_time_expression_is_resolved() -> None:
    state, diff = merge(BookingState(), delta(start_time_expr="明晚八点"), NOW)
    assert state.start_time == datetime(2026, 10, 2, 20)
    assert diff.time_result is not None and diff.time_result.ok


def test_relative_time_uses_current_start_time() -> None:
    state, diff = merge(BASE, delta(start_time_expr="晚一小时"), NOW)
    assert state.start_time == datetime(2026, 10, 2, 21)
    assert state.start_time_expr == "晚一小时"
    assert diff.changed["start_time"] == (datetime(2026, 10, 2, 20), datetime(2026, 10, 2, 21))


def test_ambiguous_time_clears_start_time_and_reports_reason() -> None:
    state, diff = merge(BASE, delta(start_time_expr="周六下午"), NOW)
    assert state.start_time is None
    assert state.start_time_expr == "周六下午"
    assert diff.time_result is not None
    assert diff.time_result.ambiguous and diff.time_result.reason == Reason.AMBIGUOUS_HOUR


def test_past_time_is_rejected() -> None:
    state, diff = merge(BookingState(), delta(start_time_expr="今天上午十点"), NOW)
    assert state.start_time is None
    assert diff.time_result is not None and diff.time_result.reason == Reason.PAST


def test_pending_ambiguous_expression_is_completed_by_next_turn() -> None:
    state, _ = merge(BookingState(), delta(start_time_expr="明晚"), NOW)
    assert state.start_time is None
    state, diff = merge(state, delta(start_time_expr="八点"), NOW)
    assert state.start_time == datetime(2026, 10, 2, 20)
    assert state.start_time_expr == "明晚八点"
    assert diff.time_result is not None and diff.time_result.ok


def test_pending_expression_ignored_when_combination_fails() -> None:
    state, _ = merge(BookingState(), delta(start_time_expr="明晚"), NOW)
    state, _ = merge(state, delta(start_time_expr="后天晚上九点"), NOW)
    assert state.start_time == datetime(2026, 10, 3, 21)
    assert state.start_time_expr == "后天晚上九点"


def test_null_time_expression_clears_both_fields() -> None:
    state, diff = merge(BASE, delta(start_time_expr=None), NOW)
    assert state.start_time is None and state.start_time_expr is None
    assert diff.time_result is None
    assert set(diff.changed_fields) == {"start_time", "start_time_expr"}


# --- derived-state invalidation ----------------------------------------------------------------

CANDIDATES = (Candidate(companion_id=1, name="阿狸酱"), Candidate(companion_id=2, name="小鱼"))
QUOTED = BASE.model_copy(
    update={
        "candidates": CANDIDATES,
        "companion_name": "阿狸酱",
        "selected_companion_id": 1,
        "quote": QUOTE,
        "relaxations": ("time_window",),
        "pending_action": PendingAction.AWAIT_CONFIRM_BOOKING,
    }
)


def test_companion_change_clears_quote_and_selection() -> None:
    state, diff = merge(QUOTED, delta(companion_name="小鱼"), NOW)
    assert state.companion_name == "小鱼"
    assert state.quote is None
    assert state.selected_companion_id is None
    assert state.candidates == CANDIDATES  # the list the user picked from stays valid
    assert state.pending_action is None
    assert diff.invalidated == ("pending_action", "quote", "selected_companion_id")


def test_duration_change_clears_quote_and_candidates() -> None:
    state, diff = merge(QUOTED, delta(duration_hours=3), NOW)
    assert state.quote is None and state.candidates == () and state.relaxations == ()
    assert "candidates" in diff.invalidated and "quote" in diff.invalidated


def test_time_change_keeps_quote_but_drops_candidates() -> None:
    state, diff = merge(QUOTED, delta(start_time_expr="晚一小时"), NOW)
    assert state.quote == QUOTE
    assert state.candidates == ()
    assert "quote" not in diff.invalidated


def test_non_filter_change_keeps_candidates() -> None:
    state, diff = merge(QUOTED, delta(companion_name="阿狸酱"), NOW)  # same name
    assert state == QUOTED
    assert not diff


def test_nothing_to_invalidate_on_fresh_state() -> None:
    _, diff = merge(BASE, delta(companion_name="阿狸酱", duration_hours=3), NOW)
    assert diff.invalidated == ()
