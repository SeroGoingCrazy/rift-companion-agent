"""MCP local timestamps must work with clients enforcing JSON Schema formats."""

from datetime import datetime

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from booking_mcp.tools.schemas import FreeSlot


def test_local_slot_matches_strict_client_schema() -> None:
    checker = FormatChecker()

    # jsonschema's optional RFC3339 dependency is not installed in every env.
    # A date-time format requires an offset; local booking clocks have none.
    @checker.checks("date-time")
    def has_timezone(value: object) -> bool:
        return not isinstance(value, str) or datetime.fromisoformat(value).tzinfo is not None

    slot = FreeSlot(start=datetime(2026, 10, 4, 20), end=datetime(2026, 10, 4, 22))
    validator = Draft202012Validator(
        FreeSlot.model_json_schema(mode="serialization"), format_checker=checker
    )
    validator.validate(slot.model_dump(mode="json"))
    with pytest.raises(ValidationError, match="does not match"):
        validator.validate({"start": "not-a-time", "end": slot.end.isoformat()})
