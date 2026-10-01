"""Protocol layer of the L2 score: is the raw output a valid ``SlotExtraction``?

Checks, in order: JSON syntax, a JSON object at the top, then the strict C2 schema
(extra keys, missing keys, enum values, JSON types). The output must be the bare JSON
object, exactly what the online extractor parses; no Markdown fences are stripped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from rift_domain.slots import SlotExtraction, parse_extraction

ProtocolError = Literal["empty", "json", "not_object", "extra_key", "missing_key", "enum", "type"]

#: When one output breaks several rules, the report counts it under the first of these.
_PRIORITY: tuple[ProtocolError, ...] = ("extra_key", "missing_key", "enum", "type")
_ENUM_TYPES = frozenset({"enum", "literal_error"})


@dataclass(frozen=True)
class ProtocolResult:
    extraction: SlotExtraction | None
    error: ProtocolError | None = None
    messages: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.extraction is not None


def _kind(error_type: str, value: Any) -> ProtocolError:
    if error_type == "extra_forbidden":
        return "extra_key"
    if error_type == "missing":
        return "missing_key"
    # An unknown string is a bad enum value; ``1`` against ``bool | "any"`` is a type error.
    if error_type in _ENUM_TYPES and isinstance(value, str):
        return "enum"
    return "type"


def classify_validation(exc: ValidationError) -> tuple[ProtocolError, tuple[str, ...]]:
    """Main error kind plus one readable message per schema violation."""
    kinds: set[ProtocolError] = set()
    messages: list[str] = []
    for e in exc.errors():
        kinds.add(_kind(e["type"], e.get("input")))
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        messages.append(f"{loc}: {e['msg']}")
    main: ProtocolError = next((k for k in _PRIORITY if k in kinds), "type")
    return main, tuple(messages)


def check_protocol(raw: str) -> ProtocolResult:
    text = raw.strip()
    if not text:
        return ProtocolResult(None, "empty", ("empty output",))
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return ProtocolResult(None, "json", (f"invalid JSON: {exc.msg} at {exc.pos}",))
    if not isinstance(data, dict):
        return ProtocolResult(None, "not_object", (f"top level is {type(data).__name__}",))
    try:
        return ProtocolResult(parse_extraction(text))
    except ValidationError as exc:
        kind, messages = classify_validation(exc)
        return ProtocolResult(None, kind, messages)
