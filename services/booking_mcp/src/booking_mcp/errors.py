"""Business errors of the booking service; each carries a stable, client-visible code.

The MCP layer turns these into tool errors (``SLOT_CONFLICT``, ``NOT_FOUND``,
``FORBIDDEN``, ``INVALID_ARGUMENT``) that the agent maps back to Python exceptions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    SLOT_CONFLICT = "SLOT_CONFLICT"
    NOT_FOUND = "NOT_FOUND"
    FORBIDDEN = "FORBIDDEN"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INTERNAL = "INTERNAL"


#: JSON-RPC numeric codes: application errors live in the reserved -32000..-32099 range,
#: bad arguments reuse the standard "invalid params" code.
JSONRPC_CODES: dict[ErrorCode, int] = {
    ErrorCode.SLOT_CONFLICT: -32001,
    ErrorCode.NOT_FOUND: -32002,
    ErrorCode.FORBIDDEN: -32003,
    ErrorCode.INVALID_ARGUMENT: -32602,
    ErrorCode.INTERNAL: -32603,
}


class BookingError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code.value,
            "jsonrpc_code": JSONRPC_CODES[self.code],
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class SlotConflict(BookingError):
    """The companion is not free for the requested time (checked inside the write txn)."""

    code = ErrorCode.SLOT_CONFLICT


class NotFound(BookingError):
    code = ErrorCode.NOT_FOUND


class Forbidden(BookingError):
    """The caller does not own the resource."""

    code = ErrorCode.FORBIDDEN


class InvalidArgument(BookingError):
    code = ErrorCode.INVALID_ARGUMENT


class InvalidTransition(InvalidArgument):
    """A booking status change the state machine does not allow."""
