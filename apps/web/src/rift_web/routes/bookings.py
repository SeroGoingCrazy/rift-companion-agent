"""My orders: page + JSON API (all data through booking-mcp, like the agent)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from rift_agent.mcp_clients import (
    ForbiddenError,
    InvalidArgumentError,
    McpUnavailable,
    NotFoundError,
    ToolError,
)
from rift_web.auth import current_user, login_redirect
from rift_web.pages import page
from rift_web.services import WebServices

router = APIRouter()

_STATUS = {
    ForbiddenError: 403,
    NotFoundError: 404,
    InvalidArgumentError: 400,
    McpUnavailable: 503,
}


class CancelIn(BaseModel):
    dry_run: bool = True


def tool_error(exc: ToolError) -> JSONResponse:
    status = next((code for cls, code in _STATUS.items() if isinstance(exc, cls)), 502)
    return JSONResponse({"error": exc.message, "code": exc.code}, status_code=status)


def _unauthorized() -> JSONResponse:
    return JSONResponse({"error": "请先登录"}, status_code=401)


@router.get("/bookings", response_class=HTMLResponse)
async def bookings_page(request: Request) -> Response:
    if current_user(request) is None:
        return login_redirect(request)
    return page(request, "bookings.html", {"active": "bookings"})


@router.get("/api/bookings")
async def list_bookings(request: Request) -> Response:
    user = current_user(request)
    if user is None:
        return _unauthorized()
    services: WebServices = request.app.state.services
    try:
        bookings = await services.booking.list_my_bookings(user.user_id)
    except ToolError as exc:
        return tool_error(exc)
    bookings.sort(
        key=lambda b: (b["status"] not in ("pending_payment", "confirmed"), b["start_time"])
    )
    return JSONResponse({"bookings": bookings})


@router.post("/api/bookings/{booking_id}/pay")
async def pay(request: Request, booking_id: int) -> Response:
    user = current_user(request)
    if user is None:
        return _unauthorized()
    services: WebServices = request.app.state.services
    try:
        out: dict[str, Any] = await services.booking.pay_booking(user.user_id, booking_id)
    except ToolError as exc:
        return tool_error(exc)
    return JSONResponse({"booking": out})


@router.post("/api/bookings/{booking_id}/cancel")
async def cancel(request: Request, booking_id: int, body: CancelIn) -> Response:
    user = current_user(request)
    if user is None:
        return _unauthorized()
    services: WebServices = request.app.state.services
    try:
        out = await services.booking.cancel_booking(user.user_id, booking_id, dry_run=body.dry_run)
    except ToolError as exc:
        return tool_error(exc)
    return JSONResponse(out)
