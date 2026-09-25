"""Chat page (G1: login-protected shell; G2 adds the SSE API)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from rift_web.auth import current_user, login_redirect
from rift_web.pages import page

router = APIRouter()


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Response:
    user = current_user(request)
    if user is None:
        return login_redirect(request)
    return page(request, "chat.html", {"active": "chat"})
