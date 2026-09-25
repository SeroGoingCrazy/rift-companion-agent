"""Chat page and API.

``POST /api/chat`` answers with Server-Sent Events:

- ``token``      ``{"text": "..."}`` reply text, in order
- ``candidates`` ``{"candidates": [...]}`` companion cards to pick from
- ``confirm``    order / cancellation summary awaiting a yes/no
- ``booked``     the created order
- ``done``       turn metadata (phase, reply_type, trace_id, ...)
- ``error``      ``{"message": "..."}`` — failures never leak into the reply text

The browser tab owns a random ``session_id``; the agent thread is namespaced by the
logged-in user (``u<uid>-<session_id>``) so a session id alone reveals nothing.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from rift_web.auth import WebUser, current_user, login_redirect
from rift_web.pages import page
from rift_web.services import WebServices

logger = logging.getLogger(__name__)
router = APIRouter()

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


class ChatIn(BaseModel):
    session_id: str = Field(min_length=6, max_length=64)
    message: str = Field(min_length=1, max_length=500)


def thread_id(user: WebUser, session_id: str) -> str:
    return f"u{user.user_id}-{session_id}"


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _unauthorized() -> JSONResponse:
    return JSONResponse({"error": "请先登录"}, status_code=401)


@router.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request) -> Response:
    if current_user(request) is None:
        return login_redirect(request)
    return page(request, "chat.html", {"active": "chat"})


@router.post("/api/chat")
async def chat_api(request: Request, body: ChatIn) -> Response:
    user = current_user(request)
    if user is None:
        return _unauthorized()
    if not SESSION_ID.fullmatch(body.session_id):
        return JSONResponse({"error": "invalid session_id"}, status_code=422)
    services: WebServices = request.app.state.services
    thread = thread_id(user, body.session_id)
    text = body.message.strip()

    async def stream() -> AsyncIterator[str]:
        try:
            async for event in services.agent.run_turn_events(thread, user.user_id, text):
                if event.type == "token":
                    yield sse("token", {"text": event.data})
                elif event.type == "candidates":
                    yield sse("candidates", {"candidates": event.data})
                elif event.type == "error":
                    yield sse("error", {"message": event.data.get("message", "出错了")})
                else:
                    yield sse(event.type, event.data)
        except Exception:
            logger.exception("chat stream failed for %s", thread)
            yield sse("error", {"message": "抱歉，服务出了点问题，请稍后再试。"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/chat/history")
async def chat_history(request: Request, session_id: str) -> Response:
    user = current_user(request)
    if user is None:
        return _unauthorized()
    if not SESSION_ID.fullmatch(session_id):
        return JSONResponse({"error": "invalid session_id"}, status_code=422)
    services: WebServices = request.app.state.services
    messages = await services.agent.history(thread_id(user, session_id))
    return JSONResponse({"messages": messages})
