"""Login / logout pages."""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from rift_agent.mcp_clients import InvalidArgumentError, ToolError
from rift_web.auth import COOKIE_NAME, WebUser, current_user, safe_next
from rift_web.pages import page
from rift_web.services import WebServices

router = APIRouter()


def _login_page(
    request: Request, *, error: str = "", nickname: str = "", status: int = 200
) -> Response:
    ctx = {"error": error, "nickname": nickname, "next": request.query_params.get("next", "")}
    return page(request, "login.html", ctx, status=status)


@router.get("/", include_in_schema=False)
async def root(request: Request) -> Response:
    return RedirectResponse("/chat" if current_user(request) else "/login", status_code=303)


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request) -> Response:
    if current_user(request):
        return RedirectResponse(safe_next(request.query_params.get("next")), status_code=303)
    return _login_page(request)


@router.post("/login")
async def login(request: Request, nickname: str = Form(""), next: str = Form("")) -> Response:
    services: WebServices = request.app.state.services
    nickname = nickname.strip()
    if not nickname or len(nickname) > 16:
        return _login_page(request, error="昵称需为 1–16 个字符", nickname=nickname, status=400)
    try:
        out = await services.booking.ensure_user(nickname)
    except InvalidArgumentError:
        return _login_page(request, error="昵称不合法，换一个试试", nickname=nickname, status=400)
    except ToolError:
        return _login_page(
            request, error="服务暂时不可用，请稍后再试", nickname=nickname, status=503
        )
    user = WebUser(int(out["user_id"]), str(out["nickname"]))
    response = RedirectResponse(safe_next(next), status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        services.auth.issue(user),
        max_age=services.auth.max_age_s,
        httponly=True,
        samesite="lax",
        secure=services.cookie_secure,
    )
    return response


@router.post("/logout")
async def logout() -> Response:
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response
