"""Nickname login with a signed cookie (no passwords in v1).

The cookie carries ``{"uid", "nick"}`` signed with ``itsdangerous``; a tampered or
foreign-key cookie simply reads as "not logged in".
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, URLSafeTimedSerializer

COOKIE_NAME = "rift_session"
MAX_AGE_S = 30 * 24 * 3600


@dataclass(frozen=True)
class WebUser:
    user_id: int
    nickname: str


class Auth:
    def __init__(self, secret: str, *, max_age_s: int = MAX_AGE_S) -> None:
        self._serializer = URLSafeTimedSerializer(secret, salt="rift-web-login")
        self.max_age_s = max_age_s

    def issue(self, user: WebUser) -> str:
        return self._serializer.dumps({"uid": user.user_id, "nick": user.nickname})

    def read(self, token: str | None) -> WebUser | None:
        if not token:
            return None
        try:
            data = self._serializer.loads(token, max_age=self.max_age_s)
        except BadSignature:  # includes SignatureExpired
            return None
        if not isinstance(data, dict):
            return None
        uid, nick = data.get("uid"), data.get("nick")
        if not isinstance(uid, int) or not isinstance(nick, str):
            return None
        return WebUser(uid, nick)


def current_user(request: Request) -> WebUser | None:
    auth: Auth = request.app.state.services.auth
    return auth.read(request.cookies.get(COOKIE_NAME))


def login_redirect(request: Request) -> RedirectResponse:
    target = request.url.path
    if request.url.query:
        target += f"?{request.url.query}"
    return RedirectResponse(f"/login?next={quote(target)}", status_code=303)


def safe_next(target: str | None, default: str = "/chat") -> str:
    """Only same-site relative paths; ``//evil.com`` and absolute URLs are refused."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return default
    return target
