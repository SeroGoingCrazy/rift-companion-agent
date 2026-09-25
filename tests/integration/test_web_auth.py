"""G1: app factory, nickname login with signed cookie, protected pages, CORS."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rift_web.auth import COOKIE_NAME, Auth, WebUser, safe_next
from rift_web.services import WebServices

Make = Callable[..., tuple[TestClient, WebServices]]


def test_logged_out_chat_redirects_to_login(make_web: Make) -> None:
    client, _ = make_web()
    resp = client.get("/chat")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/chat"
    assert client.get("/").headers["location"] == "/login"


def test_login_page_renders(make_web: Make) -> None:
    client, _ = make_web()
    resp = client.get("/login?next=/chat")
    assert resp.status_code == 200
    assert 'name="nickname"' in resp.text and 'value="/chat"' in resp.text


def test_login_sets_signed_cookie_and_opens_chat(make_web: Make) -> None:
    client, services = make_web()
    resp = client.post("/login", data={"nickname": "  小明 ", "next": "/chat"})
    assert resp.status_code == 303 and resp.headers["location"] == "/chat"
    cookie = resp.cookies[COOKIE_NAME]
    user = services.auth.read(cookie)
    assert user is not None and user.nickname == "小明"
    set_cookie = resp.headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "samesite=lax" in set_cookie

    page = client.get("/chat")
    assert page.status_code == 200
    assert "小明" in page.text and 'id="composer"' in page.text
    assert client.get("/").headers["location"] == "/chat"
    assert client.get("/login").headers["location"] == "/chat"


def test_same_nickname_same_user(make_web: Make) -> None:
    client, services = make_web()
    ids = []
    for _ in range(2):
        resp = client.post("/login", data={"nickname": "阿杰"})
        user = services.auth.read(resp.cookies[COOKIE_NAME])
        assert user is not None
        ids.append(user.user_id)
    assert ids[0] == ids[1]
    # The seeded demo user keeps id 1.
    resp = client.post("/login", data={"nickname": "demo"})
    demo = services.auth.read(resp.cookies[COOKIE_NAME])
    assert demo is not None and demo.user_id == 1


@pytest.mark.parametrize("cookie", ["garbage", "e30.aaaa.bbbb"])
def test_tampered_cookie_is_rejected(make_web: Make, cookie: str) -> None:
    client, _ = make_web()
    client.cookies.set(COOKIE_NAME, cookie)
    assert client.get("/chat").status_code == 303


def test_cookie_from_another_secret_is_rejected(make_web: Make) -> None:
    client, _ = make_web()
    forged = Auth("attacker-secret").issue(WebUser(1, "demo"))
    client.cookies.set(COOKIE_NAME, forged)
    assert client.get("/chat").status_code == 303


def test_modified_payload_is_rejected(make_web: Make) -> None:
    client, services = make_web()
    token = services.auth.issue(WebUser(5, "eve"))
    payload, rest = token.split(".", 1)
    client.cookies.set(COOKIE_NAME, payload[:-2] + "AA." + rest)
    assert client.get("/chat").status_code == 303


def test_expired_cookie_is_rejected() -> None:
    auth = Auth("s", max_age_s=-1)
    assert auth.read(auth.issue(WebUser(1, "a"))) is None
    assert auth.read(None) is None


def test_logout_clears_cookie(make_web: Make, web_login: Callable[..., None]) -> None:
    client, _ = make_web()
    web_login(client)
    resp = client.post("/logout")
    assert resp.status_code == 303 and resp.headers["location"] == "/login"
    assert COOKIE_NAME in resp.headers["set-cookie"]
    client.cookies.clear()
    assert client.get("/chat").status_code == 303


@pytest.mark.parametrize(("nickname", "status"), [("", 400), ("   ", 400), ("x" * 17, 400)])
def test_invalid_nicknames(make_web: Make, nickname: str, status: int) -> None:
    client, _ = make_web()
    resp = client.post("/login", data={"nickname": nickname})
    assert resp.status_code == status
    assert "昵称" in resp.text


def test_booking_service_down_shows_error(make_web: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    from rift_agent.mcp_clients import McpUnavailable

    client, services = make_web()

    async def down(nickname: str) -> Any:
        raise McpUnavailable("booking unavailable")

    monkeypatch.setattr(services.booking, "ensure_user", down)
    resp = client.post("/login", data={"nickname": "a"})
    assert resp.status_code == 503 and "服务暂时不可用" in resp.text


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("/bookings", "/bookings"),
        ("/chat?x=1", "/chat?x=1"),
        ("//evil.com", "/chat"),
        ("https://evil.com", "/chat"),
        ("/\\evil.com", "/chat"),
        ("", "/chat"),
        (None, "/chat"),
    ],
)
def test_safe_next(target: str | None, expected: str) -> None:
    assert safe_next(target) == expected


def test_login_does_not_open_redirect(make_web: Make) -> None:
    client, _ = make_web()
    resp = client.post("/login", data={"nickname": "a", "next": "//evil.com/x"})
    assert resp.headers["location"] == "/chat"


def test_cors_only_local_origins(make_web: Make) -> None:
    client, _ = make_web()
    headers = {"Access-Control-Request-Method": "POST"}
    ok = client.options("/login", headers={**headers, "Origin": "http://localhost:8000"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8000"
    bad = client.options("/login", headers={**headers, "Origin": "https://evil.com"})
    assert "access-control-allow-origin" not in bad.headers


def test_static_css_served(make_web: Make) -> None:
    client, _ = make_web()
    resp = client.get("/static/style.css")
    assert resp.status_code == 200 and "--accent" in resp.text
