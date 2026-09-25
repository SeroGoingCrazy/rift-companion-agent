"""G5: shared layout guards for small screens (verified manually at 375px as well)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from rift_web.services import WebServices

Make = Callable[..., tuple[TestClient, WebServices]]


@pytest.fixture
def client(make_web: Make, web_login: Callable[..., None]) -> TestClient:
    c, _ = make_web()
    web_login(c, "demo")
    return c


@pytest.mark.parametrize("path", ["/chat", "/bookings", "/companions"])
def test_pages_share_the_mobile_ready_shell(client: TestClient, path: str) -> None:
    html = client.get(path).text
    assert 'name="viewport" content="width=device-width, initial-scale=1' in html
    assert '<link rel="stylesheet" href="/static/style.css">' in html
    assert 'class="nav"' in html and "退出" in html


def test_css_guards_against_horizontal_scroll(client: TestClient) -> None:
    css = client.get("/static/style.css").text
    assert "overflow-x: hidden" in css  # page never scrolls sideways
    assert "overflow-wrap: anywhere" in css  # long tokens in chat bubbles wrap
    assert "minmax(min(100%," in css  # grids collapse to one column on phones
    assert "100dvh" in css  # chat fits mobile browser chrome
    assert "@media (max-width: 480px)" in css
    assert "prefers-reduced-motion" in css


def test_loading_and_error_states_exist(client: TestClient) -> None:
    chat_js = client.get("/static/chat.js").text
    assert "typing" in chat_js and "网络异常" in chat_js
    bookings_js = client.get("/static/bookings.js").text
    assert "loading" in bookings_js and "订单加载失败" in bookings_js
