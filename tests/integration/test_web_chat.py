"""G2: chat API as Server-Sent Events, session namespacing, history, chat page assets."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from rift_agent.api import Agent, AgentEvent
from rift_agent.graph.builder import build_graph
from rift_web.services import WebServices

if TYPE_CHECKING:
    from tests.integration.conftest import AgentHarness, FakeChatAgent

Make = Callable[..., tuple[TestClient, WebServices]]
SID = "tab123456"


def parse_sse(body: str) -> list[tuple[str, Any]]:
    events = []
    for frame in body.strip().split("\n\n"):
        event, data = "message", ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data += line[6:]
        events.append((event, json.loads(data) if data else None))
    return events


CARDS = [{"companion_id": 3, "name": "阿狸酱", "rank": "diamond", "roles": ["mid"]}]
SCRIPT = {
    "找人": [
        AgentEvent("token", "为你找到"),
        AgentEvent("token", "这些陪玩师："),
        AgentEvent("candidates", CARDS),
        AgentEvent("done", {"phase": "BOOKING", "reply_type": "candidates"}),
    ],
    "选阿狸酱": [
        AgentEvent("token", "请确认订单"),
        AgentEvent("confirm", {"kind": "booking", "companion": "阿狸酱", "total": "120.00"}),
        AgentEvent("done", {"awaiting_confirmation": True}),
    ],
    "坏了": [AgentEvent("error", {"message": "抱歉，服务出了点问题，请稍后再试。", "detail": "x"})],
}


@pytest.fixture
def agent(fake_chat_agent: type[FakeChatAgent]) -> FakeChatAgent:
    return fake_chat_agent(SCRIPT)


@pytest.fixture
def client(make_web: Make, agent: FakeChatAgent, web_login: Callable[..., None]) -> TestClient:
    c, _ = make_web(agent)
    web_login(c, "小明")
    return c


def _chat(client: TestClient, message: str, session_id: str = SID) -> list[tuple[str, Any]]:
    resp = client.post("/api/chat", json={"session_id": session_id, "message": message})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    return parse_sse(resp.text)


def test_stream_tokens_candidates_done(client: TestClient, agent: FakeChatAgent) -> None:
    events = _chat(client, "找人")
    assert [e for e, _ in events] == ["token", "token", "candidates", "done"]
    assert "".join(d["text"] for e, d in events if e == "token") == "为你找到这些陪玩师："
    assert events[2][1] == {"candidates": CARDS}
    session, user_id, text = agent.calls[0]
    assert session == f"u{user_id}-{SID}" and text == "找人"


def test_confirm_event(client: TestClient) -> None:
    events = _chat(client, "选阿狸酱")
    confirm = next(d for e, d in events if e == "confirm")
    assert confirm["companion"] == "阿狸酱" and confirm["total"] == "120.00"


def test_errors_are_events_not_text(client: TestClient) -> None:
    events = _chat(client, "坏了")
    assert events == [("error", {"message": "抱歉，服务出了点问题，请稍后再试。"})]
    assert all("[ERROR]" not in json.dumps(d, ensure_ascii=False) for _, d in events)


def test_crash_mid_stream_becomes_error_event(
    make_web: Make, web_login: Callable[..., None]
) -> None:
    class Crashing:
        async def run_turn_events(self, *_: Any) -> AsyncIterator[AgentEvent]:
            yield AgentEvent("token", "半句")
            raise RuntimeError("boom")

        async def history(self, session_id: str) -> list[Any]:
            return []

    c, _ = make_web(Crashing())
    web_login(c)
    events = parse_sse(c.post("/api/chat", json={"session_id": SID, "message": "hi"}).text)
    assert events[0] == ("token", {"text": "半句"})
    assert events[-1][0] == "error"


def test_requires_login(make_web: Make) -> None:
    c, _ = make_web()
    resp = c.post("/api/chat", json={"session_id": SID, "message": "hi"})
    assert resp.status_code == 401
    assert c.get(f"/api/chat/history?session_id={SID}").status_code == 401


@pytest.mark.parametrize("sid", ["short", "has space x", "../../etc", "x" * 65])
def test_bad_session_ids(client: TestClient, sid: str) -> None:
    resp = client.post("/api/chat", json={"session_id": sid, "message": "hi"})
    assert resp.status_code == 422


def test_empty_message_rejected(client: TestClient) -> None:
    assert client.post("/api/chat", json={"session_id": SID, "message": ""}).status_code == 422


def test_sessions_are_per_user(
    make_web: Make, agent: FakeChatAgent, web_login: Callable[..., None]
) -> None:
    c, _ = make_web(agent)
    web_login(c, "alice")
    _chat(c, "hi")
    web_login(c, "bob")
    _chat(c, "hi")
    threads = [call[0] for call in agent.calls]
    assert threads[0] != threads[1]
    assert all(t.endswith(SID) for t in threads)


def test_history(client: TestClient, agent: FakeChatAgent) -> None:
    _chat(client, "找人")
    session = agent.calls[0][0]
    agent.histories[session] = [{"role": "user", "content": "找人"}]
    resp = client.get(f"/api/chat/history?session_id={SID}")
    assert resp.json() == {"messages": [{"role": "user", "content": "找人"}]}
    assert client.get("/api/chat/history?session_id=bad").status_code == 422


def test_chat_page_and_script(client: TestClient) -> None:
    page = client.get("/chat").text
    assert '<script src="/static/chat.js"' in page and 'id="new-session"' in page
    js = client.get("/static/chat.js").text
    for hook in ('"candidates"', '"confirm"', '"booked"', '"error"', "sessionStorage", "选这位"):
        assert hook in js


def test_real_agent_behind_the_api(
    make_web: Make, make_agent: Callable[..., AgentHarness], web_login: Callable[..., None]
) -> None:
    table = {
        "约个大乱斗明晚八点两小时": {
            "turn_intent": "booking",
            "delta": {"game_mode": "aram", "start_time_expr": "明晚八点", "duration_hours": 2},
            "confirmation": "none",
        }
    }
    harness = make_agent(table)
    c, _ = make_web(Agent(build_graph(MemorySaver()), harness.deps))
    web_login(c, "demo")
    events = _chat(c, "约个大乱斗明晚八点两小时")
    kinds = [e for e, _ in events]
    assert kinds[0] == "token" and kinds[-1] == "done" and "candidates" in kinds
    cards = next(d for e, d in events if e == "candidates")["candidates"]
    assert 1 <= len(cards) <= 3 and all("name" in card for card in cards)
    done = events[-1][1]
    assert done["phase"] == "BOOKING" and done["reply_type"] == "candidates"
    history = c.get(f"/api/chat/history?session_id={SID}").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]
