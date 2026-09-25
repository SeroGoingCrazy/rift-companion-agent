"""Order management: ``manage`` (list / choose / ask to cancel) and ``manage_confirm``.

Understanding here is rule based (and cheap): cancel words, ordinals ("第二个"), order
numbers ("#12"), companion names and day words pick the booking; the refund comes from
``cancel_booking(dry_run=true)`` and nothing is cancelled before an explicit yes. Text
that is not about orders while in MANAGE is sent back to ``classify``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from langchain_core.runnables import RunnableConfig

from rift_agent.deps import get_deps
from rift_agent.graph.state import AgentState, Phase
from rift_agent.graph.tracing import traced_node
from rift_agent.mcp_clients import McpUnavailable, ToolError
from rift_agent.replies.templates import STATUS_LABELS
from rift_domain.enums import PendingAction
from rift_domain.timeparse import cn_to_int

ACTIVE = ("pending_payment", "confirmed")
CANCEL_WORDS = ("取消", "退掉", "退订", "退单", "不要了")
LIST_WORDS = ("订单", "哪些单", "我的单", "约了什么", "预约记录")
NO_WORDS = ("不", "别", "算了", "先不", "保留", "再想想", "no")
YES_WORDS = ("确认", "确定", "是", "好", "对", "可以", "行", "yes", "ok", "没问题")
#: Unambiguous enough to count even inside a question.
STRONG_YES = ("确认", "确定")
CANCEL = PendingAction.AWAIT_CONFIRM_CANCEL.value

_ORDINAL = re.compile(r"第\s*([一二两三四五六七八九十\d]+)\s*(?:个|单|笔|条)?")
_ORDER_ID = re.compile(r"(?:#|订单号?\s*|单号\s*)(\d+)|(\d+)\s*号(?:订单|单)")
_DAY_WORDS = {"今天": 0, "明天": 1, "后天": 2}


def parse_yes_no(text: str) -> bool | None:
    """True / False / None (unclear), conservative because cancelling is destructive:
    "no" words win, and a question ("好？") only counts with an explicit 确认/确定."""
    t = text.strip().lower()
    if any(w in t for w in NO_WORDS):
        return False
    if "?" in t or "？" in t:
        return True if any(w in t for w in STRONG_YES) else None
    if any(w in t for w in YES_WORDS):
        return True
    return None


def pick_booking(
    text: str, bookings: list[dict[str, Any]], shown: list[int], now: datetime
) -> dict[str, Any] | None:
    """The booking ``text`` refers to, or None."""
    by_id = {b["booking_id"]: b for b in bookings}
    m = _ORDER_ID.search(text)
    if m:
        return by_id.get(int(m.group(1) or m.group(2)))
    m = _ORDINAL.search(text)
    if m:
        n = cn_to_int(m.group(1)) if not m.group(1).isdigit() else int(m.group(1))
        order = [i for i in shown if i in by_id] or [b["booking_id"] for b in bookings]
        if n is not None and 1 <= n <= len(order):
            return by_id[order[n - 1]]
        return None
    named = [b for b in bookings if b["companion_name"] in text]
    if len(named) == 1:
        return named[0]
    for word, offset in _DAY_WORDS.items():
        if word in text:
            day = (now + timedelta(days=offset)).date()
            on_day = [b for b in bookings if datetime.fromisoformat(b["start_time"]).date() == day]
            if len(on_day) == 1:
                return on_day[0]
    return None


def _order_view(b: dict[str, Any]) -> dict[str, Any]:
    keys = ("booking_id", "companion_name", "game_mode", "start_time", "hours", "total", "status")
    return {k: b.get(k) for k in keys}


@traced_node("manage")
async def manage(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    text = state.get("user_input", "")
    memo: dict[str, Any] = dict(state.get("manage") or {})
    came_from_manage = state.get("phase") == Phase.MANAGE.value and state.get("intent") is None
    wants_cancel = any(w in text for w in CANCEL_WORDS)
    wants_list = any(w in text for w in LIST_WORDS)

    try:
        bookings = await deps.require_booking().list_my_bookings(int(state.get("user_id") or 0))
    except McpUnavailable:
        return {"action": "unavailable", "reply_type": "service_unavailable"}

    active = [b for b in bookings if b["status"] in ACTIVE]
    target = pick_booking(text, bookings, memo.get("shown", []), deps.now())
    choosing = memo.get("awaiting") == "choose_cancel" and target is not None

    if wants_cancel or choosing:
        if not active:
            return _reply("manage_empty", {}, memo={}, phase=Phase.IDLE)
        if target is None and len(active) == 1:
            target = active[0]
        if target is None:
            shown = [b["booking_id"] for b in active]
            return _reply(
                "manage_which",
                {"bookings": [_order_view(b) for b in active]},
                memo={"shown": shown, "awaiting": "choose_cancel"},
                phase=Phase.MANAGE,
            )
        if target["status"] not in ACTIVE:
            status = STATUS_LABELS.get(target["status"], target["status"])
            message = f"订单 #{target['booking_id']} {status}，不能再取消"
            return _reply("manage_error", {"message": message}, memo={}, phase=Phase.IDLE)
        try:
            quote = await deps.require_booking().cancel_booking(
                int(state.get("user_id") or 0), target["booking_id"], dry_run=True
            )
        except ToolError as exc:
            return _reply("manage_error", {"message": exc.message}, memo={}, phase=Phase.IDLE)
        facts = cancel_facts(quote)
        memo = {"target": target["booking_id"], "refund": quote}
        update = _reply("cancel_confirm", facts, memo=memo, phase=Phase.MANAGE)
        update.update(pending_action=CANCEL, ui={"confirm": facts})
        return update

    if wants_list or not came_from_manage:
        shown = sorted(bookings, key=lambda b: (b["status"] not in ACTIVE, b["start_time"]))[:10]
        return _reply(
            "manage_list",
            {"bookings": [_order_view(b) for b in shown], "active": len(active)},
            memo={"shown": [b["booking_id"] for b in shown]},
            phase=Phase.MANAGE if bookings else Phase.IDLE,
        )

    # In MANAGE but not talking about orders: let the classifier route it.
    return {"action": "reclassify", "manage": {}, "phase": Phase.IDLE.value}


def cancel_facts(quote: dict[str, Any]) -> dict[str, Any]:
    b = quote["booking"]
    return {
        "kind": "cancel",
        "order": _order_view(b),
        "paid": quote["paid"],
        "refund_label": quote["refund_label"],
        "refund_amount": quote["refund_amount"],
        "policy_refund_amount": quote["policy_refund_amount"],
        "hours_before": quote["hours_before_start"],
    }


def _reply(
    reply_type: str, facts: dict[str, Any], *, memo: dict[str, Any], phase: Phase
) -> dict[str, Any]:
    return {
        "action": reply_type,
        "reply_type": reply_type,
        "facts": facts,
        "manage": memo,
        "phase": phase.value,
        "pending_action": None,
    }


def route_manage(state: AgentState) -> str:
    if state.get("error"):
        return "render"
    return "classify" if state.get("action") == "reclassify" else "render"


@traced_node("manage_confirm")
async def manage_confirm(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    deps = get_deps(config)
    memo: dict[str, Any] = dict(state.get("manage") or {})
    booking_id = int(memo.get("target") or 0)
    answer = parse_yes_no(state.get("user_input", ""))
    order = (memo.get("refund") or {}).get("booking", {})

    if answer is None:
        facts = cancel_facts(memo["refund"])
        return {
            "action": "ask_again",
            "reply_type": "cancel_unclear",
            "facts": facts,
            "ui": {"confirm": facts},
        }
    if answer is False:
        return _reply("cancel_aborted", {"order": _order_view(order)}, memo={}, phase=Phase.IDLE)
    try:
        done = await deps.require_booking().cancel_booking(
            int(state.get("user_id") or 0), booking_id, dry_run=False
        )
    except ToolError as exc:
        return _reply("manage_error", {"message": exc.message}, memo={}, phase=Phase.IDLE)
    return _reply("cancelled", cancel_facts(done), memo={}, phase=Phase.IDLE)
