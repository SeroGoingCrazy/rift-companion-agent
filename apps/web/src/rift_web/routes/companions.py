"""Companion list page: filters (mode, rank, role, gender) and cards with free time."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from rift_agent.mcp_clients import ToolError
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType
from rift_web.auth import current_user, login_redirect
from rift_web.pages import page
from rift_web.services import WebServices

router = APIRouter()
WEEKDAYS = "一二三四五六日"


def _choice[E: (GameMode, Rank, Role, Gender)](enum: type[E], raw: str | None) -> E | None:
    if not raw:
        return None
    try:
        return enum(raw)
    except ValueError:
        return None


def fmt_slot(slot: dict[str, Any]) -> str:
    start = datetime.fromisoformat(slot["start"])
    end = datetime.fromisoformat(slot["end"])
    day = f"{start.month}/{start.day} 周{WEEKDAYS[start.weekday()]}"
    end_text = f"{end:%H:%M}" if end.date() == start.date() or end.hour else "24:00"
    return f"{day} {start:%H:%M}–{end_text}"


@router.get("/companions", response_class=HTMLResponse)
async def companions_page(
    request: Request,
    mode: str | None = None,
    rank: str | None = None,
    role: str | None = None,
    gender: str | None = None,
) -> Response:
    if current_user(request) is None:
        return login_redirect(request)
    services: WebServices = request.app.state.services
    domain = services.domain
    filters = {
        "game_mode": _choice(GameMode, mode),
        "rank_requirement": _choice(Rank, rank),
        "role": _choice(Role, role),
        "companion_gender": _choice(Gender, gender),
    }
    error = ""
    companions: list[dict[str, Any]] = []
    try:
        companions = await services.booking.browse_companions(days=3, **filters)
    except ToolError as exc:
        error = f"陪玩师列表暂时加载不了：{exc.message}"

    for c in companions:
        c["slot_texts"] = [fmt_slot(s) for s in c["free_slots"][:6]]
        c["rank_label"] = domain.rank_label(Rank(c["rank"]))
        c["role_labels"] = [domain.roles[Role(r)].label for r in c["roles"]]
        c["mode_labels"] = [domain.game_modes[GameMode(m)].label for m in c["modes"]]
        c["gender_label"] = domain.genders[Gender(c["gender"])].label
        c["price_labels"] = [
            (domain.service_types[ServiceType(st)].label, price)
            for st, price in c["prices"].items()
        ]
        c["chat_link"] = "/chat?q=" + quote(f"我想约{c['name']}")

    options = {
        "mode": [(m.value, cfg.label) for m, cfg in domain.game_modes.items()],
        "rank": [(r.key.value, r.label) for r in domain.ranks],
        "role": [(r.value, cfg.label) for r, cfg in domain.roles.items()],
        "gender": [(g.value, cfg.label) for g, cfg in domain.genders.items()],
    }
    selected = {
        "mode": filters["game_mode"].value if filters["game_mode"] else "",
        "rank": filters["rank_requirement"].value if filters["rank_requirement"] else "",
        "role": filters["role"].value if filters["role"] else "",
        "gender": filters["companion_gender"].value if filters["companion_gender"] else "",
    }
    return page(
        request,
        "companions.html",
        {
            "active": "companions",
            "companions": companions,
            "options": options,
            "selected": selected,
            "error": error,
        },
    )
