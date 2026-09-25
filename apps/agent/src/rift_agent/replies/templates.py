"""Template replies: ``config/prompts/templates/replies.yaml`` keyed by ``reply_type``.

Templates are Jinja2 with the turn's ``facts`` plus a few helpers, and filters that turn
domain keys into Chinese labels (``mode`` / ``rank`` / ``role`` / ``service`` /
``gender`` / ``field``), format times (``dt``) and money (``money``). Rendering is
deterministic and never calls a model.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, Undefined

from rift_agent.prompts import PROMPTS_DIR
from rift_domain.config import DomainConfig
from rift_domain.enums import GameMode, Gender, Rank, Role, ServiceType, SlotField
from rift_domain.slots import ANY, BookingState

DEFAULT_TEMPLATES = PROMPTS_DIR / "templates" / "replies.yaml"
WEEKDAYS = "一二三四五六日"

FIELD_LABELS: dict[str, str] = {
    SlotField.GAME_MODE: "游戏模式",
    SlotField.START_TIME: "开始时间",
    SlotField.DURATION_HOURS: "时长",
    SlotField.RANK_REQUIREMENT: "段位要求",
    SlotField.ROLE_PREFERENCE: "位置",
    SlotField.SERVICE_TYPE: "服务类型",
    SlotField.COMPANION_GENDER: "性别",
    SlotField.VOICE_REQUIRED: "开麦",
    SlotField.BUDGET_PER_HOUR: "预算",
    SlotField.STYLE_PREFERENCE: "风格",
    SlotField.COMPANION_NAME: "指定陪玩师",
}


class TemplateError(KeyError):
    """Unknown reply type."""


def fmt_dt(value: datetime | str | None) -> str:
    if value is None or value == "":
        return ""
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return f"{dt.month}月{dt.day}日（周{WEEKDAYS[dt.weekday()]}）{dt:%H:%M}"


def fmt_money(value: Any) -> str:
    return f"{Decimal(str(value)):.2f}"


def fmt_hours(value: Any) -> str:
    d = Decimal(str(value)).normalize()
    return format(d, "f")


class ReplyRenderer:
    def __init__(self, domain: DomainConfig, path: str | Path = DEFAULT_TEMPLATES) -> None:
        self.domain = domain
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        self.env = Environment(
            # Optional facts are simply absent; lenient undefined renders them as empty.
            undefined=Undefined,
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            keep_trailing_newline=False,
        )
        self.env.filters.update(
            mode=self.mode_label,
            rank=self.rank_label,
            role=self.role_label,
            service=self.service_label,
            gender=self.gender_label,
            field=lambda f: FIELD_LABELS.get(str(f), str(f)),
            dt=fmt_dt,
            money=fmt_money,
            hours=fmt_hours,
        )
        self.templates = {k: self.env.from_string(str(v)) for k, v in raw.items()}

    # --- labels ------------------------------------------------------------------------

    def mode_label(self, value: str | None) -> str:
        return self.domain.game_modes[GameMode(value)].label if value else ""

    def rank_label(self, value: str | None) -> str:
        if not value:
            return ""
        return "不限" if value == ANY else self.domain.rank_label(Rank(value))

    def role_label(self, value: str) -> str:
        return self.domain.roles[Role(value)].label

    def service_label(self, value: str | None) -> str:
        return self.domain.service_types[ServiceType(value)].label if value else ""

    def gender_label(self, value: str | None) -> str:
        if not value:
            return ""
        return "不限" if value == ANY else self.domain.genders[Gender(value)].label

    # --- summaries ---------------------------------------------------------------------

    def summary(self, booking: BookingState) -> str:
        """One line of what has been collected so far, e.g. ``单双排 · 10月2日（周五）20:00``."""
        parts: list[str] = []
        b = booking
        if b.game_mode:
            parts.append(self.mode_label(b.game_mode))
        if b.start_time:
            parts.append(fmt_dt(b.start_time))
        if b.duration_hours:
            parts.append(f"{fmt_hours(b.duration_hours)} 小时")
        if isinstance(b.rank_requirement, str) and b.rank_requirement != ANY:
            parts.append(f"{self.rank_label(b.rank_requirement)}起")
        if isinstance(b.role_preference, tuple):
            parts.append("/".join(self.role_label(r) for r in b.role_preference))
        if isinstance(b.companion_gender, str) and b.companion_gender != ANY:
            parts.append(self.gender_label(b.companion_gender))
        if b.voice_required is True:
            parts.append("开麦")
        if isinstance(b.budget_per_hour, int | float):
            parts.append(f"每小时 ≤{fmt_hours(b.budget_per_hour)} 元")
        if isinstance(b.style_preference, str) and b.style_preference != ANY:
            parts.append(f"「{b.style_preference}」")
        if b.service_type:
            parts.append(self.service_label(b.service_type))
        return " · ".join(parts)

    # --- rendering ---------------------------------------------------------------------

    def has(self, reply_type: str) -> bool:
        return reply_type in self.templates

    def render(
        self, reply_type: str, facts: dict[str, Any], booking: BookingState | None = None
    ) -> str:
        template = self.templates.get(reply_type)
        if template is None:
            raise TemplateError(reply_type)
        b = booking or BookingState()
        text = template.render(
            **facts,
            summary=self.summary(b),
            booking=b.model_dump(mode="json"),
            domain=self.domain,
        )
        lines = [line.rstrip() for line in text.strip().splitlines()]
        return "\n".join(lines)
