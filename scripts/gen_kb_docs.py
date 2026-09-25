"""Generate the knowledge base (Markdown) from the single sources of truth.

- ``kb/platform_rules/*.md``    <- ``config/domain.yaml`` (billing, refund, late arrival,
                                   violations, companion levels, matching rules)
- ``kb/modes_and_ranks/*.md``   <- ``config/domain.yaml`` (+ companion counts from the seed)
- ``kb/companion_profiles/*.md`` <- companion seed data, one page per companion

Every number is interpolated from config or computed with ``rift_domain`` (prices via
``pricing.quote``, refund examples via ``refund.refund``), so the knowledge base cannot
disagree with the code.

Usage::

    uv run python scripts/gen_kb_docs.py            # writes kb/
    uv run python scripts/gen_kb_docs.py --out /tmp/kb
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from booking_mcp.seed import DEFAULT_SEED, CompanionSeed, generate_companions
from rift_domain.config import DomainConfig, load_domain_config
from rift_domain.enums import GameMode, ServiceType, SlotField
from rift_domain.pricing import quote
from rift_domain.refund import refund

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = Path(__file__).resolve().parent / "kb_templates"
COLLECTIONS = ("platform_rules", "modes_and_ranks", "companion_profiles")

#: Chinese names of slot fields as they appear in the docs.
FIELD_LABELS: dict[SlotField, str] = {
    SlotField.GAME_MODE: "游戏模式",
    SlotField.START_TIME: "开始时间",
    SlotField.DURATION_HOURS: "时长",
    SlotField.RANK_REQUIREMENT: "段位要求",
    SlotField.ROLE_PREFERENCE: "位置偏好",
    SlotField.SERVICE_TYPE: "服务类型",
    SlotField.COMPANION_GENDER: "陪玩师性别",
    SlotField.VOICE_REQUIRED: "是否开麦",
    SlotField.BUDGET_PER_HOUR: "预算",
    SlotField.STYLE_PREFERENCE: "风格偏好",
    SlotField.COMPANION_NAME: "指定陪玩师",
}
RELAX_LABELS = {
    "time_window": "开始时间（前后 {h} 小时）",
    "companion_gender": "性别",
    "role_preference": "位置",
}

EXAMPLE_UNIT_PRICE = Decimal(60)
EXAMPLE_HOURS = Decimal(2)


@dataclass(frozen=True)
class Doc:
    collection: str
    name: str
    content: str

    def path(self, root: Path) -> Path:
        return root / self.collection / f"{self.name}.md"


def _num(value: Decimal | float | int) -> str:
    """Plain number: 1.0 -> 1, 1.20 -> 1.2, 144.00 -> 144."""
    d = Decimal(str(value))
    text = format(d.normalize(), "f")
    return text


def _money(value: Decimal) -> str:
    return f"{value:.2f}"


def _percent(ratio: Decimal) -> str:
    return f"{_num(ratio * 100)}%"


def _mode_labels(domain: DomainConfig, modes: tuple[GameMode, ...]) -> list[str]:
    return [domain.game_modes[m].label for m in modes]


# --- contexts ------------------------------------------------------------------------------


def billing_context(domain: DomainConfig) -> dict[str, Any]:
    ranked = [m for m in GameMode if domain.is_ranked(m)]
    casual = next(m for m in GameMode if not domain.is_ranked(m))
    quotes = []
    for st, cfg in domain.service_types.items():
        q = quote(EXAMPLE_UNIT_PRICE, st, EXAMPLE_HOURS, domain)
        quotes.append(
            {
                "label": cfg.label,
                "multiplier": _num(cfg.multiplier),
                "total": _money(q.total),
                "effective": _money(q.effective_hourly),
            }
        )
    return {
        "service_types": [
            {"label": c.label, "aliases": c.aliases, "multiplier": _num(c.multiplier)}
            for c in domain.service_types.values()
        ],
        "ranked_modes": _mode_labels(domain, tuple(ranked)),
        "default_ranked_service": domain.service_types[
            domain.default_service_type(ranked[0])
        ].label,
        "default_casual_service": domain.service_types[domain.default_service_type(casual)].label,
        "duration": {
            "min": _num(domain.duration.min_hours),
            "max": _num(domain.duration.max_hours),
            "step": _num(domain.duration.step_hours),
        },
        "example": {
            "unit_price": _num(EXAMPLE_UNIT_PRICE),
            "hours": _num(EXAMPLE_HOURS),
            "quotes": quotes,
        },
    }


@dataclass(frozen=True)
class _ExampleBooking:
    start_time: datetime
    total: Decimal


def refund_context(domain: DomainConfig) -> dict[str, Any]:
    tiers = domain.refund.tiers
    rows = []
    for i, t in enumerate(tiers):
        lo = _num(t.min_hours_before)
        if i == 0:
            rng = f"≥ {lo} 小时"
        elif t.min_hours_before == 0:
            rng = f"< {_num(tiers[i - 1].min_hours_before)} 小时（含已开局）"
        else:
            rng = f"{lo}–{_num(tiers[i - 1].min_hours_before)} 小时"
        rows.append({"range": rng, "label": t.label, "percent": _percent(t.ratio), "min_hours": lo})

    total = quote(EXAMPLE_UNIT_PRICE, ServiceType.CLIMB, EXAMPLE_HOURS, domain).total
    start = datetime(2026, 1, 3, 20, 0)
    booking = _ExampleBooking(start, total)
    # One example inside each tier (bounds +1h, the last tier 1h before start).
    hours = [t.min_hours_before + 1 for t in tiers[:-1]] + [Decimal(1)]
    cases = []
    for h in hours:
        r = refund(booking, start - timedelta(hours=float(h)), domain)
        cases.append({"hours_before": _num(h), "label": r.label, "amount": _money(r.amount)})
    return {
        "tiers": rows,
        "example": {"total": _money(total), "start": "晚上 8 点", "cases": cases},
    }


def late_context(domain: DomainConfig) -> dict[str, Any]:
    late = domain.policies.late_arrival
    example_late = late.grace_minutes + 10
    over = example_late - late.grace_minutes
    return {
        "late": {
            "grace_minutes": late.grace_minutes,
            "makeup": _num(late.makeup_minutes_per_late_minute),
            "full_refund_after_minutes": late.full_refund_after_minutes,
            "example_late": example_late,
            "example_makeup": _num(over * late.makeup_minutes_per_late_minute),
        }
    }


def levels_context(domain: DomainConfig) -> dict[str, Any]:
    levels = domain.policies.companion_levels
    rows = []
    for i, lv in enumerate(levels):
        if i == 0:
            req = f"≥ {_num(lv.min_rating)}"
        elif lv.min_rating == 0:
            req = f"< {_num(levels[i - 1].min_rating)}"
        else:
            req = f"{_num(lv.min_rating)}–{_num(levels[i - 1].min_rating)}"
        rows.append({"name": lv.name, "requirement": req, "perks": lv.perks})
    return {"levels": rows, "rating_max": _num(Decimal(str(domain.matching.rating_max)))}


def matching_context(domain: DomainConfig) -> dict[str, Any]:
    band = domain.rules.rank_band
    example_from = domain.rank_order[len(domain.rank_order) // 2 + 1]
    example_to = domain.rank_offset(example_from, band.max_tier_gap)
    window = _num(domain.relaxation.time_window_hours)
    return {
        "required": [FIELD_LABELS[f] for f in domain.rules.required_fields],
        "conditional": [
            {
                "modes": _mode_labels(domain, c.when_modes),
                "fields": [FIELD_LABELS[f] for f in c.fields],
            }
            for c in domain.rules.conditional_required
        ],
        "not_applicable": [
            {
                "modes": _mode_labels(domain, c.when_modes),
                "fields": [FIELD_LABELS[f] for f in c.fields],
            }
            for c in domain.rules.not_applicable
        ],
        "band": {
            "modes": _mode_labels(domain, band.modes),
            "gap": band.max_tier_gap,
            "example_from": domain.rank_label(example_from),
            "example_range": f"{domain.rank_label(example_from)}到{domain.rank_label(example_to)}",
        },
        "top_k": domain.matching.top_k,
        "relax_steps": [RELAX_LABELS[s.value].format(h=window) for s in domain.relaxation.order],
        "never_relax": [FIELD_LABELS[f] for f in domain.relaxation.never_relax],
    }


def modes_context(domain: DomainConfig, companions: list[CompanionSeed]) -> dict[str, Any]:
    modes = []
    for mode, cfg in domain.game_modes.items():
        modes.append(
            {
                "label": cfg.label,
                "aliases": [a for a in cfg.aliases if a != cfg.label] or [cfg.label],
                "players": cfg.players,
                "ranked": cfg.ranked,
                "default_service": domain.service_types[cfg.default_service_type].label,
                "description": cfg.description,
                "not_applicable": [FIELD_LABELS[f] for f in domain.not_applicable_for(mode)],
                "companion_count": sum(mode in c.modes for c in companions),
            }
        )
    return {"modes": modes}


def ranks_context(domain: DomainConfig) -> dict[str, Any]:
    band = domain.rules.rank_band
    ranks = []
    for r in domain.ranks:
        top = domain.rank_offset(r.key, band.max_tier_gap)
        span = r.label if top == r.key else f"{r.label}–{domain.rank_label(top)}"
        ranks.append({"label": r.label, "aliases": r.aliases, "band": span})
    return {
        "ranks": ranks,
        "band_modes": _mode_labels(domain, band.modes),
        "gap": band.max_tier_gap,
    }


def roles_context(domain: DomainConfig, companions: list[CompanionSeed]) -> dict[str, Any]:
    return {
        "roles": [
            {
                "label": cfg.label,
                "aliases": cfg.aliases,
                "companion_count": sum(role in c.roles for c in companions),
            }
            for role, cfg in domain.roles.items()
        ]
    }


def companion_context(domain: DomainConfig, c: CompanionSeed) -> dict[str, Any]:
    return {
        "c": {
            "name": c.name,
            "gender": domain.genders[c.gender].label,
            "rank": domain.rank_label(c.rank),
            "level": domain.policies.level_for(c.rating).name,
            "rating": _num(Decimal(str(c.rating))),
            "roles": [domain.roles[r].label for r in c.roles],
            "modes": _mode_labels(domain, c.modes),
            "voice": c.voice,
            "tags": c.tags,
            "unit_price": _money(c.hourly_price),
            "prices": [
                {
                    "label": domain.service_types[st].label,
                    "effective": _money(quote(c.hourly_price, st, 1, domain).effective_hourly),
                }
                for st in c.service_types
            ],
            "bio": c.bio,
        }
    }


# --- rendering -------------------------------------------------------------------------------


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )


def build_docs(domain: DomainConfig, companions: list[CompanionSeed]) -> list[Doc]:
    env = _env()

    def render(template: str, ctx: dict[str, Any]) -> str:
        text = env.get_template(template).render(**ctx)
        return "\n".join(line.rstrip() for line in text.strip().splitlines()) + "\n"

    pages: list[tuple[str, str, dict[str, Any]]] = [
        ("platform_rules", "billing", billing_context(domain)),
        ("platform_rules", "refund", refund_context(domain)),
        ("platform_rules", "late_arrival", late_context(domain)),
        ("platform_rules", "violations", {"violations": domain.policies.violations}),
        ("platform_rules", "companion_levels", levels_context(domain)),
        ("platform_rules", "matching", matching_context(domain)),
        ("modes_and_ranks", "game_modes", modes_context(domain, companions)),
        ("modes_and_ranks", "ranks", ranks_context(domain)),
        ("modes_and_ranks", "roles", roles_context(domain, companions)),
    ]
    docs = [
        Doc(collection, name, render(f"{collection}/{name}.md.j2", ctx))
        for collection, name, ctx in pages
    ]
    for i, c in enumerate(companions, start=1):
        docs.append(
            Doc(
                "companion_profiles",
                f"companion_{i:02d}",
                render("companion_profiles/companion.md.j2", companion_context(domain, c)),
            )
        )
    return docs


def write_docs(docs: list[Doc], out: Path, *, clean: bool = True) -> list[Path]:
    if clean:
        for collection in COLLECTIONS:
            shutil.rmtree(out / collection, ignore_errors=True)
    paths = []
    for doc in docs:
        path = doc.path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(doc.content, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def generate(domain: DomainConfig, out: Path, *, seed: int = DEFAULT_SEED) -> list[Path]:
    return write_docs(build_docs(domain, generate_companions(domain, seed)), out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default=str(REPO_ROOT / "config" / "domain.yaml"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "kb")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    paths = generate(load_domain_config(args.domain), args.out, seed=args.seed)
    counts = {c: sum(p.parent.name == c for p in paths) for c in COLLECTIONS}
    print(
        f"wrote {len(paths)} docs to {args.out}: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
