"""Generate the L3 RAG golden set from the single sources of truth.

Every question has a reference answer computed from ``config/domain.yaml`` and the
companion seed (prices via ``rift_domain.pricing.quote``, refunds via
``rift_domain.refund.refund``), plus the KB document that must be retrieved for it,
so the golden set cannot disagree with the generated knowledge base.

Writes ``eval/datasets/rag_golden.jsonl`` and ``rag_golden.jsonl.sha256`` (checked by
``eval/runners/run_rag_eval.py``; the runner also rebuilds the set in memory and warns
when it no longer matches ``domain.yaml``).

Usage::

    uv run python scripts/gen_rag_golden.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType

from booking_mcp.seed import DEFAULT_SEED, CompanionSeed, generate_companions
from rift_domain.config import DomainConfig, load_domain_config
from rift_domain.enums import GameMode, ServiceType
from rift_domain.pricing import quote
from rift_domain.refund import refund

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "eval" / "datasets" / "rag_golden.jsonl"
#: Companions (1-based, as in ``companion_XX.md``) that get profile questions.
PROFILE_COMPANIONS = (1, 5, 9, 13, 17, 21, 25)


def _load_kb_helpers() -> ModuleType:
    """``scripts/gen_kb_docs.py`` holds the number formatting and labels used in the KB."""
    name = "gen_kb_docs"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


kb = _load_kb_helpers()


@dataclass(frozen=True)
class GoldenQA:
    id: str
    collection: str
    category: str
    query: str
    reference_answer: str
    expected_sources: tuple[str, ...]

    def to_json(self) -> str:
        data = asdict(self)
        data["expected_sources"] = list(self.expected_sources)
        return json.dumps(data, ensure_ascii=False)


@dataclass(frozen=True)
class _Booking:
    start_time: datetime
    total: Decimal


def _qa(id: str, collection: str, doc: str, category: str, query: str, answer: str) -> GoldenQA:
    return GoldenQA(id, collection, category, query, answer, (f"{collection}/{doc}.md",))


def platform_rules(domain: DomainConfig) -> list[GoldenQA]:
    rules = "platform_rules"
    tiers = domain.refund.tiers
    top, last = tiers[0], tiers[-1]
    cutoff = kb._duration(tiers[-2].min_hours_before) if len(tiers) > 1 else "0 小时"
    start = datetime(2026, 10, 1, 20, 0)
    total = Decimal("200.00")
    booking = _Booking(start, total)
    ten_hours = refund(booking, start - timedelta(hours=10), domain)
    three_hours = refund(booking, start - timedelta(hours=3), domain)
    # Just inside the tier below the top one (e.g. 10 minutes before a 15-minute cutoff).
    below_top = top.min_hours_before * 2 / 3
    just_below = refund(booking, start - timedelta(hours=float(below_top)), domain)

    st = domain.service_types
    climb_quote = quote(80, ServiceType.CLIMB, 3, domain)
    dur = domain.duration
    late = domain.policies.late_arrival
    late_minutes = late.grace_minutes + 5
    makeup = (late_minutes - late.grace_minutes) * late.makeup_minutes_per_late_minute
    violations = {v.name: v for v in domain.policies.violations}
    top_level = domain.policies.companion_levels[0]
    relax = domain.relaxation
    relax_steps = " → ".join(
        kb.RELAX_LABELS[s].format(h=kb._num(relax.time_window_hours)) for s in relax.order
    )
    never = "、".join(kb.FIELD_LABELS[f] for f in relax.never_relax)
    required = "、".join(kb.FIELD_LABELS[f] for f in domain.rules.required_fields)
    ranked_modes = "、".join(
        domain.game_modes[m].label for m in GameMode if domain.game_modes[m].ranked
    )
    ranked_default = st[domain.game_modes[GameMode.RANKED_SOLO_DUO].default_service_type].label

    return [
        _qa(
            "refund_3h",
            rules,
            "refund",
            "refund",
            "开局前三小时取消能退多少钱？",
            f"开局前 3 小时取消{three_hours.label}，退款比例 {kb._percent(three_hours.ratio)}。",
        ),
        _qa(
            "refund_full",
            rules,
            "refund",
            "refund",
            f"提前 {kb._duration(top.min_hours_before)}取消订单能全额退款吗？",
            f"距开局 ≥ {kb._duration(top.min_hours_before)}取消{top.label}"
            f"（{kb._percent(top.ratio)}）。",
        ),
        _qa(
            "refund_none",
            rules,
            "refund",
            "refund",
            "马上就要开局了，现在取消还能退钱吗？",
            f"距开局不足 {cutoff}（含已开局）取消{last.label}。",
        ),
        _qa(
            "refund_amount",
            rules,
            "refund",
            "refund",
            f"一单 {kb._money(total)} 元的订单，提前 10 小时取消能退多少？",
            f"{ten_hours.label}，退 {kb._money(ten_hours.amount)} 元。",
        ),
        _qa(
            "refund_boundary",
            rules,
            "refund",
            "refund",
            f"恰好提前 {kb._duration(top.min_hours_before)}取消，按哪一档退款？",
            f"边界按更有利于用户的一档：恰好提前 {kb._duration(top.min_hours_before)}仍"
            f"{top.label}；提前 {kb._duration(below_top)}则{just_below.label}。",
        ),
        _qa(
            "refund_unpaid",
            rules,
            "refund",
            "refund",
            "还没付款的订单取消要扣钱吗？",
            "待支付订单取消不产生任何费用，也没有退款。",
        ),
        _qa(
            "billing_formula",
            rules,
            "billing",
            "billing",
            "陪玩的费用是怎么算的？",
            "总价 = 陪玩师基础单价 × 服务类型系数 × 时长（小时），保留两位小数。",
        ),
        _qa(
            "billing_coaching",
            rules,
            "billing",
            "billing",
            "教学服务比普通陪玩贵多少？",
            f"教学系数 {kb._num(st[ServiceType.COACHING].multiplier)}，"
            f"娱乐系数 {kb._num(st[ServiceType.CASUAL].multiplier)}。",
        ),
        _qa(
            "billing_compute",
            rules,
            "billing",
            "billing",
            "基础单价 80 元的陪玩师，上分 3 小时一共多少钱？",
            f"80 × {kb._num(st[ServiceType.CLIMB].multiplier)} × 3 = "
            f"{kb._money(climb_quote.total)} 元。",
        ),
        _qa(
            "billing_duration",
            rules,
            "billing",
            "billing",
            "可以只约 1.2 个小时吗？",
            f"不行：时长 {kb._num(dur.min_hours)}–{kb._num(dur.max_hours)} 小时，"
            f"步长 {kb._num(dur.step_hours)} 小时。",
        ),
        _qa(
            "billing_default_type",
            rules,
            "billing",
            "billing",
            "约单双排没说要什么服务，按哪种收费？",
            f"排位模式（{ranked_modes}）默认按「{ranked_default}」计费。",
        ),
        _qa(
            "late_grace",
            rules,
            "late_arrival",
            "late",
            "陪玩师晚几分钟上线不算迟到？",
            f"约定开局后 {late.grace_minutes} 分钟内上线视为准时。",
        ),
        _qa(
            "late_makeup",
            rules,
            "late_arrival",
            "late",
            f"陪玩师迟到了 {late_minutes} 分钟，会补偿多少时长？",
            f"超出宽限的每 1 分钟顺延 {kb._num(late.makeup_minutes_per_late_minute)} 分钟，"
            f"免费补时 {kb._num(makeup)} 分钟。",
        ),
        _qa(
            "late_no_show",
            rules,
            "late_arrival",
            "late",
            "陪玩师一直没上线怎么办？",
            f"迟到达到 {late.full_refund_after_minutes} 分钟或未上线，可直接取消并全额退款。",
        ),
        _qa(
            "violation_afk",
            rules,
            "violations",
            "violation",
            "陪玩师打游戏挂机、故意送人头怎么处理？",
            f"属于{violations['消极比赛'].name}：{violations['消极比赛'].penalty}。",
        ),
        _qa(
            "violation_private",
            rules,
            "violations",
            "violation",
            "陪玩师让我绕过平台私下转账可以吗？",
            f"属于{violations['私下交易'].name}：{violations['私下交易'].penalty}。",
        ),
        _qa(
            "level_top",
            rules,
            "companion_levels",
            "level",
            "评分要多少才能成为王牌陪玩？",
            f"评分 ≥ {kb._num(Decimal(str(top_level.min_rating)))} 为{top_level.name}，"
            f"权益：{top_level.perks}。",
        ),
        _qa(
            "level_price",
            rules,
            "companion_levels",
            "level",
            "陪玩师等级高是不是更贵？",
            "等级只影响展示顺序，不影响价格。",
        ),
        _qa(
            "matching_relax",
            rules,
            "matching",
            "matching",
            "找不到合适的陪玩师时，系统会怎么放宽条件？",
            f"依次放宽：{relax_steps}；{never}永远不会放宽。",
        ),
        _qa(
            "matching_required",
            rules,
            "matching",
            "matching",
            "预约陪玩需要提供哪些信息？",
            f"必须提供{required}；单双排、灵活组排还需要段位要求。",
        ),
    ]


def modes_and_ranks(domain: DomainConfig, companions: list[CompanionSeed]) -> list[GoldenQA]:
    col = "modes_and_ranks"
    modes = domain.game_modes
    ranks = domain.ranks
    gap = domain.rules.rank_band.max_tier_gap
    diamond = next(i for i, r in enumerate(ranks) if r.label == "钻石")
    band_top = ranks[min(diamond + gap, len(ranks) - 1)].label
    mayhem_count = sum(GameMode.ARAM_MAYHEM in c.modes for c in companions)
    support = next(r for r in domain.roles.values() if "软辅" in r.aliases)
    platinum = next(i for i, r in enumerate(ranks) if r.label == "铂金")
    emerald = next(i for i, r in enumerate(ranks) if r.label == "翡翠")
    higher = ranks[max(platinum, emerald)].label

    return [
        _qa(
            "mode_alias_flex",
            col,
            "game_modes",
            "mode",
            "「五排」指的是哪个模式？",
            f"五排是{modes[GameMode.RANKED_FLEX].label}，{modes[GameMode.RANKED_FLEX].players}。",
        ),
        _qa(
            "mode_arena_players",
            col,
            "game_modes",
            "mode",
            "斗魂竞技场是几个人一队？",
            f"{modes[GameMode.ARENA].label}：{modes[GameMode.ARENA].players}。",
        ),
        _qa(
            "mode_aram_rank",
            col,
            "game_modes",
            "mode",
            "约大乱斗需要选段位吗？",
            f"{modes[GameMode.ARAM].label}不区分段位要求和位置偏好。",
        ),
        _qa(
            "mode_mayhem_count",
            col,
            "game_modes",
            "mode",
            "能接海克斯大乱斗的陪玩师有几位？",
            f"可接{modes[GameMode.ARAM_MAYHEM].label}的陪玩师有 {mayhem_count} 位。",
        ),
        _qa(
            "rank_band_diamond",
            col,
            "ranks",
            "rank",
            "要求钻石的单双排，能匹配到哪些段位的陪玩师？",
            f"钻石–{band_top}（要求段位到其上 {gap} 个大段）。",
        ),
        _qa(
            "rank_order",
            col,
            "ranks",
            "rank",
            "翡翠和铂金哪个段位更高？",
            f"{higher}更高；顺序为 " + " < ".join(r.label for r in ranks) + "。",
        ),
        _qa(
            "role_alias_support",
            col,
            "roles",
            "role",
            "「软辅」是什么位置？",
            f"软辅指{support.label}位置。",
        ),
    ]


def companion_profiles(domain: DomainConfig, companions: list[CompanionSeed]) -> list[GoldenQA]:
    col = "companion_profiles"
    questions = []
    kinds = ("price", "roles", "rank", "voice", "modes", "level", "unit_price")
    for kind, index in zip(kinds, PROFILE_COMPANIONS, strict=True):
        c = companions[index - 1]
        doc = f"companion_{index:02d}"
        if kind == "price":
            st = c.service_types[0]
            label = domain.service_types[st].label
            price = quote(c.hourly_price, st, 1, domain).effective_hourly
            q, a = f"{c.name}{label}一小时多少钱？", f"{c.name}{label} {kb._money(price)} 元/小时。"
        elif kind == "roles":
            roles = "、".join(domain.roles[r].label for r in c.roles)
            q, a = f"{c.name}擅长打什么位置？", f"{c.name}擅长{roles}。"
        elif kind == "rank":
            q, a = f"{c.name}是什么段位？", f"{c.name}单双排段位{domain.rank_label(c.rank)}。"
        elif kind == "voice":
            voice = "可以开麦" if c.voice else "不开麦"
            q, a = f"{c.name}打游戏能开麦吗？", f"{c.name}{voice}。"
        elif kind == "modes":
            modes = "、".join(kb._mode_labels(domain, c.modes))
            q, a = f"{c.name}可以陪玩哪些模式？", f"{c.name}可接{modes}。"
        elif kind == "level":
            level = domain.policies.level_for(c.rating).name
            q = f"{c.name}的评分和等级是多少？"
            a = f"{c.name}评分 {kb._num(Decimal(str(c.rating)))}，{level}。"
        else:
            q, a = f"{c.name}的基础单价是多少？", f"基础单价 {kb._money(c.hourly_price)} 元/小时。"
        questions.append(_qa(f"companion_{index:02d}_{kind}", col, doc, "companion", q, a))
    return questions


def build_golden(domain: DomainConfig, companions: list[CompanionSeed]) -> list[GoldenQA]:
    return [
        *platform_rules(domain),
        *modes_and_ranks(domain, companions),
        *companion_profiles(domain, companions),
    ]


def render(items: list[GoldenQA]) -> str:
    return "".join(item.to_json() + "\n" for item in items)


def write(items: list[GoldenQA], out: Path) -> str:
    """Write the JSONL file and its ``.sha256`` sidecar; return the digest."""
    data = render(items).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sidecar = out.with_name(out.name + ".sha256")
    sidecar.write_bytes(f"{digest}  {out.name}\n".encode())  # LF on every OS
    return digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--domain", default=str(REPO_ROOT / "config" / "domain.yaml"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    domain = load_domain_config(args.domain)
    items = build_golden(domain, generate_companions(domain, seed=args.seed))
    digest = write(items, args.out)
    by_collection: dict[str, int] = {}
    for item in items:
        by_collection[item.collection] = by_collection.get(item.collection, 0) + 1
    counts = ", ".join(f"{k}={v}" for k, v in by_collection.items())
    print(f"wrote {len(items)} QA to {args.out} ({counts}); sha256 {digest[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
