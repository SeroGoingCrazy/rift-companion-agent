"""E3: knowledge-base docs are generated from domain.yaml and the companion seed."""

from __future__ import annotations

import copy
import importlib.util
import re
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from booking_mcp.seed import generate_companions
from rift_domain.config import DomainConfig, parse_domain_config

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW: dict[str, Any] = yaml.safe_load((REPO_ROOT / "config" / "domain.yaml").read_text("utf-8"))


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_kb_docs", REPO_ROOT / "scripts" / "gen_kb_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while executing
    spec.loader.exec_module(module)
    return module


gen = _load()


def _docs(domain: DomainConfig) -> dict[str, str]:
    docs = gen.build_docs(domain, generate_companions(domain))
    return {f"{d.collection}/{d.name}": d.content for d in docs}


@pytest.fixture(scope="module")
def docs(domain: DomainConfig) -> dict[str, str]:
    return _docs(domain)


def test_three_collections_and_one_page_per_companion(docs: dict[str, str]) -> None:
    collections = {k.split("/")[0] for k in docs}
    assert collections == set(gen.COLLECTIONS)
    assert sum(k.startswith("companion_profiles/") for k in docs) == 30
    for key in (
        "platform_rules/billing",
        "platform_rules/refund",
        "platform_rules/late_arrival",
        "platform_rules/violations",
        "platform_rules/companion_levels",
        "platform_rules/matching",
        "modes_and_ranks/game_modes",
        "modes_and_ranks/ranks",
        "modes_and_ranks/roles",
    ):
        assert key in docs


def test_every_doc_has_a_title_and_no_template_leftovers(docs: dict[str, str]) -> None:
    for key, text in docs.items():
        assert text.startswith("# "), key
        assert "{{" not in text and "{%" not in text, key
        assert text.endswith("\n") and not text.endswith("\n\n"), key


def test_billing_numbers_come_from_config(docs: dict[str, str]) -> None:
    billing = docs["platform_rules/billing"]
    assert "| 上分 | 上分、冲分、带飞 | 1.2 |" in billing
    assert "60 × 1.2 × 2 = **144.00 元**" in billing
    assert "60 × 1.5 × 2 = **180.00 元**" in billing
    assert "1–8 小时，以 0.5 小时为步长" in billing


def test_multiplier_change_propagates(domain: DomainConfig) -> None:
    raw = copy.deepcopy(RAW)
    raw["service_types"]["climb"]["multiplier"] = 1.3
    changed = _docs(parse_domain_config(raw))
    billing = changed["platform_rules/billing"]
    assert "| 上分 | 上分、冲分、带飞 | 1.3 |" in billing
    assert "60 × 1.3 × 2 = **156.00 元**" in billing
    # The refund example uses a climb booking, so it follows too.
    assert "以一笔 156.00 元" in changed["platform_rules/refund"]
    # Companion price lists are recomputed as well.
    page = changed["companion_profiles/companion_01"]
    base = _number_after(page, "基础单价 ")
    assert f"上分：{base * Decimal('1.3'):.2f} 元/小时" in page


def _number_after(text: str, prefix: str) -> Decimal:
    m = re.search(re.escape(prefix) + r"([\d.]+)", text)
    assert m is not None
    return Decimal(m.group(1))


def test_refund_tiers_and_examples(docs: dict[str, str]) -> None:
    refund = docs["platform_rules/refund"]
    assert "| ≥ 24 小时 | 全额退款（100%） |" in refund
    assert "| 2–24 小时 | 退一半（50%） |" in refund
    assert "| < 2 小时（含已开局） | 不退款（0%） |" in refund
    assert "提前 3 小时取消：退一半，退 **72.00 元**" in refund
    assert "提前 25 小时取消：全额退款，退 **144.00 元**" in refund


def test_refund_tier_change_propagates() -> None:
    raw = copy.deepcopy(RAW)
    raw["refund"]["tiers"][1]["ratio"] = 0.3
    refund = _docs(parse_domain_config(raw))["platform_rules/refund"]
    assert "退一半（30%）" in refund
    assert "退 **43.20 元**" in refund


def test_policies_rendered(docs: dict[str, str], domain: DomainConfig) -> None:
    late = docs["platform_rules/late_arrival"]
    p = domain.policies.late_arrival
    assert f"{p.grace_minutes} 分钟内上线视为准时" in late
    assert f"迟到达到 {p.full_refund_after_minutes} 分钟" in late
    violations = docs["platform_rules/violations"]
    for v in domain.policies.violations:
        assert f"## {v.name}" in violations and v.penalty in violations
    levels = docs["platform_rules/companion_levels"]
    assert "| 王牌陪玩 | ≥ 4.8 |" in levels
    assert "| 新星陪玩 | < 4.5 |" in levels


def test_matching_rules(docs: dict[str, str]) -> None:
    matching = docs["platform_rules/matching"]
    assert "必须提供：游戏模式、开始时间、时长" in matching
    assert "开始时间（前后 1 小时） → 性别 → 位置" in matching
    assert "预算与段位要求永远不会放宽" in matching
    assert "要求钻石，可匹配钻石到宗师" in matching


def test_modes_and_ranks(docs: dict[str, str], domain: DomainConfig) -> None:
    modes = docs["modes_and_ranks/game_modes"]
    for cfg in domain.game_modes.values():
        assert f"## {cfg.label}" in modes
        assert cfg.players in modes
    assert "8 队 × 2 人" in modes  # 斗魂竞技场是几个人
    ranks = docs["modes_and_ranks/ranks"]
    assert "| 钻石 | 钻石–宗师 |" in ranks
    assert "| 王者 | 王者 |" in ranks


def test_companion_pages_match_seed(docs: dict[str, str], domain: DomainConfig) -> None:
    seeds = generate_companions(domain)
    for i, c in enumerate(seeds, start=1):
        page = docs[f"companion_profiles/companion_{i:02d}"]
        assert page.startswith(f"# 陪玩师 {c.name}\n")
        assert f"单双排段位：{domain.rank_label(c.rank)}" in page
        assert f"基础单价 {c.hourly_price:.2f} 元/小时" in page
        assert domain.policies.level_for(c.rating).name in page


def test_main_writes_files_and_cleans_stale_ones(tmp_path: Path) -> None:
    stale = tmp_path / "companion_profiles" / "companion_99.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    assert gen.main(["--out", str(tmp_path)]) == 0
    assert not stale.exists()
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.md"))
    assert len(files) == 39
    assert "platform_rules/refund.md" in files
    text = (tmp_path / "platform_rules" / "refund.md").read_bytes()
    assert b"\r\n" not in text
