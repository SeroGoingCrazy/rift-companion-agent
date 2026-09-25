import copy
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from rift_domain.config import (
    DomainConfig,
    DomainConfigError,
    load_domain_config,
    parse_domain_config,
)
from rift_domain.enums import GameMode, Rank, ServiceType, SlotField

DOMAIN_PATH = Path(__file__).resolve().parents[2] / "config" / "domain.yaml"
RAW: dict[str, Any] = yaml.safe_load(DOMAIN_PATH.read_text(encoding="utf-8"))


def _raw() -> dict[str, Any]:
    return copy.deepcopy(RAW)


@pytest.fixture(scope="module")
def domain() -> DomainConfig:
    return load_domain_config(DOMAIN_PATH)


# --- repo config loads and exposes the rules -----------------------------------------


def test_repo_domain_yaml_loads(domain: DomainConfig) -> None:
    assert set(domain.game_modes) == set(GameMode)
    assert domain.rank_order[0] is Rank.IRON and domain.rank_order[-1] is Rank.CHALLENGER
    assert domain.multiplier(ServiceType.CLIMB) == Decimal("1.2")
    assert domain.multiplier(ServiceType.CASUAL) == Decimal("1.0")
    assert domain.multiplier(ServiceType.COACHING) == Decimal("1.5")


def test_rank_helpers(domain: DomainConfig) -> None:
    assert domain.rank_index(Rank.DIAMOND) > domain.rank_index(Rank.EMERALD)
    assert domain.rank_offset(Rank.DIAMOND, 2) is Rank.GRANDMASTER
    assert domain.rank_offset(Rank.GRANDMASTER, 2) is Rank.CHALLENGER  # clamped
    assert domain.rank_offset(Rank.IRON, -1) is Rank.IRON
    assert domain.rank_label(Rank.EMERALD) == "翡翠"


def test_required_fields_order(domain: DomainConfig) -> None:
    base = (SlotField.GAME_MODE, SlotField.START_TIME, SlotField.DURATION_HOURS)
    assert domain.required_fields_for(None) == base
    assert domain.required_fields_for(GameMode.ARAM) == base
    assert domain.required_fields_for(GameMode.RANKED_SOLO_DUO) == (
        *base,
        SlotField.RANK_REQUIREMENT,
    )


def test_not_applicable_fields(domain: DomainConfig) -> None:
    expected = (SlotField.RANK_REQUIREMENT, SlotField.ROLE_PREFERENCE)
    for mode in (GameMode.ARAM, GameMode.ARAM_MAYHEM, GameMode.ARENA):
        assert domain.not_applicable_for(mode) == expected
    assert domain.not_applicable_for(GameMode.RANKED_FLEX) == ()
    assert domain.not_applicable_for(None) == ()


def test_rank_band_only_for_solo_duo(domain: DomainConfig) -> None:
    assert domain.max_tier_gap_for(GameMode.RANKED_SOLO_DUO) == 2
    assert domain.max_tier_gap_for(GameMode.RANKED_FLEX) is None


def test_mode_defaults(domain: DomainConfig) -> None:
    assert domain.is_ranked(GameMode.RANKED_FLEX)
    assert not domain.is_ranked(GameMode.ARENA)
    assert domain.default_service_type(GameMode.RANKED_SOLO_DUO) is ServiceType.CLIMB
    assert domain.default_service_type(GameMode.ARAM) is ServiceType.CASUAL


@pytest.mark.parametrize(
    ("text", "mode"),
    [
        ("想打海克斯大乱斗", GameMode.ARAM_MAYHEM),
        ("来把乱斗", GameMode.ARAM),
        ("极地大乱斗走起", GameMode.ARAM),
        ("斗魂", GameMode.ARENA),
        ("双排上分", GameMode.RANKED_SOLO_DUO),
        ("灵活组排五个人", GameMode.RANKED_FLEX),
        ("普通匹配", GameMode.NORMAL_DRAFT),
        ("随便玩玩", None),
    ],
)
def test_find_mode_prefers_longest_alias(domain: DomainConfig, text: str, mode: GameMode) -> None:
    assert domain.find_mode(text) == mode


def test_duration_validity(domain: DomainConfig) -> None:
    assert domain.duration.is_valid(1)
    assert domain.duration.is_valid(2.5)
    assert domain.duration.is_valid(8)
    assert not domain.duration.is_valid(0.5)
    assert not domain.duration.is_valid(8.5)
    assert not domain.duration.is_valid(1.3)


# --- invalid configs fail at load -----------------------------------------------------


def _expect_error(data: dict[str, Any], match: str) -> None:
    with pytest.raises(DomainConfigError, match=match):
        parse_domain_config(data)


def test_negative_multiplier_rejected() -> None:
    data = _raw()
    data["service_types"]["climb"]["multiplier"] = -1.2
    _expect_error(data, r"service_types\.climb\.multiplier")


def test_zero_multiplier_rejected() -> None:
    data = _raw()
    data["service_types"]["casual"]["multiplier"] = 0
    _expect_error(data, "greater than 0")


@pytest.mark.parametrize("step", ["budget_per_hour", "rank_requirement"])
def test_relaxing_budget_or_rank_rejected(step: str) -> None:
    data = _raw()
    data["relaxation"]["order"].append(step)
    _expect_error(data, r"relaxation\.order")


def test_never_relax_must_cover_budget_and_rank() -> None:
    data = _raw()
    data["relaxation"]["never_relax"] = ["budget_per_hour"]
    _expect_error(data, "never_relax must include rank_requirement")


def test_duplicate_relax_step_rejected() -> None:
    data = _raw()
    data["relaxation"]["order"] = ["time_window", "time_window"]
    _expect_error(data, "duplicate steps")


def test_duplicate_mode_alias_rejected() -> None:
    data = _raw()
    data["game_modes"]["aram"]["aliases"].append("双排")
    _expect_error(data, "alias '双排' used by both ranked_solo_duo and aram")


def test_missing_mode_rejected() -> None:
    data = _raw()
    del data["game_modes"]["arena"]
    _expect_error(data, r"missing=\['arena'\]")


def test_unknown_mode_rejected() -> None:
    data = _raw()
    data["game_modes"]["tft"] = data["game_modes"]["aram"]
    _expect_error(data, r"game_modes\.tft")


def test_missing_rank_rejected() -> None:
    data = _raw()
    data["ranks"] = [r for r in data["ranks"] if r["key"] != "emerald"]
    _expect_error(data, r"ranks must list exactly the enum values; missing=\['emerald'\]")


def test_refund_tiers_must_decrease_and_end_at_zero() -> None:
    data = _raw()
    data["refund"]["tiers"] = list(reversed(data["refund"]["tiers"]))
    _expect_error(data, "strictly decreasing")
    data = _raw()
    data["refund"]["tiers"] = data["refund"]["tiers"][:2]
    _expect_error(data, "must start at min_hours_before: 0")


def test_refund_ratio_increasing_toward_start_rejected() -> None:
    data = _raw()
    data["refund"]["tiers"][2]["ratio"] = 0.8
    _expect_error(data, "must not increase")


def test_refund_tier_names_unique() -> None:
    data = _raw()
    data["refund"]["tiers"][1]["name"] = "full"
    _expect_error(data, "names must be unique")


def test_duplicate_rank_rejected() -> None:
    data = _raw()
    data["ranks"].append(dict(data["ranks"][0], label="黑铁2", aliases=[]))
    _expect_error(data, "ranks has duplicate keys")


def test_refund_ratio_out_of_range_rejected() -> None:
    data = _raw()
    data["refund"]["tiers"][0]["ratio"] = 1.5
    _expect_error(data, r"refund\.tiers\.0\.ratio")


def test_duration_range_checked() -> None:
    data = _raw()
    data["duration"]["min_hours"] = 9
    _expect_error(data, "min_hours must be <= max_hours")
    data = _raw()
    data["duration"]["min_hours"] = 1.2
    _expect_error(data, "multiples of step_hours")


def test_unknown_required_field_rejected() -> None:
    data = _raw()
    data["rules"]["required_fields"].append("favourite_champion")
    _expect_error(data, r"rules\.required_fields")


def test_unknown_top_level_key_rejected() -> None:
    data = _raw()
    data["surprise"] = 1
    _expect_error(data, "surprise")


def test_config_is_immutable(domain: DomainConfig) -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic frozen instance error
        domain.version = 2  # type: ignore[misc]


def test_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(DomainConfigError, match="not found"):
        load_domain_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("game_modes: [unclosed", encoding="utf-8")
    with pytest.raises(DomainConfigError, match="invalid YAML"):
        load_domain_config(bad)
