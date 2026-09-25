"""``DomainConfig``: immutable, validated view of ``config/domain.yaml``."""

from __future__ import annotations

from decimal import Decimal
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rift_domain.enums import (
    GameMode,
    Gender,
    Rank,
    RelaxStepName,
    Role,
    ServiceType,
    SlotField,
)

DEFAULT_DOMAIN_PATH = Path("config/domain.yaml")


class DomainConfigError(ValueError):
    """Raised when ``domain.yaml`` is missing, malformed or inconsistent."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabeledEntry(_Frozen):
    label: str
    aliases: tuple[str, ...] = ()


class ModeConfig(LabeledEntry):
    ranked: bool
    players: str
    default_service_type: ServiceType
    description: str


class RankEntry(LabeledEntry):
    key: Rank


class ServiceTypeConfig(LabeledEntry):
    multiplier: Decimal = Field(gt=0)


class ConditionalFields(_Frozen):
    when_modes: tuple[GameMode, ...] = Field(min_length=1)
    fields: tuple[SlotField, ...] = Field(min_length=1)


class RankBand(_Frozen):
    modes: tuple[GameMode, ...]
    max_tier_gap: int = Field(ge=0)


class RulesConfig(_Frozen):
    required_fields: tuple[SlotField, ...] = Field(min_length=1)
    conditional_required: tuple[ConditionalFields, ...] = ()
    not_applicable: tuple[ConditionalFields, ...] = ()
    rank_band: RankBand


class DurationConfig(_Frozen):
    min_hours: Decimal = Field(gt=0)
    max_hours: Decimal = Field(gt=0)
    step_hours: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def _check_range(self) -> DurationConfig:
        if self.min_hours > self.max_hours:
            raise ValueError("min_hours must be <= max_hours")
        if self.min_hours % self.step_hours or self.max_hours % self.step_hours:
            raise ValueError("min_hours and max_hours must be multiples of step_hours")
        return self

    def is_valid(self, hours: float | Decimal) -> bool:
        value = Decimal(str(hours))
        return self.min_hours <= value <= self.max_hours and value % self.step_hours == 0


class RefundTier(_Frozen):
    name: str
    min_hours_before: Decimal = Field(ge=0)
    ratio: Decimal = Field(ge=0, le=1)
    label: str


class RefundConfig(_Frozen):
    tiers: tuple[RefundTier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_tiers(self) -> RefundConfig:
        bounds = [t.min_hours_before for t in self.tiers]
        if bounds != sorted(bounds, reverse=True) or len(set(bounds)) != len(bounds):
            raise ValueError("refund tiers must have strictly decreasing min_hours_before")
        if bounds[-1] != 0:
            raise ValueError("the last refund tier must start at min_hours_before: 0")
        ratios = [t.ratio for t in self.tiers]
        if ratios != sorted(ratios, reverse=True):
            raise ValueError("refund ratios must not increase as the start time approaches")
        if len({t.name for t in self.tiers}) != len(self.tiers):
            raise ValueError("refund tier names must be unique")
        return self


class RelaxationConfig(_Frozen):
    order: tuple[RelaxStepName, ...]
    time_window_hours: Decimal = Field(gt=0)
    never_relax: tuple[SlotField, ...]

    @model_validator(mode="after")
    def _check_order(self) -> RelaxationConfig:
        # Budget/rank cannot appear in ``order``: it only accepts RelaxStepName values.
        if len(set(self.order)) != len(self.order):
            raise ValueError("relaxation.order has duplicate steps")
        missing = {SlotField.BUDGET_PER_HOUR, SlotField.RANK_REQUIREMENT} - set(self.never_relax)
        if missing:
            raise ValueError(f"relaxation.never_relax must include {', '.join(sorted(missing))}")
        return self


class MatchingWeights(_Frozen):
    role: float = Field(ge=0)
    style: float = Field(ge=0)
    rating: float = Field(ge=0)


class MatchingConfig(_Frozen):
    top_k: int = Field(gt=0)
    rating_max: float = Field(gt=0)
    weights: MatchingWeights


class LateArrivalPolicy(_Frozen):
    grace_minutes: int = Field(ge=0)
    makeup_minutes_per_late_minute: Decimal = Field(ge=0)
    full_refund_after_minutes: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_order(self) -> LateArrivalPolicy:
        if self.full_refund_after_minutes <= self.grace_minutes:
            raise ValueError("full_refund_after_minutes must exceed grace_minutes")
        return self


class ViolationRule(_Frozen):
    name: str
    description: str
    penalty: str


class CompanionLevel(_Frozen):
    name: str
    min_rating: float = Field(ge=0)
    perks: str = ""


class PoliciesConfig(_Frozen):
    late_arrival: LateArrivalPolicy
    violations: tuple[ViolationRule, ...] = ()
    companion_levels: tuple[CompanionLevel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_levels(self) -> PoliciesConfig:
        bounds = [lv.min_rating for lv in self.companion_levels]
        if bounds != sorted(bounds, reverse=True) or len(set(bounds)) != len(bounds):
            raise ValueError("companion_levels must have strictly decreasing min_rating")
        if bounds[-1] != 0:
            raise ValueError("the last companion level must start at min_rating: 0")
        return self

    def level_for(self, rating: float) -> CompanionLevel:
        return next(lv for lv in self.companion_levels if rating >= lv.min_rating)


class DomainConfig(_Frozen):
    version: int
    game_modes: dict[GameMode, ModeConfig]
    ranks: tuple[RankEntry, ...]
    roles: dict[Role, LabeledEntry]
    service_types: dict[ServiceType, ServiceTypeConfig]
    genders: dict[Gender, LabeledEntry]
    rules: RulesConfig
    duration: DurationConfig
    refund: RefundConfig
    relaxation: RelaxationConfig
    matching: MatchingConfig
    policies: PoliciesConfig

    @model_validator(mode="after")
    def _check_consistency(self) -> DomainConfig:
        _require_all("game_modes", set(self.game_modes), set(GameMode))
        _require_all("roles", set(self.roles), set(Role))
        _require_all("service_types", set(self.service_types), set(ServiceType))
        _require_all("genders", set(self.genders), set(Gender))
        rank_keys = [r.key for r in self.ranks]
        if len(rank_keys) != len(set(rank_keys)):
            raise ValueError("ranks has duplicate keys")
        _require_all("ranks", set(rank_keys), set(Rank))

        for kind, entries in self._alias_groups().items():
            seen: dict[str, str] = {}
            for key, entry in entries:
                for alias in (entry.label, *entry.aliases):
                    other = seen.setdefault(alias, key)
                    if other != key:
                        raise ValueError(f"{kind} alias {alias!r} used by both {other} and {key}")
        return self

    def _alias_groups(self) -> dict[str, list[tuple[str, LabeledEntry]]]:
        return {
            "game_modes": [(k.value, v) for k, v in self.game_modes.items()],
            "ranks": [(r.key.value, r) for r in self.ranks],
            "roles": [(k.value, v) for k, v in self.roles.items()],
            "service_types": [(k.value, v) for k, v in self.service_types.items()],
            "genders": [(k.value, v) for k, v in self.genders.items()],
        }

    # --- ranks -----------------------------------------------------------------------

    @cached_property
    def rank_order(self) -> tuple[Rank, ...]:
        return tuple(r.key for r in self.ranks)

    def rank_index(self, rank: Rank) -> int:
        return self.rank_order.index(rank)

    def rank_offset(self, rank: Rank, tiers: int) -> Rank:
        """``rank`` shifted by ``tiers``, clamped to the ladder."""
        idx = min(max(self.rank_index(rank) + tiers, 0), len(self.rank_order) - 1)
        return self.rank_order[idx]

    def rank_label(self, rank: Rank) -> str:
        return self.ranks[self.rank_index(rank)].label

    # --- modes / rules ---------------------------------------------------------------

    def is_ranked(self, mode: GameMode) -> bool:
        return self.game_modes[mode].ranked

    def default_service_type(self, mode: GameMode) -> ServiceType:
        return self.game_modes[mode].default_service_type

    def required_fields_for(self, mode: GameMode | None) -> tuple[SlotField, ...]:
        """Required slots in asking order: base fields, then mode-specific ones."""
        fields = list(self.rules.required_fields)
        if mode is not None:
            for cond in self.rules.conditional_required:
                if mode in cond.when_modes:
                    fields.extend(f for f in cond.fields if f not in fields)
        return tuple(fields)

    def not_applicable_for(self, mode: GameMode | None) -> tuple[SlotField, ...]:
        if mode is None:
            return ()
        fields: list[SlotField] = []
        for cond in self.rules.not_applicable:
            if mode in cond.when_modes:
                fields.extend(f for f in cond.fields if f not in fields)
        return tuple(fields)

    def max_tier_gap_for(self, mode: GameMode) -> int | None:
        band = self.rules.rank_band
        return band.max_tier_gap if mode in band.modes else None

    def multiplier(self, service_type: ServiceType) -> Decimal:
        return self.service_types[service_type].multiplier

    # --- aliases ---------------------------------------------------------------------

    @cached_property
    def mode_alias_table(self) -> list[tuple[str, GameMode]]:
        pairs = [
            (alias, mode)
            for mode, cfg in self.game_modes.items()
            for alias in {cfg.label, *cfg.aliases}
        ]
        return sorted(pairs, key=lambda p: -len(p[0]))

    def find_mode(self, text: str) -> GameMode | None:
        """Mode named in ``text`` by label/alias; longest alias wins (海克斯大乱斗 > 乱斗)."""
        for alias, mode in self.mode_alias_table:
            if alias in text:
                return mode
        return None


def _require_all(name: str, present: set[Any], expected: set[Any]) -> None:
    if present != expected:
        missing = sorted(str(v) for v in expected - present)
        extra = sorted(str(v) for v in present - expected)
        raise ValueError(
            f"{name} must list exactly the enum values; missing={missing} extra={extra}"
        )


def _format_errors(err: ValidationError) -> str:
    parts = []
    for e in err.errors():
        path = ".".join(str(p) for p in e["loc"]) or "<root>"
        msg = e["msg"].removeprefix("Value error, ")
        parts.append(f"{path}: {msg}")
    return "; ".join(parts)


def parse_domain_config(data: Any) -> DomainConfig:
    try:
        return DomainConfig.model_validate(data)
    except ValidationError as exc:
        raise DomainConfigError(_format_errors(exc)) from exc


def load_domain_config(path: str | Path = DEFAULT_DOMAIN_PATH) -> DomainConfig:
    p = Path(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DomainConfigError(f"domain config not found: {p}") from exc
    except yaml.YAMLError as exc:
        raise DomainConfigError(f"invalid YAML in {p}: {exc}") from exc
    return parse_domain_config(data)
