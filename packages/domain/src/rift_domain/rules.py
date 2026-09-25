"""Business rules over the merged state: required fields, not-applicable fields, service type.

``compute_rules`` only reports; ``apply_rules`` also writes the result into the state
(clears not-applicable/invalid slots, sets ``missing_fields`` and
``service_type_effective``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rift_domain.config import DomainConfig
from rift_domain.enums import ServiceType, SlotField
from rift_domain.slots import BookingState


@dataclass(frozen=True)
class RulesResult:
    #: Required slots still empty, in asking order (base fields, then mode-specific).
    missing_fields: tuple[SlotField, ...]
    #: Slots that do not apply to the chosen mode and were (or must be) cleared.
    cleared_fields: tuple[SlotField, ...]
    #: Explicit service type if the user gave one, else the mode default.
    service_type_effective: ServiceType | None
    #: Slots whose value breaks a domain rule -> human-readable reason; also cleared.
    invalid_fields: dict[SlotField, str] = field(default_factory=dict)


def _invalid_fields(state: BookingState, domain: DomainConfig) -> dict[SlotField, str]:
    invalid: dict[SlotField, str] = {}
    if state.duration_hours is not None and not domain.duration.is_valid(state.duration_hours):
        d = domain.duration
        invalid[SlotField.DURATION_HOURS] = (
            f"时长需在 {d.min_hours:g}–{d.max_hours:g} 小时之间，按 {d.step_hours:g} 小时递增"
        )
    return invalid


def compute_rules(state: BookingState, domain: DomainConfig) -> RulesResult:
    mode = state.game_mode
    cleared = tuple(f for f in domain.not_applicable_for(mode) if state.slot(f) is not None)
    invalid = _invalid_fields(state, domain)
    emptied = set(cleared) | set(invalid)

    missing = tuple(
        f for f in domain.required_fields_for(mode) if f in emptied or not state.is_filled(f)
    )

    if state.service_type is not None:
        service_type: ServiceType | None = state.service_type
    elif mode is not None:
        service_type = domain.default_service_type(mode)
    else:
        service_type = None

    return RulesResult(
        missing_fields=missing,
        cleared_fields=cleared,
        service_type_effective=service_type,
        invalid_fields=invalid,
    )


def apply_rules(state: BookingState, domain: DomainConfig) -> tuple[BookingState, RulesResult]:
    result = compute_rules(state, domain)
    updates: dict[str, object] = {
        f.value: None for f in (*result.cleared_fields, *result.invalid_fields)
    }
    updates["missing_fields"] = result.missing_fields
    updates["service_type_effective"] = result.service_type_effective
    return state.model_copy(update=updates), result
