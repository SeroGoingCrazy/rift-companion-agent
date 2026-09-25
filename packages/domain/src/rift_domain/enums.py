"""Domain enumerations. Values must match the keys in ``config/domain.yaml``."""

from __future__ import annotations

from enum import StrEnum


class GameMode(StrEnum):
    RANKED_SOLO_DUO = "ranked_solo_duo"
    RANKED_FLEX = "ranked_flex"
    NORMAL_DRAFT = "normal_draft"
    ARAM = "aram"
    ARAM_MAYHEM = "aram_mayhem"
    ARENA = "arena"


class Rank(StrEnum):
    """Declared low -> high; ``DomainConfig.rank_index`` is the authoritative order."""

    IRON = "iron"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    EMERALD = "emerald"
    DIAMOND = "diamond"
    MASTER = "master"
    GRANDMASTER = "grandmaster"
    CHALLENGER = "challenger"


class Role(StrEnum):
    TOP = "top"
    JUNGLE = "jungle"
    MID = "mid"
    ADC = "adc"
    SUPPORT = "support"


class ServiceType(StrEnum):
    CLIMB = "climb"
    CASUAL = "casual"
    COACHING = "coaching"


class Gender(StrEnum):
    FEMALE = "female"
    MALE = "male"


class TurnIntent(StrEnum):
    """What the user did this turn, as seen by the slot extractor."""

    BOOKING = "booking"
    CONSULT = "consult"
    UNRELATED = "unrelated"


class Confirmation(StrEnum):
    YES = "yes"
    NO = "no"
    NONE = "none"


class PendingAction(StrEnum):
    AWAIT_CONFIRM_BOOKING = "await_confirm_booking"
    AWAIT_CONFIRM_CANCEL = "await_confirm_cancel"


class SlotField(StrEnum):
    """Booking-state slot names (``start_time`` is the resolved form of ``start_time_expr``)."""

    GAME_MODE = "game_mode"
    START_TIME = "start_time"
    DURATION_HOURS = "duration_hours"
    RANK_REQUIREMENT = "rank_requirement"
    ROLE_PREFERENCE = "role_preference"
    SERVICE_TYPE = "service_type"
    COMPANION_GENDER = "companion_gender"
    VOICE_REQUIRED = "voice_required"
    BUDGET_PER_HOUR = "budget_per_hour"
    STYLE_PREFERENCE = "style_preference"
    COMPANION_NAME = "companion_name"


class RelaxStepName(StrEnum):
    TIME_WINDOW = "time_window"
    COMPANION_GENDER = "companion_gender"
    ROLE_PREFERENCE = "role_preference"
