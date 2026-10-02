"""Coverage audit of an SFT set: categories, variants, styles, tri-state fields, mode aliases.

``note`` of a generated sample is ``"<variant>; <style>"`` (see ``validate.assemble``).
Alias coverage counts how many user inputs contain each surface form from ``domain.yaml``,
so gaps such as "海克斯" never appearing show up before training, not in the eval.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from rift_domain.config import DomainConfig
from rift_training.data.card import delta_field_counts
from rift_training.evaluation.dataset import CATEGORIES, SlotSample


def _note_parts(sample: SlotSample) -> tuple[str, str]:
    variant, _, style = (sample.note or "").partition("; ")
    return variant or "-", style or "-"


def alias_groups(domain: DomainConfig) -> dict[str, dict[str, tuple[str, ...]]]:
    """Kind -> canonical value -> surface forms (label + aliases)."""

    def forms(label: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys((label, *aliases)))

    return {
        "game_mode": {k.value: forms(v.label, v.aliases) for k, v in domain.game_modes.items()},
        "rank": {r.key.value: forms(r.label, r.aliases) for r in domain.ranks},
        "role": {k.value: forms(v.label, v.aliases) for k, v in domain.roles.items()},
        "service_type": {
            k.value: forms(v.label, v.aliases) for k, v in domain.service_types.items()
        },
        "gender": {k.value: forms(v.label, v.aliases) for k, v in domain.genders.items()},
    }


def alias_coverage(samples: Sequence[SlotSample], domain: DomainConfig) -> dict[str, Any]:
    texts = [s.user_input.lower() for s in samples]
    counts: dict[str, dict[str, int]] = {}
    unseen: list[str] = []
    for kind, values in alias_groups(domain).items():
        for value, surface in values.items():
            for form in surface:
                n = sum(form.lower() in t for t in texts)
                counts.setdefault(kind, {})[f"{value}:{form}"] = n
                if n == 0:
                    unseen.append(f"{kind}.{value}:{form}")
    return {"counts": counts, "unseen": unseen}


def audit(samples: Sequence[SlotSample], domain: DomainConfig) -> dict[str, Any]:
    variants = Counter(f"{s.category}.{_note_parts(s)[0]}" for s in samples)
    styles = Counter(_note_parts(s)[1] for s in samples)
    modes = Counter(
        str(s.expected["delta"].get("game_mode") or s.current_state.get("game_mode") or "-")
        for s in samples
    )
    lengths = [len(s.user_input) for s in samples] or [0]
    categories = Counter(s.category for s in samples)
    return {
        "total": len(samples),
        "categories": {k: categories[k] for k in CATEGORIES},
        "variants": dict(sorted(variants.items())),
        "styles": dict(styles.most_common()),
        "turn_intents": dict(Counter(s.expected["turn_intent"] for s in samples)),
        "confirmations": dict(Counter(s.expected["confirmation"] for s in samples)),
        "game_modes": dict(modes.most_common()),
        "history_messages": dict(sorted(Counter(len(s.history) for s in samples).items())),
        "candidates": dict(sorted(Counter(len(s.candidates) for s in samples).items())),
        "user_input_chars": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
        },
        "delta_fields": delta_field_counts(samples),
        "aliases": alias_coverage(samples, domain),
    }


def missing_categories(report: dict[str, Any]) -> list[str]:
    return [k for k, n in report["categories"].items() if n == 0]
