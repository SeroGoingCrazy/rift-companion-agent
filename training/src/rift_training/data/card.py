"""Data cards: what a training-data version contains and which prompt it was rendered for.

Every dataset version under ``training/data/processed`` ships ``data_card.json`` (machine
readable) and ``DATA_CARD.md`` (for review). The card pins the sha256 of
``config/prompts/slot_extract.txt``; ``prompt_mismatch`` tells training scripts (and the
agent at start-up) when the prompt changed after the data was rendered.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rift_training.contract import PROMPT_FILE, prompt_hash
from rift_training.evaluation.dataset import CATEGORIES, SlotSample

CARD_JSON = "data_card.json"
CARD_MD = "DATA_CARD.md"
#: How a delta key was used: a value, "any" (no preference) or null (withdrawn).
VALUE_KINDS = ("value", "any", "null")


class DataCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    created_at: str
    num_samples: int
    prompt_file: str = PROMPT_FILE
    prompt_sha256: str
    categories: dict[str, int] = Field(default_factory=dict)
    turn_intents: dict[str, int] = Field(default_factory=dict)
    confirmations: dict[str, int] = Field(default_factory=dict)
    #: field -> {"value": n, "any": n, "null": n}
    delta_fields: dict[str, dict[str, int]] = Field(default_factory=dict)
    #: Teacher backend and model, e.g. {"provider": "anthropic", "model": "..."}.
    teacher: dict[str, str] | None = None
    spec: str | None = None
    notes: list[str] = Field(default_factory=list)
    #: Free-form sections added by later steps (audit, dedup, rejected counts ...).
    extra: dict[str, Any] = Field(default_factory=dict)


def value_kind(value: Any) -> str:
    if value is None:
        return "null"
    if value == "any":
        return "any"
    return "value"


def delta_field_counts(samples: Sequence[SlotSample]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for sample in samples:
        for key, value in sample.expected["delta"].items():
            counts.setdefault(key, Counter())[value_kind(value)] += 1
    return {key: {kind: counts[key][kind] for kind in VALUE_KINDS} for key in sorted(counts)}


def build_card(
    samples: Sequence[SlotSample],
    *,
    name: str,
    version: str,
    teacher: dict[str, str] | None = None,
    spec: str | None = None,
    notes: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> DataCard:
    categories = Counter(s.category for s in samples)
    return DataCard(
        name=name,
        version=version,
        created_at=(created_at or datetime.now()).isoformat(timespec="seconds"),
        num_samples=len(samples),
        prompt_sha256=prompt_hash(),
        categories={key: categories[key] for key in CATEGORIES if categories[key]},
        turn_intents=dict(sorted(Counter(s.expected["turn_intent"] for s in samples).items())),
        confirmations=dict(sorted(Counter(s.expected["confirmation"] for s in samples).items())),
        delta_fields=delta_field_counts(samples),
        teacher=teacher,
        spec=spec,
        notes=list(notes),
        extra=extra or {},
    )


def prompt_mismatch(card: DataCard) -> str | None:
    """A readable warning when the prompt file changed since the data was rendered."""
    current = prompt_hash()
    if card.prompt_sha256 == current:
        return None
    return (
        f"{card.name} {card.version} was rendered for {card.prompt_file} "
        f"sha256 {card.prompt_sha256[:12]}, current file is {current[:12]}"
    )


def _counts_table(title: str, counts: dict[str, int], total: int) -> list[str]:
    rows = [f"| {title} | 数量 | 占比 |", "|---|---:|---:|"]
    for key, n in counts.items():
        rows.append(f"| `{key}` | {n} | {n / total:.1%} |" if total else f"| `{key}` | {n} | – |")
    return rows


def render_markdown(card: DataCard) -> str:
    total = card.num_samples
    teacher = " / ".join(f"{k}={v}" for k, v in card.teacher.items()) if card.teacher else "–"
    lines = [
        f"# 数据卡：{card.name} {card.version}",
        "",
        f"- 生成时间：{card.created_at}",
        f"- 样本数：**{total}**",
        f"- 场景规格：{card.spec or '–'}",
        f"- Teacher：{teacher}",
        f"- Prompt：`{card.prompt_file}` sha256 `{card.prompt_sha256}`",
        "",
        "## 场景类别",
        "",
        *_counts_table("类别", card.categories, total),
        "",
        "## turn_intent / confirmation",
        "",
        *_counts_table("turn_intent", card.turn_intents, total),
        "",
        *_counts_table("confirmation", card.confirmations, total),
        "",
        "## delta 字段（三态）",
        "",
        "| 字段 | 值 | any | null |",
        "|---|---:|---:|---:|",
    ]
    for key, kinds in card.delta_fields.items():
        lines.append(f"| `{key}` | {kinds['value']} | {kinds['any']} | {kinds['null']} |")
    if card.notes:
        lines += ["", "## 备注", "", *(f"- {note}" for note in card.notes)]
    for section, value in card.extra.items():
        lines += [
            "",
            f"## {section}",
            "",
            "```json",
            json.dumps(value, ensure_ascii=False, indent=2),
            "```",
        ]
    return "\n".join(lines) + "\n"


def write_card(card: DataCard, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / CARD_JSON, out_dir / CARD_MD
    with json_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(card.model_dump_json(indent=2) + "\n")
    with md_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(render_markdown(card))
    return json_path, md_path


def load_card(path: Path) -> DataCard:
    """Read ``data_card.json`` (a directory containing it also works)."""
    if path.is_dir():
        path = path / CARD_JSON
    return DataCard.model_validate_json(path.read_text("utf-8"))
