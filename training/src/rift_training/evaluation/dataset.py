"""L2 slot-extraction samples (``eval/datasets/slot_*.jsonl``) and their checks.

One JSON object per line: the extractor's input context (``now``, ``current_state``,
``candidates``, ``pending_confirmation``, ``history``, ``user_input``) and the expected
``SlotExtraction``. ``expected`` keeps the tri-state contract literally: a key absent from
``expected.delta`` means "not mentioned", ``null`` means "withdrawn", ``"any"`` means "no
preference".

The sets are written by hand, reviewed line by line and frozen before training: their
sha256 sums live in ``eval/datasets/CHECKSUMS`` (``sha256sum`` format).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rift_domain.enums import SlotField
from rift_domain.slots import ANY, BookingState, SlotExtraction, parse_extraction

#: Scenario categories of DEV_SPEC 3.7 (key -> label used in reports).
CATEGORIES: dict[str, str] = {
    "first_turn": "首轮抽取",
    "multi_turn": "多轮增量",
    "tri_state": "三态语义",
    "relative_time": "相对时间表达",
    "candidate_ref": "候选引用",
    "consult": "插话咨询",
    "unrelated": "无关话题",
    "confirmation": "确认 / 拒绝",
    "mode_slang": "模式说法与黑话",
}
MIN_PER_CATEGORY = 3
#: Keys ``current_state`` may carry: the merged slots plus the raw time expression.
STATE_KEYS = frozenset({f.value for f in SlotField} | {"start_time_expr"})


class DatasetError(ValueError):
    """A dataset file could not be read or failed validation."""


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class SlotSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    category: str
    now: datetime
    current_state: dict[str, Any] = Field(default_factory=dict)
    candidates: list[str] = Field(default_factory=list)
    pending_confirmation: bool = False
    history: list[HistoryMessage] = Field(default_factory=list)
    user_input: str = Field(min_length=1)
    expected: dict[str, Any]
    note: str | None = None

    @field_validator("category")
    @classmethod
    def _known_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"unknown category {value!r}")
        return value

    @field_validator("current_state")
    @classmethod
    def _state_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = set(value) - STATE_KEYS
        if unknown:
            raise ValueError(f"unknown current_state keys {sorted(unknown)}")
        BookingState.model_validate(value)
        return value

    @field_validator("expected")
    @classmethod
    def _expected_contract(cls, value: dict[str, Any]) -> dict[str, Any]:
        parse_extraction(value)  # C2 schema: strict types, enums, no extra keys
        return value

    def state(self) -> BookingState:
        return BookingState.model_validate(self.current_state)

    def extraction(self) -> SlotExtraction:
        return parse_extraction(self.expected)

    def history_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.history]


# --- loading -------------------------------------------------------------------------------


def parse_samples(lines: Iterable[str], *, source: str = "<memory>") -> list[SlotSample]:
    """Parse JSONL lines; every bad line is reported (``DatasetError``), not just the first."""
    samples: list[SlotSample] = []
    errors: list[str] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            samples.append(SlotSample.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            errors.append(f"{source}:{lineno}: invalid JSON: {exc}")
        except ValidationError as exc:
            for e in exc.errors():
                loc = ".".join(str(p) for p in e["loc"]) or "<root>"
                errors.append(f"{source}:{lineno}: {loc}: {e['msg']}")
    if errors:
        raise DatasetError("\n".join(errors))
    return samples


def load_samples(path: Path) -> list[SlotSample]:
    if not path.exists():
        raise DatasetError(f"{path}: not found")
    return parse_samples(path.read_text("utf-8").splitlines(), source=path.name)


# --- checks --------------------------------------------------------------------------------


def sample_problems(sample: SlotSample) -> list[str]:
    """Consistency rules beyond the schema (things a reviewer could easily miss)."""
    problems: list[str] = []
    ext = sample.extraction()
    delta = ext.delta.provided()
    name = delta.get("companion_name")
    if sample.category == "candidate_ref" and not sample.candidates:
        problems.append("candidate_ref sample without candidates")
    if (
        isinstance(name, str)
        and name != ANY
        and sample.candidates
        and name not in sample.candidates
    ):
        problems.append(f"companion_name {name!r} is not one of the candidates")
    if ext.confirmation.value == "yes" and not sample.pending_confirmation:
        problems.append("confirmation=yes only answers a pending confirmation")
    if ext.turn_intent.value != "booking" and delta:
        problems.append(f"{ext.turn_intent.value} turn should not carry slot values")
    if sample.history and sample.history[-1].role != "assistant":
        problems.append("history must end with the assistant's message")
    return problems


def category_counts(samples: Sequence[SlotSample]) -> Counter[str]:
    return Counter(s.category for s in samples)


def normalize_input(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?~～]+", "", text).lower()


def check_datasets(datasets: dict[str, Sequence[SlotSample]]) -> list[str]:
    """Cross-set rules: unique ids, enough samples per category, no input shared by two sets."""
    problems: list[str] = []
    seen_ids: dict[str, str] = {}
    inputs: dict[str, dict[str, str]] = {}
    for name, samples in datasets.items():
        for s in samples:
            if s.id in seen_ids:
                problems.append(f"{name}: duplicate id {s.id} (also in {seen_ids[s.id]})")
            seen_ids[s.id] = name
            for p in sample_problems(s):
                problems.append(f"{name}:{s.id}: {p}")
            inputs.setdefault(name, {})[normalize_input(s.user_input)] = s.id
        counts = category_counts(samples)
        for category in CATEGORIES:
            if counts[category] < MIN_PER_CATEGORY:
                problems.append(
                    f"{name}: category {category} has {counts[category]} samples "
                    f"(< {MIN_PER_CATEGORY})"
                )
    names = list(inputs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for text in sorted(inputs[a].keys() & inputs[b].keys()):
                problems.append(
                    f"user_input shared by {a}:{inputs[a][text]} and {b}:{inputs[b][text]}"
                )
    return problems


def distribution_table(datasets: dict[str, Sequence[SlotSample]]) -> str:
    """Markdown table: one row per category, one column per dataset."""
    names = list(datasets)
    counts = {name: category_counts(samples) for name, samples in datasets.items()}
    rows = [
        "| 类别 | key | " + " | ".join(names) + " |",
        "|---|---|" + "---:|" * len(names),
    ]
    for key, label in CATEGORIES.items():
        rows.append(
            f"| {label} | `{key}` | " + " | ".join(str(counts[n][key]) for n in names) + " |"
        )
    totals = " | ".join(f"**{len(datasets[n])}**" for n in names)
    rows.append(f"| **合计** | | {totals} |")
    return "\n".join(rows)


# --- checksums -----------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """``<sha256>  <file name>`` per line, as written by ``sha256sum``."""
    sums: dict[str, str] = {}
    if not path.exists():
        return sums
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            digest, _, name = line.strip().partition("  ")
            sums[name.strip()] = digest
    return sums


def write_checksums(path: Path, files: Sequence[Path]) -> None:
    lines = [f"{sha256_file(f)}  {f.name}\n" for f in sorted(files, key=lambda p: p.name)]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)


def checksum_problems(path: Path, files: Sequence[Path]) -> list[str]:
    sums = read_checksums(path)
    problems: list[str] = []
    for f in files:
        recorded = sums.get(f.name)
        if recorded is None:
            problems.append(f"{f.name}: no checksum recorded in {path.name}")
        elif recorded != sha256_file(f):
            problems.append(f"{f.name}: checksum mismatch (dataset changed after freezing)")
    return problems
