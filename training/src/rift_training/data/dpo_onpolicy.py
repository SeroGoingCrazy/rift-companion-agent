"""On-policy preference pairs (DEV_SPEC I8 ②).

The best SFT model samples ``k`` answers per train prompt (temperature 0.7,
``training/scripts/train/sample_onpolicy.py``). Every distinct sample the L2 scorer fails
becomes ``rejected`` -- verbatim, as the model wrote it -- against the Teacher answer as
``chosen``. The scorer is the one L2 uses (same style matcher), so a correct paraphrase of a
style is never punished. ``kind`` records why a sample failed: the protocol error, or the
first wrong field.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rift_training.data.preference import PreferencePair
from rift_training.evaluation.dataset import SlotSample
from rift_training.evaluation.preferences import StyleMatcher
from rift_training.evaluation.task import SampleScore, score_output


def read_samples(path: Path) -> dict[str, list[str]]:
    """``{"id": ..., "samples": [...]}`` per line -> id -> samples."""
    out: dict[str, list[str]] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = list(row["samples"])
    return out


def failure_kind(score: SampleScore) -> str:
    if not score.protocol.ok:
        return f"protocol:{score.protocol.error}"
    errors = score.task.errors if score.task else ()
    return f"field:{errors[0].field}" if errors else "field:?"


@dataclass
class OnPolicyStats:
    samples: int = 0
    generations: int = 0
    passed: int = 0
    samples_with_failure: int = 0
    failure_kinds: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "generations": self.generations,
            "generation_pass_rate": round(self.passed / self.generations, 4)
            if self.generations
            else None,
            "samples_with_failure": self.samples_with_failure,
            "failure_kinds": dict(self.failure_kinds.most_common()),
        }


def onpolicy_pairs(
    samples: Sequence[SlotSample],
    sampled: Mapping[str, Sequence[str]],
    style: StyleMatcher,
    *,
    max_per_sample: int = 2,
) -> tuple[list[PreferencePair], OnPolicyStats]:
    stats = OnPolicyStats()
    pairs: list[PreferencePair] = []
    for sample in samples:
        texts = sampled.get(sample.id)
        if texts is None:
            continue
        stats.samples += 1
        expected = sample.extraction()
        seen: set[str] = set()
        failed = 0
        for text in texts:
            stats.generations += 1
            score = score_output(text, expected, state=sample.state(), now=sample.now, style=style)
            if score.passed:
                stats.passed += 1
                continue
            kind = failure_kind(score)
            stats.failure_kinds[kind] += 1
            if text in seen or failed >= max_per_sample:
                continue
            seen.add(text)
            failed += 1
            pairs.append(PreferencePair(sample, text, f"onpolicy:{kind}"))
        stats.samples_with_failure += failed > 0
    return pairs, stats
