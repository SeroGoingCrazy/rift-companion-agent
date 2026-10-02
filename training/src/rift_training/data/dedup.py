"""Near-duplicate removal: against the L2 eval sets (leakage) and within the training set.

The L2 sets were written by hand and frozen before any data was generated; generation never
reads them. This step is the only place training data meets them: a generated sample whose
``user_input`` is the same as an eval input after normalization, or whose embedding cosine
similarity reaches the threshold, is dropped and listed in the data card.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

from rift_common.embedding import BaseEmbedding, Vector, cosine_similarity
from rift_training.evaluation.dataset import SlotSample, normalize_input

#: bge-small-zh cosine at or above which two user inputs count as the same utterance.
DEFAULT_THRESHOLD = 0.92


@dataclass(frozen=True)
class Leak:
    sample_id: str
    user_input: str
    eval_id: str
    eval_input: str
    similarity: float
    reason: str  # "exact" | "embedding"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def nearest_eval(
    samples: Sequence[SlotSample],
    eval_samples: Sequence[SlotSample],
    embedder: BaseEmbedding | None,
) -> list[tuple[SlotSample, SlotSample, float, str]]:
    """For every training sample, its most similar eval sample, the similarity and why."""
    exact = {normalize_input(e.user_input): e for e in eval_samples}
    train_vecs: list[Vector] = []
    eval_vecs: list[Vector] = []
    if embedder is not None and eval_samples:
        train_vecs = embedder.embed([s.user_input for s in samples])
        eval_vecs = embedder.embed([e.user_input for e in eval_samples])
    out: list[tuple[SlotSample, SlotSample, float, str]] = []
    for i, sample in enumerate(samples):
        hit = exact.get(normalize_input(sample.user_input))
        if hit is not None:
            out.append((sample, hit, 1.0, "exact"))
            continue
        if not eval_vecs:
            continue
        sims = [cosine_similarity(train_vecs[i], v) for v in eval_vecs]
        best = max(range(len(sims)), key=sims.__getitem__)
        out.append((sample, eval_samples[best], sims[best], "embedding"))
    return out


def find_leaks(
    samples: Sequence[SlotSample],
    eval_samples: Sequence[SlotSample],
    embedder: BaseEmbedding | None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Leak]:
    return [
        Leak(s.id, s.user_input, e.id, e.user_input, round(sim, 4), reason)
        for s, e, sim, reason in nearest_eval(samples, eval_samples, embedder)
        if reason == "exact" or sim >= threshold
    ]


def _sample_key(sample: SlotSample) -> str:
    context = {
        "state": sample.current_state,
        "candidates": sample.candidates,
        "pending": sample.pending_confirmation,
        "last": sample.history[-1].content if sample.history else "",
    }
    return normalize_input(sample.user_input) + json.dumps(
        context, sort_keys=True, ensure_ascii=False
    )


def dedup_within(
    samples: Sequence[SlotSample],
) -> tuple[list[SlotSample], list[tuple[str, str]]]:
    """Drop samples whose input and context repeat an earlier one; returns (kept, (dup, of))."""
    seen: dict[str, str] = {}
    kept: list[SlotSample] = []
    removed: list[tuple[str, str]] = []
    for sample in samples:
        key = _sample_key(sample)
        if key in seen:
            removed.append((sample.id, seen[key]))
            continue
        seen[key] = sample.id
        kept.append(sample)
    return kept, removed


def similarity_histogram(sims: Sequence[float], edges: Sequence[float]) -> dict[str, int]:
    """Counts per ``[edge_i, edge_i+1)`` bucket, for choosing / reporting the threshold."""
    buckets = {f"{lo}-{hi}": 0 for lo, hi in pairwise(edges)}
    buckets[f">={edges[-1]}"] = 0
    for sim in sims:
        if sim >= edges[-1]:
            buckets[f">={edges[-1]}"] += 1
            continue
        for lo, hi in pairwise(edges):
            if lo <= sim < hi:
                buckets[f"{lo}-{hi}"] += 1
                break
    return buckets
