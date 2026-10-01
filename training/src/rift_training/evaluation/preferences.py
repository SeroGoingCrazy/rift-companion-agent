"""Fuzzy matching of free-text preferences (``style_preference``) for the L2 task score.

``style_preference`` is copied from the user's words, so "温柔会聊天" and "温柔一点、会聊天的"
should both count. Texts equal after normalization match outright; otherwise the cosine
similarity of their embeddings (the project's local bge model) must reach a threshold.
Tests and offline runs use ``ExactStyleMatcher`` or the mock embedding.
"""

from __future__ import annotations

import re
from typing import Protocol

from rift_common.embedding import BaseEmbedding, EmbeddingFactory, Vector, cosine_similarity
from rift_common.settings import EmbeddingConfig

#: Calibrated on bge-small-zh-v1.5 with seed style tags: paraphrases ("温柔会聊天" /
#: "性格温柔健谈", "声音好听" / "嗓音好") score >= 0.74, different styles ("幽默" / "安静",
#: "会教学" / "会哄人") <= 0.59.
DEFAULT_STYLE_THRESHOLD = 0.7

_NOISE = re.compile(r"[\s，。！？、,.!?~～]+|的|一点|一些")


def normalize_style(text: str) -> str:
    """Drop whitespace, punctuation and filler ("的", "一点") before comparing."""
    return _NOISE.sub("", text).lower()


class StyleMatcher(Protocol):
    threshold: float

    def similarity(self, expected: str, predicted: str) -> float: ...

    def matches(self, expected: str, predicted: str) -> bool: ...


class ExactStyleMatcher:
    """Normalized string equality (no model needed)."""

    threshold = 1.0

    def similarity(self, expected: str, predicted: str) -> float:
        return 1.0 if normalize_style(expected) == normalize_style(predicted) else 0.0

    def matches(self, expected: str, predicted: str) -> bool:
        return self.similarity(expected, predicted) >= self.threshold


class EmbeddingStyleMatcher:
    """Embedding cosine similarity with a per-text vector cache."""

    def __init__(
        self, embedder: BaseEmbedding, *, threshold: float = DEFAULT_STYLE_THRESHOLD
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self._cache: dict[str, Vector] = {}

    def _vector(self, text: str) -> Vector:
        if text not in self._cache:
            self._cache[text] = self.embedder.embed_one(text)
        return self._cache[text]

    def similarity(self, expected: str, predicted: str) -> float:
        if normalize_style(expected) == normalize_style(predicted):
            return 1.0
        return cosine_similarity(self._vector(expected), self._vector(predicted))

    def matches(self, expected: str, predicted: str) -> bool:
        return self.similarity(expected, predicted) >= self.threshold


def make_style_matcher(
    config: EmbeddingConfig | None, *, threshold: float = DEFAULT_STYLE_THRESHOLD
) -> StyleMatcher:
    """Embedding matcher for ``config`` (``settings.embedding``); exact match without one."""
    if config is None:
        return ExactStyleMatcher()
    return EmbeddingStyleMatcher(EmbeddingFactory.create(config), threshold=threshold)
