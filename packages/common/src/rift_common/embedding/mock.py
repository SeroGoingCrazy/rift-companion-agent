"""Deterministic hash-based embedding for tests and offline CI.

Character unigrams and bigrams are hashed (blake2b, stable across processes) into a fixed
number of signed buckets and L2-normalized, so identical texts give identical vectors and
texts sharing characters get a higher cosine than unrelated ones.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from itertools import pairwise

from rift_common.embedding.base import BaseEmbedding, Vector

DEFAULT_DIMENSION = 64


class MockEmbedding(BaseEmbedding):
    provider_name = "mock"

    @property
    def dimension(self) -> int:
        return self.config.dimension or DEFAULT_DIMENSION

    def _features(self, text: str) -> list[str]:
        chars = list(text.strip())
        return chars + [a + b for a, b in pairwise(chars)]

    def _vector(self, text: str) -> Vector:
        vec = [0.0] * self.dimension
        for feature in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self.dimension] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._vector(t) for t in texts]
