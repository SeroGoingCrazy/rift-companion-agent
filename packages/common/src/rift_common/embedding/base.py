"""Abstract embedding interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from rift_common.settings import EmbeddingConfig

Vector = list[float]


class EmbeddingError(Exception):
    """Embedding backend failed (unreachable, bad response, missing dependency)."""


class BaseEmbedding(ABC):
    provider_name: ClassVar[str] = ""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[Vector]:
        """Embed ``texts`` in order; returns one vector per text."""

    def embed_one(self, text: str) -> Vector:
        return self.embed([text])[0]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 when either vector is all zeros."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
