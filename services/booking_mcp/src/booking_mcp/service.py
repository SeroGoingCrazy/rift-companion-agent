"""``BookingService``: what every tool needs (DB, domain rules, clock, style embedder)."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from booking_mcp.db import SessionFactory
from rift_common.embedding import BaseEmbedding, EmbeddingError, Vector, cosine_similarity
from rift_domain.config import DomainConfig
from rift_domain.matching import CompanionProfile

logger = logging.getLogger(__name__)


def style_text(c: CompanionProfile) -> str:
    """What a companion's style embedding is computed from."""
    return " ".join((*c.tags, c.bio)).strip()


@dataclass
class BookingService:
    factory: SessionFactory
    domain: DomainConfig
    #: Optional: without it ``style_preference`` does not affect ranking.
    embedder: BaseEmbedding | None = None
    clock: Callable[[], datetime] = datetime.now
    _style_vectors: dict[int, Vector] = field(default_factory=dict, repr=False)

    def now(self) -> datetime:
        return self.clock()

    def style_scores(
        self, style: str | None, companions: Sequence[CompanionProfile]
    ) -> Mapping[int, float] | None:
        """Cosine similarity of ``style`` to each companion's tags + bio.

        Companion vectors are cached per process. Returns ``None`` (no style component)
        when there is no style, no embedder, or the embedder fails.
        """
        if not style or self.embedder is None or not companions:
            return None
        try:
            missing = [c for c in companions if c.id not in self._style_vectors]
            if missing:
                vectors = self.embedder.embed([style_text(c) for c in missing])
                self._style_vectors.update(zip((c.id for c in missing), vectors, strict=True))
            query = self.embedder.embed_one(style)
        except EmbeddingError:
            logger.warning("style embedding failed; ranking without style", exc_info=True)
            return None
        return {c.id: cosine_similarity(query, self._style_vectors[c.id]) for c in companions}
