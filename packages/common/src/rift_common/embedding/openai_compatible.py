"""OpenAI-compatible ``/embeddings`` endpoint over httpx."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from rift_common.embedding.base import BaseEmbedding, EmbeddingError, Vector
from rift_common.settings import EmbeddingConfig


class OpenAICompatibleEmbedding(BaseEmbedding):
    provider_name = "openai_compatible"

    def __init__(
        self, config: EmbeddingConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        super().__init__(config)
        if not config.base_url or not config.model:
            raise ValueError("openai_compatible embedding requires base_url and model")
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_s,
            transport=transport,
        )

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            resp = self._client.post(
                "/embeddings", json={"model": self.config.model, "input": list(texts)}
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise EmbeddingError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            items = sorted(resp.json()["data"], key=lambda d: d["index"])
            vectors = [[float(x) for x in item["embedding"]] for item in items]
        except (ValueError, KeyError, TypeError) as exc:
            raise EmbeddingError(f"malformed embedding payload: {resp.text[:300]}") from exc
        if len(vectors) != len(texts):
            raise EmbeddingError(f"expected {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def close(self) -> None:
        self._client.close()
