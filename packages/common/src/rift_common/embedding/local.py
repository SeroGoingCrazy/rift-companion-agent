"""Local sentence-transformers embedding (Chinese small model), lazily loaded and cached.

Requires the ``local-embedding`` extra: ``uv sync --extra local-embedding`` or
``uv pip install 'rift-common[local-embedding]'``.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Sequence
from typing import Any

from rift_common.embedding.base import BaseEmbedding, EmbeddingError, Vector

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

# Process-wide cache: loading a model takes seconds and hundreds of MB, so every
# LocalEmbedding with the same model name shares one instance.
_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def _load_model(name: str) -> Any:
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(name)
        if model is None:
            try:
                st = importlib.import_module("sentence_transformers")
            except ImportError as exc:
                raise EmbeddingError(
                    "local embedding needs sentence-transformers; "
                    "install the rift-common[local-embedding] extra"
                ) from exc
            model = st.SentenceTransformer(name)
            _MODEL_CACHE[name] = model
        return model


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


class LocalEmbedding(BaseEmbedding):
    provider_name = "local"

    @property
    def model_name(self) -> str:
        return self.config.model or DEFAULT_MODEL

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        model = _load_model(self.model_name)
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in v] for v in vectors]
