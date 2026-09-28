"""Local sentence-transformers Embedding (e.g. BAAI/bge-small-zh-v1.5).

Runs fully offline once the model is cached, which suits the Chinese knowledge
base: no embedding API key is needed (DeepSeek has no embedding endpoint).
The model is loaded lazily on first ``embed()`` and shared per model name.

Requires the ``local-embedding`` extra (``sentence-transformers``).
"""

from __future__ import annotations

import importlib
import threading
from typing import Any, Dict, List, Optional

from src.libs.embedding.base_embedding import BaseEmbedding

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


class LocalEmbeddingError(RuntimeError):
    """Raised when the local model cannot be loaded or run."""


def _load_model(name: str, device: Optional[str]) -> Any:
    key = f"{name}@{device or 'auto'}"
    with _CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            try:
                st = importlib.import_module("sentence_transformers")
            except ImportError as exc:
                raise LocalEmbeddingError(
                    "local embedding needs sentence-transformers; "
                    "install it with `uv sync --extra local-embedding`"
                ) from exc
            model = st.SentenceTransformer(name, device=device)
            _MODEL_CACHE[key] = model
        return model


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


class LocalEmbedding(BaseEmbedding):
    """sentence-transformers provider returning L2-normalized vectors."""

    def __init__(self, settings: Any, device: Optional[str] = None, **kwargs: Any) -> None:
        self.model = settings.embedding.model or DEFAULT_MODEL
        self.dimension = settings.embedding.dimensions
        self.device = device

    def embed(
        self,
        texts: List[str],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        self.validate_texts(texts)
        model = _load_model(self.model, self.device)
        try:
            vectors = model.encode(list(texts), normalize_embeddings=True)
        except Exception as exc:
            raise LocalEmbeddingError(f"Local embedding failed ({self.model}): {exc}") from exc

        result = [[float(x) for x in vector] for vector in vectors]
        if result and self.dimension and len(result[0]) != self.dimension:
            raise LocalEmbeddingError(
                f"Model {self.model} returned {len(result[0])}-dim vectors, "
                f"but embedding.dimensions is {self.dimension}"
            )
        return result

    def get_dimension(self) -> int:
        return self.dimension
