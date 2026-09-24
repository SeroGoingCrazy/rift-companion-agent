"""Embedding providers behind a registration-based factory."""

from rift_common.embedding.base import BaseEmbedding, EmbeddingError, Vector, cosine_similarity
from rift_common.embedding.factory import EmbeddingFactory, available_providers, register_provider
from rift_common.embedding.local import LocalEmbedding
from rift_common.embedding.mock import MockEmbedding
from rift_common.embedding.openai_compatible import OpenAICompatibleEmbedding
from rift_common.registry import UnknownProviderError

register_provider(LocalEmbedding.provider_name, LocalEmbedding)
register_provider(OpenAICompatibleEmbedding.provider_name, OpenAICompatibleEmbedding)
register_provider(MockEmbedding.provider_name, MockEmbedding)

__all__ = [
    "BaseEmbedding",
    "EmbeddingError",
    "EmbeddingFactory",
    "LocalEmbedding",
    "MockEmbedding",
    "OpenAICompatibleEmbedding",
    "UnknownProviderError",
    "Vector",
    "available_providers",
    "cosine_similarity",
    "register_provider",
]
