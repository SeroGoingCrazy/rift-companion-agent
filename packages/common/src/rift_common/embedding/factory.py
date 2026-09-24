"""Registration-based embedding factory."""

from __future__ import annotations

from typing import Any

from rift_common.embedding.base import BaseEmbedding
from rift_common.registry import ProviderRegistry
from rift_common.settings import EmbeddingConfig

_registry: ProviderRegistry[BaseEmbedding] = ProviderRegistry("embedding", BaseEmbedding)


def register_provider(name: str, cls: type[BaseEmbedding]) -> None:
    """Register ``cls`` under ``name``; raises ``TypeError`` if it is not a ``BaseEmbedding``."""
    _registry.register(name, cls)


def available_providers() -> list[str]:
    return _registry.names()


class EmbeddingFactory:
    @staticmethod
    def create(cfg: EmbeddingConfig, **kwargs: Any) -> BaseEmbedding:
        cls = _registry.get(cfg.provider)
        return cls(cfg, **kwargs)
