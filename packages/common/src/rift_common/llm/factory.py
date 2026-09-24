"""Registration-based LLM factory."""

from __future__ import annotations

from typing import Any

from rift_common.llm.base import BaseLLM
from rift_common.registry import ProviderRegistry
from rift_common.settings import LLMConfig

_registry: ProviderRegistry[BaseLLM] = ProviderRegistry("llm", BaseLLM)


def register_provider(name: str, cls: type[BaseLLM]) -> None:
    """Register ``cls`` under ``name``; raises ``TypeError`` if it is not a ``BaseLLM``."""
    _registry.register(name, cls)


def available_providers() -> list[str]:
    return _registry.names()


class LLMFactory:
    @staticmethod
    def create(cfg: LLMConfig, **kwargs: Any) -> BaseLLM:
        """Instantiate the provider named by ``cfg.provider``.

        Extra keyword arguments go to the provider constructor (e.g. an httpx transport
        or a mock handler in tests).
        """
        cls = _registry.get(cfg.provider)
        return cls(cfg, **kwargs)
