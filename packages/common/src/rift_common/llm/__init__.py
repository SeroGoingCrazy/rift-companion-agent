"""LLM providers behind a registration-based factory."""

from rift_common.llm.base import BaseLLM
from rift_common.llm.factory import LLMFactory, available_providers, register_provider
from rift_common.llm.mock import MockLLM, MockNoMatchError, MockRule
from rift_common.llm.openai_compatible import OpenAICompatibleLLM
from rift_common.llm.types import (
    ChatOptions,
    LLMError,
    LLMResponseError,
    LLMResult,
    LLMTimeout,
    LLMUnavailable,
    Message,
    Usage,
)
from rift_common.registry import UnknownProviderError

register_provider(OpenAICompatibleLLM.provider_name, OpenAICompatibleLLM)
register_provider(MockLLM.provider_name, MockLLM)

__all__ = [
    "BaseLLM",
    "ChatOptions",
    "LLMError",
    "LLMFactory",
    "LLMResponseError",
    "LLMResult",
    "LLMTimeout",
    "LLMUnavailable",
    "Message",
    "MockLLM",
    "MockNoMatchError",
    "MockRule",
    "OpenAICompatibleLLM",
    "UnknownProviderError",
    "Usage",
    "available_providers",
    "register_provider",
]
