"""Abstract LLM interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any, ClassVar

from rift_common.llm.types import ChatOptions, LLMResult, Message
from rift_common.settings import LLMConfig
from rift_common.trace.context import span


def _first_set[T](*values: T | None) -> T | None:
    return next((v for v in values if v is not None), None)


class BaseLLM(ABC):
    """A chat-completion backend.

    ``chat`` / ``astream`` are template methods: they merge call arguments with config
    defaults and delegate to ``_chat`` / ``_astream``, which providers implement.
    """

    #: Registry name; set by each concrete provider.
    provider_name: ClassVar[str] = ""
    #: Provider-level fallbacks used when neither the call nor the config sets a value.
    default_temperature: ClassVar[float | None] = None
    default_max_tokens: ClassVar[int | None] = None

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @property
    def model(self) -> str:
        return self.config.model

    def options(
        self,
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> ChatOptions:
        """Resolve each option as: call argument > config > provider default."""
        return ChatOptions(
            response_format=response_format,
            temperature=_first_set(temperature, self.config.temperature, self.default_temperature),
            max_tokens=_first_set(max_tokens, self.config.max_tokens, self.default_max_tokens),
            timeout=timeout if timeout is not None else self.config.timeout_s,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> LLMResult:
        opts = self.options(
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        msgs = list(messages)
        with span(
            f"llm:{self.provider_name or type(self).__name__}",
            kind="generation",
            model=self.model,
            input=msgs,
            model_parameters={"temperature": opts.temperature, "max_tokens": opts.max_tokens},
        ) as s:
            result = self._chat(msgs, opts)
            s.set_attrs(output=result.text, usage=result.usage.as_dict())
            return result

    def astream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas."""
        opts = self.options(temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        return self._astream(list(messages), opts)

    @abstractmethod
    def _chat(self, messages: list[Message], opts: ChatOptions) -> LLMResult: ...

    @abstractmethod
    def _astream(self, messages: list[Message], opts: ChatOptions) -> AsyncIterator[str]: ...

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release network resources, if any."""
