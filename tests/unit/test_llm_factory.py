from collections.abc import AsyncIterator

import pytest

from rift_common.llm import (
    BaseLLM,
    ChatOptions,
    LLMFactory,
    LLMResult,
    Message,
    UnknownProviderError,
    available_providers,
    register_provider,
)
from rift_common.registry import ProviderRegistry
from rift_common.settings import LLMConfig


class EchoLLM(BaseLLM):
    provider_name = "echo_test"
    default_max_tokens = 99

    def __init__(self, config: LLMConfig, *, prefix: str = "") -> None:
        super().__init__(config)
        self.prefix = prefix
        self.last_opts: ChatOptions | None = None

    def _chat(self, messages: list[Message], opts: ChatOptions) -> LLMResult:
        self.last_opts = opts
        return LLMResult(text=self.prefix + messages[-1]["content"], model=self.model)

    async def _astream(self, messages: list[Message], opts: ChatOptions) -> AsyncIterator[str]:
        for ch in messages[-1]["content"]:
            yield ch


register_provider("echo_test", EchoLLM)


def _cfg(**kw: object) -> LLMConfig:
    return LLMConfig.model_validate({"provider": "echo_test", "model": "m", **kw})


def test_create_registered_provider_with_kwargs() -> None:
    llm = LLMFactory.create(_cfg(), prefix=">")
    assert isinstance(llm, EchoLLM)
    result = llm.chat([{"role": "user", "content": "hi"}])
    assert result.text == ">hi"
    assert result.model == "m"


def test_unknown_provider_is_readable() -> None:
    cfg = LLMConfig(provider="nope", model="m")
    with pytest.raises(UnknownProviderError, match=r"unknown llm provider 'nope'; available: "):
        LLMFactory.create(cfg)


def test_register_non_subclass_raises_type_error() -> None:
    class NotAnLLM:
        pass

    with pytest.raises(TypeError, match="must be a subclass of BaseLLM"):
        register_provider("bad", NotAnLLM)  # type: ignore[arg-type]


def test_register_instance_raises_type_error() -> None:
    with pytest.raises(TypeError):
        register_provider("bad", object())  # type: ignore[arg-type]


def test_reregister_same_class_is_idempotent_but_conflict_rejected() -> None:
    register_provider("echo_test", EchoLLM)

    class Other(EchoLLM):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_provider("echo_test", Other)


def test_available_providers_lists_registered() -> None:
    assert "echo_test" in available_providers()


def test_options_precedence_call_over_config_over_default() -> None:
    llm = LLMFactory.create(_cfg(temperature=0.5, timeout_s=7))
    assert isinstance(llm, EchoLLM)
    llm.chat([{"role": "user", "content": "x"}])
    assert llm.last_opts == ChatOptions(temperature=0.5, max_tokens=99, timeout=7)
    llm.chat(
        [{"role": "user", "content": "x"}],
        temperature=0.1,
        max_tokens=5,
        timeout=1,
        response_format={"type": "json_object"},
    )
    assert llm.last_opts == ChatOptions(
        response_format={"type": "json_object"}, temperature=0.1, max_tokens=5, timeout=1
    )


async def test_astream_yields_deltas() -> None:
    llm = LLMFactory.create(_cfg())
    chunks = [c async for c in llm.astream([{"role": "user", "content": "abc"}])]
    assert chunks == ["a", "b", "c"]


def test_registry_rejects_empty_name() -> None:
    reg: ProviderRegistry[BaseLLM] = ProviderRegistry("llm", BaseLLM)
    with pytest.raises(ValueError, match="non-empty"):
        reg.register("", EchoLLM)
