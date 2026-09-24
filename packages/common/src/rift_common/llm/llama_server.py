"""Local llama.cpp server (OpenAI-compatible ``/v1``) for the GGUF slot extractor."""

from __future__ import annotations

import httpx

from rift_common.llm.openai_compatible import OpenAICompatibleLLM


class LlamaServerLLM(OpenAICompatibleLLM):
    """Deterministic defaults (temperature 0, 160 max tokens) plus a health check.

    Timeouts raise ``LLMTimeout`` and connection failures ``LLMUnavailable`` so the routed
    extractor can tell "slow" from "down" when deciding to fall back.
    """

    provider_name = "llama_server"
    default_temperature = 0.0
    default_max_tokens = 160

    @property
    def health_url(self) -> str:
        """llama.cpp serves ``/health`` at the root, not under ``/v1``."""
        root = self.base_url.removesuffix("/v1")
        return f"{root}/health"

    def ping(self, timeout: float = 2.0) -> bool:
        """True when the server is up and the model is loaded."""
        try:
            resp = self._client.get(self.health_url, timeout=timeout)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200
