"""Ragas-based evaluator for RAG quality assessment.

This evaluator wraps the Ragas framework to compute LLM-as-Judge metrics:
- Faithfulness: Does the answer stick to the retrieved context?
- Answer Relevancy: Is the answer relevant to the query?
- Context Precision: Are the retrieved chunks relevant and well-ordered?

Design Principles:
- Pluggable: Implements BaseEvaluator interface, swappable via factory.
- Config-Driven: LLM/Embedding backend read from settings.yaml.
- Graceful Degradation: Clear ImportError if ragas not installed.
"""

from __future__ import annotations

import logging
import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metric name constants
FAITHFULNESS = "faithfulness"
ANSWER_RELEVANCY = "answer_relevancy"
CONTEXT_PRECISION = "context_precision"

SUPPORTED_METRICS = {FAITHFULNESS, ANSWER_RELEVANCY, CONTEXT_PRECISION}

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def _project_embeddings(settings: Any) -> Any:
    """Ragas embeddings backed by the project's own provider (e.g. local bge)."""
    from ragas.embeddings.base import BaseRagasEmbedding
    from src.libs.embedding.embedding_factory import EmbeddingFactory

    class ProjectEmbeddings(BaseRagasEmbedding):
        def __init__(self, embedding: Any) -> None:
            super().__init__()
            self._embedding = embedding

        def embed_text(self, text: str, **kwargs: Any) -> List[float]:
            return self._embedding.embed([text])[0]

        async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
            return (await asyncio.to_thread(self._embedding.embed, [text]))[0]

        def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return self._embedding.embed(list(texts))

        async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
            return await asyncio.to_thread(self._embedding.embed, list(texts))

    return ProjectEmbeddings(EmbeddingFactory.create(settings))


def _import_ragas() -> None:
    """Validate that ragas is importable, raising a clear error if not."""
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'ragas' package is required for RagasEvaluator. "
            "Install it with: pip install ragas datasets"
        ) from exc


class RagasEvaluator(BaseEvaluator):
    """Evaluator that uses the Ragas framework for LLM-as-Judge metrics.

    The three metrics configured here do not require reference labels.
    Other Ragas metrics may require them. ``context_precision`` here is
    ContextPrecisionWithoutReference, judged against the generated answer.

    Supported metrics:
        - faithfulness: Measures factual consistency with context.
        - answer_relevancy: Measures how relevant the answer is to the query.
        - context_precision: Measures relevance/ordering of retrieved chunks.

    Example::

        evaluator = RagasEvaluator(settings=settings)
        metrics = evaluator.evaluate(
            query="What is RAG?",
            retrieved_chunks=[{"id": "c1", "text": "RAG is ..."}],
            generated_answer="RAG stands for ...",
        )
        # metrics == {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
    """

    def __init__(
        self,
        settings: Any = None,
        metrics: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize RagasEvaluator.

        Args:
            settings: Application settings (used to configure LLM backend).
            metrics: Metric names to compute. Defaults to all supported.
            **kwargs: Additional parameters (reserved).

        Raises:
            ImportError: If ragas is not installed.
            ValueError: If unsupported metric names are requested.
        """
        _import_ragas()

        self.settings = settings
        self.kwargs = kwargs

        if metrics is None:
            metrics = self._metrics_from_settings(settings)

        normalised = [m.strip().lower() for m in (metrics or [])]
        if not normalised:
            normalised = sorted(SUPPORTED_METRICS)

        unsupported = [m for m in normalised if m not in SUPPORTED_METRICS]
        if unsupported:
            raise ValueError(
                f"Unsupported ragas metrics: {', '.join(unsupported)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_METRICS))}"
            )

        self._metric_names = normalised

    # ── public API ────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Any],
        generated_answer: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Evaluate RAG quality using Ragas LLM-as-Judge metrics.

        Args:
            query: The user query string.
            retrieved_chunks: Retrieved chunks (dicts with 'text' key or strings).
            generated_answer: The generated answer text. Required for Ragas.
            ground_truth: Ignored by these reference-free metrics.
            trace: Optional TraceContext for observability.
            **kwargs: Additional parameters.

        Returns:
            Dictionary mapping metric names to float scores (0.0 – 1.0).

        Raises:
            ValueError: If query/chunks are invalid or generated_answer is missing.
        """
        self.validate_query(query)
        self.validate_retrieved_chunks(retrieved_chunks)

        if not generated_answer or not generated_answer.strip():
            raise ValueError(
                "RagasEvaluator requires a non-empty 'generated_answer'. "
                "Ragas uses LLM-as-Judge and needs the answer text to evaluate."
            )

        contexts = self._extract_texts(retrieved_chunks)

        try:
            result = self._run_ragas(query, contexts, generated_answer)
        except Exception as exc:
            logger.error("Ragas evaluation failed: %s", exc, exc_info=True)
            raise RuntimeError(f"Ragas evaluation failed: {exc}") from exc

        return result

    # ── private helpers ───────────────────────────────────────────

    def _run_ragas(
        self,
        query: str,
        contexts: List[str],
        answer: str,
    ) -> Dict[str, float]:
        """Run all metrics on one event loop so async clients can be reused safely."""
        return asyncio.run(self._arun_ragas(query, contexts, answer))

    async def _arun_ragas(
        self, query: str, contexts: List[str], answer: str,
    ) -> Dict[str, float]:
        """Execute Ragas collections metrics with shared async client lifetimes."""
        # Build LLM / Embedding wrappers from settings
        self._open_clients: List[Any] = []
        llm, embeddings = self._build_wrappers()
        try:
            return await self._score_all(query, contexts, answer, llm, embeddings)
        finally:
            # Close clients on this loop: asyncio.run() closes the loop right after, and
            # clients left for the GC would fail with "Event loop is closed".
            for client in self._open_clients:
                await client.close()

    async def _score_all(
        self, query: str, contexts: List[str], answer: str, llm: Any, embeddings: Any,
    ) -> Dict[str, float]:
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecisionWithoutReference,
        )

        scores: Dict[str, float] = {}

        for metric_name in self._metric_names:
            if metric_name == FAITHFULNESS:
                m = Faithfulness(llm=llm)
                result = await m.ascore(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            elif metric_name == ANSWER_RELEVANCY:
                m = AnswerRelevancy(llm=llm, embeddings=embeddings)
                result = await m.ascore(user_input=query, response=answer)
            elif metric_name == CONTEXT_PRECISION:
                m = ContextPrecisionWithoutReference(llm=llm)
                result = await m.ascore(
                    user_input=query, response=answer, retrieved_contexts=contexts,
                )
            else:
                continue

            # Undefined results (e.g. no claims in a refusal) are not zero-quality answers.
            scores[metric_name] = float(result.value) if result.value is not None else float("nan")

        return scores

    def _build_wrappers(self) -> tuple:
        """Build Ragas LLM and Embedding wrappers from project settings.

        Uses Ragas 0.4+ native API (InstructorLLM + OpenAIEmbeddings)
        instead of deprecated LangchainLLMWrapper.

        Returns:
            Tuple of (llm_wrapper, embeddings_wrapper).
        """
        from openai import AsyncAzureOpenAI, AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings

        if self.settings is None:
            raise ValueError("Settings required to create LLM for Ragas evaluation")

        # ── LLM ──
        llm_cfg = self.settings.llm
        provider = llm_cfg.provider.lower()
        llm_azure_endpoint = getattr(llm_cfg, "azure_endpoint", None)

        # Azure-compatible mode: if azure_endpoint is configured, use Azure
        # client even when provider is "openai" (matches project convention).
        use_azure_llm = (
            provider == "azure"
            or (provider == "openai" and llm_azure_endpoint)
        )

        client_options = {}
        if self.kwargs.get("http_client") is not None:
            client_options["http_client"] = self.kwargs["http_client"]
        if use_azure_llm:
            llm_client = AsyncAzureOpenAI(
                api_key=llm_cfg.api_key,
                azure_endpoint=llm_azure_endpoint or llm_cfg.azure_endpoint,
                api_version=getattr(llm_cfg, "api_version", None) or "2024-02-15-preview",
                **client_options,
            )
        elif provider == "openai":
            llm_client = AsyncOpenAI(api_key=llm_cfg.api_key, **client_options)
        elif provider == "deepseek":
            # OpenAI-compatible API; the key falls back to the env like DeepSeekLLM
            llm_client = AsyncOpenAI(
                api_key=llm_cfg.api_key or os.environ.get("DEEPSEEK_API_KEY"),
                base_url=DEEPSEEK_BASE_URL,
                **client_options,
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider for Ragas: '{provider}'. "
                "Supported: azure, openai, deepseek"
            )

        if "http_client" not in client_options:  # a caller-supplied client stays open
            self._track(llm_client)
        llm = llm_factory(
            llm_cfg.model, client=llm_client,
            max_tokens=self.kwargs.get("judge_max_tokens", 8192),
            temperature=self.kwargs.get("judge_temperature", 0),
        )

        # ── Embeddings ──
        emb_cfg = self.settings.embedding
        emb_provider = emb_cfg.provider.lower()
        emb_azure_endpoint = getattr(emb_cfg, "azure_endpoint", None)

        # Same Azure-compatible mode detection for embeddings
        use_azure_emb = (
            emb_provider == "azure"
            or (emb_provider == "openai" and emb_azure_endpoint)
        )

        if use_azure_emb:
            emb_client = AsyncAzureOpenAI(
                api_key=emb_cfg.api_key,
                azure_endpoint=emb_azure_endpoint or emb_cfg.azure_endpoint,
                api_version=getattr(emb_cfg, "api_version", None) or "2024-02-15-preview",
                **client_options,
            )
        elif emb_provider == "openai":
            emb_client = AsyncOpenAI(api_key=emb_cfg.api_key, **client_options)
        else:
            # Any other provider (local, ollama, ...) goes through the project factory
            return llm, _project_embeddings(self.settings)

        if "http_client" not in client_options:
            self._track(emb_client)
        embeddings = OpenAIEmbeddings(model=emb_cfg.model, client=emb_client)

        return llm, embeddings

    def _track(self, client: Any) -> None:
        if not hasattr(self, "_open_clients"):
            self._open_clients = []
        self._open_clients.append(client)

    def _extract_texts(self, chunks: List[Any]) -> List[str]:
        """Extract text strings from various chunk representations.

        Args:
            chunks: List of chunk dicts, strings, or objects with .text.

        Returns:
            List of text strings.
        """
        texts: List[str] = []
        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
                texts.append(str(text))
            elif hasattr(chunk, "text"):
                texts.append(str(getattr(chunk, "text")))
            else:
                texts.append(str(chunk))
        return texts

    def _metrics_from_settings(self, settings: Any) -> List[str]:
        """Extract metrics list from settings if available."""
        if settings is None:
            return []
        evaluation = getattr(settings, "evaluation", None)
        if evaluation is None:
            return []
        raw_metrics = getattr(evaluation, "metrics", None)
        if raw_metrics is None:
            return []
        # Filter to only ragas-supported metrics
        return [m for m in raw_metrics if m.lower() in SUPPORTED_METRICS]
