import json
import sys
import types
from collections.abc import Iterator, Sequence

import httpx
import pytest

from rift_common.embedding import (
    EmbeddingError,
    EmbeddingFactory,
    LocalEmbedding,
    MockEmbedding,
    OpenAICompatibleEmbedding,
    UnknownProviderError,
    cosine_similarity,
    register_provider,
)
from rift_common.embedding.local import clear_model_cache
from rift_common.settings import EmbeddingConfig

# --- mock ---------------------------------------------------------------------


def _mock(dimension: int | None = None) -> MockEmbedding:
    emb = EmbeddingFactory.create(EmbeddingConfig(provider="mock", dimension=dimension))
    assert isinstance(emb, MockEmbedding)
    return emb


def test_mock_same_text_same_vector_across_instances() -> None:
    assert _mock().embed_one("温柔会聊天") == _mock().embed_one("温柔会聊天")


def test_mock_different_texts_cosine_below_one() -> None:
    a, b = _mock().embed(["温柔会聊天", "暴躁老哥带飞"])
    assert cosine_similarity(a, b) < 1.0


def test_mock_shared_characters_are_more_similar() -> None:
    base, near, far = _mock().embed(["温柔会聊天", "温柔爱聊天", "打野带飞"])
    assert cosine_similarity(base, near) > cosine_similarity(base, far)


def test_mock_vectors_are_unit_length_with_configured_dimension() -> None:
    (v,) = _mock(dimension=16).embed(["阿狸"])
    assert len(v) == 16
    assert cosine_similarity(v, v) == pytest.approx(1.0)
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_mock_empty_text_is_zero_vector() -> None:
    (v,) = _mock().embed([""])
    assert not any(v)


# --- cosine / factory ---------------------------------------------------------


def test_cosine_handles_zero_and_mismatch() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_unknown_embedding_provider() -> None:
    with pytest.raises(UnknownProviderError, match="unknown embedding provider 'nope'"):
        EmbeddingFactory.create(EmbeddingConfig(provider="nope"))


def test_register_non_subclass_rejected() -> None:
    with pytest.raises(TypeError):
        register_provider("bad", dict)  # type: ignore[arg-type]


# --- local (sentence-transformers stubbed) ---------------------------------


class _FakeModel:
    instances = 0

    def __init__(self, name: str) -> None:
        _FakeModel.instances += 1
        self.name = name
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = False) -> list[list[float]]:
        self.calls.append((list(texts), normalize_embeddings))
        return [[float(len(t)), 1.0] for t in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    _FakeModel.instances = 0
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.mark.usefixtures("fake_sentence_transformers")
def test_local_is_lazy_and_cached_across_instances() -> None:
    cfg = EmbeddingConfig(provider="local", model="fake/zh-small")
    first = EmbeddingFactory.create(cfg)
    assert isinstance(first, LocalEmbedding)
    assert _FakeModel.instances == 0  # nothing loaded at construction

    assert first.embed(["ab", "c"]) == [[2.0, 1.0], [1.0, 1.0]]
    EmbeddingFactory.create(cfg).embed(["d"])
    assert _FakeModel.instances == 1  # second instance reuses the cached model


@pytest.mark.usefixtures("fake_sentence_transformers")
def test_local_defaults_model_and_normalizes() -> None:
    emb = EmbeddingFactory.create(EmbeddingConfig(provider="local"))
    assert isinstance(emb, LocalEmbedding)
    assert emb.model_name == "BAAI/bge-small-zh-v1.5"
    assert emb.embed([]) == []
    emb.embed(["x"])
    from rift_common.embedding.local import _MODEL_CACHE

    assert _MODEL_CACHE["BAAI/bge-small-zh-v1.5"].calls == [(["x"], True)]


def test_local_without_dependency_raises_readable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    clear_model_cache()
    emb = EmbeddingFactory.create(EmbeddingConfig(provider="local", model="missing"))
    with pytest.raises(EmbeddingError, match="local-embedding"):
        emb.embed(["x"])


# --- openai_compatible ----------------------------------------------------------


def _remote(handler: object) -> OpenAICompatibleEmbedding:
    cfg = EmbeddingConfig(
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        model="text-embedding-3-small",
        api_key="k",
    )
    emb = EmbeddingFactory.create(cfg, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    assert isinstance(emb, OpenAICompatibleEmbedding)
    return emb


def test_openai_embedding_orders_by_index() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        data = [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]
        return httpx.Response(200, json={"data": data})

    assert _remote(handler).embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert str(seen[0].url) == "https://api.example.com/v1/embeddings"
    assert json.loads(seen[0].content) == {"model": "text-embedding-3-small", "input": ["a", "b"]}
    assert seen[0].headers["Authorization"] == "Bearer k"


def test_openai_embedding_errors() -> None:
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        _remote(lambda r: httpx.Response(500, text="x")).embed(["a"])
    with pytest.raises(EmbeddingError, match="expected 2"):
        _remote(
            lambda r: httpx.Response(200, json={"data": [{"index": 0, "embedding": [1]}]})
        ).embed(["a", "b"])


def test_openai_embedding_requires_model_and_url() -> None:
    with pytest.raises(ValueError, match="base_url and model"):
        EmbeddingFactory.create(EmbeddingConfig(provider="openai_compatible"))
