"""Tests for E4 – local sentence-transformers embedding provider.

sentence-transformers is replaced by a fake module, so no model is downloaded.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from src.libs.embedding import embedding_factory, local_embedding
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.embedding.local_embedding import LocalEmbedding, LocalEmbeddingError


class FakeModel:
    instances = 0

    def __init__(self, name, device=None):
        FakeModel.instances += 1
        self.name = name
        self.device = device
        self.calls = []

    def encode(self, texts, normalize_embeddings=False):
        self.calls.append((list(texts), normalize_embeddings))
        return [[float(len(t)), 0.5, 0.25] for t in texts]


@pytest.fixture(autouse=True)
def fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeModel.instances = 0
    local_embedding.clear_model_cache()
    yield
    local_embedding.clear_model_cache()


def _settings(model="BAAI/bge-small-zh-v1.5", dimensions=3, provider="local"):
    return SimpleNamespace(
        embedding=SimpleNamespace(provider=provider, model=model, dimensions=dimensions)
    )


def test_factory_creates_local_provider():
    # other suites clear the registry; re-run the built-in registration under test
    embedding_factory._register_builtin_providers()

    embedding = EmbeddingFactory.create(_settings())

    assert isinstance(embedding, LocalEmbedding)
    assert "local" in EmbeddingFactory.list_providers()
    assert embedding.get_dimension() == 3


def test_embed_returns_normalized_float_vectors():
    embedding = LocalEmbedding(_settings())

    vectors = embedding.embed(["退款", "计费规则"])

    assert vectors == [[2.0, 0.5, 0.25], [4.0, 0.5, 0.25]]
    model = local_embedding._MODEL_CACHE["BAAI/bge-small-zh-v1.5@auto"]
    assert model.calls == [(["退款", "计费规则"], True)]


def test_model_is_loaded_lazily_and_shared():
    first = LocalEmbedding(_settings())
    second = LocalEmbedding(_settings())
    assert FakeModel.instances == 0

    first.embed(["a"])
    second.embed(["b"])

    assert FakeModel.instances == 1


def test_dimension_mismatch_fails_fast():
    embedding = LocalEmbedding(_settings(dimensions=512))

    with pytest.raises(LocalEmbeddingError, match="3-dim vectors.*512"):
        embedding.embed(["段位"])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        LocalEmbedding(_settings()).embed([])


def test_missing_dependency_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(LocalEmbeddingError, match="--extra local-embedding"):
        LocalEmbedding(_settings()).embed(["x"])


def test_encode_errors_are_wrapped(monkeypatch):
    def broken(self, texts, normalize_embeddings=False):
        raise RuntimeError("oom")

    monkeypatch.setattr(FakeModel, "encode", broken)

    with pytest.raises(LocalEmbeddingError, match="oom"):
        LocalEmbedding(_settings()).embed(["x"])
