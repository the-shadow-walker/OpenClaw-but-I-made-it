"""Unit tests for jarvis.memory.embeddings — provider, pipeline, cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.memory.embeddings import (
    EMBEDDING_CACHE_MAX_ROWS,
    EmbeddingError,
    EmbeddingPipeline,
    _DeterministicEmbeddings,
    fingerprint_for,
    l2_normalize,
)
from jarvis.memory.index import get_connection, init_schema


def test_l2_normalize_unit_norm():
    out = l2_normalize([3.0, 4.0])
    assert pytest.approx(out[0] ** 2 + out[1] ** 2, abs=1e-6) == 1.0


def test_l2_normalize_zero_vector_passthrough():
    assert l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_fingerprint_format():
    assert fingerprint_for("ollama", "nomic-embed-text", 768) == "ollama:nomic-embed-text:768"


def test_pipeline_normalizes_outputs(tmp_path: Path):
    """Every vector returned must be unit-norm regardless of provider raw output."""
    pipe = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=16))
    [v] = pipe.embed_batch(["hello world"])
    assert v is not None
    norm_sq = sum(x * x for x in v)
    assert pytest.approx(norm_sq, abs=1e-6) == 1.0


def test_pipeline_caches_hits(tmp_path: Path):
    """A second call hits the cache and skips the provider."""
    db = tmp_path / "cache.sqlite"
    conn = get_connection(db)
    init_schema(conn)

    class CountingProvider(_DeterministicEmbeddings):
        calls: int = 0

        def embed_batch(self, texts):
            type(self).calls += 1
            return super().embed_batch(texts)

    provider = CountingProvider(dimensions=8)
    pipe = EmbeddingPipeline(provider, cache_conn=conn)

    pipe.embed_batch(["alpha", "beta"])
    assert CountingProvider.calls == 1
    pipe.embed_batch(["alpha", "beta"])  # all cache hits
    assert CountingProvider.calls == 1
    pipe.embed_batch(["alpha", "gamma"])  # one new
    assert CountingProvider.calls == 2

    conn.close()


def test_pipeline_returns_none_on_provider_failure():
    class FailingProvider(_DeterministicEmbeddings):
        def embed_batch(self, texts):
            raise EmbeddingError("network down")

    pipe = EmbeddingPipeline(FailingProvider())
    out = pipe.embed_batch(["a", "b"])
    assert out == [None, None]


def test_pipeline_partial_cache_partial_provider(tmp_path: Path):
    db = tmp_path / "cache.sqlite"
    conn = get_connection(db)
    init_schema(conn)

    pipe = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=8), cache_conn=conn)
    # Prime the cache with one entry.
    pipe.embed_batch(["alpha"])

    class FailNew(_DeterministicEmbeddings):
        def embed_batch(self, texts):
            raise EmbeddingError("provider mid-call failure")

    # Mix cached "alpha" with uncached "beta": provider fails on miss → None.
    pipe2 = EmbeddingPipeline(FailNew(dimensions=8), cache_conn=conn)
    out = pipe2.embed_batch(["alpha", "beta"])
    assert out[0] is not None  # cache hit (same fingerprint)
    assert out[1] is None       # provider failure on miss

    conn.close()


def test_cache_lru_eviction_triggers_above_threshold(tmp_path: Path, monkeypatch):
    """Force a tiny cap and verify writes evict oldest rows."""
    db = tmp_path / "cache.sqlite"
    conn = get_connection(db)
    init_schema(conn)

    monkeypatch.setattr("jarvis.memory.embeddings.EMBEDDING_CACHE_MAX_ROWS", 5)

    pipe = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=4), cache_conn=conn)
    # Insert 10 distinct rows; cap is 5 so eviction must fire.
    pipe.embed_batch([f"text-{i}" for i in range(10)])

    n = conn.execute("SELECT COUNT(*) AS c FROM embedding_cache").fetchone()["c"]
    assert n <= 5  # capped, even if precise count varies by 5% sweep math

    conn.close()


def test_cache_default_max_rows_is_50k():
    """Sanity-check the §5 LRU evict-at-50k constant."""
    assert EMBEDDING_CACHE_MAX_ROWS == 50_000
