"""Embedding providers + pipeline + cache (BUILD_SPEC §6.3).

Provider stack
--------------
``EmbeddingProvider`` is a Protocol; ``OllamaEmbeddings`` (default,
``nomic-embed-text``, 768d) is the sole built provider in P3.
``EmbeddingPipeline`` wraps a chosen provider with an on-disk LRU cache
(the ``embedding_cache`` table from §5) and returns L2-normalized vectors
so downstream L2-distance queries against ``chunks_vec`` rank identically
to cosine.

P3 is **Ollama-only**. The §5 schema fixes ``chunks_vec`` at 768
dimensions; OpenAI's 1536d wouldn't fit without a schema migration, and
shipping a multi-provider story with that latent dimension mismatch is
worse than shipping fewer providers. P14 revisits multi-provider with a
proper migration path.

L2-normalization is done once here, on-write — query and storage both go
through this pipeline so the geometry stays consistent. Anything new that
needs raw vectors should call ``provider.embed`` directly (and explain why
in a comment). Mixing normalized and unnormalized rows in the same
``chunks_vec`` would silently corrupt ranking.

Fingerprint
-----------
``"<kind>:<model>:<dimensions>"`` — e.g. ``ollama:nomic-embed-text:768``.
Stored on every chunk in ``chunks.embedding_model``. A pipeline whose
fingerprint differs from a chunk's fingerprint treats that chunk as
"unembedded for me" and either re-embeds it (during reconcile) or skips
it in vector search and falls back to BM25 alone (graceful degradation).

Failure modes
-------------
- Ollama unreachable, model not pulled, etc. → providers raise
  ``EmbeddingError``. ``EmbeddingPipeline.embed_batch`` returns one
  ``None`` per failed input rather than partial-success-with-exception,
  so the indexer can still write FTS rows for chunks that failed to embed.
- The cache is best-effort: a SQLite write failure logs a warning and
  keeps going. Search quality is the cache's only product; it must never
  block reconciliation.
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import struct
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "OllamaEmbeddings",
    "EmbeddingPipeline",
    "build_provider_from_config",
    "l2_normalize",
    "fingerprint_for",
    "EMBEDDING_CACHE_MAX_ROWS",
]


EMBEDDING_CACHE_MAX_ROWS = 50_000  # §5: LRU evict at 50k rows.


class EmbeddingError(RuntimeError):
    """Raised by providers when embedding cannot be produced for a batch."""


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def l2_normalize(vec: Sequence[float]) -> list[float]:
    """Return a unit-norm copy of ``vec``. Zero vectors pass through unchanged."""
    s = math.sqrt(sum(x * x for x in vec))
    if s == 0.0:
        return list(vec)
    return [x / s for x in vec]


def fingerprint_for(kind: str, model: str, dimensions: int) -> str:
    return f"{kind}:{model}:{dimensions}"


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """A provider produces embeddings for a batch of strings.

    ``embed_batch`` may raise ``EmbeddingError`` for an *entire* batch
    (e.g. host unreachable). Per-item failures are not modeled — callers
    can chunk into single-item batches if they need item-level recovery.
    """

    kind: str
    model: str
    dimensions: int

    @property
    def fingerprint(self) -> str: ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...


# ---------------------------------------------------------------------------
# Ollama provider — default, local, nomic-embed-text 768d
# ---------------------------------------------------------------------------


@dataclass
class OllamaEmbeddings:
    """Synchronous Ollama embedding client. Uses the ``/api/embed`` endpoint.

    Failure semantics: any non-2xx, network failure, or malformed payload
    raises ``EmbeddingError`` for the entire batch. The pipeline catches it
    and returns ``None`` for each input so the indexer can keep going.
    """

    host: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    timeout_s: float = 60.0
    kind: str = "ollama"

    @property
    def fingerprint(self) -> str:
        return fingerprint_for(self.kind, self.model, self.dimensions)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with httpx.Client(base_url=self.host.rstrip("/"), timeout=self.timeout_s) as c:
                resp = c.post("/api/embed", json={"model": self.model, "input": list(texts)})
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise EmbeddingError(f"Ollama embed failed: {e}") from e

        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama embed returned malformed payload (got "
                f"{len(embeddings) if isinstance(embeddings, list) else 'non-list'} "
                f"vectors for {len(texts)} inputs)"
            )
        for vec in embeddings:
            if not isinstance(vec, list) or len(vec) != self.dimensions:
                raise EmbeddingError(
                    f"Ollama embed dim mismatch: expected {self.dimensions}, "
                    f"got {len(vec) if isinstance(vec, list) else 'non-list'}"
                )
        return [[float(x) for x in v] for v in embeddings]


# ---------------------------------------------------------------------------
# Pipeline — provider + cache + L2 normalization
# ---------------------------------------------------------------------------


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_floats(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


class EmbeddingPipeline:
    """Provider-agnostic embedding pipeline with persistent LRU cache.

    Every output is L2-normalized exactly once (here) — the indexer and
    search code assume unit-norm vectors so the L2 metric in ``chunks_vec``
    ranks like cosine.

    The cache uses the ``embedding_cache`` table created by ``init_schema``.
    Eviction at 50k rows is opportunistic: triggered after batch writes,
    drops the oldest-accessed rows. (Strict per-row LRU would require an
    extra lookup per cache hit; we update ``accessed_at`` on hit but only
    sweep on write.)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache_conn: sqlite3.Connection | None = None,
    ) -> None:
        self.provider = provider
        self.cache_conn = cache_conn

    # ----- properties -----------------------------------------------------

    @property
    def fingerprint(self) -> str:
        return self.provider.fingerprint

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    # ----- public API -----------------------------------------------------

    def embed_one(self, text: str) -> list[float] | None:
        """Embed a single string. Returns None on provider failure."""
        out = self.embed_batch([text])
        return out[0] if out else None

    def embed_batch(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Embed a batch. Per-item ``None`` on failure for that batch.

        Cache hits are read first and skip the provider. Misses are sent in
        a single request to the provider; on full-batch failure, every miss
        becomes ``None``.
        """
        if not texts:
            return []

        # Phase 1 — cache lookup.
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        miss_hashes: list[str] = []

        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(t)
                miss_hashes.append(_hash_text(t))

        if not miss_texts:
            return results

        # Phase 2 — call provider on misses.
        try:
            raw_vecs = self.provider.embed_batch(miss_texts)
        except EmbeddingError as e:
            logger.warning("embedding provider failure (%d items): %s", len(miss_texts), e)
            # Leave all misses as None.
            return results

        # Phase 3 — normalize, write cache, fill results.
        for slot_idx, idx_in_results in enumerate(miss_indices):
            normalized = l2_normalize(raw_vecs[slot_idx])
            results[idx_in_results] = normalized
            self._cache_put(miss_hashes[slot_idx], normalized)

        # Phase 4 — opportunistic LRU sweep.
        self._maybe_evict()
        return results

    # ----- cache helpers --------------------------------------------------

    def _cache_get(self, text: str) -> list[float] | None:
        if self.cache_conn is None:
            return None
        text_hash = _hash_text(text)
        try:
            row = self.cache_conn.execute(
                "SELECT embedding FROM embedding_cache "
                "WHERE text_hash = ? AND model_fingerprint = ?",
                (text_hash, self.fingerprint),
            ).fetchone()
        except sqlite3.Error as e:
            logger.warning("embedding cache read failed: %s", e)
            return None

        if row is None:
            return None

        # Touch accessed_at for LRU.
        try:
            with self.cache_conn:
                self.cache_conn.execute(
                    "UPDATE embedding_cache SET accessed_at = ? "
                    "WHERE text_hash = ? AND model_fingerprint = ?",
                    (int(time.time()), text_hash, self.fingerprint),
                )
        except sqlite3.Error as e:
            logger.warning("embedding cache touch failed: %s", e)

        return _unpack_floats(row["embedding"])

    def _cache_put(self, text_hash: str, vec: Sequence[float]) -> None:
        if self.cache_conn is None:
            return
        try:
            with self.cache_conn:
                self.cache_conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache "
                    "(text_hash, model_fingerprint, embedding, accessed_at) "
                    "VALUES (?, ?, ?, ?)",
                    (text_hash, self.fingerprint, _pack_floats(vec), int(time.time())),
                )
        except sqlite3.Error as e:
            logger.warning("embedding cache write failed: %s", e)

    def _maybe_evict(self) -> None:
        if self.cache_conn is None:
            return
        try:
            row = self.cache_conn.execute(
                "SELECT COUNT(*) AS c FROM embedding_cache"
            ).fetchone()
            n = int(row["c"]) if row else 0
            if n <= EMBEDDING_CACHE_MAX_ROWS:
                return
            # Evict the oldest 5% so we don't sweep on every write.
            to_evict = max(1, n - EMBEDDING_CACHE_MAX_ROWS + n // 20)
            with self.cache_conn:
                self.cache_conn.execute(
                    "DELETE FROM embedding_cache WHERE rowid IN ("
                    "  SELECT rowid FROM embedding_cache "
                    "  ORDER BY accessed_at ASC LIMIT ?"
                    ")",
                    (to_evict,),
                )
        except sqlite3.Error as e:
            logger.warning("embedding cache eviction failed: %s", e)


# ---------------------------------------------------------------------------
# Provider factory from JarvisConfig
# ---------------------------------------------------------------------------


def build_provider_from_config(cfg) -> EmbeddingProvider | None:  # type: ignore[no-untyped-def]
    """Pick the first ``ollama`` provider in ``cfg.embeddings.providers``.

    P3 is Ollama-only; the config schema rejects other ``kind`` values at
    load time, so this loop only ever sees ollama entries. Returns
    ``None`` if the providers list is empty — caller treats that as
    degraded mode (BM25-only).
    """
    for p in cfg.embeddings.providers:
        if p.kind == "ollama":
            return OllamaEmbeddings(
                host=cfg.llm.ollama_host,
                model=p.model,
                dimensions=p.dimensions,
            )
    return None


# ---------------------------------------------------------------------------
# Test-only no-op embedder (importable but kept here for cohesion)
# ---------------------------------------------------------------------------


@dataclass
class _DeterministicEmbeddings:
    """Deterministic hash-based fake embedder.

    Useful in unit tests where we want stable vectors without a network. Not
    re-exported in ``__all__``; tests import it explicitly.
    """

    dimensions: int = 16
    model: str = "fake-deterministic"
    kind: str = "fake"

    @property
    def fingerprint(self) -> str:
        return fingerprint_for(self.kind, self.model, self.dimensions)

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vec: list[float] = []
            for i in range(self.dimensions):
                b = h[i % len(h)]
                # Map [0,255] to [-1,1] roughly.
                vec.append((b - 127.5) / 127.5)
            out.append(vec)
        return out
