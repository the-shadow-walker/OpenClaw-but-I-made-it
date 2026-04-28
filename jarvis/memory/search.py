"""Hybrid search over the SQLite index (BUILD_SPEC §6.4).

Two entry points:
  * ``search_bm25`` — pure BM25, the P2 default and the degraded-mode fallback
    when no embedder is available. Score is ``-bm25(chunks_fts)`` (raw,
    higher-is-better, unnormalized) — it's intentional that this stays
    raw, since a single-source ranker has nothing to fuse with.
  * ``memory_search`` — the canonical P3 hybrid: pulls a candidate pool
    (``k * candidate_multiplier``) from each of BM25 and vector, min-max
    normalizes each pool, weighted-fuses (0.7 vec / 0.3 text), applies
    30-day temporal decay (skipping evergreen files), then runs MMR
    (λ=0.7) using vectors prefetched once into ``vectors_by_id`` so the
    inner loop never re-embeds.

Score normalization is the silent killer of hybrid search (§6.4). BM25 is
unbounded; cosine is bounded. ALWAYS min-max within each result set
before fusion. Tested with synthetic numbers in ``test_search_hybrid.py``.

MMR caches the query vector once. The inner loop reads pre-fetched chunk
vectors; any chunk lacking a current-fingerprint vector is treated as
maximally dissimilar (so its diversity term is generous) — that way the
hybrid degrades to "BM25-flavored" for partially-embedded indices rather
than dropping unembedded chunks entirely.

P3 logs every ``memory_search`` call to ``search_queries`` (REM Sleep
needs the signal in P11+). ``search_bm25`` does not log — it's a low-level
probe used by tests and the degraded-mode CLI.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import sqlite_vec

from jarvis.memory.embeddings import EmbeddingPipeline, l2_normalize

logger = logging.getLogger(__name__)

__all__ = [
    "SearchResult",
    "SearchOptions",
    "search_bm25",
    "memory_search",
    "min_max_normalize",
    "mmr_select",
]


_QUERY_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_\s]")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    content: str
    start_line: int
    end_line: int
    heading_path: str | None
    score: float                          # higher is better
    score_components: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchOptions:
    k: int = 10
    candidate_multiplier: int = 4
    vector_weight: float = 0.7
    text_weight: float = 0.3
    mmr_lambda: float = 0.7
    half_life_days: float = 30.0
    file_kinds: list[str] | None = None
    include_conversations: bool = False   # honored in P5+; P3 has no transcripts


# ---------------------------------------------------------------------------
# Query escaping (shared with search_bm25)
# ---------------------------------------------------------------------------


def _sanitize_query(query: str) -> str:
    """Strip non-alphanumerics; collapse whitespace; return MATCH-safe string."""
    sanitized = _QUERY_SANITIZE_RE.sub(" ", query)
    tokens = [t for t in sanitized.split() if t]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# BM25-only search (degraded-mode + low-level probe)
# ---------------------------------------------------------------------------


def search_bm25(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    file_kinds: list[str] | None = None,
) -> list[SearchResult]:
    """BM25-only search. Used by tests and the BM25-only CLI fallback.

    Score is ``-bm25(chunks_fts)`` raw — no min-max here because a
    single-source ranker has nothing to fuse with. The hybrid path
    re-pulls these as one candidate pool and normalizes inside
    ``memory_search``.
    """
    match_query = _sanitize_query(query)
    if not match_query:
        return []

    sql_parts = [
        "SELECT c.id AS chunk_id, c.file_path AS file_path, c.content AS content,",
        "       c.start_line AS start_line, c.end_line AS end_line,",
        "       c.heading_path AS heading_path,",
        "       -bm25(chunks_fts) AS score",
        "FROM chunks_fts",
        "JOIN chunks c ON c.id = chunks_fts.rowid",
        "JOIN files  f ON f.path = c.file_path",
        "WHERE chunks_fts MATCH ?",
    ]
    params: list[Any] = [match_query]

    if file_kinds:
        placeholders = ",".join("?" for _ in file_kinds)
        sql_parts.append(f"  AND f.file_kind IN ({placeholders})")
        params.extend(file_kinds)

    sql_parts.append("ORDER BY score DESC LIMIT ?")
    params.append(k)

    sql = "\n".join(sql_parts)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        SearchResult(
            chunk_id=r["chunk_id"],
            file_path=r["file_path"],
            content=r["content"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            heading_path=r["heading_path"],
            score=float(r["score"]),
            score_components={"bm25": float(r["score"])},
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Score helpers — pure functions, easy to unit-test
# ---------------------------------------------------------------------------


def min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a ``{key: score}`` map to ``[0.0, 1.0]``.

    All-equal scores (lo == hi) collapse to 0.5 — neutral, doesn't bias
    fusion when one channel had no signal variance.
    """
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi == lo:
        return {k: 0.5 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


def _decay_factor(age_days: float, half_life_days: float) -> float:
    """Exponential decay: 1.0 at age 0, 0.5 at age == half_life."""
    if half_life_days <= 0:
        return 1.0
    return math.exp(-math.log(2.0) * max(0.0, age_days) / half_life_days)


def _cosine_similarity_unit(a: list[float], b: list[float]) -> float:
    """Cosine similarity for already-unit-norm vectors. Unsafe if not unit-norm."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))


def mmr_select(
    candidates: list[int],
    relevance: dict[int, float],
    vectors: dict[int, list[float]],
    k: int,
    lambda_: float = 0.7,
) -> list[int]:
    """Greedy MMR. Picks ``k`` ids from ``candidates`` maximizing
    ``λ * rel(i) - (1 - λ) * max_j sim(i, j)`` over already-selected ``j``.

    Candidates without a vector get a neutral diversity term (max-sim == 0)
    so they can still be picked when the embedded portion exhausts.
    """
    if k <= 0 or not candidates:
        return []

    selected: list[int] = []
    remaining = [c for c in candidates if c in relevance]

    # Take the first by pure relevance, with chunk_id asc as a deterministic
    # tiebreaker so two identical fused scores always resolve the same way
    # across rebuilds. (CPython dict insertion order isn't a contract worth
    # leaning on for ranking determinism.)
    remaining.sort(key=lambda c: (-relevance[c], c))
    if not remaining:
        return []
    first = remaining.pop(0)
    selected.append(first)

    while remaining and len(selected) < k:
        best_id = None
        best_score = -math.inf
        # Iterate in chunk_id-ascending order so MMR ties also break
        # toward the lower id deterministically.
        for c in sorted(remaining):
            rel = relevance[c]
            v_c = vectors.get(c)
            if v_c is None:
                max_sim = 0.0
            else:
                sims = []
                for s in selected:
                    v_s = vectors.get(s)
                    if v_s is None:
                        continue
                    sims.append(_cosine_similarity_unit(v_c, v_s))
                max_sim = max(sims) if sims else 0.0
            score = lambda_ * rel - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_id = c
        if best_id is None:
            break
        selected.append(best_id)
        remaining.remove(best_id)
    return selected


# ---------------------------------------------------------------------------
# memory_search — the canonical P3 hybrid entry point
# ---------------------------------------------------------------------------


def _hash_query(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_bm25(
    conn: sqlite3.Connection,
    match_query: str,
    pool_size: int,
    file_kinds: list[str] | None,
) -> dict[int, float]:
    """Return ``{chunk_id: -bm25_score}`` (higher = better)."""
    sql = (
        "SELECT c.id AS chunk_id, -bm25(chunks_fts) AS score "
        "FROM chunks_fts "
        "JOIN chunks c ON c.id = chunks_fts.rowid "
        "JOIN files  f ON f.path = c.file_path "
        "WHERE chunks_fts MATCH ?"
    )
    params: list[Any] = [match_query]
    if file_kinds:
        placeholders = ",".join("?" for _ in file_kinds)
        sql += f" AND f.file_kind IN ({placeholders})"
        params.extend(file_kinds)
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(pool_size)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {int(r["chunk_id"]): float(r["score"]) for r in rows}


def _candidate_vector(
    conn: sqlite3.Connection,
    query_vec: list[float],
    pool_size: int,
    fingerprint: str,
    file_kinds: list[str] | None,
) -> dict[int, float]:
    """Return ``{chunk_id: cosine_similarity}`` for the top vector candidates.

    L2 distance on unit vectors equals ``2 - 2*cos``; we pull more than we
    need from vec0 (since file_kind filter happens after) and convert to
    cosine similarity ∈ [-1, 1] for fusion.
    """
    # Pull a generous pool so the post-filter still has enough candidates.
    raw_limit = pool_size * 4 if file_kinds else pool_size
    try:
        rows = conn.execute(
            "SELECT v.chunk_id AS chunk_id, v.distance AS distance, "
            "       c.file_path AS file_path "
            "FROM chunks_vec v "
            "JOIN chunks c ON c.id = v.chunk_id "
            "WHERE v.embedding MATCH ? "
            "  AND k = ? "
            "  AND c.embedding_model = ? "
            "ORDER BY v.distance",
            (sqlite_vec.serialize_float32(query_vec), raw_limit, fingerprint),
        ).fetchall()
    except sqlite3.OperationalError as e:
        logger.warning("vector candidate query failed: %s", e)
        return {}

    out: dict[int, float] = {}
    kept_paths: set[str] = set()
    if file_kinds:
        kept_paths = {
            r["path"]
            for r in conn.execute(
                "SELECT path FROM files WHERE file_kind IN ("
                + ",".join("?" for _ in file_kinds)
                + ")",
                tuple(file_kinds),
            ).fetchall()
        }
        if not kept_paths:
            return {}

    for r in rows:
        if file_kinds and r["file_path"] not in kept_paths:
            continue
        # L2 distance on unit vectors → cosine sim = 1 - dist^2 / 2.
        d = float(r["distance"])
        cos = 1.0 - (d * d) / 2.0
        out[int(r["chunk_id"])] = cos
        if len(out) >= pool_size:
            break
    return out


def _fetch_chunk_meta(
    conn: sqlite3.Connection, chunk_ids: list[int]
) -> dict[int, dict[str, Any]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT c.id, c.file_path, c.content, c.start_line, c.end_line, "
        f"       c.heading_path, c.embedding_model, "
        f"       f.evergreen, f.modified_at "
        f"FROM chunks c JOIN files f ON f.path = c.file_path "
        f"WHERE c.id IN ({placeholders})",
        tuple(chunk_ids),
    ).fetchall()
    return {int(r["id"]): dict(r) for r in rows}


def _fetch_chunk_vectors(
    conn: sqlite3.Connection, chunk_ids: list[int]
) -> dict[int, list[float]]:
    """Pre-fetch unit vectors for a batch of chunk_ids. Skips chunks lacking one."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    try:
        rows = conn.execute(
            f"SELECT chunk_id, embedding FROM chunks_vec "
            f"WHERE chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        ).fetchall()
    except sqlite3.OperationalError as e:
        logger.warning("chunk vector pre-fetch failed: %s", e)
        return {}

    out: dict[int, list[float]] = {}
    for r in rows:
        blob = r["embedding"]
        if blob is None:
            continue
        # vec0 stores rows as little-endian float32 blobs.
        n = len(blob) // 4
        vec = list(struct.unpack(f"<{n}f", blob))
        out[int(r["chunk_id"])] = vec
    return out


def _log_query(conn: sqlite3.Connection, query: str, result_count: int) -> None:
    """Append a row to ``search_queries`` (REM Sleep signal source)."""
    try:
        with conn:
            conn.execute(
                "INSERT INTO search_queries (query, query_hash, queried_at, result_count) "
                "VALUES (?, ?, ?, ?)",
                (query, _hash_query(query), int(time.time()), int(result_count)),
            )
    except sqlite3.Error as e:
        logger.warning("search_queries log write failed: %s", e)


def memory_search(
    conn: sqlite3.Connection,
    query: str,
    embedder: EmbeddingPipeline | None = None,
    options: SearchOptions | None = None,
) -> list[SearchResult]:
    """Hybrid BM25 + vector search with MMR + temporal decay (§6.4).

    Without an ``embedder`` (or when its provider fails on the query),
    vector retrieval is skipped and the result reduces to normalized
    BM25 + decay + MMR-by-vectors-prefetched. That's the documented
    degraded path.
    """
    options = options or SearchOptions()
    pool_size = max(options.k, options.k * options.candidate_multiplier)

    match_query = _sanitize_query(query)
    bm25_scores: dict[int, float] = {}
    if match_query:
        bm25_scores = _candidate_bm25(conn, match_query, pool_size, options.file_kinds)

    # Embed the query exactly once (cached for MMR's inner loop too).
    vec_scores: dict[int, float] = {}
    query_vec: list[float] | None = None
    fingerprint: str | None = None
    if embedder is not None:
        try:
            qv = embedder.embed_one(query)
        except Exception as e:  # provider misbehaving outside our wrapper
            logger.warning("query embed raised unexpectedly: %s", e)
            qv = None
        if qv is not None:
            # The pipeline already L2-normalizes; defensive re-normalize is
            # cheap and keeps invariants explicit.
            query_vec = l2_normalize(qv)
            fingerprint = embedder.fingerprint
            vec_scores = _candidate_vector(
                conn, query_vec, pool_size, fingerprint, options.file_kinds
            )

    if not bm25_scores and not vec_scores:
        _log_query(conn, query, 0)
        return []

    # Min-max normalize each pool independently (§6.4).
    bm25_norm = min_max_normalize(bm25_scores)
    vec_norm = min_max_normalize(vec_scores)

    # Union of candidate ids; missing-channel scores default to 0.
    all_ids = sorted(set(bm25_norm) | set(vec_norm))
    meta = _fetch_chunk_meta(conn, all_ids)

    now = time.time()
    fused: dict[int, float] = {}
    components: dict[int, dict[str, float]] = {}
    for cid in all_ids:
        m = meta.get(cid)
        if m is None:
            continue
        b = bm25_norm.get(cid, 0.0)
        v = vec_norm.get(cid, 0.0)
        base = options.vector_weight * v + options.text_weight * b

        if m["evergreen"]:
            decay = 1.0
        else:
            age_days = max(0.0, (now - float(m["modified_at"])) / 86400.0)
            decay = _decay_factor(age_days, options.half_life_days)

        fused[cid] = base * decay
        components[cid] = {
            "bm25": float(bm25_scores.get(cid, 0.0)),
            "bm25_norm": b,
            "vector": float(vec_scores.get(cid, 0.0)),
            "vector_norm": v,
            "decay": decay,
            "fused": fused[cid],
        }

    if not fused:
        _log_query(conn, query, 0)
        return []

    # Pre-fetch vectors for MMR diversity term (§6.4 — no re-embedding).
    # Sort by fused score desc with chunk_id asc as deterministic tiebreaker.
    candidate_ids = sorted(fused.keys(), key=lambda c: (-fused[c], c))
    candidate_ids = candidate_ids[: pool_size]
    vectors_by_id = _fetch_chunk_vectors(conn, candidate_ids)

    selected = mmr_select(
        candidate_ids, fused, vectors_by_id, options.k, lambda_=options.mmr_lambda
    )

    results: list[SearchResult] = []
    for cid in selected:
        m = meta[cid]
        c = components[cid]
        results.append(
            SearchResult(
                chunk_id=cid,
                file_path=m["file_path"],
                content=m["content"],
                start_line=m["start_line"],
                end_line=m["end_line"],
                heading_path=m["heading_path"],
                score=fused[cid],
                score_components=c,
            )
        )

    _log_query(conn, query, len(results))
    return results
