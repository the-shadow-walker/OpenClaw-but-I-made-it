"""Unit tests for jarvis.memory.search hybrid path — normalization, MMR, decay."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.embeddings import EmbeddingPipeline, _DeterministicEmbeddings
from jarvis.memory.index import Indexer
from jarvis.memory.search import (
    SearchOptions,
    _decay_factor,
    memory_search,
    min_max_normalize,
    mmr_select,
)
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from tests.fixtures.populate_workspace import populate


@pytest.fixture(autouse=True)
def _approx_tokenizer():
    configure_tokenizer("approximation")


def _make_paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    return paths


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_min_max_normalize_basic():
    out = min_max_normalize({1: 0.0, 2: 5.0, 3: 10.0})
    assert out[1] == 0.0
    assert out[3] == 1.0
    assert pytest.approx(out[2], abs=1e-9) == 0.5


def test_min_max_normalize_all_equal_collapses_to_neutral():
    out = min_max_normalize({1: 3.0, 2: 3.0, 3: 3.0})
    assert all(v == 0.5 for v in out.values())


def test_min_max_normalize_empty():
    assert min_max_normalize({}) == {}


def test_decay_factor_evergreen_skip_via_caller():
    # The function itself just decays; evergreen-skip is the caller's job.
    # Half-life: at t=half_life, decay should be 0.5.
    assert pytest.approx(_decay_factor(0.0, 30.0), abs=1e-9) == 1.0
    assert pytest.approx(_decay_factor(30.0, 30.0), abs=1e-9) == 0.5
    assert pytest.approx(_decay_factor(60.0, 30.0), abs=1e-9) == 0.25


def test_mmr_select_picks_first_by_relevance():
    """Highest relevance candidate is always picked first (no diversity term yet)."""
    relevance = {1: 0.1, 2: 0.9, 3: 0.5}
    vectors = {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.7071, 0.7071]}
    selected = mmr_select([1, 2, 3], relevance, vectors, k=1, lambda_=0.7)
    assert selected == [2]


def test_mmr_select_diverse_when_top_two_are_similar():
    """λ=0.5: similar twin to the top should lose to a less-similar third."""
    relevance = {1: 1.0, 2: 0.95, 3: 0.6}
    vectors = {
        1: [1.0, 0.0],     # top
        2: [0.999, 0.045], # almost identical to 1
        3: [0.0, 1.0],     # orthogonal
    }
    selected = mmr_select([1, 2, 3], relevance, vectors, k=2, lambda_=0.5)
    assert selected[0] == 1
    assert selected[1] == 3   # diversity wins over the near-twin


def test_mmr_select_handles_missing_vectors():
    """Candidates without vectors get a neutral 0 diversity term and stay pickable."""
    relevance = {1: 1.0, 2: 0.5}
    vectors = {1: [1.0, 0.0]}  # 2 has no vector
    selected = mmr_select([1, 2], relevance, vectors, k=2, lambda_=0.7)
    assert set(selected) == {1, 2}


# ---------------------------------------------------------------------------
# End-to-end memory_search with deterministic embedder
# ---------------------------------------------------------------------------


def test_memory_search_roundtrip_with_deterministic_embedder(tmp_path: Path):
    """Hybrid path runs without error, returns results, and logs the query."""
    paths = _make_paths(tmp_path)
    populate(paths)

    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)
    pipe = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=768))
    indexer.embedder = pipe

    stats = indexer.reconcile_all()
    assert stats.chunks_embedded > 0
    assert stats.chunks_embed_failed == 0

    # The deterministic embedder is hash-based noise — vector signal is
    # uncorrelated with semantics. Lean on text weight here so the eval
    # is meaningful; the acceptance test exercises the default weights.
    results = memory_search(
        indexer.conn,
        "trapezoidal",
        embedder=pipe,
        options=SearchOptions(k=5, vector_weight=0.0, text_weight=1.0),
    )
    assert results
    assert any("rocket-sim" in r.file_path for r in results)
    # search_queries was written to.
    n = indexer.conn.execute(
        "SELECT COUNT(*) AS c FROM search_queries"
    ).fetchone()["c"]
    assert n >= 1
    indexer.close()


def test_memory_search_empty_query_returns_empty(tmp_path: Path):
    """Empty/whitespace/punctuation-only query: no candidates → empty result."""
    paths = _make_paths(tmp_path)
    populate(paths)
    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)
    pipe = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=768))
    indexer.embedder = pipe
    indexer.reconcile_all()

    # Punctuation-only sanitizes to empty match query; deterministic embedder
    # would still produce a vector, so we call without an embedder to ensure
    # no candidates emerge anywhere.
    results = memory_search(indexer.conn, "!@#$", embedder=None)
    assert results == []
    indexer.close()


def test_memory_search_evergreen_skips_decay(tmp_path: Path):
    """Evergreen rows are not subjected to decay even if 'old' on disk."""
    paths = _make_paths(tmp_path)
    populate(paths)

    # Backdate every file in the DB to "180 days ago" by editing modified_at.
    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)
    pipe = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=768))
    indexer.embedder = pipe
    indexer.reconcile_all()

    old_ts = int(time.time()) - 180 * 86400
    indexer.conn.execute("UPDATE files SET modified_at = ?", (old_ts,))
    indexer.conn.commit()

    # Use a query that hits both evergreen (MEMORY/USER/SOUL) and non-evergreen
    # files, lean on text weight so we exercise decay deterministically.
    results = memory_search(
        indexer.conn,
        "standup",
        embedder=pipe,
        options=SearchOptions(k=10, vector_weight=0.0, text_weight=1.0),
    )
    assert results
    evergreen_paths = {"MEMORY.md", "USER.md", "SOUL.md"}
    evergreen_decay = [
        r.score_components["decay"] for r in results if r.file_path in evergreen_paths
    ]
    non_evergreen_decay = [
        r.score_components["decay"]
        for r in results
        if r.file_path not in evergreen_paths
    ]
    assert evergreen_decay or non_evergreen_decay
    assert all(d == 1.0 for d in evergreen_decay)
    assert all(d < 1.0 for d in non_evergreen_decay)

    indexer.close()


def test_memory_search_degraded_mode_without_embedder(tmp_path: Path):
    """No embedder → BM25-only fusion, still returns sane results, still logs."""
    paths = _make_paths(tmp_path)
    populate(paths)
    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)
    indexer.reconcile_all()  # no embedder set → no vectors

    results = memory_search(
        indexer.conn,
        "trapezoidal",
        embedder=None,
        options=SearchOptions(k=5),
    )
    assert results
    assert any("rocket-sim" in r.file_path for r in results)
    indexer.close()
