"""P3 acceptance: ≥80% top-5 chunk overlap across delete + rebuild.

Disposability at the *ranking* level: deleting the index database and
re-running ``reconcile_all`` must produce search results whose top-5 sets
overlap at ≥80% with the pre-rebuild run for the 10-query eval set. This
is the hybrid-search counterpart to the P2 byte-equal CLI disposability
check (which only used BM25 + heading line numbers).

Uses the deterministic fake embedder so the test runs in CI without an
Ollama daemon. The fingerprint changes between an Ollama deploy and a
test run, but as long as both pre and post use the same embedder the
top-5 sets should match — we're validating that the index is disposable,
not that the embedder is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.embeddings import EmbeddingPipeline, _DeterministicEmbeddings
from jarvis.memory.index import Indexer
from jarvis.memory.search import SearchOptions, memory_search
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from tests.fixtures.populate_workspace import populate

_EVAL_QUERIES = [
    "rocket fin",
    "typescript",
    "daily log",
    "jarvis-rebuild",
    "garden",
    "standup",
    "preferences",
    "fast model",
    "grant",
    "reconcile",
]

_OVERLAP_THRESHOLD = 0.80
_K = 5


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


def _top_keys(results) -> set:
    """Use (file_path, start_line, end_line) instead of chunk_id, since the
    chunk autoincrement re-counts after a wipe + rebuild."""
    return {(r.file_path, r.start_line, r.end_line) for r in results}


def _run_eval(indexer: Indexer, embedder: EmbeddingPipeline) -> dict[str, set]:
    out: dict[str, set] = {}
    for q in _EVAL_QUERIES:
        results = memory_search(
            indexer.conn, q, embedder=embedder, options=SearchOptions(k=_K)
        )
        out[q] = _top_keys(results)
    return out


def test_top5_overlap_at_least_80_percent_across_rebuild(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)

    db_path = paths.index_dir / "memory.sqlite"

    # First run: build, embed, query.
    embedder1 = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=768))
    indexer1 = Indexer(db_path, paths.root, embedder=embedder1)
    indexer1.reconcile_all()
    before = _run_eval(indexer1, embedder1)
    indexer1.close()

    # Wipe DB + WAL/SHM siblings; rebuild.
    for ext in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + ext)
        if candidate.exists():
            candidate.unlink()

    embedder2 = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=768))
    indexer2 = Indexer(db_path, paths.root, embedder=embedder2)
    indexer2.reconcile_all()
    after = _run_eval(indexer2, embedder2)
    indexer2.close()

    # Compute per-query Jaccard-style overlap and require average ≥ 0.8.
    overlaps: list[float] = []
    for q in _EVAL_QUERIES:
        b = before[q]
        a = after[q]
        if not b and not a:
            overlaps.append(1.0)
            continue
        if not b or not a:
            overlaps.append(0.0)
            continue
        common = b & a
        overlaps.append(len(common) / max(len(b), len(a)))

    avg = sum(overlaps) / len(overlaps)
    assert avg >= _OVERLAP_THRESHOLD, (
        f"avg top-{_K} overlap was {avg:.2%} (< {_OVERLAP_THRESHOLD:.0%})\n"
        + "\n".join(
            f"  {q!r:30s} overlap={ov:.0%}  before={before[q]}  after={after[q]}"
            for q, ov in zip(_EVAL_QUERIES, overlaps, strict=True)
        )
    )
