"""Unit tests for jarvis.memory.search — BM25 ranking and edge cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.files import write_markdown_atomic
from jarvis.memory.index import Indexer
from jarvis.memory.search import search_bm25
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from tests.fixtures.populate_workspace import populate


@pytest.fixture(autouse=True)
def _approx_tokenizer():
    configure_tokenizer("approximation")


def _populated(tmp_path: Path) -> tuple[WorkspacePaths, Indexer]:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    populate(paths)
    indexer = Indexer(paths.index_dir / "memory.sqlite", paths.root)
    indexer.reconcile_all()
    return paths, indexer


def test_search_returns_empty_on_empty_query(tmp_path: Path):
    paths, indexer = _populated(tmp_path)
    try:
        assert search_bm25(indexer.conn, "") == []
        assert search_bm25(indexer.conn, "   ") == []
        assert search_bm25(indexer.conn, "!@#$%") == []
    finally:
        indexer.close()


def test_search_finds_planted_term(tmp_path: Path):
    paths, indexer = _populated(tmp_path)
    try:
        # "trapezoidal" appears once in projects/rocket-sim.md fin-design section.
        results = search_bm25(indexer.conn, "trapezoidal", k=5)
        assert results
        assert any("rocket-sim" in r.file_path for r in results)
    finally:
        indexer.close()


def test_search_orders_by_relevance(tmp_path: Path):
    """Plant a unique token more in fileA than fileB; fileA must rank ahead."""
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)

    a = paths.projects_dir / "alpha.md"
    b = paths.projects_dir / "beta.md"
    write_markdown_atomic(
        a, "# Alpha\n\nflumberknack flumberknack flumberknack body.\n", tmp_dir=paths.tmp_dir
    )
    write_markdown_atomic(
        b, "# Beta\n\none flumberknack mention only.\n", tmp_dir=paths.tmp_dir
    )

    indexer = Indexer(paths.index_dir / "memory.sqlite", paths.root)
    try:
        indexer.reconcile_all()
        results = search_bm25(indexer.conn, "flumberknack", k=5)
        assert len(results) >= 2
        # alpha should outrank beta.
        ranks = {r.file_path: i for i, r in enumerate(results)}
        assert ranks["projects/alpha.md"] < ranks["projects/beta.md"]
    finally:
        indexer.close()


def test_search_filters_by_file_kind(tmp_path: Path):
    """Same term in MEMORY.md and a project; ``file_kinds=['memory']`` returns only the MEMORY hit."""
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)

    write_markdown_atomic(
        paths.memory_md,
        "# MEMORY.md\n\nGrant likes zorgblatts everywhere.\n",
        tmp_dir=paths.tmp_dir,
    )
    write_markdown_atomic(
        paths.project("foo"),
        "# foo\n\nA project also mentioning zorgblatts.\n",
        tmp_dir=paths.tmp_dir,
    )

    indexer = Indexer(paths.index_dir / "memory.sqlite", paths.root)
    try:
        indexer.reconcile_all()
        unfiltered = search_bm25(indexer.conn, "zorgblatts", k=5)
        assert len(unfiltered) >= 2
        only_memory = search_bm25(indexer.conn, "zorgblatts", k=5, file_kinds=["memory"])
        assert all(r.file_path == "MEMORY.md" for r in only_memory)
        assert only_memory
    finally:
        indexer.close()


def test_search_query_with_punctuation_does_not_crash(tmp_path: Path):
    paths, indexer = _populated(tmp_path)
    try:
        results = search_bm25(indexer.conn, "rocket-sim's fins!")
        # Don't assert content; just assert no exception and a list.
        assert isinstance(results, list)
    finally:
        indexer.close()
