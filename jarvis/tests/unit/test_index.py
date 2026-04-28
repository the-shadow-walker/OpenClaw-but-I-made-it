"""Unit tests for jarvis.memory.index — schema, reconcile, disposability."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.index import Indexer, get_connection, init_schema
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


def test_init_schema_creates_all_tables(tmp_path: Path):
    db = tmp_path / "memory.sqlite"
    conn = get_connection(db)
    init_schema(conn)

    rows = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','index','trigger')"
    ).fetchall()
    names = {r["name"] for r in rows}

    expected_tables = {
        "files",
        "chunks",
        "chunks_fts",
        "chunks_vec",
        "embedding_cache",
        "conversations",
        "search_queries",
    }
    expected_indexes = {
        "idx_chunks_file",
        "idx_chunks_model",
        "idx_emb_cache_accessed",
        "idx_conv_started",
        "idx_query_hash",
        "idx_query_time",
    }
    expected_triggers = {"chunks_ai", "chunks_ad", "chunks_au"}

    missing = (expected_tables | expected_indexes | expected_triggers) - names
    assert not missing, f"missing schema objects: {missing}"
    conn.close()


def test_sqlite_vec_extension_loads(tmp_path: Path):
    db = tmp_path / "memory.sqlite"
    conn = get_connection(db)
    row = conn.execute("SELECT vec_version() AS v").fetchone()
    assert row["v"]  # non-empty version string
    conn.close()


def test_reconcile_inserts_then_noop_on_same_hash(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        target = paths.memory_md
        first = indexer.reconcile(target)
        before = indexer.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        second = indexer.reconcile(target)
        after = indexer.conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        assert first is True
        assert second is False
        assert before == after
    finally:
        indexer.close()


def test_reconcile_after_edit_replaces_chunks(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        target = paths.memory_md
        indexer.reconcile(target)
        ids_before = {
            r["id"]
            for r in indexer.conn.execute(
                "SELECT id FROM chunks WHERE file_path = ?",
                (target.relative_to(paths.root).as_posix(),),
            ).fetchall()
        }
        # Edit the file (different content → different hash).
        target.write_text(target.read_text() + "\n\n## Added\n\nfresh content here.\n")
        changed = indexer.reconcile(target)
        ids_after = {
            r["id"]
            for r in indexer.conn.execute(
                "SELECT id FROM chunks WHERE file_path = ?",
                (target.relative_to(paths.root).as_posix(),),
            ).fetchall()
        }
        assert changed is True
        assert ids_before.isdisjoint(ids_after), "chunk ids must rotate after edit"
    finally:
        indexer.close()


def test_remove_file_cascades_chunks_and_fts(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        target = paths.memory_md
        indexer.reconcile(target)
        rel = target.relative_to(paths.root).as_posix()

        chunk_ids = [
            r["id"]
            for r in indexer.conn.execute(
                "SELECT id FROM chunks WHERE file_path = ?", (rel,)
            ).fetchall()
        ]
        assert chunk_ids, "expected non-empty chunks before removal"

        indexer.remove_file(target)

        files_left = indexer.conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE path = ?", (rel,)
        ).fetchone()["c"]
        chunks_left = indexer.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks WHERE file_path = ?", (rel,)
        ).fetchone()["c"]
        # FTS rows are dropped via the chunks_ad trigger.
        placeholders = ",".join("?" for _ in chunk_ids)
        fts_left = indexer.conn.execute(
            f"SELECT COUNT(*) AS c FROM chunks_fts WHERE rowid IN ({placeholders})",
            chunk_ids,
        ).fetchone()["c"]

        assert files_left == 0
        assert chunks_left == 0
        assert fts_left == 0
    finally:
        indexer.close()


def test_reconcile_all_skips_dotfile_dirs(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)

    # Drop two dotfile-dir markdowns that should be ignored.
    (paths.tmp_dir / "should-not-index.md").write_text("# nope\n\nsecret tmp content.\n")
    (paths.dreams_staging_dir / "candidate.md").write_text("# candidate\n\nfodder.\n")

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        indexer.reconcile_all()
        rows = indexer.conn.execute("SELECT path FROM files").fetchall()
        paths_indexed = [r["path"] for r in rows]
        assert not any(p.startswith(".tmp/") for p in paths_indexed)
        assert not any(p.startswith("memory/.dreams/") for p in paths_indexed)
        # Sanity: the planted files are present.
        assert "MEMORY.md" in paths_indexed
        assert "USER.md" in paths_indexed
    finally:
        indexer.close()


def test_reconcile_all_removes_vanished_files(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)
    extra = paths.projects_dir / "ephemeral.md"
    extra.write_text("# Ephemeral\n\ntemporary project file.\n")

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        indexer.reconcile_all()
        rel = extra.relative_to(paths.root).as_posix()
        before = indexer.conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE path = ?", (rel,)
        ).fetchone()["c"]
        assert before == 1

        extra.unlink()
        stats = indexer.reconcile_all()
        after = indexer.conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE path = ?", (rel,)
        ).fetchone()["c"]
        assert after == 0
        assert stats.files_removed >= 1
    finally:
        indexer.close()


def test_disposability(tmp_path: Path):
    """Delete index, rebuild, chunk row count must match exactly."""
    paths = _make_paths(tmp_path)
    populate(paths)

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        indexer.reconcile_all()
        before = indexer.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks"
        ).fetchone()["c"]
    finally:
        indexer.close()

    # Wipe DB + WAL/SHM siblings.
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db) + suffix)
        if p.exists():
            p.unlink()

    indexer2 = Indexer(db, paths.root)
    try:
        indexer2.reconcile_all()
        after = indexer2.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks"
        ).fetchone()["c"]
    finally:
        indexer2.close()

    assert before == after
    assert before > 0


def test_classifier_evergreen_flags(tmp_path: Path):
    """Evergreen flag must only fire for MEMORY/USER/SOUL — AGENTS stays False."""
    paths = _make_paths(tmp_path)
    populate(paths)

    db = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db, paths.root)
    try:
        indexer.reconcile_all()
        rows = {
            r["path"]: (r["file_kind"], r["evergreen"])
            for r in indexer.conn.execute(
                "SELECT path, file_kind, evergreen FROM files"
            ).fetchall()
        }
        assert rows["MEMORY.md"] == ("memory", 1)
        assert rows["USER.md"] == ("user", 1)
        assert rows["SOUL.md"] == ("soul", 1)
        assert rows["AGENTS.md"] == ("agents", 0)
        assert rows["HEARTBEAT.md"] == ("heartbeat", 0)
        # Daily logs.
        daily_rows = [k for k, v in rows.items() if v[0] == "daily"]
        assert daily_rows, "expected at least one daily-log row"
        # Project rows.
        project_rows = [k for k, v in rows.items() if v[0] == "project"]
        assert any("rocket-sim" in p for p in project_rows)
    finally:
        indexer.close()


def test_vec0_orphans_when_chunks_deleted_directly(tmp_path: Path):
    """Documenting footgun: vec0 virtual tables do NOT honor parent FK CASCADE.

    A direct ``DELETE FROM chunks`` (or ``DELETE FROM files``) bypasses the
    Python-side cleanup in ``Indexer.reconcile`` / ``Indexer.remove_file``
    that explicitly drops matching ``chunks_vec`` rows first. The result is
    orphan rows in ``chunks_vec`` whose ``chunk_id`` no longer references a
    real chunk. This test pins that behavior so anyone adding a third
    delete path sees the bug and routes through ``Indexer.remove_file``.

    If you're here because this test is failing: vec0 finally honors FK
    CASCADE (or we added a database-side trigger). Update the indexer
    methods to drop the manual cleanup and delete this test.
    """
    from jarvis.memory.embeddings import EmbeddingPipeline, _DeterministicEmbeddings

    paths = _make_paths(tmp_path)
    populate(paths)
    db_path = paths.index_dir / "memory.sqlite"

    indexer = Indexer(db_path, paths.root)
    indexer.embedder = EmbeddingPipeline(_DeterministicEmbeddings(dimensions=768))
    indexer.reconcile_all()

    # Sanity: every chunk has a vector.
    chunk_ids = [r["id"] for r in indexer.conn.execute("SELECT id FROM chunks").fetchall()]
    assert chunk_ids
    vec_count_before = indexer.conn.execute(
        "SELECT COUNT(*) AS c FROM chunks_vec"
    ).fetchone()["c"]
    assert vec_count_before == len(chunk_ids)

    # Bypass remove_file: delete a chunk directly. CASCADE behavior here is
    # irrelevant — chunks has no parent FK; we're just simulating "any code
    # path that drops a chunks row without going through the indexer".
    target_id = chunk_ids[0]
    indexer.conn.execute("DELETE FROM chunks WHERE id = ?", (target_id,))
    indexer.conn.commit()

    # The vec0 row for that chunk_id is now an orphan.
    orphan_count = indexer.conn.execute(
        "SELECT COUNT(*) AS c FROM chunks_vec "
        "WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    ).fetchone()["c"]
    assert orphan_count == 1, (
        "expected exactly one orphan vec0 row after a direct chunks delete; "
        "if this fails, vec0 may now honor CASCADE — update indexer methods accordingly"
    )

    indexer.close()
