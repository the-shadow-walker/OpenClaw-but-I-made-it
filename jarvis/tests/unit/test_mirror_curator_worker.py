"""Worker unit tests for jarvis.workers.mirror_curator.

Real SQLite tmp file populated by the test, real central_context.md
target in tmp_path. Threading via ``_wait_until`` polling with a 2s
deadline.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from jarvis.config import MirrorConfig
from jarvis.workers.mirror_curator import MirrorCurator


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _init_db(db_path: Path, *, wal: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    try:
        if wal:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE shared_context (
                key TEXT PRIMARY KEY,
                value TEXT,
                agent_id TEXT,
                created_at REAL,
                expires_at REAL
            )
            """
        )
        conn.execute("CREATE INDEX idx_created ON shared_context(created_at)")
        conn.commit()
    finally:
        conn.close()


def _insert_row(
    db_path: Path,
    key: str,
    value: str = "v",
    agent_id: str = "cmd",
    created_at: float | None = None,
    expires_at: float | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO shared_context(key, value, agent_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                value,
                agent_id,
                time.time() if created_at is None else created_at,
                expires_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _make_curator(tmp_path: Path, *, poll_interval_s: float = 0.05) -> tuple[MirrorCurator, Path, Path]:
    db_path = tmp_path / "memory.db"
    out_path = tmp_path / "central_context.md"
    cfg = MirrorConfig(
        enabled=True,
        central_context_md=out_path,
        shared_db_path=db_path,
        poll_interval_s=poll_interval_s,
    )
    curator = MirrorCurator(
        cfg,
        shared_db_path=db_path,
        central_context_md=out_path,
        tmp_dir=tmp_path / ".tmp",
    )
    return curator, db_path, out_path


# ---------------------------------------------------------------------------
# Lifecycle / atomic write
# ---------------------------------------------------------------------------


def test_curator_creates_mirror_on_first_cycle(tmp_path: Path) -> None:
    curator, db_path, out_path = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="hello")

    curator.start()
    try:
        assert _wait_until(out_path.exists, timeout=2.0)
        content = out_path.read_text(encoding="utf-8")
        assert "convo_a" in content
        assert "hello" in content
    finally:
        curator.stop()


def test_curator_writes_atomic_via_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The os.replace step must be invoked at least once — that's how
    write_markdown_atomic lands the .tmp on the target path.
    """
    curator, db_path, out_path = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="hello")

    import jarvis.memory.files as files_mod

    real_replace = files_mod.os.replace
    calls: list[tuple[str, str]] = []

    def counting_replace(src, dst):
        calls.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(files_mod.os, "replace", counting_replace)

    curator.start()
    try:
        assert _wait_until(lambda: len(calls) >= 1, timeout=2.0)
        # Source path was a .tmp staging file; dest is the mirror path.
        src, dst = calls[0]
        assert ".tmp." in src
        assert dst == str(out_path)
    finally:
        curator.stop()


def test_curator_skips_unchanged_db(tmp_path: Path) -> None:
    curator, db_path, out_path = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="hello")

    curator.start()
    try:
        assert _wait_until(out_path.exists, timeout=2.0)
        first_mtime = out_path.stat().st_mtime_ns
        # Wait through several poll cycles with no DB change.
        time.sleep(0.5)
        second_mtime = out_path.stat().st_mtime_ns
        assert first_mtime == second_mtime
    finally:
        curator.stop()


def test_curator_rewrites_when_new_row_appears(tmp_path: Path) -> None:
    curator, db_path, out_path = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="first", created_at=time.time() - 60)

    curator.start()
    try:
        assert _wait_until(out_path.exists, timeout=2.0)
        first_mtime = out_path.stat().st_mtime_ns

        # Insert a strictly newer row → MAX(created_at) advances.
        _insert_row(db_path, "convo_b", value="second", created_at=time.time())

        assert _wait_until(
            lambda: out_path.stat().st_mtime_ns != first_mtime, timeout=2.0
        )
        content = out_path.read_text(encoding="utf-8")
        assert "second" in content
    finally:
        curator.stop()


# ---------------------------------------------------------------------------
# Read-only contract
# ---------------------------------------------------------------------------


def test_curator_uses_readonly_uri(tmp_path: Path) -> None:
    curator, db_path, _ = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="hello")

    conn = curator._connect_ro()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO shared_context(key, value, agent_id, created_at, expires_at) "
                "VALUES ('x','x','x',0,0)"
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_curator_handles_missing_db_quietly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    curator, db_path, out_path = _make_curator(tmp_path)
    # Do NOT init the DB — the file simply doesn't exist.
    assert not db_path.exists()

    with caplog.at_level(logging.WARNING, logger="jarvis.workers.mirror_curator"):
        curator.start()
        try:
            time.sleep(0.3)  # let several poll cycles attempt + fail
        finally:
            curator.stop()

    # Curator survived, no mirror file written.
    assert not out_path.exists()


def test_curator_handles_db_with_no_wal_yet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Empty (touched but never written-to) memory.db has no WAL/SHM
    sidecars. mode=ro can't create them; connect() raises
    OperationalError. Curator must log WARNING and survive; next tick
    after a writer creates the WAL files succeeds.
    """
    curator, db_path, out_path = _make_curator(tmp_path)
    # Create an empty file with no schema and no WAL — this is the
    # CMD-cold-start race condition.
    db_path.touch()

    with caplog.at_level(logging.WARNING, logger="jarvis.workers.mirror_curator"):
        curator.start()
        try:
            assert _wait_until(
                lambda: any(
                    "cycle failed" in rec.getMessage() for rec in caplog.records
                ),
                timeout=2.0,
            )
            assert not out_path.exists()

            # Now bring up the schema as if CMD did its first write.
            _init_db(db_path)
            _insert_row(db_path, "convo_a", value="hello")

            assert _wait_until(out_path.exists, timeout=2.0)
        finally:
            curator.stop()


def test_curator_continues_after_sqlite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curator, db_path, out_path = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="hello")

    real_read = curator._read_all_rows
    flag = {"raised": False}

    def flaky(conn):
        if not flag["raised"]:
            flag["raised"] = True
            raise sqlite3.OperationalError("simulated transient")
        return real_read(conn)

    monkeypatch.setattr(curator, "_read_all_rows", flaky)

    curator.start()
    try:
        assert _wait_until(out_path.exists, timeout=2.0)
        # On success, _consecutive_failures resets to 0.
        assert curator._consecutive_failures == 0
    finally:
        curator.stop()


def test_curator_escalates_log_after_three_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    curator, db_path, _ = _make_curator(tmp_path)
    _init_db(db_path)
    _insert_row(db_path, "convo_a", value="hello")

    def always_raise(conn):
        raise sqlite3.OperationalError("simulated permanent")

    monkeypatch.setattr(curator, "_read_all_rows", always_raise)

    with caplog.at_level(logging.WARNING, logger="jarvis.workers.mirror_curator"):
        curator.start()
        try:
            # Wait for an ERROR-level record from the curator logger.
            assert _wait_until(
                lambda: any(
                    rec.levelno >= logging.ERROR
                    and "curator unhealthy" in rec.getMessage()
                    for rec in caplog.records
                ),
                timeout=3.0,
            )
        finally:
            curator.stop()


# ---------------------------------------------------------------------------
# Lifecycle idempotency
# ---------------------------------------------------------------------------


def test_curator_stop_joins_thread_within_timeout(tmp_path: Path) -> None:
    curator, db_path, _ = _make_curator(tmp_path)
    _init_db(db_path)

    curator.start()
    t = curator._thread
    assert t is not None and t.is_alive()
    curator.stop(timeout=2.0)
    assert curator._thread is None
    assert not t.is_alive()


def test_curator_stop_is_idempotent(tmp_path: Path) -> None:
    curator, db_path, _ = _make_curator(tmp_path)
    _init_db(db_path)
    curator.start()
    curator.stop()
    curator.stop()  # second call is a noop


def test_curator_double_start_is_noop(tmp_path: Path) -> None:
    curator, db_path, _ = _make_curator(tmp_path)
    _init_db(db_path)
    curator.start()
    t1 = curator._thread
    curator.start()  # should be a noop
    t2 = curator._thread
    assert t1 is t2
    curator.stop()
