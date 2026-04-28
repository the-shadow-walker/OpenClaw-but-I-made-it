"""Runtime integration test for the mirror curator wiring.

Exercises the same construction/shutdown pattern as ``jarvis.run.main``
but in-process and against a tmp SQLite. We assert that:

  * mirror.enabled=True spins up a curator that writes the mirror file
    within a couple of polls of startup.
  * mirror.enabled=False does NOT construct a curator (mirror file
    never appears).
  * Shutdown stops the worker thread cleanly.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from jarvis.config import JarvisConfig, MirrorConfig, PathsConfig
from jarvis.workers.mirror_curator import MirrorCurator


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
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
        conn.execute(
            "INSERT INTO shared_context VALUES "
            "('convo_test', 'integration test', 'jarvis', ?, NULL)",
            (time.time(),),
        )
        conn.commit()
    finally:
        conn.close()


def _make_cfg(tmp_path: Path, *, enabled: bool) -> tuple[JarvisConfig, Path, Path]:
    db_path = tmp_path / "memory.db"
    out_path = tmp_path / "central_context.md"
    cfg = JarvisConfig(
        paths=PathsConfig(
            workspace=tmp_path / "ws",
            shared_board=tmp_path / "agent_bin",
        ),
        mirror=MirrorConfig(
            enabled=enabled,
            central_context_md=out_path,
            shared_db_path=db_path,
            poll_interval_s=0.05,
        ),
    )
    return cfg, db_path, out_path


def _spawn_if_enabled(cfg: JarvisConfig, tmp_path: Path) -> MirrorCurator | None:
    """Mirrors the construction branch in jarvis.run.main."""
    if not cfg.mirror.enabled:
        return None
    curator = MirrorCurator(
        cfg.mirror,
        shared_db_path=cfg.mirror.shared_db_path,
        central_context_md=cfg.mirror.central_context_md,
        tmp_dir=tmp_path / ".tmp",
    )
    curator.start()
    return curator


def test_runtime_creates_curator_when_enabled(tmp_path: Path) -> None:
    cfg, db_path, out_path = _make_cfg(tmp_path, enabled=True)
    _init_db(db_path)

    curator = _spawn_if_enabled(cfg, tmp_path)
    assert curator is not None
    try:
        assert _wait_until(out_path.exists, timeout=2.0)
        assert "convo_test" in out_path.read_text(encoding="utf-8")
    finally:
        curator.stop()


def test_runtime_skips_curator_when_disabled(tmp_path: Path) -> None:
    cfg, db_path, out_path = _make_cfg(tmp_path, enabled=False)
    _init_db(db_path)

    curator = _spawn_if_enabled(cfg, tmp_path)
    assert curator is None
    # Give the system a moment; the file must NEVER appear.
    time.sleep(0.3)
    assert not out_path.exists()


def test_runtime_shutdown_stops_curator(tmp_path: Path) -> None:
    cfg, db_path, out_path = _make_cfg(tmp_path, enabled=True)
    _init_db(db_path)

    curator = _spawn_if_enabled(cfg, tmp_path)
    assert curator is not None
    assert _wait_until(out_path.exists, timeout=2.0)
    thread = curator._thread
    assert thread is not None and thread.is_alive()

    curator.stop(timeout=2.0)
    assert curator._thread is None
    assert not thread.is_alive()
