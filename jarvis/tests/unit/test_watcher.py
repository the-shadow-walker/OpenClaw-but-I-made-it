"""Unit tests for jarvis.memory.watcher.

Filesystem-event tests use ``_wait_until`` rather than ``time.sleep`` and
pray. macOS fsevents has variable latency, so the deadline is generous;
the actual debounce we exercise (50ms) is much tighter.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.index import Indexer
from jarvis.memory.watcher import WatcherConfig, WorkspaceWatcher
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from tests.fixtures.populate_workspace import populate

FAST = WatcherConfig(debounce_ms=50, drain_interval_ms=10)
DRAIN_GRACE = 0.25  # ~5x drain_interval; covers FS event latency on macOS


@pytest.fixture(autouse=True)
def _approx_tokenizer():
    configure_tokenizer("approximation")


def _wait_until(predicate, timeout=2.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _make_paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    return paths


class _RecordingIndexer:
    """Hand-rolled fake — fast tests skip the SQLite roundtrip."""

    def __init__(self) -> None:
        self.reconcile_calls: list[Path] = []
        self.remove_file_calls: list[Path] = []
        self.reconcile_all_calls: int = 0

    def reconcile(self, p: Path) -> bool:
        self.reconcile_calls.append(p)
        return True

    def remove_file(self, p: Path) -> None:
        self.remove_file_calls.append(p)

    def reconcile_all(self):
        self.reconcile_all_calls += 1
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Event-driven behavior
# ---------------------------------------------------------------------------


def test_create_triggers_reconcile(tmp_path: Path):
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()
    target = paths.root / "projects" / "newfile.md"

    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w  # silence "unused" warning; we only need the context-manager
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# new\nhello\n")
        assert _wait_until(lambda: any(p.name == "newfile.md" for p in idx.reconcile_calls))

    assert any(p.name == "newfile.md" for p in idx.reconcile_calls)


def test_modify_triggers_reconcile(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)
    idx = _RecordingIndexer()

    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w
        # Drain any startup-noise events, then take a baseline.
        time.sleep(DRAIN_GRACE)
        baseline = len(idx.reconcile_calls)

        paths.memory_md.write_text(paths.memory_md.read_text() + "\nnew line\n")
        assert _wait_until(lambda: len(idx.reconcile_calls) > baseline)

    assert len(idx.reconcile_calls) > baseline
    assert any(p.name == "MEMORY.md" for p in idx.reconcile_calls[baseline:])


def test_delete_triggers_remove_file(tmp_path: Path):
    paths = _make_paths(tmp_path)
    target = paths.root / "projects" / "deleteme.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# delete\n")

    idx = _RecordingIndexer()
    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w
        # Wait for the create event to flush.
        assert _wait_until(lambda: any(p.name == "deleteme.md" for p in idx.reconcile_calls))

        target.unlink()
        assert _wait_until(
            lambda: any(p.name == "deleteme.md" for p in idx.remove_file_calls)
        )

    assert any(p.name == "deleteme.md" for p in idx.remove_file_calls)


def test_rapid_10_edits_coalesce_to_one(tmp_path: Path):
    """The spec's central debounce guarantee."""
    paths = _make_paths(tmp_path)
    target = paths.memory_md
    idx = _RecordingIndexer()

    cfg = WatcherConfig(debounce_ms=120, drain_interval_ms=10)
    with WorkspaceWatcher(paths.root, idx, cfg) as w:
        del w
        # Consume any initial event from the bootstrap-written file.
        time.sleep(0.3)
        baseline = sum(1 for p in idx.reconcile_calls if p.name == "MEMORY.md")

        # Burst of 10 edits inside 2 * debounce_ms.
        burst_deadline = time.monotonic() + (2 * cfg.debounce_ms / 1000.0)
        i = 0
        while time.monotonic() < burst_deadline and i < 10:
            target.write_text(f"# v{i}\nhello {i}\n")
            i += 1
            time.sleep(0.005)

        # Wait well past the debounce so the drainer flushes.
        time.sleep(5 * cfg.debounce_ms / 1000.0)

        post = sum(1 for p in idx.reconcile_calls if p.name == "MEMORY.md")

    assert post - baseline == 1, (
        f"expected exactly 1 coalesced reconcile, got {post - baseline} "
        f"(burst writes: {i})"
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_dotfile_dirs_ignored(tmp_path: Path):
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()

    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w
        time.sleep(DRAIN_GRACE)
        baseline_paths = list(idx.reconcile_calls)

        # All under dotfile dirs.
        (paths.root / ".tmp").mkdir(parents=True, exist_ok=True)
        (paths.root / ".dreams").mkdir(parents=True, exist_ok=True)
        (paths.root / ".git").mkdir(parents=True, exist_ok=True)
        (paths.root / ".tmp" / "foo.md").write_text("# foo\n")
        (paths.root / ".dreams" / "bar.md").write_text("# bar\n")
        (paths.root / ".git" / "HEAD.md").write_text("# head\n")

        time.sleep(5 * FAST.debounce_ms / 1000.0)

    new_calls = [p for p in idx.reconcile_calls if p not in baseline_paths]
    for p in new_calls:
        assert ".tmp" not in p.parts
        assert ".dreams" not in p.parts
        assert ".git" not in p.parts


def test_non_md_files_ignored(tmp_path: Path):
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()

    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w
        time.sleep(DRAIN_GRACE)
        baseline = len(idx.reconcile_calls)

        (paths.root / "notes.txt").write_text("hello")
        (paths.root / "image.png").write_bytes(b"\x89PNG\r\n")
        (paths.root / "data.json").write_text("{}")

        time.sleep(5 * FAST.debounce_ms / 1000.0)

    new_calls = idx.reconcile_calls[baseline:]
    for p in new_calls:
        assert p.suffix == ".md", f"non-md slipped through: {p}"


def test_atomic_rename_pattern(tmp_path: Path):
    """Simulate vim atomic-save: write swap+tmp, rename tmp → target.

    vim's ``.swp`` lives next to the target file in the workspace root,
    NOT in ``.tmp/``. The dotfile-skip filter saves us via the leading-dot
    in the file's own basename matching ``part.startswith(".")``. If P1's
    atomic-write ever moves ``.swp`` somewhere else, this assumption needs
    revisiting — the filter checks every part of the relative path,
    including the basename.
    """
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()

    target = paths.memory_md
    swap = paths.root / ".MEMORY.md.swp"
    tmp = paths.root / "MEMORY.md.tmp"

    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w
        time.sleep(DRAIN_GRACE)
        baseline_target = sum(1 for p in idx.reconcile_calls if p.name == "MEMORY.md")

        swap.write_text("vim swap")
        tmp.write_text("# new memory\nfresh content\n")
        os.rename(tmp, target)

        time.sleep(5 * FAST.debounce_ms / 1000.0)
        post_target = sum(1 for p in idx.reconcile_calls if p.name == "MEMORY.md")

    # MEMORY.md.tmp has suffix .tmp not .md — already filtered.
    # .MEMORY.md.swp basename starts with "." — filtered by dotfile rule.
    # Only the final MEMORY.md rename should produce a reconcile.
    assert post_target - baseline_target == 1, (
        f"expected exactly 1 reconcile for MEMORY.md, got {post_target - baseline_target}"
    )
    assert all(
        p.name != "MEMORY.md.tmp" and p.name != ".MEMORY.md.swp"
        for p in idx.reconcile_calls
    )


# ---------------------------------------------------------------------------
# Initial scan
# ---------------------------------------------------------------------------


def test_initial_scan_indexes_existing_files(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)
    idx = _RecordingIndexer()

    w = WorkspaceWatcher(paths.root, idx, FAST)
    try:
        w.start(initial_scan=True)
    finally:
        w.stop()

    assert idx.reconcile_all_calls == 1


def test_initial_scan_skipped_when_disabled(tmp_path: Path):
    paths = _make_paths(tmp_path)
    populate(paths)
    idx = _RecordingIndexer()

    w = WorkspaceWatcher(paths.root, idx, FAST)
    try:
        w.start(initial_scan=False)
    finally:
        w.stop()

    assert idx.reconcile_all_calls == 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_stop_is_idempotent(tmp_path: Path):
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()

    w = WorkspaceWatcher(paths.root, idx, FAST)
    w.start(initial_scan=False)
    w.stop()
    # Second stop is a no-op.
    w.stop()


def test_stop_within_timeout(tmp_path: Path):
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()

    w = WorkspaceWatcher(paths.root, idx, FAST)
    w.start(initial_scan=False)
    t0 = time.monotonic()
    w.stop(timeout=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 3.0, f"stop took too long: {elapsed:.2f}s"


def test_indexer_exception_does_not_kill_drainer(tmp_path: Path, caplog):
    """First file raises; second file still reconciles. Error is logged, not swallowed."""
    paths = _make_paths(tmp_path)

    class _BoomOnAlpha:
        def __init__(self) -> None:
            self.calls: list[Path] = []

        def reconcile(self, p: Path) -> bool:
            self.calls.append(p)
            if p.name == "alpha.md":
                raise RuntimeError("boom")
            return True

        def remove_file(self, p: Path) -> None:
            pass

        def reconcile_all(self):
            return None

    idx = _BoomOnAlpha()
    file_a = paths.root / "projects" / "alpha.md"
    file_b = paths.root / "projects" / "beta.md"
    file_a.parent.mkdir(parents=True, exist_ok=True)

    with (
        caplog.at_level(logging.ERROR, logger="jarvis.memory.watcher"),
        WorkspaceWatcher(paths.root, idx, FAST) as w,
    ):
        del w
        # Let bootstrap-time events drain harmlessly first.
        time.sleep(DRAIN_GRACE)
        file_a.write_text("# a\n")
        assert _wait_until(lambda: any(p.name == "alpha.md" for p in idx.calls))
        # Give the drainer time to take and discard the bad call.
        time.sleep(DRAIN_GRACE)
        file_b.write_text("# b\n")
        assert _wait_until(lambda: any(p.name == "beta.md" for p in idx.calls))

    assert any(p.name == "beta.md" for p in idx.calls), "drainer died on first exception"

    err_records = [
        r
        for r in caplog.records
        if r.levelno == logging.ERROR and r.name == "jarvis.memory.watcher"
    ]
    assert err_records, "no ERROR log emitted from watcher"
    assert any("alpha.md" in r.getMessage() for r in err_records), (
        "ERROR log doesn't mention alpha.md"
    )
    assert any(r.exc_info is not None for r in err_records), (
        "ERROR log has no exc_info — exception is not observable"
    )


def test_clock_jump_backward_does_not_stall(tmp_path: Path, monkeypatch):
    """Patches ``time.monotonic`` in the watcher module to return a backward
    sequence, exercising the fact that the watcher uses monotonic time. If
    a future change "simplifies" back to wall-clock, this test catches it.
    """
    paths = _make_paths(tmp_path)
    idx = _RecordingIndexer()

    # Sequence: first call returns 100.0, then jumps backward to 50.0 and
    # advances slowly. Real elapsed wall-clock will quickly exceed
    # debounce_ms, so the drainer must use *real* monotonic for its sleep
    # but we feed the watcher's reads of monotonic this sequence to model
    # an NTP-corrected clock. We patch only the module-local reference so
    # the drainer's ``self._stop.wait`` (which uses internal threading
    # primitives) is unaffected.
    sequence = iter([100.0] + [50.0 + 0.05 * i for i in range(10_000)])

    def _fake_monotonic() -> float:
        try:
            return next(sequence)
        except StopIteration:
            return 1_000_000.0

    import jarvis.memory.watcher as watcher_mod

    monkeypatch.setattr(watcher_mod.time, "monotonic", _fake_monotonic)

    target = paths.root / "projects" / "clockjump.md"
    target.parent.mkdir(parents=True, exist_ok=True)

    with WorkspaceWatcher(paths.root, idx, FAST) as w:
        del w
        target.write_text("# tick\n")
        assert _wait_until(
            lambda: any(p.name == "clockjump.md" for p in idx.reconcile_calls),
            timeout=3.0,
        )


# ---------------------------------------------------------------------------
# One real-indexer integration smoke
# ---------------------------------------------------------------------------


def test_create_triggers_reconcile_real_indexer(tmp_path: Path):
    """Wire the watcher through to a real Indexer + SQLite to prove the path."""
    paths = _make_paths(tmp_path)
    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)
    try:
        with WorkspaceWatcher(paths.root, indexer, FAST) as w:
            del w
            # Drain any bootstrap-time events.
            time.sleep(DRAIN_GRACE)

            paths.memory_md.write_text(
                paths.memory_md.read_text() + "\nthe magic phrase is purplemonkey-2026\n"
            )

            def _has_phrase() -> bool:
                row = indexer.conn.execute(
                    "SELECT 1 FROM chunks WHERE content LIKE ? LIMIT 1",
                    ("%purplemonkey-2026%",),
                ).fetchone()
                return row is not None

            assert _wait_until(_has_phrase, timeout=3.0)
    finally:
        indexer.close()
