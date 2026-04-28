"""Workspace file watcher → debounced reconciliation.

Spec reference: BUILD_SPEC §6.1.

This module wires watchdog's filesystem-event stream into ``Indexer.reconcile``
/ ``Indexer.remove_file``. Editor saves (especially vim's atomic-rename) emit
multiple events per save; we debounce per-path so a single human edit produces
at most one reconcile call.

Concurrency contract
====================
Three threads touch state inside this module:

1. **watchdog observer thread** (managed by the ``Observer``) — runs the
   ``FileSystemEventHandler``. It only writes to ``_pending`` under
   ``_lock``. It never touches the indexer.
2. **drainer thread** (we own) — wakes every ``drain_interval_ms``,
   snapshots-and-pops ready items from ``_pending`` under ``_lock``,
   releases the lock, then calls ``indexer.reconcile`` /
   ``indexer.remove_file`` *without* holding the lock.
3. **caller thread** — calls ``start()`` and ``stop()``.

The ``Indexer`` holds a single ``sqlite3.Connection`` and is **not**
thread-safe (one connection per ``Indexer``, no internal locking — see
``jarvis/memory/index.py``). The drainer is the **sole** indexer caller
after ``start()`` returns. A future "let's parallelize indexing" change
must preserve this invariant or move the indexer behind a lock first.

Implementation choices that diverge from the spec pseudocode
============================================================
* ``time.monotonic()`` instead of ``time.time()`` — debounce should
  measure elapsed time, not wall-clock. NTP corrections / manual clock
  fixes can otherwise stall the drainer for hours.
* ``threading.Lock`` around every ``_pending`` access — the
  iter-then-pop sequence in ``_drain`` is a check-then-act race that
  silently drops events under bare-dict access.
* ``os.path.abspath`` for path normalization — ``Path.resolve()`` follows
  symlinks; if the workspace contains symlinks (e.g.
  ``projects/active -> ../archived/...``) ``resolve()`` would coalesce
  events under a different key from the one the user actually edits,
  causing double-indexing or missed updates.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from jarvis.memory.index import Indexer

logger = logging.getLogger(__name__)

__all__ = ["WatcherConfig", "WorkspaceWatcher"]


# WatcherConfig defaults are spec-derived (§6.1). Intentionally NOT exposed via
# JarvisConfig — promote to config only when there's a real operational reason.
# Worst-case save→indexed latency = debounce_ms + drain_interval_ms = 350ms with
# defaults; the 1s exit criterion has comfortable headroom.
@dataclass(frozen=True)
class WatcherConfig:
    debounce_ms: int = 250
    drain_interval_ms: int = 100


class _Handler(FileSystemEventHandler):
    """Watchdog handler — pushes path candidates to the watcher's callback.

    Filtering (dotfile, workspace-relative, .md-only) happens in the
    callback, since the handler doesn't know the workspace root.
    """

    def __init__(self, on_path: Callable[[Path], None]) -> None:
        self._on_path = on_path

    def on_created(self, event: FileSystemEvent) -> None:
        self._maybe(event.src_path, event.is_directory)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._maybe(event.src_path, event.is_directory)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._maybe(event.src_path, event.is_directory)

    def on_moved(self, event: FileSystemEvent) -> None:
        # vim atomic-rename and explicit ``mv`` both produce moved events.
        # Treat both endpoints as pending; the drainer decides reconcile
        # vs remove based on existence at drain time.
        self._maybe(event.src_path, event.is_directory)
        dest = getattr(event, "dest_path", None)
        if dest:
            self._maybe(dest, event.is_directory)

    def _maybe(self, raw: str, is_dir: bool) -> None:
        if is_dir:
            return
        p = Path(raw)
        if p.suffix != ".md":
            return
        self._on_path(p)


class WorkspaceWatcher:
    """Observes ``workspace`` and routes debounced events to ``indexer``.

    Usage::

        with WorkspaceWatcher(paths.root, indexer) as w:
            ...  # observer + drainer running
        # threads cleanly stopped on exit
    """

    def __init__(
        self,
        workspace: Path,
        indexer: Indexer,
        config: WatcherConfig | None = None,
    ) -> None:
        self.workspace = Path(os.path.abspath(workspace))
        self.indexer = indexer
        self._cfg = config or WatcherConfig()

        self._pending: dict[Path, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._observer: Observer | None = None  # type: ignore[assignment]
        self._drainer: threading.Thread | None = None
        self._started = False

    # -- public API --------------------------------------------------------

    def start(self, *, initial_scan: bool = True) -> None:
        """Start the watchdog observer and the drainer thread.

        With ``initial_scan=True`` (default) the watcher first asks the
        indexer to walk the workspace and bring the index in sync with
        on-disk state. This is delegated to ``Indexer.reconcile_all``
        because it already implements the same dotfile-skip / *.md-only
        / vanished-file pruning rules. ``initial_scan=False`` is for
        tests that want to isolate event-driven behavior.
        """
        if self._started:
            return

        if initial_scan:
            try:
                stats = self.indexer.reconcile_all()
                logger.info("watcher: initial scan complete — %s", stats)
            except Exception:
                # Don't refuse to start on an initial-scan failure — log it
                # loudly, but the watcher is more useful running than not.
                logger.exception("watcher: initial scan failed; continuing anyway")

        self._stop.clear()
        self._observer = Observer()
        handler = _Handler(self._on_path)
        self._observer.schedule(handler, str(self.workspace), recursive=True)
        self._observer.start()

        self._drainer = threading.Thread(
            target=self._drain, name="jarvis-watcher-drain", daemon=True
        )
        self._drainer.start()
        self._started = True

    def stop(self, timeout: float = 2.0) -> None:
        """Idempotent shutdown. Best-effort; does not flush ``_pending``.

        If the user is hitting Ctrl-C, they want fast exit, not a final
        reconcile sweep. Any remaining pending paths are logged at DEBUG
        and dropped.
        """
        if not self._started:
            return

        self._stop.set()

        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=timeout)
            except Exception:
                logger.exception("watcher: observer shutdown raised")
            self._observer = None

        if self._drainer is not None:
            self._drainer.join(timeout=timeout)
            self._drainer = None

        with self._lock:
            if self._pending:
                logger.debug(
                    "watcher: dropping %d pending path(s) on shutdown", len(self._pending)
                )
                self._pending.clear()

        self._started = False

    def __enter__(self) -> WorkspaceWatcher:
        self.start()
        return self

    def __exit__(self, *exc) -> None:  # type: ignore[no-untyped-def]
        self.stop()

    # -- internals ---------------------------------------------------------

    def _on_path(self, p: Path) -> None:
        # ``os.path.abspath`` normalizes (cwd + lexical cleanup) without
        # following symlinks. ``Path.resolve()`` would silently coalesce
        # symlinked paths with the resolved path under different keys,
        # causing double-indexing or worse on workspaces that use symlinks.
        abs_path = Path(os.path.abspath(p))
        try:
            rel = abs_path.relative_to(self.workspace)
        except ValueError:
            # Outside the workspace — ignore.
            return
        if any(part.startswith(".") for part in rel.parts):
            return  # skip .index/, .dreams/, .git/, .tmp/, hidden files, etc.
        with self._lock:
            self._pending[abs_path] = time.monotonic()

    def _drain(self) -> None:
        cfg = self._cfg
        while not self._stop.is_set():
            now = time.monotonic()
            ready: list[Path] = []
            with self._lock:
                for path, ts in list(self._pending.items()):
                    if (now - ts) * 1000.0 >= cfg.debounce_ms:
                        ready.append(path)
                for path in ready:
                    self._pending.pop(path, None)

            for path in ready:
                try:
                    if path.exists():
                        self.indexer.reconcile(path)
                    else:
                        self.indexer.remove_file(path)
                except Exception:
                    # One bad file shouldn't kill the indexer for the rest of
                    # the workspace. Log with exc_info so the failure is
                    # observable, never silently swallowed.
                    logger.exception("watcher: failed to handle %s", path)

            # Sleep on the stop event so stop() wakes us promptly.
            self._stop.wait(cfg.drain_interval_ms / 1000.0)
