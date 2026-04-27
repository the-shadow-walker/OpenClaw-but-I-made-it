"""Atomic Markdown I/O primitives — file-first memory layer.

Pure helpers: no globals, no config dependency. Callers pass paths explicitly.

Atomic write strategy:
    Tempfile lives in a sibling ``workspace/.tmp/`` directory (NOT next to the
    target). Same-filesystem rename is atomic on POSIX. The dedicated ``.tmp/``
    directory is naturally excluded from the future P4 watcher by its dotfile
    prefix — see BUILD_SPEC §6.1.

Conversation note: anywhere a docstring needs the word for "context across
turns," we use "conversation" (never "session"); see §22 glossary.
"""

from __future__ import annotations

import contextlib
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

__all__ = [
    "read_markdown",
    "write_markdown_atomic",
    "read_lines",
    "append_to_daily_log",
]


def read_markdown(path: Path) -> str:
    """Return the full UTF-8 contents of ``path``."""
    return path.read_text(encoding="utf-8")


def write_markdown_atomic(path: Path, content: str, *, tmp_dir: Path) -> None:
    """Atomically write ``content`` to ``path`` via a sibling tempfile.

    Steps:
        1. Ensure ``path.parent`` exists.
        2. Write content to ``tmp_dir / "<name>.tmp.<pid>.<rand>"``.
        3. fsync the tempfile.
        4. ``os.replace`` it onto the target (atomic on POSIX same-fs).

    ``tmp_dir`` MUST be on the same filesystem as ``path`` (callers obtain it
    from ``WorkspacePaths.tmp_dir``, which lives at ``workspace/.tmp/``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_name = f"{path.name}.tmp.{os.getpid()}.{uuid4().hex[:8]}"
    tmp_path = tmp_dir / tmp_name

    # Write + fsync so the bytes are durable before the rename.
    with tmp_path.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    try:
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; ignore if already gone.
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def read_lines(path: Path, start: int, end: int) -> str:
    """Return lines [start, end] from ``path``, 1-indexed and inclusive.

    Matches the contract that the future P5 ``memory_get`` tool will expose to
    the LLM. Out-of-range bounds are clipped silently (matches POSIX
    ``sed -n``-style behavior).
    """
    if start < 1:
        raise ValueError(f"start must be >= 1, got {start}")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Convert 1-indexed inclusive to 0-indexed half-open.
    return "".join(lines[start - 1 : end])


def append_to_daily_log(
    daily_log_path: Path,
    text: str,
    tags: list[str] | None = None,
) -> None:
    """Append a timestamped entry to a daily log Markdown file.

    Format: ``[HH:MM] <text> #tag1 #tag2`` on a single line, terminated by ``\\n``.

    Strict: raises ``FileNotFoundError`` if the daily log is missing. The
    expected creator is ``bootstrap_workspace()`` for today's log on first run,
    and the daily-rollover cron (P5+) for subsequent days. Lazy creation here
    would mask a broken rollover.
    """
    if not daily_log_path.exists():
        raise FileNotFoundError(
            f"daily log missing at {daily_log_path}: bootstrap_workspace() creates "
            "today's log; the daily-rollover cron creates each new day's file"
        )

    timestamp = datetime.now().strftime("[%H:%M]")
    tag_part = ""
    if tags:
        tag_part = " " + " ".join(f"#{t.lstrip('#')}" for t in tags)
    line = f"{timestamp} {text}{tag_part}\n"

    with daily_log_path.open("a", encoding="utf-8") as f:
        f.write(line)
