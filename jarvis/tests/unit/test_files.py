"""Unit tests for jarvis.memory.files atomic-I/O primitives."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from jarvis.memory.files import (
    append_to_daily_log,
    read_lines,
    write_markdown_atomic,
)


def _tmp_dir(root: Path) -> Path:
    d = root / ".tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_write_markdown_atomic_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "out.md"
    write_markdown_atomic(target, "hello world\n", tmp_dir=_tmp_dir(tmp_path))
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello world\n"


def test_write_markdown_atomic_no_partial_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the write step blows up, the target file must remain untouched."""
    target = tmp_path / "preexisting.md"
    target.write_text("ORIGINAL\n", encoding="utf-8")

    # Force the os.replace step to raise so the rename never lands.
    import jarvis.memory.files as files_mod

    def boom(*_a, **_kw):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(files_mod.os, "replace", boom)

    with pytest.raises(OSError, match="simulated rename failure"):
        write_markdown_atomic(target, "NEW\n", tmp_dir=_tmp_dir(tmp_path))

    # Original content preserved.
    assert target.read_text(encoding="utf-8") == "ORIGINAL\n"
    # No leftover staging files in workspace root (they live in .tmp/, and
    # cleanup attempted on failure).
    leftover = list(tmp_path.glob("preexisting.md.tmp.*"))
    assert leftover == []


def test_read_lines_inclusive(tmp_path: Path) -> None:
    p = tmp_path / "f.md"
    p.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    assert read_lines(p, 1, 1) == "line1\n"
    assert read_lines(p, 2, 3) == "line2\nline3\n"
    assert read_lines(p, 1, 4) == "line1\nline2\nline3\nline4\n"

    with pytest.raises(ValueError):
        read_lines(p, 0, 2)
    with pytest.raises(ValueError):
        read_lines(p, 3, 2)


def test_append_to_daily_log_missing_raises(tmp_path: Path) -> None:
    """Strict: the daily log must exist already; bootstrap creates today's log."""
    missing = tmp_path / "memory" / "2026-04-26.md"
    with pytest.raises(FileNotFoundError):
        append_to_daily_log(missing, "anything")


def test_append_to_daily_log_appends(tmp_path: Path) -> None:
    log = tmp_path / "memory" / f"{date.today().isoformat()}.md"
    log.parent.mkdir(parents=True)
    log.write_text(f"# {date.today().isoformat()}\n", encoding="utf-8")

    append_to_daily_log(log, "started rocket-sim refactor", tags=["work", "rocket-sim"])
    append_to_daily_log(log, "no tags here")

    text = log.read_text(encoding="utf-8")
    # Header preserved.
    assert text.startswith(f"# {date.today().isoformat()}\n")
    # Two appended lines.
    assert "started rocket-sim refactor" in text
    assert "#work #rocket-sim" in text
    assert "no tags here" in text
    # Timestamp formatting: each appended line starts with [HH:MM]
    appended_lines = [
        line for line in text.splitlines()
        if line and not line.startswith("#")
    ]
    assert len(appended_lines) == 2
    for line in appended_lines:
        assert line[0] == "["
        # [HH:MM] is 7 chars
        assert line[6] == "]"
