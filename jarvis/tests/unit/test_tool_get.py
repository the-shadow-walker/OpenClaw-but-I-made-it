"""Path safety + happy path for memory_get_tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.memory.tool_get import memory_get_tool


def _ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "MEMORY.md").write_text(
        "# MEMORY.md\nline2\nline3\nline4\nline5\n", encoding="utf-8"
    )
    sub = root / "memory"
    sub.mkdir()
    (sub / "2026-04-27.md").write_text("# 2026-04-27\nentry one\n", encoding="utf-8")
    return root


def test_happy_path_returns_lines(tmp_path: Path):
    root = _ws(tmp_path)
    out = memory_get_tool(
        file_path="MEMORY.md", start_line=2, end_line=4, workspace_root=root
    )
    assert out["file_path"] == "MEMORY.md"
    assert out["start_line"] == 2 and out["end_line"] == 4
    assert out["content"].splitlines() == ["line2", "line3", "line4"]


def test_subdir_file_works(tmp_path: Path):
    root = _ws(tmp_path)
    out = memory_get_tool(
        file_path="memory/2026-04-27.md", start_line=1, end_line=2,
        workspace_root=root,
    )
    assert "entry one" in out["content"]


def test_absolute_path_rejected(tmp_path: Path):
    root = _ws(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        memory_get_tool(
            file_path=str(root / "MEMORY.md"),
            start_line=1, end_line=1, workspace_root=root,
        )


def test_dotdot_traversal_rejected(tmp_path: Path):
    root = _ws(tmp_path)
    # Plant a sibling outside the workspace.
    (tmp_path / "secret.md").write_text("nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes workspace root"):
        memory_get_tool(
            file_path="../secret.md", start_line=1, end_line=1,
            workspace_root=root,
        )


def test_missing_file_raises(tmp_path: Path):
    root = _ws(tmp_path)
    with pytest.raises(FileNotFoundError):
        memory_get_tool(
            file_path="missing.md", start_line=1, end_line=1, workspace_root=root,
        )


def test_start_zero_raises(tmp_path: Path):
    root = _ws(tmp_path)
    with pytest.raises(ValueError, match="start"):
        memory_get_tool(
            file_path="MEMORY.md", start_line=0, end_line=2, workspace_root=root,
        )


def test_end_past_eof_clips_silently(tmp_path: Path):
    root = _ws(tmp_path)
    out = memory_get_tool(
        file_path="MEMORY.md", start_line=1, end_line=999, workspace_root=root,
    )
    # 5 newline-terminated lines; clipping returns all.
    assert out["content"].splitlines() == [
        "# MEMORY.md", "line2", "line3", "line4", "line5"
    ]
