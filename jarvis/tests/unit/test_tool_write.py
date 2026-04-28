"""memory_write_tool — daily-log + MEMORY.md branches + missing-log strictness."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.tool_write import memory_write_tool
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    p = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(p)
    return p


def test_daily_appends_timestamped_line(paths):
    res = memory_write_tool(
        content="ate a bagel", where="daily", tags=["food"], paths=paths
    )
    assert res["where"] == "daily"
    text = paths.daily_log().read_text(encoding="utf-8")
    # Last line is the appended entry.
    last = text.splitlines()[-1]
    assert "ate a bagel" in last
    assert "#food" in last
    # Has the [HH:MM] prefix.
    assert last.lstrip().startswith("[")


def test_memory_appends_bullet(paths):
    before = paths.memory_md.read_text(encoding="utf-8")
    res = memory_write_tool(
        content="prefers ruff over flake8", where="memory", paths=paths
    )
    assert res["where"] == "memory"
    after = paths.memory_md.read_text(encoding="utf-8")
    assert after.startswith(before.rstrip("\n"))
    assert "- prefers ruff over flake8\n" in after


def test_memory_strips_existing_bullet_prefix(paths):
    memory_write_tool(
        content="- already a bullet", where="memory", paths=paths
    )
    text = paths.memory_md.read_text(encoding="utf-8")
    # Should be exactly one bullet line, not "- - ".
    assert "- - already a bullet" not in text
    assert "- already a bullet\n" in text


def test_daily_missing_log_raises(paths):
    # Remove today's daily log to verify strict semantics.
    paths.daily_log().unlink()
    with pytest.raises(FileNotFoundError):
        memory_write_tool(content="x", where="daily", paths=paths)


def test_unknown_where_raises(paths):
    with pytest.raises(ValueError, match="unknown 'where'"):
        memory_write_tool(content="x", where="elsewhere", paths=paths)  # type: ignore[arg-type]
