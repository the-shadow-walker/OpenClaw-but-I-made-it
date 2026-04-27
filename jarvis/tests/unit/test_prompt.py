"""Unit tests for jarvis.core.prompt system-prompt assembly."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.prompt import assemble_system_prompt
from jarvis.memory.files import write_markdown_atomic
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    p = WorkspacePaths.from_config(
        JarvisConfig(
            paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
        )
    )
    bootstrap_workspace(p)
    return p


def test_dm_includes_memory_user_soul_etc(paths: WorkspacePaths) -> None:
    prompt = assemble_system_prompt(paths, "dm")
    for marker in (
        "# ===== USER.md =====",
        "# ===== MEMORY.md =====",
        "# ===== SOUL.md =====",
        "# ===== AGENTS.md =====",
        "# ===== TOOLS.md =====",
        "# ===== daily/today =====",
    ):
        assert marker in prompt, f"missing section {marker!r}"


def test_group_only_includes_user(paths: WorkspacePaths) -> None:
    prompt = assemble_system_prompt(paths, "group")
    assert "# ===== USER.md =====" in prompt
    # Multi-party leak guard — these MUST NOT appear.
    assert "# ===== MEMORY.md =====" not in prompt
    assert "# ===== SOUL.md =====" not in prompt
    assert "# ===== AGENTS.md =====" not in prompt
    assert "# ===== TOOLS.md =====" not in prompt


def test_heartbeat_includes_heartbeat_user_daily(paths: WorkspacePaths) -> None:
    prompt = assemble_system_prompt(paths, "heartbeat")
    assert "# ===== HEARTBEAT.md =====" in prompt
    assert "# ===== USER.md =====" in prompt
    assert "# ===== AGENTS.md =====" in prompt
    assert "# ===== daily/today =====" in prompt
    # Heartbeat must NOT pull MEMORY/SOUL/TOOLS.
    assert "# ===== MEMORY.md =====" not in prompt
    assert "# ===== SOUL.md =====" not in prompt
    assert "# ===== TOOLS.md =====" not in prompt


def test_dm_with_active_project_appends_project_file(paths: WorkspacePaths) -> None:
    project_path = paths.project("rocket-sim")
    write_markdown_atomic(
        project_path,
        "# rocket-sim\n\nfins are still wrong.\n",
        tmp_dir=paths.tmp_dir,
    )

    prompt = assemble_system_prompt(paths, "dm", active_project_slug="rocket-sim")
    assert "# ===== projects/rocket-sim.md =====" in prompt
    assert "fins are still wrong." in prompt


def test_dm_with_active_project_missing_file_is_silent(paths: WorkspacePaths) -> None:
    """A nonexistent project slug should not raise; just skip."""
    prompt = assemble_system_prompt(paths, "dm", active_project_slug="ghost")
    assert "# ===== projects/ghost.md =====" not in prompt
    # Other DM sections still present.
    assert "# ===== MEMORY.md =====" in prompt


def test_dm_skips_missing_yesterday_daily(paths: WorkspacePaths) -> None:
    """Only today's log exists after bootstrap — yesterday's is silently skipped."""
    yesterday = date.today() - timedelta(days=1)
    assert not paths.daily_log(yesterday).exists()

    prompt = assemble_system_prompt(paths, "dm")
    assert "# ===== daily/today =====" in prompt
    assert "# ===== daily/yesterday =====" not in prompt
    # And the rest of the DM bundle is still present.
    assert "# ===== USER.md =====" in prompt


def test_dm_with_yesterday_log(paths: WorkspacePaths) -> None:
    yesterday = date.today() - timedelta(days=1)
    write_markdown_atomic(
        paths.daily_log(yesterday),
        f"# {yesterday.isoformat()}\n\nyesterday note\n",
        tmp_dir=paths.tmp_dir,
    )
    prompt = assemble_system_prompt(paths, "dm")
    assert "# ===== daily/yesterday =====" in prompt
    assert "yesterday note" in prompt


def test_size_warning_emitted(
    paths: WorkspacePaths, caplog: pytest.LogCaptureFixture
) -> None:
    # Stuff MEMORY.md with > 20K bytes.
    write_markdown_atomic(
        paths.memory_md,
        "x" * 25_000,
        tmp_dir=paths.tmp_dir,
    )
    with caplog.at_level("WARNING", logger="jarvis.core.prompt"):
        assemble_system_prompt(paths, "dm")
    assert any("system prompt is" in rec.message for rec in caplog.records)
