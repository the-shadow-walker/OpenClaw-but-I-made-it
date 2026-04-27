"""Unit tests for jarvis.memory.workspace path resolution + bootstrap."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


def test_bootstrap_creates_all_files_and_dirs(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_config(
        JarvisConfig(
            paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
        )
    )
    bootstrap_workspace(paths)

    # Directories.
    for d in (
        paths.root,
        paths.projects_dir,
        paths.memory_dir,
        paths.conversations_dir,
        paths.index_dir,
        paths.dreams_staging_dir,
        paths.tmp_dir,
    ):
        assert d.is_dir(), f"{d} should be a directory"

    # Top-level Markdown files.
    for md in (
        paths.memory_md,
        paths.user_md,
        paths.soul_md,
        paths.agents_md,
        paths.tools_md,
        paths.heartbeat_md,
        paths.dreams_md,
    ):
        assert md.is_file(), f"{md} should exist"
        # Must be non-empty starter templates.
        assert md.read_text(encoding="utf-8").strip(), f"{md} is empty"

    # Today's daily log.
    today_log = paths.daily_log()
    assert today_log.is_file()
    assert today_log.read_text(encoding="utf-8").startswith(f"# {date.today().isoformat()}")

    # .tmp/.gitignore present and correct.
    gi = paths.tmp_dir / ".gitignore"
    assert gi.is_file()
    assert gi.read_text(encoding="utf-8") == "*\n!.gitignore\n"

    # TOOLS.md flagged as auto-generated.
    assert "auto-generated" in paths.tools_md.read_text(encoding="utf-8").lower()


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_config(
        JarvisConfig(
            paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
        )
    )
    bootstrap_workspace(paths)

    # Mutate USER.md to verify it is preserved.
    paths.user_md.write_text("# USER.md\n\ncustom content\n", encoding="utf-8")
    custom_mtime = paths.user_md.stat().st_mtime

    # Sleep a hair so any rewrite would change mtime detectably.
    time.sleep(0.02)
    bootstrap_workspace(paths)

    assert paths.user_md.read_text(encoding="utf-8") == "# USER.md\n\ncustom content\n"
    assert paths.user_md.stat().st_mtime == custom_mtime


def test_workspace_paths_from_config(tmp_path: Path) -> None:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)

    assert paths.root == (tmp_path / "ws").resolve() or paths.root == tmp_path / "ws"
    assert paths.memory_md == paths.root / "MEMORY.md"
    assert paths.user_md == paths.root / "USER.md"
    assert paths.tmp_dir == paths.root / ".tmp"
    assert paths.dreams_staging_dir == paths.root / "memory" / ".dreams"


def test_from_config_rejects_relative_workspace(tmp_path: Path) -> None:
    """from_config asserts the workspace is absolute — defense against bad YAML."""
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    # Patch the resolved path to a relative one to exercise the assertion.
    bad_cfg = cfg.model_copy(deep=True)
    object.__setattr__(bad_cfg.paths, "workspace", Path("relative/ws"))
    with pytest.raises(AssertionError, match="absolute"):
        WorkspacePaths.from_config(bad_cfg)


def test_daily_log_path_format(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_config(
        JarvisConfig(
            paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
        )
    )
    assert paths.daily_log(date(2026, 4, 26)) == paths.root / "memory" / "2026-04-26.md"


def test_project_path_format(tmp_path: Path) -> None:
    paths = WorkspacePaths.from_config(
        JarvisConfig(
            paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
        )
    )
    assert paths.project("rocket-sim") == paths.root / "projects" / "rocket-sim.md"
