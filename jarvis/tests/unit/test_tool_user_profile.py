"""user_profile_append_tool — section-keyed appends to USER.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.tool_user_profile import user_profile_append_tool
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    p = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(p)
    return p


def test_append_under_existing_section(paths):
    paths.user_md.write_text(
        "# USER\n\n## Identity\n- Grant\n\n## School\n- 11th grade\n",
        encoding="utf-8",
    )
    user_profile_append_tool(section="Identity", content="lives in Seattle", paths=paths)
    text = paths.user_md.read_text(encoding="utf-8")
    # Bullet landed under Identity, before the School heading.
    identity_block = text.split("## School")[0]
    assert "- Grant" in identity_block
    assert "- lives in Seattle" in identity_block
    # School section unaltered.
    assert "## School\n- 11th grade" in text


def test_creates_section_when_missing(paths):
    paths.user_md.write_text("# USER\n\n## Identity\n- Grant\n", encoding="utf-8")
    user_profile_append_tool(section="Preferences", content="prefers terse replies", paths=paths)
    text = paths.user_md.read_text(encoding="utf-8")
    assert "## Preferences" in text
    assert "- prefers terse replies" in text
    # Original Identity section still intact.
    assert "- Grant" in text


def test_strips_dash_prefix(paths):
    paths.user_md.write_text("# USER\n\n## Identity\n", encoding="utf-8")
    user_profile_append_tool(section="Identity", content="- already a bullet", paths=paths)
    text = paths.user_md.read_text(encoding="utf-8")
    # Should not be double-dashed.
    assert "- already a bullet" in text
    assert "- - already a bullet" not in text


def test_case_insensitive_section_match(paths):
    paths.user_md.write_text("# USER\n\n## Identity\n- a\n", encoding="utf-8")
    user_profile_append_tool(section="identity", content="b", paths=paths)
    text = paths.user_md.read_text(encoding="utf-8")
    # Should not have created a duplicate "## identity" — append under existing.
    assert text.count("## Identity") == 1
    assert "## identity" not in text  # no lowercase duplicate
    assert "- a" in text and "- b" in text


def test_empty_section_or_content_raises(paths):
    with pytest.raises(ValueError):
        user_profile_append_tool(section="", content="x", paths=paths)
    with pytest.raises(ValueError):
        user_profile_append_tool(section="x", content="", paths=paths)


def test_appends_below_blank_line_in_section(paths):
    """Bullet should land at the end of the existing list, not after blank lines."""
    paths.user_md.write_text(
        "# USER\n\n## Identity\n- a\n- b\n\n## School\n- 11th\n",
        encoding="utf-8",
    )
    user_profile_append_tool(section="Identity", content="c", paths=paths)
    text = paths.user_md.read_text(encoding="utf-8")
    # The bullet should appear before the blank line / next heading,
    # forming a contiguous list.
    identity_block = text.split("## School")[0]
    bullets = [ln for ln in identity_block.splitlines() if ln.startswith("- ")]
    assert bullets == ["- a", "- b", "- c"]
