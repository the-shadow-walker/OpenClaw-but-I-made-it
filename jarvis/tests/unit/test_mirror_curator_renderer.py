"""Renderer unit tests for jarvis.workers.mirror_curator.

Tests the pure renderer in isolation. ``now`` is pinned to a fixed epoch
so output is byte-deterministic.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from jarvis.workers.mirror_curator import MirrorRow, render_mirror

# 2026-04-27T12:00:00Z — fixed epoch so timestamps in rendered output are
# byte-deterministic across machines and runs.
NOW = 1761566400.0


def _row(
    key: str,
    value: str = "v",
    agent_id: str = "cmd",
    *,
    created_at: float | None = None,
    expires_at: float = 0.0,
) -> MirrorRow:
    return MirrorRow(
        key=key,
        value=value,
        agent_id=agent_id,
        created_at=NOW - 60.0 if created_at is None else created_at,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# Namespace grouping
# ---------------------------------------------------------------------------


def test_renderer_groups_convo_keys_under_active_conversations() -> None:
    rows = [_row("convo_abc", value="hello convo")]
    out = render_mirror(rows, now=NOW)
    assert "## Active conversations" in out
    # The convo body appears under its section, before any other section.
    convo_idx = out.index("## Active conversations")
    body_idx = out.index("hello convo")
    next_section_idx = out.index("## Active projects")
    assert convo_idx < body_idx < next_section_idx


def test_renderer_groups_project_keys_under_active_projects() -> None:
    rows = [_row("project_rocket", value="rocket plan")]
    out = render_mirror(rows, now=NOW)
    proj_idx = out.index("## Active projects")
    body_idx = out.index("rocket plan")
    next_idx = out.index("## In-flight jobs")
    assert proj_idx < body_idx < next_idx


def test_renderer_groups_user_keys_under_user_context() -> None:
    rows = [_row("user_pref", value="prefer markdown")]
    out = render_mirror(rows, now=NOW)
    user_idx = out.index("## User context")
    body_idx = out.index("prefer markdown")
    next_idx = out.index("## Recent handoffs")
    assert user_idx < body_idx < next_idx


@pytest.mark.parametrize(
    "key,marker",
    [
        ("chain_xyz", "chain val"),
        ("gui_xyz", "gui val"),
        ("cmd_xyz", "cmd val"),
        ("swarm_xyz", "swarm val"),
        ("session_xyz", "session val"),
    ],
)
def test_renderer_groups_inflight_prefixes_under_inflight_jobs(
    key: str, marker: str
) -> None:
    rows = [_row(key, value=marker)]
    out = render_mirror(rows, now=NOW)
    inflight_idx = out.index("## In-flight jobs")
    body_idx = out.index(marker)
    next_idx = out.index("## User context")
    assert inflight_idx < body_idx < next_idx
    # Should NOT be in Recent handoffs.
    handoff_section = out[out.index("## Recent handoffs"):]
    assert marker not in handoff_section


# ---------------------------------------------------------------------------
# Filtering: expired and ephemera
# ---------------------------------------------------------------------------


def test_renderer_skips_expired_entries() -> None:
    rows = [
        _row("project_alive", value="ALIVE", expires_at=NOW + 7200),
        _row(
            "project_dead",
            value="DEAD",
            created_at=NOW - 7200,
            expires_at=NOW - 60,  # expired
        ),
    ]
    out = render_mirror(rows, now=NOW)
    assert "ALIVE" in out
    assert "DEAD" not in out


def test_renderer_skips_ephemera() -> None:
    """expires_at - created_at < 3600 → in-flight tool plumbing → skipped."""
    rows = [
        _row(
            "cmd_durable",
            value="KEEP",
            created_at=NOW - 60,
            expires_at=NOW + 7200,  # 2h+ TTL → kept
        ),
        _row(
            "cmd_ephemeral",
            value="DROP",
            created_at=NOW - 60,
            expires_at=NOW + 60,  # ~2 minutes → dropped
        ),
    ]
    out = render_mirror(rows, now=NOW)
    assert "KEEP" in out
    assert "DROP" not in out


# ---------------------------------------------------------------------------
# Excerpt rules
# ---------------------------------------------------------------------------


def test_renderer_excerpts_values_over_2kb() -> None:
    big = "a" * 3000
    rows = [_row("project_big", value=big)]
    out = render_mirror(rows, now=NOW)
    # 200-char excerpt + pointer line; full 3000 chars must NOT be inlined.
    assert "a" * 200 in out
    assert "a" * 3000 not in out
    assert "see SQLite key `project_big` for full content" in out


def test_renderer_keeps_brief_full_under_8kb() -> None:
    val = "B" * 5000   # 5KB, well under the 8KB always-full cap
    rows = [_row("project_x_brief", value=val)]
    out = render_mirror(rows, now=NOW)
    assert val in out  # full content inlined verbatim


def test_renderer_keeps_result_full_under_8kb() -> None:
    val = "R" * 5000
    rows = [_row("project_x_result", value=val)]
    out = render_mirror(rows, now=NOW)
    assert val in out


def test_renderer_brief_over_8kb_warns_and_excerpts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    val = "X" * 10000  # 10KB — exceeds 8KB always-full cap
    rows = [_row("project_x_brief", value=val)]
    with caplog.at_level(logging.WARNING, logger="jarvis.workers.mirror_curator"):
        out = render_mirror(rows, now=NOW)
    assert "X" * 10000 not in out  # NOT inlined verbatim
    assert "X" * 200 in out          # 200-char excerpt
    assert "exceeded 8KB promoted-summary cap" in out
    assert "publisher bug for key `project_x_brief`" in out
    assert any(
        "exceeded 8KB always-full cap" in rec.getMessage() for rec in caplog.records
    )


def test_renderer_brief_v2_follows_excerpt_rule() -> None:
    """Key ending in ``_v2`` (not ``_brief`` exactly) is NOT always-full.

    Pins exact-suffix-match contract: ``_brief_v2`` follows the standard
    2KB excerpt rule, NOT always-full. Documented in the renderer
    docstring; future agents must not "improve" this to glob-match.
    """
    val = "V" * 5000  # 5KB — would be full under always-full, excerpted under standard
    rows = [_row("project_x_brief_v2", value=val)]
    out = render_mirror(rows, now=NOW)
    assert "V" * 5000 not in out  # NOT always-full
    assert "V" * 200 in out
    assert "see SQLite key `project_x_brief_v2` for full content" in out


# ---------------------------------------------------------------------------
# Recent handoffs
# ---------------------------------------------------------------------------


def test_renderer_recent_handoffs_dedup_against_grouped_sections() -> None:
    rows = [
        _row("convo_a", value="CONVO_BODY"),
        _row("project_a", value="PROJ_BODY"),
        _row("user_a", value="USER_BODY"),
        _row("chain_a", value="CHAIN_BODY"),
        _row("gui_a", value="GUI_BODY"),
        _row("cmd_a", value="CMD_BODY"),
        _row("swarm_a", value="SWARM_BODY"),
        _row("session_a", value="SESSION_BODY"),
    ]
    out = render_mirror(rows, now=NOW)
    handoff_section = out[out.index("## Recent handoffs"):]
    for marker in (
        "CONVO_BODY", "PROJ_BODY", "USER_BODY",
        "CHAIN_BODY", "GUI_BODY", "CMD_BODY",
        "SWARM_BODY", "SESSION_BODY",
    ):
        assert marker not in handoff_section, (
            f"{marker} appeared in Recent handoffs but should be in a grouped section"
        )


def test_renderer_recent_handoffs_includes_unknown_prefix() -> None:
    """Unknown future namespaces (e.g. ``dream_*``) gracefully degrade
    into Recent handoffs by default.
    """
    rows = [_row("dream_xyz", value="DREAM_BODY")]
    out = render_mirror(rows, now=NOW)
    handoff_section = out[out.index("## Recent handoffs"):]
    assert "DREAM_BODY" in handoff_section


def test_renderer_recent_handoffs_capped_at_20() -> None:
    rows = [
        _row(f"misc_{i:02d}", value=f"BODY_{i:02d}", created_at=NOW - i)
        for i in range(30)
    ]
    out = render_mirror(rows, now=NOW)
    handoff_section = out[out.index("## Recent handoffs"):]
    # The 20 most recent (lowest i, since created_at descends with i)
    # should be present; the rest should not.
    for i in range(20):
        assert f"BODY_{i:02d}" in handoff_section
    for i in range(20, 30):
        assert f"BODY_{i:02d}" not in handoff_section


def test_renderer_empty_board_emits_empty_marker() -> None:
    out = render_mirror([], now=NOW)
    assert "shared board empty" in out
    # Header still renders so consumers can confirm the curator ran.
    assert "Jarvis Central Context Mirror" in out


def test_renderer_section_order() -> None:
    out = render_mirror([], now=NOW)
    # Header first
    assert out.startswith("# Jarvis Central Context Mirror")
    # Empty-marker case has no section headers; populate to assert order.
    rows = [
        _row("convo_a"), _row("project_a"), _row("chain_a"),
        _row("user_a"), _row("misc_a"),
    ]
    out2 = render_mirror(rows, now=NOW)
    indices = [
        out2.index("## Active conversations"),
        out2.index("## Active projects"),
        out2.index("## In-flight jobs"),
        out2.index("## User context"),
        out2.index("## Recent handoffs"),
    ]
    assert indices == sorted(indices), "sections out of spec order"


# ---------------------------------------------------------------------------
# Byte-exact golden snapshot
# ---------------------------------------------------------------------------


CANONICAL_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "mirror" / "canonical.md"
)


def _canonical_rows() -> list[MirrorRow]:
    """Deterministic row set for the golden-file test. Mix every section.

    All timestamps are explicit so the rendered output is byte-stable.
    """
    return [
        MirrorRow(
            key="convo_abc",
            value="active conversation summary",
            agent_id="jarvis",
            created_at=NOW - 30,
            expires_at=0.0,
        ),
        MirrorRow(
            key="project_rocket",
            value="rocket-sim plan body",
            agent_id="jarvis",
            created_at=NOW - 90,
            expires_at=NOW + 86400,
        ),
        MirrorRow(
            key="cmd_job_42",
            value="cmd job in flight",
            agent_id="cmd",
            created_at=NOW - 20,
            expires_at=NOW + 7200,
        ),
        MirrorRow(
            key="swarm_dispatch_7",
            value="swarm dispatch in flight",
            agent_id="swarm:math",
            created_at=NOW - 15,
            expires_at=NOW + 7200,
        ),
        MirrorRow(
            key="user_pref_lang",
            value="prefer python",
            agent_id="jarvis",
            created_at=NOW - 600,
            expires_at=0.0,
        ),
        MirrorRow(
            key="dream_seed_1",
            value="recent unscoped handoff",
            agent_id="jarvis",
            created_at=NOW - 5,
            expires_at=0.0,
        ),
    ]


def test_renderer_byte_exact_snapshot(request: pytest.FixtureRequest) -> None:
    """Golden-file test. Re-generate with ``UPDATE_MIRROR_GOLDEN=1``."""
    import os

    rendered = render_mirror(_canonical_rows(), now=NOW)
    if os.environ.get("UPDATE_MIRROR_GOLDEN") == "1":
        CANONICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        CANONICAL_PATH.write_text(rendered, encoding="utf-8")
        pytest.skip("regenerated golden file at " + str(CANONICAL_PATH))
    expected = CANONICAL_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "renderer output drifted from golden file; if intentional, "
        "rerun with UPDATE_MIRROR_GOLDEN=1"
    )
