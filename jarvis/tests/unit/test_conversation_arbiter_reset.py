"""Conversation lifecycle ↔ arbiter reset hooks (P10).

Verifies:

  * ``Conversation.close()`` resets the per-conv master entry when an
    arbiter is wired in.
  * Same call is a no-op when ``arbiter`` is left ``None`` (back-compat
    for tests / channels that don't construct one).
  * The stale-row reset path inside ``Conversation.open`` resets the
    PRIOR conversation's master before closing the row, so the new
    conversation gets fresh arbitration.
  * A busted ``arbiter.reset`` (raises) does NOT block the DB row close;
    the swallow is logged loudly so the failure is visible.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    p = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(p)
    return p


@pytest.fixture
def conn(paths: WorkspacePaths):
    c = get_connection(paths.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


def test_close_resets_arbiter_when_wired(paths, conn):
    arbiter = RoleArbiter()
    conv = Conversation.open(
        channel_kind="cli", channel_id="t1", paths=paths, conn=conn,
        cfg=ConversationConfig(), arbiter=arbiter,
    )
    arbiter.claim(conv.conv_id, "code")
    assert arbiter.master_for(conv.conv_id) == "code"
    conv.close(slug=None, summary=None)
    assert arbiter.master_for(conv.conv_id) is None


def test_close_no_arbiter_is_noop(paths, conn):
    """No arbiter wired → close still succeeds, doesn't reach for None."""
    conv = Conversation.open(
        channel_kind="cli", channel_id="t2", paths=paths, conn=conn,
        cfg=ConversationConfig(),
    )
    # Should not raise.
    conv.close(slug=None, summary=None)


def test_open_stale_row_resets_arbiter(paths, conn):
    """A stale-row reset on Conversation.open must reset the prior
    conversation's master entry so the new conversation starts clean."""
    arbiter = RoleArbiter()
    cfg = ConversationConfig(idle_minutes=60)

    # Open a conversation, claim a master.
    fixed_then = datetime(2026, 4, 28, 12, 0, 0)
    first = Conversation.open(
        channel_kind="cli", channel_id="stale-test",
        paths=paths, conn=conn, cfg=cfg, arbiter=arbiter, now=fixed_then,
    )
    arbiter.claim(first.conv_id, "code")
    assert arbiter.master_for(first.conv_id) == "code"
    # Release the FD without stamping ended_at.
    first.__exit__(None, None, None)

    # Hand-edit the row's last_event_at to be older than idle_minutes ago,
    # but DON'T close it — that's the "stale open row" condition.
    # We do this by manipulating the JSONL transcript mtime, since
    # last_event_at is reconstructed from disk after a restart.
    transcript = first.transcript_path
    old_ts = (fixed_then - timedelta(hours=3)).timestamp()
    os_utime_set(transcript, old_ts)

    # Re-open with a `now` that's well past the idle threshold relative
    # to the (mtime-derived) last_event_at.
    fresh_now = fixed_then  # idle_minutes=60, mtime is 3h earlier
    second = Conversation.open(
        channel_kind="cli", channel_id="stale-test",
        paths=paths, conn=conn, cfg=cfg, arbiter=arbiter, now=fresh_now,
    )
    # New conv_id (the stale row was closed, a new row was created).
    assert second.conv_id != first.conv_id
    # The prior conv's master entry was reset BEFORE the new conv started.
    assert arbiter.master_for(first.conv_id) is None
    # And the new conversation has no master claimed yet.
    assert arbiter.master_for(second.conv_id) is None
    second.__exit__(None, None, None)


def test_close_arbiter_failure_does_not_block_db_close(paths, conn, caplog):
    """A raising arbiter.reset must not block the DB row close. The
    swallow must be logged at WARNING/ERROR — silent exception-eating
    is the failure mode that hides future bugs."""

    class BustedArbiter:
        def reset(self, conv_id: str) -> None:
            raise RuntimeError("simulated bust")

    conv = Conversation.open(
        channel_kind="cli", channel_id="t3", paths=paths, conn=conn,
        cfg=ConversationConfig(), arbiter=BustedArbiter(),
    )
    with caplog.at_level(logging.WARNING, logger="jarvis.core.conversation"):
        conv.close(slug=None, summary=None)
    # close() returned cleanly; the DB row close happened first.
    assert conv._closed is True
    # The swallowed exception was logged loud enough to find later.
    assert any(
        "arbiter.reset" in rec.getMessage() for rec in caplog.records
    ), "arbiter.reset failure was swallowed without a log line"


def os_utime_set(p: Path, ts: float) -> None:
    """Set both atime and mtime on a path to the given epoch second."""
    import os
    os.utime(p, (ts, ts))
