"""Unit tests for jarvis.core.conversation + jarvis.core.scheduler (P5a)."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.conversation import (
    Conversation,
    ConversationConfig,
    _crosses_daily_boundary,
)
from jarvis.core.scheduler import ResetScheduler
from jarvis.memory.index import (
    get_connection,
    get_open_conversation,
    init_schema,
    insert_conversation,
)
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    return paths


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    return _make_paths(tmp_path)


@pytest.fixture
def conn(paths: WorkspacePaths):
    c = get_connection(paths.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def cfg() -> ConversationConfig:
    return ConversationConfig(idle_minutes=120, dm_daily_at="04:00", group_daily_at="02:00")


# ---------------------------------------------------------------------------
# Conversation.open / append / close
# ---------------------------------------------------------------------------


def test_open_creates_row_and_jsonl(paths, conn, cfg):
    now = datetime(2026, 4, 27, 10, 0, 0)
    convo = Conversation.open(
        channel_kind="dm", channel_id="grant", paths=paths, conn=conn, cfg=cfg, now=now
    )
    try:
        # Row exists with ended_at NULL.
        row = get_open_conversation(conn, "dm", "grant")
        assert row is not None
        assert row["id"] == convo.conv_id
        assert row["channel_kind"] == "dm"
        assert row["channel_id"] == "grant"
        assert row["started_at"] == int(now.timestamp())
        assert row["ended_at"] is None
        # transcript_path is workspace-relative.
        assert row["transcript_path"] == f".conversations/{convo.conv_id}.jsonl"
        # File exists (empty).
        assert convo.transcript_path.exists()
        assert convo.transcript_path.read_text() == ""
    finally:
        convo.__exit__(None, None, None)


def test_append_writes_jsonl_line(paths, conn, cfg):
    convo = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg
    )
    try:
        convo.append("user_message", {"content": "hello"})
        convo.append("assistant_message", {"content": "hi", "tool_calls": None})
        lines = convo.transcript_path.read_text().splitlines()
        assert len(lines) == 2
        e1 = json.loads(lines[0])
        e2 = json.loads(lines[1])
        assert e1["kind"] == "user_message"
        assert e1["payload"] == {"content": "hello"}
        assert e2["kind"] == "assistant_message"
        assert isinstance(e1["ts"], float)
    finally:
        convo.__exit__(None, None, None)


def test_resume_within_idle(paths, conn, cfg):
    t0 = datetime(2026, 4, 27, 10, 0, 0)
    convo1 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t0
    )
    cid1 = convo1.conv_id
    convo1.append("user_message", {"content": "hi"})
    convo1.__exit__(None, None, None)

    # Same channel 30 minutes later → resume.
    t1 = t0 + timedelta(minutes=30)
    convo2 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t1
    )
    try:
        assert convo2.conv_id == cid1
        # The original row is still the open row.
        row = get_open_conversation(conn, "dm", "g")
        assert row["id"] == cid1
    finally:
        convo2.__exit__(None, None, None)


def test_idle_timeout_starts_new(paths, conn, cfg):
    t0 = datetime(2026, 4, 27, 10, 0, 0)
    convo1 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t0
    )
    cid1 = convo1.conv_id
    convo1.append("user_message", {"content": "first"})
    convo1.__exit__(None, None, None)

    # Touch the file's mtime to t0 so reconstruction sees the right last_event_at.
    import os
    os.utime(convo1.transcript_path, (t0.timestamp(), t0.timestamp()))

    # 121 minutes later → idle threshold tripped.
    t1 = t0 + timedelta(minutes=121)
    convo2 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t1
    )
    try:
        assert convo2.conv_id != cid1
        # Old row has ended_at stamped (no slug/summary).
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (cid1,)
        ).fetchone()
        assert row["ended_at"] is not None
        assert row["slug"] is None
        assert row["summary"] is None
        # New row is the open one.
        open_row = get_open_conversation(conn, "dm", "g")
        assert open_row["id"] == convo2.conv_id
    finally:
        convo2.__exit__(None, None, None)


def test_daily_reset_dm_at_0400(paths, conn, cfg):
    # First message yesterday at 23:50.
    t0 = datetime(2026, 4, 26, 23, 50, 0)
    convo1 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t0
    )
    cid1 = convo1.conv_id
    convo1.append("user_message", {"content": "evening"})
    convo1.__exit__(None, None, None)

    import os
    os.utime(convo1.transcript_path, (t0.timestamp(), t0.timestamp()))

    # Today at 04:01 → crossed the dm 04:00 boundary.
    t1 = datetime(2026, 4, 27, 4, 1, 0)
    convo2 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t1
    )
    try:
        assert convo2.conv_id != cid1
    finally:
        convo2.__exit__(None, None, None)


def test_group_reset_uses_group_time(paths, conn, cfg):
    # Last seen at 01:45 (well under the 120-min idle threshold), today.
    t0 = datetime(2026, 4, 27, 1, 45, 0)
    convo1 = Conversation.open(
        channel_kind="group", channel_id="room1", paths=paths, conn=conn, cfg=cfg, now=t0
    )
    cid1 = convo1.conv_id
    convo1.append("user_message", {"content": "x"})
    convo1.__exit__(None, None, None)

    import os
    os.utime(convo1.transcript_path, (t0.timestamp(), t0.timestamp()))

    # 02:01 — crossed the group 02:00 boundary (DM 04:00 NOT yet crossed,
    # 16-minute gap is well under idle).
    t1 = datetime(2026, 4, 27, 2, 1, 0)
    convo2 = Conversation.open(
        channel_kind="group", channel_id="room1", paths=paths, conn=conn, cfg=cfg, now=t1
    )
    try:
        assert convo2.conv_id != cid1
    finally:
        convo2.__exit__(None, None, None)

    # And: a DM channel at the same wall-clock gap WOULD resume (DM threshold is 04:00).
    convo_dm1 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t0
    )
    convo_dm1.append("user_message", {"content": "x"})
    convo_dm1.__exit__(None, None, None)
    os.utime(convo_dm1.transcript_path, (t0.timestamp(), t0.timestamp()))

    convo_dm2 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t1
    )
    try:
        assert convo_dm2.conv_id == convo_dm1.conv_id, "DM should not reset at 02:01"
    finally:
        convo_dm2.__exit__(None, None, None)


def test_close_stamps_slug_and_summary(paths, conn, cfg):
    convo = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg
    )
    cid = convo.conv_id
    convo.append("user_message", {"content": "hi"})
    convo.close(slug="greeting", summary="user said hi")
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
    assert row["slug"] == "greeting"
    assert row["summary"] == "user said hi"
    assert row["ended_at"] is not None


def test_50_event_synthetic_conversation(paths, conn, cfg):
    convo = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg
    )
    try:
        kinds = ["user_message", "assistant_message", "tool_call", "tool_result", "system_prompt"]
        for i in range(50):
            convo.append(kinds[i % len(kinds)], {"i": i, "blob": "x" * (i * 10)})
        text = convo.transcript_path.read_text()
        lines = text.splitlines()
        assert len(lines) == 50
        for i, ln in enumerate(lines):
            evt = json.loads(ln)
            assert evt["kind"] == kinds[i % len(kinds)]
            assert evt["payload"]["i"] == i
            assert isinstance(evt["ts"], float)
    finally:
        convo.__exit__(None, None, None)


def test_jsonl_append_concurrent_lines_dont_interleave(paths, conn, cfg):
    convo = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg
    )
    try:
        # Pick payload sizes that straddle the 4KB POSIX-pipe atomicity boundary
        # so we prove the lock is what makes this safe, not happenstance.
        big_blob = "A" * 5000  # >4KB
        small_blob = "b" * 50

        errors: list[str] = []

        def writer(label: str, blob: str):
            try:
                for i in range(100):
                    convo.append("user_message", {"who": label, "i": i, "blob": blob})
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}: {e!r}")

        t1 = threading.Thread(target=writer, args=("big", big_blob))
        t2 = threading.Thread(target=writer, args=("small", small_blob))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors, errors

        lines = convo.transcript_path.read_text().splitlines()
        assert len(lines) == 200, f"expected 200, got {len(lines)}"
        big_count = 0
        small_count = 0
        for ln in lines:
            evt = json.loads(ln)
            assert evt["kind"] == "user_message"
            who = evt["payload"]["who"]
            if who == "big":
                assert evt["payload"]["blob"] == big_blob
                big_count += 1
            else:
                assert evt["payload"]["blob"] == small_blob
                small_count += 1
        assert big_count == 100 and small_count == 100
    finally:
        convo.__exit__(None, None, None)


def test_append_after_close_raises(paths, conn, cfg):
    convo = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg
    )
    convo.close(slug=None, summary=None)
    with pytest.raises(RuntimeError):
        convo.append("user_message", {"content": "x"})


# ---------------------------------------------------------------------------
# _crosses_daily_boundary unit tests (pure function)
# ---------------------------------------------------------------------------


def test_crosses_daily_boundary_obvious_cross():
    last = datetime(2026, 4, 26, 23, 50)
    now = datetime(2026, 4, 27, 4, 1)
    assert _crosses_daily_boundary(last, now, "04:00") is True


def test_crosses_daily_boundary_not_yet():
    last = datetime(2026, 4, 27, 0, 30)
    now = datetime(2026, 4, 27, 3, 30)
    assert _crosses_daily_boundary(last, now, "04:00") is False


def test_crosses_daily_boundary_within_same_day_after_target():
    last = datetime(2026, 4, 27, 5, 0)
    now = datetime(2026, 4, 27, 6, 0)
    assert _crosses_daily_boundary(last, now, "04:00") is False


def test_crosses_daily_boundary_long_gap_includes_target():
    # 3-day gap definitely contains the target.
    last = datetime(2026, 4, 25, 12, 0)
    now = datetime(2026, 4, 28, 12, 0)
    assert _crosses_daily_boundary(last, now, "04:00") is True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_scheduler_fires_close_at_configured_time(cfg):
    """In lieu of advancing wall-clock through APScheduler (flaky in CI), drive
    the private callbacks directly. They're the same code path APScheduler
    invokes; the cron triggers themselves are APScheduler's contract.
    """
    fired: list[list[str]] = []

    def on_reset(kinds: list[str]) -> None:
        fired.append(list(kinds))

    sched = ResetScheduler(cfg, on_reset)
    sched.start()
    try:
        sched._fire_dm()       # noqa: SLF001 — testing the private fire path
        sched._fire_group()    # noqa: SLF001
    finally:
        sched.stop()

    assert fired == [["dm", "cli", "heartbeat"], ["group"]]


def test_scheduler_swallows_callback_exceptions(cfg):
    def raising(_kinds):
        raise RuntimeError("boom")

    sched = ResetScheduler(cfg, raising)
    sched.start()
    try:
        # Should not raise out — just log.
        sched._fire_dm()       # noqa: SLF001
    finally:
        sched.stop()


def test_open_after_restart_reconstructs_last_event_from_mtime(paths, cfg):
    """Simulate a daemon restart: drop the in-memory Conversation, reopen the
    DB connection, and verify open() reconstructs last_event_at from the
    transcript mtime.
    """
    db = paths.index_dir / "memory.sqlite"
    conn = get_connection(db)
    init_schema(conn)
    t0 = datetime(2026, 4, 27, 10, 0, 0)
    convo1 = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn, cfg=cfg, now=t0
    )
    cid1 = convo1.conv_id
    convo1.append("user_message", {"content": "hi"})
    convo1.__exit__(None, None, None)
    conn.close()

    # New process: new connection, same workspace.
    import os
    os.utime(paths.conversations_dir / f"{cid1}.jsonl",
             (t0.timestamp() + 60, t0.timestamp() + 60))
    conn2 = get_connection(db)
    try:
        # 30 minutes after the mtime — should resume.
        t1 = t0 + timedelta(minutes=30, seconds=60)
        convo2 = Conversation.open(
            channel_kind="dm", channel_id="g", paths=paths, conn=conn2, cfg=cfg, now=t1
        )
        try:
            assert convo2.conv_id == cid1
        finally:
            convo2.__exit__(None, None, None)
    finally:
        conn2.close()


def test_insert_and_close_helpers_round_trip(paths, conn):
    # Direct test of the helpers in jarvis.memory.index.
    insert_conversation(
        conn,
        conv_id="20260427T100000-abc123",
        started_at=int(datetime(2026, 4, 27, 10, 0).timestamp()),
        channel_kind="dm",
        channel_id="g",
        transcript_path=".conversations/20260427T100000-abc123.jsonl",
    )
    row = get_open_conversation(conn, "dm", "g")
    assert row["id"] == "20260427T100000-abc123"

    from jarvis.memory.index import close_conversation
    close_conversation(
        conn,
        "20260427T100000-abc123",
        ended_at=int(time.time()),
        slug="x",
        summary="y",
    )
    after = get_open_conversation(conn, "dm", "g")
    assert after is None
