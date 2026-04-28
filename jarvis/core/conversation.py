"""Conversation primitive — JSONL transcript + DB row lifecycle (BUILD_SPEC §7).

A ``Conversation`` owns one ``conv_id``, one ``conversations`` table row, and
one append-only ``workspace/.conversations/<conv_id>.jsonl`` transcript file.

Resume vs reset rule (called by ``Conversation.open``):
  * No open row for ``(channel_kind, channel_id)``           → create new.
  * ``now - last_event_at > idle_minutes``                    → close old, create new.
  * Configured daily HH:MM falls between ``last_event_at`` and ``now`` (local
    time) → close old, create new. ``dm_daily_at`` covers ``dm`` / ``cli`` /
    ``heartbeat``; ``group_daily_at`` covers ``group``.

``last_event_at`` is **not** persisted — it's derived. Live sessions track it
in-memory (``_last_event_at``, set by ``append()``); reconstruction after a
daemon restart falls back to ``os.path.getmtime(transcript_path)``. mtime has
1-second resolution on most filesystems, so the daily-reset boundary check
can theoretically miss by a second on the very first message after a restart
that lands within a second of the boundary; it self-corrects on the message
after that. Accepted — boundary precision is to-the-minute, not millisecond.

JSONL writes are serialized through a ``threading.Lock`` so multi-line
appends from concurrent threads can never interleave. POSIX append-atomicity
is per-write, but Python's buffered text I/O can flush partial lines without
the lock; the test suite proves the lock is what makes the guarantee.

Naming note: "conversation" not "session" (anti-pattern §19 #11). The
``/api/session`` HTTP endpoint is named for HTTP convention only — its
response key is ``conv_id``.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from typing import IO, Literal

from jarvis.memory.index import (
    close_conversation,
    get_open_conversation,
    insert_conversation,
)
from jarvis.memory.workspace import WorkspacePaths

logger = logging.getLogger(__name__)

__all__ = [
    "ChannelKind",
    "ConversationConfig",
    "Conversation",
    "TranscriptEventKind",
]

ChannelKind = Literal["dm", "group", "heartbeat", "cli"]
TranscriptEventKind = Literal[
    "user_message",
    "system_prompt",
    "assistant_message",
    "tool_call",
    "tool_result",
    "compaction",
    "delegation_snapshot",
    "delegation_envelope",
]


@dataclass(frozen=True)
class ConversationConfig:
    """Reset cadence for a conversation. Mirrors ``cfg.conversation.reset``.

    Three knobs only — passive idle threshold and the two cron HH:MM windows.
    Construct from a ``JarvisConfig`` via :meth:`from_jarvis_config`.
    """

    idle_minutes: int = 120
    dm_daily_at: str = "04:00"      # covers dm / cli / heartbeat
    group_daily_at: str = "02:00"   # covers group

    @classmethod
    def from_jarvis_config(cls, cfg) -> ConversationConfig:  # type: ignore[no-untyped-def]
        """Build from a ``JarvisConfig`` (avoids two ``ConversationConfig`` collisions)."""
        r = cfg.conversation.reset
        return cls(
            idle_minutes=r.idle_minutes,
            dm_daily_at=r.daily_at,
            group_daily_at=r.group_daily_at,
        )

    def daily_at_for(self, channel_kind: ChannelKind) -> str:
        """Pick which HH:MM applies to a channel. Group uses its own slot."""
        return self.group_daily_at if channel_kind == "group" else self.dm_daily_at


def _new_conv_id(now: datetime) -> str:
    """``YYYYMMDDTHHMMSS-<6-hex>`` — sortable, file-system-safe, 21 chars."""
    stamp = now.strftime("%Y%m%dT%H%M%S")
    tail = secrets.token_hex(3)  # 6 hex chars
    return f"{stamp}-{tail}"


def _parse_hhmm(value: str) -> dtime:
    """Parse ``HH:MM`` → ``datetime.time``. Raises ValueError on bad input."""
    hh, mm = value.split(":", 1)
    return dtime(hour=int(hh), minute=int(mm))


def _crosses_daily_boundary(
    last_event_at: datetime, now: datetime, hhmm: str
) -> bool:
    """True iff the configured HH:MM lies in (last_event_at, now] in local time.

    Calendar-aware: counts every HH:MM occurrence at the daily cadence between
    the two timestamps. Uses local time (``datetime.now()`` is naive local).
    """
    if now <= last_event_at:
        return False
    target_t = _parse_hhmm(hhmm)
    # Walk day-by-day from last_event_at's date forward; check each calendar
    # day's HH:MM against the open interval (last_event_at, now].
    cursor_date = last_event_at.date()
    end_date = now.date()
    while cursor_date <= end_date:
        boundary = datetime.combine(cursor_date, target_t)
        if last_event_at < boundary <= now:
            return True
        # Next day.
        cursor_date = cursor_date.fromordinal(cursor_date.toordinal() + 1)
    return False


class Conversation:
    """One live conversation. JSONL writer + DB row lifecycle.

    Construct via :meth:`open` — direct construction is reserved for tests
    and for the scheduler's reconstruction path.
    """

    def __init__(
        self,
        conv_id: str,
        channel_kind: ChannelKind,
        channel_id: str | None,
        paths: WorkspacePaths,
        conn,
        *,
        _last_event_at: datetime | None = None,
    ) -> None:
        self.conv_id = conv_id
        self.channel_kind: ChannelKind = channel_kind
        self.channel_id = channel_id
        self.paths = paths
        self.conn = conn

        self._transcript_path = paths.conversations_dir / f"{conv_id}.jsonl"
        self._transcript_path.parent.mkdir(parents=True, exist_ok=True)
        # Text mode, line buffer; append so the file is created if missing
        # and never truncated.
        self._fp: IO[str] | None = open(  # noqa: SIM115 — closed in close()/__exit__
            self._transcript_path, "a", encoding="utf-8", buffering=1
        )
        self._append_lock = threading.Lock()
        self._last_event_at = _last_event_at or datetime.now()
        self._closed = False

    # -- factory -----------------------------------------------------------

    @classmethod
    def open(
        cls,
        *,
        channel_kind: ChannelKind,
        channel_id: str | None,
        paths: WorkspacePaths,
        conn,
        cfg: ConversationConfig,
        now: datetime | None = None,
    ) -> Conversation:
        """Resume the open conversation for the channel, or create a fresh one.

        If a stale row exists (idle threshold or daily boundary crossed), it
        is closed without slug/summary — those require an LLM round-trip,
        which the scheduler / explicit ``close()`` path handles. The
        timestamp on the closed row is the actual ``last_event_at`` of the
        old conversation, not ``now`` — that way "when did this conversation
        end?" reflects the user's last message, not the wakeup that retired
        it.
        """
        now = now or datetime.now()
        existing = get_open_conversation(conn, channel_kind, channel_id)

        if existing is not None:
            transcript_rel = existing["transcript_path"]
            transcript_abs = paths.root / transcript_rel
            last_event_at = _last_event_at_from_disk(transcript_abs, existing)
            if not cls._should_reset(now, last_event_at, channel_kind, cfg):
                # Resume — re-attach to the existing row.
                return cls(
                    conv_id=existing["id"],
                    channel_kind=channel_kind,
                    channel_id=channel_id,
                    paths=paths,
                    conn=conn,
                    _last_event_at=last_event_at,
                )
            # Reset — close the stale row at its last_event_at (no slug/summary).
            close_conversation(
                conn,
                existing["id"],
                int(last_event_at.timestamp()),
                slug=None,
                summary=None,
            )

        conv_id = _new_conv_id(now)
        transcript_rel = (paths.conversations_dir / f"{conv_id}.jsonl").relative_to(
            paths.root
        ).as_posix()
        insert_conversation(
            conn,
            conv_id=conv_id,
            started_at=int(now.timestamp()),
            channel_kind=channel_kind,
            channel_id=channel_id,
            transcript_path=transcript_rel,
        )
        return cls(
            conv_id=conv_id,
            channel_kind=channel_kind,
            channel_id=channel_id,
            paths=paths,
            conn=conn,
            _last_event_at=now,
        )

    # -- reset decision ----------------------------------------------------

    @staticmethod
    def _should_reset(
        now: datetime,
        last_event_at: datetime,
        channel_kind: ChannelKind,
        cfg: ConversationConfig,
    ) -> bool:
        idle_seconds = (now - last_event_at).total_seconds()
        if idle_seconds > cfg.idle_minutes * 60:
            return True
        return _crosses_daily_boundary(
            last_event_at, now, cfg.daily_at_for(channel_kind)
        )

    # -- lifecycle ---------------------------------------------------------

    def append(self, kind: TranscriptEventKind, payload: dict) -> None:
        """Append a single JSONL event line. Updates in-memory ``last_event_at``."""
        if self._closed or self._fp is None:
            raise RuntimeError(f"Conversation {self.conv_id!r} is closed")
        line = json.dumps(
            {"ts": time.time(), "kind": kind, "payload": payload},
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n"
        with self._append_lock:
            self._fp.write(line)
            self._fp.flush()
        self._last_event_at = datetime.now()

    def close(
        self,
        *,
        slug: str | None,
        summary: str | None,
        now: datetime | None = None,
    ) -> None:
        """Stamp ended_at + slug + summary onto the row. Idempotent on this object."""
        if self._closed:
            return
        ended_at = int((now or datetime.now()).timestamp())
        close_conversation(self.conn, self.conv_id, ended_at, slug, summary)
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                logger.exception("conversation %s: close raised on transcript fp", self.conv_id)
            self._fp = None
        self._closed = True

    # -- properties / context manager -------------------------------------

    @property
    def transcript_path(self) -> Path:
        return self._transcript_path

    @property
    def last_event_at(self) -> datetime:
        return self._last_event_at

    def __enter__(self) -> Conversation:
        return self

    def __exit__(self, *_exc) -> None:  # type: ignore[no-untyped-def]
        # Don't auto-close (would clobber the row); just release the FD.
        if self._fp is not None and not self._closed:
            try:
                self._fp.close()
            finally:
                self._fp = None


def _last_event_at_from_disk(transcript_abs: Path, row) -> datetime:  # type: ignore[no-untyped-def]
    """Best-effort reconstruction of last_event_at after a daemon restart.

    Falls back through: transcript mtime → row.started_at. Never raises.
    """
    if transcript_abs.exists():
        try:
            return datetime.fromtimestamp(os.path.getmtime(transcript_abs))
        except OSError:
            pass
    return datetime.fromtimestamp(int(row["started_at"]))
