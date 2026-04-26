#!/usr/bin/env python3
"""
AgentMemory — swarm-side mirror of CMD's AgentMemory shared_context board.

Single source of truth lives in `~/.agent_bin/memory.db` (CMD-owned).
Swarm writes through CMD's REST endpoint (`POST /api/v1/context`) so the
markdown mirror at `~/.agent_bin/central_context.md` stays current.

If CMD is unreachable, swarm falls back to a direct SQLite write with a
3-attempt retry on SQLITE_BUSY and a stale-mirror warning. Reads always
go to SQLite directly (no mirror impact).

Schema lock: SCHEMA_VERSION = 1, recorded in a `_meta` table that both
sides create with CREATE TABLE IF NOT EXISTS. CMD has been informed via
handoff that this table will appear after first swarm startup.
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

DB_DIR = os.path.expanduser("~/.agent_bin")
DB_PATH = os.path.join(DB_DIR, "memory.db")
SCHEMA_VERSION = 1
DEFAULT_CMD_URL = os.getenv("AGENT_MEMORY_CMD_URL", "http://localhost:5000")
DEFAULT_TIMEOUT_S = float(os.getenv("AGENT_MEMORY_HTTP_TIMEOUT", "5"))


class AgentMemory:
    """Thin wrapper that prefers CMD's REST API for writes; reads SQLite directly."""

    def __init__(
        self,
        agent_id: str = "swarm",
        cmd_url: str = DEFAULT_CMD_URL,
        db_path: str = DB_PATH,
    ):
        self.agent_id = agent_id
        self.cmd_url = cmd_url.rstrip("/")
        self.db_path = db_path
        self._lock = Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    # ── DB init ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_context (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    agent_id    TEXT,
                    updated_at  TEXT NOT NULL,
                    ttl_seconds INTEGER DEFAULT 86400
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _meta (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO _meta(key, value, updated_at) VALUES('schema_version', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                                              updated_at=excluded.updated_at
                """,
                (str(SCHEMA_VERSION), datetime.utcnow().isoformat()),
            )
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_context(
        self,
        key: str,
        value: str,
        agent_id: Optional[str] = None,
        ttl: int = 86400,
    ) -> bool:
        """Publish to the shared context board.

        Tries CMD's REST endpoint first (so the markdown mirror is rebuilt).
        Falls back to a direct SQLite write on connection error / timeout.
        Returns True on any successful write, False on hard failure.
        """
        agent = agent_id or self.agent_id
        ttl_hours = max(1, int(round(ttl / 3600))) if ttl else 0

        # 1. Preferred path — CMD REST
        try:
            resp = requests.post(
                f"{self.cmd_url}/api/v1/context",
                json={
                    "key": key,
                    "value": value,
                    "agent_id": agent,
                    "ttl_hours": ttl_hours,
                },
                timeout=DEFAULT_TIMEOUT_S,
            )
            if resp.ok and resp.json().get("ok"):
                return True
            # If CMD responds but rejects, fall through to direct write
            print(f"⚠️ CMD REST set_context for {key!r} returned {resp.status_code}; falling back to direct SQLite")
        except (requests.ConnectionError, requests.Timeout):
            print(
                "⚠️ CMD REST unreachable; writing SQLite directly. "
                "central_context.md may be stale until CMD restarts."
            )
        except Exception as e:
            print(f"⚠️ CMD REST error ({type(e).__name__}: {e}); falling back to direct SQLite")

        # 2. Fallback — direct SQLite with WAL retry
        return self._direct_set(key, value, agent, ttl)

    def _direct_set(self, key: str, value: str, agent: str, ttl: int) -> bool:
        now = datetime.utcnow().isoformat()
        for attempt in range(3):
            try:
                with self._lock, sqlite3.connect(self.db_path, timeout=5) as conn:
                    conn.execute(
                        """
                        INSERT INTO shared_context(key, value, agent_id, updated_at, ttl_seconds)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                          value=excluded.value, agent_id=excluded.agent_id,
                          updated_at=excluded.updated_at, ttl_seconds=excluded.ttl_seconds
                        """,
                        (key, value, agent, now, ttl),
                    )
                    conn.commit()
                return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(0.1 * (attempt + 1))
                    continue
                print(f"❌ SQLite write failed for {key!r}: {e}")
                return False
            except Exception as e:
                print(f"❌ SQLite write failed for {key!r}: {e}")
                return False
        print(f"❌ SQLite write for {key!r} gave up after 3 SQLITE_BUSY retries")
        return False

    def get_context(self, key: str) -> Optional[str]:
        """Read a single key from SQLite. Auto-deletes expired entries."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                row = conn.execute(
                    "SELECT value, updated_at, ttl_seconds FROM shared_context WHERE key = ?",
                    (key,),
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        value, updated_at, ttl = row
        if ttl and ttl > 0:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(updated_at)).total_seconds()
                if age > ttl:
                    try:
                        with self._lock, sqlite3.connect(self.db_path, timeout=5) as conn:
                            conn.execute("DELETE FROM shared_context WHERE key = ?", (key,))
                            conn.commit()
                    except Exception:
                        pass
                    return None
            except Exception:
                pass
        return value

    def list_context(self, prefix: str = "") -> List[Dict[str, Any]]:
        """Return all non-expired context entries, optionally filtered by key prefix."""
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                rows = conn.execute(
                    """
                    SELECT key, value, agent_id, updated_at, ttl_seconds
                    FROM shared_context
                    WHERE key LIKE ?
                    ORDER BY updated_at DESC
                    """,
                    (f"{prefix}%",),
                ).fetchall()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        now = datetime.utcnow()
        for key, value, agent_id, updated_at, ttl in rows:
            if ttl and ttl > 0:
                try:
                    age = (now - datetime.fromisoformat(updated_at)).total_seconds()
                    if age > ttl:
                        continue
                except Exception:
                    pass
            out.append(
                {
                    "key": key,
                    "value": value,
                    "agent_id": agent_id,
                    "updated_at": updated_at,
                    "ttl_seconds": ttl,
                }
            )
        return out


# ── Module singleton ─────────────────────────────────────────────────────────

_default: Optional[AgentMemory] = None


def get_default(agent_id: str = "swarm") -> AgentMemory:
    """Return a process-wide AgentMemory instance."""
    global _default
    if _default is None:
        _default = AgentMemory(agent_id=agent_id)
    return _default


if __name__ == "__main__":
    # smoke test
    mem = AgentMemory(agent_id="swarm.probe")
    assert mem.set_context("swarm_self_test", "hello", ttl=60), "set_context failed"
    assert mem.get_context("swarm_self_test") == "hello", "get_context mismatch"
    listed = mem.list_context(prefix="swarm_self_test")
    print(f"OK — set/get/list round-trip ({len(listed)} entries)")
