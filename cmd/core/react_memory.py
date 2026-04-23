#!/usr/bin/env python3
"""
AgentMemory: SQLite-backed command history and system survey cache.
No Ollama dependency — can be tested in isolation.
"""

import sqlite3
import subprocess
import json
import os
import time
import glob as glob_module
from threading import Lock
from typing import Dict, List, Optional, Any
from datetime import datetime


class AgentMemory:
    DB_DIR = os.path.expanduser("~/.agent_bin")
    DB_PATH = os.path.join(DB_DIR, "memory.db")
    RUNBOOK_DIR = os.path.join(DB_DIR, "runbooks")
    BACKUP_DIR = os.path.join(DB_DIR, "backups")

    SURVEY_COMMANDS = [
        ("uname",           "uname -a"),
        ("memory",          "free -h"),
        ("cpu_info",        "lscpu 2>/dev/null | head -10 || sysctl -n machdep.cpu.brand_string 2>/dev/null"),
        ("cpu_count",       "nproc 2>/dev/null || sysctl -n hw.ncpu"),
        ("disk",            "df -h"),
        ("hostname",        "hostname"),
        ("whoami",          "whoami"),
        ("home_dir",        "echo $HOME"),
        ("shell",           "echo $SHELL"),
        ("uptime",          "uptime"),
        ("active_services", "systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null | head -20 || echo '(systemd unavailable)'"),
        ("failed_units",    "systemctl list-units --state=failed --no-pager --no-legend 2>/dev/null || echo '(systemd unavailable)'"),
        ("kernel",          "uname -r"),
        ("open_ports",      "ss -tulnp 2>/dev/null | grep LISTEN | head -20"),
        ("lang_versions",   "python3 --version 2>&1; node --version 2>&1; rustc --version 2>&1; go version 2>&1; java -version 2>&1 | head -1; echo done"),
        ("pip_packages",    "pip3 list 2>/dev/null | wc -l | xargs echo 'pip packages:'"),
        ("firewall",        "sudo nft list ruleset 2>/dev/null | grep -E 'chain|tcp dport|udp dport' | head -30 || echo '(nft: needs sudo or not installed)'"),
        ("installed_packages",
         "pacman -Qe 2>/dev/null | awk '{print $1}' | tr '\\n' ' '"),
    ]

    def __init__(self):
        os.makedirs(self.DB_DIR, exist_ok=True)
        os.makedirs(self.RUNBOOK_DIR, exist_ok=True)
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
        self._lock = Lock()
        self._init_db()
        self._survey_cache: Optional[Dict[str, str]] = None
        self._survey_cache_time: float = 0.0

    # ------------------------------------------------------------------ DB --

    def _init_db(self):
        with sqlite3.connect(self.DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_history (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    command       TEXT NOT NULL,
                    context       TEXT,
                    task          TEXT,
                    exit_code     INTEGER,
                    duration_ms   INTEGER,
                    used_at       TEXT NOT NULL,
                    success_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shared_context (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    agent_id    TEXT,
                    updated_at  TEXT NOT NULL,
                    ttl_seconds INTEGER DEFAULT 86400
                )
            """)
            conn.commit()

    # --------------------------------------------------------- system survey -

    def get_system_survey(self) -> Dict[str, str]:
        """Run 11 survey commands; cache for 10 minutes."""
        now = time.time()
        if self._survey_cache and (now - self._survey_cache_time) < 600:
            return self._survey_cache

        survey: Dict[str, str] = {}
        for key, cmd in self.SURVEY_COMMANDS:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    executable='/bin/bash',
                )
                if result.returncode == 0:
                    survey[key] = result.stdout.strip()
                else:
                    survey[key] = f"(error: {result.stderr.strip()[:100]})"
            except Exception as e:
                survey[key] = f"(unavailable: {e})"

        self._survey_cache = survey
        self._survey_cache_time = now
        return survey

    # ------------------------------------------------------- memory lookup --

    def lookup(self, query: str) -> List[Dict[str, Any]]:
        """LIKE search across command, context, and task columns."""
        try:
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    """
                    SELECT command, context, task, exit_code, duration_ms,
                           used_at, success_count
                    FROM   command_history
                    WHERE  command LIKE ?
                        OR context LIKE ?
                        OR task    LIKE ?
                    ORDER  BY success_count DESC, used_at DESC
                    LIMIT  10
                    """,
                    (f"%{query}%", f"%{query}%", f"%{query}%"),
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception:
            return []

    # ---------------------------------------------------- record a success --

    def record_success(
        self,
        command: str,
        context: str,
        task: str,
        exit_code: int,
        duration_ms: int,
    ):
        """UPSERT a successful command; increment success_count if it already exists."""
        try:
            used_at = datetime.now().isoformat()
            with sqlite3.connect(self.DB_PATH) as conn:
                cur = conn.execute(
                    "SELECT id, success_count FROM command_history WHERE command = ?",
                    (command,),
                )
                row = cur.fetchone()
                if row:
                    conn.execute(
                        """UPDATE command_history
                              SET success_count = ?,
                                  used_at      = ?,
                                  context      = ?,
                                  task         = ?,
                                  exit_code    = ?,
                                  duration_ms  = ?
                            WHERE id = ?""",
                        (row[1] + 1, used_at, context, task, exit_code, duration_ms, row[0]),
                    )
                else:
                    conn.execute(
                        """INSERT INTO command_history
                               (command, context, task, exit_code, duration_ms, used_at)
                           VALUES (?,?,?,?,?,?)""",
                        (command, context, task, exit_code, duration_ms, used_at),
                    )
                conn.commit()
        except Exception:
            pass  # Non-fatal; memory failures should not break execution

    # ------------------------------------------------------------ runbooks --

    def load_runbook(self, task_keyword: str) -> Optional[str]:
        """Scan ~/.agent_bin/runbooks/*.md for a keyword match; return content."""
        try:
            runbook_files = glob_module.glob(
                os.path.join(self.RUNBOOK_DIR, "*.md")
            )
            keyword_lower = task_keyword.lower()
            for fpath in runbook_files:
                try:
                    with open(fpath, "r") as f:
                        content = f.read()
                    basename = os.path.basename(fpath).lower()
                    if keyword_lower in content.lower() or keyword_lower in basename:
                        return content
                except Exception:
                    continue
        except Exception:
            pass
        return None


    # --------------------------------------------------- shared context board --

    def set_context(self, key: str, value: str,
                    agent_id: str = "cmd", ttl: int = 86400) -> None:
        """Publish a fact to the shared cross-agent context board."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO shared_context(key, value, agent_id, updated_at, ttl_seconds)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                      value=excluded.value, agent_id=excluded.agent_id,
                      updated_at=excluded.updated_at, ttl_seconds=excluded.ttl_seconds
                """, (key, value, agent_id, now, ttl))
                conn.commit()

    def get_context(self, key: str) -> Optional[str]:
        """Read a single key from shared context (None if missing or expired)."""
        row = None
        with sqlite3.connect(self.DB_PATH) as conn:
            row = conn.execute(
                "SELECT value, updated_at, ttl_seconds FROM shared_context WHERE key = ?",
                (key,)
            ).fetchone()
        if not row:
            return None
        value, updated_at, ttl = row
        if ttl and ttl > 0:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(updated_at)).total_seconds()
                if age > ttl:
                    with sqlite3.connect(self.DB_PATH) as conn:
                        conn.execute("DELETE FROM shared_context WHERE key = ?", (key,))
                        conn.commit()
                    return None
            except Exception:
                pass
        return value

    def list_context(self, prefix: str = "") -> List[Dict]:
        """Return all non-expired context entries, optionally filtered by key prefix."""
        with sqlite3.connect(self.DB_PATH) as conn:
            rows = conn.execute("""
                SELECT key, value, agent_id, updated_at, ttl_seconds
                FROM shared_context
                WHERE key LIKE ?
                ORDER BY updated_at DESC
            """, (f"{prefix}%",)).fetchall()
        now = datetime.utcnow()
        result = []
        expired_keys = []
        for key, value, agent_id, updated_at, ttl in rows:
            if ttl and ttl > 0:
                try:
                    age = (now - datetime.fromisoformat(updated_at)).total_seconds()
                    if age > ttl:
                        expired_keys.append(key)
                        continue
                except Exception:
                    pass
            result.append({"key": key, "value": value, "agent": agent_id,
                            "updated_at": updated_at})
        if expired_keys:
            try:
                with sqlite3.connect(self.DB_PATH) as conn:
                    conn.executemany("DELETE FROM shared_context WHERE key = ?",
                                     [(k,) for k in expired_keys])
                    conn.commit()
            except Exception:
                pass
        return result


# ----------------------------------------------------------------- CLI test --

if __name__ == "__main__":
    m = AgentMemory()
    print("=== System Survey ===")
    survey = m.get_system_survey()
    for k, v in survey.items():
        print(f"  {k}: {v[:80]}")
    print("\n=== Memory lookup: 'systemctl' ===")
    rows = m.lookup("systemctl")
    for r in rows:
        print(f"  {r['command']} (used {r['success_count']}x)")
    print("\nOK")
