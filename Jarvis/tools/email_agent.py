"""
email_agent.py
==============
JARVIS Email Memory Agent

Runs in the background, polls Gmail, classifies emails with an LLM,
and stores only meaningful notes into SQLite — so JARVIS can inject
relevant email context into conversations naturally.

Storage: memory/email_memory.db
"""

import json
import sqlite3
import time
import re
import base64
import pickle
import os
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import List, Dict, Optional

import requests

# ─── Config ───────────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))
from config.server_config import OLLAMA_HOST, MODELS, MEMORY_DIR, TOKEN_PATH

FAST_MODEL         = MODELS["chat"]
DB_PATH            = MEMORY_DIR / "email_memory.db"
RECENT_EMAILS_FILE = MEMORY_DIR / "recent-emails.md"
POLL_INTERVAL      = 15 * 60
MAX_FETCH          = 50
RECENT_DAYS        = 7
OLD_UNIMPORTANT_DAYS = 7


# ─── Gmail helpers ────────────────────────────────────────────────────────────

def _load_gmail_service():
    """Load Gmail API service from saved token."""
    try:
        from googleapiclient.discovery import build
        from google.auth.transport.requests import Request
        with open(TOKEN_PATH, "rb") as f:
            creds = pickle.load(f)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return build("gmail", "v1", credentials=creds)
    except Exception as e:
        print(f"[EmailAgent] Gmail auth failed: {e}")
        return None


def _extract_body(payload) -> str:
    """Recursively extract plain-text body from Gmail payload."""
    mime = payload.get("mimeType", "")
    if mime == "text/plain":
        data = payload["body"].get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    for part in payload.get("parts", []):
        text = _extract_body(part)
        if text:
            return text
    return ""


# ─── LLM helpers ──────────────────────────────────────────────────────────────

def _ollama_chat(messages: List[Dict], model: str = FAST_MODEL, timeout: int = 60) -> str:
    try:
        r = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        if r.status_code == 200:
            content = r.json()["message"]["content"]
            # Strip <think> blocks from qwen3
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
            return content.strip()
    except Exception:
        pass
    return ""


CLASSIFY_SYSTEM = """You are an email classifier for a personal AI assistant.
Decide if an email is worth remembering as a concise note.

SKIP these: marketing, newsletters, promotions, rewards programs, subscription
  confirmations with no ongoing relevance, automated social-media notifications,
  generic service alerts (e.g. "your password was used").

REMEMBER these: personal messages from real people, meetings/appointments,
  orders/shipping updates with meaningful detail, financial statements (bank,
  invoice, bill amount), legal/medical correspondence, job/work related emails,
  travel confirmations, anything the user would want to recall later.

Respond with ONLY valid JSON (no markdown, no explanation):
{
  "action": "skip" | "remember",
  "importance": 1 | 2 | 3,
  "note": "one or two sentences capturing the key fact — omit filler words",
  "tags": ["tag1", "tag2"]
}

importance: 1=low (good to know), 2=medium (somewhat important), 3=high (action needed / time-sensitive)
tags: choose from: meeting, finance, order, shipping, personal, work, legal, medical, travel, project, other"""


def _classify_email(sender: str, subject: str, body: str) -> Optional[Dict]:
    """Ask the LLM to classify and summarize one email. Returns dict or None on skip."""
    snippet = body[:600].strip()
    user_msg = f"From: {sender}\nSubject: {subject}\nBody:\n{snippet}"

    raw = _ollama_chat(
        messages=[
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        timeout=45,
    )

    # Extract JSON — handle LLM adding markdown fences
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        result = json.loads(match.group())
        if result.get("action") == "skip":
            return None
        return result
    except json.JSONDecodeError:
        return None


# ─── Database ─────────────────────────────────────────────────────────────────

def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS email_notes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id     TEXT    UNIQUE,
            from_name    TEXT,
            from_addr    TEXT,
            subject      TEXT,
            date         TEXT,
            importance   INTEGER DEFAULT 1,
            note         TEXT,
            tags         TEXT,
            body         TEXT,
            processed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_importance ON email_notes(importance);
        CREATE INDEX IF NOT EXISTS idx_date       ON email_notes(date);
        CREATE INDEX IF NOT EXISTS idx_processed  ON email_notes(processed_at);

        -- Archive for old important emails (searchable)
        CREATE TABLE IF NOT EXISTS old_important_emails (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id     TEXT    UNIQUE,
            from_name    TEXT,
            from_addr    TEXT,
            subject      TEXT,
            date         TEXT,
            importance   INTEGER,
            note         TEXT,
            tags         TEXT,
            body         TEXT,
            archived_at  TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_old_subject ON old_important_emails(subject);
        CREATE INDEX IF NOT EXISTS idx_old_from    ON old_important_emails(from_name);
    """)
    conn.commit()


# ─── Main Agent Class ──────────────────────────────────────────────────────────

class EmailAgent:
    """
    Background email memory agent.

    Fetches Gmail, classifies with LLM, stores notes.
    JARVIS calls get_relevant_notes(query) to get context.
    """

    def __init__(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        _init_db(self.conn)
        self._service = None  # loaded lazily
        self._running = False
        print("   📧 Email agent ready")

    # ── Gmail access ──────────────────────────────────────────────────────────

    def _get_service(self):
        if self._service is None:
            self._service = _load_gmail_service()
        return self._service

    def _already_processed(self, email_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM email_notes WHERE email_id = ?", (email_id,)
        ).fetchone()
        return row is not None

    # ── Fetch & process ───────────────────────────────────────────────────────

    def fetch_and_process(self) -> int:
        """
        Fetch recent emails, classify new ones, save notes.
        Returns number of new notes saved.
        """
        svc = self._get_service()
        if not svc:
            return 0

        try:
            results = svc.users().messages().list(
                userId="me", maxResults=MAX_FETCH
            ).execute()
        except Exception as e:
            print(f"[EmailAgent] Fetch error: {e}")
            return 0

        messages = results.get("messages", [])
        print(f"[EmailAgent] Retrieved {len(messages)} messages from Gmail")
        saved = 0
        skipped = 0
        already_processed = 0

        for idx, msg in enumerate(messages, 1):
            email_id = msg["id"]

            # Fetch basic info for logging
            try:
                full = svc.users().messages().get(
                    userId="me", id=email_id, format="full"
                ).execute()
                headers = {
                    h["name"]: h["value"]
                    for h in full["payload"]["headers"]
                }
                from_raw = headers.get("From", "")
                subject  = headers.get("Subject", "(no subject)")
                date_str = headers.get("Date", "")

                print(f"\n[EmailAgent] Email {idx}/{len(messages)}: '{subject[:70]}'")
                print(f"             From: {from_raw[:50]}")
                print(f"             Date: {date_str[:30]}")

            except Exception as e:
                print(f"[EmailAgent] Error fetching email {idx}: {e}")
                continue

            if self._already_processed(email_id):
                already_processed += 1
                print(f"             Status: ✓ Already processed (in database)")
                continue

            try:
                body     = _extract_body(full["payload"])

                # Parse from into name + addr
                m = re.match(r'"?([^"<]+)"?\s*<([^>]+)>', from_raw)
                if m:
                    from_name = m.group(1).strip()
                    from_addr = m.group(2).strip()
                else:
                    from_name = from_raw
                    from_addr = from_raw

                # Skip if no body content to evaluate
                if not body and not subject:
                    print(f"             Status: ⊘ Skipped (no body/subject)")
                    continue

                print(f"             Status: 🤖 Classifying with LLM...")
                result = _classify_email(from_name or from_addr, subject, body)

                # Always record it as processed so we don't retry
                if result is None:
                    # Store with importance 0 = skipped (so we don't re-fetch)
                    self.conn.execute(
                        """INSERT OR IGNORE INTO email_notes
                           (email_id, from_name, from_addr, subject, date, importance, note, tags, body, processed_at)
                           VALUES (?, ?, ?, ?, ?, 0, '', '[]', '', ?)""",
                        (email_id, from_name, from_addr, subject, date_str,
                         datetime.now().isoformat()),
                    )
                    self.conn.commit()
                    skipped += 1
                    print(f"             Result: ⊘ Skipped by LLM (unimportant/spam)")
                    continue

                importance = result.get("importance", 1)
                note = result.get("note", "")
                tags_json = json.dumps(result.get("tags", []))

                print(f"             Result: ✅ SAVED - Importance: {importance}, Note: {note[:50]}")

                self.conn.execute(
                    """INSERT OR IGNORE INTO email_notes
                       (email_id, from_name, from_addr, subject, date, importance, note, tags, body, processed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        email_id, from_name, from_addr, subject, date_str,
                        importance,
                        note,
                        tags_json,
                        body[:2000],  # Store first 2000 chars of body
                        datetime.now().isoformat(),
                    ),
                )
                self.conn.commit()
                saved += 1

            except Exception as e:
                print(f"[EmailAgent] Error processing {email_id}: {e}")

        print(f"\n[EmailAgent] ========== SUMMARY ==========")
        print(f"[EmailAgent] Total fetched: {len(messages)}")
        print(f"[EmailAgent] Already in DB: {already_processed}")
        print(f"[EmailAgent] Newly saved:   {saved}")
        print(f"[EmailAgent] Skipped:       {skipped}")
        print(f"[EmailAgent] ================================\n")

        # Show what's in the database now
        total_in_db = self.conn.execute("SELECT COUNT(*) FROM email_notes").fetchone()[0]
        important_in_db = self.conn.execute("SELECT COUNT(*) FROM email_notes WHERE importance >= 1").fetchone()[0]
        print(f"[EmailAgent] Database status: {important_in_db} important emails (out of {total_in_db} total)")

        # Update recent-emails.md file
        if saved > 0 or important_in_db > 0:
            self.write_recent_emails_file()

        # Run maintenance (archive old, cleanup unimportant)
        self.perform_maintenance()

        return saved

    # ── Context retrieval ─────────────────────────────────────────────────────

    def get_relevant_notes(self, query: str = None, max_notes: int = 5) -> List[str]:
        """
        Return a list of concise note strings relevant to the current query.
        Called by JARVIS when building context.
        """
        # Only return importance >= 1 (not skipped)
        rows = self.conn.execute(
            """SELECT note, tags, from_name, subject, date, importance
               FROM email_notes
               WHERE importance >= 1 AND note != ''
               ORDER BY importance DESC, processed_at DESC
               LIMIT 30"""
        ).fetchall()

        if not rows:
            return []

        if not query:
            # No query — return the most recent/important ones
            return [row["note"] for row in rows[:max_notes]]

        # Score by keyword overlap with query
        q_words = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        stopwords = {"the", "a", "an", "is", "are", "can", "you", "i", "my", "me",
                     "do", "did", "have", "what", "when", "any", "about", "please"}
        q_words -= stopwords

        scored = []
        for row in rows:
            text = f"{row['note']} {row['subject']} {row['tags']} {row['from_name']}".lower()
            text_words = set(re.sub(r"[^\w\s]", "", text).split())
            overlap = len(q_words & text_words)
            # Always include high-importance notes
            if overlap > 0 or row["importance"] >= 3:
                scored.append((overlap + row["importance"] * 0.5, row["note"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [note for _, note in scored[:max_notes]]

    def get_recent_notes(self, n: int = 10) -> List[Dict]:
        """Return n most recent important email notes (for 'check my email' command)."""
        rows = self.conn.execute(
            """SELECT from_name, subject, date, importance, note, tags
               FROM email_notes
               WHERE importance >= 1 AND note != ''
               ORDER BY processed_at DESC
               LIMIT ?""",
            (n,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_summary(self) -> str:
        """Return a human-readable summary of recent important emails for JARVIS to speak."""
        notes = self.get_recent_notes(10)
        if not notes:
            return "No important emails noted recently, sir."

        high   = [n for n in notes if n["importance"] == 3]
        medium = [n for n in notes if n["importance"] == 2]
        low    = [n for n in notes if n["importance"] == 1]

        lines = []
        if high:
            lines.append("High priority:")
            for n in high:
                lines.append(f"  • {n['note']}")
        if medium:
            lines.append("Also notable:")
            for n in medium[:3]:
                lines.append(f"  • {n['note']}")
        if low and not high and not medium:
            for n in low[:3]:
                lines.append(f"  • {n['note']}")

        return "\n".join(lines)

    # ── Smart email management ────────────────────────────────────────────────

    def write_recent_emails_file(self):
        """Write recent important emails to markdown file for on-demand access."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=RECENT_DAYS)).isoformat()

        rows = self.conn.execute(
            """SELECT from_name, from_addr, subject, date, importance, note, body, processed_at
               FROM email_notes
               WHERE importance >= 2 AND processed_at > ?
               ORDER BY importance DESC, processed_at DESC""",
            (cutoff,),
        ).fetchall()

        if not rows:
            RECENT_EMAILS_FILE.write_text("# Recent Important Emails\n\nNo recent important emails.\n")
            return

        lines = ["# Recent Important Emails", ""]
        lines.append(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        lines.append(f"*Showing {len(rows)} important emails from the last {RECENT_DAYS} days*")
        lines.append("")

        for row in rows:
            importance_emoji = "🔴" if row['importance'] == 3 else "🟡"
            lines.append(f"## {importance_emoji} {row['subject']}")
            lines.append(f"**From:** {row['from_name']} <{row['from_addr']}>")
            lines.append(f"**Date:** {row['date']}")
            lines.append(f"**Summary:** {row['note']}")
            lines.append("")
            if row['body']:
                lines.append("**Full Email:**")
                lines.append("```")
                lines.append(row['body'][:1500])  # First 1500 chars
                if len(row['body']) > 1500:
                    lines.append("... (truncated)")
                lines.append("```")
                lines.append("")
            lines.append("---")
            lines.append("")

        RECENT_EMAILS_FILE.write_text("\n".join(lines))
        print(f"[EmailAgent] ✓ Wrote {len(rows)} recent emails to {RECENT_EMAILS_FILE.name}")

    def archive_old_important_emails(self):
        """Move old important emails to archive table."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=RECENT_DAYS)).isoformat()

        # Get old important emails
        old_emails = self.conn.execute(
            """SELECT email_id, from_name, from_addr, subject, date, importance, note, tags, body
               FROM email_notes
               WHERE importance >= 2 AND processed_at < ?""",
            (cutoff,),
        ).fetchall()

        if not old_emails:
            return 0

        # Move to archive
        for row in old_emails:
            self.conn.execute(
                """INSERT OR IGNORE INTO old_important_emails
                   (email_id, from_name, from_addr, subject, date, importance, note, tags, body, archived_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (*row, datetime.now().isoformat()),
            )

        # Delete from main table
        self.conn.execute(
            """DELETE FROM email_notes WHERE importance >= 2 AND processed_at < ?""",
            (cutoff,),
        )
        self.conn.commit()

        print(f"[EmailAgent] ✓ Archived {len(old_emails)} old important emails")
        return len(old_emails)

    def cleanup_old_unimportant_emails(self):
        """Delete old unimportant emails."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=OLD_UNIMPORTANT_DAYS)).isoformat()

        result = self.conn.execute(
            """DELETE FROM email_notes WHERE importance <= 1 AND processed_at < ?""",
            (cutoff,),
        )
        self.conn.commit()

        deleted = result.rowcount
        if deleted > 0:
            print(f"[EmailAgent] ✓ Cleaned up {deleted} old unimportant emails")
        return deleted

    def search_old_emails(self, query: str, limit: int = 10) -> List[Dict]:
        """Search archived important emails."""
        query_lower = f"%{query.lower()}%"

        rows = self.conn.execute(
            """SELECT from_name, subject, date, importance, note, body
               FROM old_important_emails
               WHERE LOWER(subject) LIKE ? OR LOWER(note) LIKE ? OR LOWER(from_name) LIKE ?
               ORDER BY importance DESC, archived_at DESC
               LIMIT ?""",
            (query_lower, query_lower, query_lower, limit),
        ).fetchall()

        return [dict(row) for row in rows]

    def get_recent_email_summary(self) -> str:
        """Get brief email highlights for context injection (JARVIS can answer from these directly)."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=RECENT_DAYS)).isoformat()

        rows = self.conn.execute(
            """SELECT from_name, subject, importance, note
               FROM email_notes
               WHERE importance >= 2 AND processed_at > ?
               ORDER BY importance DESC, processed_at DESC
               LIMIT 5""",
            (cutoff,),
        ).fetchall()

        if not rows:
            return "No recent important emails."

        total = self.conn.execute(
            "SELECT COUNT(*) FROM email_notes WHERE importance >= 2 AND processed_at > ?",
            (cutoff,),
        ).fetchone()[0]

        LABEL = {3: "HIGH", 2: "MED"}
        lines = [f"Top {len(rows)} of {total} recent emails (use [READ_RECENT_EMAILS] only if user asks for full list):"]
        for row in rows:
            label = LABEL.get(row["importance"], "MED")
            note = (row["note"] or row["subject"])[:120]
            lines.append(f"[{label}] {row['from_name']}: {note}")

        return "\n".join(lines)

    def get_email_digest(self, days: int = RECENT_DAYS) -> str:
        """Return compact email digest (subject/from/date/note, NO body) — ~3KB vs 47KB."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """SELECT from_name, subject, date, importance, note
               FROM email_notes
               WHERE importance >= 2 AND processed_at > ?
               ORDER BY importance DESC, processed_at DESC""",
            (cutoff,),
        ).fetchall()

        if not rows:
            return "No recent important emails."

        LABEL = {3: "HIGH", 2: "MED"}
        lines = [f"{len(rows)} important email(s) in the last {days} days:\n"]
        for row in rows:
            label = LABEL.get(row["importance"], "MED")
            lines.append(f"[{label}] {row['from_name']} — \"{row['subject']}\" ({row['date'][:10]})")
            if row["note"]:
                lines.append(f"  → {row['note']}")
        return "\n".join(lines)

    def perform_maintenance(self):
        """Run all maintenance tasks."""
        print("[EmailAgent] Running maintenance...")
        self.write_recent_emails_file()
        self.archive_old_important_emails()
        self.cleanup_old_unimportant_emails()

    # ── Background polling ────────────────────────────────────────────────────

    def _poll_loop(self):
        # First run immediately
        try:
            n = self.fetch_and_process()
            if n:
                print(f"[EmailAgent] Processed {n} new notable emails")
        except Exception as e:
            print(f"[EmailAgent] Poll error: {e}")

        while self._running:
            time.sleep(POLL_INTERVAL)
            if not self._running:
                break
            try:
                n = self.fetch_and_process()
                if n:
                    print(f"[EmailAgent] {n} new email notes saved")
            except Exception as e:
                print(f"[EmailAgent] Poll error: {e}")

    def start_background_polling(self):
        """Start background thread that polls Gmail every POLL_INTERVAL seconds."""
        self._running = True
        Thread(target=self._poll_loop, daemon=True, name="email-agent").start()

    def stop(self):
        self._running = False

    def is_alive(self) -> bool:
        return self._running


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = EmailAgent()
    print("Fetching and processing emails...")
    n = agent.fetch_and_process()
    print(f"\nSaved {n} new notes")
    print("\nRecent important emails:")
    print(agent.get_summary())
