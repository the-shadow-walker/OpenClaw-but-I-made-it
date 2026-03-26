"""
Session Manager for JARVIS Distributed System
Tracks conversation history per client session
"""

import sqlite3
import uuid
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from config.server_config import SESSION_DB_PATH, SESSION_TIMEOUT_HOURS, MAX_HISTORY_MESSAGES


@dataclass
class Session:
    """Represents a client session"""
    id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    history: List[Dict]  # [{"role": "user|assistant", "content": "..."}]


class SessionManager:
    """Manages client sessions and conversation history"""

    def __init__(self, db_path: Path = SESSION_DB_PATH):
        self.db_path = db_path
        self.sessions = {}  # In-memory cache: session_id -> Session
        self._init_db()
        self._load_active_sessions()

    def _init_db(self):
        """Initialize SQLite database for session persistence"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    history TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_activity
                ON sessions(last_activity)
            """)
            conn.commit()

    def _load_active_sessions(self):
        """Load active sessions from database into memory"""
        cutoff = datetime.now() - timedelta(hours=SESSION_TIMEOUT_HOURS)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id, user_id, created_at, last_activity, history "
                "FROM sessions WHERE last_activity > ?",
                (cutoff.isoformat(),)
            )

            for row in cursor:
                session_id, user_id, created_at, last_activity, history_json = row
                self.sessions[session_id] = Session(
                    id=session_id,
                    user_id=user_id,
                    created_at=datetime.fromisoformat(created_at),
                    last_activity=datetime.fromisoformat(last_activity),
                    history=json.loads(history_json)
                )

    def create_session(self, user_id: str = "grant") -> str:
        """Create new session and return session ID"""
        session_id = str(uuid.uuid4())
        now = datetime.now()

        session = Session(
            id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            history=[]
        )

        self.sessions[session_id] = session
        self._save_session(session)

        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID, return None if expired or not found"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check if expired
        if datetime.now() - session.last_activity > timedelta(hours=SESSION_TIMEOUT_HOURS):
            self.delete_session(session_id)
            return None

        return session

    def get_history(self, session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> List[Dict]:
        """Get conversation history for session (last N messages)"""
        session = self.get_session(session_id)
        if not session:
            return []

        return session.history[-limit:]

    def add_to_history(self, session_id: str, role: str, content: str):
        """Add message to session history"""
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found or expired")

        session.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        session.last_activity = datetime.now()
        self._save_session(session)

    def delete_session(self, session_id: str):
        """Delete session from memory and database"""
        if session_id in self.sessions:
            del self.sessions[session_id]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    def _save_session(self, session: Session):
        """Persist session to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, user_id, created_at, last_activity, history)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    session.created_at.isoformat(),
                    session.last_activity.isoformat(),
                    json.dumps(session.history)
                )
            )
            conn.commit()

    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory and database"""
        cutoff = datetime.now() - timedelta(hours=SESSION_TIMEOUT_HOURS)

        # Clean memory
        expired = [
            sid for sid, session in self.sessions.items()
            if session.last_activity < cutoff
        ]
        for session_id in expired:
            del self.sessions[session_id]

        # Clean database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM sessions WHERE last_activity < ?",
                (cutoff.isoformat(),)
            )
            conn.commit()

    def get_active_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session metadata (for debugging/status)"""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(session.history)
        }
