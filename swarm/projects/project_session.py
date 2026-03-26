"""
Project Session State Machine
==============================

Wraps project_mode.py's LLM-driven Q&A flow in a stateless HTTP-friendly
session model.  No blocking `input()` calls — the caller drives state
forward by POSTing answers.

State flow:
  init  →  qa  (after /project/start with description)
  qa    →  qa  (after /project/respond while more questions remain)
  qa    →  done (after last answer — triggers compute_requirements)

Amazon sourcing is intentionally skipped (requires interactive selection).
"""

import uuid
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# ── Lazy import from project_mode ────────────────────────────────────────────
try:
    from project_mode import get_next_question, compute_requirements
    _HAS_PROJECT_MODE = True
except ImportError:
    _HAS_PROJECT_MODE = False
    def get_next_question(specs, history):
        return {"done": True}
    def compute_requirements(specs):
        return {"requirements": {}, "component_categories": [], "notes": "project_mode.py unavailable"}


# ── Session dataclass ─────────────────────────────────────────────────────────

@dataclass
class ProjectSession:
    session_id: str
    state: str                   # "qa" | "done"
    specs: Dict[str, Any]        # accumulated Q&A answers
    history: List[Dict]          # [{"question": ..., "answer": ...}]
    pending_question: Optional[Dict]  # last question issued to the client
    qa_count: int
    result_markdown: Optional[str]
    requirements: Optional[Dict]
    created_at: float            # unix timestamp
    updated_at: float


def _new_session(description: str) -> ProjectSession:
    now = time.time()
    return ProjectSession(
        session_id=str(uuid.uuid4())[:12],
        state="qa",
        specs={"description": description},
        history=[],
        pending_question=None,
        qa_count=0,
        result_markdown=None,
        requirements=None,
        created_at=now,
        updated_at=now,
    )


# ── Simple in-memory markdown generator ──────────────────────────────────────

def _build_result_markdown(specs: Dict, req: Dict) -> str:
    lines = ["# Project Brief\n"]

    desc = specs.get("description", "")
    if desc:
        lines.append(f"**Description:** {desc}\n")

    lines.append("\n## Vision & Answers\n")
    for k, v in specs.items():
        if k.startswith("_") or k == "description":
            continue
        lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")

    user_notes = specs.get("_user_notes", [])
    if user_notes:
        lines.append("\n## Additional Notes\n")
        for note in user_notes:
            lines.append(f"- {note}")

    lines.append("\n## Technical Requirements\n")
    requirements = req.get("requirements", {})
    if requirements:
        for k, v in requirements.items():
            lines.append(f"- **{k.replace('_', ' ').title()}:** {v}")
    else:
        lines.append("_(no requirements computed)_")

    decisions = req.get("engineering_decisions", {})
    if decisions:
        lines.append("\n## Engineering Decisions\n")
        for decision, rationale in decisions.items():
            lines.append(f"- **{decision}:** {rationale}")

    categories = req.get("component_categories", [])
    if categories:
        lines.append("\n## Component Categories to Source\n")
        for cat in categories:
            lines.append(f"- {cat}")

    notes = req.get("notes", "")
    if notes:
        lines.append(f"\n## Notes\n{notes}")

    return "\n".join(lines)


# ── Session manager ───────────────────────────────────────────────────────────

class ProjectSessionManager:
    """
    In-memory store for active project sessions.
    Sessions expire after TTL_SECONDS of inactivity.
    """

    TTL_SECONDS = 2 * 3600  # 2 hours

    def __init__(self):
        self._sessions: Dict[str, ProjectSession] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def create(self, description: str) -> ProjectSession:
        """Create a new session and issue the first question."""
        session = _new_session(description)
        self._cleanup_expired()
        self._sessions[session.session_id] = session
        # Immediately fetch first question so the client gets it in the
        # /project/start response without needing a separate round-trip.
        self._advance_question(session)
        return session

    def get(self, session_id: str) -> Optional[ProjectSession]:
        """Return session or None if not found / expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.updated_at > self.TTL_SECONDS:
            del self._sessions[session_id]
            return None
        return session

    def advance(self, session_id: str, user_answer: str) -> Optional[ProjectSession]:
        """
        Record the user's answer to the current pending_question, then either
        issue the next question or finish by computing requirements.

        Returns the updated session (or None if session_id invalid).
        """
        session = self.get(session_id)
        if session is None or session.state == "done":
            return session

        pq = session.pending_question
        if pq:
            session.history.append({
                "question": pq.get("question", ""),
                "answer":   user_answer,
            })
            key = pq.get("key", f"answer_{session.qa_count}")
            session.specs[key] = user_answer
            session.qa_count += 1

        session.updated_at = time.time()
        self._advance_question(session)
        return session

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _advance_question(self, session: ProjectSession):
        """Ask the LLM for the next question; finalize if done."""
        try:
            nq = get_next_question(session.specs, session.history)
        except Exception:
            nq = {"done": True}

        # Treat a missing/empty question as implicit done (LLM returned malformed JSON)
        if nq.get("done") or not nq.get("question", "").strip():
            self._finalize(session)
        else:
            session.pending_question = nq

    def _finalize(self, session: ProjectSession):
        """Compute requirements and render markdown; transition to 'done'."""
        try:
            req = compute_requirements(session.specs)
        except Exception as e:
            req = {"requirements": {}, "notes": f"compute_requirements error: {e}"}

        session.requirements = req
        session.result_markdown = _build_result_markdown(session.specs, req)
        session.state = "done"
        session.pending_question = None
        session.updated_at = time.time()

    def _cleanup_expired(self):
        """Remove sessions older than TTL."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.updated_at > self.TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]

    # ── Serialization ─────────────────────────────────────────────────────────

    @staticmethod
    def to_dict(session: ProjectSession) -> Dict:
        """Convert a ProjectSession to a JSON-serialisable dict."""
        return {
            "session_id":       session.session_id,
            "state":            session.state,
            "specs":            session.specs,
            "history":          session.history,
            "pending_question": session.pending_question,
            "qa_count":         session.qa_count,
            "result_markdown":  session.result_markdown,
            "requirements":     session.requirements,
            "created_at":       session.created_at,
            "updated_at":       session.updated_at,
        }

    @staticmethod
    def to_response(session: ProjectSession) -> Dict:
        """
        Build the JSON response returned to the API client.

        During Q&A:
          {"session_id", "state":"qa", "question", "type", "options", "recommendation"}

        When done:
          {"session_id", "state":"done", "result_markdown", "requirements"}
        """
        if session.state == "done":
            return {
                "session_id":      session.session_id,
                "state":           "done",
                "result_markdown": session.result_markdown,
                "requirements":    session.requirements,
            }

        pq = session.pending_question or {}
        return {
            "session_id":     session.session_id,
            "state":          "qa",
            "qa_count":       session.qa_count,
            "question":       pq.get("question", ""),
            "type":           pq.get("type", "text"),
            "options":        pq.get("options", []),
            "recommendation": pq.get("recommendation", ""),
            "key":            pq.get("key", ""),
        }


# ── Module-level singleton ────────────────────────────────────────────────────

session_manager = ProjectSessionManager()
