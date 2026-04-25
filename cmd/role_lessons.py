#!/usr/bin/env python3
"""
role_lessons.py — structured failure lessons for role-scoped memory.

Written to AgentMemory.shared_context (sqlite) with keys:
  "lesson_{role}_{pattern}_{goal_hash}"

Schema (Lesson dict):
  {
    "pattern": str,        # canonical id, e.g. "sqlalchemy_lazy_load"
    "fix": str,            # positive-framed guidance, ≤200 chars
    "confidence": float,   # 0.0–0.95
    "role": str,           # "builder" | "tester" | "commander"
    "goal_hash": str,      # 16-char sha256 prefix of the chain goal
    "source_phase": int,
    "occurrences": int
  }

Lessons are injected into minion prompts as positive guidance ("prefer X") to
avoid avoidance-overfitting on negative framings ("don't Y").
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional


# Canonical failure patterns → positive-framed fix text
PATTERN_DB: Dict[str, Dict[str, str]] = {
    "sqlalchemy_lazy_load": {
        "signals": r"(MissingGreenlet|greenlet_spawn|lazy.load.*async)",
        "fix": "When using SQLAlchemy async relationships, prefer selectinload() on queries; set relationship attrs manually after create.",
    },
    "bcrypt_passlib_py314": {
        "signals": r"(passlib.*bcrypt|bcrypt.*backend|__about__)",
        "fix": "For password hashing on Python 3.14+, prefer `import bcrypt as _bcrypt` directly; passlib's bcrypt backend is broken.",
    },
    "pydantic_schema_naming_drift": {
        "signals": r"(RegisterRequest|cannot import name.*Request|UserCreate.*not found)",
        "fix": "Keep Pydantic schema names aligned between routers and schemas.py; export aliases when in doubt.",
    },
    "fk_ambiguity": {
        "signals": r"(AmbiguousForeignKey|Could not determine join condition|foreign_keys)",
        "fix": "For multi-FK relationships to the same table, prefer specifying foreign_keys=[...] on relationship().",
    },
    "missing_selectinload": {
        "signals": r"(selectinload|eager load|detached instance)",
        "fix": "When reading relationships outside the session, prefer eager loading via selectinload().",
    },
    "async_greenlet": {
        "signals": r"(greenlet_spawn|MissingGreenlet|async_session)",
        "fix": "Use async_sessionmaker and await every DB call; avoid sync-only SQLAlchemy calls in async handlers.",
    },
    "db_flush_before_id": {
        "signals": r"(NoneType.*id|DetachedInstance|before commit)",
        "fix": "Before reading obj.id on a newly-created row, prefer db.flush() to populate the primary key.",
    },
    "import_cycle": {
        "signals": r"(partially initialized module|circular import)",
        "fix": "Break import cycles by moving shared types into a deps or types module; prefer local imports inside functions if needed.",
    },
    "missing_dependency": {
        "signals": r"(ModuleNotFoundError|No module named)",
        "fix": "Add the module to requirements.txt early; tester will install before syntax checks.",
    },
    "markdown_fences_in_file": {
        "signals": r"(```python|```bash|SyntaxError.*`)",
        "fix": "Write files with raw code only — never include markdown fences inside file content.",
    },
}


def goal_hash(goal: str) -> str:
    """Stable 16-char hex prefix of sha256(goal[:200])."""
    return hashlib.sha256(goal[:200].encode("utf-8")).hexdigest()[:16]


def extract_lesson(
    failure_summary: str,
    artifact_notes: Optional[List[str]] = None,
    *,
    role: str = "builder",
    source_phase: int = 0,
    goal: str = "",
) -> Optional[Dict[str, Any]]:
    """Classify a failure summary into a canonical pattern.

    Returns a Lesson dict. Unknown failures produce pattern="other" with confidence=0.3.
    Returns None only if `failure_summary` is empty.
    """
    text = (failure_summary or "") + "\n" + "\n".join(artifact_notes or [])
    text = text.strip()
    if not text:
        return None

    for pattern, meta in PATTERN_DB.items():
        if re.search(meta["signals"], text, re.IGNORECASE):
            return {
                "pattern": pattern,
                "fix": meta["fix"],
                "confidence": 0.6,
                "role": role,
                "goal_hash": goal_hash(goal),
                "source_phase": source_phase,
                "occurrences": 1,
            }

    # Fallback: low-confidence "other" lesson using raw summary
    return {
        "pattern": "other",
        "fix": (text[:200].replace("\n", " ") or "review failure context"),
        "confidence": 0.3,
        "role": role,
        "goal_hash": goal_hash(goal),
        "source_phase": source_phase,
        "occurrences": 1,
    }


def merge_lesson(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """When patterns match, bump occurrences and raise confidence (cap 0.95)."""
    if existing.get("pattern") != new.get("pattern"):
        # Patterns mismatch — caller should not have called merge; return new
        return new
    merged = dict(existing)
    merged["occurrences"] = int(existing.get("occurrences", 1)) + int(new.get("occurrences", 1))
    bumped = float(existing.get("confidence", 0.3)) + 0.1
    merged["confidence"] = min(0.95, bumped)
    # Keep the most recent goal_hash/source_phase — latest occurrence context wins
    merged["goal_hash"] = new.get("goal_hash", existing.get("goal_hash"))
    merged["source_phase"] = new.get("source_phase", existing.get("source_phase"))
    return merged


def format_for_prompt(lessons: List[Dict[str, Any]], max_items: int = 5) -> str:
    """Render lessons as positive-framed guidance for minion prompt injection.

    Positive phrasing only — we explicitly avoid "don't X" constructions because
    LLMs tend to overfit to avoidance (producing awkward code that dodges the
    trap rather than code that uses the right pattern).
    """
    if not lessons:
        return ""
    sorted_lessons = sorted(
        lessons,
        key=lambda L: (float(L.get("confidence", 0)), int(L.get("occurrences", 0))),
        reverse=True,
    )[:max_items]

    lines = ["LESSONS LEARNED (high-confidence first):"]
    for L in sorted_lessons:
        conf = float(L.get("confidence", 0))
        occ = int(L.get("occurrences", 1))
        fix = str(L.get("fix", ""))[:220]
        # Sanitize any accidental negative framing — we want positive guidance only
        if fix and fix.lower().startswith(("don't ", "do not ", "avoid ")):
            # Rewrite prefix to positive cue; leaves content unchanged otherwise
            fix = "Prefer the opposite of: " + fix
        lines.append(f"- {fix} (conf={conf:.2f}, seen {occ}×)")
    return "\n".join(lines)
