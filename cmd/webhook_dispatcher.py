"""
webhook_dispatcher.py — Fire-and-forget outbound webhook POSTs.

Configure target URLs via env var (comma-separated):
    AGENT_WEBHOOK_URLS=http://host1/webhook,http://host2/events

Events dispatched (event field values):
    job_completed, job_failed
    chain_completed, chain_failed, chain_cancelled
    chain_subtask_passed, chain_subtask_failed

Payload shape (all events):
    {
        "event":      "<event_type>",
        "timestamp":  "<iso8601>",
        "service":    "ollama-agent",
        ...event-specific fields...
    }
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_TIMEOUT = 10  # seconds per call


def _webhook_urls() -> List[str]:
    raw = os.environ.get("AGENT_WEBHOOK_URLS", "")
    return [u.strip() for u in raw.split(",") if u.strip()]


def _fire(url: str, payload: Dict[str, Any]) -> None:
    if not _HAS_REQUESTS:
        return
    try:
        _requests.post(url, json=payload, timeout=_TIMEOUT)
    except Exception:
        pass  # fire-and-forget — never raises


def dispatch(event_type: str, data: Dict[str, Any]) -> None:
    """POST event to all configured webhook URLs (non-blocking threads)."""
    urls = _webhook_urls()
    if not urls:
        return
    payload = {
        "event": event_type,
        "timestamp": datetime.now().isoformat(),
        "service": "ollama-agent",
        **data,
    }
    for url in urls:
        threading.Thread(target=_fire, args=(url, payload), daemon=True).start()


# ─────────────────────────── convenience wrappers ───────────────────────────

def job_completed(job_id: str, instruction: str, success: bool,
                  summary: str = "", iterations_used: int = 0,
                  chain_id: Optional[str] = None):
    event = "job_completed" if success else "job_failed"
    dispatch(event, {
        "job_id": job_id,
        "instruction": instruction[:300],
        "success": success,
        "summary": summary[:500],
        "iterations_used": iterations_used,
        "chain_id": chain_id,
    })


def chain_status_changed(chain_id: str, goal: str, status: str, subtask_count: int):
    dispatch(f"chain_{status}", {
        "chain_id": chain_id,
        "goal": goal[:300],
        "status": status,
        "subtask_count": subtask_count,
    })


def subtask_result(chain_id: str, subtask_index: int, instruction: str,
                   ac_passed: bool, summary: str = ""):
    event = "chain_subtask_passed" if ac_passed else "chain_subtask_failed"
    dispatch(event, {
        "chain_id": chain_id,
        "subtask_index": subtask_index,
        "instruction": instruction[:300],
        "ac_passed": ac_passed,
        "summary": summary[:300],
    })
