"""LLM tool wrappers for the Gmail surface.

Four tools, one per public ``GmailClient`` method:

* ``email_search`` — list/search messages, return summaries
* ``email_read``   — fetch one message in full
* ``email_send``   — send a new email (LLM should confirm with user first)
* ``email_draft``  — save a draft (no send; safe by default)

Each handler catches ``GmailError`` and returns a structured dict with
``error`` set, so the chat tool loop surfaces the failure to the model
as ``tool_result`` content rather than killing the turn.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.clients.gmail import GmailClient, GmailError

logger = logging.getLogger(__name__)

__all__ = [
    "email_search_tool",
    "email_read_tool",
    "email_send_tool",
    "email_draft_tool",
]


def email_search_tool(
    *,
    query: str = "",
    max_results: int = 10,
    gmail: GmailClient,
) -> dict[str, Any]:
    """Wrap ``GmailClient.search``. Always returns a dict (not list)."""
    try:
        results = gmail.search(query=query, max_results=max_results)
    except GmailError as e:
        return {"error": str(e), "results": []}
    return {"error": None, "results": [r.to_dict() for r in results]}


def email_read_tool(*, email_id: str, gmail: GmailClient) -> dict[str, Any]:
    """Wrap ``GmailClient.read``."""
    try:
        return {"error": None, "email": gmail.read(email_id)}
    except GmailError as e:
        return {"error": str(e), "email": None}


def email_send_tool(
    *,
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    bcc: str | None = None,
    gmail: GmailClient,
) -> dict[str, Any]:
    """Wrap ``GmailClient.send``. The LLM should always confirm with the
    user before calling this — see workspace/AGENTS.md."""
    try:
        result = gmail.send(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
        return {"error": None, "sent": True, **result}
    except GmailError as e:
        return {"error": str(e), "sent": False}


def email_draft_tool(
    *,
    to: str,
    subject: str,
    body: str,
    cc: str | None = None,
    bcc: str | None = None,
    gmail: GmailClient,
) -> dict[str, Any]:
    """Wrap ``GmailClient.draft``. Safer than send — no delivery."""
    try:
        result = gmail.draft(to=to, subject=subject, body=body, cc=cc, bcc=bcc)
        return {"error": None, "drafted": True, **result}
    except GmailError as e:
        return {"error": str(e), "drafted": False}
