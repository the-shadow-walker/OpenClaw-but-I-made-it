"""Gmail client — sync wrapper around ``googleapiclient.discovery.build``.

Restored in P10+ from Mk2's ``tools/email_agent.py`` (archived at
``/mnt/storage/NAS/jarvis-archived/Jarvis-2026-04-26/tools/email_agent.py``).
The Mk2 version was 647 lines because it bundled four concerns:

1. Gmail API access (this file's job).
2. LLM classification + SQLite digest cache (dropped — live API only).
3. Markdown digest writer (dropped).
4. Background poller (dropped — call live each turn).

Mk3's split is cleaner: this client exposes the raw Gmail surface; the
LLM-facing tool wrappers live in ``jarvis/mail/tool_email.py`` and the
handlers know how to convert raw payloads into LLM-friendly dicts.

OAuth token storage:

* The token file is loaded from ``~/.config/jarvis/gmail_token.pickle``
  by default; override via ``EmailConfig.token_path``.
* Refresh-token flow runs lazily on first use (``_get_service``); if the
  token is expired and has a valid ``refresh_token``, ``Credentials.refresh``
  re-mints the access token in-place and writes the updated pickle back.
* If the token is missing, expired without a refresh_token, or revoked,
  every call returns a structured error envelope rather than raising —
  the chat loop surfaces the error to the model, which can tell the user
  to re-authorize.

The client never auto-runs the OAuth flow. Bootstrap auth is a one-shot
script (see ``scripts/gmail_oauth_bootstrap.py``) the user runs once.
"""

from __future__ import annotations

import base64
import logging
import pickle
import re
from dataclasses import dataclass
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["GmailClient", "GmailError", "GmailNotConfigured"]


class GmailError(Exception):
    """Wraps a Gmail API failure (auth, quota, network)."""


class GmailNotConfigured(GmailError):
    """Token file missing or unusable. User needs to re-authorize."""


@dataclass(frozen=True)
class EmailSummary:
    """Compact email descriptor for list/search results."""

    id: str
    thread_id: str
    from_name: str
    from_addr: str
    subject: str
    date: str
    snippet: str
    unread: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "from": f"{self.from_name} <{self.from_addr}>" if self.from_name and self.from_addr != self.from_name else self.from_addr,
            "subject": self.subject,
            "date": self.date,
            "snippet": self.snippet,
            "unread": self.unread,
        }


class GmailClient:
    """Lazy-loading Gmail v1 client.

    Service is loaded on first call and cached. Token refresh is
    transparent. All public methods catch ``HttpError`` / network
    failures and raise ``GmailError`` so the tool layer has one
    exception type to filter on.
    """

    def __init__(self, token_path: Path) -> None:
        self._token_path = Path(token_path).expanduser()
        self._service: Any = None  # lazily loaded

    # -- internals ---------------------------------------------------------

    def _get_service(self) -> Any:
        if self._service is not None:
            return self._service
        if not self._token_path.exists():
            raise GmailNotConfigured(
                f"gmail: token file not found at {self._token_path} — "
                f"run scripts/gmail_oauth_bootstrap.py to authorize"
            )
        try:
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build
        except ImportError as e:
            raise GmailError(f"gmail: missing dependency: {e}") from e
        try:
            with self._token_path.open("rb") as f:
                creds = pickle.load(f)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Persist refreshed token back so the next process inherits.
                with self._token_path.open("wb") as f:
                    pickle.dump(creds, f)
                logger.info("gmail: refreshed access token")
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            return self._service
        except GmailError:
            raise
        except Exception as e:  # noqa: BLE001
            raise GmailError(f"gmail: auth failed: {e}") from e

    @staticmethod
    def _parse_from(raw: str) -> tuple[str, str]:
        """Split a ``"Name <addr>"`` header into ``(name, addr)``."""
        if not raw:
            return ("", "")
        m = re.match(r'"?([^"<]+)"?\s*<([^>]+)>', raw)
        if m:
            return (m.group(1).strip(), m.group(2).strip())
        return ("", raw.strip())

    @staticmethod
    def _extract_body(payload: dict) -> str:
        """Walk the MIME tree, return the first text/plain body found.

        Falls back to text/html (stripped of tags, naively) if no plain
        part exists. Mk2 only handled text/plain — Mk3 makes a token
        attempt at HTML so newsletter/marketing reads aren't empty.
        """
        mime = payload.get("mimeType", "")
        body = payload.get("body", {}) or {}
        data = body.get("data")
        if mime == "text/plain" and data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        for part in payload.get("parts", []) or []:
            text = GmailClient._extract_body(part)
            if text:
                return text
        # HTML fallback — only after exhausting plain-text descent.
        if mime == "text/html" and data:
            html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            # Strip tags/scripts/styles — naive but adequate for a digest.
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<[^>]+>", " ", html)
            html = re.sub(r"\s+", " ", html).strip()
            return html
        return ""

    @staticmethod
    def _summarize(full: dict) -> EmailSummary:
        """Build an EmailSummary from a Gmail messages.get(format=metadata|full) response."""
        payload = full.get("payload", {}) or {}
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
        from_name, from_addr = GmailClient._parse_from(headers.get("From", ""))
        return EmailSummary(
            id=full["id"],
            thread_id=full.get("threadId", ""),
            from_name=from_name,
            from_addr=from_addr,
            subject=headers.get("Subject", "(no subject)"),
            date=headers.get("Date", ""),
            snippet=full.get("snippet", "") or "",
            unread="UNREAD" in (full.get("labelIds") or []),
        )

    # -- public surface ----------------------------------------------------

    def search(self, query: str = "", max_results: int = 10) -> list[EmailSummary]:
        """List or search messages.

        ``query`` uses Gmail's search syntax (``from:foo subject:bar
        is:unread newer_than:2d``). Empty query returns most recent.
        """
        svc = self._get_service()
        try:
            kwargs: dict[str, Any] = {"userId": "me", "maxResults": max_results}
            if query:
                kwargs["q"] = query
            resp = svc.users().messages().list(**kwargs).execute()
        except Exception as e:  # noqa: BLE001
            raise GmailError(f"gmail.search: {e}") from e
        ids = resp.get("messages") or []
        results: list[EmailSummary] = []
        for stub in ids:
            try:
                full = svc.users().messages().get(
                    userId="me", id=stub["id"], format="metadata",
                    metadataHeaders=["From", "Subject", "Date"],
                ).execute()
                results.append(self._summarize(full))
            except Exception:  # noqa: BLE001
                logger.exception("gmail.search: failed to fetch %s", stub.get("id"))
        return results

    def read(self, email_id: str) -> dict[str, Any]:
        """Fetch one message in full. Returns dict with body + headers."""
        svc = self._get_service()
        try:
            full = svc.users().messages().get(
                userId="me", id=email_id, format="full"
            ).execute()
        except Exception as e:  # noqa: BLE001
            raise GmailError(f"gmail.read({email_id}): {e}") from e
        payload = full.get("payload", {}) or {}
        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
        body = self._extract_body(payload)
        from_name, from_addr = self._parse_from(headers.get("From", ""))
        return {
            "id": full["id"],
            "thread_id": full.get("threadId", ""),
            "from": f"{from_name} <{from_addr}>" if from_name and from_addr != from_name else from_addr,
            "to": headers.get("To", ""),
            "cc": headers.get("Cc", ""),
            "subject": headers.get("Subject", "(no subject)"),
            "date": headers.get("Date", ""),
            "snippet": full.get("snippet", ""),
            "body": body[:8000],  # cap so the LLM doesn't choke on a 200KB email
            "body_truncated": len(body) > 8000,
            "labels": full.get("labelIds") or [],
        }

    def send(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
    ) -> dict[str, Any]:
        """Send a plain-text email. Returns ``{id, thread_id}`` on success."""
        svc = self._get_service()
        msg = MIMEText(body)
        msg["to"] = to
        msg["subject"] = subject
        if cc:
            msg["cc"] = cc
        if bcc:
            msg["bcc"] = bcc
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        try:
            sent = svc.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
        except Exception as e:  # noqa: BLE001
            raise GmailError(f"gmail.send: {e}") from e
        return {"id": sent.get("id"), "thread_id": sent.get("threadId")}

    def draft(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        cc: str | None = None,
        bcc: str | None = None,
    ) -> dict[str, Any]:
        """Save a draft. Returns ``{id, message_id}`` for later edit/send."""
        svc = self._get_service()
        msg = MIMEText(body)
        msg["to"] = to
        msg["subject"] = subject
        if cc:
            msg["cc"] = cc
        if bcc:
            msg["bcc"] = bcc
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        try:
            d = svc.users().drafts().create(
                userId="me", body={"message": {"raw": raw}}
            ).execute()
        except Exception as e:  # noqa: BLE001
            raise GmailError(f"gmail.draft: {e}") from e
        return {"id": d.get("id"), "message_id": (d.get("message") or {}).get("id")}
