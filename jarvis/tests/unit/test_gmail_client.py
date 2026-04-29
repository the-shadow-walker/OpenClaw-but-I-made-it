"""Unit tests for GmailClient + email tool wrappers.

Mock pattern: substitute a fake ``service`` object onto ``GmailClient._service``
so ``_get_service`` short-circuits — no real OAuth, no real HTTP.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from jarvis.clients.gmail import GmailClient, GmailError, GmailNotConfigured
from jarvis.mail.tool_email import (
    email_draft_tool,
    email_read_tool,
    email_search_tool,
    email_send_tool,
)


# ---------------------------------------------------------------------------
# Fake service shim — mimics the bits of googleapiclient we touch.
# ---------------------------------------------------------------------------


class _FakeExec:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def execute(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeMessages:
    def __init__(self, list_resp: Any, get_map: dict[str, Any], send_resp: Any = None) -> None:
        self._list_resp = list_resp
        self._get_map = get_map
        self._send_resp = send_resp
        self.last_send_body: dict | None = None

    def list(self, **kwargs) -> _FakeExec:
        return _FakeExec(self._list_resp)

    def get(self, *, userId, id, **kwargs) -> _FakeExec:
        return _FakeExec(self._get_map.get(id, KeyError(f"unknown id {id}")))

    def send(self, *, userId, body) -> _FakeExec:
        self.last_send_body = body
        return _FakeExec(self._send_resp or {"id": "sent_xyz", "threadId": "thr_xyz"})


class _FakeDrafts:
    def __init__(self, create_resp: Any) -> None:
        self._create_resp = create_resp
        self.last_body: dict | None = None

    def create(self, *, userId, body) -> _FakeExec:
        self.last_body = body
        return _FakeExec(self._create_resp)


class _FakeUsers:
    def __init__(self, messages: _FakeMessages, drafts: _FakeDrafts) -> None:
        self._messages = messages
        self._drafts = drafts

    def messages(self) -> _FakeMessages:
        return self._messages

    def drafts(self) -> _FakeDrafts:
        return self._drafts


class _FakeService:
    def __init__(self, users: _FakeUsers) -> None:
        self._users = users

    def users(self) -> _FakeUsers:
        return self._users


def _build_client(*, list_resp, get_map, send_resp=None, draft_resp=None) -> tuple[GmailClient, _FakeMessages, _FakeDrafts]:
    msgs = _FakeMessages(list_resp, get_map, send_resp)
    drafts = _FakeDrafts(draft_resp or {"id": "draft_1", "message": {"id": "m1"}})
    svc = _FakeService(_FakeUsers(msgs, drafts))
    client = GmailClient(token_path=Path("/nonexistent/token.pickle"))
    client._service = svc  # bypass OAuth
    return client, msgs, drafts


# ---------------------------------------------------------------------------
# Token-loading edge cases
# ---------------------------------------------------------------------------


def test_missing_token_raises_not_configured(tmp_path: Path) -> None:
    client = GmailClient(token_path=tmp_path / "absent.pickle")
    with pytest.raises(GmailNotConfigured):
        client._get_service()


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_returns_summaries() -> None:
    list_resp = {"messages": [{"id": "m1"}, {"id": "m2"}]}
    get_map = {
        "m1": {
            "id": "m1", "threadId": "t1",
            "snippet": "first preview",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {"headers": [
                {"name": "From", "value": "Alice <alice@x.com>"},
                {"name": "Subject", "value": "Hello"},
                {"name": "Date", "value": "Mon, 1 Jan 2026 12:00:00 +0000"},
            ]},
        },
        "m2": {
            "id": "m2", "threadId": "t2",
            "snippet": "second preview",
            "labelIds": ["INBOX"],
            "payload": {"headers": [
                {"name": "From", "value": "bob@x.com"},
                {"name": "Subject", "value": "Re: Hi"},
                {"name": "Date", "value": "Mon, 1 Jan 2026 13:00:00 +0000"},
            ]},
        },
    }
    client, _, _ = _build_client(list_resp=list_resp, get_map=get_map)
    out = client.search(query="is:unread", max_results=2)
    assert len(out) == 2
    assert out[0].id == "m1"
    assert out[0].from_name == "Alice"
    assert out[0].from_addr == "alice@x.com"
    assert out[0].unread is True
    assert out[1].from_name == ""
    assert out[1].from_addr == "bob@x.com"
    assert out[1].unread is False


def test_search_empty_results() -> None:
    client, _, _ = _build_client(list_resp={}, get_map={})
    assert client.search() == []


def test_search_api_error_raises_gmail_error() -> None:
    client, _, _ = _build_client(list_resp=RuntimeError("api boom"), get_map={})
    with pytest.raises(GmailError):
        client.search()


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------


def _b64(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode("utf-8")).decode("utf-8")


def test_read_extracts_plaintext_body() -> None:
    full = {
        "id": "m1", "threadId": "t1",
        "snippet": "preview",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "Alice <a@x.com>"},
                {"name": "To", "value": "me@x.com"},
                {"name": "Subject", "value": "Hi"},
                {"name": "Date", "value": "Mon"},
            ],
            "parts": [
                {"mimeType": "text/plain", "body": {"data": _b64("plain body here")}},
                {"mimeType": "text/html", "body": {"data": _b64("<p>html version</p>")}},
            ],
        },
    }
    client, _, _ = _build_client(list_resp={}, get_map={"m1": full})
    out = client.read("m1")
    assert out["body"] == "plain body here"
    assert out["body_truncated"] is False
    assert out["from"] == "Alice <a@x.com>"
    assert out["subject"] == "Hi"


def test_read_falls_back_to_html() -> None:
    full = {
        "id": "m2", "threadId": "t2",
        "snippet": "",
        "labelIds": [],
        "payload": {
            "mimeType": "text/html",
            "headers": [{"name": "From", "value": "x@y.com"}],
            "body": {"data": _b64("<style>x{}</style><p>hello <b>world</b></p>")},
        },
    }
    client, _, _ = _build_client(list_resp={}, get_map={"m2": full})
    out = client.read("m2")
    assert "hello" in out["body"]
    assert "<p>" not in out["body"]
    assert "<style>" not in out["body"]


def test_read_caps_body_at_8000() -> None:
    long = "x" * 12_000
    full = {
        "id": "m3", "threadId": "t3", "snippet": "", "labelIds": [],
        "payload": {
            "mimeType": "text/plain",
            "headers": [],
            "body": {"data": _b64(long)},
        },
    }
    client, _, _ = _build_client(list_resp={}, get_map={"m3": full})
    out = client.read("m3")
    assert len(out["body"]) == 8000
    assert out["body_truncated"] is True


# ---------------------------------------------------------------------------
# send() / draft()
# ---------------------------------------------------------------------------


def test_send_returns_id_and_thread() -> None:
    client, msgs, _ = _build_client(
        list_resp={}, get_map={},
        send_resp={"id": "s1", "threadId": "thr1"},
    )
    out = client.send(to="bob@x.com", subject="hi", body="hello", cc="cc@x.com")
    assert out == {"id": "s1", "thread_id": "thr1"}
    # The raw body is base64 — decode and check headers landed.
    raw = base64.urlsafe_b64decode(msgs.last_send_body["raw"]).decode("utf-8")
    assert "to: bob@x.com" in raw
    assert "subject: hi" in raw
    assert "cc: cc@x.com" in raw


def test_draft_returns_id_and_message_id() -> None:
    client, _, drafts = _build_client(
        list_resp={}, get_map={},
        draft_resp={"id": "d1", "message": {"id": "m1"}},
    )
    out = client.draft(to="bob@x.com", subject="x", body="y")
    assert out == {"id": "d1", "message_id": "m1"}


# ---------------------------------------------------------------------------
# Tool wrappers
# ---------------------------------------------------------------------------


def test_email_search_tool_wraps_results() -> None:
    list_resp = {"messages": [{"id": "m1"}]}
    get_map = {"m1": {
        "id": "m1", "threadId": "t1", "snippet": "p",
        "labelIds": ["UNREAD"],
        "payload": {"headers": [
            {"name": "From", "value": "Alice <a@x.com>"},
            {"name": "Subject", "value": "Hi"},
            {"name": "Date", "value": ""},
        ]},
    }}
    client, _, _ = _build_client(list_resp=list_resp, get_map=get_map)
    out = email_search_tool(query="is:unread", gmail=client)
    assert out["error"] is None
    assert len(out["results"]) == 1
    assert out["results"][0]["unread"] is True


def test_email_search_tool_catches_gmail_error() -> None:
    client, _, _ = _build_client(list_resp=RuntimeError("nope"), get_map={})
    out = email_search_tool(gmail=client)
    assert out["error"] is not None
    assert out["results"] == []


def test_email_read_tool_envelope() -> None:
    full = {
        "id": "m1", "threadId": "t1", "snippet": "", "labelIds": [],
        "payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "From", "value": "x@y.com"}],
            "body": {"data": _b64("body")},
        },
    }
    client, _, _ = _build_client(list_resp={}, get_map={"m1": full})
    out = email_read_tool(email_id="m1", gmail=client)
    assert out["error"] is None
    assert out["email"]["body"] == "body"


def test_email_send_tool_returns_sent_true() -> None:
    client, _, _ = _build_client(
        list_resp={}, get_map={},
        send_resp={"id": "s1", "threadId": "t1"},
    )
    out = email_send_tool(to="x@y.com", subject="s", body="b", gmail=client)
    assert out["error"] is None
    assert out["sent"] is True
    assert out["id"] == "s1"


def test_email_draft_tool_returns_drafted_true() -> None:
    client, _, _ = _build_client(
        list_resp={}, get_map={},
        draft_resp={"id": "d1", "message": {"id": "m1"}},
    )
    out = email_draft_tool(to="x@y.com", subject="s", body="b", gmail=client)
    assert out["error"] is None
    assert out["drafted"] is True
    assert out["id"] == "d1"
