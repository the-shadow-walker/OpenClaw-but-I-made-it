"""Invoker — snapshot, dispatch, merge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.core.invoker import (
    _quick_to_envelope,
    _safe_truncate,
    dispatch,
    merge,
    restore_from_snapshot,
    snapshot,
)
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    p = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(p)
    return p


@pytest.fixture
def conn(paths: WorkspacePaths):
    c = get_connection(paths.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def shared_board(tmp_path: Path) -> Path:
    p = tmp_path / "agent_bin"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def conversation(paths: WorkspacePaths, conn) -> Conversation:
    cfg = ConversationConfig()
    return Conversation.open(
        channel_kind="cli", channel_id="test", paths=paths, conn=conn, cfg=cfg
    )


class FakeCMDClient:
    """Records every call; returns scripted envelopes for execute / quick."""

    def __init__(
        self,
        *,
        execute_response: dict | None = None,
        quick_response: dict | None = None,
        execute_raises: Exception | None = None,
        quick_raises: Exception | None = None,
    ) -> None:
        self.execute_calls: list[dict] = []
        self.quick_calls: list[dict] = []
        self._execute_response = execute_response or {
            "success": True, "summary": "ok",
            "deliverables": [], "context_keys_written": [],
            "sidechain_path": "/sidechains/x.jsonl", "error": None,
        }
        self._quick_response = quick_response or {
            "returncode": 0, "stdout": "load average\n", "stderr": ""
        }
        self._execute_raises = execute_raises
        self._quick_raises = quick_raises

    def execute(self, instruction, *, context_keys=None, model=None, timeout_s=None):
        self.execute_calls.append({
            "instruction": instruction, "context_keys": context_keys,
            "model": model, "timeout_s": timeout_s,
        })
        if self._execute_raises is not None:
            raise self._execute_raises
        return self._execute_response

    def quick(self, *, command=None, question=None, timeout_s=None, allow_risk="low"):
        self.quick_calls.append({
            "command": command, "question": question,
            "timeout_s": timeout_s, "allow_risk": allow_risk,
        })
        if self._quick_raises is not None:
            raise self._quick_raises
        return self._quick_response


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_writes_file_and_round_trips(conversation, shared_board):
    conversation.append("user_message", {"content": "hi"})
    p = snapshot(conversation=conversation, label="pre_test", shared_board=shared_board)
    assert p.exists()
    assert "jarvis_" in p.name and "pre_test" in p.name
    payload = restore_from_snapshot(p)
    assert payload["conv_id"] == conversation.conv_id
    assert payload["label"] == "pre_test"
    assert payload["transcript_path"] == str(conversation.transcript_path)
    assert payload["transcript_offset_lines"] == 1


def test_snapshot_idempotent_dir_create(conversation, tmp_path: Path):
    sb = tmp_path / "fresh_board"  # does not yet exist
    assert not sb.exists()
    p = snapshot(conversation=conversation, label="pre_test", shared_board=sb)
    assert p.parent == sb / "sessions"
    assert p.exists()


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------


def test_dispatch_routes_cmd_quick_to_quick(conversation, paths, shared_board):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    env = dispatch(
        target="cmd:quick", task="is uptime ok?",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert len(cmd.quick_calls) == 1
    assert cmd.quick_calls[0]["question"] == "is uptime ok?"
    assert env["success"] is True
    # Summary derived from stdout.
    assert "load average" in env["summary"]


def test_dispatch_routes_cmd_react_to_execute(conversation, paths, shared_board):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    env = dispatch(
        target="cmd:react", task="build a thing",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter, context_keys=["k1"],
    )
    assert len(cmd.execute_calls) == 1
    assert cmd.execute_calls[0]["instruction"] == "build a thing"
    assert cmd.execute_calls[0]["context_keys"] == ["k1"]
    assert env["success"] is True
    assert env["sidechain_path"] == "/sidechains/x.jsonl"


@pytest.mark.parametrize(
    "target",
    ["cmd:chain", "cmd:gui", "cmd:blue"],
)
def test_dispatch_unsupported_cmd_targets_return_err_envelope(
    conversation, paths, shared_board, target,
):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    env = dispatch(
        target=target, task="x",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert env["success"] is False
    assert "not implemented" in env["error"]
    # Neither client method was called.
    assert cmd.execute_calls == []
    assert cmd.quick_calls == []


@pytest.mark.parametrize(
    "target",
    ["swarm:math", "swarm:engineer", "swarm:research"],
)
def test_dispatch_swarm_targets_with_no_swarm_client_return_err_envelope(
    conversation, paths, shared_board, target,
):
    """P8: swarm:* targets are wired, but a missing swarm_client (the
    pre-P8 backward-compat path) lands a degraded-mode envelope."""
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    env = dispatch(
        target=target, task="x",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert env["success"] is False
    assert "Swarm client" in env["error"] or "swarm" in env["error"].lower()


def test_dispatch_envelope_excludes_meta(conversation, paths, shared_board):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    env = dispatch(
        target="cmd:react", task="build x",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    # LLM-facing envelope: contract keys only.
    assert set(env.keys()) == {
        "success", "summary", "deliverables",
        "context_keys_written", "sidechain_path", "error",
    }


def test_dispatch_envelope_error_passes_through(conversation, paths, shared_board):
    cmd = FakeCMDClient(execute_response={
        "success": False, "summary": None, "deliverables": [],
        "context_keys_written": [], "sidechain_path": None,
        "error": "safety: rm -rf / blocked",
    })
    arbiter = RoleArbiter()
    env = dispatch(
        target="cmd:react", task="dangerous",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert env["success"] is False
    assert env["error"] == "safety: rm -rf / blocked"


def test_dispatch_no_cmd_client_returns_err_envelope(
    conversation, paths, shared_board,
):
    arbiter = RoleArbiter()
    env = dispatch(
        target="cmd:react", task="x",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=None, arbiter=arbiter,
    )
    assert env["success"] is False
    assert "no CMD client" in env["error"] or "degraded" in env["error"]


def test_dispatch_master_mode_flag_recorded_in_jsonl(
    conversation, paths, shared_board,
):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    arbiter.set_master(conversation.conv_id, "code")
    dispatch(
        target="cmd:gui", task="paint", conversation=conversation,
        paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    # cmd:gui returns "not implemented" envelope, but master_mode=True is
    # still recorded in the JSONL event.
    transcript = conversation.transcript_path.read_text(encoding="utf-8")
    events = [json.loads(ln) for ln in transcript.splitlines() if ln.strip()]
    delegations = [e for e in events if e["kind"] == "delegation_envelope"]
    assert len(delegations) == 1
    assert delegations[0]["payload"]["master_mode"] is True
    assert delegations[0]["payload"]["target"] == "cmd:gui"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_quick_to_envelope_returncode_nonzero_is_failure():
    env = _quick_to_envelope({"returncode": 1, "stdout": "", "stderr": ""})
    assert env["success"] is False
    assert env["error"] == "exit 1"


def test_quick_to_envelope_returncode_zero_is_success():
    env = _quick_to_envelope({"returncode": 0, "stdout": "x", "stderr": ""})
    assert env["success"] is True
    assert env["summary"] == "x"
    assert env["error"] is None


def test_safe_truncate_handles_multibyte():
    """A multi-byte char straddling the cut boundary must not produce
    an invalid UTF-8 sequence."""
    # Build a string where byte position ~500 lands inside a 3-byte char.
    # Use the heart emoji's underlying char "❤" (3 bytes in UTF-8).
    s = "a" * 498 + "❤❤❤"  # bytes: 498 + 9 = 507
    out = _safe_truncate(s, 500)
    # Round-trip must succeed (no UnicodeDecodeError).
    out.encode("utf-8")
    # Length check: <= 500 bytes.
    assert len(out.encode("utf-8")) <= 500


def test_safe_truncate_short_input_unchanged():
    assert _safe_truncate("hi", 500) == "hi"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_appends_jsonl_event(conversation):
    envelope = {
        "success": True, "summary": "did x", "deliverables": ["/tmp/x"],
        "context_keys_written": [], "sidechain_path": "/sc/y.jsonl",
        "error": None,
    }
    merge(
        envelope=envelope, conversation=conversation,
        target="cmd:react", task="t" * 1000,  # >500 → truncated
        snapshot_path=Path("/tmp/snap.context"),
        ms_elapsed=42, master_mode=False,
    )
    transcript = conversation.transcript_path.read_text(encoding="utf-8")
    events = [json.loads(ln) for ln in transcript.splitlines() if ln.strip()]
    delegations = [e for e in events if e["kind"] == "delegation_envelope"]
    assert len(delegations) == 1
    p = delegations[0]["payload"]
    assert p["target"] == "cmd:react"
    assert len(p["task"].encode("utf-8")) <= 500
    assert p["snapshot_path"] == "/tmp/snap.context"
    assert p["envelope"] == envelope
    assert p["ms_elapsed"] == 42
    assert p["master_mode"] is False
