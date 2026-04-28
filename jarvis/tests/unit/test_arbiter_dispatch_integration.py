"""End-to-end arbiter ↔ dispatch integration (P10).

Verifies the two main P10 invariants:

  * The first ``cmd:code`` / ``cmd:gui`` dispatch in a conversation
    claims the master role (first-write-wins).
  * Subsequent cross-mode (``cmd:code`` after ``gui`` master, or
    ``cmd:gui`` after ``code`` master) dispatches carry
    ``master_mode: "<other-mode>"`` in the HTTP body. Same-mode and
    non-arbitrated targets (cmd:react / cmd:quick / swarm:*) never
    grow the field.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.core.invoker import dispatch
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
    return Conversation.open(
        channel_kind="cli",
        channel_id="test",
        paths=paths,
        conn=conn,
        cfg=ConversationConfig(),
    )


class RecordingCMDClient:
    """Records every execute() body and returns a canned success envelope.

    Mirrors the relevant subset of ``CMDClient.execute``'s signature so the
    invoker can call it transparently. ``quick()`` is included for
    completeness even though P10 arbitration never touches it.
    """

    def __init__(self) -> None:
        self.execute_calls: list[dict] = []
        self.quick_calls: list[dict] = []

    def execute(
        self,
        instruction,
        *,
        context_keys=None,
        model=None,
        timeout_s=None,
        mode=None,
        master_mode=None,
    ):
        self.execute_calls.append({
            "instruction": instruction,
            "context_keys": context_keys,
            "model": model,
            "timeout_s": timeout_s,
            "mode": mode,
            "master_mode": master_mode,
        })
        return {
            "success": True,
            "summary": "ok",
            "deliverables": [],
            "context_keys_written": [],
            "sidechain_path": "/sc/x.jsonl",
            "error": None,
        }

    def quick(self, *, command=None, question=None, timeout_s=None, allow_risk="low"):
        self.quick_calls.append({
            "command": command,
            "question": question,
            "timeout_s": timeout_s,
            "allow_risk": allow_risk,
        })
        return {"returncode": 0, "stdout": "", "stderr": ""}


class RecordingSwarmClient:
    """Stub swarm client: records dispatch calls; the body shape isn't
    arbitrated so no master_mode threading exists here."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def dispatch(
        self, role, task, *, context_keys=None, max_iterations=40, timeout_s=None,
    ):
        self.calls.append({
            "role": role,
            "task": task,
            "context_keys": context_keys,
            "max_iterations": max_iterations,
            "timeout_s": timeout_s,
        })
        return {
            "success": True,
            "summary": "ok",
            "deliverables": [],
            "context_keys_written": [],
            "sidechain_path": "/sc/y.jsonl",
            "error": None,
        }


# ---------------------------------------------------------------------------
# claim() invariants
# ---------------------------------------------------------------------------


def test_first_cmd_code_claims_master_no_subordinate_flag(
    conversation, paths, shared_board,
):
    cmd = RecordingCMDClient()
    arbiter = RoleArbiter()
    dispatch(
        target="cmd:code", task="list /tmp",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert arbiter.master_for(conversation.conv_id) == "code"
    assert len(cmd.execute_calls) == 1
    body = cmd.execute_calls[0]
    assert body["mode"] == "code"
    # First-master dispatch is NOT subordinate → master_mode field omitted.
    assert body["master_mode"] is None


def test_subsequent_cmd_gui_dispatches_with_master_mode_code(
    conversation, paths, shared_board,
):
    cmd = RecordingCMDClient()
    arbiter = RoleArbiter()
    arbiter.claim(conversation.conv_id, "code")
    dispatch(
        target="cmd:gui", task="screenshot",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    body = cmd.execute_calls[0]
    assert body["mode"] == "gui"
    assert body["master_mode"] == "code"
    # JSONL bool flips True for the subordinate dispatch.
    transcript = conversation.transcript_path.read_text(encoding="utf-8")
    events = [json.loads(ln) for ln in transcript.splitlines() if ln.strip()]
    delegations = [e for e in events if e["kind"] == "delegation_envelope"]
    assert len(delegations) == 1
    assert delegations[0]["payload"]["master_mode"] is True
    assert delegations[0]["payload"]["target"] == "cmd:gui"


def test_first_cmd_gui_then_cmd_code_subordinate_with_master_mode_gui(
    conversation, paths, shared_board,
):
    cmd = RecordingCMDClient()
    arbiter = RoleArbiter()
    dispatch(
        target="cmd:gui", task="paint",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    dispatch(
        target="cmd:code", task="build",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert arbiter.master_for(conversation.conv_id) == "gui"
    first, second = cmd.execute_calls
    assert first["mode"] == "gui"
    assert first["master_mode"] is None
    assert second["mode"] == "code"
    assert second["master_mode"] == "gui"


def test_cmd_react_after_code_claim_no_master_mode_field(
    conversation, paths, shared_board,
):
    """cmd:react is exempt from arbitration: body has no mode/master_mode."""
    cmd = RecordingCMDClient()
    arbiter = RoleArbiter()
    arbiter.claim(conversation.conv_id, "code")
    dispatch(
        target="cmd:react", task="do thing",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    body = cmd.execute_calls[0]
    assert body["mode"] is None
    assert body["master_mode"] is None


def test_swarm_dispatch_after_code_claim_no_master_mode_field(
    conversation, paths, shared_board,
):
    """swarm:* dispatches never consult or set the arbiter."""
    cmd = RecordingCMDClient()
    swarm = RecordingSwarmClient()
    arbiter = RoleArbiter()
    arbiter.claim(conversation.conv_id, "code")
    dispatch(
        target="swarm:engineer", task="design",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, swarm_client=swarm, arbiter=arbiter,
    )
    # CMD untouched; swarm got the call without mode-related fields.
    assert cmd.execute_calls == []
    assert len(swarm.calls) == 1
    assert "mode" not in swarm.calls[0]
    assert "master_mode" not in swarm.calls[0]
    # Master role unchanged.
    assert arbiter.master_for(conversation.conv_id) == "code"


def test_cmd_code_dispatched_twice_first_write_wins(
    conversation, paths, shared_board,
):
    """Two cmd:code dispatches: master stays code; both bodies carry
    mode='code'; neither has master_mode set (same-mode, not subordinate)."""
    cmd = RecordingCMDClient()
    arbiter = RoleArbiter()
    dispatch(
        target="cmd:code", task="t1",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    dispatch(
        target="cmd:code", task="t2",
        conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=cmd, arbiter=arbiter,
    )
    assert arbiter.master_for(conversation.conv_id) == "code"
    assert len(cmd.execute_calls) == 2
    for body in cmd.execute_calls:
        assert body["mode"] == "code"
        assert body["master_mode"] is None
