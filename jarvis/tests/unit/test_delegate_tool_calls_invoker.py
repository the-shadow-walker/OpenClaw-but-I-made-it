"""Delegate tool — verifies the registry handler routes through invoker
when all delegation deps are wired, and falls back to the stub otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.core.tools import build_default_registry
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


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
def conversation(paths, conn) -> Conversation:
    return Conversation.open(
        channel_kind="cli", channel_id="t",
        paths=paths, conn=conn, cfg=ConversationConfig(),
    )


class FakeCMDClient:
    def __init__(self) -> None:
        self.execute_calls: list[dict] = []
        self.quick_calls: list[dict] = []

    def execute(self, instruction, *, context_keys=None, model=None, timeout_s=None):
        self.execute_calls.append({"instruction": instruction,
                                   "context_keys": context_keys})
        return {
            "success": True, "summary": "ok", "deliverables": [],
            "context_keys_written": [], "sidechain_path": "/sc/x.jsonl",
            "error": None,
        }

    def quick(self, *, command=None, question=None, timeout_s=None, allow_risk="low"):
        self.quick_calls.append({"command": command, "question": question})
        return {"returncode": 0, "stdout": "ok", "stderr": ""}


# ---------------------------------------------------------------------------


def test_delegate_dispatches_to_cmd_react(conn, paths, conversation, shared_board):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    r = build_default_registry(
        conn=conn, embedder=None, paths=paths, channel_kind="cli",
        cmd_client=cmd, arbiter=arbiter,
        conversation=conversation, shared_board=shared_board,
    )
    out = r.execute("delegate", {"target": "cmd:react", "task": "build x"})
    assert len(cmd.execute_calls) == 1
    assert cmd.execute_calls[0]["instruction"] == "build x"
    assert out["success"] is True
    assert out["sidechain_path"] == "/sc/x.jsonl"


def test_delegate_returns_envelope_dict(conn, paths, conversation, shared_board):
    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    r = build_default_registry(
        conn=conn, embedder=None, paths=paths, channel_kind="cli",
        cmd_client=cmd, arbiter=arbiter,
        conversation=conversation, shared_board=shared_board,
    )
    out = r.execute("delegate", {"target": "cmd:quick", "task": "uptime"})
    assert {"success", "summary", "error", "deliverables",
            "context_keys_written", "sidechain_path"} <= set(out.keys())


def test_existing_stub_path_still_works(conn, paths):
    """When delegation deps are not wired, the registry registers the legacy
    stub. This test covers backward compatibility for the existing
    test_tools_registry suite — the assertion mirrors
    test_delegate_stub_returns_error.
    """
    r = build_default_registry(
        conn=conn, embedder=None, paths=paths, channel_kind="cli",
    )
    out = r.execute("delegate", {"target": "cmd:quick", "task": "hi"})
    assert out["error"].startswith("delegate not implemented")
    assert out["target"] == "cmd:quick"
