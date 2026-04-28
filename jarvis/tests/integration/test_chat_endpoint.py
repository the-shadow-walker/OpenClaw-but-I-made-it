"""FastAPI chat endpoint — in-process via httpx.ASGITransport, fake Ollama.

No real network. Spins up ``create_app`` against a tmp workspace, an empty
indexer, and a scripted ``OllamaClient`` subclass, and verifies the NDJSON
stream is well-formed and event ordering is correct.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jarvis.clients.ollama import OllamaClient, OllamaResponse, OllamaToolCall
from jarvis.config import (
    CompactionConfig,
    JarvisConfig,
    LLMConfig,
    PathsConfig,
)
from jarvis.config import (
    ConversationConfig as JarvisConvSection,
)
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.index import Indexer
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from jarvis.server import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeOllama(OllamaClient):
    """OllamaClient subclass that returns scripted responses (no httpx)."""

    def __init__(self, scripted: list[OllamaResponse]) -> None:
        self._scripted = deque(scripted)
        self._calls: list[dict] = []

    def chat(self, model, messages, *, tools=None, system=None, **opts) -> OllamaResponse:
        self._calls.append({"model": model, "messages": list(messages),
                            "tools": tools, "system": system})
        if not self._scripted:
            raise RuntimeError("FakeOllama: ran out of scripted responses")
        return self._scripted.popleft()

    def close(self) -> None:
        pass


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    return paths


@pytest.fixture
def cfg() -> JarvisConfig:
    return JarvisConfig()


@pytest.fixture
def indexer(workspace: WorkspacePaths):
    idx = Indexer(workspace.index_dir / "memory.sqlite", workspace.root)
    yield idx
    idx.close()


def _make_client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_returns_ok(workspace, cfg, indexer):
    ollama = FakeOllama([])
    app = create_app(paths=workspace, cfg=cfg, indexer=indexer,
                     ollama=ollama, embedder=None)
    with _make_client(app) as cli:
        resp = cli.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["embedder"] == "degraded"   # no embedder wired in this test


def test_session_returns_conv_id(workspace, cfg, indexer):
    ollama = FakeOllama([])
    app = create_app(paths=workspace, cfg=cfg, indexer=indexer,
                     ollama=ollama, embedder=None)
    with _make_client(app) as cli:
        resp = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "test-host",
        })
    assert resp.status_code == 200
    body = resp.json()
    assert "conv_id" in body
    assert body["channel_kind"] == "cli"
    assert len(body["conv_id"]) == 22   # YYYYMMDDTHHMMSS-XXXXXX


def _read_ndjson(resp_text: str) -> list[dict]:
    return [json.loads(ln) for ln in resp_text.splitlines() if ln.strip()]


def test_chat_streams_ordered_events(workspace, cfg, indexer):
    """One tool round-trip: tool_call → tool_result → delta → done."""
    tc = OllamaToolCall(call_id="tc-1", name="memory_search",
                        arguments={"query": "anything"})
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[tc],
                       done_reason=None),
        OllamaResponse(role="assistant", content="found nothing relevant",
                       tool_calls=[], done_reason="stop"),
    ])
    app = create_app(paths=workspace, cfg=cfg, indexer=indexer,
                     ollama=ollama, embedder=None)

    with _make_client(app) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "test-host",
        }).json()
        conv_id = sess["conv_id"]

        resp = cli.post("/api/chat", json={
            "conv_id": conv_id, "text": "search for nothing",
            "channel_kind": "cli", "channel_id": "test-host",
        })

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    events = _read_ndjson(resp.text)
    types = [e["type"] for e in events]
    assert types == [
        "system_prompt", "tool_call", "tool_result", "delta", "done",
    ]
    # All events carry conv_id.
    assert all(e["conv_id"] == conv_id for e in events)
    # The delta must contain the model's final answer.
    assert "found nothing relevant" in events[-2]["text"]
    assert events[-1]["stop_reason"] == "stop"


def test_chat_handles_ollama_failure_gracefully(workspace, cfg, indexer):
    """If Ollama raises mid-stream, the response still terminates with
    error + done events (never a 500 mid-stream)."""

    class BoomOllama(FakeOllama):
        def chat(self, *_a, **_kw):
            raise RuntimeError("connection refused")

    ollama = BoomOllama([])
    app = create_app(paths=workspace, cfg=cfg, indexer=indexer,
                     ollama=ollama, embedder=None)

    with _make_client(app) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "boom",
        }).json()
        resp = cli.post("/api/chat", json={
            "conv_id": sess["conv_id"], "text": "x",
            "channel_kind": "cli", "channel_id": "boom",
        })

    assert resp.status_code == 200
    events = _read_ndjson(resp.text)
    types = [e["type"] for e in events]
    assert "error" in types
    assert events[-1]["type"] == "done"
    assert events[-1]["stop_reason"] == "error"


def test_chat_endpoint_emits_compaction_event_when_triggered(workspace, indexer):
    """A user_text larger than ``trigger_pct * context_window`` forces
    ``maybe_compact`` to fire on the first iteration. The endpoint streams
    normally; the JSONL transcript records one ``compaction`` event."""
    configure_tokenizer("approximation")
    tight_cfg = JarvisConfig(
        llm=LLMConfig(context_window=400),
        conversation=JarvisConvSection(
            compaction=CompactionConfig(
                trigger_pct=0.5,
                keep_recent_turns=2,
                reserve_tokens_floor=50,
                auto_flush=False,
            )
        ),
    )
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="ok", tool_calls=[],
                       done_reason="stop"),
    ])
    app = create_app(paths=workspace, cfg=tight_cfg, indexer=indexer,
                     ollama=ollama, embedder=None)
    big_text = "filler word " * 200  # ~600 tokens, well past 200-token trigger

    with _make_client(app) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "compaction-host",
        }).json()
        conv_id = sess["conv_id"]
        resp = cli.post("/api/chat", json={
            "conv_id": conv_id, "text": big_text,
            "channel_kind": "cli", "channel_id": "compaction-host",
        })

    assert resp.status_code == 200
    # Find the JSONL transcript and look for a compaction event.
    transcript = workspace.conversations_dir / f"{conv_id}.jsonl"
    events = [json.loads(ln) for ln in transcript.read_text().splitlines()]
    compactions = [e for e in events if e["kind"] == "compaction"]
    assert compactions, "expected at least one compaction event in the JSONL"
    payload = compactions[0]["payload"]
    assert payload["fired"] is True


def test_chat_delegation_event_flows_through(workspace, cfg, indexer):
    """Scripted Ollama emits delegate(cmd:react), then a final answer.
    FakeCMDClient.execute returns a scripted envelope. Verify the NDJSON
    stream carries tool_call(delegate) → tool_result(success=True), and
    the JSONL transcript captures a delegation_envelope event."""
    tc = OllamaToolCall(
        call_id="tc-1", name="delegate",
        arguments={"target": "cmd:react", "task": "build x"},
    )
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[tc],
                       done_reason=None),
        OllamaResponse(role="assistant", content="done — see /tmp/x",
                       tool_calls=[], done_reason="stop"),
    ])

    class FakeCMDClient:
        def execute(self, instruction, *, context_keys=None, model=None,
                    timeout_s=None, mode=None, master_mode=None):
            return {
                "success": True, "summary": "wrote /tmp/x",
                "deliverables": ["/tmp/x"], "context_keys_written": [],
                "sidechain_path": "/home/foo/.agent_bin/sidechains/j-1.jsonl",
                "error": None,
            }

        def quick(self, **kw):
            raise RuntimeError("quick should not be called")

        def close(self) -> None:
            pass

    from jarvis.core.arbiter import RoleArbiter

    cmd = FakeCMDClient()
    arbiter = RoleArbiter()
    app = create_app(
        paths=workspace, cfg=cfg, indexer=indexer,
        ollama=ollama, embedder=None,
        cmd_client=cmd, arbiter=arbiter,
    )

    with _make_client(app) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "delegation-host",
        }).json()
        conv_id = sess["conv_id"]
        resp = cli.post("/api/chat", json={
            "conv_id": conv_id, "text": "build me a thing",
            "channel_kind": "cli", "channel_id": "delegation-host",
        })
    assert resp.status_code == 200

    events = _read_ndjson(resp.text)
    types_with_names = [(e["type"], e.get("name")) for e in events]
    assert ("tool_call", "delegate") in types_with_names
    tool_results = [e for e in events
                    if e["type"] == "tool_result" and e.get("name") == "delegate"]
    assert len(tool_results) == 1
    assert tool_results[0]["error"] is None
    assert tool_results[0]["result"]["success"] is True

    # JSONL transcript records the delegation_envelope event.
    transcript = workspace.conversations_dir / f"{conv_id}.jsonl"
    raw_events = [json.loads(ln) for ln in transcript.read_text().splitlines()]
    delegations = [e for e in raw_events if e["kind"] == "delegation_envelope"]
    assert len(delegations) == 1
    assert delegations[0]["payload"]["target"] == "cmd:react"
    # Anti-pattern §19 #3: never inline ReAct internals.
    blob = json.dumps(delegations[0]["payload"])
    for forbidden in ("react_log", "execution_log", "tool_internal"):
        assert forbidden not in blob, (
            f"delegation_envelope payload leaked {forbidden!r}"
        )


def test_chat_rocket_sim_plan_and_execute_flow(workspace, cfg, indexer):
    """P8 rocket-sim shape: LLM emits plan_and_execute → planner returns
    3-node DAG → orchestrator dispatches via swarm/cmd → final answer
    contains all 4 absolute deliverable paths.

    Asserts §16 binding gates:
      A. 3 delegation_snapshot + 3 delegation_envelope events; ZERO
         react_log/sub_thought/tool_internal substrings in delegation
         payloads (parsed JSON, not raw grep).
      C. Final delta contains all 4 absolute paths verbatim.
    """
    # Iteration 1: model picks plan_and_execute.
    plan_call = OllamaToolCall(
        call_id="tc-1", name="plan_and_execute",
        arguments={
            "user_request": (
                "Build me a single-stage rocket simulator: math model, "
                "Python implementation, README, and a brief background "
                "research note."
            )
        },
    )
    # Planner LLM call: emits submit_plan with 3 nodes.
    submit_plan = OllamaToolCall(
        call_id="tc-plan", name="submit_plan",
        arguments={
            "rationale": "decompose into research → math → engineer",
            "nodes": [
                {"id": "R", "target": "swarm:research",
                 "task": "background note on rocket sims",
                 "depends_on": [], "publish_keys": ["bg"], "consume_keys": []},
                {"id": "M", "target": "swarm:math",
                 "task": "derive equations of motion",
                 "depends_on": ["R"], "publish_keys": ["eom"],
                 "consume_keys": ["bg"]},
                {"id": "E", "target": "cmd:react",
                 "task": "implement RK4 + write README",
                 "depends_on": ["M"], "publish_keys": [], "consume_keys": ["eom"]},
            ],
        },
    )
    # Iteration 2: model summarizes after the orchestrator returns.
    final_text = (
        "Done. See /tmp/research.md, /tmp/eom.md, /tmp/sim.py, /tmp/README.md."
    )
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[plan_call],
                       done_reason=None),
        OllamaResponse(role="assistant", content="", tool_calls=[submit_plan],
                       done_reason=None),
        OllamaResponse(role="assistant", content=final_text,
                       tool_calls=[], done_reason="stop"),
    ])

    class FakeCMDClient:
        def execute(self, instruction, *, context_keys=None, model=None,
                    timeout_s=None, mode=None, master_mode=None):
            return {
                "success": True, "summary": "wrote sim + README",
                "deliverables": ["/tmp/sim.py", "/tmp/README.md"],
                "context_keys_written": [],
                "sidechain_path": "/sidechains/cmd-1.jsonl",
                "error": None,
            }

        def quick(self, **kw):
            raise RuntimeError("not used in this test")

        def close(self) -> None:
            pass

    class FakeSwarmClient:
        def __init__(self):
            self.calls = []

        def dispatch(self, role, task, *, context_keys=None,
                     max_iterations=40, timeout_s=None):
            self.calls.append({"role": role, "task": task,
                               "context_keys": list(context_keys or [])})
            if role == "research":
                return {
                    "success": True, "summary": "background written",
                    "deliverables": ["/tmp/research.md"],
                    "context_keys_written": ["bg"],
                    "sidechain_path": "/sidechains/sw-r.jsonl",
                    "error": None,
                }
            if role == "math":
                return {
                    "success": True, "summary": "EOM derived",
                    "deliverables": ["/tmp/eom.md"],
                    "context_keys_written": ["eom"],
                    "sidechain_path": "/sidechains/sw-m.jsonl",
                    "error": None,
                }
            return {
                "success": False, "summary": None, "deliverables": [],
                "context_keys_written": [], "sidechain_path": None,
                "error": f"unexpected role {role}",
            }

        def close(self) -> None:
            pass

    from jarvis.core.arbiter import RoleArbiter

    cmd = FakeCMDClient()
    swarm = FakeSwarmClient()
    arbiter = RoleArbiter()
    app = create_app(
        paths=workspace, cfg=cfg, indexer=indexer,
        ollama=ollama, embedder=None,
        cmd_client=cmd, swarm_client=swarm, arbiter=arbiter,
    )

    with _make_client(app) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "rocket-host",
        }).json()
        conv_id = sess["conv_id"]
        resp = cli.post("/api/chat", json={
            "conv_id": conv_id,
            "text": (
                "Build me a single-stage rocket simulator: math model, "
                "Python implementation, README, and a brief background "
                "research note."
            ),
            "channel_kind": "cli", "channel_id": "rocket-host",
        })
    assert resp.status_code == 200

    events = _read_ndjson(resp.text)
    # Final delta must mention all 4 absolute paths verbatim (assertion C).
    delta_text = next(e for e in events if e["type"] == "delta")["text"]
    for p in ("/tmp/research.md", "/tmp/eom.md", "/tmp/sim.py", "/tmp/README.md"):
        assert p in delta_text, f"final delta missing {p}"

    # JSONL transcript: assertion A.
    transcript = workspace.conversations_dir / f"{conv_id}.jsonl"
    raw_events = [json.loads(ln) for ln in transcript.read_text().splitlines()]
    snapshots = [e for e in raw_events if e["kind"] == "delegation_snapshot"]
    envelopes = [e for e in raw_events if e["kind"] == "delegation_envelope"]
    assert len(snapshots) == 3, f"expected 3 delegation_snapshot, got {len(snapshots)}"
    assert len(envelopes) == 3, f"expected 3 delegation_envelope, got {len(envelopes)}"

    # ZERO ReAct substrings in delegation payloads (parsed JSON, not raw grep).
    for evt in snapshots + envelopes:
        blob = json.dumps(evt["payload"])
        for forbidden in ("react_log", "sub_thought", "tool_internal"):
            assert forbidden not in blob, (
                f"{evt['kind']} payload leaked {forbidden!r}"
            )


def test_chat_no_tool_call_yields_minimal_stream(workspace, cfg, indexer):
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="hi there", tool_calls=[],
                       done_reason="stop"),
    ])
    app = create_app(paths=workspace, cfg=cfg, indexer=indexer,
                     ollama=ollama, embedder=None)

    with _make_client(app) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "host-x",
        }).json()
        resp = cli.post("/api/chat", json={
            "conv_id": sess["conv_id"], "text": "say hi",
            "channel_kind": "cli", "channel_id": "host-x",
        })

    events = _read_ndjson(resp.text)
    types = [e["type"] for e in events]
    assert types == ["system_prompt", "delta", "done"]
    assert events[1]["text"] == "hi there"
