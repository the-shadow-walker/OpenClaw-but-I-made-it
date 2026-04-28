"""Orchestrator — DAG topo execution with first-failure-stops."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.core.orchestrator import execute
from jarvis.core.planner import Plan, PlanNode
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(
            workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin"
        )
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


# ---------------------------------------------------------------------------
# Fake invoker dispatch
# ---------------------------------------------------------------------------


def _ok(summary="ok", deliverables=None, sidechain="/sc/x.jsonl"):
    return {
        "success": True, "summary": summary,
        "deliverables": list(deliverables or []),
        "context_keys_written": [], "sidechain_path": sidechain,
        "error": None,
    }


def _fail(error="boom"):
    return {
        "success": False, "summary": None, "deliverables": [],
        "context_keys_written": [], "sidechain_path": None,
        "error": error,
    }


class _Recorder:
    """Records every invoker_dispatch call with start/end timestamps."""

    def __init__(self, behavior: dict[str, dict] | None = None) -> None:
        # behavior maps task -> envelope OR callable(task, kwargs) -> envelope
        self.behavior = behavior or {}
        self.calls: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, **kwargs):
        started = time.monotonic()
        task = kwargs.get("task")
        target = kwargs.get("target")
        with self.lock:
            self.calls.append({
                "task": task, "target": target,
                "context_keys": list(kwargs.get("context_keys") or []),
                "started": started,
            })
        beh = self.behavior.get(task)
        if callable(beh):
            env = beh(task, kwargs)
        elif isinstance(beh, dict):
            env = beh
        else:
            env = _ok(summary=f"did {task}")
        ended = time.monotonic()
        with self.lock:
            self.calls[-1]["ended"] = ended
        return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _node(nid, task, deps=(), pubs=(), cons=(), target="cmd:react"):
    return PlanNode(
        id=nid, target=target, task=task,
        depends_on=tuple(deps), publish_keys=tuple(pubs),
        consume_keys=tuple(cons),
    )


def test_topo_executes_dag_in_dependency_order(monkeypatch, conversation, paths, shared_board):
    p = Plan(nodes=(
        _node("A", "task-a"),
        _node("B", "task-b"),
        _node("C", "task-c", deps=("A", "B")),
    ))
    rec = _Recorder()
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    result = execute(
        p, conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=None, swarm_client=None, arbiter=RoleArbiter(),
    )
    assert result.failed_node_id is None
    # C must have started after both A and B finished.
    starts = {c["task"]: c["started"] for c in rec.calls}
    ends = {c["task"]: c["ended"] for c in rec.calls}
    assert starts["task-c"] >= ends["task-a"]
    assert starts["task-c"] >= ends["task-b"]


def test_independent_leaves_run_in_parallel(monkeypatch, conversation, paths, shared_board):
    """Property test: A and B share a Barrier(2). If serialized, the second
    deadlocks the barrier (5s timeout → call returns failure). If parallel,
    both pass through and return success."""
    barrier = threading.Barrier(2, timeout=5.0)

    def parallel_behavior(task, kwargs):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return _fail(error=f"{task}: barrier deadlocked (serialized)")
        return _ok(summary=f"did {task}")

    p = Plan(nodes=(
        _node("A", "task-a"),
        _node("B", "task-b"),
    ))
    rec = _Recorder(behavior={"task-a": parallel_behavior, "task-b": parallel_behavior})
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    result = execute(
        p, conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=None, swarm_client=None, arbiter=RoleArbiter(),
    )
    assert result.failed_node_id is None, (
        f"barrier deadlocked → orchestrator runs siblings serially. "
        f"summary={result.summary}"
    )


def test_first_failure_stops_dag(monkeypatch, conversation, paths, shared_board):
    p = Plan(nodes=(
        _node("A", "task-a"),
        _node("B", "task-b"),
        _node("C", "task-c", deps=("A",)),
        _node("D", "task-d", deps=("B",)),
    ))
    rec = _Recorder(behavior={"task-b": _fail(error="B failed")})
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    result = execute(
        p, conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=None, swarm_client=None, arbiter=RoleArbiter(),
    )
    assert result.failed_node_id == "B"
    tasks_run = {c["task"] for c in rec.calls}
    # First level (A, B) ran; second level (C, D) skipped after failure.
    assert "task-a" in tasks_run
    assert "task-b" in tasks_run
    assert "task-c" not in tasks_run
    assert "task-d" not in tasks_run


def test_failure_waits_for_in_flight_siblings(monkeypatch, conversation, paths, shared_board):
    """A fails fast; B and C are still running. Orchestrator must NOT
    return until B and C complete."""
    hold_b = threading.Event()
    hold_c = threading.Event()
    b_finished_at: list[float] = []
    c_finished_at: list[float] = []

    def b_behavior(task, kwargs):
        hold_b.wait(timeout=3.0)
        b_finished_at.append(time.monotonic())
        return _ok(summary="B ok")

    def c_behavior(task, kwargs):
        hold_c.wait(timeout=3.0)
        c_finished_at.append(time.monotonic())
        return _ok(summary="C ok")

    p = Plan(nodes=(
        _node("A", "task-a"),
        _node("B", "task-b"),
        _node("C", "task-c"),
        _node("D", "task-d", deps=("A",)),  # deeper level
    ))
    rec = _Recorder(behavior={
        "task-a": _fail(error="A failed"),
        "task-b": b_behavior, "task-c": c_behavior,
    })
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    # Run execute on a thread; release B and C after a delay.
    result_holder: list = []

    def runner():
        result_holder.append(execute(
            p, conversation=conversation, paths=paths,
            shared_board=shared_board, cmd_client=None,
            swarm_client=None, arbiter=RoleArbiter(),
        ))

    t = threading.Thread(target=runner)
    t.start()
    # Give A time to fail.
    time.sleep(0.05)
    return_at_release = time.monotonic()
    hold_b.set()
    hold_c.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    result = result_holder[0]
    assert result.failed_node_id == "A"
    # B and C completed.
    assert b_finished_at and c_finished_at
    assert b_finished_at[0] >= return_at_release
    assert c_finished_at[0] >= return_at_release
    # D (deeper level) was never dispatched.
    tasks_run = {c["task"] for c in rec.calls}
    assert "task-d" not in tasks_run


def test_consume_keys_transitive_closure_then_self_first_seen(
    monkeypatch, conversation, paths, shared_board,
):
    """DAG: A→D, B→D, C→A. D.consume_keys=[x]; ancestor publishes:
    A=[a1], B=[b1], C=[c1]. Expected merged for D = transitive ancestors
    (C, A, B) publishes + self consume_keys."""
    p = Plan(nodes=(
        _node("C", "c", pubs=("c1",)),
        _node("A", "a", deps=("C",), pubs=("a1",)),
        _node("B", "b", pubs=("b1",)),
        _node("D", "d", deps=("A", "B"), cons=("x",)),
    ))
    rec = _Recorder()
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    execute(
        p, conversation=conversation, paths=paths, shared_board=shared_board,
        cmd_client=None, swarm_client=None, arbiter=RoleArbiter(),
    )
    d_call = next(c for c in rec.calls if c["task"] == "d")
    keys = d_call["context_keys"]
    # All ancestor publishes present + self consume_keys; "x" comes last.
    assert "a1" in keys
    assert "b1" in keys
    assert "c1" in keys
    assert keys[-1] == "x"
    # Dedup: no key appears twice.
    assert len(keys) == len(set(keys))


def test_missing_consume_key_warns_does_not_fail(
    monkeypatch, conversation, paths, shared_board, caplog,
):
    p = Plan(nodes=(
        _node("A", "a", pubs=("real_key",)),
        _node("B", "b", deps=("A",), cons=("missing_key",)),
    ))
    rec = _Recorder()
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    import logging
    with caplog.at_level(logging.WARNING):
        result = execute(
            p, conversation=conversation, paths=paths,
            shared_board=shared_board, cmd_client=None,
            swarm_client=None, arbiter=RoleArbiter(),
        )
    assert result.failed_node_id is None
    assert any("missing_key" in r.message for r in caplog.records)


def test_deliverable_paths_dedup_and_absolute(
    monkeypatch, conversation, paths, shared_board, caplog,
):
    p = Plan(nodes=(
        _node("A", "a"), _node("B", "b"),
    ))
    rec = _Recorder(behavior={
        "a": _ok(deliverables=["/tmp/shared.md"]),
        "b": _ok(deliverables=["/tmp/shared.md", "/tmp/b.md"]),
    })
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    import logging
    with caplog.at_level(logging.WARNING):
        result = execute(
            p, conversation=conversation, paths=paths,
            shared_board=shared_board, cmd_client=None,
            swarm_client=None, arbiter=RoleArbiter(),
        )
    assert result.deliverable_paths == ["/tmp/shared.md", "/tmp/b.md"]
    assert any("shared.md" in r.message and "multiple" in r.message
               for r in caplog.records)


def test_summary_does_not_contain_react_substrings(
    monkeypatch, conversation, paths, shared_board,
):
    """§16-A textual purity guard: orchestrator builds prose from
    envelope.summary only. If a node leaks ReAct hints into its summary,
    the orchestrator passes them through (we test the orchestrator's own
    output: it should not synthesize them itself)."""
    p = Plan(nodes=(_node("A", "a"), _node("B", "b"),))
    rec = _Recorder()
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)

    result = execute(
        p, conversation=conversation, paths=paths,
        shared_board=shared_board, cmd_client=None,
        swarm_client=None, arbiter=RoleArbiter(),
    )
    for forbidden in ("react_log", "sub_thought", "tool_internal"):
        assert forbidden not in result.summary


def test_three_node_rocket_dag_emits_three_envelope_events_in_jsonl(
    monkeypatch, conversation, paths, shared_board,
):
    """3 nodes → 3 delegation_envelope events (the orchestrator dispatches
    via invoker, which appends the event)."""
    import json

    p = Plan(nodes=(
        _node("M", "math task", target="swarm:math"),
        _node("E", "engineer task", deps=("M",), target="swarm:engineer"),
        _node("R", "research task", target="swarm:research"),
    ))

    def fake_dispatch(**kwargs):
        # Mimic invoker.dispatch's side-effect: append delegation_envelope.
        env = _ok(summary=f"did {kwargs.get('task')}")
        conversation.append("delegation_envelope", {
            "target": kwargs.get("target"),
            "task": kwargs.get("task"),
            "envelope": env,
        })
        return env

    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", fake_dispatch)
    execute(
        p, conversation=conversation, paths=paths,
        shared_board=shared_board, cmd_client=None,
        swarm_client=None, arbiter=RoleArbiter(),
    )
    transcript = conversation.transcript_path.read_text(encoding="utf-8")
    events = [json.loads(ln) for ln in transcript.splitlines() if ln.strip()]
    delegations = [e for e in events if e["kind"] == "delegation_envelope"]
    assert len(delegations) == 3


def test_empty_results_yields_placeholder_summary(
    monkeypatch, conversation, paths, shared_board,
):
    """Edge case: an empty Plan. topo_order returns []; nothing runs."""
    p = Plan(nodes=())
    # _build_result handles empty results gracefully.
    rec = _Recorder()
    monkeypatch.setattr("jarvis.core.orchestrator.invoker_dispatch", rec)
    result = execute(
        p, conversation=conversation, paths=paths,
        shared_board=shared_board, cmd_client=None,
        swarm_client=None, arbiter=RoleArbiter(),
    )
    assert result.failed_node_id is None
    assert "no nodes ran" in result.summary
    assert result.deliverable_paths == []
