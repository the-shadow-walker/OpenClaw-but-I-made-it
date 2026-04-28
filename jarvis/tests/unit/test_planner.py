"""Planner — one LLM call → validated DAG."""

from __future__ import annotations

import pytest

from jarvis.clients.ollama import OllamaResponse, OllamaToolCall
from jarvis.core.planner import Plan, PlanError, plan


class FakeOllama:
    """Scriptable OllamaClient stand-in. Records the call args."""

    def __init__(self, response: OllamaResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def chat(self, model, messages, *, tools=None, system=None, num_ctx=None, **opts):
        self.calls.append({
            "model": model, "messages": messages, "tools": tools,
            "system": system, "num_ctx": num_ctx, "opts": opts,
        })
        return self._response


def _resp(tool_calls: list[OllamaToolCall]) -> OllamaResponse:
    return OllamaResponse(role="assistant", content="", tool_calls=tool_calls)


def _submit_plan_call(args: dict) -> OllamaToolCall:
    return OllamaToolCall(call_id="c1", name="submit_plan", arguments=args)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_plan_returns_dataclass_for_well_formed_dag():
    args = {
        "rationale": "decompose into math then engineering",
        "nodes": [
            {"id": "W1", "target": "swarm:math",
             "task": "derive equations of motion",
             "depends_on": [], "publish_keys": ["eom"], "consume_keys": []},
            {"id": "W2", "target": "swarm:engineer",
             "task": "implement RK4 in Python",
             "depends_on": ["W1"], "publish_keys": ["impl"],
             "consume_keys": ["eom"]},
        ],
    }
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    p = plan("rocket sim", ollama=ollama, model="qwen2.5:3b", num_ctx=8192)
    assert isinstance(p, Plan)
    assert len(p.nodes) == 2
    assert p.nodes[0].id == "W1"
    assert p.nodes[0].target == "swarm:math"
    assert p.nodes[1].depends_on == ("W1",)
    assert p.nodes[1].consume_keys == ("eom",)
    assert "decompose" in p.rationale
    assert ollama.calls[0]["model"] == "qwen2.5:3b"
    assert ollama.calls[0]["num_ctx"] == 8192


def test_topo_order_layers_independent_nodes():
    args = {
        "nodes": [
            {"id": "A", "target": "swarm:math", "task": "a", "depends_on": []},
            {"id": "B", "target": "swarm:research", "task": "b", "depends_on": []},
            {"id": "C", "target": "swarm:engineer", "task": "c",
             "depends_on": ["A", "B"]},
        ]
    }
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    p = plan("x", ollama=ollama, model="m", num_ctx=4096)
    layers = p.topo_order()
    assert len(layers) == 2
    layer0_ids = {n.id for n in layers[0]}
    assert layer0_ids == {"A", "B"}
    assert [n.id for n in layers[1]] == ["C"]


def test_plan_arguments_as_json_string_coerced():
    """Some models emit JSON arguments as a string; planner must coerce."""
    import json
    args_obj = {"nodes": [
        {"id": "W1", "target": "cmd:react", "task": "x", "depends_on": []},
    ]}
    ollama = FakeOllama(_resp([
        OllamaToolCall(call_id="c1", name="submit_plan",
                       arguments=json.dumps(args_obj)),  # string, not dict
    ]))
    p = plan("x", ollama=ollama, model="m", num_ctx=4096)
    assert len(p.nodes) == 1


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------


def test_plan_raises_on_empty_text():
    ollama = FakeOllama(_resp([]))
    with pytest.raises(PlanError):
        plan("", ollama=ollama, model="m", num_ctx=4096)
    with pytest.raises(PlanError):
        plan("   ", ollama=ollama, model="m", num_ctx=4096)


def test_plan_raises_on_no_tool_call():
    ollama = FakeOllama(_resp([]))
    with pytest.raises(PlanError) as ei:
        plan("rocket sim", ollama=ollama, model="m", num_ctx=4096)
    assert "submit_plan" in str(ei.value) or "tool" in str(ei.value).lower()


def test_plan_raises_on_wrong_tool_name():
    ollama = FakeOllama(_resp([
        OllamaToolCall(call_id="c1", name="something_else", arguments={}),
    ]))
    with pytest.raises(PlanError):
        plan("x", ollama=ollama, model="m", num_ctx=4096)


def test_plan_raises_on_unknown_target():
    args = {"nodes": [
        {"id": "W1", "target": "swarm:bogus", "task": "x", "depends_on": []},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError) as ei:
        plan("x", ollama=ollama, model="m", num_ctx=4096)
    assert "unknown target" in str(ei.value)


def test_plan_raises_on_missing_id():
    args = {"nodes": [
        {"target": "cmd:react", "task": "x", "depends_on": []},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError):
        plan("x", ollama=ollama, model="m", num_ctx=4096)


def test_plan_raises_on_duplicate_id():
    args = {"nodes": [
        {"id": "W1", "target": "cmd:react", "task": "x", "depends_on": []},
        {"id": "W1", "target": "swarm:math", "task": "y", "depends_on": []},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError) as ei:
        plan("x", ollama=ollama, model="m", num_ctx=4096)
    assert "duplicate" in str(ei.value).lower()


def test_plan_raises_on_dangling_dependency():
    args = {"nodes": [
        {"id": "W1", "target": "cmd:react", "task": "x",
         "depends_on": ["NOPE"]},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError) as ei:
        plan("x", ollama=ollama, model="m", num_ctx=4096)
    assert "unknown id" in str(ei.value).lower() or "depends" in str(ei.value).lower()


def test_plan_raises_on_cycle():
    args = {"nodes": [
        {"id": "A", "target": "cmd:react", "task": "a", "depends_on": ["B"]},
        {"id": "B", "target": "cmd:react", "task": "b", "depends_on": ["A"]},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError) as ei:
        plan("x", ollama=ollama, model="m", num_ctx=4096)
    assert "cycle" in str(ei.value).lower()


def test_plan_rejects_long_task():
    args = {"nodes": [
        {"id": "W1", "target": "cmd:react",
         "task": "x" * 1001, "depends_on": []},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError) as ei:
        plan("x", ollama=ollama, model="m", num_ctx=4096)
    assert "1001" in str(ei.value) or "chars" in str(ei.value).lower()


def test_plan_raises_on_empty_nodes():
    args = {"nodes": []}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError):
        plan("x", ollama=ollama, model="m", num_ctx=4096)


def test_plan_raises_on_empty_task():
    args = {"nodes": [
        {"id": "W1", "target": "cmd:react", "task": "  ", "depends_on": []},
    ]}
    ollama = FakeOllama(_resp([_submit_plan_call(args)]))
    with pytest.raises(PlanError):
        plan("x", ollama=ollama, model="m", num_ctx=4096)


def test_plan_raises_on_llm_call_exception():
    class BadOllama(FakeOllama):
        def chat(self, *a, **kw):
            raise RuntimeError("network down")

    bad = BadOllama(_resp([]))
    with pytest.raises(PlanError) as ei:
        plan("x", ollama=bad, model="m", num_ctx=4096)
    assert "network down" in str(ei.value)
