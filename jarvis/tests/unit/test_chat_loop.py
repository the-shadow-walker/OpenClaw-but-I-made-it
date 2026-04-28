"""run_turn() exercised against a fake OllamaClient — no network required."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import pytest

from jarvis.clients.ollama import OllamaClient, OllamaResponse, OllamaToolCall
from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.chat import ChatTurnConfig, run_turn
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.core.tools import ToolRegistry, ToolSpec
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
def cfg() -> JarvisConfig:
    return JarvisConfig()


@pytest.fixture
def conn(paths: WorkspacePaths):
    c = get_connection(paths.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def conversation(paths, conn):
    convo = Conversation.open(
        channel_kind="dm", channel_id="g", paths=paths, conn=conn,
        cfg=ConversationConfig(),
    )
    yield convo
    convo.__exit__(None, None, None)


class FakeOllama(OllamaClient):
    """OllamaClient subclass that returns scripted responses and counts calls."""

    def __init__(self, scripted: list[OllamaResponse]) -> None:
        # Don't open a real httpx client — we override chat()/complete() entirely.
        self._scripted = deque(scripted)
        self._chat_calls: list[dict] = []

    def chat(self, model, messages, *, tools=None, system=None, **opts) -> OllamaResponse:
        self._chat_calls.append({
            "model": model,
            "messages": [dict(m) for m in messages],
            "tools": tools,
            "system": system,
        })
        if not self._scripted:
            raise RuntimeError("FakeOllama: ran out of scripted responses")
        return self._scripted.popleft()

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_tool_turn_emits_system_delta_done(paths, cfg, conversation):
    ollama = FakeOllama([OllamaResponse(role="assistant", content="hello", tool_calls=[],
                                        done_reason="stop")])
    registry = ToolRegistry()  # empty — no tools
    events = list(run_turn(
        user_text="hi",
        conversation=conversation,
        paths=paths,
        cfg=cfg,
        ollama=ollama,
        registry=registry,
        channel_kind="dm",
    ))
    types = [e["type"] for e in events]
    assert types == ["system_prompt", "delta", "done"]
    assert events[1]["text"] == "hello"
    assert events[2]["stop_reason"] == "stop"
    assert len(ollama._chat_calls) == 1

    # Transcript: system_prompt, user_message, assistant_message.
    lines = [json.loads(ln) for ln in conversation.transcript_path.read_text().splitlines()]
    kinds = [ln["kind"] for ln in lines]
    assert kinds == ["system_prompt", "user_message", "assistant_message"]


def test_tool_then_answer(paths, cfg, conversation):
    """First model call requests a tool; second call returns plain text."""
    tc = OllamaToolCall(call_id="tc-1", name="echo", arguments={"text": "hi"})
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[tc], done_reason=None),
        OllamaResponse(role="assistant", content="echoed: hi", tool_calls=[], done_reason="stop"),
    ])
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="echo", description="echo back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda text: {"echoed": text},
    ))
    events = list(run_turn(
        user_text="please echo",
        conversation=conversation,
        paths=paths,
        cfg=cfg,
        ollama=ollama,
        registry=registry,
        channel_kind="dm",
    ))
    types = [e["type"] for e in events]
    assert types == ["system_prompt", "tool_call", "tool_result", "delta", "done"]
    assert events[1]["name"] == "echo"
    assert events[2]["result"] == {"echoed": "hi"}
    assert events[3]["text"] == "echoed: hi"
    assert len(ollama._chat_calls) == 2

    # Second call's messages must include the tool result.
    second = ollama._chat_calls[1]
    roles = [m["role"] for m in second["messages"]]
    assert "tool" in roles

    # Transcript has both tool_call + tool_result events.
    lines = [json.loads(ln) for ln in conversation.transcript_path.read_text().splitlines()]
    kinds = [ln["kind"] for ln in lines]
    assert kinds == [
        "system_prompt", "user_message", "assistant_message",
        "tool_call", "tool_result", "assistant_message",
    ]


def test_tool_loop_hits_iteration_cap(paths, cfg, conversation):
    """Model keeps requesting tools forever; loop stops at max_tool_iterations."""
    looping_tc = OllamaToolCall(call_id="tc-x", name="echo", arguments={"text": "x"})
    # 12 responses, all of which request another tool — never terminates.
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[looping_tc],
                       done_reason=None)
        for _ in range(12)
    ])
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="echo", description="",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda text: {"echoed": text},
    ))
    events = list(run_turn(
        user_text="loop forever",
        conversation=conversation,
        paths=paths,
        cfg=cfg,
        ollama=ollama,
        registry=registry,
        channel_kind="dm",
        turn_cfg=ChatTurnConfig(max_tool_iterations=3),
    ))
    # Should stop at 3 iterations → 3 tool_call/tool_result pairs, no answer.
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 3
    done = events[-1]
    assert done["type"] == "done"
    assert done["stop_reason"] == "tool_limit"
    assert len(ollama._chat_calls) == 3


def test_tool_handler_error_wraps_into_result(paths, cfg, conversation):
    """When a tool raises, the loop emits a tool_result with error set
    rather than crashing, and the model can still produce a final answer.
    """
    tc = OllamaToolCall(call_id="tc-1", name="boom", arguments={})
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[tc], done_reason=None),
        OllamaResponse(role="assistant", content="sorry, that broke", tool_calls=[],
                       done_reason="stop"),
    ])
    registry = ToolRegistry()

    def boom_handler():
        raise RuntimeError("kaboom")

    registry.register(ToolSpec(
        name="boom", description="", parameters={"type": "object"},
        handler=boom_handler,
    ))

    events = list(run_turn(
        user_text="trigger it",
        conversation=conversation,
        paths=paths,
        cfg=cfg,
        ollama=ollama,
        registry=registry,
        channel_kind="dm",
    ))
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["error"] == "RuntimeError: kaboom"
    assert tool_results[0]["result"] is None
    assert events[-2]["text"] == "sorry, that broke"


def test_unknown_tool_emits_error_result(paths, cfg, conversation):
    tc = OllamaToolCall(call_id="tc-1", name="ghost_tool", arguments={"x": 1})
    ollama = FakeOllama([
        OllamaResponse(role="assistant", content="", tool_calls=[tc], done_reason=None),
        OllamaResponse(role="assistant", content="ok", tool_calls=[], done_reason="stop"),
    ])
    registry = ToolRegistry()
    events = list(run_turn(
        user_text="x",
        conversation=conversation,
        paths=paths,
        cfg=cfg,
        ollama=ollama,
        registry=registry,
        channel_kind="dm",
    ))
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["error"].startswith("unknown tool")


def test_ollama_failure_yields_error_event(paths, cfg, conversation):
    """If ollama.chat() raises, the loop yields {type:error} + {type:done}."""

    class BoomOllama(FakeOllama):
        def chat(self, *_a, **_kw):
            raise RuntimeError("connection refused")

    ollama = BoomOllama([])
    registry = ToolRegistry()
    events = list(run_turn(
        user_text="x",
        conversation=conversation,
        paths=paths,
        cfg=cfg,
        ollama=ollama,
        registry=registry,
        channel_kind="dm",
    ))
    types = [e["type"] for e in events]
    assert "error" in types
    assert events[-1]["type"] == "done"
    assert events[-1]["stop_reason"] == "error"


# ---------------------------------------------------------------------------
# OllamaClient.chat parsing helpers
# ---------------------------------------------------------------------------


def test_parse_response_synthesizes_call_id():
    raw = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "memory_search", "arguments": {"query": "x"}}}],
        },
        "done_reason": None,
    }
    parsed = OllamaClient._parse_response(raw, tools_present=True)
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.name == "memory_search"
    assert tc.arguments == {"query": "x"}
    assert tc.call_id.startswith("tc-")


def test_parse_response_drops_hallucinated_tool_calls_when_no_tools():
    raw = {
        "message": {
            "role": "assistant",
            "content": "anyway",
            "tool_calls": [{"function": {"name": "ghost", "arguments": {}}}],
        }
    }
    parsed = OllamaClient._parse_response(raw, tools_present=False)
    assert parsed.tool_calls == []
    assert parsed.content == "anyway"


def test_parse_response_handles_string_arguments():
    raw = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "x", "arguments": '{"a": 1}'}}],
        }
    }
    parsed = OllamaClient._parse_response(raw, tools_present=True)
    assert parsed.tool_calls[0].arguments == {"a": 1}
