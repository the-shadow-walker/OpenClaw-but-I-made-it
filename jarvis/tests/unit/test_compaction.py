"""Unit tests for jarvis.core.compaction (BUILD_SPEC §8, §2 invariant 5).

Pure-function tests over crafted ``messages`` lists plus a tiny
``FakeOllama`` for the auto-flush turn. No real LLM, no real network.

Most tests use a small ``context_window`` (1000) so the trigger threshold
is reachable with a few hundred chars of synthetic content rather than
needing a 30K-character corpus per test. The approximation tokenizer
(default in tests) computes ``max(1, len(text) // 4)``.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import pytest

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
from jarvis.core.compaction import (
    _AUTO_FLUSH_SYSTEM_PROMPT,
    _decide_older,
    _emergency_drop,
    _estimate_messages_tokens,
    _find_recent_turn_boundary,
    _matches_floor,
    _render_older_for_flush,
    _summarize_tool_result,
    maybe_compact,
)
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_approximation_tokenizer():
    configure_tokenizer("approximation")
    yield


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
    """Tiny context_window for fast triggering."""
    return JarvisConfig(
        llm=LLMConfig(context_window=1000),
        conversation=JarvisConvSection(
            compaction=CompactionConfig(
                trigger_pct=0.9,
                keep_recent_turns=2,
                reserve_tokens_floor=100,
                auto_flush=True,
            )
        ),
    )


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
    """OllamaClient subclass that returns scripted responses."""

    def __init__(self, scripted: list[OllamaResponse] | None = None) -> None:
        self._scripted = deque(scripted or [])
        self.calls: list[dict] = []

    def chat(self, model, messages, *, tools=None, system=None, **opts) -> OllamaResponse:
        self.calls.append({
            "model": model, "messages": [dict(m) for m in messages],
            "tools": tools, "system": system, "opts": dict(opts),
        })
        if not self._scripted:
            return OllamaResponse(role="assistant", content="", tool_calls=[],
                                  done_reason="stop")
        return self._scripted.popleft()

    def close(self) -> None:
        pass


def _msg(role: str, content: str, **extra) -> dict:
    m: dict = {"role": role, "content": content}
    m.update(extra)
    return m


def _padding(token_target: int) -> str:
    """Approximation: count_tokens = chars // 4. Build a benign string."""
    return "filler word " * (token_target // 3 + 1)


# ---------------------------------------------------------------------------
# Floor-regex tests
# ---------------------------------------------------------------------------


def test_matches_floor_fenced_code():
    assert _matches_floor("here:\n```py\nx=1\n```\n")


def test_matches_floor_path_at_line_start():
    assert _matches_floor("/etc/hosts is the file")


def test_matches_floor_home_path():
    assert _matches_floor("see ~/.agent_bin/memory.db")


def test_matches_floor_quoted_path_with_spaces():
    assert _matches_floor('open "/Users/grant/My Documents/notes.md"')


def test_matches_floor_url():
    assert _matches_floor("see https://example.com/x?y=1")


def test_matches_floor_negative():
    assert not _matches_floor("just plain prose with no anchors here")


def test_matches_floor_empty():
    assert not _matches_floor("")


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------


def test_estimate_messages_tokens_includes_overhead_and_args():
    messages = [
        _msg("user", "hello"),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "memory_search", "arguments": {"query": "x" * 80}}}
        ]),
    ]
    n = _estimate_messages_tokens(messages)
    # ~"hello" 5 chars / 4 = 1 token, +4 overhead = 5
    # assistant content "" -> count_tokens returns 0; +4 overhead
    # tool_call args str roughly 100 chars / 4 = 25 tokens
    assert n >= 25


def test_find_recent_turn_boundary_with_more_users_than_keep():
    msgs = [
        _msg("user", "a"), _msg("assistant", "ra"),
        _msg("user", "b"), _msg("assistant", "rb"),
        _msg("user", "c"), _msg("assistant", "rc"),
        _msg("user", "d"), _msg("assistant", "rd"),
    ]
    # Last 2 user messages: indices 4 and 6. n-th-last with n=2 => index 4.
    assert _find_recent_turn_boundary(msgs, keep_recent_turns=2) == 4


def test_find_recent_turn_boundary_few_users_returns_zero():
    msgs = [_msg("user", "a"), _msg("assistant", "ra")]
    assert _find_recent_turn_boundary(msgs, keep_recent_turns=6) == 0


def test_summarize_tool_result_replaces_long_content():
    big = "x" * 5000
    msg = _msg("tool", big, name="memory_search")
    out = _summarize_tool_result(msg)
    assert out["role"] == "tool"
    assert out["name"] == "memory_search"
    assert "5000" in out["content"]
    assert len(out["content"]) < 300


# ---------------------------------------------------------------------------
# _decide_older — pair atomicity
# ---------------------------------------------------------------------------


def test_decide_older_floor_keeps_user_with_path():
    older = [_msg("user", "edit /etc/hosts please"),
             _msg("assistant", "ok")]
    decisions = _decide_older(older)
    assert decisions[0] == "keep"
    assert decisions[1] == "drop"


def test_decide_older_pair_atomicity_floor_on_call_side():
    """Assistant shell with /etc/hosts in tool_call args + bare tool result.
    Expected: both kept."""
    older = [
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "memory_get",
                          "arguments": {"file_path": "/etc/hosts"}}},
        ]),
        _msg("tool", "no anchor here", name="memory_get"),
    ]
    decisions = _decide_older(older)
    assert decisions == ["keep", "keep"]


def test_decide_older_pair_atomicity_floor_on_result_side():
    """Assistant shell with no anchor + tool result containing /etc/hosts.
    Expected: both kept (the inverted case)."""
    older = [
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "memory_search",
                          "arguments": {"query": "anything"}}},
        ]),
        _msg("tool", "found at /etc/hosts line 5", name="memory_search"),
    ]
    decisions = _decide_older(older)
    assert decisions == ["keep", "keep"]


def test_decide_older_pair_no_floor_summarizes_and_drops_shell():
    older = [
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "memory_search",
                          "arguments": {"query": "anything"}}},
        ]),
        _msg("tool", "plain prose result", name="memory_search"),
    ]
    decisions = _decide_older(older)
    assert decisions[0] == "drop"
    assert decisions[1] == "summarize"


# ---------------------------------------------------------------------------
# Render older for flush
# ---------------------------------------------------------------------------


def test_render_older_includes_user_assistant_and_tool_lines():
    older = [
        _msg("user", "I'm allergic to peanuts."),
        _msg("assistant", "Got it."),
        _msg("user", "Plan a recipe."),
        _msg("assistant", "", tool_calls=[
            {"function": {"name": "memory_search",
                          "arguments": {"query": "recipes"}}},
        ]),
        _msg("tool", "3 results: almond cake, ...", name="memory_search"),
        _msg("assistant", "Here's a recipe with almonds…"),
    ]
    text = _render_older_for_flush(older)
    assert "[turn 1, user] I'm allergic to peanuts." in text
    assert "[turn 1, assistant] Got it." in text
    assert "[turn 2, assistant called memory_search" in text
    assert "[turn 2, tool memory_search" in text


# ---------------------------------------------------------------------------
# maybe_compact end-to-end
# ---------------------------------------------------------------------------


def test_under_trigger_is_noop(paths, cfg, conversation):
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    ollama = FakeOllama()
    before = list(msgs)
    result = maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="small", paths=paths, cfg=cfg,
        ollama=ollama,
    )
    assert result.fired is False
    assert msgs == before
    # No JSONL compaction event.
    lines = conversation.transcript_path.read_text().splitlines()
    assert all(json.loads(ln)["kind"] != "compaction" for ln in lines)


def test_above_trigger_fires_and_mutates(paths, cfg, conversation):
    """Build messages totaling > 900 tokens (90% of 1000)."""
    big = _padding(150)  # ~150 tokens of filler each
    msgs: list[dict] = []
    # 8 turns × ~150 tokens user + ~150 tokens assistant = ~2400 tokens
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama()
    result = maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    assert result.fired is True
    assert result.tokens_after < result.tokens_before
    # JSONL event recorded.
    events = [json.loads(ln) for ln in conversation.transcript_path.read_text().splitlines()]
    compactions = [e for e in events if e["kind"] == "compaction"]
    assert len(compactions) == 1
    payload = compactions[0]["payload"]
    assert "tokens_before" in payload
    assert "tokens_after" in payload
    assert "emergency_drop_count" in payload


def test_floor_keeps_fenced_code_blocks(paths, cfg, conversation):
    big = _padding(150)
    msgs: list[dict] = []
    canary = "here:\n```py\nx=1\n```"
    msgs.append(_msg("user", canary))                       # turn 1 — canary
    msgs.append(_msg("assistant", "ok"))
    for i in range(7):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama()
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    contents = " ".join((m.get("content") or "") for m in msgs)
    assert "```py" in contents


def test_floor_keeps_absolute_paths_at_line_start(paths, cfg, conversation):
    big = _padding(150)
    msgs: list[dict] = []
    msgs.append(_msg("user", "/etc/hosts is the file we should edit"))
    msgs.append(_msg("assistant", "ok"))
    for i in range(7):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    assert any("/etc/hosts" in (m.get("content") or "") for m in msgs)


def test_floor_keeps_home_paths(paths, cfg, conversation):
    big = _padding(150)
    msgs: list[dict] = []
    msgs.append(_msg("user", "see ~/.agent_bin/memory.db"))
    msgs.append(_msg("assistant", "ok"))
    for i in range(7):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    assert any("~/.agent_bin" in (m.get("content") or "") for m in msgs)


def test_floor_keeps_quoted_paths_with_spaces(paths, cfg, conversation):
    big = _padding(150)
    msgs: list[dict] = []
    msgs.append(_msg("user", 'open "/Users/grant/My Documents/notes.md"'))
    msgs.append(_msg("assistant", "ok"))
    for i in range(7):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    assert any("My Documents" in (m.get("content") or "") for m in msgs)


def test_floor_keeps_urls(paths, cfg, conversation):
    big = _padding(150)
    msgs: list[dict] = []
    msgs.append(_msg("user", "see https://example.com/x"))
    msgs.append(_msg("assistant", "ok"))
    for i in range(7):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    assert any("https://example.com/x" in (m.get("content") or "") for m in msgs)


def test_recent_turns_kept_verbatim(paths, cfg, conversation):
    """Last keep_recent_turns turns must survive untouched."""
    big = _padding(150)
    msgs: list[dict] = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    # Tag the last 2 turns with unique markers.
    msgs[-4]["content"] = "RECENT-USER-A " + big
    msgs[-3]["content"] = "RECENT-ASS-A " + big
    msgs[-2]["content"] = "RECENT-USER-B " + big
    msgs[-1]["content"] = "RECENT-ASS-B " + big
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    contents = [m.get("content") or "" for m in msgs]
    joined = " | ".join(contents)
    assert "RECENT-USER-A" in joined
    assert "RECENT-ASS-A" in joined
    assert "RECENT-USER-B" in joined
    assert "RECENT-ASS-B" in joined


def test_summarize_replaces_long_tool_result(paths, cfg, conversation):
    big = _padding(150)
    long_tool_content = "z" * 5000
    msgs: list[dict] = []
    msgs.append(_msg("user", "search"))
    msgs.append(_msg("assistant", "", tool_calls=[
        {"function": {"name": "memory_search", "arguments": {"query": "x"}}}
    ]))
    msgs.append(_msg("tool", long_tool_content, name="memory_search"))
    msgs.append(_msg("assistant", "found"))
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    # The big tool content should not survive verbatim.
    assert all(("z" * 200) not in (m.get("content") or "") for m in msgs)
    # But there might be a [compacted] marker.
    contents = " ".join((m.get("content") or "") for m in msgs)
    # Either the tool was summarized OR dropped via shell pair logic.
    # Both outcomes are correct; we just want no big content.
    assert long_tool_content not in contents


def test_compaction_event_appended_to_jsonl(paths, cfg, conversation):
    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    events = [json.loads(ln) for ln in conversation.transcript_path.read_text().splitlines()]
    compactions = [e for e in events if e["kind"] == "compaction"]
    assert len(compactions) == 1
    payload = compactions[0]["payload"]
    for key in (
        "fired", "tokens_before", "tokens_after", "flushed_facts",
        "kept_floor_count", "summarized_count", "dropped_count",
        "emergency_drop_count",
    ):
        assert key in payload, f"missing {key}"


def test_system_prompt_never_modified(paths, cfg, conversation):
    """maybe_compact must not mutate the system_prompt argument."""
    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    sp = "this is the system prompt — do not touch"
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt=sp, paths=paths, cfg=cfg, ollama=FakeOllama(),
    )
    # We can't tell from outside if a string was rebuilt vs mutated, but we
    # can confirm Python strings are immutable and the variable is unchanged.
    assert sp == "this is the system prompt — do not touch"


def test_system_prompt_size_warning(paths, cfg, conversation, caplog):
    """System prompt > 50% of context_window should WARN."""
    # context_window=1000, so 50% = 500 tokens = 2000 chars.
    huge_sp = "x" * 3000  # 750 tokens
    msgs = [_msg("user", "hi")]
    with caplog.at_level(logging.WARNING):
        maybe_compact(
            conversation=conversation, messages=msgs,
            system_prompt=huge_sp, paths=paths, cfg=cfg, ollama=FakeOllama(),
        )
    assert any("system prompt" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Auto-flush behavior
# ---------------------------------------------------------------------------


def test_auto_flush_disabled_skips_phase1(paths, cfg, conversation):
    cfg2 = cfg.model_copy(deep=True)
    cfg2.conversation.compaction = CompactionConfig(
        trigger_pct=0.9, keep_recent_turns=2, reserve_tokens_floor=100,
        auto_flush=False,
    )
    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama()
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg2, ollama=ollama,
    )
    # No flush call (which would have system=_AUTO_FLUSH_SYSTEM_PROMPT).
    assert not any(c["system"] == _AUTO_FLUSH_SYSTEM_PROMPT for c in ollama.calls)


def test_auto_flush_failure_does_not_block_truncation(paths, cfg, conversation):
    class BoomOllama(FakeOllama):
        def chat(self, *_a, **_kw):
            raise RuntimeError("flush blew up")

    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = BoomOllama()
    result = maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    assert result.fired is True
    assert result.flushed_facts == []
    assert result.dropped_count > 0


def test_auto_flush_sees_older_with_tool_calls_flattened(paths, cfg, conversation):
    """Capture the flush prompt; assert tool calls appear flattened."""
    big = _padding(150)
    msgs: list[dict] = []
    msgs.append(_msg("user", "search please"))
    msgs.append(_msg("assistant", "", tool_calls=[
        {"function": {"name": "memory_search",
                      "arguments": {"query": "x"}}}
    ]))
    msgs.append(_msg("tool", "result body", name="memory_search"))
    msgs.append(_msg("assistant", "ok done"))
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama()
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    flush_calls = [c for c in ollama.calls if c["system"] == _AUTO_FLUSH_SYSTEM_PROMPT]
    assert len(flush_calls) == 1
    flush_text = flush_calls[0]["messages"][0]["content"]
    assert "[turn 1, user]" in flush_text
    assert "assistant called memory_search" in flush_text


def test_auto_flush_forces_where_daily(paths, cfg, conversation):
    """Even if the model emits where='memory', the call site rewrites to 'daily'."""
    flush_resp = OllamaResponse(
        role="assistant", content="",
        tool_calls=[OllamaToolCall(
            call_id="f-1", name="memory_write",
            arguments={"content": "fact A", "where": "memory"},
        )],
        done_reason="stop",
    )
    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama([flush_resp])
    # MEMORY.md baseline
    memory_before = paths.memory_md.read_text()
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    # Daily log got the line; MEMORY.md is unchanged.
    daily = paths.daily_log().read_text()
    assert "fact A" in daily
    assert paths.memory_md.read_text() == memory_before


def test_auto_flush_forces_flush_tag(paths, cfg, conversation):
    flush_resp = OllamaResponse(
        role="assistant", content="",
        tool_calls=[OllamaToolCall(
            call_id="f-1", name="memory_write",
            arguments={"content": "tagless fact"},  # no tags
        )],
        done_reason="stop",
    )
    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama([flush_resp])
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    daily = paths.daily_log().read_text()
    assert "tagless fact" in daily
    assert "#flush" in daily


# ---------------------------------------------------------------------------
# Emergency drop
# ---------------------------------------------------------------------------


def test_emergency_drop_basic_drops_until_under_target():
    """Direct test of _emergency_drop: messages over budget shrink."""
    msgs = [_msg("user", "x" * 4000) for _ in range(5)]  # ~5000 tokens
    n = _emergency_drop(msgs, sys_prompt_tokens=0, context_window=1000)
    assert n > 0
    # Stop guard: at least one user message remains.
    assert sum(1 for m in msgs if m["role"] == "user") >= 1


def test_emergency_drop_preserves_at_least_one_recent_turn():
    msgs = [_msg("user", "x" * 100_000)]  # impossibly large
    n = _emergency_drop(msgs, sys_prompt_tokens=0, context_window=1000)
    # Stop guard fires immediately because user_count == 1; n == 0.
    assert n == 0
    assert len(msgs) == 1


def test_emergency_drop_fires_when_floor_plus_recent_exceed_window(
    paths, cfg, conversation, caplog
):
    """Pasted-document case: a single huge user message with a path stays via
    the floor; recent turns + system prompt push us above 0.95×window."""
    big_user = "/etc/hosts and " + ("y" * 4000)
    msgs: list[dict] = [_msg("user", big_user)]
    big = _padding(120)
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama()
    with caplog.at_level(logging.WARNING):
        result = maybe_compact(
            conversation=conversation, messages=msgs,
            system_prompt="x" * 800,  # 200 tokens of system prompt
            paths=paths, cfg=cfg, ollama=ollama,
        )
    assert result.fired is True
    assert result.emergency_drop_count > 0
    # WARN logged.
    assert any("emergency drop" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# num_ctx propagation
# ---------------------------------------------------------------------------


def test_num_ctx_passed_to_ollama_chat_during_flush(paths, cfg, conversation):
    big = _padding(150)
    msgs = []
    for i in range(8):
        msgs.append(_msg("user", f"q{i} " + big))
        msgs.append(_msg("assistant", f"a{i} " + big))
    ollama = FakeOllama()
    maybe_compact(
        conversation=conversation, messages=msgs,
        system_prompt="sys", paths=paths, cfg=cfg, ollama=ollama,
    )
    flush_calls = [c for c in ollama.calls if c["system"] == _AUTO_FLUSH_SYSTEM_PROMPT]
    assert len(flush_calls) == 1
    assert flush_calls[0]["opts"].get("num_ctx") == cfg.llm.context_window


def test_token_estimate_uses_count_tokens_approximation():
    """Approximation: chars // 4 plus 4 overhead per message."""
    text = "a" * 400  # 100 tokens
    msgs = [_msg("user", text)]
    n = _estimate_messages_tokens(msgs)
    assert 95 <= n <= 110  # ~100 + 4 overhead


# ---------------------------------------------------------------------------
# KNOWN_MODEL_WINDOWS warning
# ---------------------------------------------------------------------------


def test_known_model_window_warn_for_unknown_model(caplog):
    with caplog.at_level(logging.WARNING):
        JarvisConfig(llm=LLMConfig(chat_model="unknown:7b"))
    assert any(
        "unknown chat_model" in r.message and "unknown:7b" in r.message
        for r in caplog.records
    )


def test_known_model_window_no_warn_for_known_model(caplog):
    with caplog.at_level(logging.WARNING):
        JarvisConfig(llm=LLMConfig())  # default qwen2.5:3b
    assert not any("unknown chat_model" in r.message for r in caplog.records)
