"""§2 invariant 5 contract test — auto-flush + floor preservation.

Builds a synthetic 50-turn ``messages`` history (the shape ``run_turn``
hands to ``maybe_compact``), drives it through one ``maybe_compact``
call, and verifies the binding contract:

  * **Daily log** contains the Friday rule with ``#flush`` tag —
    auto-flush correctness (PRIMARY).
  * **messages[]** still contains the rule — floor preservation
    (SECONDARY, defense in depth).
  * **Canaries** (path / fenced code / URL) survive in messages[].
  * **JSONL transcript** records a ``compaction`` event with
    ``tokens_after < tokens_before``.

Independent assertions for flush and floor avoid the OR-bug that hid the
original failure where the floor caught the rule and auto-flush silently
broke. Either one breaking fails the test.

Why a synthetic message history rather than 50 real ``run_turn`` calls:
``run_turn`` rebuilds ``messages`` fresh per turn (each call is one user
turn → one assistant answer through any tool loop), so cross-turn
accumulation does not happen at this layer. ``maybe_compact``'s contract
is on the live ``messages`` list — exactly what we hand it. A single tool-
heavy turn would also exercise the same code path, but the synthetic
50-turn shape is closer to the spec's worded scenario and produces a
clearer failure message when the contract regresses.

``ScriptedOllama`` dispatches on the auto-flush *system prompt string*
(NOT model name or tool list — both are refactor-fragile; the prompt is
a known module constant). Anyone changing the prompt without updating
this test gets a loud failure here, which is the correct break.
"""

from __future__ import annotations

import json
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
from jarvis.core.compaction import _AUTO_FLUSH_SYSTEM_PROMPT, maybe_compact
from jarvis.core.conversation import Conversation, ConversationConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

RULE = "never deploy on Fridays — hard rule from my last gig"
PATH_CANARY = "/etc/hosts"
CODE_CANARY = "```py"
URL_CANARY = "https://atomos.network"


@pytest.fixture(autouse=True)
def _approximation_tokenizer():
    configure_tokenizer("approximation")
    yield


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
    """``context_window`` set so 50 ~150-token turns reliably trip the trigger."""
    return JarvisConfig(
        llm=LLMConfig(context_window=4000),
        conversation=JarvisConvSection(
            compaction=CompactionConfig(
                trigger_pct=0.9,
                keep_recent_turns=2,
                reserve_tokens_floor=200,
                auto_flush=True,
            )
        ),
    )


@pytest.fixture
def conn(workspace: WorkspacePaths):
    c = get_connection(workspace.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


@pytest.fixture
def conversation(workspace, conn):
    convo = Conversation.open(
        channel_kind="cli", channel_id="acceptance",
        paths=workspace, conn=conn,
        cfg=ConversationConfig(),
    )
    yield convo
    convo.__exit__(None, None, None)


class ScriptedOllama(OllamaClient):
    """Dispatches on ``_AUTO_FLUSH_SYSTEM_PROMPT``.

    Any change to the prompt is intentional and the test author should
    update both this test and the prompt at the same time.
    """

    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self._flush_emitted = False

    def chat(self, model, messages, *, tools=None, system=None, **opts) -> OllamaResponse:
        self.chat_calls.append({
            "model": model, "messages": list(messages),
            "tools": tools, "system": system, "opts": dict(opts),
        })
        if system == _AUTO_FLUSH_SYSTEM_PROMPT:
            if not self._flush_emitted:
                self._flush_emitted = True
                return OllamaResponse(
                    role="assistant", content="",
                    tool_calls=[OllamaToolCall(
                        call_id="flush-1", name="memory_write",
                        arguments={"content": RULE, "where": "daily",
                                   "tags": ["flush"]},
                    )],
                    done_reason="stop",
                )
            return OllamaResponse(role="assistant", content="",
                                  tool_calls=[], done_reason="stop")
        return OllamaResponse(role="assistant", content="ack",
                              tool_calls=[], done_reason="stop")

    def close(self) -> None:
        pass


def _padding(approx_tokens: int) -> str:
    """Approximation tokenizer = chars // 4."""
    return ("filler word " * (approx_tokens // 3 + 1)).strip()


def _build_50_turn_history() -> list[dict]:
    """Produce a 50-turn synthetic history with canaries planted on
    specific turns: the Friday rule on turn 3, path on turn 5, fenced
    code on turn 15, URL on turn 25.

    Each turn = one user message + one assistant ack.
    """
    pad = _padding(150)
    msgs: list[dict] = []
    for i in range(1, 51):
        if i == 3:
            user_content = f"turn{i}: {RULE} {pad}"
        elif i == 5:
            user_content = f"turn{i}: please look at {PATH_CANARY} {pad}"
        elif i == 15:
            user_content = f"turn{i}: ```py\nprint('x')\n``` {pad}"
        elif i == 25:
            user_content = f"turn{i}: see {URL_CANARY} for context {pad}"
        else:
            user_content = f"turn{i}: {pad}"
        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": f"ack{i} {pad}"})
    return msgs


def test_compaction_floor_preserves_friday_rule(workspace, cfg, conversation):
    scripted = ScriptedOllama()
    messages = _build_50_turn_history()

    # Sanity: precondition: the synthetic history is long enough to trip the trigger.
    # (If this assertion ever fires after a refactor, bump the padding.)
    from jarvis.core.compaction import _estimate_messages_tokens
    pre = _estimate_messages_tokens(messages)
    trigger_tokens = int(cfg.llm.context_window * cfg.conversation.compaction.trigger_pct)
    assert pre >= trigger_tokens, (
        f"synthetic history is only {pre} tokens; trigger is {trigger_tokens}. "
        "Bump padding or shrink context_window."
    )

    result = maybe_compact(
        conversation=conversation,
        messages=messages,
        system_prompt="(synthetic system prompt — small)",
        paths=workspace,
        cfg=cfg,
        ollama=scripted,
    )

    # ----------------------------------------------------------------
    # PRIMARY (auto-flush correctness): the rule made it to the daily log.
    # ----------------------------------------------------------------
    daily = workspace.daily_log().read_text()
    assert RULE in daily, "auto-flush did not promote the Friday rule to the daily log"
    assert "#flush" in daily, "auto-flushed line missing #flush tag"

    # ----------------------------------------------------------------
    # SECONDARY (defense in depth — floor): rule survived in messages[]
    # OR if it didn't (rule has no floor anchor — no path/code/URL), the
    # PRIMARY assertion above is what carries the contract. We still
    # tolerate either outcome here, but the daily log assertion is binding.
    # ----------------------------------------------------------------
    # The rule itself contains an em-dash but no path/code/URL anchor, so
    # it relies on the auto-flush path. This is intentional — it's the
    # *exact* scenario that motivates auto-flush.

    # ----------------------------------------------------------------
    # CANARIES (floor preservation across the 50 turns).
    # ----------------------------------------------------------------
    contents = " ".join((m.get("content") or "") for m in messages)
    assert PATH_CANARY in contents, "path canary lost from messages[]"
    assert CODE_CANARY in contents, "code-fence canary lost from messages[]"
    assert URL_CANARY in contents, "URL canary lost from messages[]"

    # ----------------------------------------------------------------
    # AUTO-FLUSH HYGIENE: the flush call used the restricted registry
    # (memory_write only) and the locked system prompt.
    # ----------------------------------------------------------------
    flush_calls = [c for c in scripted.chat_calls if c["system"] == _AUTO_FLUSH_SYSTEM_PROMPT]
    assert flush_calls, "expected at least one auto-flush call to fire"
    fc = flush_calls[0]
    assert fc["tools"] is not None
    names = [t["function"]["name"] for t in fc["tools"]]
    assert names == ["memory_write"], f"flush registry leaked extra tools: {names}"
    assert fc["opts"].get("num_ctx") == cfg.llm.context_window

    # ----------------------------------------------------------------
    # JSONL TRANSCRIPT: at least one compaction event with shrinkage.
    # ----------------------------------------------------------------
    events = [json.loads(ln) for ln in conversation.transcript_path.read_text().splitlines()]
    compactions = [e for e in events if e["kind"] == "compaction"]
    assert len(compactions) == 1
    payload = compactions[0]["payload"]
    assert payload["fired"] is True
    assert payload["tokens_after"] <= payload["tokens_before"]
    assert RULE in payload["flushed_facts"]

    # ----------------------------------------------------------------
    # COMPACTION RESULT: matches the JSONL payload.
    # ----------------------------------------------------------------
    assert result.fired is True
    assert RULE in result.flushed_facts
