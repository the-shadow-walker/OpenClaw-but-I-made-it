"""Auto-compaction with auto-flush (BUILD_SPEC §8, §2 invariant 5).

Single entry point: :func:`maybe_compact`. Called inside ``run_turn`` before
each ``ollama.chat()`` call. No-op below the trigger; on trip, runs three
phases:

  1. **Auto-flush** (best-effort): a silent, agentic round-trip on the fast
     model with a *restricted* registry exposing only ``memory_write``. The
     model sees the about-to-be-truncated older segment with tool roles
     flattened into readable text (NOT filtered — durable facts often live
     mid-tool-use). Each ``memory_write`` call is forced to
     ``where='daily'`` with ``'flush'`` in ``tags`` regardless of what the
     model passed (belt-and-suspenders against small-model drift; the
     system prompt also requests it). Failure here logs and continues to
     Phase 2 — auto-flush is never a hard blocker on truncation.

  2. **Truncate with floor preservation**: the last ``keep_recent_turns``
     user-turn-rooted blocks survive verbatim. For older messages we keep
     anything matching the floor regex (fenced code, absolute or ~/ paths,
     URLs); summarize tool_results that don't; drop empty assistant shells
     that carried tool_calls; drop user/assistant prose that hit no floor.
     Pairs are atomic: a ``tool_call`` assistant shell + its tool_result(s)
     stay together — if **either** matches the floor, **both** are kept.

  3. **Emergency drop pass**: if Phases 1+2 still leave us above
     ``0.95 * context_window`` (typical when the user pasted a long
     document with paths/code that the floor caught), drop the oldest
     floor-preserved messages first, then the oldest of the recent turns,
     until under target. Always preserves at least one recent turn. WARN —
     known degraded state, not an error.

Then append one ``compaction`` event to the JSONL transcript.

The system prompt is read for token accounting only — never mutated. A
single WARN fires when the system prompt itself exceeds 50% of the window
(long conversations grow the daily log, which grows the assembled prompt;
compaction can't shrink it from this layer).

Anti-patterns avoided (§19, §8):
  * Mid-pair truncation — pair atomicity with "floor wins for the pair".
  * Filtering tool roles in flush — facts get lost mid-tool-use.
  * Hard auto-flush dependency — failure logs, truncation continues.
  * Touching the system prompt — never.
  * Storing flushed facts in MEMORY.md — locked to ``where='daily'``.
  * Re-implementing token counting — uses ``chunker.count_tokens``.
  * Compacting below trigger — early-return after one cheap sweep.
  * Wire-exposing compaction events — JSONL only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from jarvis.clients.ollama import OllamaClient
from jarvis.config import JarvisConfig
from jarvis.core.conversation import ChannelKind, Conversation
from jarvis.core.tools import ToolRegistry, ToolSpec
from jarvis.memory.chunker import count_tokens
from jarvis.memory.tool_write import memory_write_tool
from jarvis.memory.workspace import WorkspacePaths

logger = logging.getLogger(__name__)

__all__ = ["CompactionResult", "maybe_compact", "_AUTO_FLUSH_SYSTEM_PROMPT"]


# ---------------------------------------------------------------------------
# Floor regexes — over-keep aggressively. Asymmetry: a false positive bloats;
# a false negative drops a real rule (silent data loss). Lean into bloat.
# ---------------------------------------------------------------------------

# Fenced code — any triple backtick anywhere opens preservation.
_FENCED_CODE_RE = re.compile(r"```", re.MULTILINE)

# Paths — absolute (/...) or home-relative (~/...). Allows boundary at
# line-start or any whitespace / open-delim; matches /xxx or ~/xxx with
# at least 2 chars after / so casual prose like "and/or" doesn't trigger.
_PATH_RE = re.compile(
    r"""(?:^|[\s`'"(\[<])"""
    r"""(/[^\s`'")\]>]{2,}|~/[^\s`'")\]>]+)""",
    re.MULTILINE,
)

# URLs — http(s) plus path/query, stops at whitespace or close-delim.
_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)


def _matches_floor(content: str) -> bool:
    """True iff ``content`` matches any floor pattern (code / path / URL)."""
    if not content:
        return False
    return bool(
        _FENCED_CODE_RE.search(content)
        or _PATH_RE.search(content)
        or _URL_RE.search(content)
    )


# ---------------------------------------------------------------------------
# Auto-flush system prompt — locked to daily + flush tag.
# ---------------------------------------------------------------------------

_AUTO_FLUSH_SYSTEM_PROMPT = """\
You are the auto-flush phase of a memory compactor. The conversation
segment below is about to be truncated from the model's working context.
Identify any DURABLE rules, preferences, constraints, or facts the user
stated (or that were established about the user) that should survive
truncation.

For each such fact, call:
    memory_write(content=<one concise sentence>, where="daily",
                 tags=["flush"])

CRITICAL CONSTRAINTS — non-negotiable:
  - You MUST set where="daily". Do NOT use where="memory". MEMORY.md
    is reserved for the Dreaming pipeline; auto-flush only writes to
    the daily log.
  - You MUST include "flush" in the tags list so future passes can
    distinguish auto-flushed facts from manually-stated ones.
  - Do NOT write questions the user asked, one-time tasks, or facts
    already trivially deducible from existing memory.
  - If no durable facts are present, write nothing and respond with an
    empty message. This is a silent background turn — your prose is not
    shown to the user.
"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionResult:
    fired: bool
    tokens_before: int
    tokens_after: int
    flushed_facts: list[str] = field(default_factory=list)
    kept_floor_count: int = 0
    summarized_count: int = 0
    dropped_count: int = 0
    emergency_drop_count: int = 0


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_messages_tokens(messages: list[dict]) -> int:
    """Sum count_tokens(content) plus 4-token overhead per message plus the
    serialized tool_call arguments. Cheap and conservative."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += count_tokens(content)
        total += 4
        for tc in m.get("tool_calls") or []:
            args = (tc.get("function") or {}).get("arguments", {})
            total += count_tokens(str(args))
    return total


def _find_recent_turn_boundary(messages: list[dict], keep_recent_turns: int) -> int:
    """Index in ``messages`` where the recent block starts.

    Walks backward through user-role messages; the n-th-last user message
    marks the boundary. If there are <= keep_recent_turns user messages,
    everything is recent (boundary == 0).
    """
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_recent_turns:
        return 0
    return user_indices[-keep_recent_turns]


# ---------------------------------------------------------------------------
# Decide / summarize / drop on the older segment
# ---------------------------------------------------------------------------


def _floor_text_for(msg: dict) -> str:
    """Build the haystack for floor matching: content + serialized tool_call args.

    An assistant shell whose ``tool_calls`` mention an absolute path is just
    as durable as a content match — preserve.
    """
    parts: list[str] = []
    content = msg.get("content") or ""
    if content:
        parts.append(content)
    for tc in msg.get("tool_calls") or []:
        args = (tc.get("function") or {}).get("arguments", {})
        if args:
            parts.append(str(args))
    return " ".join(parts)


def _decide_older(older: list[dict]) -> list[str]:
    """Return per-index decision: 'keep' | 'summarize' | 'drop'.

    "floor wins for the pair": a tool_call assistant shell + its
    tool_result(s) form an atomic group; if any group member matches the
    floor, the whole group is promoted to 'keep'.
    """
    n = len(older)
    decisions: list[str] = ["drop"] * n

    # Per-message default decisions.
    for i, m in enumerate(older):
        role = m.get("role")
        if _matches_floor(_floor_text_for(m)):
            decisions[i] = "keep"
        elif role == "tool":
            decisions[i] = "summarize"
        else:
            decisions[i] = "drop"

    # Pair atomicity: assistant-with-tool_calls + trailing tool messages.
    i = 0
    while i < n:
        m = older[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            j = i + 1
            while j < n and older[j].get("role") == "tool":
                j += 1
            pair = list(range(i, j))
            if any(decisions[k] == "keep" for k in pair):
                for k in pair:
                    decisions[k] = "keep"
            i = j
        else:
            i += 1

    return decisions


def _summarize_tool_result(msg: dict) -> dict:
    """Replace a tool_result message's content with a compact summary line."""
    content = msg.get("content") or ""
    name = msg.get("name", "?")
    head = content[:120].replace("\n", " ")
    return {
        "role": "tool",
        "name": name,
        "content": f"[compacted] tool {name} returned {len(content)} chars; summary: {head}…",
    }


# ---------------------------------------------------------------------------
# Flush rendering and registry
# ---------------------------------------------------------------------------


def _render_older_for_flush(older: list[dict]) -> str:
    """Flatten ``older`` to readable text with tool calls preserved (NOT filtered).

    Format::

        [turn N, user] ...
        [turn N, assistant] ...
        [turn N, assistant called toolname(args)]
        [turn N, tool toolname returned <N> chars: head]
        [turn N, assistant] final
    """
    lines: list[str] = []
    turn = 0
    for m in older:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            turn += 1
            lines.append(f"[turn {turn}, user] {content}")
        elif role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "?")
                    args = fn.get("arguments", {})
                    lines.append(f"[turn {turn}, assistant called {name}({args})]")
                if content.strip():
                    lines.append(f"[turn {turn}, assistant] {content}")
            else:
                lines.append(f"[turn {turn}, assistant] {content}")
        elif role == "tool":
            name = m.get("name", "?")
            head = content[:200].replace("\n", " ")
            lines.append(f"[turn {turn}, tool {name} returned {len(content)} chars: {head}]")
    return "\n".join(lines)


_FLUSH_MEMORY_WRITE_PARAMS: dict = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "Durable fact, one sentence."},
        "where": {"type": "string", "enum": ["daily"], "default": "daily"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["content"],
}


def _build_flush_registry(paths: WorkspacePaths) -> ToolRegistry:
    """Restricted registry — exposes only ``memory_write``.

    ``fast_model`` == ``chat_model`` in the current config (both
    ``qwen2.5:3b``) makes the "smaller model for flush" optimization a
    no-op today; preserve the seam so a future tuning pass can drop a
    1B-param model in for ``fast_model`` without touching this layer.
    """
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="memory_write",
            description="Persist a durable fact to today's daily log.",
            parameters=_FLUSH_MEMORY_WRITE_PARAMS,
            handler=lambda **kw: memory_write_tool(**kw, paths=paths),
        )
    )
    return registry


def _auto_flush(
    older: list[dict],
    *,
    cfg: JarvisConfig,
    ollama: OllamaClient,
    paths: WorkspacePaths,
) -> list[str]:
    """Run one round-trip on the fast model. Returns the list of contents
    actually persisted. Best-effort — caller swallows exceptions."""
    flush_registry = _build_flush_registry(paths)
    flush_text = _render_older_for_flush(older)
    user_msg = {"role": "user", "content": flush_text}

    resp = ollama.chat(
        cfg.llm.fast_model,
        [user_msg],
        tools=flush_registry.schemas(),
        system=_AUTO_FLUSH_SYSTEM_PROMPT,
        num_ctx=cfg.llm.context_window,
    )

    flushed: list[str] = []
    for tc in resp.tool_calls:
        if tc.name != "memory_write":
            # Restricted registry would refuse anyway; skip without dispatch.
            continue
        args: dict[str, Any] = dict(tc.arguments or {})
        # Force where='daily' regardless of what the model passed (small models drift).
        args["where"] = "daily"
        # Ensure 'flush' tag is present.
        tags = list(args.get("tags") or [])
        if "flush" not in tags:
            tags.append("flush")
        args["tags"] = tags

        try:
            flush_registry.execute("memory_write", args)
            content = args.get("content") or ""
            if content:
                flushed.append(content)
        except Exception:  # noqa: BLE001
            logger.exception("compaction: auto-flush memory_write failed")
    return flushed


# ---------------------------------------------------------------------------
# Emergency drop
# ---------------------------------------------------------------------------


def _emergency_drop(
    messages: list[dict],
    *,
    sys_prompt_tokens: int,
    context_window: int,
) -> int:
    """Drop oldest message groups until under 0.95 * window.

    Pair-atomic: an assistant-with-tool_calls drops together with its
    trailing tool_result messages. Stops once only one user-message turn
    remains (the user must always have at least their latest turn).
    """
    target = int(context_window * 0.95)
    dropped = 0
    while True:
        cur = _estimate_messages_tokens(messages) + sys_prompt_tokens
        if cur <= target:
            break
        if not messages:
            break
        # Stop guard: keep at least one user message.
        user_count = sum(1 for m in messages if m.get("role") == "user")
        if user_count <= 1:
            break

        first = messages[0]
        to_drop = 1
        if first.get("role") == "assistant" and first.get("tool_calls"):
            j = 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            to_drop = j
        elif first.get("role") == "user":
            # Drop the entire turn: user + following assistant/tool messages
            # up to (but not including) the next user message. This is the
            # "drop oldest of the recent turns" path.
            j = 1
            while j < len(messages) and messages[j].get("role") != "user":
                j += 1
            to_drop = j

        for _ in range(to_drop):
            if messages:
                messages.pop(0)
                dropped += 1
    return dropped


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def maybe_compact(
    *,
    conversation: Conversation,
    messages: list[dict],
    system_prompt: str,
    paths: WorkspacePaths,
    cfg: JarvisConfig,
    ollama: OllamaClient,
    embedder: Any = None,
    channel_kind: ChannelKind | None = None,
) -> CompactionResult:
    """Mutate ``messages`` in place if the assembled token total exceeds the
    trigger. Returns a :class:`CompactionResult` describing what happened.

    No-op below the trigger (one ``count_tokens`` sweep). When fired,
    appends one ``compaction`` event to the JSONL transcript.

    ``embedder`` and ``channel_kind`` are accepted for forward compatibility
    (a future pass may restrict the flush registry per channel or use the
    embedder for fact-de-dup); current implementation does not consume them.
    """
    del embedder, channel_kind  # reserved for future use

    context_window = cfg.llm.context_window
    sys_prompt_tokens = count_tokens(system_prompt)

    # System-prompt size warning (compaction does not shrink it).
    if sys_prompt_tokens > context_window * 0.5:
        logger.warning(
            "compaction: system prompt is %d tokens (>%d, 50%% of context_window=%d) "
            "— compaction does not shrink the system prompt; long-conversation tuning "
            "should revisit prompt assembly",
            sys_prompt_tokens, int(context_window * 0.5), context_window,
        )

    tokens_before = _estimate_messages_tokens(messages) + sys_prompt_tokens
    compaction_cfg = cfg.conversation.compaction
    trigger_tokens = int(context_window * compaction_cfg.trigger_pct)

    if tokens_before < trigger_tokens:
        return CompactionResult(
            fired=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
        )

    boundary = _find_recent_turn_boundary(messages, compaction_cfg.keep_recent_turns)
    older = list(messages[:boundary])

    # Phase 1 — auto-flush. Best-effort; failure does not block truncation.
    flushed_facts: list[str] = []
    if compaction_cfg.auto_flush and older:
        try:
            flushed_facts = _auto_flush(older, cfg=cfg, ollama=ollama, paths=paths)
        except Exception:  # noqa: BLE001
            logger.exception("compaction: auto-flush raised — continuing to truncate")
            flushed_facts = []

    # Phase 2 — truncate older segment with floor preservation.
    decisions = _decide_older(older)
    new_older: list[dict] = []
    kept_floor = 0
    summarized = 0
    dropped = 0
    for m, decision in zip(older, decisions, strict=True):
        if decision == "keep":
            new_older.append(m)
            kept_floor += 1
        elif decision == "summarize":
            new_older.append(_summarize_tool_result(m))
            summarized += 1
        else:
            dropped += 1

    recent = list(messages[boundary:])
    messages.clear()
    messages.extend(new_older)
    messages.extend(recent)

    # Phase 3 — emergency drop (rare; pasted-document case).
    emergency = _emergency_drop(
        messages,
        sys_prompt_tokens=sys_prompt_tokens,
        context_window=context_window,
    )
    if emergency:
        logger.warning(
            "compaction: emergency drop fired (n=%d) — context window pressure; "
            "degraded context this turn",
            emergency,
        )

    tokens_after = _estimate_messages_tokens(messages) + sys_prompt_tokens

    result = CompactionResult(
        fired=True,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        flushed_facts=flushed_facts,
        kept_floor_count=kept_floor,
        summarized_count=summarized,
        dropped_count=dropped,
        emergency_drop_count=emergency,
    )
    try:
        conversation.append("compaction", asdict(result))
    except Exception:  # noqa: BLE001
        logger.exception("compaction: failed to append compaction event to JSONL")
    return result
