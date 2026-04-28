"""Chat tool loop (BUILD_SPEC §6.5, §7).

Synchronous generator that drives one user turn → final assistant answer
through any number of intermediate tool calls. Yields events as dicts so
the FastAPI layer can NDJSON-stream them with no extra translation.

Per §7: the system prompt is reassembled fresh from USER.md + MEMORY.md +
daily logs every turn, so live edits to memory take effect immediately.

The loop runs synchronously — async lands when token streaming does (P5+).

Event shapes yielded::

    {"type": "system_prompt", "size": int}
    {"type": "tool_call",   "call_id": str, "name": str, "arguments": dict}
    {"type": "tool_result", "call_id": str, "name": str,
                            "result": Any | None, "error": str | None}
    {"type": "delta",       "text": str}            # final assistant answer
    {"type": "done",        "stop_reason": str}     # "stop" | "tool_limit" | "error"

Tool-iteration cap: ``max_tool_iterations=8`` catches "let me search just
one more thing" loops. Hitting the cap surfaces ``stop_reason="tool_limit"``
in the transcript so the user sees something happened.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

from jarvis.clients.ollama import OllamaClient
from jarvis.config import JarvisConfig
from jarvis.core.compaction import maybe_compact
from jarvis.core.conversation import ChannelKind, Conversation
from jarvis.core.prompt import assemble_system_prompt
from jarvis.core.tools import ToolRegistry
from jarvis.memory.embeddings import EmbeddingPipeline
from jarvis.memory.workspace import WorkspacePaths

logger = logging.getLogger(__name__)

__all__ = ["ChatTurnConfig", "run_turn"]


@dataclass(frozen=True)
class ChatTurnConfig:
    max_tool_iterations: int = 8


def _serialize_for_tool_result(value) -> str:  # type: ignore[no-untyped-def]
    """JSON-encode a tool result for the assistant message thread.

    Ollama's chat API expects the tool message ``content`` to be a string.
    Falls back to ``repr`` for non-JSON-serializable objects.
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)


def run_turn(
    *,
    user_text: str,
    conversation: Conversation,
    paths: WorkspacePaths,
    cfg: JarvisConfig,
    ollama: OllamaClient,
    registry: ToolRegistry,
    channel_kind: ChannelKind,
    active_project_slug: str | None = None,
    turn_cfg: ChatTurnConfig | None = None,
    embedder: EmbeddingPipeline | None = None,
) -> Iterator[dict]:
    """Drive one user turn. Yields events for the streaming layer.

    See module docstring for event shapes. The loop appends every event to
    the conversation's JSONL transcript; the wire never sees the system
    prompt content (size only).
    """
    turn_cfg = turn_cfg or ChatTurnConfig()

    # P8 (§16-D): zero the per-turn token-usage counter so the rocket-sim
    # acceptance test can read it back after the stream terminates.
    # Test doubles (e.g. FakeOllama subclasses that don't call super())
    # may not have the attribute; tolerate.
    import contextlib as _contextlib
    with _contextlib.suppress(AttributeError):
        ollama.chat_input_tokens_total = 0

    # 1. Reassemble fresh system prompt every turn (§7).
    # Group / heartbeat have their own loading rules; "cli" uses the DM scaffold.
    prompt_kind = channel_kind
    if channel_kind == "cli":
        prompt_kind = "cli"  # assemble_system_prompt handles cli==dm in P5
    system_prompt = assemble_system_prompt(
        paths, prompt_kind, active_project_slug=active_project_slug
    )
    conversation.append("system_prompt", {"content": system_prompt, "size": len(system_prompt)})
    yield {"type": "system_prompt", "size": len(system_prompt)}

    # 2. Append user_message to JSONL + chat thread.
    conversation.append("user_message", {"content": user_text})
    messages: list[dict] = [{"role": "user", "content": user_text}]

    # Rule-based router hint for forward-compat with the P10 LLM router.
    # DEBUG only — the hint is noisy and never short-circuits the loop;
    # the LLM is the routing arbiter (it emits ``delegate(target=...)``).
    from jarvis.core.router import classify
    logger.debug("router: hint=%s", classify(user_text))

    # 3. Tool loop.
    schemas = registry.schemas()
    stop_reason = "stop"
    final_content = ""
    for iteration in range(turn_cfg.max_tool_iterations):
        # Compact if we'd otherwise blow past the context window. Mutates
        # ``messages`` in place; no-op below the trigger. Per-iteration so
        # a tool-heavy turn that grows mid-loop still gets caught.
        maybe_compact(
            conversation=conversation,
            messages=messages,
            system_prompt=system_prompt,
            paths=paths,
            cfg=cfg,
            ollama=ollama,
            embedder=embedder,
            channel_kind=channel_kind,
        )
        try:
            resp = ollama.chat(
                cfg.llm.chat_model,
                messages,
                tools=schemas,
                system=system_prompt,
                num_ctx=cfg.llm.context_window,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("chat: ollama.chat raised on iteration %d", iteration)
            yield {"type": "error", "message": f"ollama error: {e}"}
            yield {"type": "done", "stop_reason": "error"}
            return

        # Persist the assistant turn (with whatever tool_calls it contained).
        tool_call_payload = [
            {"call_id": tc.call_id, "name": tc.name, "arguments": tc.arguments}
            for tc in resp.tool_calls
        ] or None
        conversation.append(
            "assistant_message",
            {"content": resp.content, "tool_calls": tool_call_payload},
        )
        # Mirror into the in-memory message thread so subsequent iterations
        # see the assistant's tool requests.
        assistant_msg: dict = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            assistant_msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in resp.tool_calls
            ]
        messages.append(assistant_msg)

        if not resp.tool_calls:
            final_content = resp.content
            stop_reason = resp.done_reason or "stop"
            break

        # 4. Dispatch each tool call; emit events; thread results back.
        for tc in resp.tool_calls:
            yield {
                "type": "tool_call",
                "call_id": tc.call_id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
            conversation.append("tool_call", {
                "call_id": tc.call_id, "name": tc.name, "arguments": tc.arguments,
            })

            error: str | None = None
            result = None
            try:
                result = registry.execute(tc.name, tc.arguments)
            except KeyError as e:
                error = f"unknown tool: {e}"
            except Exception as e:  # noqa: BLE001
                logger.exception("tool %s raised", tc.name)
                error = f"{type(e).__name__}: {e}"

            yield {
                "type": "tool_result",
                "call_id": tc.call_id,
                "name": tc.name,
                "result": result,
                "error": error,
            }
            conversation.append("tool_result", {
                "call_id": tc.call_id, "name": tc.name,
                "result": result, "error": error,
            })

            # Thread the result into the chat for the next iteration.
            tool_msg_content = (
                _serialize_for_tool_result({"error": error})
                if error is not None
                else _serialize_for_tool_result(result)
            )
            messages.append({
                "role": "tool",
                "content": tool_msg_content,
                "name": tc.name,
            })
    else:
        # Hit the iteration cap — surface what we last got.
        stop_reason = "tool_limit"
        final_content = final_content or "(tool iteration limit reached)"

    yield {"type": "delta", "text": final_content}
    yield {"type": "done", "stop_reason": stop_reason}
