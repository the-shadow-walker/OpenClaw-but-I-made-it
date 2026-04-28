"""Ollama client — sync, two methods (P1 ``complete()`` + P5 ``chat()``).

``complete()`` is preserved verbatim so the P1 integration test
(``tests/integration/test_ollama_dm.py``) keeps working as-is.

``chat()`` adds tool-call support: passes a tools schema list, parses
``message.tool_calls`` from the response, synthesizes a ``call_id`` per
call (Ollama doesn't always provide one), and returns a structured
``OllamaResponse``. qwen2.5:3b supports tool calls in Ollama 0.4+.

Retries: a single retry on ``httpx.ConnectError`` / ``httpx.ReadTimeout``
with 1s backoff. Aggressive retry policies are P5+ tuning concerns; this
is the bare minimum to not flake on a one-off blip.

If the model returns ``tool_calls`` but the registered tools list is
empty, treat it as a hallucination — drop the tool calls and pass the
content through. (Ollama tool-format docs:
https://ollama.com/blog/tool-support).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = ["OllamaClient", "OllamaToolCall", "OllamaResponse"]


@dataclass(frozen=True)
class OllamaToolCall:
    """One tool-call request from the model. ``call_id`` is locally synthesized."""
    call_id: str
    name: str
    arguments: dict


@dataclass
class OllamaResponse:
    """Parsed response from ``/api/chat``.

    ``content`` is the assistant message text (may be empty when the model
    returned only tool_calls). ``tool_calls`` is empty when the model has
    no requests. ``done_reason`` mirrors Ollama's ``done_reason`` field
    (e.g. ``"stop"``, ``"length"``).
    """
    role: str
    content: str
    tool_calls: list[OllamaToolCall] = field(default_factory=list)
    done_reason: str | None = None
    raw: dict | None = None


class OllamaClient:
    """Synchronous Ollama client. Both ``/api/chat`` paths.

    P1 ``complete()``: returns the assistant text. Untouched.
    P5 ``chat()``:    structured response with tool-call parsing.
    """

    def __init__(self, host: str, timeout_s: float = 120.0) -> None:
        self._client = httpx.Client(base_url=host.rstrip("/"), timeout=timeout_s)
        # P8: per-call estimated input-token counter, summed across all
        # ``chat()`` invocations. Callers reset to 0 at turn boundaries
        # (see ``run_turn``); the rocket-sim acceptance test reads it
        # after the streaming response completes to assert the §16-D
        # token-budget gate. Measured at the model-input boundary
        # (messages + system + tools payload), not at JSONL events —
        # see plan pre-decision #9.
        self.chat_input_tokens_total: int = 0

    # -- P1 ----------------------------------------------------------------

    def complete(
        self,
        model: str,
        messages: list[dict],
        system: str | None = None,
        **opts: Any,
    ) -> str:
        """POST a chat completion and return the assistant message content.

        ``messages`` is the standard Ollama-style list of
        ``{"role": "...", "content": "..."}`` dicts. ``system`` is prepended
        as a ``{"role": "system", ...}`` entry — Ollama's ``/api/chat``
        treats system content this way (the top-level ``system`` field is
        only honored by the older ``/api/generate`` endpoint). ``opts``
        (e.g. ``temperature``, ``num_predict``) are passed through under
        ``options``.
        """
        full_messages: list[dict] = list(messages)
        if system is not None:
            full_messages = [{"role": "system", "content": system}, *full_messages]

        payload: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "stream": False,
        }
        if opts:
            payload["options"] = dict(opts)

        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    # -- P5 ----------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        system: str | None = None,
        **opts: Any,
    ) -> OllamaResponse:
        """Structured chat completion with optional tool-call support.

        Caller is responsible for passing ``num_ctx`` to bind the model's
        context window. Without it, Ollama applies its own default and may
        silently truncate long conversations underneath us — exactly the
        condition P6 compaction is meant to handle predictably.
        """
        full_messages: list[dict] = list(messages)
        if system is not None:
            full_messages = [{"role": "system", "content": system}, *full_messages]

        payload: dict[str, Any] = {
            "model": model,
            "messages": full_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if opts:
            payload["options"] = dict(opts)

        # P8 (§16-D): instrument input-token usage at the model-input
        # boundary. Estimate via the configured tokenizer over messages
        # + tools payload; failure to import the helper (cycle / partial
        # init) silently degrades to 0 rather than crashing chat.
        try:
            from jarvis.core.compaction import _estimate_messages_tokens
            extra = 0
            if tools:
                # Tools are part of the model input every call; encode the
                # JSON for a conservative estimate.
                import json as _json

                from jarvis.memory.chunker import count_tokens as _ct
                extra = _ct(_json.dumps(tools, ensure_ascii=False, default=str))
            self.chat_input_tokens_total += (
                _estimate_messages_tokens(full_messages) + extra
            )
        except Exception:  # noqa: BLE001
            pass

        data = self._post_with_retry("/api/chat", payload)
        return self._parse_response(data, tools_present=bool(tools))

    # -- internals ---------------------------------------------------------

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in (0, 1):
            try:
                resp = self._client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exc = e
                if attempt == 0:
                    logger.warning("ollama %s: %s — retrying once in 1s", type(e).__name__, e)
                    time.sleep(1.0)
                    continue
                raise
        # Unreachable — the loop either returns or raises.
        raise RuntimeError(f"unreachable: {last_exc}")  # pragma: no cover

    @staticmethod
    def _parse_response(data: dict, *, tools_present: bool) -> OllamaResponse:
        msg = data.get("message", {}) or {}
        role = msg.get("role", "assistant")
        content = msg.get("content", "") or ""
        done_reason = data.get("done_reason")

        raw_calls = msg.get("tool_calls") or []
        if raw_calls and not tools_present:
            # Hallucinated tool calls — fold into plain content so the
            # caller's loop terminates instead of trying to dispatch.
            logger.warning("ollama: ignoring %d hallucinated tool_calls (no tools registered)",
                           len(raw_calls))
            raw_calls = []

        parsed: list[OllamaToolCall] = []
        for tc in raw_calls:
            fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
            name = fn.get("name")
            if not name:
                continue
            args = fn.get("arguments")
            if isinstance(args, str):
                # Some models emit a JSON-stringified arguments blob.
                import json
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {"_raw": args}
            elif args is None:
                args = {}
            parsed.append(OllamaToolCall(
                call_id=f"tc-{uuid.uuid4().hex[:8]}",
                name=name,
                arguments=args,
            ))

        return OllamaResponse(
            role=role,
            content=content,
            tool_calls=parsed,
            done_reason=done_reason,
            raw=data,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
