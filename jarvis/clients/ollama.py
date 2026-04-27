"""Minimal sync Ollama client for P1.

Single ``complete()`` method. No streaming, no retries — those land in P5.
Raises ``httpx.HTTPStatusError`` on non-2xx so callers can decide policy.
"""

from __future__ import annotations

from typing import Any

import httpx

__all__ = ["OllamaClient"]


class OllamaClient:
    """Tiny synchronous Ollama client.

    Uses ``/api/chat`` with ``stream=False``. The result is the assistant's
    full message content as a string.
    """

    def __init__(self, host: str, timeout_s: float = 120.0) -> None:
        self._client = httpx.Client(base_url=host.rstrip("/"), timeout=timeout_s)

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

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
