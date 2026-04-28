"""CMD client — sync HTTP wrapper for the peer agent at ``cfg.orchestration.cmd.base``.

Mirrors the ``OllamaClient`` pattern (``jarvis/clients/ollama.py``):
``httpx.Client`` over a base URL, ``raise_for_status``, ``close()``,
context-manager protocol, scriptable subclass for tests.

P7 surface — three categories:

* **Shared board** — ``publish`` / ``read_context`` / ``delete_context`` hit
  ``/api/v1/context``; not semaphore-guarded (book-keeping only).
* **Quick** — ``quick(command=... | question=...)`` hits ``/api/v1/quick``
  synchronously. Semaphored.
* **Execute (ReAct)** — ``execute(instruction, context_keys=...)`` submits
  ``POST /api/v1/execute`` with ``async:true``, then polls
  ``GET /api/v1/jobs/<id>?envelope_only=1`` until the canonical contract
  envelope arrives. Semaphored across the whole submit + poll.

Concurrency: ``threading.Semaphore(max_concurrent)`` wraps the entire
submit-and-poll block. A slow ReAct job actually backpressures the third
concurrent caller (anti-pattern §19 — a sloppy "submit then return" lets
the daemon stack arbitrary in-flight jobs).

No ``chain()`` method in P7 — deferred to P8 alongside the multi-phase
planner. Adding it later is a clean append; the chain status route shape
(``GET /api/v1/chains/<id>``) is not yet verified, so shipping without it
keeps the contract honest.

Anti-patterns explicitly avoided here:

* **§19 #7** — never auto-retry on CMD failure. Single submit; one
  ``job_id``; one poll loop. The user needs to see ``rm -rf /`` was
  blocked, not a sanitised "command failed".
* **Translating envelope errors into exceptions.** ``execute()`` returns
  the envelope dict unchanged when CMD ships ``{success: false,
  error: "..."}`` — the LLM reads it via ``tool_result.result.error``.
  Only ``CMDTimeout`` and HTTP submit errors raise.
* **Shortening the safety-block message.** Verbatim. (``/quick`` 403 →
  ``CMDError`` whose message is the body's ``reason`` field.)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

__all__ = ["CMDClient", "CMDError", "CMDTimeout"]


class CMDError(Exception):
    """CMD returned an error at the HTTP layer (e.g. 403 safety block)."""


class CMDTimeout(CMDError):
    """Polling exceeded the configured deadline."""


_AllowRisk = Literal["safe", "low", "medium"]


class CMDClient:
    """Synchronous CMD HTTP client.

    Constructor args mirror ``cfg.orchestration.cmd``:

    * ``base`` — e.g. ``http://10.0.0.58:5000``.
    * ``max_concurrent`` — semaphore size (default 2).
    * ``quick_timeout_s`` — default per-call timeout for ``/quick``.
    * ``react_max_wait_s`` — default poll deadline for ``execute``.
    * ``chain_max_wait_s`` — reserved for P8.
    * ``poll_interval_s`` — sleep between status polls.
    * ``http_timeout_s`` — httpx connect/read timeout for individual calls.
    """

    def __init__(
        self,
        base: str,
        *,
        max_concurrent: int = 2,
        quick_timeout_s: int = 15,
        react_max_wait_s: int = 1_800,
        chain_max_wait_s: int = 7_200,
        poll_interval_s: float = 1.0,
        http_timeout_s: float = 30.0,
    ) -> None:
        self._client = httpx.Client(base_url=base.rstrip("/"), timeout=http_timeout_s)
        self._sem = threading.Semaphore(max_concurrent)
        self._quick_timeout_s = quick_timeout_s
        self._react_max_wait_s = react_max_wait_s
        self._chain_max_wait_s = chain_max_wait_s
        self._poll_interval_s = poll_interval_s
        self._http_timeout_s = http_timeout_s

    # -- shared board ------------------------------------------------------

    def publish(
        self,
        key: str,
        value: str,
        *,
        agent_id: str = "jarvis",
        ttl_hours: int = 24,
    ) -> dict:
        """Publish a context blob to the shared board (no semaphore)."""
        resp = self._client.post(
            "/api/v1/context",
            json={"key": key, "value": value, "agent_id": agent_id,
                  "ttl_hours": ttl_hours},
        )
        resp.raise_for_status()
        return resp.json()

    def read_context(
        self,
        *,
        prefix: str | None = None,
        key: str | None = None,
    ) -> dict:
        """Fetch one or many context entries. Pass ``prefix`` or ``key`` (or neither)."""
        params: dict[str, str] = {}
        if prefix is not None:
            params["prefix"] = prefix
        if key is not None:
            params["key"] = key
        resp = self._client.get("/api/v1/context", params=params or None)
        resp.raise_for_status()
        return resp.json()

    def delete_context(self, key: str) -> dict:
        """Delete a context entry by key."""
        resp = self._client.delete(f"/api/v1/context/{key}")
        resp.raise_for_status()
        return resp.json()

    # -- quick -------------------------------------------------------------

    def quick(
        self,
        *,
        command: str | None = None,
        question: str | None = None,
        timeout_s: int | None = None,
        allow_risk: _AllowRisk = "low",
    ) -> dict:
        """One-shot synchronous question or shell command.

        Exactly one of ``command`` / ``question`` must be supplied. The CMD
        server returns 200 sync; on a safety block it returns 403 and we
        raise ``CMDError`` whose message is the body's ``reason`` (or
        ``error``) field verbatim — the LLM should never see a sanitised
        version (anti-pattern §19 #?: "Shortening the safety-block
        message").

        NOTE: ``quick()`` is exempt from ``mode`` / ``master_mode``
        arbitration. Subordinate semantics only apply to ``cmd:code`` /
        ``cmd:gui`` (spec §14). Do NOT add ``mode`` / ``master_mode``
        kwargs here without re-reading §14 — quietly extending arbitration
        to ``cmd:quick`` would force CMD's quick path to interpret a flag
        it doesn't act on.
        """
        if (command is None) == (question is None):
            raise ValueError("quick requires exactly one of command/question")
        body: dict[str, Any] = {
            "timeout": timeout_s if timeout_s is not None else self._quick_timeout_s,
            "allow_risk": allow_risk,
        }
        if command is not None:
            body["command"] = command
        else:
            body["question"] = question

        with self._sem:
            resp = self._client.post("/api/v1/quick", json=body)

        if resp.status_code == 403:
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001
                payload = {}
            msg = (
                payload.get("reason")
                or payload.get("error")
                or resp.text
                or "safety block"
            )
            raise CMDError(msg)

        resp.raise_for_status()
        return resp.json()

    # -- execute (ReAct) ---------------------------------------------------

    def execute(
        self,
        instruction: str,
        *,
        context_keys: list[str] | None = None,
        model: str | None = None,
        timeout_s: int | None = None,
        mode: str | None = None,
        master_mode: str | None = None,
    ) -> dict:
        """Submit a ReAct task; poll envelope_only=1 until terminal.

        Returns the canonical contract envelope dict unchanged — including
        the case where CMD shipped an envelope with ``success: false`` /
        ``error: "..."`` (e.g. the safety-block path on ``/execute``). The
        LLM reads it via ``tool_result.result.error``; only
        ``CMDTimeout`` / HTTP submit failure raise.

        ``mode`` is ``"code"`` or ``"gui"`` for ``cmd:code`` / ``cmd:gui``
        dispatches; ``None`` for ``cmd:react`` (the legacy default — CMD
        picks the mode itself). ``master_mode`` is the master conversation
        role's name (also ``"code"`` or ``"gui"``) — set IFF this dispatch
        is subordinate, telling CMD to run with reduced autonomy. Both
        fields are omitted from the body when ``None``; never serialised
        as a JSON ``null`` (CMD's reads use ``body.get(...)``, but a
        ``cmd:react`` regression that started sending ``mode: null`` would
        be a wire-protocol divergence with no upside).
        """
        body: dict[str, Any] = {"instruction": instruction, "async": True}
        if context_keys:
            body["context_keys"] = list(context_keys)
        if model:
            body["model"] = model
        if timeout_s is not None:
            body["timeout"] = timeout_s
        if mode is not None:
            body["mode"] = mode
        if master_mode is not None:
            body["master_mode"] = master_mode

        deadline_s = timeout_s if timeout_s is not None else self._react_max_wait_s

        with self._sem:
            submit = self._client.post("/api/v1/execute", json=body)
            submit.raise_for_status()
            data = submit.json()
            job_id = data.get("job_id")
            if not job_id:
                raise CMDError(f"execute: missing job_id in submit response: {data!r}")

            return self._poll_envelope(job_id, deadline_s)

    # -- internals ---------------------------------------------------------

    def _poll_envelope(self, job_id: str, deadline_s: float) -> dict:
        deadline = time.monotonic() + deadline_s
        while True:
            resp = self._client.get(
                f"/api/v1/jobs/{job_id}",
                params={"envelope_only": 1},
            )
            resp.raise_for_status()
            payload = resp.json() or {}
            envelope = payload.get("envelope")
            # Server returns the envelope under the key ``envelope`` once
            # the job is terminal; some shapes inline the keys at top-level.
            if envelope is None and "success" in payload:
                envelope = payload
            if isinstance(envelope, dict) and "success" in envelope:
                return envelope
            if time.monotonic() >= deadline:
                raise CMDTimeout(
                    f"execute: job {job_id} did not return an envelope within "
                    f"{deadline_s:.0f}s"
                )
            time.sleep(self._poll_interval_s)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CMDClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
