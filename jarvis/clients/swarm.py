"""Swarm client — sync long-poll HTTP wrapper for the Swarm peer (P8).

Mirrors the ``CMDClient`` shape (``jarvis/clients/cmd.py``):
``httpx.Client`` over a base URL, ``raise_for_status``, ``close()``,
context-manager protocol, scriptable subclass for tests.

§12.0 reality check (the live Swarm surface differs from spec §12)
-------------------------------------------------------------------
Live server at ``http://10.0.0.58:5002``:

* Submit endpoint: ``POST /subagent/<role>`` where the server-side role
  name ∈ ``{math, engineer, deep_search}``.
* Body shape: ``{task, context_keys: list[str], extra: {timeout_s,
  max_iterations}}``. ``context_keys`` is **top-level**, NOT nested in
  ``extra`` — the spec's ``parent_context_keys`` field does not exist.
* **Sync long-poll** — one POST returns the canonical envelope directly.
  There is no submit-and-poll flow on Swarm (unlike CMD's ``/execute``).
* HTTP codes: 200 success, 400 bad-request, 429 busy, 500 crash, 504
  timeout. **500 / 504 still ship a contract envelope** with
  ``success=false`` per §12.0.
* Hard server-side cap on ``timeout_s`` is 3600s; we clamp client-side
  to be polite.

Spec name mapping
-----------------
Spec §13 calls one role ``swarm:research``. The live server names that
role ``deep_search``. ``_ROLE_MAP`` accepts both spellings client-side
for tolerance toward direct ``delegate(target="swarm:deep_search")``
calls; the **canonical name the planner emits is ``swarm:research``** —
``deep_search`` is reachable only through that compatibility path.

Anti-patterns explicitly avoided here
-------------------------------------
* **§19 #7** — never auto-retry on Swarm failure. Single POST per
  ``dispatch``; first failure surfaces as an envelope error or raises.
* **§19 #3 / §10.4** — the canonical envelope contains ``sidechain_path``
  but we never read its content. Only contract keys (``success``,
  ``summary``, ``deliverables``, ``context_keys_written``,
  ``sidechain_path``, ``error``) are honored on coercion.
* **§19 #9** — keep ``task`` short; rich context flows via top-level
  ``context_keys`` references, not inline payload bloat.
* **Translating envelope errors into exceptions** — when Swarm ships
  ``{"success": false, "error": "..."}`` (envelope path), return the
  envelope unchanged. Only HTTP-layer issues (400, 429, ReadTimeout)
  raise.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = ["SwarmClient", "SwarmError", "SwarmBusy", "SwarmTimeout"]


class SwarmError(Exception):
    """Swarm returned an HTTP-layer error (e.g. 400 bad-request)."""


class SwarmBusy(SwarmError):
    """Swarm returned 429 — all worker slots in use."""


class SwarmTimeout(SwarmError):
    """Sync long-poll exceeded the local read deadline."""


# Spec name → live server role name. ``swarm:research`` is the canonical
# spec name; the live server uses ``deep_search``. Both are accepted on
# the spec side; the wire only ever sees the live name.
_ROLE_MAP: dict[str, str] = {
    "math": "math",
    "engineer": "engineer",
    "research": "deep_search",
    "deep_search": "deep_search",
}
_ALLOWED_ROLES = frozenset(_ROLE_MAP.keys())

# Server-side hard cap per §12.0.
_SERVER_HARD_CAP_S = 3600

# Contract envelope keys we honor when coercing the response. Anything
# else the server ships is preserved on the dict (so the JSONL captures
# meta) but never consumed by the orchestrator.
_CONTRACT_KEYS = {
    "success",
    "summary",
    "deliverables",
    "context_keys_written",
    "sidechain_path",
    "error",
}


class SwarmClient:
    """Synchronous Swarm HTTP client. One method: :meth:`dispatch`.

    Constructor args mirror ``cfg.orchestration.swarm``:

    * ``base`` — e.g. ``http://10.0.0.58:5002``.
    * ``max_concurrent`` — semaphore size (default 2). Wraps the entire
      sync long-poll so the third concurrent caller backpressures.
    * ``dispatch_max_wait_s`` — local read deadline (default 1800,
      clamped at the 3600s server-side hard cap).
    * ``http_timeout_s`` — explicit override; defaults to
      ``dispatch_max_wait_s`` so long-poll calls don't trip httpx's
      default read timeout.
    * ``auth_token`` — optional; sent as ``Authorization: Bearer <tok>``.
    """

    def __init__(
        self,
        base: str,
        *,
        max_concurrent: int = 2,
        dispatch_max_wait_s: int = 1_800,
        http_timeout_s: float | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._dispatch_max_wait_s = min(int(dispatch_max_wait_s), _SERVER_HARD_CAP_S)
        # Long-poll: read timeout must be >= dispatch deadline so the
        # server gets to respond on its own schedule.
        timeout = (
            float(http_timeout_s)
            if http_timeout_s is not None
            else float(self._dispatch_max_wait_s + 30)
        )
        headers: dict[str, str] = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        self._client = httpx.Client(
            base_url=base.rstrip("/"),
            timeout=timeout,
            headers=headers or None,
        )
        self._sem = threading.Semaphore(max_concurrent)
        self._auth_token = auth_token

    # -- dispatch ----------------------------------------------------------

    def dispatch(
        self,
        role: str,
        task: str,
        *,
        context_keys: list[str] | None = None,
        max_iterations: int = 40,
        timeout_s: int | None = None,
    ) -> dict:
        """Submit a task to ``/subagent/<server_role>`` (sync long-poll).

        Body shape::

            {
              "task": "...",
              "context_keys": ["k1", "k2"],
              "extra": {"timeout_s": 1800, "max_iterations": 40}
            }

        Returns the canonical contract envelope dict on 200/500/504 (the
        server ships envelope shape per §12.0 even on terminal errors).
        Raises ``ValueError`` for an unknown role, ``SwarmError`` on 400,
        ``SwarmBusy`` on 429, ``SwarmTimeout`` on local read timeout.

        No retries (§19 #7). The semaphore wraps the whole call so a slow
        long-poll genuinely backpressures the third concurrent caller.
        """
        if role not in _ALLOWED_ROLES:
            raise ValueError(
                f"swarm: unknown role {role!r}; must be one of "
                f"{sorted(_ALLOWED_ROLES)}"
            )
        server_role = _ROLE_MAP[role]
        deadline_s = (
            min(int(timeout_s), _SERVER_HARD_CAP_S)
            if timeout_s is not None
            else self._dispatch_max_wait_s
        )

        body: dict[str, Any] = {
            "task": task,
            "context_keys": list(context_keys or []),
            "extra": {
                "timeout_s": deadline_s,
                "max_iterations": int(max_iterations),
            },
        }

        with self._sem:
            try:
                resp = self._client.post(f"/subagent/{server_role}", json=body)
            except httpx.ReadTimeout as e:
                raise SwarmTimeout(
                    f"swarm: long-poll read timeout for role={role!r} "
                    f"after {deadline_s}s"
                ) from e
            except httpx.RequestError as e:
                # Connect errors / network issues — surface as SwarmError.
                raise SwarmError(f"swarm: HTTP error: {e}") from e

        status = resp.status_code
        if status == 429:
            raise SwarmBusy(
                f"swarm: 429 busy for role={role!r}: {_safe_text(resp)}"
            )
        if status == 400:
            raise SwarmError(
                f"swarm: 400 bad-request for role={role!r}: {_safe_text(resp)}"
            )
        # 200, 500, 504 — all ship envelope shape per §12.0. We coerce
        # whatever JSON came back into the contract shape; if the body
        # isn't JSON or doesn't carry ``success``, synthesize an err
        # envelope so the orchestrator never sees a raw HTTP error.
        if status in (200, 500, 504):
            return _coerce_envelope(resp, role=role, http_status=status)
        # Anything else — treat as a hard SwarmError (raise).
        raise SwarmError(
            f"swarm: unexpected status {status} for role={role!r}: "
            f"{_safe_text(resp)}"
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SwarmClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_text(resp: httpx.Response) -> str:
    """Return a short text snippet from a response without raising."""
    try:
        return resp.text[:200]
    except Exception:  # noqa: BLE001
        return ""


def _coerce_envelope(
    resp: httpx.Response, *, role: str, http_status: int
) -> dict:
    """Normalize the response body to the contract envelope shape.

    Honors only the contract keys; preserves whatever extras Swarm ships
    on the dict so the JSONL captures meta. If the body is not a JSON
    object or doesn't have ``success``, synthesize an err envelope —
    we never raise out of dispatch on a body shape we can't read.
    """
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = None
    if not isinstance(body, dict):
        return {
            "success": False,
            "summary": None,
            "deliverables": [],
            "context_keys_written": [],
            "sidechain_path": None,
            "error": (
                f"swarm: role={role!r} status={http_status} "
                f"non-envelope body"
            ),
        }
    out = dict(body)
    out.setdefault("success", False)
    out.setdefault("summary", None)
    out.setdefault("deliverables", [])
    out.setdefault("context_keys_written", [])
    out.setdefault("sidechain_path", None)
    out.setdefault("error", None)
    # Meta fields (e.g. execution_log) are preserved on the dict but the
    # invoker filters to contract keys before the LLM sees the result.
    return out
