"""Delegation invoker — snapshot, dispatch, merge (BUILD_SPEC §10).

Three top-level functions, called by the ``delegate`` tool handler:

* :func:`snapshot` — write a small JSON file under ``shared_board/sessions/``
  capturing the conversation's current state pre-delegation. P7's payload
  is honest: ``conv_id``, ``label``, ``ts``, ``transcript_path``, and a
  line offset into the JSONL transcript. **No** ``messages``
  thread-through — the in-memory ``messages`` list isn't accessible at
  the tool-handler boundary in P7. Restoration (not wired in P7) reads
  the JSONL up to the offset.

* :func:`dispatch` — top-level entry from the delegate tool. Routes by
  ``target``: ``cmd:quick`` → ``cmd_client.quick``; ``cmd:react`` →
  ``cmd_client.execute``; everything else (``cmd:chain``, ``cmd:gui``,
  ``cmd:blue``, ``swarm:*``) returns a "not implemented in P7" error
  envelope. NEVER raises. CMD timeouts / HTTP errors land in
  ``envelope.error``. The envelope returned is the LLM-facing contract
  shape (``success``, ``summary``, ``deliverables``,
  ``context_keys_written``, ``sidechain_path``, ``error``); meta fields
  like ``target`` / ``snapshot_path`` / ``ms_elapsed`` / ``master_mode``
  live in the JSONL ``delegation_envelope`` event only.

* :func:`merge` — append a ``delegation_envelope`` event to the
  conversation's JSONL transcript with the full payload (envelope +
  meta).

Anti-patterns explicitly avoided:

* **§19 #3** Reading ``sidechain_path`` content or ``execution_log``
  into the JSONL. Path strings only; the LLM never sees the ReAct
  internals.
* **§10.4 binding** Consuming ``execution_log`` from the specialist's
  status response. Drop on the floor.
* **§19 #9** Inlining 50KB of context in ``task``. Caller publishes
  context keys; ``task`` references them by name (``context_keys=...``
  on ``execute``).
* **§19 #11** Naming things "session". We use ``conversation`` for the
  chat thread, ``snapshot`` for the pre-delegation save. The shared
  directory at ``~/.agent_bin/sessions/`` is named for CMD's
  convention; we don't fight it, but we don't propagate the term into
  Jarvis types.
* Raising out of ``dispatch``. Every failure is an envelope.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from jarvis.clients.cmd import CMDClient, CMDError, CMDTimeout
from jarvis.clients.swarm import SwarmBusy, SwarmClient, SwarmError, SwarmTimeout
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import Conversation
from jarvis.memory.workspace import WorkspacePaths

logger = logging.getLogger(__name__)

__all__ = ["snapshot", "restore_from_snapshot", "dispatch", "merge"]


_LLM_FACING_KEYS = {
    "success",
    "summary",
    "deliverables",
    "context_keys_written",
    "sidechain_path",
    "error",
}


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def snapshot(
    *,
    conversation: Conversation,
    label: str,
    shared_board: Path,
) -> Path:
    """Write a pre-delegation snapshot file to ``shared_board/sessions/``.

    File name: ``jarvis_{conv_id}_{label}_{ts}.context``. Payload keys:

    * ``conv_id``
    * ``label`` (e.g. ``"pre_cmd_react"``)
    * ``ts`` (epoch seconds, float)
    * ``transcript_path`` (absolute string)
    * ``transcript_offset_lines`` (current line count, used to bound
      restoration reads)

    Returns the absolute path to the file.
    """
    sessions_dir = shared_board / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = conversation.transcript_path
    line_count = _count_jsonl_lines(transcript_path)

    ts = time.time()
    ts_str = f"{int(ts)}"
    file_path = sessions_dir / f"jarvis_{conversation.conv_id}_{label}_{ts_str}.context"

    payload = {
        "conv_id": conversation.conv_id,
        "label": label,
        "ts": ts,
        "transcript_path": str(transcript_path),
        "transcript_offset_lines": line_count,
    }
    file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return file_path


def restore_from_snapshot(path: Path) -> dict:
    """Load and return the snapshot payload. Not wired in P7."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _count_jsonl_lines(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        with p.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch(
    *,
    target: str,
    task: str,
    conversation: Conversation,
    paths: WorkspacePaths,
    shared_board: Path,
    cmd_client: CMDClient | None,
    arbiter: RoleArbiter,
    swarm_client: SwarmClient | None = None,
    context_keys: list[str] | None = None,
    snapshot_label: str | None = None,
) -> dict:
    """Top-level delegation entry. Returns the LLM-facing envelope dict.

    Never raises. Every error path lands in ``envelope.error``. Internal
    metadata (target, snapshot path, elapsed ms, master-mode flag) is
    written to the JSONL ``delegation_envelope`` event but **filtered
    out** of the value returned to the caller — the LLM does not need to
    reason about whether arbitration kicked in.
    """
    started = time.monotonic()
    label = snapshot_label or _label_for_target(target)
    # P10: claim master role on first cmd:code / cmd:gui (first-write-wins).
    # cmd:react / cmd:quick / swarm:* never claim or consult — they're
    # exempt from arbitration.
    if target in ("cmd:code", "cmd:gui"):
        arbiter.claim(
            conversation.conv_id, "code" if target == "cmd:code" else "gui"
        )
    subordinate = arbiter.is_subordinate(conversation.conv_id, target)
    # Single source of truth for the master's identity — derive from the
    # arbiter, never hardcode "code"/"gui" in the dispatch branches. Keeps
    # the body string consistent if is_subordinate() semantics evolve.
    master_for_body = (
        arbiter.master_for(conversation.conv_id) if subordinate else None
    )

    snap_path: Path | None = None
    try:
        snap_path = snapshot(
            conversation=conversation, label=label, shared_board=shared_board
        )
    except Exception:  # noqa: BLE001
        logger.exception("invoker: snapshot failed (continuing without)")

    # P8 (§16-A): emit ``delegation_snapshot`` JSONL event so the
    # rocket-sim acceptance assertion ("3 snapshot + 3 envelope events")
    # can count them. Best-effort — a failed append must not block
    # dispatch.
    if snap_path is not None:
        try:
            conversation.append("delegation_snapshot", {
                "target": target,
                "label": label,
                "snapshot_path": str(snap_path),
            })
        except Exception:  # noqa: BLE001
            logger.exception("invoker: delegation_snapshot append failed")

    envelope: dict
    if target == "cmd:quick":
        if cmd_client is None:
            envelope = _err_envelope(
                "delegation: no CMD client wired (degraded mode)", target=target
            )
        else:
            envelope = _safe_quick(cmd_client, task)
    elif target == "cmd:react":
        if cmd_client is None:
            envelope = _err_envelope(
                "delegation: no CMD client wired (degraded mode)", target=target
            )
        else:
            envelope = _safe_execute(cmd_client, task, context_keys=context_keys)
    elif target == "cmd:code":
        if cmd_client is None:
            envelope = _err_envelope(
                "delegation: no CMD client wired (degraded mode)", target=target
            )
        else:
            envelope = _safe_execute(
                cmd_client, task, context_keys=context_keys,
                mode="code",
                master_mode=master_for_body,
                target_label="cmd:code",
            )
    elif target == "cmd:gui":
        if cmd_client is None:
            envelope = _err_envelope(
                "delegation: no CMD client wired (degraded mode)", target=target
            )
        else:
            envelope = _safe_execute(
                cmd_client, task, context_keys=context_keys,
                mode="gui",
                master_mode=master_for_body,
                target_label="cmd:gui",
            )
    elif target.startswith("swarm:"):
        if swarm_client is None:
            envelope = _err_envelope(
                "delegation: no Swarm client wired (degraded mode)", target=target
            )
        else:
            envelope = _safe_swarm(
                swarm_client, target, task, context_keys=context_keys
            )
    elif target in ("cmd:chain", "cmd:blue"):
        envelope = _err_envelope(
            f"delegation target {target!r} not implemented in P10", target=target
        )
    else:
        envelope = _err_envelope(
            f"delegation target {target!r} unknown", target=target
        )

    elapsed_ms = int((time.monotonic() - started) * 1000)

    try:
        merge(
            envelope=envelope,
            conversation=conversation,
            target=target,
            task=task,
            snapshot_path=snap_path,
            ms_elapsed=elapsed_ms,
            master_mode=subordinate,
        )
    except Exception:  # noqa: BLE001
        logger.exception("invoker: merge failed (envelope still returned)")

    return _filter_to_contract(envelope)


def _safe_quick(cmd_client: CMDClient, task: str) -> dict:
    try:
        raw = cmd_client.quick(question=task)
    except CMDTimeout as e:
        return _err_envelope(f"cmd:quick timed out: {e}", target="cmd:quick")
    except CMDError as e:
        return _err_envelope(str(e), target="cmd:quick")
    except Exception as e:  # noqa: BLE001
        logger.exception("invoker: cmd:quick raised unexpectedly")
        return _err_envelope(f"cmd:quick error: {e}", target="cmd:quick")
    return _quick_to_envelope(raw)


def _safe_execute(
    cmd_client: CMDClient,
    task: str,
    *,
    context_keys: list[str] | None,
    mode: str | None = None,
    master_mode: str | None = None,
    target_label: str = "cmd:react",
) -> dict:
    """Wrap ``CMDClient.execute`` for the cmd:react / cmd:code / cmd:gui paths.

    ``target_label`` only shapes error-envelope messages — the wire body
    is driven by ``mode`` (``"code"`` / ``"gui"`` for cmd:code/cmd:gui;
    ``None`` for cmd:react). ``master_mode`` is set IFF this dispatch is
    subordinate (cross-mode); ``None`` otherwise.
    """
    try:
        env = cmd_client.execute(
            task,
            context_keys=context_keys,
            mode=mode,
            master_mode=master_mode,
        )
    except CMDTimeout as e:
        return _err_envelope(f"{target_label} timed out: {e}", target=target_label)
    except CMDError as e:
        return _err_envelope(str(e), target=target_label)
    except Exception as e:  # noqa: BLE001
        logger.exception("invoker: %s raised unexpectedly", target_label)
        return _err_envelope(f"{target_label} error: {e}", target=target_label)
    return _normalise_envelope(env)


def _safe_swarm(
    swarm_client: SwarmClient,
    target: str,
    task: str,
    *,
    context_keys: list[str] | None,
) -> dict:
    """Route a ``swarm:<role>`` target through the SwarmClient.

    Maps:

    * ``swarm:math`` → role ``math``
    * ``swarm:engineer`` → role ``engineer``
    * ``swarm:research`` → role ``research`` (server-side ``deep_search``)
    * ``swarm:deep_search`` → also accepted (server-side ``deep_search``)

    Mirrors :func:`_safe_execute`: never raises; every failure path lands
    in ``envelope.error``.
    """
    role = target.split(":", 1)[1] if ":" in target else target
    try:
        env = swarm_client.dispatch(role, task, context_keys=context_keys)
    except SwarmTimeout as e:
        return _err_envelope(f"{target} timed out: {e}", target=target)
    except SwarmBusy as e:
        return _err_envelope(f"{target} busy: {e}", target=target)
    except SwarmError as e:
        return _err_envelope(f"{target}: {e}", target=target)
    except ValueError as e:
        # Unknown role.
        return _err_envelope(f"{target}: {e}", target=target)
    except Exception as e:  # noqa: BLE001
        logger.exception("invoker: %s raised unexpectedly", target)
        return _err_envelope(f"{target} error: {e}", target=target)
    return _normalise_envelope(env)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge(
    *,
    envelope: dict,
    conversation: Conversation,
    target: str,
    task: str,
    snapshot_path: Path | None,
    ms_elapsed: int,
    master_mode: bool,
) -> None:
    """Append a ``delegation_envelope`` event to the JSONL transcript."""
    payload = {
        "target": target,
        "task": _safe_truncate(task, 500),
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "envelope": envelope,
        "ms_elapsed": ms_elapsed,
        "master_mode": master_mode,
    }
    conversation.append("delegation_envelope", payload)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_for_target(target: str) -> str:
    """Snapshot label, e.g. ``cmd:react`` → ``pre_cmd_react``."""
    sanitized = target.replace(":", "_").replace("/", "_")
    return f"pre_{sanitized}"


def _err_envelope(message: str, *, target: str) -> dict:
    return {
        "success": False,
        "summary": None,
        "deliverables": [],
        "context_keys_written": [],
        "sidechain_path": None,
        "error": message,
    }


def _quick_to_envelope(quick_response: dict) -> dict:
    """Map ``/quick``'s sync response to the canonical envelope shape.

    ``success`` ties to ``returncode == 0`` — a silent non-zero exit is
    still a failure. ``stdout[:1000]`` becomes the summary; ``stderr``
    or a synthesised ``"exit N"`` becomes the error when nonzero.
    """
    rc_raw = quick_response.get("returncode")
    rc = int(rc_raw) if rc_raw is not None else 0
    stdout = (quick_response.get("stdout") or "")
    stderr = (quick_response.get("stderr") or "")
    success = rc == 0
    error: str | None = None if success else (stderr.strip() or f"exit {rc}")
    summary = stdout[:1000] if stdout else None
    return {
        "success": success,
        "summary": summary,
        "deliverables": [],
        "context_keys_written": [],
        "sidechain_path": None,
        "error": error,
    }


def _normalise_envelope(env: Any) -> dict:
    """Coerce the CMD envelope into the canonical contract shape.

    Missing keys are filled with safe defaults; unknown extras are
    preserved on the dict so the JSONL captures everything CMD shipped.
    """
    if not isinstance(env, dict):
        return _err_envelope(f"cmd: unexpected envelope shape: {type(env).__name__}",
                             target="cmd:react")
    result = dict(env)
    result.setdefault("success", False)
    result.setdefault("summary", None)
    result.setdefault("deliverables", [])
    result.setdefault("context_keys_written", [])
    result.setdefault("sidechain_path", None)
    result.setdefault("error", None)
    return result


def _filter_to_contract(envelope: dict) -> dict:
    """Strip non-contract keys (meta lives in the JSONL event, not LLM-facing)."""
    return {k: envelope.get(k) for k in _LLM_FACING_KEYS}


def _safe_truncate(s: str, n: int) -> str:
    """UTF-8-safe truncate to ``n`` bytes. Mid-codepoint cuts break JSON parsers."""
    if s is None:
        return ""
    encoded = s.encode("utf-8")
    if len(encoded) <= n:
        return s
    return encoded[:n].decode("utf-8", errors="ignore")
