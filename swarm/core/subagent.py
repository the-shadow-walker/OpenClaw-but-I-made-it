#!/usr/bin/env python3
"""
SubAgentInvoker — swarm-side mirror of CMD's primitive.

Lets a swarm agent delegate to another agent via a uniform interface:
  - target = "cmd"             → POST to CMD's /api/v1/execute (HTTP)
  - target = "swarm:math"      → in-process call into subagent_handler
  - target = "swarm:engineer"  → in-process call into subagent_handler
  - target = "swarm:deep_search" → in-process call into subagent_handler

The parent's transcript is NOT polluted by the sub-agent's full ReAct trace —
only the SubAgentResult dict comes back. Full traces land in sidechain JSONLs.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

SESSIONS_DIR = os.path.expanduser("~/.agent_bin/sessions")
SIDECHAINS_DIR = os.path.expanduser("~/.agent_bin/sidechains")
RESULTS_DIR = os.path.expanduser("~/.agent_bin/results")
DEFAULT_CMD_URL = os.getenv("CMD_API_URL", "http://localhost:5000")
DEFAULT_SWARM_BASE = os.getenv("SWARM_API_URL", "http://localhost:5002")


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class SubAgentResult:
    target: str
    success: bool
    summary: str
    deliverables: List[str] = field(default_factory=list)
    context_keys_written: List[str] = field(default_factory=list)
    sidechain_path: str = ""
    parent_summary: str = ""              # back-compat alias for `summary`
    files_created: List[str] = field(default_factory=list)
    iterations_used: int = 0
    elapsed_ms: int = 0
    snapshot_path: str = ""
    error: Optional[str] = None

    def __post_init__(self) -> None:
        # Keep parent_summary in sync with summary for back-compat
        if not self.parent_summary and self.summary:
            self.parent_summary = self.summary

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Invoker ──────────────────────────────────────────────────────────────────

class SubAgentInvoker:
    """Uniform delegation primitive. Swarm version."""

    SUPPORTED_TARGETS = {
        "cmd",
        "swarm:engineer",
        "swarm:math",
        "swarm:deep_search",
    }

    def __init__(self, parent_agent: Any = None, agent_memory: Any = None):
        self.parent = parent_agent
        self.memory = agent_memory or getattr(parent_agent, "memory", None)
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(SIDECHAINS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        target: str,
        task: str,
        *,
        max_iterations: int = 20,
        context_keys: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        if target not in self.SUPPORTED_TARGETS:
            return SubAgentResult(
                target=target,
                success=False,
                summary=f"Unknown sub-agent target {target!r}. Valid: {sorted(self.SUPPORTED_TARGETS)}",
                error="unknown_target",
            )
        if not task or not task.strip():
            return SubAgentResult(
                target=target,
                success=False,
                summary="task is required (non-empty string)",
                error="empty_task",
            )

        sid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        snapshot_path = self._snapshot_parent(target, sid)
        ctx_block = self._gather_context(context_keys or [])
        bridged_task = self._build_bridged_task(task, ctx_block)

        t0 = time.time()
        try:
            if target == "cmd":
                result = self._run_cmd(bridged_task, max_iterations, extra or {})
            else:
                role = target.split(":", 1)[1]
                result = self._run_swarm_role(role, bridged_task, max_iterations, context_keys or [], extra or {})
        except Exception as e:
            result = SubAgentResult(
                target=target,
                success=False,
                summary=f"Sub-agent crashed: {type(e).__name__}: {e}",
                error=f"{type(e).__name__}: {e}",
            )

        result.elapsed_ms = int((time.time() - t0) * 1000)
        result.snapshot_path = snapshot_path
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _slug(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in s)[:30] or "x"

    def _snapshot_parent(self, target: str, sid: str) -> str:
        save_cb = getattr(self.parent, "save_context", None)
        if not callable(save_cb):
            return ""
        try:
            label = f"pre_{self._slug(target)}_{sid}"
            return save_cb(label) or ""
        except Exception as e:
            print(f"⚠️ SubAgentInvoker snapshot failed: {e}")
            return ""

    def _gather_context(self, keys: List[str]) -> str:
        if not keys or self.memory is None:
            return ""
        getter = getattr(self.memory, "get_context", None)
        if not callable(getter):
            return ""
        lines: List[str] = []
        for k in keys:
            try:
                v = getter(k)
                if v is not None:
                    lines.append(f"  {k}: {v}")
            except Exception:
                pass
        if not lines:
            return ""
        return "## Shared context (from central board)\n" + "\n".join(lines) + "\n\n"

    @staticmethod
    def _build_bridged_task(task: str, ctx_block: str) -> str:
        if not ctx_block:
            return task
        return f"{ctx_block}## Your task\n{task}"

    # ── Target dispatchers ────────────────────────────────────────────────────

    def _run_cmd(
        self,
        task: str,
        max_iter: int,
        extra: Dict[str, Any],
    ) -> SubAgentResult:
        """POST to CMD's /api/v1/execute and poll /api/v1/jobs/<id> until done."""
        base = extra.get("cmd_url", DEFAULT_CMD_URL).rstrip("/")
        timeout_s = int(extra.get("timeout_s", 1800))
        try:
            r = requests.post(
                f"{base}/api/v1/execute",
                json={"task": task, "max_iterations": max_iter},
                timeout=10,
            )
        except Exception as e:
            return SubAgentResult(
                target="cmd",
                success=False,
                summary=f"CMD execute POST failed: {e}",
                error=f"{type(e).__name__}: {e}",
            )
        if not r.ok:
            return SubAgentResult(
                target="cmd",
                success=False,
                summary=f"CMD execute returned {r.status_code}: {r.text[:200]}",
                error="cmd_http_error",
            )
        try:
            payload = r.json()
        except Exception:
            return SubAgentResult(
                target="cmd",
                success=False,
                summary="CMD execute returned non-JSON",
                error="bad_json",
            )

        job_id = payload.get("job_id")
        if not job_id:
            # Sync response, treat payload as result
            return SubAgentResult(
                target="cmd",
                success=bool(payload.get("success", True)),
                summary=str(payload.get("summary", payload.get("result", "")))[:4000],
                deliverables=list(payload.get("deliverables", []) or []),
                files_created=list(payload.get("files_created", []) or []),
            )

        # Poll for completion
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                jr = requests.get(f"{base}/api/v1/jobs/{job_id}", timeout=10)
                if jr.ok:
                    jp = jr.json()
                    status = jp.get("status", "")
                    if status in ("completed", "succeeded"):
                        return SubAgentResult(
                            target="cmd",
                            success=True,
                            summary=str(jp.get("summary", jp.get("result", "")))[:4000],
                            deliverables=list(jp.get("deliverables", []) or []),
                            files_created=list(jp.get("files_created", []) or []),
                            iterations_used=int(jp.get("iterations_used", 0)),
                        )
                    if status in ("failed", "error", "cancelled"):
                        return SubAgentResult(
                            target="cmd",
                            success=False,
                            summary=str(jp.get("summary", jp.get("error", "CMD job failed")))[:4000],
                            error=str(jp.get("error", "")) or "cmd_job_failed",
                        )
            except Exception:
                pass
            time.sleep(2)
        return SubAgentResult(
            target="cmd",
            success=False,
            summary=f"CMD job {job_id} long-poll timeout after {timeout_s}s",
            error="timeout",
        )

    def _run_swarm_role(
        self,
        role: str,
        task: str,
        max_iter: int,
        context_keys: List[str],
        extra: Dict[str, Any],
    ) -> SubAgentResult:
        """In-process delegation to swarm subagent_handler."""
        try:
            from subagent_handler import run_role_sync  # type: ignore
        except ImportError as e:
            return SubAgentResult(
                target=f"swarm:{role}",
                success=False,
                summary=f"subagent_handler not importable: {e}",
                error="handler_missing",
            )
        try:
            return run_role_sync(
                role=role,
                task=task,
                max_iterations=max_iter,
                context_keys=context_keys,
                extra=extra,
            )
        except Exception as e:
            return SubAgentResult(
                target=f"swarm:{role}",
                success=False,
                summary=f"swarm:{role} crashed: {type(e).__name__}: {e}",
                error=f"{type(e).__name__}: {e}",
            )


# ── module CLI smoke test ────────────────────────────────────────────────────

if __name__ == "__main__":
    r = SubAgentResult(
        target="swarm:math",
        success=True,
        summary="solved x=2,3",
        deliverables=["/home/Grindlewalt/.agent_bin/results/math_test.md"],
        context_keys_written=["swarm_math_test_result"],
        iterations_used=4,
        elapsed_ms=1234,
    )
    print(json.dumps(r.to_dict(), indent=2))
