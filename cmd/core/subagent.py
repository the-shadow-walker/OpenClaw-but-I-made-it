#!/usr/bin/env python3
"""
SubAgentInvoker — uniform delegation primitive.

The "snapshot → run sub-agent → merge clean results back" pattern that all
tool-driven sub-agent calls (gui_task, code_task, swarm_task, etc.) should
route through. Keeps the parent's ReAct context clean: only the sub-agent's
final summary + artifact list comes back, not its hundreds of intermediate
ReAct iterations.

Design contract:
  1. **Snapshot**: caller's pinned slots + files manifest get serialized to
     ~/.agent_bin/sessions/{sid}.json BEFORE the sub-agent runs.
  2. **Context bridge**: optional context_keys (list[str]) are pulled from the
     central shared_context board and passed verbatim to the sub-agent.
  3. **Run**: sub-agent runs in isolation. Its full ReAct trace is dumped to
     ~/.agent_bin/sidechains/{sid}.jsonl, NOT into the parent's history.
  4. **Merge**: SubAgentResult contains only the clean summary, files_created,
     and any artifacts. Caller merges those back into parent's _files_created
     and pins one "📥 SUBAGENT RESULT" message.
  5. **Recovery**: if sub-agent crashes or stop_event fires, snapshot can be
     re-applied via agent.restore_context(snapshot_path).

Targets supported:
  - "gui"            : runs guiagent.GUIAgent in-process
  - "cmd"            : runs OllamaCommandAgent in-process (used by GUI->CMD)
  - "swarm:engineer" : POSTs to http://localhost:5002/query (engineer mode)
  - "swarm:math"     : POSTs to http://localhost:5002/query (math mode)
  - "swarm:search"   : POSTs to http://localhost:5002/search (deep search)
"""

from __future__ import annotations

import os
import json
import time
import uuid
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


SESSIONS_DIR = os.path.expanduser("~/.agent_bin/sessions")
SIDECHAINS_DIR = os.path.expanduser("~/.agent_bin/sidechains")
DEFAULT_SWARM_BASE = os.getenv("SWARM_API_URL", "http://localhost:5002")


# ---------------------------------------------------------------- result --

@dataclass
class SubAgentResult:
    """Clean output of a sub-agent run — ONLY this comes back into parent context.

    parent_summary is the human-readable string the parent agent sees.
    files_created and artifacts are merged back into parent's manifest.
    sidechain_path points at the dumped trace for post-mortem debugging.
    """
    target: str
    success: bool
    parent_summary: str
    files_created: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path
    iterations_used: int = 0
    elapsed_ms: int = 0
    sidechain_path: str = ""
    snapshot_path: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------- main invoker --

class SubAgentInvoker:
    """Uniform sub-agent delegation. Construct with parent agent ref, then call run()."""

    SUPPORTED_TARGETS = {
        "gui", "cmd",
        "swarm:engineer", "swarm:math", "swarm:search",
    }

    def __init__(self, parent_agent: Any, agent_memory: Optional[Any] = None):
        """
        parent_agent : the OllamaCommandAgent (or compatible) calling this.
                       Must expose save_context(label) -> str path and
                       restore_context(path) -> None.
                       Optional: _files_created list, _update_pinned(key, msg).
        agent_memory : AgentMemory instance for shared_context lookups. Optional.
        """
        self.parent = parent_agent
        self.memory = agent_memory or getattr(parent_agent, "memory", None)
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        os.makedirs(SIDECHAINS_DIR, exist_ok=True)

    # ---------- public ----------

    def run(
        self,
        target: str,
        task: str,
        max_iterations: int = 20,
        context_keys: Optional[List[str]] = None,
        merge_files: bool = True,
        merge_pin: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> SubAgentResult:
        """Snapshot parent → run sub-agent → merge clean results back.

        target          : one of SUPPORTED_TARGETS
        task            : natural-language task description for the sub-agent
        max_iterations  : iteration budget (sub-agent specific)
        context_keys    : list of shared_context keys to fetch and pass through
        merge_files     : if True, sub-agent's files_created are appended to
                          parent._files_created (deduplicated).
        merge_pin       : if True, pin a "📥 SUBAGENT RESULT" slot on parent.
        extra           : target-specific options dict.
        """
        if target not in self.SUPPORTED_TARGETS:
            return SubAgentResult(
                target=target, success=False,
                parent_summary=f"Unknown sub-agent target '{target}'. "
                f"Valid: {sorted(self.SUPPORTED_TARGETS)}",
                error="unknown_target",
            )

        if not task or not task.strip():
            return SubAgentResult(
                target=target, success=False,
                parent_summary="task is required (non-empty string)",
                error="empty_task",
            )

        sid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        snapshot_path = self._snapshot_parent(target, sid, task)
        sidechain_path = os.path.join(SIDECHAINS_DIR, f"{sid}_{self._slug(target)}.jsonl")

        # Pull any requested context keys from the shared board so the sub-agent
        # has the exact facts the parent does (without the ReAct chatter).
        ctx_block = self._gather_context(context_keys or [])
        bridged_task = self._build_bridged_task(task, ctx_block)

        t0 = time.time()
        try:
            if target == "gui":
                result = self._run_gui(bridged_task, max_iterations, sidechain_path, extra or {})
            elif target == "cmd":
                result = self._run_cmd(bridged_task, max_iterations, sidechain_path, extra or {})
            elif target.startswith("swarm:"):
                mode = target.split(":", 1)[1]
                result = self._run_swarm(mode, bridged_task, sidechain_path, extra or {})
            else:
                # Defensive — should never fall here
                result = SubAgentResult(
                    target=target, success=False,
                    parent_summary=f"target {target} dispatch missing",
                    error="dispatch_missing",
                )
        except Exception as e:
            result = SubAgentResult(
                target=target, success=False,
                parent_summary=f"Sub-agent crashed: {type(e).__name__}: {e}",
                error=f"{type(e).__name__}: {e}",
            )

        result.elapsed_ms = int((time.time() - t0) * 1000)
        result.snapshot_path = snapshot_path
        result.sidechain_path = sidechain_path

        # Merge files back into parent manifest
        if merge_files and result.files_created:
            try:
                files_attr = getattr(self.parent, "_files_created", None)
                if isinstance(files_attr, list):
                    for f in result.files_created:
                        if f and f not in files_attr:
                            files_attr.append(f)
                    # Refresh the pinned manifest if the parent supports it
                    refresh = getattr(self.parent, "_refresh_file_manifest_pin", None)
                    if callable(refresh):
                        try:
                            refresh()
                        except Exception:
                            pass
            except Exception:
                pass

        # Pin a single clean result message so the parent agent sees the outcome
        # without the sub-agent's full ReAct chatter
        if merge_pin:
            try:
                update_pinned = getattr(self.parent, "_update_pinned", None)
                if callable(update_pinned):
                    icon = "✅" if result.success else "⚠️"
                    files_line = (
                        f"\nFiles: {', '.join(result.files_created[:8])}"
                        f"{' (+more)' if len(result.files_created) > 8 else ''}"
                        if result.files_created else ""
                    )
                    msg_text = (
                        f"📥 SUBAGENT RESULT [{target}] {icon}\n"
                        f"Task: {task[:140]}\n"
                        f"{result.parent_summary[:1200]}{files_line}\n"
                        f"(sidechain: {os.path.basename(sidechain_path)} | "
                        f"iters: {result.iterations_used} | {result.elapsed_ms}ms)"
                    )
                    update_pinned(
                        f"subagent_{target.replace(':', '_')}",
                        {"role": "system", "content": msg_text},
                    )
            except Exception:
                pass

        return result

    # ---------- internals ----------

    @staticmethod
    def _slug(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in s)[:30]

    def _snapshot_parent(self, target: str, sid: str, task: str) -> str:
        """Save parent's pinned + files state. Returns snapshot path."""
        save_cb = getattr(self.parent, "save_context", None)
        if callable(save_cb):
            try:
                label = f"pre_{self._slug(target)}_{sid}"
                path = save_cb(label)
                return path or ""
            except Exception as e:
                print(f"⚠️  SubAgentInvoker snapshot failed: {e}")
        return ""

    def _gather_context(self, keys: List[str]) -> str:
        if not keys or self.memory is None:
            return ""
        getter = getattr(self.memory, "get_context", None)
        if not callable(getter):
            return ""
        lines = []
        for k in keys:
            try:
                v = getter(k)
                if v is not None:
                    lines.append(f"  {k}: {v}")
            except Exception:
                pass
        if not lines:
            return ""
        return (
            "## Shared context (from central board)\n" + "\n".join(lines) + "\n\n"
        )

    @staticmethod
    def _build_bridged_task(task: str, ctx_block: str) -> str:
        if not ctx_block:
            return task
        return f"{ctx_block}## Your task\n{task}"

    # ---------- target dispatchers ----------

    def _run_gui(
        self, task: str, max_iter: int, sidechain_path: str, extra: Dict[str, Any]
    ) -> SubAgentResult:
        try:
            # Lazy import — gui_agent has heavy deps (xdotool etc.)
            from gui_agent import GUIAgent  # type: ignore
        except ImportError as e:
            return SubAgentResult(
                target="gui", success=False,
                parent_summary=f"GUI agent not available on this host: {e}",
                error="gui_unavailable",
            )

        agent = GUIAgent()
        try:
            result = agent.run(task=task, max_iterations=max_iter)
        except Exception as e:
            return SubAgentResult(
                target="gui", success=False,
                parent_summary=f"GUI agent crashed: {e}",
                error=f"{type(e).__name__}: {e}",
            )

        # Extract files created during GUI run
        files: List[str] = []
        try:
            inner = getattr(agent, "agent", None)
            trace = getattr(inner, "react_trace", []) if inner else []
            for ev in trace:
                if ev.get("tool") == "create_file":
                    res = ev.get("result")
                    if res and getattr(res, "success", False):
                        p = ev.get("args", {}).get("path", "")
                        if p:
                            files.append(p)
        except Exception:
            pass

        # Dump the GUI sidechain trace
        try:
            with open(sidechain_path, "w") as f:
                inner = getattr(agent, "agent", None)
                trace = getattr(inner, "react_trace", []) if inner else []
                for ev in trace:
                    f.write(json.dumps(ev, default=str) + "\n")
            # Zero out the inner trace so subsequent GUI calls start clean
            if inner is not None and hasattr(inner, "react_trace"):
                inner.react_trace = []
        except Exception:
            pass

        success = bool(result.get("success", False))
        summary = result.get("summary", result.get("finish_summary", ""))
        return SubAgentResult(
            target="gui", success=success,
            parent_summary=summary or ("GUI task completed" if success else "GUI task failed"),
            files_created=files,
            iterations_used=int(result.get("iterations_used", 0)),
        )

    def _run_cmd(
        self, task: str, max_iter: int, sidechain_path: str, extra: Dict[str, Any]
    ) -> SubAgentResult:
        """Run an OllamaCommandAgent as a subordinate (used when GUI/swarm need code)."""
        try:
            from ollama_agent_core import OllamaCommandAgent  # type: ignore
        except ImportError as e:
            return SubAgentResult(
                target="cmd", success=False,
                parent_summary=f"OllamaCommandAgent unavailable: {e}",
                error="cmd_unavailable",
            )

        sub = OllamaCommandAgent()
        sub.max_react_iterations = max_iter
        try:
            sub.run(task)
        except Exception as e:
            return SubAgentResult(
                target="cmd", success=False,
                parent_summary=f"CMD subordinate crashed: {e}",
                error=f"{type(e).__name__}: {e}",
            )

        # Pull files from the subordinate's manifest
        files = list(getattr(sub, "_files_created", []) or [])

        # Dump trace
        try:
            with open(sidechain_path, "w") as f:
                for ev in getattr(sub, "react_trace", []) or []:
                    f.write(json.dumps(ev, default=str) + "\n")
        except Exception:
            pass

        # Use last execution log entry as summary if available
        log = getattr(sub, "execution_log", []) or []
        summary = ""
        if log:
            last = log[-1]
            summary = (
                last.get("finish_summary")
                or last.get("summary")
                or "CMD subordinate completed"
            )
        else:
            summary = "CMD subordinate completed"

        # Detect success: did it finish() with success=true?
        success = False
        for ev in reversed(getattr(sub, "react_trace", []) or []):
            if ev.get("tool") == "finish":
                res = ev.get("result")
                if res:
                    success = bool(getattr(res, "success", False))
                    break

        return SubAgentResult(
            target="cmd", success=success,
            parent_summary=summary,
            files_created=files,
            iterations_used=len(getattr(sub, "react_trace", []) or []),
        )

    def _run_swarm(
        self, mode: str, task: str, sidechain_path: str, extra: Dict[str, Any]
    ) -> SubAgentResult:
        """POST to swarm API. mode: engineer|math|search."""
        target = f"swarm:{mode}"
        base = extra.get("swarm_url", DEFAULT_SWARM_BASE).rstrip("/")
        timeout = int(extra.get("timeout", 300))

        if mode == "search":
            url = f"{base}/api/search"
            payload = {"query": task, "max_results": int(extra.get("max_results", 8))}
        else:
            url = f"{base}/query"
            payload = {
                "question": task,
                "mode": mode,
                "max_iterations": int(extra.get("max_iterations", 30)),
            }

        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-X", "POST", url,
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload),
                    "--max-time", str(timeout),
                ],
                capture_output=True, text=True, timeout=timeout + 10,
            )
        except subprocess.TimeoutExpired:
            return SubAgentResult(
                target=target, success=False,
                parent_summary=f"Swarm {mode} call timed out after {timeout}s",
                error="timeout",
            )
        except Exception as e:
            return SubAgentResult(
                target=target, success=False,
                parent_summary=f"Swarm {mode} call failed: {e}",
                error=f"{type(e).__name__}: {e}",
            )

        if result.returncode != 0:
            return SubAgentResult(
                target=target, success=False,
                parent_summary=f"Swarm {mode} curl exit {result.returncode}",
                error=(result.stderr or "")[:200],
            )

        # Dump raw response to sidechain
        try:
            with open(sidechain_path, "w") as f:
                f.write(result.stdout)
        except Exception:
            pass

        # Parse swarm response
        try:
            data = json.loads(result.stdout)
        except Exception:
            return SubAgentResult(
                target=target, success=False,
                parent_summary=f"Swarm {mode} returned non-JSON ({len(result.stdout)} bytes)",
                error="bad_json",
            )

        # Different swarm endpoints return different shapes — normalize
        success = bool(data.get("success", True)) and "error" not in data
        summary = (
            data.get("answer")
            or data.get("result")
            or data.get("summary")
            or data.get("response")
            or json.dumps(data)[:500]
        )
        files = data.get("files_created") or data.get("artifacts") or []
        if isinstance(files, dict):
            files = list(files.values())
        if not isinstance(files, list):
            files = []
        # Filter to strings only
        files = [str(f) for f in files if f]

        return SubAgentResult(
            target=target, success=success,
            parent_summary=str(summary)[:4000],
            files_created=files,
            iterations_used=int(data.get("iterations_used", 0)),
        )


# ----------------------------------------------------------- module CLI --

if __name__ == "__main__":
    # Smoke test (no parent — just exercise the dataclass)
    r = SubAgentResult(
        target="gui", success=True,
        parent_summary="opened firefox and navigated to example.com",
        files_created=["/tmp/screenshot.png"],
        iterations_used=4, elapsed_ms=1234,
    )
    print(json.dumps(r.to_dict(), indent=2))
