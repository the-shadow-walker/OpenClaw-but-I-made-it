#!/usr/bin/env python3
"""
task_chain.py — Multi-agent task chain system.

Classes:
  HandoffExtractor       — structured JSON handoff between sub-tasks (Phases 1 & 5)
  AcceptanceCriteriaRunner — shell-command acceptance gate (Phase 3)
  TaskDecomposer         — breaks a goal into scoped sub-tasks (Phase 4)
  SubtaskReplanner       — adjusts next sub-task based on previous handoff (Phase 2)
  TaskChain              — persists chain state to ~/.agent_bin/chains/<id>.json (Phase 6)
"""

import json
import os
import re as _re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ollama_agent_core import CommandSafetyValidator

CHAINS_DIR = os.path.expanduser("~/.agent_bin/chains")
INCOMPLETE_TASK_PATH = os.path.expanduser("~/.agent_bin/last_incomplete_task.json")
SIDECHAIN_DIR = os.path.expanduser("~/.agent_bin/sidechains")

# Common dev ports to clean up between chain phases
_CLEANUP_PORTS = [5000, 8000, 3000, 8080, 8443]

# ---------------------------------------------------------------------------
# ROLES — four specialist subagent configurations
# ---------------------------------------------------------------------------

ROLES: Dict[str, Dict] = {
    "planner": {
        "tools": {"read_file", "create_file", "web_search", "memory_lookup",
                  "validate_arch", "write_plan", "finish"},
        "system_prefix": (
            "You are the PLANNER. Read existing code/docs and write the architecture contract.\n"
            "Do NOT write implementation code. Do NOT run commands.\n"
            "If AGENT_ARCH_JSON is in force, emit DOCS/ARCH.json — a structured JSON document\n"
            "with keys: routes[], models[], ports[], files[], dependencies[].\n"
            "Each route MUST have method, path, handler, request_schema, response_schema.\n"
            "Each model MUST have name, fields:[{name,type}], relationships[].\n"
            "After writing, call validate_arch {\"path\": \"DOCS/ARCH.json\"} before finish().\n"
            "Tester role will auto-generate test stubs from this contract — do NOT write tests.\n"
            "Then finish() with files_created listing the contract file path."
        ),
        "first_action": "read_file",
        "single_minion": True,   # skip micro-task decomposition
    },
    "builder": {
        "tools": {"read_file", "create_file", "patch_file", "validate_arch",
                  "get_deps", "finish", "write_plan"},
        "system_prefix": (
            "You are the BUILDER. Write and modify code files only.\n"
            "Do NOT run servers or execute shell commands.\n"
            "Your FIRST action MUST be write_plan — write a full markdown plan covering:\n"
            "  ## Architecture, ## Files (- [ ] /path — description), ## Dependencies\n"
            "After each file is written, re-call write_plan with that item checked (- [x]).\n"
            "You MAY call validate_arch to re-check the contract at any time (idempotent).\n"
            "finish() only when ALL [ ] items are checked. List EVERY file in files_created."
        ),
        "first_action": "write_plan",
        "single_minion": False,
    },
    "tester": {
        "tools": {"read_file", "execute_command", "finish"},
        "system_prefix": (
            "You are the TESTER. Run verification checks only — do NOT modify source files.\n"
            "Run: syntax checks (python -m py_compile *.py), import checks, unit tests.\n"
            "finish() with a test_results summary. Set success=false if any check fails."
        ),
        "first_action": "execute_command",
        "single_minion": True,
    },
    "commander": {
        "tools": {"read_file", "execute_command", "manage_server", "finish"},
        "system_prefix": (
            "You are the COMMANDER. Handle infrastructure: install deps, migrate db,\n"
            "start services, configure ports. Do NOT write application code.\n"
            "finish() with a summary of services_running and ports_open."
        ),
        "first_action": "execute_command",
        "single_minion": False,
    },
    # NOTE — see _build_tool_restrictions_block below; the tools+first_action
    # in each role above are the SINGLE source of truth for the prompt block.
    # Internal-only role — filtered out of TaskDecomposer prompt; only runs as an
    # inline gate from SubtaskOrchestrator._run_inline_reconciler.
    "reconciler": {
        "tools": {"read_file", "patch_file", "create_file", "validate_arch",
                  "get_deps", "execute_command", "write_plan", "finish"},
        "system_prefix": (
            "You are the RECONCILER. Close drift between ARCH.json and the codebase.\n"
            "You may: read any file, patch/create files to align code with ARCH, or\n"
            "rewrite ARCH.json when code has intentionally evolved past the contract.\n"
            "A classification of findings is provided — obey the prescribed direction\n"
            "(patch_code vs update_arch). Do NOT introduce new features. Run\n"
            "validate_arch after writing ARCH.json. finish() with summary including\n"
            "'PATCHED: N' and 'ARCH_UPDATED: True/False'."
        ),
        "first_action": "validate_arch",
        "single_minion": True,
        "internal_only": True,
    },
}


# ---------------------------------------------------------------------------
# _build_tool_restrictions_block — single source of truth for prompt tool list
# ---------------------------------------------------------------------------

def _build_tool_restrictions_block(role_tools, first_action: str = "") -> str:
    """Generate the TOOL RESTRICTIONS block from a role's whitelist + TOOL_SCHEMAS.

    Replaces the hardcoded "YOUR ONLY TOOLS" prose that drifted from ROLES[role]
    over time (e.g. advertised write_plan when it wasn't registered, omitted
    validate_arch when it was). Walks TOOL_SCHEMAS in deterministic order so
    every minion sees a consistent block.
    """
    try:
        from ollama_agent_core import TOOL_SCHEMAS
    except Exception:
        TOOL_SCHEMAS = {}
    role_tool_set = set(role_tools or [])
    allowed = [t for t in TOOL_SCHEMAS if t in role_tool_set]
    blocked = [t for t in TOOL_SCHEMAS if t not in role_tool_set]
    lines = ["YOUR ONLY TOOLS (others are HARD-BLOCKED and will error):"]
    for t in allowed:
        lines.append(f"  ✅ {t}")
    for t in blocked:
        lines.append(f"  ❌ {t} — BLOCKED. Do NOT attempt it.")
    if first_action:
        lines.append(f"\nCRITICAL: Your FIRST tool call MUST be {first_action}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HandoffResult — structured bidirectional handoff between subagents
# ---------------------------------------------------------------------------

@dataclass
class HandoffResult:
    """Structured result returned from SubtaskOrchestrator to the chain parent."""
    success: bool
    summary: str
    role: str                          # which role produced this
    files_created: List[str] = field(default_factory=list)   # verified to exist on disk
    files_modified: List[str] = field(default_factory=list)
    services_running: List[str] = field(default_factory=list)  # "name:port" strings
    ports_open: List[int] = field(default_factory=list)
    test_results: Dict = field(default_factory=dict)           # {"passed": N, "failed": N, "errors": [...]}
    pinned_facts: Dict = field(default_factory=dict)           # key→value carried to all future subtasks
    next_subtask_hints: str = ""

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "summary": self.summary,
            "role": self.role,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "services_running": self.services_running,
            "ports_open": self.ports_open,
            "test_results": self.test_results,
            "pinned_facts": self.pinned_facts,
            "next_subtask_hints": self.next_subtask_hints,
        }


# ---------------------------------------------------------------------------
# cleanup_between_phases
# ---------------------------------------------------------------------------

def cleanup_between_phases() -> None:
    """Kill any processes occupying common dev ports. Safe no-op if nothing is listening."""
    for port in _CLEANUP_PORTS:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
        )  # ignore errors — port may not be in use
    time.sleep(1)  # let processes die


# ---------------------------------------------------------------------------
# ImplementationArtifact
# ---------------------------------------------------------------------------

@dataclass
class ImplementationArtifact:
    """Structured record of what a SubtaskOrchestrator phase produced."""
    subtask_index: int
    subtask_instruction: str
    status: str                          # "completed" | "partial" | "failed"
    summary: str                         # one paragraph prose
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    services_running: List[Dict] = field(default_factory=list)   # [{name, port, pid}]
    credentials: Dict[str, str] = field(default_factory=dict)    # {db_url: ..., secret_key: ...}
    micro_task_reports: List[Dict] = field(default_factory=list) # raw Self-Correction Reports
    notes: List[str] = field(default_factory=list)               # e.g. "route expects 'email'"

    @property
    def iterations_used(self) -> int:
        """Total iterations consumed across all micro-tasks in this phase."""
        return sum(int(r.get("iterations_used", 0) or 0) for r in self.micro_task_reports)

    @property
    def failure_count(self) -> int:
        """Number of micro-tasks that did not succeed."""
        return sum(1 for r in self.micro_task_reports if not r.get("success"))

    def to_dict(self) -> Dict:
        return {
            "subtask_index": self.subtask_index,
            "subtask_instruction": self.subtask_instruction,
            "status": self.status,
            "summary": self.summary,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "services_running": self.services_running,
            "credentials": self.credentials,
            "micro_task_reports": self.micro_task_reports,
            "notes": self.notes,
            "iterations_used": self.iterations_used,
            "failure_count": self.failure_count,
        }

    def compact_summary(self, max_chars: int = 500) -> str:
        """Produce a ≤500-char summary for context injection into subsequent phases."""
        parts = [f"[Phase {self.subtask_index}] {self.status.upper()}: {self.summary[:200]}"]
        if self.files_created:
            parts.append(f"Created: {', '.join(self.files_created[:5])}")
        if self.files_modified:
            parts.append(f"Modified: {', '.join(self.files_modified[:5])}")
        if self.services_running:
            svc = ", ".join(f"{s.get('name')}:{s.get('port')}" for s in self.services_running[:3])
            parts.append(f"Services: {svc}")
        if self.notes:
            parts.append(f"Notes: {'; '.join(self.notes[:3])}")
        return "\n".join(parts)[:max_chars]


# ---------------------------------------------------------------------------
# HandoffExtractor
# ---------------------------------------------------------------------------

class HandoffExtractor:
    """Extracts a structured JSON handoff from a completed (or partial) react result."""

    SCHEMA_VERSION = 1

    def __init__(self, agent):
        self.agent = agent

    def _read_incomplete_state(self, instruction: str) -> Optional[Dict]:
        """Read last_incomplete_task.json if fresh (≤10 min) and matches instruction."""
        try:
            if not os.path.exists(INCOMPLETE_TASK_PATH):
                return None
            age = datetime.now().timestamp() - os.path.getmtime(INCOMPLETE_TASK_PATH)
            if age > 600:
                return None
            with open(INCOMPLETE_TASK_PATH) as f:
                state = json.load(f)
            if state.get("instruction") != instruction:
                return None
            return state
        except Exception:
            return None

    def extract(self, instruction: str, react_result: Dict) -> Dict:
        """
        Build a structured handoff dict from a react_result.
        Never returns None — always falls back gracefully.
        """
        trace = react_result.get("trace", [])
        finish_summary = react_result.get("finish_summary", "")
        success = react_result.get("success", False)
        iterations_used = react_result.get("iterations_used", 0)

        # Build a digest of successful trace entries only
        digest_lines = []
        for entry in trace:
            result = entry.get("result")
            if result is None:
                continue
            # Handle both ToolResult objects and plain dicts
            if hasattr(result, "success"):
                result_success = result.success
                result_output = (result.output or "")[:200]
            elif isinstance(result, dict):
                result_success = result.get("success", False)
                result_output = str(result.get("output", ""))[:200]
            else:
                continue

            if not result_success:
                continue

            tool = entry.get("tool", "")
            args = entry.get("args", {})
            args_str = json.dumps(args)[:400]
            digest_lines.append(f"[{tool}] args={args_str} output={result_output!r}")

        digest_str = "\n".join(digest_lines) if digest_lines else "(no successful actions)"

        prompt = f"""Extract concrete facts from this completed agent task.

INSTRUCTION: {instruction}
SUCCESS: {success}
FINISH SUMMARY: {finish_summary}

SUCCESSFUL ACTIONS (digest):
{digest_str}

Return ONLY valid JSON matching this schema (omit fields with empty values):
{{
  "schema_version": 1,
  "source_subtask_index": 0,
  "source_instruction": "...",
  "completed_at": "ISO timestamp",
  "success": {str(success).lower()},
  "iterations_used": {iterations_used},
  "finish_summary": "one sentence",
  "facts": {{
    "files_created": [],
    "files_modified": [],
    "services_running": [],
    "ports_open": [{{"port": 5432, "service": "postgresql"}}],
    "packages_installed": [],
    "credentials": {{}},
    "versions": {{}},
    "config_values": {{}},
    "custom_notes": []
  }}
}}

Only populate facts fields that have actual data from the actions above. Return JSON only."""

        system = "You are a task fact extractor. Return only valid JSON. No prose."

        try:
            raw = self.agent._call_model_oneshot(
                self.agent.fast_model, prompt, system, timeout=45
            )
            handoff = self.agent.extract_json(raw)
            if not handoff or not isinstance(handoff, dict):
                raise ValueError("invalid JSON from LLM")
        except Exception:
            handoff = {
                "schema_version": self.SCHEMA_VERSION,
                "facts": {},
                "finish_summary": finish_summary,
                "success": success,
                "iterations_used": iterations_used,
                "source_instruction": instruction,
                "completed_at": datetime.now().isoformat(),
            }

        # Ensure required fields are present
        handoff.setdefault("schema_version", self.SCHEMA_VERSION)
        handoff.setdefault("completed_at", datetime.now().isoformat())
        handoff.setdefault("facts", {})
        handoff.setdefault("source_instruction", instruction)
        handoff.setdefault("success", success)
        handoff.setdefault("iterations_used", iterations_used)

        # Fold in partial completion info if task was incomplete
        if not success:
            incomplete_state = self._read_incomplete_state(instruction)
            if incomplete_state:
                accomplished = incomplete_state.get("accomplished", [])
                handoff["partial_completion"] = {
                    "accomplished": accomplished,
                    "what_remains": finish_summary,
                    "source": "last_incomplete_task.json",
                }

        return handoff


# ---------------------------------------------------------------------------
# AcceptanceCriteriaRunner
# ---------------------------------------------------------------------------

class AcceptanceCriteriaRunner:
    """Runs a shell command as an acceptance gate. No LLM involved."""

    def run(self, command: str, timeout: int = 30, cwd: Optional[str] = None) -> Dict:
        """
        Run acceptance criteria command.
        Returns dict with passed, exit_code, stdout, stderr, command, checked_at, cwd.

        Hardening: cwd parameter resolves relative paths (e.g. "test -f DOCS/ARCH.json")
        against the workspace rather than the server's current working directory. When
        cwd is None, behavior is unchanged.
        """
        checked_at = datetime.now().isoformat()

        if not command or not command.strip():
            return {
                "passed": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": "Empty command",
                "command": command,
                "checked_at": checked_at,
                "cwd": cwd,
            }

        # Safety validation first
        is_safe, risk_level, reason = CommandSafetyValidator.validate_command(command)
        if not is_safe or risk_level == "blocked":
            return {
                "passed": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Blocked by safety validator: {reason}",
                "command": command,
                "checked_at": checked_at,
                "cwd": cwd,
            }

        # Validate cwd before passing to subprocess — a missing directory would
        # otherwise raise FileNotFoundError and look like an AC failure.
        run_cwd = cwd if (cwd and os.path.isdir(cwd)) else None

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
                cwd=run_cwd,
            )
            return {
                "passed": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:500],
                "command": command,
                "checked_at": checked_at,
                "cwd": run_cwd,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "command": command,
                "checked_at": checked_at,
                "cwd": run_cwd,
            }
        except Exception as e:
            return {
                "passed": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "command": command,
                "checked_at": checked_at,
                "cwd": run_cwd,
            }

    @staticmethod
    def soft_re_verify(command: str, cwd: Optional[str] = None) -> Optional[Dict]:
        """Disk-truth recovery: if the AC subprocess returned non-zero but the
        artifact files actually exist on disk (and any embedded JSON parses),
        return a soft-pass dict so the chain can advance. None if it can't parse
        or files genuinely don't exist.

        Targets common AC patterns:
          test -f /abs/path
          [ -f /abs/path ]
          test -f relative/path  (resolved against cwd)
          python3 -c "... json.load(open('/path/to/file.json'))"

        This is intentionally narrow — only file-existence + JSON-parseability.
        Anything more complex (curl, port checks, exit codes) is not soft-verified.
        """
        if not command or not isinstance(command, str):
            return None

        # Extract candidate paths from `test -f <path>` and `[ -f <path> ]`
        path_patterns = [
            r'test\s+-[fde]\s+([^\s;&|]+)',
            r'\[\s+-[fde]\s+([^\s;&|]+)\s+\]',
        ]
        candidate_paths = []
        for pat in path_patterns:
            for m in _re.finditer(pat, command):
                candidate_paths.append(m.group(1).strip("'\""))

        # Extract JSON-load paths: open('/path/to/file')
        json_paths = []
        if "json.load" in command and "open(" in command:
            for m in _re.finditer(r"open\(['\"]([^'\"]+)['\"]\)", command):
                json_paths.append(m.group(1))

        if not candidate_paths and not json_paths:
            return None

        verified_files = []

        def _resolve(p: str) -> str:
            if os.path.isabs(p):
                return p
            base = cwd if cwd else os.getcwd()
            return os.path.join(base, p)

        for p in candidate_paths:
            full = _resolve(p)
            if not os.path.exists(full):
                return None
            verified_files.append(full)

        # JSON parseability check (any path mentioned in json.load)
        for p in json_paths:
            full = _resolve(p)
            if not os.path.exists(full):
                return None
            try:
                with open(full) as _jf:
                    json.load(_jf)
            except Exception:
                return None
            if full not in verified_files:
                verified_files.append(full)

        if not verified_files:
            return None

        return {
            "passed": True,
            "soft_pass": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
            "command": command,
            "checked_at": datetime.now().isoformat(),
            "cwd": cwd,
            "verified_files": verified_files,
            "note": "AC subprocess returned non-zero but files exist on disk",
        }


# ---------------------------------------------------------------------------
# TaskDecomposer
# ---------------------------------------------------------------------------

class TaskDecomposer:
    """Breaks a high-level goal into a list of scoped sub-tasks with iteration budgets."""

    COMPLEXITY_DEFAULTS = {"small": 25, "medium": 50, "large": 100}
    MIN_ITERATIONS_PER_TASK = 20

    def __init__(self, agent):
        self.agent = agent

    def decompose(self, goal: str, total_budget: int = 100) -> List[Dict]:
        """
        Decompose goal into ordered sub-tasks.
        Returns a list of sub-task dicts with enforced budget constraints.
        Falls back to single sub-task on LLM failure.
        """
        # Give LLM a realistic per-phase target so it doesn't output tiny values
        target_phases = min(8, max(3, total_budget // 100))
        per_phase_hint = max(25, total_budget // target_phases)

        # Extract an explicit workspace path from the goal if provided
        # Matches: "workspace in /abs/path" or "workspace: /abs/path" etc.
        ws_match = _re.search(
            r'workspace[:\s]+(/[^\s,."\']+)',
            goal,
            _re.IGNORECASE,
        )
        workspace = ws_match.group(1).rstrip('/') if ws_match else ""

        # PR0: ARCH.json feature flag. Default ON — planner emits structured JSON.
        # If off, falls back to the legacy ARCH.md prompt for safe rollback.
        _arch_json = os.getenv("AGENT_ARCH_JSON", "1") != "0"
        _arch_filename = "ARCH.json" if _arch_json else "ARCH.md"

        if workspace:
            arch_path      = f"{workspace}/DOCS/{_arch_filename}"
            if _arch_json:
                arch_ac = (
                    f"test -f {arch_path} && "
                    f"python3 -c \"import json; json.load(open('{arch_path}'))\""
                )
                phase0_instr = (
                    f"Create {arch_path} — a structured JSON architecture contract with keys: "
                    f"routes[{{method,path,handler,request_schema,response_schema}}], "
                    f"models[{{name,fields,relationships}}], ports[{{service,port}}], "
                    f"files[{{path,purpose}}], dependencies[]. "
                    f"See DOCS/ARCH_SCHEMA.md for exact shape. "
                    f"After writing, call validate_arch {{\"path\":\"{arch_path}\"}} to verify. "
                    f"No implementation code — contract only."
                )
            else:
                arch_ac = f"test -f {arch_path}"
                phase0_instr = (
                    f"Create {arch_path} with: module list, DB schema (if needed), "
                    f"API route table, port assignments, auth approach, and file layout. "
                    f"No code — specification only."
                )
            workspace_rule = (
                f"\n0. WORKSPACE: ALL files MUST be created under {workspace}/. "
                f"Use ABSOLUTE paths only — no relative paths ever. "
                f"acceptance_criteria shell commands must also use the full absolute path."
            )
        else:
            arch_path      = f"DOCS/{_arch_filename}"
            if _arch_json:
                arch_ac = (
                    f"test -f {arch_path} && "
                    f"python3 -c \"import json; json.load(open('{arch_path}'))\""
                )
                phase0_instr = (
                    f"Create {arch_path} — a structured JSON architecture contract with keys: "
                    f"routes[{{method,path,handler,request_schema,response_schema}}], "
                    f"models[{{name,fields,relationships}}], ports[{{service,port}}], "
                    f"files[{{path,purpose}}], dependencies[]. "
                    f"See DOCS/ARCH_SCHEMA.md for exact shape. "
                    f"After writing, call validate_arch {{\"path\":\"{arch_path}\"}} to verify. "
                    f"No implementation code — contract only."
                )
            else:
                arch_ac = f"test -f {arch_path}"
                phase0_instr = (
                    f"Create {arch_path} with: module list, DB schema (if needed), "
                    f"API route table, port assignments, auth approach, and file layout. "
                    f"No code — specification only."
                )
            workspace_rule = ""

        prompt = f"""Decompose this goal into sequential sub-tasks for an autonomous agent.

GOAL: {goal}
TOTAL ITERATION BUDGET: {total_budget}
TARGET PHASES: {target_phases} (aim for this many, each getting ~{per_phase_hint} iterations)

Return a JSON array of sub-tasks. Each element:
{{
  "index": 0,
  "instruction": "specific, actionable instruction for one sub-task",
  "acceptance_criteria": "shell command that exits 0 on success, or null",
  "estimated_complexity": "small|medium|large",
  "max_iterations": {per_phase_hint},
  "role": "planner|builder|tester|commander"
}}

ROLE ASSIGNMENTS (REQUIRED — every subtask must have a role):
- "planner": read existing code/docs and write one specification/plan file. No code implementation.
- "builder": write/modify code files only. No shell commands.
- "tester": run syntax checks, import checks, unit tests. Does NOT modify source files.
- "commander": install deps, migrate db, start services. No application code writing.
- Never assign builder to a phase that only runs shell commands.
- Never assign commander to a phase that writes code files.
- Phase 0 (architecture/spec) MUST use role=planner.
- Verification phases (curl health, syntax check) MUST use role=tester.
- Dependency install / service start phases MUST use role=commander.

MANDATORY DECOMPOSITION RULES:{workspace_rule}
1. PHASE 0 MUST be: "{phase0_instr}"
   Its acceptance_criteria MUST be: "{arch_ac}"
   Its role MUST be: "planner"
2. MODULAR STRUCTURE: Decompose so each phase touches ≤3 files. Prefer src/auth/, src/models/,
   src/api/ layout over a monolithic app.py.
3. DB MIGRATION: If a database is needed, one phase must be: "Initialize DB schema using schema.sql
   or Alembic migration — do NOT create tables at app startup."
4. SERVER VERIFICATION: Any phase that starts a server must be followed immediately by a
   tester phase: "Run syntax checks and curl the /health endpoint to confirm it started."
5. ORDER: Always put install/dependency phases before code phases, code before server start.
6. Order tasks so each builds on the previous.
7. acceptance_criteria must be a simple verifiable shell command using ABSOLUTE paths
   (e.g. "test -f {arch_path}", "curl -sf http://localhost:PORT")
8. Set max_iterations proportional to complexity: simpler phases get fewer, complex phases get more.
9. Total max_iterations across all tasks must sum to approximately {total_budget}.
10. Aim for {target_phases} sub-tasks; prefer fewer larger phases over many tiny ones.

Return ONLY the JSON array, no other text."""

        system = "You are a task decomposer. Return only a JSON array of sub-tasks. No prose."

        # Use non-thinking coder model for decomposition — qwen3.6-Grindlewalt
        # burns all num_predict tokens on thinking and returns empty content.
        DECOMP_MODEL = "qwen3-coder:30b"

        try:
            raw = self.agent._call_model_oneshot(
                DECOMP_MODEL, prompt, system, timeout=180
            )
            # Prefer direct JSON array parse; fall back to extract_json which
            # only grabs the first {} object when the response is an array.
            subtasks = None
            try:
                parsed = json.loads(raw.strip())
                if isinstance(parsed, list):
                    subtasks = parsed
            except Exception:
                pass
            if subtasks is None:
                # Try to extract an array via regex
                m = _re.search(r'\[\s*\{.*\}\s*\]', raw, _re.DOTALL)
                if m:
                    try:
                        subtasks = json.loads(m.group(0))
                    except Exception:
                        pass
            if subtasks is None:
                subtasks = self.agent.extract_json(raw)
            if not subtasks or not isinstance(subtasks, list) or len(subtasks) == 0:
                raise ValueError("invalid decomposition result")
        except Exception:
            # Fallback: single sub-task with full budget
            return [{
                "index": 0,
                "instruction": goal,
                "acceptance_criteria": None,
                "estimated_complexity": "large",
                "max_iterations": total_budget,
            }]

        # Fill in max_iterations from complexity if missing or zero
        _internal_only = {k for k, v in ROLES.items() if v.get("internal_only")}
        for task in subtasks:
            if not task.get("max_iterations"):
                complexity = task.get("estimated_complexity", "medium")
                task["max_iterations"] = self.COMPLEXITY_DEFAULTS.get(complexity, 25)
            task.setdefault("acceptance_criteria", None)
            task.setdefault("estimated_complexity", "medium")
            task.setdefault("role", "builder")  # default if LLM omitted the field
            # Safety: never allow decomposer to assign an internal-only role (e.g. reconciler).
            if task.get("role") in _internal_only:
                task["role"] = "builder"
            # Preserve original budget so the rebalancer can compute slack + caps later.
            task["original_max_iterations"] = task["max_iterations"]
            task.setdefault("failure_count", 0)

        # Re-index sequentially
        for i, task in enumerate(subtasks):
            task["index"] = i

        # Budget enforcement: always rescale so phases fill the total budget exactly
        total = sum(t.get("max_iterations", 25) for t in subtasks)
        if total != total_budget:
            scale = total_budget / total
            for task in subtasks:
                task["max_iterations"] = max(
                    self.MIN_ITERATIONS_PER_TASK,
                    int(task["max_iterations"] * scale)
                )
            # Assign any rounding remainder to the last phase
            used = sum(t["max_iterations"] for t in subtasks)
            remainder = total_budget - used
            if remainder > 0:
                subtasks[-1]["max_iterations"] += remainder

        return subtasks


# ---------------------------------------------------------------------------
# SubtaskReplanner
# ---------------------------------------------------------------------------

class SubtaskReplanner:
    """Adjusts the next sub-task instruction based on facts from the previous handoff."""

    def __init__(self, agent):
        self.agent = agent

    def replan(self, original_subtask: Dict, handoff: Dict, chain_goal: str) -> Dict:
        """
        Given the previous handoff, decide if the sub-task needs adjustment or skip.
        Always returns a valid dict — never blocks execution on failure.
        """
        facts = handoff.get("facts", {})
        finish_summary = handoff.get("finish_summary", "")
        partial = handoff.get("partial_completion")

        facts_str = json.dumps(facts, indent=2) if facts else "{}"
        partial_str = ""
        if partial:
            partial_str = f"\nPartial completion: {json.dumps(partial, indent=2)}"

        prompt = f"""Review a planned sub-task in light of what the previous agent just completed.

OVERALL CHAIN GOAL: {chain_goal}

PREVIOUS TASK RESULT:
Summary: {finish_summary}{partial_str}
Facts discovered:
{facts_str}

NEXT PLANNED SUB-TASK:
Instruction: {original_subtask.get('instruction', '')}
Acceptance Criteria: {original_subtask.get('acceptance_criteria', 'none')}

Decide:
1. Proceed unchanged — it's still needed and correct as-is
2. Adjust — modify the instruction or criteria based on discovered facts
3. Skip — it was already completed by the previous task or is now irrelevant

Return JSON only:
{{
  "instruction": "same or adjusted instruction",
  "acceptance_criteria": "same, adjusted, or null",
  "skip": false,
  "reason": "brief explanation"
}}"""

        system = "You are a task replanner. Return only valid JSON. Never block execution."

        try:
            raw = self.agent._call_model_oneshot(
                self.agent.fast_model, prompt, system, timeout=30
            )
            result = self.agent.extract_json(raw)
            if not result or not isinstance(result, dict):
                raise ValueError("invalid JSON")
            result.setdefault("instruction", original_subtask.get("instruction", ""))
            result.setdefault("acceptance_criteria", original_subtask.get("acceptance_criteria"))
            result.setdefault("skip", False)
            result.setdefault("reason", "replan succeeded")
            return result
        except Exception:
            return {
                "instruction": original_subtask.get("instruction", ""),
                "acceptance_criteria": original_subtask.get("acceptance_criteria"),
                "skip": False,
                "reason": "replan failed",
            }


# ---------------------------------------------------------------------------
# SubtaskOrchestrator
# ---------------------------------------------------------------------------

class SubtaskOrchestrator:
    """
    Producer tier. Given one subtask + full chain context, dispatches to a
    role-specific handler, writes a sidechain transcript, and auto-injects a
    tester gate after every builder phase.

    Legacy tool-set aliases kept for TDA helpers:
    """

    CODER_TOOLS = {"read_file", "create_file", "patch_file", "validate_arch",
                   "get_deps", "finish", "write_plan"}
    COMMANDER_TOOLS = {"execute_command", "manage_server", "read_file", "finish"}

    def __init__(self, agent):
        """agent: shared OllamaCommandAgent instance (history cleared per minion)."""
        self.agent = agent
        # PR2 — file ownership tracking: file_path -> {owner_role, phase_created, last_modified_phase}
        self._file_ownership: Dict[str, Dict] = {}
        # PR2 — active dep-graph reverify queue: files to re-check on next reconciler gate
        self._pending_reverify_set: set = set()

    # -------------------------------------------------------- public API --

    def orchestrate(
        self,
        subtask: Dict,
        chain_context: Dict,
    ) -> ImplementationArtifact:
        """
        Dispatch subtask to a role-specific handler, write sidechain transcript,
        and auto-inject tester gate after builder phases.
        Returns an ImplementationArtifact aggregating all results.
        """
        subtask_index = subtask.get("index", 0)
        subtask_instruction = subtask.get("instruction", "")
        role = subtask.get("role", "builder")
        role_cfg = ROLES.get(role, ROLES["builder"])

        print(f"\n{'=' * 70}")
        print(f"🎬 SubtaskOrchestrator: phase {subtask_index} [{role}] — {subtask_instruction[:80]}")

        # Role-based dispatch:
        # planner / tester → single-minion with role's tool whitelist + system prefix
        # builder / commander → micro-task decomposition with appropriate tool set
        if role_cfg.get("single_minion") or self._is_single_file_task(subtask_instruction):
            artifact = self._run_as_role_minion(subtask, chain_context, role_cfg)
        else:
            # builder / commander: micro-task decomposition with role-appropriate tool set
            # Map role → tool whitelist (fall back to type-based selection inside loop)
            _role_tools = role_cfg["tools"]
            artifact = self._orchestrate_with_microtasks(
                subtask, chain_context, role_override_tools=_role_tools
            )

        # Write the builder/planner/commander sidechain BEFORE running gates, so each
        # gate gets its own isolated trace file (fixes pre-existing overwrite bug).
        self._write_sidechain(subtask_index, role)

        # ── Auto-tester gate: run inline verification after every builder phase ──
        tester_summary: Dict = {}
        if role == "builder" and artifact.status in ("completed", "partial"):
            tester_summary = self._run_inline_tester(artifact, chain_context, subtask_index)
            if tester_summary.get("failed", 0) > 0 or not tester_summary.get("success", True):
                artifact.notes.append(
                    f"Tester gate detected issues: {tester_summary.get('summary', '')[:200]}"
                )
                if tester_summary.get("contract_failed"):
                    artifact.notes.append(
                        f"Contract tests failed: {tester_summary['contract_failed']}"
                    )
                artifact.status = "partial"
                print(f"  ⚠️  Tester gate: issues detected — marked phase as partial")
            else:
                print(f"  ✅ Tester gate: all checks passed")
            # Separate sidechain file for tester trace
            self._write_sidechain(subtask_index, "tester")

        # ── Reconciler gate: only after builder phases that didn't hard-fail ──
        _reconciler_on = os.getenv("AGENT_RECONCILER", "1") != "0"
        if _reconciler_on and role == "builder" and artifact.status != "failed":
            builder_iters = int(subtask.get("max_iterations", 25))
            rec_budget = max(10, builder_iters // 3)
            # PR3 — integration-shaped tester failures get more reconciler budget
            if tester_summary.get("needs_reconciler"):
                rec_budget = max(rec_budget, builder_iters // 2)
                print(f"  ⬆️  Reconciler budget boosted to {rec_budget} (integration failure detected)")
            # PR2 — active dep graph: prepend queued reverify files so they get re-checked first
            phase_files = list(artifact.files_created) + list(artifact.files_modified)
            reverify = [f for f in self._pending_reverify_set if f not in phase_files]
            files_for_rec = reverify + phase_files if reverify else None
            if reverify:
                print(f"  🔁 Active dep-graph: re-verifying {len(reverify)} dependent file(s)")
            rec_summary = self._run_inline_reconciler(
                artifact, chain_context, subtask_index,
                budget=rec_budget, scope="phase",
                files_override=files_for_rec,
            )
            # Clear the reverify queue — consumed by this gate
            self._pending_reverify_set.clear()
            if rec_summary.get("patched_count", 0) > 0 or rec_summary.get("arch_updated"):
                artifact.notes.append(
                    f"Reconciler: {rec_summary.get('summary', '')[:200]}"
                )
            self._write_sidechain(subtask_index, "reconciler")

        # PR2 — record structured lesson on partial/failed phases
        self._record_role_lesson(role, artifact, chain_context)
        # PR2 — dump ownership map sidecar for audit
        self._write_ownership_sidecar(subtask_index)

        print(f"\n✅ SubtaskOrchestrator phase {subtask_index} [{role}] done: {artifact.status}")
        return artifact

    def _orchestrate_with_microtasks(
        self,
        subtask: Dict,
        chain_context: Dict,
        role_override_tools: Optional[set] = None,
    ) -> ImplementationArtifact:
        """Run subtask through micro-task decomposition.  Called for builder/commander roles."""
        subtask_index = subtask.get("index", 0)
        subtask_instruction = subtask.get("instruction", "")

        # 1. Decompose subtask into micro-tasks
        micro_tasks = self._decompose_to_micro_tasks(subtask_instruction, chain_context)
        print(f"  📋 {len(micro_tasks)} micro-tasks planned")

        # PR2 — inject role-scoped lessons before spawning minions
        role = subtask.get("role", "builder")
        self._inject_role_lessons(role, chain_context)

        micro_task_reports: List[Dict] = []
        all_files_created: List[str] = []
        all_files_modified: List[str] = []
        all_services: List[Dict] = []
        all_credentials: Dict[str, str] = {}
        all_notes: List[str] = []
        overall_status = "completed"

        # 2. Run each micro-task as a Minion (clean history per minion)
        for i, micro_task in enumerate(micro_tasks):
            mt_type = micro_task.get("type", "command")
            work_order = micro_task.get("work_order", micro_task.get("instruction", ""))
            file_scope = micro_task.get("file_scope", [])

            print(f"\n  🤖 Minion {i + 1}/{len(micro_tasks)} [{mt_type}]: {work_order[:80]}")

            # Snapshot files before code micro-tasks so we can revert on failure
            snapshot: Dict[str, Optional[str]] = {}
            if mt_type == "code" and file_scope:
                snapshot = self._snapshot_files(file_scope)
                print(f"  📸 Snapshotted {len(snapshot)} file(s)")

            # TDA — use test-first loop for Python code tasks
            if self._is_tda_eligible(micro_task):
                print(f"  🧪 TDA: running QA→Coder→Validator loop")
                report = self._run_tda_code_task(micro_task, chain_context, micro_task_reports)
            else:
                # Tool whitelist: role override > type-based fallback
                if role_override_tools is not None:
                    tools = role_override_tools
                else:
                    tools = self.CODER_TOOLS if mt_type == "code" else self.COMMANDER_TOOLS

                prompt = self._build_minion_prompt(
                    work_order=work_order,
                    chain_context=chain_context,
                    previous_reports=micro_task_reports,
                    file_scope=file_scope,
                    mt_type=mt_type,
                    role=role,  # parent subtask's role drives tool block
                )

                self.agent.react_trace = []
                self.agent.tool_registry.reset_phase_state()

                minion_iters = max(15, subtask.get("max_iterations", 25))
                result = self.agent.run_react(
                    instruction=prompt,
                    tool_whitelist=tools,
                    max_iterations=minion_iters,
                )

                report = {
                    "micro_task_index": i,
                    "type": mt_type,
                    "work_order": work_order,
                    "success": result.get("success", False),
                    "finish_summary": result.get("finish_summary", ""),
                    "iterations_used": result.get("iterations_used", 0),
                }

                # Extract files touched from trace
                for entry in result.get("trace", []):
                    entry_tool = entry.get("tool", "")
                    entry_args = entry.get("args", {})
                    entry_result = entry.get("result")
                    ok = getattr(entry_result, "success", False) if hasattr(entry_result, "success") \
                        else (entry_result or {}).get("success", False)
                    if ok and entry_tool == "create_file":
                        p = entry_args.get("path", "")
                        if p and p not in all_files_created:
                            all_files_created.append(p)
                    elif ok and entry_tool == "patch_file":
                        p = entry_args.get("path", "")
                        if p and p not in all_files_modified:
                            all_files_modified.append(p)
                    elif ok and entry_tool == "manage_server":
                        if entry_args.get("action") == "start":
                            all_services.append({
                                "name": entry_args.get("name", ""),
                                "command": entry_args.get("command", ""),
                            })

            micro_task_reports.append(report)

            if not report["success"] and snapshot:
                self._restore_snapshot(snapshot)
                all_notes.append(
                    f"Minion {i + 1} failed — reverted {len(snapshot)} file(s) to pre-task state"
                )
                print(f"  ↩️  Reverted {len(snapshot)} file(s) after minion failure")

            if not report["success"]:
                all_notes.append(f"Minion {i + 1} failed: {report['finish_summary'][:120]}")
                overall_status = "partial"

        summary_parts = [r.get("finish_summary", "") for r in micro_task_reports if r.get("finish_summary")]
        summary = " | ".join(summary_parts[:3])[:500] or f"Phase {subtask_index} complete"

        # PR2 — record file ownership + enqueue dependents for next reconciler gate
        own_warnings = self._record_ownership(
            all_files_created, all_files_modified, role, subtask_index,
        )
        if own_warnings:
            all_notes.extend(own_warnings)
        self._update_reverify_set(all_files_created + all_files_modified)

        return ImplementationArtifact(
            subtask_index=subtask_index,
            subtask_instruction=subtask_instruction,
            status=overall_status,
            summary=summary,
            files_created=all_files_created,
            files_modified=all_files_modified,
            services_running=all_services,
            credentials=all_credentials,
            micro_task_reports=micro_task_reports,
            notes=all_notes,
        )

    # ---------------------------------------------------- private helpers --

    # ---------------------------------------------------- PR2 bookkeeping --

    def _record_ownership(
        self,
        created: List[str],
        modified: List[str],
        role: str,
        phase: int,
    ) -> List[str]:
        """Update file-ownership map and return a list of cross-phase-edit warnings
        (strings) for patches that modified files owned by other roles/phases.
        """
        if os.getenv("AGENT_FILE_OWNERSHIP", "1") == "0":
            return []
        warnings: List[str] = []
        for p in created:
            if p and p not in self._file_ownership:
                self._file_ownership[p] = {
                    "owner_role": role,
                    "phase_created": phase,
                    "last_modified_phase": phase,
                }
        for p in modified:
            if not p:
                continue
            record = self._file_ownership.get(p)
            if record and record["owner_role"] != role and record["phase_created"] < phase:
                warnings.append(
                    f"⚠️ Cross-phase edit: {p} (owned by {record['owner_role']} from "
                    f"phase {record['phase_created']}, patched by {role} in phase {phase})"
                )
            # Still update last_modified_phase even if not owned here
            if record:
                record["last_modified_phase"] = phase
            else:
                # File we didn't know about — treat this phase as owner
                self._file_ownership[p] = {
                    "owner_role": role,
                    "phase_created": phase,
                    "last_modified_phase": phase,
                }
        return warnings

    def _update_reverify_set(self, changed_files: List[str]) -> None:
        """Active dep-graph reverify: record files that depend on any of the changed
        files, so the next reconciler gate can re-check them first.
        """
        if os.getenv("AGENT_DEP_GRAPH_ACTIVE", "1") == "0":
            return
        try:
            import dep_graph as _dep
        except ImportError:
            return
        known = list(self._file_ownership.keys())
        graph = _dep.build_graph(known)
        affected = _dep.files_affected_by(set(changed_files), graph)
        if affected:
            self._pending_reverify_set.update(affected)

    def _write_ownership_sidecar(self, subtask_index: int) -> None:
        """Dump current ownership map to sidechain dir alongside the phase transcript."""
        try:
            os.makedirs(SIDECHAIN_DIR, exist_ok=True)
            ts = int(time.time())
            fpath = os.path.join(
                SIDECHAIN_DIR, f"subtask_{subtask_index:02d}_{ts}.ownership.json"
            )
            with open(fpath, "w") as f:
                json.dump(self._file_ownership, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------------- PR2 role-memory I/O --

    def _inject_role_lessons(self, role: str, chain_context: Dict) -> None:
        """Fetch and pin goal-scoped + role-wide lessons for the current role."""
        if os.getenv("AGENT_ROLE_MEMORY", "1") == "0":
            return
        memory = getattr(self.agent, "memory", None) or getattr(self.agent, "agent_memory", None)
        if memory is None or not hasattr(memory, "list_context"):
            return
        try:
            import role_lessons
        except ImportError:
            return
        goal = chain_context.get("goal", "")
        gh = role_lessons.goal_hash(goal)
        try:
            rows = memory.list_context(prefix=f"lesson_{role}_")
        except Exception:
            return
        lessons: List[Dict] = []
        for r in rows:
            if r.get("agent") and r.get("agent") != role:
                continue
            try:
                L = json.loads(r.get("value") or "{}")
            except Exception:
                continue
            # Prefer goal-scoped, fall back to cross-goal shared patterns
            if L.get("goal_hash") in (gh, None, ""):
                lessons.append(L)
        # If we got few goal-scoped, supplement with the highest-confidence shared patterns
        if len(lessons) < 3:
            for r in rows:
                if r.get("agent") and r.get("agent") != role:
                    continue
                try:
                    L = json.loads(r.get("value") or "{}")
                except Exception:
                    continue
                if L.get("goal_hash") != gh and L.get("pattern") != "other":
                    lessons.append(L)
        if not lessons:
            return
        msg = role_lessons.format_for_prompt(lessons, max_items=5)
        if not msg:
            return
        try:
            # Pin under a role-specific slot so it updates in place across calls
            self.agent._update_pinned(f"role_lessons_{role}",
                                      {"role": "system", "content": msg})
        except Exception:
            pass

    def _record_role_lesson(
        self,
        role: str,
        artifact: "ImplementationArtifact",
        chain_context: Dict,
    ) -> None:
        """After a partial/failed phase, extract a structured lesson and merge into memory."""
        if os.getenv("AGENT_ROLE_MEMORY", "1") == "0":
            return
        if artifact.status not in ("partial", "failed"):
            return
        memory = getattr(self.agent, "memory", None) or getattr(self.agent, "agent_memory", None)
        if memory is None or not hasattr(memory, "set_context"):
            return
        try:
            import role_lessons
        except ImportError:
            return
        summary = artifact.summary or ""
        notes = list(artifact.notes or [])
        goal = chain_context.get("goal", "")
        lesson = role_lessons.extract_lesson(
            summary, notes,
            role=role,
            source_phase=artifact.subtask_index,
            goal=goal,
        )
        if not lesson:
            return
        pattern = lesson["pattern"]
        gh = lesson["goal_hash"]
        key = f"lesson_{role}_{pattern}_{gh}"
        try:
            existing_raw = memory.get_context(key)
            if existing_raw:
                try:
                    existing = json.loads(existing_raw)
                    lesson = role_lessons.merge_lesson(existing, lesson)
                except Exception:
                    pass
            memory.set_context(
                key, json.dumps(lesson),
                agent_id=role, ttl=86400 * 7,
            )
        except Exception:
            pass

    def _is_single_file_task(self, instruction: str) -> bool:
        """
        Return True when the instruction clearly involves creating/writing exactly
        one file.  Used to skip micro-task decomposition and run as a single minion.
        """
        # Match "Create /absolute/path/to/file.ext" at the start of the instruction
        return bool(_re.match(r'Create\s+/[^\s,]+\.[a-zA-Z]+', instruction.strip()))

    def _run_as_role_minion(
        self,
        subtask: Dict,
        chain_context: Dict,
        role_cfg: Dict,
    ) -> ImplementationArtifact:
        """Run the subtask as a single minion using the role's tool whitelist and system prefix."""
        subtask_index = subtask.get("index", 0)
        subtask_instruction = subtask.get("instruction", "")
        budget = max(15, subtask.get("max_iterations", 25))
        role = subtask.get("role", "builder")

        print(f"  🎭 Role={role} — single minion (budget={budget})")

        # PR2 — inject role-scoped lessons before spawning minion
        self._inject_role_lessons(role, chain_context)

        # Build the prompt using _build_minion_prompt, then prepend role's system prefix
        mt_type = "code" if role in ("planner", "builder") else "command"
        base_prompt = self._build_minion_prompt(
            work_order=subtask_instruction,
            chain_context=chain_context,
            previous_reports=[],
            file_scope=[],
            mt_type=mt_type,
            role=role,
        )
        # Prepend role persona so the model knows its constraints immediately
        full_prompt = f"{role_cfg['system_prefix']}\n\n{base_prompt}"

        self.agent.react_trace = []
        self.agent.tool_registry.reset_phase_state()
        result = self.agent.run_react(
            instruction=full_prompt,
            tool_whitelist=role_cfg["tools"],
            max_iterations=budget,
        )

        success = result.get("success", False)
        finish_summary = result.get("finish_summary", "")
        files_created: List[str] = []
        files_modified: List[str] = []

        for entry in result.get("trace", []):
            tool = entry.get("tool", "")
            args = entry.get("args", {})
            res = entry.get("result")
            ok = getattr(res, "success", False) if hasattr(res, "success") \
                else (res or {}).get("success", False)
            if ok and tool == "create_file":
                p = args.get("path", "")
                if p and p not in files_created:
                    files_created.append(p)
            elif ok and tool == "patch_file":
                p = args.get("path", "")
                if p and p not in files_modified:
                    files_modified.append(p)

        # PR2 — record file ownership + enqueue dependents for next reconciler gate
        own_notes: List[str] = []
        own_warnings = self._record_ownership(
            files_created, files_modified, role, subtask_index,
        )
        if own_warnings:
            own_notes.extend(own_warnings)
        self._update_reverify_set(files_created + files_modified)

        return ImplementationArtifact(
            subtask_index=subtask_index,
            subtask_instruction=subtask_instruction,
            status="completed" if success else "partial",
            summary=finish_summary[:500] or f"Phase {subtask_index} ({role}) complete",
            files_created=files_created,
            files_modified=files_modified,
            notes=own_notes,
            micro_task_reports=[{
                "micro_task_index": 0,
                "type": mt_type,
                "work_order": subtask_instruction,
                "success": success,
                "finish_summary": finish_summary,
                "iterations_used": result.get("iterations_used", 0),
            }],
        )

    def _run_inline_tester(
        self,
        artifact: ImplementationArtifact,
        chain_context: Dict,
        subtask_index: int,
    ) -> Dict:
        """Run a quick tester minion against the files created by a builder phase.
        Returns {"success": bool, "failed": int, "summary": str}.
        """
        all_files = artifact.files_created + artifact.files_modified
        if not all_files:
            return {"success": True, "failed": 0, "summary": "no files to check"}

        py_files = [f for f in all_files if f.endswith(".py")]
        js_files = [f for f in all_files if f.endswith((".js", ".ts"))]

        # Use the same Python interpreter running this service (guaranteed venv).
        # Bare `python3` in a subprocess is the system Python on Arch (PEP 668 blocked).
        import sys as _sys
        _py = _sys.executable          # e.g. /mnt/storage/NAS/Jarvis/.venv/bin/python3
        _pip = f"{_py} -m pip"

        # Auto-install requirements.txt files found near any created Python files.
        # Builder can't run pip (no execute_command), so we do it here before checks.
        req_candidates = set()
        for f in py_files:
            d = os.path.dirname(f)
            req_candidates.add(os.path.join(d, "requirements.txt"))
            req_candidates.add(os.path.join(os.path.dirname(d), "requirements.txt"))
        install_cmds = [
            f"[ -f {r} ] && {_pip} install -q -r {r} 2>&1 | tail -3 || true"
            for r in sorted(req_candidates)
        ]

        check_cmds = []
        for f in py_files[:5]:
            check_cmds.append(f"{_py} -m py_compile {f} && echo 'OK: {f}' || echo 'FAIL: {f}'")
        for f in js_files[:3]:
            check_cmds.append(f"node --check {f} && echo 'OK: {f}' || echo 'FAIL: {f}'")

        # PR3 — spec-driven contract tests: regenerate from latest ARCH.json and run pytest
        contract_test_cmds: List[str] = []
        contract_on = os.getenv("AGENT_CONTRACT_TESTS", "1") != "0"
        happy_on = os.getenv("AGENT_HAPPY_PATH_TESTS", "1") != "0"
        workspace = self._extract_workspace(chain_context)
        if contract_on and workspace:
            arch_path = self._guess_arch_path(chain_context)
            if arch_path and os.path.exists(arch_path):
                try:
                    import arch_schema as _as
                    import contract_test_template as _ctt
                    arch_data = _as.load_arch(arch_path)
                    emitted = _ctt.emit_all(arch_data, workspace)
                    print(f"  📝 Contract tests regenerated: {list(emitted.keys())}")
                    for name, tpath in emitted.items():
                        if name == "happy_path" and not happy_on:
                            continue
                        contract_test_cmds.append(
                            f"{_py} -m pytest {tpath} -x --tb=short -q 2>&1 | tail -15 "
                            f"&& echo 'CONTRACT_OK: {name}' || echo 'CONTRACT_FAIL: {name}'"
                        )
                except Exception as _e:
                    print(f"  ⚠️  Contract-test emit failed: {_e}")

        if not check_cmds and not contract_test_cmds:
            return {"success": True, "failed": 0, "summary": "no .py/.js files to check"}

        all_cmds = install_cmds + check_cmds + contract_test_cmds
        tester_instruction = (
            f"Run dependency install, syntax checks, and contract tests for phase {subtask_index}.\n"
            "Run each command in order and report PASS or FAIL for each check:\n"
            + "\n".join(f"  {cmd}" for cmd in all_cmds)
            + "\n\nfinish() with test_results summary. Set success=false if any syntax "
              "check FAILs or CONTRACT_FAIL appears."
        )

        tester_role_cfg = ROLES["tester"]
        base_prompt = self._build_minion_prompt(
            work_order=tester_instruction,
            chain_context=chain_context,
            previous_reports=[],
            file_scope=all_files[:8],
            mt_type="command",
        )
        full_prompt = f"{tester_role_cfg['system_prefix']}\n\n{base_prompt}"

        print(f"\n  🧪 Auto-tester gate: checking {len(all_files)} file(s)...")
        self.agent.react_trace = []
        self.agent.tool_registry.reset_phase_state()
        result = self.agent.run_react(
            instruction=full_prompt,
            tool_whitelist=tester_role_cfg["tools"],
            max_iterations=15,
        )

        summary = result.get("finish_summary", "")
        low = (summary or "").lower()
        # Per-suite parsing: CONTRACT_FAIL/OK markers, plus "FAIL:" for syntax checks
        contract_failed = [m for m in ("routes", "models", "happy_path")
                           if f"contract_fail: {m}" in low]
        contract_passed = [m for m in ("routes", "models", "happy_path")
                           if f"contract_ok: {m}" in low]
        syntax_fails = low.count("fail:")
        # Integration-shaped failures (import/attribute errors) warrant reconciler follow-up
        needs_reconciler = bool(_re.search(
            r"(importerror|modulenotfounderror|attributeerror|cannot import)",
            low,
        ))
        total_failed = syntax_fails + len(contract_failed)
        return {
            "success": result.get("success", False) and total_failed == 0,
            "failed": total_failed,
            "summary": summary[:400],
            "contract_failed": contract_failed,
            "contract_passed": contract_passed,
            "syntax_fails": syntax_fails,
            "needs_reconciler": needs_reconciler,
        }

    def _run_inline_reconciler(
        self,
        artifact: "ImplementationArtifact",
        chain_context: Dict,
        subtask_index: int,
        budget: int = 12,
        scope: str = "phase",
        files_override: Optional[List[str]] = None,
    ) -> Dict:
        """Run a deterministic static-check sweep, then a reconciler minion if issues found.

        Returns {"patched_count": int, "arch_updated": bool, "summary": str,
                 "found_issues": bool}.
        """
        try:
            import reconciler_checks as rc
        except ImportError as e:
            return {"patched_count": 0, "arch_updated": False,
                    "summary": f"reconciler_checks unavailable: {e}",
                    "found_issues": False}

        # Files to analyse
        if files_override is not None:
            files = list(files_override)
        else:
            files = list(artifact.files_created) + list(artifact.files_modified)

        # Load ARCH.json from workspace if available
        arch: Optional[Dict] = None
        arch_path = self._guess_arch_path(chain_context)
        if arch_path and os.path.exists(arch_path):
            try:
                import arch_schema
                arch = arch_schema.load_arch(arch_path)
            except Exception:
                arch = None

        findings = rc.run_all(files, arch)
        if not rc.has_any_issue(findings):
            return {"patched_count": 0, "arch_updated": False,
                    "summary": "no drift detected", "found_issues": False}

        classification = rc.classify_violations(findings)
        findings_prompt = rc.format_findings_for_prompt(findings, classification, max_items=10)

        direction_hint = []
        if classification.get("patch_code"):
            direction_hint.append(
                f"PATCH CODE: {len(classification['patch_code'])} code-side fixes needed."
            )
        if classification.get("update_arch"):
            direction_hint.append(
                f"UPDATE ARCH: {len(classification['update_arch'])} ARCH.json entries to update "
                f"(code has evolved past contract)."
            )
        if classification.get("report_only"):
            direction_hint.append(
                f"REPORT: {len(classification['report_only'])} ambiguous items — do not auto-fix."
            )
        direction = "\n".join(direction_hint) or "Minor drift only."

        rec_role_cfg = ROLES["reconciler"]
        arch_hint = f"ARCH.json path: {arch_path}\n" if arch_path else ""
        work_order = (
            f"Reconcile drift between ARCH and code for phase {subtask_index} [{scope}].\n"
            f"{arch_hint}"
            f"DIRECTION:\n{direction}\n\n"
            f"FINDINGS:\n{findings_prompt}\n\n"
            f"Resolve each item. Use patch_file or create_file on code for patch_code items;\n"
            f"rewrite ARCH.json via create_file for update_arch items. For report_only, just note.\n"
            f"Call validate_arch after any ARCH.json rewrite. finish() with:\n"
            f"  'PATCHED: N | ARCH_UPDATED: True/False | <summary>'"
        )
        base_prompt = self._build_minion_prompt(
            work_order=work_order,
            chain_context=chain_context,
            previous_reports=[],
            file_scope=files[:10],
            mt_type="code",
        )
        full_prompt = f"{rec_role_cfg['system_prefix']}\n\n{base_prompt}"

        print(f"\n  🔧 Reconciler gate [{scope}]: {sum(len(v) for v in classification.values())} "
              f"findings, budget={budget}")
        self.agent.react_trace = []
        self.agent.tool_registry.reset_phase_state()
        result = self.agent.run_react(
            instruction=full_prompt,
            tool_whitelist=rec_role_cfg["tools"],
            max_iterations=budget,
        )
        summary = result.get("finish_summary", "")
        # Parse "PATCHED: N" / "ARCH_UPDATED: True" from the summary
        patched_count = 0
        m = _re.search(r'PATCHED:\s*(\d+)', summary or "")
        if m:
            try:
                patched_count = int(m.group(1))
            except ValueError:
                patched_count = 0
        arch_updated = bool(_re.search(r'ARCH_UPDATED:\s*True', summary or "", _re.IGNORECASE))
        return {
            "patched_count": patched_count,
            "arch_updated": arch_updated,
            "summary": summary[:400],
            "found_issues": True,
        }

    def _extract_workspace(self, chain_context: Dict) -> str:
        """Extract workspace dir from chain goal (e.g. 'Deploy to /path/' → /path)."""
        goal = chain_context.get("goal", "")
        m = _re.search(r'workspace[:\s]+(/[^\s,."\']+)', goal, _re.IGNORECASE)
        if not m:
            m = _re.search(r'(?:deploy to|in|at)\s+(/[^\s,."\']+)', goal, _re.IGNORECASE)
        return m.group(1).rstrip('/') if m else ""

    def _guess_arch_path(self, chain_context: Dict) -> Optional[str]:
        """Infer ARCH.json location from goal workspace (same logic as decomposer)."""
        ws = self._extract_workspace(chain_context)
        fname = "ARCH.json" if os.getenv("AGENT_ARCH_JSON", "1") != "0" else "ARCH.md"
        if ws:
            return f"{ws}/DOCS/{fname}"
        # Fall back to cwd-relative
        return f"DOCS/{fname}"

    def _write_sidechain(self, subtask_index: int, role_label: str) -> None:
        """Write the agent's react_trace to a sidechain jsonl file and zero it out.
        This keeps the chain parent context free of full minion traces.

        role_label differentiates builder/tester/reconciler traces within one phase
        so they don't overwrite each other (was a pre-existing bug).
        """
        try:
            os.makedirs(SIDECHAIN_DIR, exist_ok=True)
            ts = int(time.time())
            fname = f"subtask_{subtask_index:02d}_{role_label}_{ts}.jsonl"
            fpath = os.path.join(SIDECHAIN_DIR, fname)
            with open(fpath, "w") as f:
                for entry in self.agent.react_trace:
                    # Serialize result (ToolResult namedtuple → dict)
                    entry_copy = dict(entry)
                    r = entry_copy.get("result")
                    if hasattr(r, "_asdict"):
                        entry_copy["result"] = r._asdict()
                    try:
                        f.write(json.dumps(entry_copy) + "\n")
                    except (TypeError, ValueError):
                        f.write(json.dumps({"serialization_error": str(entry_copy)[:200]}) + "\n")
            # Zero out the trace so it doesn't inflate parent context
            self.agent.react_trace = []
            print(f"  📼 Sidechain saved → {fpath}")
        except Exception as e:
            print(f"  ⚠️  Sidechain write failed: {e}")

    def _run_as_single_minion(
        self,
        subtask: Dict,
        chain_context: Dict,
    ) -> ImplementationArtifact:
        """Run the entire subtask as one CODER minion — no micro-task decomposition."""
        subtask_index = subtask.get("index", 0)
        subtask_instruction = subtask.get("instruction", "")
        budget = max(20, subtask.get("max_iterations", 25))

        print(f"  📝 Single-file task — running as one CODER minion (budget={budget})")

        prompt = self._build_minion_prompt(
            work_order=subtask_instruction,
            chain_context=chain_context,
            previous_reports=[],
            file_scope=[],
            mt_type="code",
        )

        self.agent.react_trace = []
        self.agent.tool_registry.reset_phase_state()
        result = self.agent.run_react(
            instruction=prompt,
            tool_whitelist=self.CODER_TOOLS,
            max_iterations=budget,
        )

        success = result.get("success", False)
        finish_summary = result.get("finish_summary", "")
        files_created: List[str] = []
        files_modified: List[str] = []

        for entry in result.get("trace", []):
            tool = entry.get("tool", "")
            args = entry.get("args", {})
            res = entry.get("result")
            ok = getattr(res, "success", False) if hasattr(res, "success") \
                else (res or {}).get("success", False)
            if ok and tool == "create_file":
                p = args.get("path", "")
                if p and p not in files_created:
                    files_created.append(p)
            elif ok and tool == "patch_file":
                p = args.get("path", "")
                if p and p not in files_modified:
                    files_modified.append(p)

        report = {
            "micro_task_index": 0,
            "type": "code",
            "work_order": subtask_instruction,
            "success": success,
            "finish_summary": finish_summary,
            "iterations_used": result.get("iterations_used", 0),
        }

        print(f"\n✅ SubtaskOrchestrator phase {subtask_index} done (single-minion): "
              f"{'completed' if success else 'partial'}")

        return ImplementationArtifact(
            subtask_index=subtask_index,
            subtask_instruction=subtask_instruction,
            status="completed" if success else "partial",
            summary=finish_summary[:500] or f"Phase {subtask_index} complete",
            files_created=files_created,
            files_modified=files_modified,
            micro_task_reports=[report],
        )

    def _decompose_to_micro_tasks(
        self, subtask_instruction: str, chain_context: Dict
    ) -> List[Dict]:
        """LLM call (fast model, one-shot) to break a subtask into 3-5 micro-tasks."""
        context_summary = self._build_context_summary(chain_context)

        prompt = f"""Break this sub-task into sequential micro-tasks for individual worker agents.

SUB-TASK: {subtask_instruction}

CHAIN CONTEXT:
{context_summary}

Return a JSON array. Each element:
{{
  "index": 0,
  "type": "code|command",
  "work_order": "specific instruction for one micro-task worker",
  "file_scope": ["list", "of", "files", "this", "worker", "should", "touch"]
}}

Rules:
- type "code": for writing/editing files (use read_file, create_file, patch_file, finish)
- type "command": for running shell commands, installing packages, starting services
- Each micro-task should be completable in ≤15 iterations
- Keep micro-tasks atomic and non-overlapping in file scope
- IMPORTANT: If the sub-task involves writing a SINGLE file (e.g. an architecture spec,
  a config file, or one source file), return exactly 1 micro-task — do NOT split it.
  Splitting single-file tasks causes workers to overwrite each other.
- For multi-file tasks: 2-5 micro-tasks, grouped by file or concern
- Maximum 5 micro-tasks

Return ONLY the JSON array, no prose."""

        system = "You are a micro-task decomposer. Return only a JSON array. No prose."
        DECOMP_MODEL = "qwen3-coder:30b"
        try:
            raw = self.agent._call_model_oneshot(
                DECOMP_MODEL, prompt, system, timeout=180
            )
            micro_tasks = None
            try:
                parsed = json.loads(raw.strip())
                if isinstance(parsed, list):
                    micro_tasks = parsed
            except Exception:
                pass
            if micro_tasks is None:
                m = _re.search(r'\[\s*\{.*\}\s*\]', raw, _re.DOTALL)
                if m:
                    try:
                        micro_tasks = json.loads(m.group(0))
                    except Exception:
                        pass
            if micro_tasks is None:
                micro_tasks = self.agent.extract_json(raw)
            if micro_tasks and isinstance(micro_tasks, list) and len(micro_tasks) >= 1:
                for i, mt in enumerate(micro_tasks):
                    mt.setdefault("index", i)
                    mt.setdefault("type", "command")
                    mt.setdefault("file_scope", [])
                return micro_tasks[:5]
        except Exception:
            pass

        # Fallback: single micro-task with the full instruction
        return [{
            "index": 0,
            "type": "command",
            "work_order": subtask_instruction,
            "file_scope": [],
        }]

    def _build_context_summary(self, chain_context: Dict) -> str:
        """Produce a compact (≤500 char) summary of chain state for minion prompts."""
        goal = chain_context.get("goal", "")[:150]
        parts = [f"Goal: {goal}"]

        subtasks = chain_context.get("subtasks", [])
        for st in subtasks:
            idx = st.get("index", "?")
            status = st.get("status", "pending")
            instr = st.get("instruction", "")[:60]
            artifact_dict = st.get("artifact")
            if artifact_dict:
                art_summary = artifact_dict.get("summary", "")[:80]
                parts.append(f"  Phase {idx} [{status}]: {instr} → {art_summary}")
            else:
                parts.append(f"  Phase {idx} [{status}]: {instr}")

        return "\n".join(parts)[:500]

    def _build_minion_prompt(
        self,
        work_order: str,
        chain_context: Dict,
        previous_reports: List[Dict],
        file_scope: List[str],
        mt_type: str,
        role: Optional[str] = None,
    ) -> str:
        """Build the full prompt for a Minion agent."""
        goal = chain_context.get("goal", "")
        arch_summary = chain_context.get("arch_summary", "(not yet written)")

        # Carry workspace constraint into minion prompts if set in goal
        _ws_match = _re.search(r'workspace[:\s]+(/[^\s,."\']+)', goal, _re.IGNORECASE)
        _workspace = _ws_match.group(1).rstrip('/') if _ws_match else ""
        workspace_constraint = (
            f"  - WORKSPACE: all files MUST live under {_workspace}/ — absolute paths only\n"
            if _workspace else ""
        )

        # Compact previous micro-task reports
        prev_summaries = []
        for r in previous_reports:
            idx = r.get("micro_task_index", "?")
            ok = "✅" if r.get("success") else "❌"
            summary = r.get("finish_summary", "")[:100]
            prev_summaries.append(f"  {ok} Micro-task {idx}: {summary}")
        prev_text = "\n".join(prev_summaries) if prev_summaries else "  (none yet)"

        file_scope_text = (
            f"Only touch these files: {', '.join(file_scope)}"
            if file_scope else "No specific file scope restriction"
        )

        # PR2 — soft cross-role nudge: list files owned by prior phases (code tasks only)
        ownership_nudge = ""
        if mt_type == "code" and os.getenv("AGENT_FILE_OWNERSHIP", "1") != "0":
            other_owned = list((self._file_ownership or {}).items())[:8]
            if other_owned:
                pairs = [
                    f"{p} (owned by {meta.get('owner_role')} phase {meta.get('phase_created')})"
                    for p, meta in other_owned
                ]
                ownership_nudge = (
                    "  - Prefer NOT to edit these files (owned by other phases/roles); "
                    "if you must, explain why in your finish summary:\n"
                    + "\n".join(f"      • {x}" for x in pairs) + "\n"
                )

        # Resolve role: explicit > inferred from mt_type. Inferring 'builder' for
        # code and 'commander' for command keeps backward-compat with existing
        # micro-task callers that don't carry a role on the micro_task itself.
        _resolved_role = role
        if not _resolved_role:
            _resolved_role = "builder" if mt_type == "code" else "commander"
        role_cfg = ROLES.get(_resolved_role, ROLES["builder"])
        tool_block = _build_tool_restrictions_block(
            role_cfg.get("tools", set()),
            first_action=role_cfg.get("first_action", ""),
        )

        return (
            f"TASK: {work_order}\n\n"
            f"TOOL RESTRICTIONS:\n{tool_block}\n\n"
            f"CONTEXT:\n"
            f"  Chain goal: {goal}\n"
            f"  Architecture: {arch_summary}\n\n"
            f"PREVIOUSLY COMPLETED IN THIS PHASE:\n{prev_text}\n\n"
            f"CONSTRAINTS:\n"
            f"  - {file_scope_text}\n"
            f"{workspace_constraint}"
            f"{ownership_nudge}"
            f"  - Budget: 15 iterations — use them efficiently\n"
            f"  - DO NOT start servers or background processes (unless type=command)\n"
            f"  - Call finish() with a clear summary of exactly what you did and any key\n"
            f"    facts (ports, credentials, file paths) for future agents\n\n"
            f"Produce your first thought and tool call as JSON."
        )

    # ------------------------------------------------- snapshot / revert ----

    def _snapshot_files(self, file_scope: List[str]) -> Dict[str, Optional[str]]:
        """Return {path: content_or_None}. None = file didn't exist (will be deleted on revert)."""
        snap: Dict[str, Optional[str]] = {}
        for path in file_scope:
            try:
                with open(os.path.expanduser(path), "r") as f:
                    snap[path] = f.read()
            except FileNotFoundError:
                snap[path] = None
        return snap

    def _restore_snapshot(self, snapshot: Dict[str, Optional[str]]) -> None:
        """Restore files to their pre-snapshot state."""
        for path, content in snapshot.items():
            expanded = os.path.expanduser(path)
            if content is None:
                try:
                    os.remove(expanded)
                except Exception:
                    pass
            else:
                try:
                    with open(expanded, "w") as f:
                        f.write(content)
                except Exception:
                    pass

    # ------------------------------------------------- TDA helpers ----------

    def _is_tda_eligible(self, micro_task: Dict) -> bool:
        """TDA applies only to code micro-tasks with .py files in scope."""
        return (
            micro_task.get("type") == "code"
            and any(f.endswith(".py") for f in micro_task.get("file_scope", []))
        )

    def _run_tda_code_task(
        self,
        micro_task: Dict,
        chain_context: Dict,
        previous_reports: List[Dict],
    ) -> Dict:
        """Run QA → Coder → Validator loop with up to 2 Coder retries."""
        work_order = micro_task.get("work_order", "")
        file_scope = micro_task.get("file_scope", [])

        # Derive test file path (e.g. foo.py → <same_dir>/test_foo.py)
        py_files = [f for f in file_scope if f.endswith(".py")]
        if py_files:
            base = os.path.basename(py_files[0])
            test_file = os.path.join(os.path.dirname(py_files[0]), f"test_{base}")
        else:
            test_file = "test_feature.py"

        # Step 1: QA minion writes test file
        qa_prompt = self._build_minion_prompt(
            work_order=(
                f"Write a pytest test file at {test_file} for this feature:\n{work_order}\n\n"
                f"Tests MUST initially be failing (the implementation doesn't exist yet). "
                f"Cover at least 2 behaviours. Use only standard library + pytest."
            ),
            chain_context=chain_context,
            previous_reports=previous_reports,
            file_scope=[test_file],
            mt_type="code",
        )
        self.agent.react_trace = []
        self.agent.tool_registry.reset_phase_state()
        self.agent.run_react(qa_prompt, tool_whitelist=self.CODER_TOOLS, max_iterations=10)

        test_output = ""
        coder_success = False
        finish_summary = ""

        for attempt in range(3):  # Coder + up to 2 retries
            # Step 2: Coder minion
            retry_ctx = (
                f"\n\nPREVIOUS TEST FAILURE (attempt {attempt}):\n{test_output}"
                if test_output else ""
            )
            coder_prompt = self._build_minion_prompt(
                work_order=work_order + retry_ctx,
                chain_context=chain_context,
                previous_reports=previous_reports,
                file_scope=file_scope,
                mt_type="code",
            )
            self.agent.react_trace = []
            self.agent.tool_registry.reset_phase_state()
            coder_result = self.agent.run_react(
                coder_prompt, tool_whitelist=self.CODER_TOOLS, max_iterations=15
            )
            finish_summary = coder_result.get("finish_summary", "")

            # Step 3: Validator commander runs pytest
            validator_prompt = self._build_minion_prompt(
                work_order=(
                    f"Run: python -m pytest {test_file} -v --tb=short 2>&1 | head -60\n"
                    f"Report pass or fail."
                ),
                chain_context=chain_context,
                previous_reports=previous_reports,
                file_scope=[],
                mt_type="command",
            )
            self.agent.react_trace = []
            self.agent.tool_registry.reset_phase_state()
            val_result = self.agent.run_react(
                validator_prompt, tool_whitelist=self.COMMANDER_TOOLS, max_iterations=5
            )
            test_output = val_result.get("finish_summary", "")

            if "passed" in test_output.lower() and "failed" not in test_output.lower():
                coder_success = True
                break

        return {
            "micro_task_index": micro_task.get("index", 0),
            "type": "code",
            "work_order": work_order,
            "success": coder_success,
            "finish_summary": finish_summary + (
                f" | Tests: {test_output[:120]}" if test_output else ""
            ),
            "iterations_used": 0,
        }


# ---------------------------------------------------------------------------
# TaskChain
# ---------------------------------------------------------------------------

class TaskChain:
    """
    Manages chain state persisted to ~/.agent_bin/chains/<chain_id>.json.
    All saves are atomic via os.replace().

    Sub-task status values: pending | running | passed | failed | ac_failed | skipped
    Chain status values: decomposing | running | completed | failed | cancelled
    """

    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        os.makedirs(CHAINS_DIR, exist_ok=True)
        self.path = os.path.join(CHAINS_DIR, f"{chain_id}.json")
        self._data: Optional[Dict] = None

    @classmethod
    def create(
        cls,
        goal: str,
        subtasks: List[Dict],
        total_budget: int = 100,
        model: str = "qwen3-coder:30b",
        retry_policy: Optional[Dict] = None,
    ) -> "TaskChain":
        """Create a new chain and write it to disk. Returns the TaskChain instance.

        Hardening: extracts an explicit workspace path from the goal text (same
        regex as TaskDecomposer) and stores it in chain.data['workspace'] so the
        AC runner can use it as cwd. Falls back to None if the goal doesn't
        specify a workspace.
        """
        chain_id = str(uuid.uuid4())
        chain = cls(chain_id)

        ws_match = _re.search(
            r'workspace[:\s]+(/[^\s,."\']+)',
            goal,
            _re.IGNORECASE,
        )
        workspace = ws_match.group(1).rstrip('/') if ws_match else None

        if retry_policy is None:
            retry_policy = {"max_retries_per_subtask": 1}

        subtask_records = []
        for st in subtasks:
            subtask_records.append({
                "index": st["index"],
                "instruction": st["instruction"],
                "acceptance_criteria": st.get("acceptance_criteria"),
                "max_iterations": st.get("max_iterations", 25),
                "original_max_iterations": st.get(
                    "original_max_iterations", st.get("max_iterations", 25)
                ),
                "estimated_complexity": st.get("estimated_complexity", "medium"),
                "role": st.get("role", "builder"),
                "status": "pending",
                "job_id": None,
                "retry_count": 0,
                "started_at": None,
                "completed_at": None,
                "acceptance_result": None,
                "handoff": None,
                "replan_applied": False,
                "replan_reason": None,
                # PR1 — budget rebalancer + convergence bookkeeping
                "iterations_used": 0,
                "iterations_donated": 0,
                "iterations_remaining_after": None,
                "failure_count": 0,
            })

        chain._data = {
            "chain_id": chain_id,
            "goal": goal,
            "total_budget": total_budget,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "current_subtask_index": 0,
            "model": model,
            "retry_policy": retry_policy,
            "workspace": workspace,
            "subtasks": subtask_records,
        }
        chain.save()
        return chain

    @classmethod
    def load(cls, chain_id: str) -> "TaskChain":
        """Load an existing chain from disk. Raises FileNotFoundError if not found."""
        chain = cls(chain_id)
        with open(chain.path) as f:
            chain._data = json.load(f)
        return chain

    def save(self):
        """Atomically write chain state to disk."""
        os.makedirs(CHAINS_DIR, exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp_path, self.path)

    @property
    def data(self) -> Dict:
        return self._data

    def update_subtask(self, index: int, updates: Dict):
        """Merge updates into sub-task record and save atomically."""
        self._data["subtasks"][index].update(updates)
        self.save()

    # ---------------------------------------------------------- PR1 budget --
    _COMPLEXITY_RANK = {"small": 1, "medium": 2, "large": 3}

    def rebalance_budget(self, completed_index: int) -> Dict:
        """Donate slack iterations from completed tasks to a pending task with highest
        need. Bias strongly toward tasks with recorded failures so repeat offenders get
        the help they need (not just the naturally-complex phases).

        Returns a small audit dict for logging.
        """
        if os.getenv("AGENT_BUDGET_REBALANCE", "1") == "0":
            return {"skipped": True, "reason": "flag off"}

        subtasks = self._data.get("subtasks") or []
        if completed_index >= len(subtasks):
            return {"skipped": True, "reason": "bad index"}

        # Slack = sum(original - used) for completed successful tasks only
        completed_ok = [
            t for t in subtasks[: completed_index + 1]
            if t.get("status") in ("passed", "completed")
            and t.get("original_max_iterations") is not None
        ]
        slack = 0
        for t in completed_ok:
            orig = int(t.get("original_max_iterations") or t.get("max_iterations") or 0)
            used = int(t.get("iterations_used") or 0)
            slack += max(0, orig - used)
        donation = int(slack * 0.5)
        if donation <= 0:
            return {"skipped": True, "reason": "no slack"}

        pending = [t for t in subtasks[completed_index + 1:]
                   if t.get("status") in ("pending", None)]
        if not pending:
            return {"skipped": True, "reason": "no pending tasks"}

        def need_score(t: Dict) -> float:
            complexity = self._COMPLEXITY_RANK.get(
                t.get("estimated_complexity", "medium"), 2
            )
            instr = (t.get("instruction") or "").lower()
            # Rough proxy for how many files this phase will touch — count path-shaped tokens
            file_factor = max(1, len(_re.findall(r'/[\w./-]+\.\w{1,5}', instr)))
            integration = any(k in instr for k in
                              ("server", "migration", "integration", "deploy"))
            integration_factor = 1.0 if integration else 0.7
            failures = int(t.get("failure_count") or 0)
            # 1.5× multiplier per failure (capped at 6 so it can't explode).
            # Rationale: we want failing phases to dominate over merely-complex ones.
            failure_multiplier = 1.5 ** min(failures, 6)
            return complexity * file_factor * integration_factor * failure_multiplier

        target = max(pending, key=need_score)
        cap = 2 * int(target.get("original_max_iterations") or target.get("max_iterations", 25))
        headroom = max(0, cap - int(target.get("max_iterations", 0)))
        actual_donation = min(donation, headroom)
        if actual_donation <= 0:
            return {"skipped": True, "reason": "target at cap"}
        target["max_iterations"] = int(target.get("max_iterations", 0)) + actual_donation
        target["iterations_donated"] = int(target.get("iterations_donated", 0)) + actual_donation
        self.save()
        return {
            "donation": actual_donation,
            "to_index": target.get("index"),
            "target_score": need_score(target),
            "slack": slack,
        }

    def update_chain(self, updates: Dict):
        """Merge updates into top-level chain record and save atomically."""
        self._data.update(updates)
        self.save()

    @classmethod
    def list_all(cls) -> List[Dict]:
        """Return summary dicts for all chains on disk, newest first."""
        os.makedirs(CHAINS_DIR, exist_ok=True)
        chains = []
        for fname in os.listdir(CHAINS_DIR):
            if not fname.endswith(".json") or fname.endswith(".tmp"):
                continue
            try:
                chain_id = fname[:-5]
                chain = cls.load(chain_id)
                d = chain.data
                chains.append({
                    "chain_id": d["chain_id"],
                    "goal": d["goal"][:100],
                    "status": d["status"],
                    "created_at": d["created_at"],
                    "completed_at": d.get("completed_at"),
                    "subtask_count": len(d.get("subtasks", [])),
                    "current_subtask_index": d.get("current_subtask_index", 0),
                })
            except Exception:
                continue
        chains.sort(key=lambda x: x["created_at"], reverse=True)
        return chains
