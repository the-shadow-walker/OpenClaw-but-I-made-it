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
        "tools": {"read_file", "create_file", "web_search", "memory_lookup", "finish"},
        "system_prefix": (
            "You are the PLANNER. Read existing code/docs and write a plan file.\n"
            "Do NOT write implementation code. Do NOT run commands.\n"
            "Output one markdown plan/spec file using create_file, then finish() "
            "with files_created listing the plan file path."
        ),
        "first_action": "read_file",
        "single_minion": True,   # skip micro-task decomposition
    },
    "builder": {
        "tools": {"read_file", "create_file", "patch_file", "finish", "write_plan"},
        "system_prefix": (
            "You are the BUILDER. Write and modify code files only.\n"
            "Do NOT run servers or execute shell commands.\n"
            "Your FIRST action MUST be write_plan — write a full markdown plan covering:\n"
            "  ## Architecture, ## Files (- [ ] /path — description), ## Dependencies\n"
            "After each file is written, re-call write_plan with that item checked (- [x]).\n"
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
}


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

    def run(self, command: str, timeout: int = 30) -> Dict:
        """
        Run acceptance criteria command.
        Returns dict with passed, exit_code, stdout, stderr, command, checked_at.
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
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
            )
            return {
                "passed": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:500],
                "command": command,
                "checked_at": checked_at,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "command": command,
                "checked_at": checked_at,
            }
        except Exception as e:
            return {
                "passed": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "command": command,
                "checked_at": checked_at,
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

        if workspace:
            arch_path     = f"{workspace}/DOCS/ARCH.md"
            arch_ac       = f"test -f {arch_path}"
            phase0_instr  = (
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
            arch_path     = "DOCS/ARCH.md"
            arch_ac       = "test -f DOCS/ARCH.md"
            phase0_instr  = (
                "Create DOCS/ARCH.md with: module list, DB schema (if needed), "
                "API route table, port assignments, auth approach, and file layout. "
                "No code — specification only."
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

        try:
            raw = self.agent._call_model_oneshot(
                self.agent.fast_model, prompt, system, timeout=180
            )
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
        for task in subtasks:
            if not task.get("max_iterations"):
                complexity = task.get("estimated_complexity", "medium")
                task["max_iterations"] = self.COMPLEXITY_DEFAULTS.get(complexity, 25)
            task.setdefault("acceptance_criteria", None)
            task.setdefault("estimated_complexity", "medium")
            task.setdefault("role", "builder")  # default if LLM omitted the field

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

    CODER_TOOLS = {"read_file", "create_file", "patch_file", "finish", "write_plan"}
    COMMANDER_TOOLS = {"execute_command", "manage_server", "read_file", "finish"}

    def __init__(self, agent):
        """agent: shared OllamaCommandAgent instance (history cleared per minion)."""
        self.agent = agent

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

        # ── Auto-tester gate: run inline verification after every builder phase ──
        if role == "builder" and artifact.status in ("completed", "partial"):
            tester_summary = self._run_inline_tester(artifact, chain_context, subtask_index)
            if tester_summary.get("failed", 0) > 0 or not tester_summary.get("success", True):
                artifact.notes.append(
                    f"Tester gate detected issues: {tester_summary.get('summary', '')[:200]}"
                )
                artifact.status = "partial"
                print(f"  ⚠️  Tester gate: issues detected — marked phase as partial")
            else:
                print(f"  ✅ Tester gate: all checks passed")

        # ── Sidechain transcript: write full trace to disk, zero out in artifact ──
        self._write_sidechain(subtask_index, role)

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

        # Build the prompt using _build_minion_prompt, then prepend role's system prefix
        mt_type = "code" if role in ("planner", "builder") else "command"
        base_prompt = self._build_minion_prompt(
            work_order=subtask_instruction,
            chain_context=chain_context,
            previous_reports=[],
            file_scope=[],
            mt_type=mt_type,
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

        return ImplementationArtifact(
            subtask_index=subtask_index,
            subtask_instruction=subtask_instruction,
            status="completed" if success else "partial",
            summary=finish_summary[:500] or f"Phase {subtask_index} ({role}) complete",
            files_created=files_created,
            files_modified=files_modified,
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
        if not check_cmds:
            return {"success": True, "failed": 0, "summary": "no .py/.js files to check"}

        all_cmds = install_cmds + check_cmds
        tester_instruction = (
            f"Run dependency install then syntax checks for files created in phase {subtask_index}.\n"
            "Run each command in order and report PASS or FAIL for each syntax check:\n"
            + "\n".join(f"  {cmd}" for cmd in all_cmds)
            + "\n\nfinish() with test_results summary. Set success=false if any syntax check FAILs."
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
        failed_count = summary.lower().count("fail") if summary else 0
        return {
            "success": result.get("success", False),
            "failed": failed_count,
            "summary": summary[:300],
        }

    def _write_sidechain(self, subtask_index: int, role: str) -> None:
        """Write the agent's react_trace to a sidechain jsonl file and zero it out.
        This keeps the chain parent context free of full minion traces.
        """
        try:
            os.makedirs(SIDECHAIN_DIR, exist_ok=True)
            ts = int(time.time())
            fname = f"subtask_{subtask_index:02d}_{role}_{ts}.jsonl"
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
        try:
            raw = self.agent._call_model_oneshot(
                self.agent.fast_model, prompt, system, timeout=120
            )
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

        if mt_type == "code":
            tool_block = (
                "YOUR ONLY TOOLS (others are HARD-BLOCKED and will error):\n"
                "  ✅ write_plan   — write your task plan (FIRST action — required)\n"
                "  ✅ read_file    — read an existing file\n"
                "  ✅ create_file  — write a new file (or overwrite)\n"
                "  ✅ patch_file   — make a targeted edit to an existing file\n"
                "  ✅ finish       — call when done (only after all plan items checked off)\n"
                "  ❌ execute_command — BLOCKED. Do NOT attempt it.\n"
                "  ❌ memory_lookup  — BLOCKED. Do NOT attempt it.\n"
                "  ❌ web_search     — BLOCKED. Do NOT attempt it.\n"
                "  ❌ manage_server  — BLOCKED. Do NOT attempt it.\n"
                "\n"
                "CRITICAL: Your FIRST tool call MUST be write_plan.\n"
                "Write a plan with ## Architecture, ## Files (- [ ] checkboxes), ## Dependencies.\n"
                "After each file is written, re-call write_plan with that item checked (- [x]).\n"
                "Do NOT call finish() until every [ ] item is checked off in your plan."
            )
        else:
            tool_block = (
                "YOUR ONLY TOOLS (others are HARD-BLOCKED and will error):\n"
                "  ✅ execute_command — run shell commands\n"
                "  ✅ manage_server   — start/stop/restart named services\n"
                "  ✅ read_file       — read an existing file\n"
                "  ✅ finish          — call when done\n"
                "  ❌ create_file  — BLOCKED. Do NOT attempt it.\n"
                "  ❌ patch_file   — BLOCKED. Do NOT attempt it.\n"
                "  ❌ memory_lookup— BLOCKED. Do NOT attempt it.\n"
                "  ❌ web_search   — BLOCKED. Do NOT attempt it.\n"
                "\n"
                "CRITICAL: Your FIRST tool call MUST be execute_command."
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
        """Create a new chain and write it to disk. Returns the TaskChain instance."""
        chain_id = str(uuid.uuid4())
        chain = cls(chain_id)

        if retry_policy is None:
            retry_policy = {"max_retries_per_subtask": 1}

        subtask_records = []
        for st in subtasks:
            subtask_records.append({
                "index": st["index"],
                "instruction": st["instruction"],
                "acceptance_criteria": st.get("acceptance_criteria"),
                "max_iterations": st.get("max_iterations", 25),
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
