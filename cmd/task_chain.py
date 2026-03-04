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
import subprocess
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ollama_agent_core import CommandSafetyValidator

CHAINS_DIR = os.path.expanduser("~/.agent_bin/chains")
INCOMPLETE_TASK_PATH = os.path.expanduser("~/.agent_bin/last_incomplete_task.json")


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
  "max_iterations": {per_phase_hint}
}}

Rules:
- Order tasks so each builds on the previous
- acceptance_criteria must be a simple verifiable shell command
  (e.g. "systemctl is-active nginx", "test -f /etc/nginx/nginx.conf", "curl -sf http://localhost")
- Set max_iterations proportional to complexity: simpler phases get fewer, complex phases get more
- Total max_iterations across all tasks must sum to approximately {total_budget}
- Aim for {target_phases} sub-tasks; prefer fewer larger phases over many tiny ones

Return ONLY the JSON array, no other text."""

        system = "You are a task decomposer. Return only a JSON array of sub-tasks. No prose."

        try:
            raw = self.agent._call_model_oneshot(
                self.agent.fast_model, prompt, system, timeout=45
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
