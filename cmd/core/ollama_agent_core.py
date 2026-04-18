#!/usr/bin/env python3
# =============================================================================
# ollama_agent_core.py  —  v3.0
# Features in this build:
#   - Single model: qwen3-coder:30b (ReAct loop + code gen)
#   - num_ctx: react loop=32768, all one-shot calls=8192 (no KV cache bloat)
#   - react timeout: 180s; one-shot calls use their own per-call timeout
#   - Chain mode CLI: --budget/-b, --yes/-y  (TaskDecomposer multi-phase)
#   - Persistent checklist: ~/.agent_bin/checklist.md (written from plan)
#   - Progress checkpointing: ~/.agent_bin/progress.md (every N iterations)
#   - JSON cascade rescue: heavy model steps in after 5 consecutive failures
#   - History window: 20 messages (first + last 19) — keeps context under 32k
#   - Observation caps: success=800 chars, failure stdout/stderr=1000 chars each
#   - AI auto-confirm: fast model safety-screens commands, only escalates UNSAFE
#   - Heavy model patch: generates both search+replace when search is empty
#   - False "binary not found" diagnosis fixed (ENOENT != command not found)
#   - PostRunVerifier: QA agent runs after finish (writes tests, runs them,
#     heavy model produces PASS/FAIL report with fix plan + improvements)
#   - Line-buffered stdout: live output works correctly when piped to tee
#   - v3.0: Premature-finish threshold capped at max(20, min(max_iter//4, 100))
#   - v3.0: File write cap (3 per path) — blocks rewrites without concrete evidence
#   - v3.0: Failed command blocking (2 identical failures → blocked)
#   - v3.0: Forward progress detection (warn @20 idle iters, abort @35)
#   - v3.0: Mechanical syntax check (py_compile / node --check) after create_file
#   - v3.0: Hard phase gate — chain aborts on acceptance check failure
#   - v3.0: manage_server tool — tracks Popen handles, prevents zombie processes
# =============================================================================
import argparse
import subprocess
import json
import os
import re
import time
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from react_memory import AgentMemory
from react_tools import ToolRegistry
try:
    import debug_logger as _debug_logger
except ImportError:
    _debug_logger = None


# ---------------------------------------------------------------------------
# ReAct system prompt — filled with {os_info} and {max_iterations}
# ---------------------------------------------------------------------------
REACT_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent running on {os_info}.
You can perform system administration AND software development tasks.

You operate in a Reason-Act-Observe loop. Each iteration you MUST output ONLY a
JSON object — no prose, no markdown fences, nothing else.

REQUIRED JSON SCHEMA (all 4 fields mandatory every iteration):
{{
  "thought": "chain-of-thought reasoning about what to do next",
  "confidence": 92,
  "tool": "execute_command",
  "args": {{ "command": "systemctl status nginx", "timeout": 15 }}
}}

AVAILABLE TOOLS:
{available_tools}

GENERAL RULES:
1. Output ONLY valid JSON — no text before or after
2. All 4 fields (thought, confidence, tool, args) are REQUIRED every iteration
3. Only call `finish` when EVERY requirement of the original task is fully implemented
   and verified, OR you have genuinely reached an unrecoverable dead end.
   A server starting does NOT mean the task is done — check ALL feature requirements.
   If you call finish(success=true) with unused budget remaining, you will be asked
   to justify it; be prepared to continue if anything is still outstanding.
4. Set confidence < 90 to trigger human confirmation for risky actions
5. confidence >= 95 AND risk safe/low → auto-execute without confirmation
6. Call `memory_lookup` before attempting tasks you may have done before
7. You have a maximum of {max_iterations} iterations — use them wisely
8. When a command fails, analyse the error output before retrying
9. Do NOT repeat the same failed action more than ONCE. On the second failure, take a
   fundamentally different approach or call memory_lookup for alternatives.
10. NEVER use reboot, shutdown, poweroff, halt, or init 0/6 — these are hard-blocked
11. For package manager commands (pacman, yay, pip, npm, cargo), set timeout: 300
12. Before installing any package: run `pacman -Q <pkg> 2>/dev/null && echo already_installed`
    or `command -v <binary>`. NEVER install a package that is already present.
13. NEVER use flags that block on stdin: --pwprompt, --interactive, password prompts.
    For PostgreSQL passwords use:
      sudo -u postgres psql -c "ALTER USER <user> PASSWORD '<pw>';"
14. --noconfirm is injected automatically for pacman. For pip use --quiet; for npm --yes.
15. At iteration >= 80% of max_iterations: if the core task is not complete, call finish
    with success=false, listing exactly what was done and what remains. Do not continue
    burning iterations on peripheral setup. (80% threshold gives more room for large tasks.)
16. When OBSERVATION shows ❌ FAILURE: read the DIAGNOSIS and the full STDOUT/STDERR
    before deciding your next action. The diagnosis identifies the root cause — follow it.
17. To start a server (uvicorn, gunicorn, node, nginx): ALWAYS background it with
    stdin explicitly closed to prevent subprocess pipe hangs:
      nohup uvicorn main:app --host 127.0.0.1 --port 8000 </dev/null >/tmp/server.log 2>&1 &
    CRITICAL: include </dev/null BEFORE the output redirect, or the command will hang.
    NEVER use --reload (it spawns child processes that hold pipes open and cause timeouts).
    After backgrounding, confirm with a separate execute_command: sleep 2 && curl -sf http://localhost:8000/
    If curl fails, read the log: cat /tmp/server.log | tail -20
17b. Prefer manage_server over nohup for persistent servers — it tracks the PID and
     lets you stop/restart cleanly. Always redirect logs in the command:
     {{"action":"start","name":"backend","command":"uvicorn main:app --port 8000 >/tmp/backend.log 2>&1"}}
     After starting, verify: execute_command 'sleep 2 && curl -sf http://localhost:8000/'
18. Wrong package names cause Python 2 SyntaxErrors inside site-packages. When you see
    a SyntaxError in .venv/lib/.../site-packages/<pkg>.py: uninstall that package
    immediately (pip uninstall <pkg> -y) then install the CORRECT Python 3 package.
    Common wrong→right mappings: jose→python-jose[cryptography], jwt→PyJWT
19. BATCH PACKAGE CHECKS: Never check packages one at a time. Check ALL deps in ONE
    command: pip show pkg1 pkg2 pkg3 2>&1 | grep -E "^(Name|WARNING)"
    Install ALL missing packages in ONE command: pip install a b c d e f
    Maximum: one batch-check iteration + one batch-install iteration.
20. CHAIN RELATED SHELL OPERATIONS with && into ONE execute_command call. Create a
    directory, write a config, and run a command in one iteration, not three.
    Bad:  mkdir /p  →  cd /p  →  pip install x   (3 wasted iterations)
    Good: mkdir -p /p && cd /p && pip install x   (1 efficient iteration)
21. TRUST SUCCESSES: After ✅ SUCCESS, do NOT re-verify it. If mkdir succeeded, it
    exists. If "Successfully installed" appeared, the package is installed.
    Move directly to the next logical step — never echo/ls/cat to confirm.
22. SKIP REDUNDANT STEPS: Only run verification commands when they are the actual
    acceptance test (e.g. curl the live endpoint). Never verify intermediate steps.
23. NEVER use `echo` to write multi-line file content. `echo 'line1\nline2'` writes
    LITERAL backslashes, not newlines. For any file content, use the create_file tool.
    For a single-line overwrite only: printf 'content\n' > file
24. NEVER start a server until you have verified the main application file has actual
    code. Always read_file the entry point first. An empty or skeleton file will fail.
    Correct order: write all code → verify syntax → start server → test endpoints.
25. NEVER clone GitHub templates or copy starter repos. ALWAYS build from scratch using
    create_file. Cloned repos have wrong structure, missing deps, and mismatched configs.
26. ALWAYS use paths from SYSTEM CONTEXT (home dir, user, active project dirs). NEVER
    guess or invent paths. If a web search, template, or example suggests a path that
    doesn't match SYSTEM CONTEXT, ignore it and use the real path from context.
27. For each concern, pick EXACTLY ONE library and be consistent throughout:
    - One web framework (FastAPI, Flask, Express, Django — not two)
    - One ORM/DB client (SQLAlchemy, Tortoise, Prisma — not two)
    - One auth library (python-jose, PyJWT — not two)
    Never install competing libraries for the same job in the same project.
28. EACH execute_command runs in a NEW bash subprocess. `cd`, `source`, `export`, and
    `activate` do NOT persist to the next iteration. Always use ABSOLUTE PATHS:
      Wrong: cd /project && python3 main.py   [cd is forgotten next iteration]
      Right: python3 /project/main.py         [works from anywhere]
      Wrong: .venv/bin/python3                [relative — depends on cwd]
      Right: /home/user/project/.venv/bin/python3  [absolute — always works]
29. After create_file succeeds, your NEXT tool call MUST be read_file on that same path.
    You need to know exactly what routes, functions, and imports the code generator wrote
    before you can test it. Never curl an endpoint without first reading the source file.
30. To kill a process on a port, NEVER use `kill $(lsof -t -i:PORT)` — lsof can hang.
    Use instead: `pkill -f 'uvicorn.*:PORT' 2>/dev/null || true` (safe, won't hang)
    Then wait a moment before starting: sleep 1 && nohup ... </dev/null >/tmp/log 2>&1 &
31. NEVER run framework "init" or "scaffold" CLI commands — most don't exist:
    ❌ python -m fastapi init, django-admin startproject, flask new, express init
    ✅ Create project structure manually using create_file for each file needed.
    If a package error says "install X to use Y command", create the files manually instead.

CODE / FILE GENERATION RULES:
10. For create_file: ALWAYS set "content": "" and write a detailed "description"
    explaining exactly what the file should contain. A separate code-generation
    model will write the actual content — do NOT attempt to write it yourself.
    Example: {{"path": "/var/www/html/index.html", "content": "",
               "description": "Full portfolio page: hero, case studies, contact form, 2-color CSS, no JS libs, WCAG AA"}}
11. For patch_file: use description mode for non-trivial changes.
    - If you know the exact text to replace: set "search": "<exact text>", "replace": "", "description": "<what to do>"
    - If you are NOT sure of the exact text (e.g. after many failed patches): set BOTH "search": "" AND "replace": "" with a detailed "description". The heavy model will read the file and locate the correct section itself.
    - For tiny changes (e.g. a port number or single word) you MAY write both inline.
    Example (unsure of search): {{"path": "app.py", "search": "", "replace": "",
               "description": "Change port to 8443 and add SSL context"}}
12. Always `read_file` before modifying any source file
13. After editing code, run it with `execute_command` to verify it works
14. For Python: use the venv python if a .venv exists (e.g. .venv/bin/python3)
15. Read error tracebacks carefully — they tell you the exact file and line
16. When debugging: read → identify → patch → run → verify in a tight loop
17. NEVER write markdown code fences (```, ```python, ```bash) inside file content.
    The file must contain ONLY raw code. Fences cause SyntaxError crashes in Python.
18. After patching a Python file always verify syntax BEFORE running it:
      python3 -c "import ast; ast.parse(open('/path/to/file.py').read()); print('syntax ok')"
"""

# Per-tool schema lines used to build the AVAILABLE TOOLS section dynamically.
# When tool_whitelist is set in run_react(), only whitelisted tools are shown.
TOOL_SCHEMAS: dict = {
    "execute_command":  '  execute_command  — {{"command": str, "timeout": int}}',
    "create_file":      '  create_file      — {{"path": str, "content": str, "description": str}}',
    "patch_file":       '  patch_file       — {{"path": str, "search": str, "replace": str, "description": str}}',
    "web_search":       '  web_search       — {{"query": str}}',
    "read_file":        '  read_file        — {{"path": str, "offset": int (opt, default 0), "limit": int (opt, default 200 lines)}}',
    "memory_lookup":    '  memory_lookup    — {{"query": str}}',
    "finish":           '  finish           — {{"summary": str, "success": bool}}',
    "manage_server":    '  manage_server    — {{"action": "start|stop|status|restart", "name": str, "command": str}}',
}
_ALL_TOOLS_TEXT = "\n".join(TOOL_SCHEMAS.values())


class FlexibleSearchAgent:
    """Simplified search agent for the command agent"""

    def __init__(self, searxng_url: str = "http://10.0.0.58:8080", timeout: int = 15):
        self.searxng_url = searxng_url
        self.timeout = timeout

    def search(self, query: str) -> str:
        """Search and return formatted results"""
        print(f"\n🔍 Searching: {query}")

        try:
            encoded_query = urllib.parse.quote(query)
            search_url = f"{self.searxng_url}/search?q={encoded_query}&format=json&categories=general"

            result = subprocess.run(
                ["curl", "-s", "-H", "Accept: application/json", search_url],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if not result.stdout.strip():
                return "No search results available."

            data = json.loads(result.stdout)
            results = data.get("results", [])[:5]

            if not results:
                return "No results found."

            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(
                    f"{i}. {r.get('title', 'No title')}\n"
                    f"   {r.get('content', 'No description')[:200]}\n"
                    f"   {r.get('url', '')}"
                )

            print(f"✅ Found {len(results)} results")
            return "\n\n".join(formatted)

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return f"Search unavailable: {e}"


class CommandSafetyValidator:
    """Validates commands for safety before execution"""

    SAFE_COMMANDS = {
        "ls", "pwd", "whoami", "date", "echo", "cat", "grep", "find", "which",
        "ps", "top", "df", "du", "free", "uptime", "uname", "hostname",
        "curl", "wget", "ping", "netstat", "ss", "lsof", "ip",
        "pacman", "yay", "systemctl", "journalctl",
        "python3", "python", "node", "npm",
        "mkdir", "touch", "cp", "mv",
        "nmap", "lynis", "clamav", "audit", "tee",
    }

    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\s+/$",
        r"\brm\s+-rf\s+/\s*$",
        r"\brm\s+-rf\s+/\*",
        r"\b>\s*/dev/sd[a-z]",
        r"\bdd\s+.*of=/dev/sd[a-z]",
        r"\bmkfs\.",
        r"\bfdisk\b",
        r"\bparted\b",
        r"\b:(){\s*:\|:&\s*};:",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bchmod\s+777\s+/",
        r"\bchown\s+.*:.*\s+/$",
        # ---- ABSOLUTE NO: reboot / shutdown / poweroff (no confirmation, ever) ----
        r"^\s*(sudo\s+)?reboot\b",
        r"^\s*(sudo\s+)?shutdown\b",
        r"^\s*(sudo\s+)?poweroff\b",
        r"^\s*(sudo\s+)?halt\b",
        r"\binit\s+[06]\b",
        r"\bsystemctl\s+(reboot|shutdown|poweroff|halt)\b",
    ]

    PROTECTED_PATHS = {
        "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64",
        "/proc", "/root", "/sbin", "/sys", "/usr",
    }

    SAFE_VAR_OPERATIONS = [
        r"rm\s+/var/lib/pacman/db\.lck",
        r"rm\s+-f\s+/var/lib/pacman/db\.lck",
        r"rm\s+/var/cache/pacman/pkg/.*\.pkg\.tar\.",
    ]

    # Regex that matches a package-installation invocation
    _INSTALL_PATTERN = re.compile(
        r"\b(pacman|yay|paru)\s+.*-[A-Za-z]*S"
        r"|\bpip3?\s+install\b"
        r"|\bnpm\s+install\b"
        r"|\byarn\s+add\b"
    )

    @classmethod
    def validate_command(cls, command: str) -> Tuple[bool, str, str]:
        """
        Validate a command for safety.

        Returns:
            (is_safe, risk_level, reason)
            risk_level: 'safe', 'low', 'medium', 'high', 'blocked'
        """
        if not command or not command.strip():
            return False, "blocked", "Empty command"

        command_lower = command.lower().strip()

        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return False, "blocked", f"Dangerous pattern: {pattern}"

        if re.search(r"\brm\b", command):
            is_safe_var = any(
                re.search(p, command) for p in cls.SAFE_VAR_OPERATIONS
            )
            if is_safe_var:
                return True, "low", "Removing lock file or clearing cache"
            for path in cls.PROTECTED_PATHS:
                if re.search(rf"\brm\b.*{re.escape(path)}(?:/|$)", command):
                    return False, "blocked", f"Attempting to remove protected path: {path}"
            return True, "medium", "File deletion command"

        if re.match(r"^\s*sudo\s*$", command):
            return False, "blocked", "sudo without command"

        first_word = command_lower.split()[0] if command_lower.split() else ""
        for prefix in ["sudo", "nohup", "time"]:
            if first_word == prefix and len(command_lower.split()) > 1:
                first_word = command_lower.split()[1]
                break

        if first_word in cls.SAFE_COMMANDS:
            if cls._INSTALL_PATTERN.search(command):
                return True, "low", "Package installation"
            if "systemctl start" in command or "systemctl enable" in command:
                return True, "medium", "System service modification"
            return True, "safe", "Safe read-only command"

        if "http.server" in command or re.search(r":\d{4,5}", command):
            if "--bind 0.0.0.0" in command or "-b 0.0.0.0" in command:
                return True, "medium", "Network-accessible server"
            return True, "low", "Local server"

        if re.search(r"\b(mkdir|touch|cp|mv)\b", command):
            if command.startswith("~") or "/home/" in command:
                return True, "low", "File operation in user directory"
            return True, "medium", "File operation in system directory"

        return True, "medium", "Unknown command pattern"


class OllamaCommandAgent:
    NUM_CTX = 16384       # ReAct loop context window
    HEAVY_NUM_CTX = 8192  # one-shot calls (code gen, explain) — short prompt in/out
    MINION_NUM_CTX = 8192 # minion agents — clean slate per micro-task

    def __init__(
        self,
        model: str = "qwen3-coder:30b",
        fast_model: str = "qwen3-coder:30b",
        searxng_url: str = "http://10.0.0.58:8080",
    ):
        # model used for both ReAct loop and code/file generation
        self.model = model
        self.fast_model = fast_model
        self.search_agent = FlexibleSearchAgent(searxng_url)
        self.safety_validator = CommandSafetyValidator()
        self.conversation_history: List[Dict] = []
        self.task_plan: List[Dict] = []
        self.execution_log: List[Dict] = []
        self.current_step = 0
        self.os_info = "Arch Linux with hardened kernel"
        self.max_retries = 3
        # ReAct fields
        self.memory = AgentMemory()
        self.tool_registry = ToolRegistry(
            self.safety_validator,
            self.search_agent,
            self.memory,
            explain_cb=self.explain_command_detailed,
        )
        self.react_trace: List[Dict] = []
        self.max_react_iterations = 50
        self.current_job_id: Optional[str] = None  # set by server before run()
        # Pinned messages: always injected after system prompt regardless of history trimming
        # (e.g. ARCH.md contents, schema.sql, key model definitions)
        self.pinned_messages: List[Dict] = []

    # ---------------------------------------------------------------- LLM --

    def _call_model_oneshot(
        self,
        model: str,
        prompt: str,
        system_prompt: str = None,
        timeout: int = 60,
    ) -> str:
        """Raw one-shot call to any Ollama model. No history, no side effects."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # One-shot calls (decomposer, planner, ai_confirm, etc.) never need the
        # full 32k window — allocating it adds seconds of overhead even for short
        # prompts. Only call_ollama_react (the ReAct loop) uses NUM_CTX=32k.
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": self.HEAVY_NUM_CTX},
        }

        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/chat",
                 "-d", json.dumps(request_data)],
                capture_output=True, text=True, timeout=timeout,
            )
            content = json.loads(result.stdout)["message"]["content"]
            # Strip qwen3-style thinking blocks so extract_json sees only the answer
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            print(f"❌ Ollama call failed ({model}): {e}")
            return ""

    def call_ollama(self, prompt: str, system_prompt: str = None, timeout: int = 60) -> str:
        """Call Ollama API and get response (uses/updates self.conversation_history)."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if "Explain this command" not in prompt:
            messages.extend(self.conversation_history[-10:])

        messages.append({"role": "user", "content": prompt})

        request_data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": self.HEAVY_NUM_CTX},
        }

        curl_cmd = [
            "curl", "-s", "http://localhost:11434/api/chat",
            "-d", json.dumps(request_data),
        ]

        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=timeout)

        try:
            response = json.loads(result.stdout)
            assistant_message = response["message"]["content"]
        except Exception:
            print("❌ Error calling Ollama API. Is Ollama running?")
            return ""

        if "Explain this command" not in prompt:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def call_ollama_react(
        self,
        react_history: List[Dict],
        system_prompt: str,
        timeout: int = 180,
    ) -> str:
        """Fast-model (14b) caller for the ReAct decision loop.
        Only picks tools and writes short args — never generates file content.
        Does NOT touch self.conversation_history.
        """
        messages = [{"role": "system", "content": system_prompt}]
        # Inject pinned architectural facts (ARCH.md, schema, models) after system prompt
        # so they're always visible regardless of history trimming.
        if self.pinned_messages:
            messages.extend(self.pinned_messages)
        messages.extend(react_history)

        request_data = {
            "model": self.fast_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": self.NUM_CTX},
        }

        try:
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/chat",
                 "-d", json.dumps(request_data)],
                capture_output=True, text=True, timeout=timeout,
            )
            return json.loads(result.stdout)["message"]["content"]
        except Exception as e:
            print(f"❌ Ollama ReAct call failed ({self.fast_model}): {e}")
            return ""

    def call_ollama_heavy(
        self,
        prompt: str,
        system_prompt: str = None,
        timeout: int = 300,
    ) -> str:
        """Heavy-model (30b) one-shot call for code/file generation only."""
        return self._call_model_oneshot(self.model, prompt, system_prompt, timeout)

    # ------------------------------------------------------------ helpers --

    def explain_command(self, command: str) -> str:
        """Get a 1-2 sentence explanation of what a command does."""
        system_prompt = """You are a Linux command explainer. Explain what the given command does in 1-2 clear sentences. Be concise and focus on the actual effect of the command.

Do not include:
- Safety warnings
- Alternative suggestions
- Additional context

Just explain what it does."""

        prompt = f"Explain this command in 1-2 sentences:\n\n{command}"
        response = self.call_ollama(prompt, system_prompt, timeout=30)

        if response:
            response = response.strip()
            response = re.sub(r'^["\']|["\']$', "", response)
            response = re.sub(r"\n+", " ", response)
            return response[:200]

        return "Command explanation unavailable"

    def explain_command_detailed(self, command: str) -> str:
        """Break a command into segments and explain each one.
        Used by ToolRegistry to show what a command does before executing it.
        Does NOT update conversation history.
        """
        system_prompt = (
            "You are a command explainer. Break a shell command into its logical "
            "segments and explain each one briefly in plain English. "
            "Return ONLY valid JSON, nothing else."
        )

        prompt = (
            f"Explain this command segment by segment.\n\n"
            f"Command: {command}\n\n"
            f"Return JSON only:\n"
            f'{{"summary": "one sentence overall description",\n'
            f' "parts": [{{"segment": "cd ~/cmd", "explanation": "change to ~/cmd directory"}}, ...]}}'
        )

        # Use the fast model — called on every command so latency matters
        response = self._call_model_oneshot(
            self.fast_model, prompt, system_prompt, timeout=20
        )

        data = self.extract_json(response)
        if not data or not isinstance(data, dict):
            # Graceful fallback — still useful without LLM breakdown
            return f"  $ {command}"

        lines = []
        summary = data.get("summary", "")
        if summary:
            lines.append(f"  💬 {summary}")
            lines.append("")

        for part in data.get("parts", []):
            seg = part.get("segment", "")
            exp = part.get("explanation", "")
            if seg and exp:
                lines.append(f"    \033[1m{seg:<35}\033[0m {exp}")

        return "\n".join(lines) if lines else f"  $ {command}"

    def extract_json(self, text: str, debug: bool = False) -> Any:
        """Extract JSON from text with multiple fallback strategies."""
        if not text or not text.strip():
            if debug:
                print("   [JSON Extract] Empty input")
            return None

        patterns = [
            r"```json\s*(\{.*?\}|\[.*?\])\s*```",
            r"```\s*(\{.*?\}|\[.*?\])\s*```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if debug:
                        print("   [JSON Extract] Success via markdown")
                    return parsed
                except Exception:
                    continue

        for start_char in ["{", "["]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            bracket_count = 0
            end_char = "}" if start_char == "{" else "]"
            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    bracket_count += 1
                elif text[i] == end_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_candidate = text[start_idx : i + 1]
                        try:
                            parsed = json.loads(json_candidate)
                            if debug:
                                print("   [JSON Extract] Success via bracket matching")
                            return parsed
                        except Exception:
                            break

        try:
            parsed = json.loads(text)
            if debug:
                print("   [JSON Extract] Success - raw JSON")
            return parsed
        except Exception:
            pass

        if debug:
            print("   [JSON Extract] All strategies failed")
        return None

    @staticmethod
    def _strip_thought_fields(msg: dict) -> dict:
        """Compress rotated-out assistant messages to just tool+args (drop thought/confidence).
        Reduces token cost for history entries that are beyond the active window.
        """
        if msg.get("role") == "assistant":
            try:
                parsed = json.loads(msg["content"])
                stripped = {"tool": parsed.get("tool"), "args": parsed.get("args")}
                return {**msg, "content": json.dumps(stripped)}
            except Exception:
                pass
        return msg

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences the heavy model adds despite instructions.
        Prevents SyntaxError when fences end up inside Python/bash files.
        """
        if not text:
            return text
        text = text.strip()
        # Remove opening fence line (```python, ```bash, ``` etc.)
        text = re.sub(r'^```[a-zA-Z0-9_-]*\r?\n', '', text)
        # Remove closing fence line
        text = re.sub(r'\n```\s*$', '', text)
        # Handle case where fences wrap everything on single-ish lines
        text = re.sub(r'^```[a-zA-Z0-9_-]*\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        return text.strip()

    # ----------------------------------------------------- legacy planning --

    def analyze_task(self, instruction: str) -> Dict[str, Any]:
        """Analyze task and identify what's needed"""
        system_prompt = f"""You are a task analyzer for {self.os_info}.

Analyze the request and determine:
1. Main goal
2. Prerequisites (software, permissions, files)
3. Whether web search is needed (ONLY if you need specific package names or unfamiliar commands)
4. Risk level (low/medium/high)

Return ONLY JSON (no explanation):
{{
    "goal": "main objective",
    "prerequisites": ["item1", "item2"],
    "needs_search": false,
    "search_query": "only if needed",
    "risk_level": "low",
    "notes": "considerations"
}}"""

        response = self.call_ollama(instruction, system_prompt)
        analysis = self.extract_json(response, debug=True)

        if not analysis:
            analysis = {
                "goal": instruction,
                "prerequisites": [],
                "needs_search": False,
                "search_query": "",
                "risk_level": "low",
                "notes": "",
            }

        return analysis

    def create_plan(self, instruction: str, analysis: Dict[str, Any], search_results: str = "") -> List[Dict[str, Any]]:
        """Create detailed execution plan - robust LLM prompting"""

        system_prompt = f"""You are a command planner for {self.os_info}. You MUST return ONLY valid JSON.

RULES:
1. Return ONLY a JSON array - no text before or after
2. Each step needs: step, description, type, command, risk, timeout
3. Commands must be actual shell commands that work on {self.os_info}

FORMAT (copy this structure exactly):
[
  {{"step": 1, "description": "what to do", "type": "execute_command", "command": "actual shell command", "risk": "low", "timeout": 30}}
]

SYSTEM INFO:
- OS: {self.os_info}
- Shell: bash
- Available: python3, curl, grep, awk, sed, ss, lsof, systemctl
- Package manager: pacman, yay

Your response must be parseable by json.loads(). Nothing else."""

        prompt = f"""Task: {instruction}

Context: {json.dumps(analysis, indent=2)}

Create a step-by-step plan. Return ONLY the JSON array.

Example for "check port 5002":
[
  {{"step": 1, "description": "Check port 5002", "type": "execute_command", "command": "ss -tuln | grep :5002", "risk": "low", "timeout": 10}}
]

Now create the plan for the task above. JSON array only:"""

        response = self.call_ollama(prompt, system_prompt, timeout=60)

        if not response or len(response) < 10:
            print("   [Planning] LLM returned empty or very short response")
            return self._create_generic_fallback(instruction)

        plan = self.extract_json(response, debug=True)

        if plan and isinstance(plan, list) and len(plan) > 0:
            valid = True
            for step in plan:
                if not isinstance(step, dict):
                    valid = False
                    break
                if "command" in step and step.get("type") == "execute_command":
                    continue
                elif step.get("type") in ["create_file", "verify"]:
                    continue
                else:
                    valid = False
                    break

            if valid:
                print(f"   [Planning] Created {len(plan)} valid steps")
                return plan
            else:
                print("   [Planning] Plan structure invalid")
        else:
            print("   [Planning] Failed to extract valid plan")

        print("   [Planning] Using generic execution fallback")
        return self._create_generic_fallback(instruction)

    def _create_generic_fallback(self, instruction: str) -> List[Dict[str, Any]]:
        """Create a generic plan when LLM fails."""
        instruction_lower = instruction.lower()

        command_keywords = [
            "ls", "ps", "df", "du", "cat", "grep", "find", "curl",
            "wget", "systemctl", "journalctl", "ss", "lsof", "netstat",
            "top", "free", "uname", "whoami", "pwd", "echo",
        ]

        first_word = instruction.split()[0].lower() if instruction.split() else ""
        if first_word in command_keywords:
            return [{
                "step": 1,
                "description": f"Execute: {instruction}",
                "type": "execute_command",
                "command": instruction,
                "risk": "medium",
                "timeout": 30,
            }]

        inspection_map = {
            "port":     "ss -tuln | grep :{port} || echo \"No service on port {port}\"",
            "process":  "ps aux | grep -i {term}",
            "service":  "systemctl status {service} 2>/dev/null || systemctl list-units | grep -i {term}",
            "disk":     "df -h",
            "memory":   "free -h",
            "file":     "ls -lah {path}",
            "running":  "ps aux --sort=-%mem | head -20",
            "listening":"ss -tuln",
            "user":     "whoami && id",
            "system":   "uname -a && uptime",
            "log":      "journalctl -n 50 --no-pager",
        }

        for keyword, command_template in inspection_map.items():
            if keyword in instruction_lower:
                command = command_template

                if "{port}" in command:
                    port_match = re.search(r"\b(\d{4,5})\b", instruction)
                    port = port_match.group(1) if port_match else "8080"
                    command = command.replace("{port}", port)

                if "{term}" in command:
                    words = instruction_lower.split()
                    if keyword in words:
                        idx = words.index(keyword)
                        term = words[idx + 1] if idx + 1 < len(words) else keyword
                    else:
                        term = keyword
                    command = command.replace("{term}", term)

                if "{service}" in command:
                    words = instruction.split()
                    service = words[-1] if words else "unknown"
                    command = command.replace("{service}", service)

                if "{path}" in command:
                    path_match = re.search(r"(/[\w/.-]+|~[\w/.-]*)", instruction)
                    path = path_match.group(1) if path_match else "."
                    command = command.replace("{path}", path)

                return [{
                    "step": 1,
                    "description": f"Check: {instruction}",
                    "type": "execute_command",
                    "command": command,
                    "risk": "low",
                    "timeout": 30,
                }]

        action_verbs = {
            "list":    "ls -lh",
            "show":    "cat",
            "display": "cat",
            "find":    "find . -name",
            "search":  "grep -r",
            "check":   "test -e",
            "create":  "mkdir -p",
            "remove":  "rm -f",
            "kill":    "pkill -f",
            "restart": "systemctl restart",
            "start":   "systemctl start",
            "stop":    "systemctl stop",
        }

        for verb, base_cmd in action_verbs.items():
            if instruction_lower.startswith(verb):
                words = instruction.split()[1:] if len(instruction.split()) > 1 else []
                obj = " ".join(words) if words else ""
                command = f"{base_cmd} {obj}".strip()
                return [{
                    "step": 1,
                    "description": instruction,
                    "type": "execute_command",
                    "command": command,
                    "risk": "medium",
                    "timeout": 30,
                }]

        simple_prompt = f"""Convert this question to a single bash command for {self.os_info}:

Question: {instruction}

Return ONLY the bash command, nothing else. One line."""

        response = self.call_ollama(
            simple_prompt,
            system_prompt="Return only the bash command.",
            timeout=30,
        )

        command = response.strip()
        command = re.sub(r"^```bash\s*|\s*```$", "", command)
        command = re.sub(r"^\$\s*", "", command)
        command = command.split("\n")[0]

        if command and len(command) < 200 and not command.startswith(("Sure", "Here", "I ", "The ")):
            return [{
                "step": 1,
                "description": f"Execute: {instruction}",
                "type": "execute_command",
                "command": command,
                "risk": "medium",
                "timeout": 30,
            }]

        return [{
            "step": 1,
            "description": "Unable to parse request",
            "type": "execute_command",
            "command": (
                f"echo 'Could not understand request: {instruction}' && "
                "echo 'Try rephrasing as a direct command or question about system state'"
            ),
            "risk": "low",
            "timeout": 5,
        }]

    def analyze_failure_and_fix(
        self,
        step: Dict[str, Any],
        result: Dict[str, Any],
        retry_count: int,
    ) -> Optional[Dict[str, Any]]:
        """Analyze failure and generate fix"""

        system_prompt = f"""Debug expert for {self.os_info}.

A command failed. Provide a corrected command.

Return ONLY JSON:
{{
    "analysis": "One sentence why it failed",
    "fixed_command": "corrected shell command"
}}

Common fixes:
- Pacman lock: sudo rm -f /var/lib/pacman/db.lck  (handled automatically on retry)
- Command not found: sudo pacman -S <package> or yay -S <package>
- Permission denied: add sudo
"""

        stderr_truncated = result.get("stderr", "")[:800]

        error_context = f"""Failed: {step.get('command', 'N/A')}
Exit: {result.get('returncode')}
Error: {stderr_truncated}

JSON only."""

        print(f"\n🔍 Analyzing failure (attempt {retry_count}/{self.max_retries})...")

        response = self.call_ollama(error_context, system_prompt)
        fix_data = self.extract_json(response, debug=True)

        if not fix_data:
            return None

        fixed_command = fix_data.get("fixed_command") or fix_data.get("command")

        if not fixed_command:
            return None

        is_safe, risk, reason = self.safety_validator.validate_command(fixed_command)

        if not is_safe:
            print(f"🛡️  BLOCKED: {reason}")
            return None

        print(f"\n💡 Analysis: {fix_data.get('analysis', 'No analysis')}")
        print(f"   Fixed Command: {fixed_command[:80]}...")
        print(f"   Risk: {risk}")

        return {
            "step": step.get("step"),
            "description": f"Fixed: {step.get('description')}",
            "type": "execute_command",
            "command": fixed_command,
            "risk": risk,
            "timeout": step.get("timeout", 30),
            "continue_on_failure": step.get("continue_on_failure", False),
        }

    def execute_command(self, command: str, risk: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a shell command with safety validation"""

        is_safe, detected_risk, reason = self.safety_validator.validate_command(command)

        if not is_safe:
            print(f"\n🛡️  BLOCKED: {reason}")
            return {"success": False, "stdout": "", "stderr": f"Blocked: {reason}", "returncode": -1}

        risk_levels = {"safe": 0, "low": 1, "medium": 2, "high": 3, "blocked": 4}
        effective_risk_level = max(
            risk_levels.get(detected_risk, 2), risk_levels.get(risk, 2)
        )
        effective_risk = list(risk_levels.keys())[effective_risk_level]

        if effective_risk in ["medium", "high"]:
            print(f"\n⚠️  {effective_risk.upper()} RISK COMMAND:")
            print(f"   Command: {command}")
            print(f"   Reason: {reason}")

            print("\n   📖 Getting explanation...")
            explanation = self.explain_command(command)
            print(f"   💬 What it does: {explanation}")

            try:
                confirm = input("\n   Execute this command? (y/n): ")
            except EOFError:
                confirm = "n"
            if confirm.lower() not in ["yes", "y"]:
                print("   ❌ Command cancelled")
                return {"success": False, "stdout": "", "stderr": "Cancelled by user", "returncode": -1}

        print(f"\n🔧 Executing: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
            )

            output = {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if output["success"]:
                print("✅ Success!")
                if output["stdout"]:
                    lines = output["stdout"].split("\n")
                    if len(lines) > 30:
                        print(f"Output (first 15 lines of {len(lines)}):")
                        print("\n".join(lines[:15]))
                        print("...")
                    else:
                        print(f"Output:\n{output['stdout']}")
            else:
                print("❌ Failed!")
                if output["stderr"]:
                    stderr_lines = output["stderr"].split("\n")
                    if len(stderr_lines) > 20:
                        print("Error (first 10 and last 10):")
                        print("\n".join(stderr_lines[:10]))
                        print("...")
                        print("\n".join(stderr_lines[-10:]))
                    else:
                        print(f"Error:\n{output['stderr']}")

            return output

        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout after {timeout}s")
            return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1}
        except Exception as e:
            print(f"❌ Exception: {e}")
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    def create_file(self, filename: str, content: str) -> bool:
        """Create a file with safety checks"""
        filename = os.path.expanduser(filename)

        protected_dirs = [
            "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64",
            "/proc", "/root", "/sbin", "/sys", "/usr/bin", "/usr/sbin",
        ]

        for protected in protected_dirs:
            if filename.startswith(protected):
                print(f"\n🛡️  BLOCKED: Cannot write to {protected}")
                return False

        if os.path.exists(filename):
            print(f"\n⚠️  File exists: {filename}")
            try:
                confirm = input("   Overwrite? (y/n): ")
            except EOFError:
                confirm = "n"
            if confirm.lower() != "y":
                print("   ❌ Cancelled")
                return False

        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)

        print(f"\n📝 Creating: {filename}")

        try:
            with open(filename, "w") as f:
                f.write(content)
            print(f"✅ Created: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False

    def execute_step(self, step: Dict[str, Any], retry_count: int = 0) -> bool:
        """Execute a single step with retry (legacy plan-execute path)."""
        print(f"\n{'=' * 70}")
        print(f"📍 Step {step.get('step', '?')}: {step['description']}")

        risk = step.get("risk", "medium")
        risk_emoji = {"safe": "🟢", "low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
        print(f"   {risk_emoji} Risk: {risk}")

        if retry_count > 0:
            print(f"   🔄 Retry {retry_count}/{self.max_retries}")

        step_type = step.get("type", "execute_command")

        if step_type == "create_file":
            success = self.create_file(step.get("filename", "output.txt"), step.get("content", ""))
            self.execution_log.append({
                "step": step.get("step"),
                "description": step["description"],
                "success": success,
                "type": step_type,
            })
            return success

        elif step_type == "execute_command":
            result = self.execute_command(
                step.get("command", ""),
                step.get("risk", "medium"),
                step.get("timeout", 30),
            )
            success = result["success"]

            self.execution_log.append({
                "step": step.get("step"),
                "description": step["description"],
                "success": success,
                "type": step_type,
                "retry_count": retry_count,
            })

            if not success and retry_count < self.max_retries:
                fixed_step = self.analyze_failure_and_fix(step, result, retry_count + 1)
                if fixed_step:
                    print("\n🔧 Applying fix...")
                    return self.execute_step(fixed_step, retry_count + 1)

            return success

        elif step_type == "verify":
            print("✅ Checkpoint")
            self.execution_log.append({
                "step": step.get("step"),
                "description": step["description"],
                "success": True,
                "type": step_type,
            })
            return True

        return False

    def verify_completion(self, instruction: str) -> bool:
        """Verify completion (legacy)."""
        unique_steps: Dict[Any, bool] = {}
        for log in self.execution_log:
            step_num = log.get("step")
            if step_num not in unique_steps or log.get("success"):
                unique_steps[step_num] = log.get("success", False)

        success_count = sum(1 for s in unique_steps.values() if s)
        total_count = len(unique_steps)

        if total_count > 0 and success_count / total_count >= 0.75:
            print(f"✅ {success_count}/{total_count} steps succeeded")
            return True
        else:
            print(f"⚠️  Only {success_count}/{total_count} succeeded")
            return False

    PROGRESS_PATH = os.path.expanduser("~/.agent_bin/progress.md")
    CHECKLIST_PATH = os.path.expanduser("~/.agent_bin/checklist.md")

    def _write_progress_md(self, instruction: str, iteration: int, max_iter: int) -> None:
        """Fast model summarises the recent trace into a human-readable progress file."""
        # Build a compact trace summary from the last 10 entries
        recent = self.react_trace[-10:] if self.react_trace else []
        trace_text = ""
        for e in recent:
            status = "✅" if (e.get("result") and e["result"].success) else "❌"
            trace_text += f"  {status} [{e['tool']}] {e.get('thought', '')[:120]}\n"

        system = (
            "You are a progress tracker for an autonomous agent. "
            "Write a concise markdown status file. "
            "Use these exact sections: "
            "## Task, ## Done, ## Current Focus, ## Struggling With, ## Still To Do, ## Notes. "
            "Be brief — max 30 lines total. No code blocks."
        )
        prompt = (
            f"Task: {instruction}\n\n"
            f"Iteration: {iteration}/{max_iter}\n\n"
            f"Recent actions:\n{trace_text}\n\n"
            f"Write the progress markdown file."
        )
        content = self._call_model_oneshot(self.fast_model, prompt, system, timeout=20)
        if not content:
            return
        try:
            os.makedirs(os.path.dirname(self.PROGRESS_PATH), exist_ok=True)
            task_fp = instruction[:120].replace("\n", " ")
            with open(self.PROGRESS_PATH, "w") as f:
                f.write(f"<!-- task: {task_fp} -->\n")
                f.write(f"<!-- updated: iteration {iteration}/{max_iter} -->\n")
                f.write(content.strip() + "\n")
            print(f"\n📝 Progress saved → {self.PROGRESS_PATH}")
        except Exception:
            pass

    # ================================================================ ReAct ==

    def _default_confirm(self, prompt: str, command_info: str = "") -> bool:
        """Interactive confirmation; returns False safely on EOFError (no TTY)."""
        try:
            answer = input(prompt)
            return answer.strip().lower() in ("y", "yes")
        except EOFError:
            return False

    def _ai_confirm(self, prompt: str, command_info: str = "") -> bool:
        """AI safety filter: fast model inspects the command with no task context.
        If it judges the command safe, auto-approve. If unsafe, ask the user."""
        if not command_info:
            return self._default_confirm(prompt)

        system = (
            "You are a security scanner for a Linux server. "
            "A command or file operation is about to run. "
            "Reply with exactly one word: SAFE or UNSAFE.\n"
            "UNSAFE means: irreversible data loss, system-wide destruction, "
            "network exfiltration, privilege escalation abuse, or overwriting "
            "critical system files. Normal installs, service restarts, "
            "writing project files, and common sysadmin tasks are SAFE."
        )
        verdict = self._call_model_oneshot(
            self.fast_model,
            f"Command/operation:\n{command_info}",
            system,
            timeout=15,
        ).strip().upper()

        if "UNSAFE" in verdict:
            print(f"\n🤖 Safety agent flagged this as potentially unsafe.")
            return self._default_confirm(prompt)
        else:
            print(f"\n🤖 Auto-approved by safety agent.")
            return True

    def _verify_react_completion(
        self,
        instruction: str,
        summary: str,
        trace: List[Dict],
    ) -> Dict[str, Any]:
        """Separate LLM call to verify whether the task was actually completed."""
        digest_lines = []
        for i, entry in enumerate(trace, 1):
            args_snippet = json.dumps(entry.get("args", {}))[:100]
            digest_lines.append(
                f"{i}. [{entry['tool']}] args={args_snippet} "
                f"success={entry['result'].success}"
            )
        digest_str = "\n".join(digest_lines) if digest_lines else "(no actions taken)"

        prompt = f"""Verify whether this task was completed successfully.

ORIGINAL TASK: {instruction}

AGENT SUMMARY: {summary}

ACTIONS TAKEN:
{digest_str}

Return JSON only:
{{
    "verified": true,
    "confidence": 85,
    "notes": "brief explanation"
}}"""

        system = (
            "You are a task verifier. Analyse whether a task was completed. "
            "Return only JSON with keys: verified (bool), confidence (int 0-100), notes (str)."
        )

        try:
            response = self.call_ollama(prompt, system, timeout=45)
            result = self.extract_json(response)
            if result and isinstance(result, dict):
                return result
        except Exception:
            pass

        return {"verified": False, "confidence": 0, "notes": "Verification call failed"}

    # ------------------------------------------ code-generation helpers ------

    def _build_context_summary(self, recent_history: List[Dict]) -> str:
        """Summarise the last few ReAct messages for injection into code-gen calls."""
        lines = []
        for msg in recent_history:
            role = msg["role"].upper()
            snippet = msg["content"][:600].replace("\n", " ")
            lines.append(f"[{role}] {snippet}")
        return "\n".join(lines)

    def _generate_file_content(
        self,
        path: str,
        description: str,
        task: str,
        context_summary: str,
    ) -> str:
        """Call the heavy (30b) model to write actual file content.
        Returns raw file content — no fences, no explanation.
        """
        ext = os.path.splitext(path)[1].lower()
        lang_hint = {
            ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
            ".html": "HTML", ".css": "CSS", ".sh": "Bash",
            ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
            ".md": "Markdown", ".sql": "SQL",
        }.get(ext, "plain text")

        print(f"\n⚙️  [{self.model}] Writing {lang_hint} → {path}")

        system = (
            f"You are an expert {lang_hint} developer. "
            "Generate complete, production-quality file content. "
            "Output ONLY the raw file content — no explanation, no markdown fences, "
            "no commentary before or after."
        )
        prompt = (
            f"Write the complete content for this file.\n\n"
            f"File path : {path}\n"
            f"Language  : {lang_hint}\n"
            f"Overall task: {task}\n\n"
            f"File requirements:\n{description}\n\n"
            f"Recent agent context (for consistency):\n{context_summary}\n\n"
            f"Output ONLY the file content. Do not wrap in markdown fences."
        )
        content = self.call_ollama_heavy(prompt, system, timeout=300)
        return self._strip_code_fences(content)

    def _generate_patch_replacement(
        self,
        path: str,
        search: str,
        description: str,
        task: str,
        context_summary: str,
    ) -> str:
        """Call the heavy (30b) model to write a patch replacement string."""
        ext = os.path.splitext(path)[1].lower()
        lang_hint = {
            ".py": "Python", ".js": "JavaScript", ".html": "HTML",
            ".css": "CSS", ".sh": "Bash",
        }.get(ext, "code")

        print(f"\n⚙️  [{self.model}] Writing patch → {path}")

        system = (
            f"You are an expert {lang_hint} developer. "
            "Generate a replacement for an existing code snippet. "
            "Output ONLY the replacement text — no explanation, no markdown fences."
        )
        prompt = (
            f"Generate the replacement for this snippet in {path}.\n\n"
            f"Original text to replace:\n{search}\n\n"
            f"What the replacement should do:\n{description}\n\n"
            f"Overall task: {task}\n"
            f"Recent context:\n{context_summary}\n\n"
            f"Output ONLY the replacement text."
        )
        replacement = self.call_ollama_heavy(prompt, system, timeout=180)
        return self._strip_code_fences(replacement)

    def _generate_patch_search_and_replace(
        self,
        path: str,
        description: str,
        task: str,
        context_summary: str,
    ) -> tuple:
        """Heavy model reads the actual file and returns (search, replace).
        Used when the fast model doesn't know the exact search string.
        Returns (None, None) on failure.
        """
        try:
            with open(path, "r") as f:
                file_content = f.read()
        except Exception as e:
            return None, None

        ext = os.path.splitext(path)[1].lower()
        lang_hint = {
            ".py": "Python", ".js": "JavaScript", ".html": "HTML",
            ".css": "CSS", ".sh": "Bash",
        }.get(ext, "code")

        print(f"\n⚙️  [{self.model}] Locating + writing patch → {path}")

        system = (
            f"You are an expert {lang_hint} developer. "
            "You will identify the exact text to replace in a file and write the replacement. "
            "Respond with ONLY a JSON object with two keys: \"search\" and \"replace\". "
            "No explanation, no markdown fences, no extra text."
        )
        prompt = (
            f"File: {path}\n\n"
            f"Current file content:\n{file_content}\n\n"
            f"What to change:\n{description}\n\n"
            f"Overall task: {task}\n"
            f"Recent context:\n{context_summary}\n\n"
            "Return a JSON object: {{\"search\": \"<exact text to find>\", \"replace\": \"<replacement text>\"}}\n"
            "The search string MUST match the file content exactly (same whitespace/indentation)."
        )
        raw = self.call_ollama_heavy(prompt, system, timeout=240)
        raw = self._strip_code_fences(raw).strip()
        try:
            result = json.loads(raw)
            return result.get("search"), result.get("replace")
        except Exception:
            return None, None

    # ================================================================ ReAct ==

    def _diagnose_error(self, tool: str, args: Dict, result) -> str:
        """Pattern-match common errors and return an actionable diagnosis string."""
        error_str = ""
        if hasattr(result, "error") and result.error:
            error_str += result.error
        if hasattr(result, "output") and result.output:
            error_str += "\n" + result.output

        # ModuleNotFoundError — wrong or missing pip package
        m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_str)
        if m:
            mod = m.group(1)
            # Map commonly misnamed packages to the correct install name
            _fixes = {
                "jose":      "python-jose[cryptography]",
                "jwt":       "PyJWT",
                "cv2":       "opencv-python",
                "sklearn":   "scikit-learn",
                "bs4":       "beautifulsoup4",
                "dotenv":    "python-dotenv",
                "PIL":       "Pillow",
                "yaml":      "PyYAML",
                "attr":      "attrs",
                "magic":     "python-magic",
            }
            correct = _fixes.get(mod, mod.replace("_", "-"))
            return (
                f"Module '{mod}' is missing from the CURRENT virtual environment. "
                f"Install it with the venv pip: <venv>/bin/python3 -m pip install {correct}\n"
                f"Do NOT use the system pip — always use the venv's python3 -m pip."
            )

        # SyntaxError inside site-packages → Python 2 / wrong package
        if "SyntaxError" in error_str and "site-packages" in error_str:
            pkg_match = re.search(r"site-packages[/\\]([^/\\]+?)(?:\.py|[/\\])", error_str)
            pkg = pkg_match.group(1) if pkg_match else "the package"
            return (
                f"SyntaxError inside a third-party package ('{pkg}'). "
                f"This means the installed package is a Python 2 era version or the WRONG package name.\n"
                f"Fix: pip uninstall {pkg} -y  →  then install the correct Python 3 package.\n"
                f"Example: 'jose' is Python 2; the correct package is 'python-jose[cryptography]'."
            )

        # SyntaxError in own code
        if "SyntaxError" in error_str:
            line_match = re.search(r"line (\d+)", error_str)
            file_match = re.search(r'File "([^"]+)"', error_str)
            line_num = line_match.group(1) if line_match else "?"
            fname = file_match.group(1) if file_match else "the file"
            return (
                f"Syntax error in {fname} at line {line_num}.\n"
                f"Most common cause: markdown fences (```python / ```) were written into the file.\n"
                f"Fix: read_file → find the fences → patch_file to remove them. "
                f"Then verify: python3 -c \"import ast; ast.parse(open('{fname}').read()); print('ok')\""
            )

        # Permission denied
        if "Permission denied" in error_str or "PermissionError" in error_str:
            return (
                "Permission denied. Options:\n"
                "• Add 'sudo' before the command\n"
                "• Check ownership: ls -la <path>\n"
                "• Write to a path you own (e.g. /home/Grindlewalt/...)"
            )

        # Connection refused / service not running
        if "Connection refused" in error_str or "could not connect to server" in error_str.lower():
            return (
                "Connection refused — the target service is not running.\n"
                "Start it first (e.g. 'sudo systemctl start postgresql') then retry.\n"
                "Check status with: systemctl status <service>"
            )

        # Already exists
        if re.search(r"already exists", error_str, re.IGNORECASE) and tool == "execute_command":
            return "Object already exists — skip creation or use IF NOT EXISTS / CREATE OR REPLACE."

        # Command not found (binary missing — NOT "file argument doesn't exist")
        # Only trigger when the binary itself is missing, not when ls/mv/etc report
        # that their *argument* path doesn't exist.
        if re.search(r"command not found", error_str):
            cmd = (args.get("command", "") or "").split()[0]
            return (
                f"Binary '{cmd}' not found. Either install the package that provides it "
                f"or check the full path. Verify with: which {cmd} || pacman -Fy {cmd}"
            )

        # Timeout (server probably started fine but blocked the shell)
        if re.search(r"[Tt]imeout", error_str):
            return (
                "Command timed out — it is likely a blocking server process.\n"
                "Use background execution instead:\n"
                "  nohup <cmd> > /tmp/<name>.log 2>&1 &\n"
                "Then confirm it started: sleep 2 && ps aux | grep <process> | grep -v grep"
            )

        # Form data / multipart missing
        if "python-multipart" in error_str:
            return "FastAPI needs python-multipart for form endpoints. Install: pip install python-multipart"

        return (
            "Examine STDOUT and STDERR above carefully to determine the root cause.\n"
            "Do NOT repeat the same command. Fix the underlying problem first."
        )

    def run_react(
        self,
        instruction: str,
        confirm_cb=None,
        max_iterations: Optional[int] = None,
        incoming_handoff: Optional[Dict] = None,
        tool_whitelist: Optional[set] = None,
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Core ReAct loop: Reason → Act → Observe × N."""
        if confirm_cb is None:
            confirm_cb = self._ai_confirm

        # Apply minion tool whitelist if provided
        if tool_whitelist is not None:
            self.tool_registry.allowed_tools = tool_whitelist
        else:
            self.tool_registry.allowed_tools = None

        max_iter = max_iterations if max_iterations is not None else self.max_react_iterations
        # Cap threshold so large --budget values don't prevent early finish
        _finish_threshold = max(20, min(max_iter // 4, 100))

        # System survey + runbook
        survey = self.memory.get_system_survey()
        keyword = " ".join(instruction.split()[:3]).lower()
        runbook_content = self.memory.load_runbook(keyword)
        runbook_text = runbook_content[:2000] if runbook_content else "No runbook found."

        # Build tool list — only show whitelisted tools to minions so the model
        # cannot plan around blocked tools.
        if tool_whitelist:
            available_tools_text = "\n".join(
                TOOL_SCHEMAS[t] for t in ("execute_command", "create_file", "patch_file",
                                           "web_search", "read_file", "memory_lookup",
                                           "finish", "manage_server")
                if t in tool_whitelist and t in TOOL_SCHEMAS
            )
        else:
            available_tools_text = _ALL_TOOLS_TEXT

        if system_prompt_override:
            system_prompt = system_prompt_override
        else:
            system_prompt = REACT_SYSTEM_PROMPT_TEMPLATE.format(
                os_info=self.os_info,
                max_iterations=max_iter,
                available_tools=available_tools_text,
            )

        home_dir = survey.get("home_dir", os.path.expanduser("~"))
        username = survey.get("whoami", "unknown")

        os_summary = (
            f"User: {username}  |  Home: {home_dir}  |  Shell: {survey.get('shell', 'N/A')}\n"
            f"OS: {survey.get('uname', 'N/A')}\n"
            f"RAM: {survey.get('memory', 'N/A')}\n"
            f"CPU: {survey.get('cpu_info', 'N/A')[:100]}\n"
            f"Disk: {survey.get('disk', 'N/A')}\n"
            f"Open ports: {survey.get('open_ports', 'N/A')[:300]}\n"
            f"Languages: {survey.get('lang_versions', 'N/A')[:200]}\n"
            f"Active services: {survey.get('active_services', 'N/A')[:300]}\n"
            f"Failed units: {survey.get('failed_units', 'none') or 'none'}\n"
            f"Firewall: {survey.get('firewall', 'N/A')[:300]}\n"
            f"Key installed packages: {survey.get('installed_packages', 'N/A')[:400]}"
        )

        # ---- pre-planning phase ----
        # Skip for whitelisted minions: they have a focused work_order already and
        # pre-planning always generates execute_command steps that are blocked.
        if tool_whitelist:
            react_plan = "(minion mode — no pre-planning)"
            print(f"\n🤖 Minion mode: tools={sorted(tool_whitelist)}\n")
        else:
            print("\n📋 Planning task...")
            plan_prompt = (
                f"Task: {instruction}\n\n"
                f"System: {self.os_info}, user={username}, home={home_dir}\n"
                f"Project directory: {home_dir}/  (use this exact path — never guess)\n"
                f"Already installed: {survey.get('installed_packages', 'unknown')[:300]}\n\n"
                f"Create a concise numbered checklist (max 12 steps) of exactly what needs to "
                f"be done. MANDATORY RULES:\n"
                f"- Step 0: ONE command that batch-checks ALL missing prerequisites at once\n"
                f"- Step 1: ONE command that installs ALL missing packages at once\n"
                f"- Combine related setup into single steps with &&: mkdir + write config + restart = 1 step\n"
                f"- Never add a step just to 'verify' something — verification is implicit after success\n"
                f"- Use exact file paths rooted at {home_dir}/ for all project files\n"
                f"- If the task has 3+ major components, order them logically (backend before frontend, DB before app)\n"
                f"- Flag tasks needing >60 iterations with: '⚠️ LARGE TASK'\n"
                f"- Pick EXACTLY ONE library per concern (one web framework, one ORM, one auth lib).\n"
                f"  Only include the libraries the task explicitly requires — no extras.\n"
                f"- ONLY include steps for services the task explicitly mentions. Do NOT add steps\n"
                f"  to check/restart nginx, postgresql, or other services unless the task targets them.\n"
                f"- NEVER clone from GitHub. All files must be created fresh with create_file.\n"
                f"- Use ABSOLUTE paths everywhere. `cd /dir && cmd` is OK within one step.\n"
                f"  But `.venv/bin/python3` is WRONG because cd does not persist between steps.\n"
                f"  Write `/home/{username}/project/.venv/bin/python3` — the full absolute path.\n"
                f"- Order: install → write all code → read files to confirm → start server → test endpoint\n"
                f"Return ONLY the numbered list, no other text."
            )
            react_plan = self._call_model_oneshot(
                self.fast_model, plan_prompt,
                "Return only a numbered checklist. No prose.",
                timeout=30,
            )
            if react_plan:
                print(f"\n{react_plan}\n")
            else:
                react_plan = "(planning unavailable)"

            # Write plan as a persistent checklist the agent can tick off
            try:
                os.makedirs(os.path.dirname(self.CHECKLIST_PATH), exist_ok=True)
                items = [f"- [ ] {l.strip()}" for l in react_plan.splitlines() if l.strip()]
                with open(self.CHECKLIST_PATH, "w") as f:
                    f.write("# Task Checklist\n\n" + "\n".join(items) + "\n")
            except Exception:
                pass

        initial_message = (
            f"TASK: {instruction}\n\n"
            f"SYSTEM CONTEXT:\n{os_summary}\n\n"
            f"⚠️  PATH RULE (CRITICAL): Use ONLY paths from SYSTEM CONTEXT above.\n"
            f"    Your home is {home_dir}. ALL project files go under {home_dir}/\n"
            f"    NEVER guess or invent paths. If a web search or example suggests a path\n"
            f"    that differs from SYSTEM CONTEXT, ignore it and use the real path.\n\n"
            f"YOUR PLAN (follow this checklist in order — do not skip steps or add unplanned steps):\n{react_plan}\n\n"
            f"RUNBOOK: {runbook_text}\n\n"
            f"Produce your first thought and tool call as JSON."
        )

        # Prepend structured handoff from previous sub-task if provided
        if incoming_handoff:
            facts = incoming_handoff.get("facts", {})
            source_instruction = incoming_handoff.get("source_instruction", "")
            prev_success = incoming_handoff.get("success", False)
            partial = incoming_handoff.get("partial_completion")
            status_str = "SUCCEEDED" if prev_success else "PARTIALLY COMPLETED"

            context_lines = [
                "CONTEXT FROM PREVIOUS TASK:",
                f"'{source_instruction}' [{status_str}]",
                "",
                "Facts discovered:",
                json.dumps(facts, indent=2),
            ]
            if partial:
                accomplished = partial.get("accomplished", [])
                if accomplished:
                    context_lines.append(f"\nWhat was done: {'; '.join(accomplished[:5])}")
                context_lines.append(f"What still remains: {partial.get('what_remains', '')}")

            context_block = "\n".join(context_lines)
            initial_message = context_block + "\n\n" + initial_message

        # Inject previous session progress notes if they match this task
        try:
            if os.path.exists(self.PROGRESS_PATH):
                with open(self.PROGRESS_PATH) as f:
                    prev_progress = f.read().strip()
                if prev_progress:
                    # Check task fingerprint: first line is <!-- task: ... -->
                    saved_fp = ""
                    for line in prev_progress.splitlines()[:2]:
                        if line.startswith("<!-- task:"):
                            saved_fp = line[len("<!-- task:"):].rstrip(" -->").strip()
                            break
                    current_fp = instruction[:120].replace("\n", " ")
                    if saved_fp and saved_fp != current_fp:
                        print(f"\n⏭️  Skipping stale progress.md (different task)")
                    else:
                        initial_message = (
                            "PREVIOUS SESSION NOTES (from last run — use this to pick up where you left off):\n"
                            f"{prev_progress}\n\n"
                            "--- END OF PREVIOUS NOTES ---\n\n"
                        ) + initial_message
                        print(f"\n📖 Loaded previous session notes from {self.PROGRESS_PATH}")
        except Exception:
            pass

        react_history: List[Dict] = [{"role": "user", "content": initial_message}]
        self.react_trace = []
        finish_summary = ""
        final_confidence = 50
        # Tracks consecutive failures per file path to break permission-denied loops
        _path_fail_counts: Dict[str, int] = {}
        # How many times we've challenged a premature finish (challenge fires once only)
        _premature_finish_challenges: int = 0
        # Consecutive JSON parse failures — triggers heavy model rescue at 5
        _consecutive_json_failures: int = 0
        # Forward progress tracking
        _last_progress_iteration: int = 0      # last iter with new file or unique cmd success
        _seen_commands: set = set()            # unique successful commands this phase
        _progress_warn_at: int = 20
        _progress_kill_at: int = 35
        # Inject a task-progress reminder every N iterations
        # Use max_iter//8 so a 100-iteration run gets a checkpoint every ~12 steps,
        # catching plan drift early rather than at 25% (too late).
        _checkpoint_interval: int = max(5, max_iter // 8)

        for iteration in range(1, max_iter + 1):
            print(f"\n{'=' * 70}")
            print(f"🔄 ReAct Iteration {iteration}/{max_iter}")

            # Trim history to keep context window manageable.
            # 12 messages × ~2k chars each ≈ 6k tokens — well inside 16k.
            # Entries being rotated out have their thought/confidence stripped
            # to compress them further before discarding.
            if len(react_history) > 10:
                keep_tail = react_history[-9:]
                react_history = react_history[:1] + keep_tail

            # Periodic task-progress reminder (every _checkpoint_interval iterations)
            if iteration > 1 and iteration % _checkpoint_interval == 0:
                self._write_progress_md(instruction, iteration, max_iter)
                remaining = max_iter - iteration
                react_history.append({"role": "user", "content": (
                    f"[PROGRESS CHECK — iteration {iteration}/{max_iter}, "
                    f"{remaining} iterations remaining]\n\n"
                    f"ORIGINAL TASK: {instruction[:600]}\n\n"
                    f"YOUR ORIGINAL PLAN:\n{react_plan}\n\n"
                    f"Look at the plan above. Which steps are DONE and which are STILL PENDING? "
                    f"If you have drifted from the plan (e.g. checking unrelated services, "
                    f"debugging the wrong thing), stop and return to the next pending plan step. "
                    f"Do not call finish until every requirement in the task is implemented and tested.\n\n"
                    f"CHECKLIST (mark steps done with patch_file: `- [ ]` → `- [x]`):\n"
                    f"{open(self.CHECKLIST_PATH).read() if os.path.exists(self.CHECKLIST_PATH) else '(unavailable)'}\n\n"
                    f"Produce your next thought and tool call as JSON."
                )})

            # Call LLM
            raw = self.call_ollama_react(react_history, system_prompt)

            if not raw:
                react_history.append({
                    "role": "user",
                    "content": (
                        "ERROR: Empty LLM response. "
                        "Produce valid JSON with thought, confidence, tool, and args."
                    ),
                })
                continue

            react_history.append({"role": "assistant", "content": raw})

            # Parse JSON from response
            parsed = self.extract_json(raw)

            if not parsed or not isinstance(parsed, dict):
                _consecutive_json_failures += 1
                print(f"⚠️  Failed to parse JSON ({_consecutive_json_failures} consecutive)")
                if _consecutive_json_failures >= 5:
                    print(f"🆘 JSON cascade — calling heavy model to rescue...")
                    ctx = self._build_context_summary(react_history[-8:])
                    rescue_raw = self._call_model_oneshot(
                        self.model,
                        f"Context:\n{ctx}\n\nBased on the above, produce the single best next tool call.",
                        'Return ONLY a JSON object: {"thought":"...","confidence":90,"tool":"...","args":{}}',
                        timeout=120,
                    )
                    if rescue_raw:
                        rescue_parsed = self.extract_json(rescue_raw)
                        if rescue_parsed and isinstance(rescue_parsed, dict):
                            print(f"  ✅ Heavy model rescue succeeded")
                            react_history.append({"role": "assistant", "content": rescue_raw})
                            parsed = rescue_parsed
                            _consecutive_json_failures = 0
                            # Purge corrupted history baggage — keep initial context + rescue response only
                            if len(react_history) > 4:
                                react_history = react_history[:1] + react_history[-2:]
                                print(f"  🧹 History pruned to {len(react_history)} messages after rescue")
                            # Fall through to normal tool dispatch below
                        else:
                            react_history.append({"role": "user", "content": "ERROR: Heavy model rescue also failed. Produce a valid JSON tool call."})
                            continue
                    else:
                        react_history.append({"role": "user", "content": "ERROR: Rescue call returned nothing. Produce a valid JSON tool call."})
                        continue
                else:
                    react_history.append({
                        "role": "user",
                        "content": (
                            'ERROR: Your response was not valid JSON. '
                            'Respond with ONLY a JSON object: {"thought":"...","confidence":90,"tool":"...","args":{}}'
                        ),
                    })
                    continue

            thought = parsed.get("thought", "")
            confidence = int(parsed.get("confidence", 50))
            tool = parsed.get("tool", "")
            args = parsed.get("args", {})

            print(f"💭 Thought: {thought[:120]}")
            print(f"🎯 Tool: {tool}  |  Confidence: {confidence}%")

            if not tool:
                react_history.append({
                    "role": "user",
                    "content": "ERROR: Missing 'tool' field. You MUST specify a tool name.",
                })
                continue

            # ---- heavy-model delegation for file content / patch replace ----
            # The fast model is expected to leave content/replace empty and
            # supply a "description" field instead.  We call the 30b model
            # here to generate the actual code before dispatching the tool.
            if tool == "create_file" and not args.get("content"):
                description = args.get("description", "Write appropriate content")
                ctx = self._build_context_summary(react_history[-6:])
                generated = self._generate_file_content(
                    args.get("path", ""), description, instruction, ctx
                )
                if generated:
                    args["content"] = generated
                    print(f"  📄 Generated {len(generated):,} chars")
                else:
                    react_history.append({
                        "role": "user",
                        "content": "ERROR: Code generation model returned empty content. Try a simpler description or break the file into smaller pieces.",
                    })
                    continue

            elif tool == "patch_file" and not args.get("search") and args.get("description"):
                # Fast model left search blank — heavy model reads the file and
                # figures out both the exact search string and the replacement.
                description = args.get("description", "")
                ctx = self._build_context_summary(react_history[-6:])
                search, replace = self._generate_patch_search_and_replace(
                    args.get("path", ""), description, instruction, ctx
                )
                if search and replace is not None:
                    args["search"] = search
                    args["replace"] = replace
                    print(f"  ✏️  Heavy model located search ({len(search):,} chars) + replacement ({len(replace):,} chars)")
                else:
                    react_history.append({
                        "role": "user",
                        "content": "ERROR: Heavy model could not locate the section to patch. Try read_file first to confirm the exact content.",
                    })
                    continue

            elif tool == "patch_file" and not args.get("replace") and args.get("description"):
                description = args.get("description", "")
                ctx = self._build_context_summary(react_history[-6:])
                generated = self._generate_patch_replacement(
                    args.get("path", ""), args.get("search", ""),
                    description, instruction, ctx
                )
                if generated:
                    args["replace"] = generated
                    print(f"  ✏️  Generated {len(generated):,} chars of replacement")
                else:
                    react_history.append({
                        "role": "user",
                        "content": "ERROR: Code generation model returned empty replacement. Try again with a clearer description.",
                    })
                    continue

            # Dispatch
            result = self.tool_registry.dispatch(tool, args, confidence, confirm_cb)

            # ---- trigger_rewrite intercept ----
            # When patch_file fails with "search not found" twice for the same file,
            # ToolRegistry returns trigger_rewrite=True.  We auto-rewrite here instead
            # of burning more iterations on hopeless patch retries.
            if tool == "patch_file" and result.metadata.get("trigger_rewrite"):
                rw_path = result.metadata["path"]
                description = args.get("description", "Rewrite the file incorporating all required changes")
                ctx = self._build_context_summary(react_history[-6:])
                new_content = self._generate_file_content(rw_path, description, instruction, ctx)
                if new_content:
                    try:
                        with open(os.path.expanduser(rw_path), "w") as _rw_f:
                            _rw_f.write(new_content)
                        observation = (
                            f"OBSERVATION [iteration {iteration}]:\n"
                            f"⚡ Auto-rewrote {rw_path} ({len(new_content):,} chars) due to "
                            f"repeated patch failures. File has been fully replaced.\n\n"
                            f"Produce your next thought and tool call as JSON."
                        )
                        print(f"  ⚡ Auto-rewrote {rw_path} ({len(new_content):,} chars)")
                    except Exception as _rw_err:
                        observation = (
                            f"OBSERVATION [iteration {iteration}]:\n"
                            f"⚡ Auto-rewrite triggered for {rw_path} but write failed: {_rw_err}.\n\n"
                            f"Produce your next thought and tool call as JSON."
                        )
                else:
                    observation = (
                        f"OBSERVATION [iteration {iteration}]:\n"
                        f"⚡ Auto-rewrite triggered for {rw_path} but heavy model returned empty content. "
                        f"Try using create_file with explicit content instead.\n\n"
                        f"Produce your next thought and tool call as JSON."
                    )
                react_history.append({"role": "user", "content": observation})
                continue

            # Record in trace
            self.react_trace.append({
                "iteration": iteration,
                "thought": thought,
                "confidence": confidence,
                "tool": tool,
                "args": args,
                "result": result,
            })
            if _debug_logger:
                _debug_logger.react_iter(
                    job_id=self.current_job_id or "",
                    iteration=iteration,
                    max_iter=max_iter,
                    thought=thought,
                    tool=tool,
                    args=args,
                    result=result,
                    confidence=confidence,
                )

            # Track forward progress
            if result.success:
                if tool in ("create_file", "patch_file"):
                    _last_progress_iteration = iteration
                elif tool == "execute_command":
                    cmd_key = args.get("command", "").strip()
                    if cmd_key not in _seen_commands:
                        _seen_commands.add(cmd_key)
                        _last_progress_iteration = iteration

            # Pin architectural reads so they survive history trimming
            _ARCH_PATTERNS = ("ARCH.md", "schema.sql", "models.py", "schema.py",
                               "config.py", "settings.py", "database.py")
            if result.success and tool == "read_file":
                read_path = args.get("path", "")
                if any(pat in read_path for pat in _ARCH_PATTERNS):
                    pin_content = (
                        f"[PINNED ARCHITECTURAL REFERENCE — {read_path}]\n"
                        f"{result.output[:3000]}"
                    )
                    # Replace existing pin for this path, or append new one
                    self.pinned_messages = [
                        m for m in self.pinned_messages
                        if read_path not in m.get("content", "")
                    ]
                    self.pinned_messages.append({
                        "role": "user",
                        "content": pin_content,
                    })

            if tool == "finish":
                # Challenge premature success: fire once if < 50% of budget used
                if (
                    args.get("success")
                    and _premature_finish_challenges < 1
                    and iteration < _finish_threshold   # was: int(max_iter * 0.5)
                ):
                    _premature_finish_challenges += 1
                    pct = iteration * 100 // max_iter
                    print(f"⚠️  Pre-finish challenge: iteration {iteration}/{max_iter} ({pct}% budget used)")
                    react_history.append({"role": "user", "content": (
                        f"⚠️  HOLD — You are calling finish(success=true) at iteration "
                        f"{iteration}/{max_iter} ({pct}% of budget used).\n\n"
                        f"ORIGINAL TASK REQUIREMENTS:\n{instruction}\n\n"
                        f"Before finishing, confirm EVERY requirement above is fully implemented:\n"
                        f"• List each requirement and whether it is DONE or NOT YET DONE\n"
                        f"• If ALL are done → call finish(success=true) again to proceed\n"
                        f"• If ANYTHING is missing → continue implementing it\n\n"
                        f"This challenge fires only once. Your next finish call will proceed.\n\n"
                        f"Produce your next thought and tool call as JSON."
                    )})
                    continue  # Do not break — agent must reconsider

                finish_summary = args.get("summary", result.output)
                final_confidence = confidence
                break

            if result.metadata.get("stuck") == "warn":
                # First stuck loop — agent gets a redirect injection and continues
                print("⚠️  Stuck loop detected — injecting redirect and continuing")
                react_history.append({"role": "user", "content": result.error})
                continue

            if result.metadata.get("stuck"):
                print("⚠️  Stuck loop detected (second time) — breaking")
                finish_summary = "Terminated: stuck in identical-call loop"
                break

            # ---- path-failure circuit breaker ----
            # If create_file / patch_file fails on the same path 3+ times in a row,
            # force the agent off that path entirely.
            if tool in ("create_file", "patch_file") and not result.success:
                fail_path = args.get("path", "")
                _path_fail_counts[fail_path] = _path_fail_counts.get(fail_path, 0) + 1
                if _path_fail_counts[fail_path] >= 3:
                    print(f"🚫 Circuit breaker: {fail_path} failed {_path_fail_counts[fail_path]} times")
                    home_dir_fb = survey.get("home_dir", os.path.expanduser("~"))
                    react_history.append({"role": "user", "content": (
                        f"CIRCUIT BREAKER: You have failed to write '{fail_path}' "
                        f"{_path_fail_counts[fail_path]} times. STOP trying this path. \n"
                        f"The correct home directory is {home_dir_fb}. "
                        f"Use a different path under {home_dir_fb}/. "
                        f"If you need to write to a system path, use execute_command with sudo tee."
                    )})
                    _path_fail_counts[fail_path] = 0  # reset after injecting
                    continue
            elif result.success and tool in ("create_file", "patch_file"):
                # Reset on success
                _path_fail_counts.pop(args.get("path", ""), None)

            # Build observation for next iteration
            if result.error == "Cancelled by user":
                obs = (
                    f"OBSERVATION [iteration {iteration}]:\n"
                    f"⛔ USER REJECTED the action: {tool} with args {json.dumps(args)}\n"
                    f"You MUST NOT attempt this action again. Choose a completely different approach.\n\n"
                    f"Produce your next thought and tool call as JSON."
                )
            elif not result.success:
                # Rich failure observation with full context and diagnosis
                diagnosis = self._diagnose_error(tool, args, result)
                exit_code = (result.metadata or {}).get("exit_code", "N/A")

                if tool == "execute_command":
                    what_ran = f"Command : {args.get('command', '(none)')}"
                elif tool in ("create_file", "patch_file"):
                    what_ran = f"File    : {args.get('path', '(none)')}"
                else:
                    what_ran = f"Args    : {json.dumps(args)}"

                stdout_text = (result.output or "").strip() or "(empty)"
                stderr_text = (result.error or "").strip() or "(empty)"

                obs = (
                    f"OBSERVATION [iteration {iteration}] — ❌ FAILURE\n"
                    f"{'=' * 64}\n"
                    f"Tool      : {tool}\n"
                    f"{what_ran}\n"
                    f"Exit code : {exit_code}\n"
                    f"\n── STDOUT ───────────────────────────────────────────────────\n"
                    f"{stdout_text[:1000]}\n"
                    f"\n── STDERR / ERROR ───────────────────────────────────────────\n"
                    f"{stderr_text[:1000]}\n"
                    f"\n── DIAGNOSIS ────────────────────────────────────────────────\n"
                    f"{diagnosis}\n"
                    f"\n── REQUIRED ACTION ──────────────────────────────────────────\n"
                    f"• Read the full STDERR and DIAGNOSIS above before acting\n"
                    f"• Do NOT repeat the exact same command/action\n"
                    f"• Fix the ROOT CAUSE, not the symptom\n"
                    f"{'=' * 64}\n\n"
                    f"Produce your next thought and tool call as JSON."
                )
            else:
                # Success observation
                extra_hint = ""
                if tool == "create_file":
                    created_path = args.get("path", "")
                    extra_hint = (
                        f"\n⚠️  REQUIRED NEXT STEP: Call read_file on '{created_path}' "
                        f"to see exactly what was generated (routes, functions, imports) "
                        f"before running or testing it."
                    )
                obs = (
                    f"OBSERVATION [iteration {iteration}] — ✅ SUCCESS\n"
                    f"Tool: {tool}\n"
                    f"Args: {json.dumps(args)}\n"
                    f"Output:\n{result.output[:800]}\n"
                    f"Metadata: {json.dumps(result.metadata)}{extra_hint}\n\n"
                    f"Produce your next thought and tool call as JSON."
                )
            react_history.append({"role": "user", "content": obs})

            # Stagnation / forward progress enforcement
            _idle = iteration - _last_progress_iteration
            if _idle == _progress_warn_at:
                print(f"⚠️  No forward progress for {_idle} iterations — injecting warning")
                react_history.append({"role": "user", "content": (
                    f"[STAGNATION WARNING — {_idle} iterations without meaningful progress]\n\n"
                    f"You have not created/modified a file or run a new successful command in {_idle} iterations.\n\n"
                    f"REQUIRED: Diagnose the blocker:\n"
                    f"• What exactly is blocking you?\n"
                    f"• Try a fundamentally different approach\n"
                    f"• If the task is unresolvable, call finish(success=false) now\n\n"
                    f"You have {_progress_kill_at - _idle} iterations before this phase is force-aborted.\n\n"
                    f"Produce your next thought and tool call as JSON."
                )})
            elif _idle >= _progress_kill_at:
                finish_summary = (
                    f"Phase aborted: {_idle} consecutive iterations with no forward progress. "
                    f"Last meaningful action at iteration {_last_progress_iteration}."
                )
                print(f"💀 {finish_summary}")
                break

        else:
            finish_summary = f"Max iterations ({max_iter}) reached without finishing"
            print(f"⚠️  {finish_summary}")

        # Verify completion via a separate LLM call
        print("\n🔬 Verifying completion...")
        verification = self._verify_react_completion(
            instruction, finish_summary, self.react_trace
        )

        # Persist successful commands to memory
        if verification.get("verified") and verification.get("confidence", 0) >= 80:
            for entry in self.react_trace:
                if entry["tool"] == "execute_command" and entry["result"].success:
                    self.memory.record_success(
                        command=entry["args"].get("command", ""),
                        context=entry["thought"][:200],
                        task=instruction[:200],
                        exit_code=entry["result"].metadata.get("exit_code", 0),
                        duration_ms=entry["result"].metadata.get("duration_ms", 0),
                    )

        iterations_used = sum(1 for e in self.react_trace if e["tool"] != "finish")

        result_dict = {
            "success": verification.get("verified", False),
            "confidence": verification.get("confidence", final_confidence),
            "verification_notes": verification.get("notes", ""),
            "finish_summary": finish_summary,
            "iterations_used": iterations_used,
            "trace": self.react_trace,
        }

        if result_dict["success"]:
            # Clear progress notes — task is done
            try:
                if os.path.exists(self.PROGRESS_PATH):
                    os.remove(self.PROGRESS_PATH)
            except Exception:
                pass
        else:
            # Write a final progress snapshot so the next run can pick up here
            self._write_progress_md(instruction, iterations_used, max_iter)

        if not result_dict["success"]:
            state_path = os.path.expanduser("~/.agent_bin/last_incomplete_task.json")
            try:
                with open(state_path, "w") as f:
                    json.dump({
                        "instruction": instruction,
                        "summary": finish_summary,
                        "iterations_used": iterations_used,
                        "accomplished": [
                            f"[{e['tool']}] {e['thought'][:100]}"
                            for e in self.react_trace
                            if e.get("result") and e["result"].success
                        ],
                        "timestamp": datetime.now().isoformat(),
                    }, f, indent=2)
                print(f"\n💾 Incomplete task state saved → {state_path}")
            except Exception:
                pass

        return result_dict

    # ================================================ public entry point ==

    def run(self, instruction: str, incoming_handoff: Optional[Dict] = None):
        """Main entry point — thin wrapper around run_react for server.py compat."""
        print(f"\n{'=' * 70}")
        print(f"🎯 TASK: {instruction}")
        print(f"💻 SYSTEM: {self.os_info}")
        print(f"🔄 MAX ITERATIONS: {self.max_react_iterations}")
        print(f"🧠 MODEL: {self.fast_model}")
        print(f"🛡️  SAFETY: Enabled")
        if incoming_handoff:
            print(f"🔗 CHAIN HANDOFF: from '{incoming_handoff.get('source_instruction', '')[:60]}'")
        print(f"{'=' * 70}")

        result = self.run_react(instruction, incoming_handoff=incoming_handoff)

        # Populate self.execution_log in the format server.py reads
        self.execution_log = [
            {
                "step": i + 1,
                "description": f"[{e['tool']}] {e['thought'][:80]}",
                "success": e["result"].success,
                "type": e["tool"],
                "confidence": e["confidence"],
            }
            for i, e in enumerate(result["trace"])
        ]

        print(f"\n{'=' * 70}")
        print(f"{'✅ COMPLETED' if result['success'] else '⚠️  INCOMPLETE'}")
        print(f"Iterations: {result['iterations_used']}")
        print(f"Summary: {result['finish_summary']}")
        print(f"Verification: {result['verification_notes']}")
        print(f"{'=' * 70}")

        return result


# ---------------------------------------------------------------------------
# PostRunVerifier
# ---------------------------------------------------------------------------

class PostRunVerifier:
    """
    QA agent that runs after the main ReAct agent calls finish.

    Steps:
      1. Fast model selects the 10 most informative trace entries.
      2. Heavy model (qwen3-coder:30b) generates 3-8 verification shell commands.
      3. Commands are executed; pass/fail recorded.
      4. Heavy model writes a PASS/FAIL report with root cause + fix plan.
    """

    MAX_CURATED_ENTRIES = 10

    def __init__(self, agent: "OllamaCommandAgent"):
        self.agent = agent

    def _curate_trace(self, trace: list) -> str:
        """Fast model picks the most informative trace entries; returns full detail text."""
        if not trace:
            return "(no trace entries)"

        summaries = []
        for i, entry in enumerate(trace):
            tool = entry.get("tool", "?")
            args = entry.get("args", {})
            result = entry.get("result")
            if hasattr(result, "success"):
                ok, out = result.success, (result.output or "")[:80]
            elif isinstance(result, dict):
                ok, out = result.get("success", False), str(result.get("output", ""))[:80]
            else:
                ok, out = False, ""
            summaries.append(
                f"[{i}] {tool} ok={ok} args={json.dumps(args)[:120]} out={out!r}"
            )

        prompt = (
            f"Select the {self.MAX_CURATED_ENTRIES} most informative entries from this agent trace.\n"
            "Prefer: file creations, file patches, service starts, major installs, config writes, finish.\n"
            "Skip: trivial reads, failed retries, memory lookups.\n\n"
            "TRACE:\n" + "\n".join(summaries) +
            f"\n\nReturn a JSON array of integer indices (max {self.MAX_CURATED_ENTRIES}). JSON only."
        )
        try:
            raw = self.agent._call_model_oneshot(
                self.agent.fast_model, prompt,
                "Return only a JSON array of integers. No prose.", timeout=30
            )
            indices = self.agent.extract_json(raw)
            if not isinstance(indices, list):
                raise ValueError()
            indices = sorted(set(i for i in indices if isinstance(i, int) and 0 <= i < len(trace)))
        except Exception:
            indices = list(range(min(len(trace), self.MAX_CURATED_ENTRIES)))

        lines = []
        for i in indices:
            entry = trace[i]
            tool = entry.get("tool", "?")
            args = entry.get("args", {})
            thought = entry.get("thought", "")[:200]
            result = entry.get("result")
            if hasattr(result, "success"):
                ok, out, err = result.success, (result.output or "")[:600], (result.error or "")[:300]
            elif isinstance(result, dict):
                ok = result.get("success", False)
                out = str(result.get("output", ""))[:600]
                err = str(result.get("error", ""))[:300]
            else:
                ok, out, err = False, "", ""
            lines.append(
                f"--- [{i}] {tool} (success={ok}) ---\n"
                f"Thought: {thought}\n"
                f"Args: {json.dumps(args, indent=2)[:500]}\n"
                f"Output: {out!r}\n"
                f"Error: {err!r}\n"
            )
        return "\n".join(lines)

    def verify(self, goal: str, react_result: dict, plan_text: str = "") -> dict:
        """
        Full QA pass.  Returns {passed, passed_count, total_count, test_results, report}.
        """
        import subprocess as _sp

        print(f"\n{'='*70}", flush=True)
        print("🔍 POST-RUN VERIFICATION (qwen3-coder:30b)", flush=True)
        print(f"{'='*70}", flush=True)

        trace = react_result.get("trace", [])
        finish_summary = react_result.get("finish_summary", "")

        # 1. Curate trace
        print("  ↳ Selecting key trace entries...", flush=True)
        curated = self._curate_trace(trace)

        context_block = (
            f"ORIGINAL GOAL:\n{goal}\n\n"
            f"PLAN / CHECKLIST:\n{plan_text if plan_text else '(not available)'}\n\n"
            f"AGENT FINISH SUMMARY:\n{finish_summary}\n\n"
            f"KEY TRACE ENTRIES (fast-model selected):\n{curated}"
        )

        # 2. Heavy model writes verification commands
        print("  ↳ Generating verification tests...", flush=True)
        test_prompt = (
            "You are a QA engineer. An autonomous agent just finished a task.\n"
            "Write 3-8 shell commands that verify it actually worked correctly.\n"
            "Each command must exit 0 on success, non-zero on failure.\n"
            "Focus on: files present, services running, endpoints responding, data correct.\n\n"
            + context_block +
            '\n\nReturn ONLY a JSON array:\n'
            '[{"command": "systemctl is-active nginx", "purpose": "nginx is running"}, ...]\n'
            "JSON only, no prose."
        )
        try:
            raw = self.agent._call_model_oneshot(
                self.agent.model, test_prompt,
                "Return only a JSON array with command and purpose fields. No prose.", timeout=120
            )
            test_commands = self.agent.extract_json(raw)
            if not isinstance(test_commands, list):
                raise ValueError()
        except Exception:
            test_commands = []
            print("  ⚠️  Could not generate test commands — skipping to analysis", flush=True)

        # 3. Run the tests
        test_results = []
        for tc in test_commands:
            cmd = (tc.get("command") or "").strip()
            purpose = tc.get("purpose", "")
            if not cmd:
                continue
            print(f"  🧪 {purpose}", flush=True)
            print(f"     $ {cmd}", flush=True)
            try:
                r = _sp.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=30, executable="/bin/bash"
                )
                passed, stdout, stderr, exit_code = (
                    r.returncode == 0, r.stdout[:500], r.stderr[:300], r.returncode
                )
            except Exception as e:
                passed, stdout, stderr, exit_code = False, "", str(e), -1
            print(f"     {'✅ PASS' if passed else '❌ FAIL'}", flush=True)
            test_results.append({
                "command": cmd, "purpose": purpose,
                "passed": passed, "stdout": stdout, "stderr": stderr, "exit_code": exit_code,
            })

        passed_count = sum(1 for r in test_results if r["passed"])
        total_count = len(test_results)
        all_passed = (passed_count == total_count) if total_count > 0 else True

        # 4. Heavy model writes the report
        print("  ↳ Writing analysis report...", flush=True)
        results_text = "\n".join(
            f"[{'PASS' if r['passed'] else 'FAIL'}] {r['purpose']}\n"
            f"  $ {r['command']}\n"
            f"  stdout: {r['stdout']!r}\n"
            f"  stderr: {r['stderr']!r}"
            for r in test_results
        ) if test_results else "(no tests were run)"

        analyze_prompt = (
            "You are a senior engineer reviewing the outcome of an automated task.\n\n"
            + context_block +
            f"\n\nVERIFICATION RESULTS ({passed_count}/{total_count} passed):\n{results_text}\n\n"
            "Write a technical report with these sections:\n"
            "1. VERDICT: PASSED or FAILED\n"
            "2. WHAT WORKS: confirmed working items\n"
            "3. WHAT FAILED: specific failures with root cause (if any)\n"
            "4. FIX PLAN: exact shell commands to fix each failure (if any)\n"
            "5. IMPROVEMENTS: optional suggestions for robustness\n\n"
            "Be specific — use exact paths, service names, and commands from the context above."
        )
        try:
            report = self.agent._call_model_oneshot(
                self.agent.model, analyze_prompt,
                "You are a senior engineer writing a concise technical verification report.", timeout=180
            )
        except Exception as e:
            report = f"(could not generate analysis report: {e})"

        print(f"\n{'='*70}", flush=True)
        verdict = "✅ PASSED" if all_passed else "❌ FAILED"
        print(f"{verdict}  ({passed_count}/{total_count} tests passed)", flush=True)
        print(f"{'='*70}", flush=True)
        print(report, flush=True)

        return {
            "passed": all_passed,
            "passed_count": passed_count,
            "total_count": total_count,
            "test_results": test_results,
            "report": report,
        }


def _read_checklist() -> str:
    """Read the agent's written plan from ~/.agent_bin/checklist.md."""
    path = os.path.expanduser("~/.agent_bin/checklist.md")
    try:
        with open(path) as f:
            return f.read()
    except Exception:
        return ""


def main():
    import sys
    from task_chain import TaskDecomposer, HandoffExtractor, AcceptanceCriteriaRunner

    # Force line-buffered output so live logs work correctly when piped to tee
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Ollama Agent CLI")
    parser.add_argument("--budget", "-b", type=int, default=500,
                        help="Total iteration budget for decomposition (default: 500)")
    parser.add_argument("--task", "-t", type=str, default=None,
                        help="Run a single task non-interactively")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip confirmation prompt for multi-phase plans")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the post-run QA verification pass")
    cli_args = parser.parse_args()

    if cli_args.task:
        instruction = cli_args.task
    else:
        instruction = input("What would you like me to do? ")

    # Decompose the instruction
    print("\n⏳ Decomposing task...", flush=True)
    decomp_agent = OllamaCommandAgent(
        model="qwen3-coder:30b",
        searxng_url="http://10.0.0.58:8080",
    )
    subtasks = TaskDecomposer(decomp_agent).decompose(instruction, total_budget=cli_args.budget)

    # Single subtask → run directly, same behaviour as before
    if len(subtasks) == 1:
        agent = OllamaCommandAgent(
            model="qwen3-coder:30b",
            searxng_url="http://10.0.0.58:8080",
        )
        agent.max_react_iterations = subtasks[0]["max_iterations"]
        result = agent.run(instruction)
        if not cli_args.no_verify:
            PostRunVerifier(agent).verify(instruction, result, plan_text=_read_checklist())
        return

    # Multi-phase: show plan and confirm
    print(f"\n📋 DECOMPOSED PLAN ({len(subtasks)} phases, {cli_args.budget} iteration budget):\n", flush=True)
    for st in subtasks:
        complexity = st.get("estimated_complexity", "medium")
        iters = st["max_iterations"]
        print(f"  Phase {st['index']+1} [{complexity}, {iters} iter]: {st['instruction']}")
        if st.get("acceptance_criteria"):
            print(f"           Check: {st['acceptance_criteria']}")
        print()

    if not cli_args.yes:
        answer = input("Proceed? [Y/n] ").strip().lower()
        if answer and answer not in ("y", "yes"):
            print("Aborted.")
            return

    # Run chain loop
    handoff = None
    last_result = None
    last_agent = None
    for st in subtasks:
        print(f"\n{'='*70}", flush=True)
        print(f"🔗 PHASE {st['index']+1}/{len(subtasks)}: {st['instruction'][:80]}", flush=True)
        iters = st["max_iterations"]
        print(f"   Budget: {iters} iterations", flush=True)

        phase_agent = OllamaCommandAgent(
            model="qwen3-coder:30b",
            searxng_url="http://10.0.0.58:8080",
        )
        phase_agent.max_react_iterations = iters
        result = phase_agent.run(st["instruction"], incoming_handoff=handoff)
        last_result = result
        last_agent = phase_agent

        # Reset per-phase state so it doesn't carry across phases
        phase_agent.tool_registry.reset_phase_state()

        # Extract structured handoff for next phase
        handoff = HandoffExtractor(phase_agent).extract(st["instruction"], result)

        # Run acceptance criteria check if provided
        if st.get("acceptance_criteria"):
            ac_result = AcceptanceCriteriaRunner().run(st["acceptance_criteria"])
            if ac_result["passed"]:
                print(f"   Acceptance check: ✅ PASSED — {st['acceptance_criteria']}", flush=True)
            else:
                print(
                    f"   Acceptance check: ❌ FAILED — {st['acceptance_criteria']}\n"
                    f"   stdout: {ac_result.get('stdout','')[:200]}\n"
                    f"   stderr: {ac_result.get('stderr','')[:200]}",
                    flush=True
                )
                if result.get("success"):
                    # Agent claimed success but acceptance check disagrees — hard abort.
                    # Something is fundamentally wrong with the output; don't build on it.
                    print(f"❌ Phase {st['index']+1} claimed success but failed acceptance — aborting chain", flush=True)
                    break
                else:
                    # Agent ran out of budget (success=False). Acceptance failure is a
                    # budget problem, not a broken artifact — warn and continue.
                    print(
                        f"⚠️  Phase {st['index']+1} hit budget limit; acceptance not met — "
                        f"continuing with partial handoff (re-run with more --budget)",
                        flush=True
                    )

        elif not result.get("success"):
            print(f"⚠️  Phase {st['index']+1} did not fully complete — continuing with partial handoff", flush=True)

    # Post-run verification after all phases complete
    if not cli_args.no_verify and last_result is not None and last_agent is not None:
        PostRunVerifier(last_agent).verify(instruction, last_result, plan_text=_read_checklist())


if __name__ == "__main__":
    main()
