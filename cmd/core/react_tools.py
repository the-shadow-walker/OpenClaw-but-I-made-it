#!/usr/bin/env python3
"""
ToolRegistry: dispatches the 7 ReAct tools with safety gating.
"""

import subprocess
import os
import re
import time
import json
from collections import deque
from typing import Optional, Callable, Any, Dict, NamedTuple


class ToolResult(NamedTuple):
    success: bool
    output: str
    error: str
    metadata: dict


class ToolRegistry:
    TOOL_NAMES = {
        "execute_command",
        "create_file",
        "patch_file",
        "web_search",
        "read_file",
        "memory_lookup",
        "finish",
        "manage_server",
        "validate_arch",   # PR0: ARCH.json validator (pure, no LLM)
        "get_deps",        # PR2: import dependency graph (pure, no LLM)
        "write_plan",      # Hardening: persists planner/builder task plan markdown
    }

    def __init__(self, safety_validator, search_agent, memory, explain_cb=None,
                 allowed_tools=None):
        self.safety_validator = safety_validator
        self.search_agent = search_agent
        self.memory = memory
        # Optional callable(command: str) -> str that returns a human-readable
        # breakdown of the command segments, shown before execution / confirmation.
        self.explain_cb = explain_cb
        # Optional set of tool names that are permitted (None = all tools allowed)
        self.allowed_tools: Optional[set] = allowed_tools
        # Tracks the last 4 (tool, args_json) calls for stuck-loop detection
        self._recent_calls: deque = deque(maxlen=4)
        # How many times we've already warned about a stuck loop this session
        self._stuck_warn_count: int = 0
        self._file_write_counts: Dict[str, int] = {}   # path → write count
        self._patch_counts: Dict[str, int] = {}         # path → patch count
        self._patch_not_found_counts: Dict[str, int] = {}  # path → search-not-found count
        self._failed_commands: Dict[str, int] = {}      # command → fail count
        self._server_procs: Dict[str, Any] = {}         # name → Popen
        self._read_file_counts: Dict[str, int] = {}     # path → read count (no-offset reads)
        # Updated by run_react before each dispatch so _handle_finish can pace early exits
        self._iteration_count: int = 0
        self._max_iterations: int = 50
        # Optional callback(plan_text: str) -> None. Wired by OllamaCommandAgent so that
        # write_plan invocations can refresh a pinned "📋 PLAN" slot in the agent's
        # message history. None when running without an agent (tests, etc).
        self._save_plan_cb: Optional[Callable[[str], None]] = None
        # Optional callback(label: str) -> str — mirrors flat react_tools.py. Used by
        # gui_task / save_context tools when those are wired in. Safe to leave None.
        self._save_context_cb: Optional[Callable[[str], str]] = None

    def reset_phase_state(self) -> None:
        """Reset per-phase state. Call between chain phases."""
        self._recent_calls.clear()
        self._stuck_warn_count = 0
        self._file_write_counts.clear()
        self._patch_counts.clear()
        self._patch_not_found_counts.clear()
        self._failed_commands.clear()
        self._read_file_counts.clear()
        # _server_procs intentionally NOT cleared — servers persist across phases

    # ---------------------------------------------------------------- gate --

    def dispatch(
        self,
        tool: str,
        args: dict,
        confidence: int,
        confirm_cb: Optional[Callable[[str], bool]] = None,
    ) -> ToolResult:
        """Route a tool call through the safety/confidence gate."""

        # Stuck-loop guard: if the last 4 calls are identical, warn first then terminate.
        # Exempt idempotent read-only tools — re-calling them is harmless and often useful.
        _STUCK_EXEMPT = {"read_file", "validate_arch", "get_deps", "write_plan"}
        call_key = (tool, json.dumps(args, sort_keys=True))
        if tool not in _STUCK_EXEMPT:
            self._recent_calls.append(call_key)
        if (
            tool not in _STUCK_EXEMPT
            and len(self._recent_calls) == 4
            and len(set(self._recent_calls)) == 1
        ):
            self._stuck_warn_count += 1
            self._recent_calls.clear()  # reset so agent gets a fresh chance
            if self._stuck_warn_count >= 2:
                # Second stuck loop in the same session → terminate
                return ToolResult(
                    False, "", "Stuck: identical call loop persists after warning. Terminating.",
                    {"stuck": True}
                )
            # First occurrence → warn the agent but continue
            stuck_tool = tool
            stuck_cmd = args.get("command", args.get("path", str(args)))[:200]
            return ToolResult(
                False,
                "",
                (
                    f"STUCK LOOP DETECTED: You have called '{stuck_tool}' with identical "
                    f"arguments 4 times in a row:\n  {stuck_cmd}\n\n"
                    f"This approach is not working. You MUST take a completely different action:\n"
                    f"• If a server won't start: read the log file with read_file /tmp/server.log\n"
                    f"• If a file isn't found: use execute_command 'find /home -name FILENAME 2>/dev/null | head -5'\n"
                    f"• If curl keeps failing: read the app source to see what routes actually exist\n"
                    f"• If install keeps failing: try a different package name or approach\n"
                    f"Do NOT retry the same command. Take a diagnostic step first."
                ),
                {"stuck": "warn"},
            )

        if tool not in self.TOOL_NAMES:
            return ToolResult(False, "", f"Unknown tool: {tool}", {})

        # ---- minion tool whitelist gate ----
        if self.allowed_tools is not None and tool not in self.allowed_tools:
            return ToolResult(
                False, "",
                f"HARD BLOCK: '{tool}' is NOT available to this agent. "
                f"Do NOT attempt it again. Your ONLY valid tools are: {sorted(self.allowed_tools)}. "
                f"Immediately switch to one of these.",
                {"blocked_tool": tool, "available_tools": list(self.allowed_tools)},
            )

        # ---- safety / confidence gate ----
        if tool == "execute_command":
            command = args.get("command", "")
            is_safe, risk, reason = self.safety_validator.validate_command(command)

            if not is_safe or risk == "blocked":
                print(f"\n🛡️  BLOCKED: {reason}")
                return ToolResult(False, "", f"Blocked: {reason}", {"risk": "blocked"})

            # Always show the command breakdown so the user can follow along
            explanation = ""
            if self.explain_cb:
                explanation = self.explain_cb(command)

            needs_confirm = (confidence < 90) or (risk in ("medium", "high"))
            if needs_confirm:
                if confirm_cb:
                    risk_icon = {"medium": "🟡", "high": "🔴"}.get(risk, "⚠️ ")
                    print(f"\n{risk_icon} [{risk.upper()} RISK]  Confidence: {confidence}%")
                    print(f"   Reason: {reason}")
                    if explanation:
                        print(f"\n{explanation}")
                    prompt = "\n   Proceed? (y/n): "
                    if not confirm_cb(prompt, command):
                        return ToolResult(False, "", "Cancelled by user", {"risk": risk})
                else:
                    # No TTY available — reject high-risk actions automatically
                    return ToolResult(
                        False, "",
                        "Requires confirmation but no confirm_cb available",
                        {"risk": risk},
                    )
            else:
                # Auto-execute — still show the breakdown so nothing is opaque
                if explanation:
                    print(f"\n{explanation}")

        elif tool == "create_file":
            if confidence < 90 and confirm_cb:
                prompt = (
                    f"\n📝 Create file: {args.get('path', '?')}\n"
                    f"   Confidence: {confidence}%\n"
                    f"   Proceed? (y/n): "
                )
                if not confirm_cb(prompt, f"create file: {args.get('path', '?')}"):
                    return ToolResult(False, "", "Cancelled by user", {})

        elif tool == "patch_file":
            if confidence < 90 and confirm_cb:
                prompt = (
                    f"\n🔧 Patch file: {args.get('path', '?')}\n"
                    f"   Search: {str(args.get('search', ''))[:80]}\n"
                    f"   Confidence: {confidence}%\n"
                    f"   Proceed? (y/n): "
                )
                if not confirm_cb(prompt, f"patch file: {args.get('path', '?')} — search: {str(args.get('search', ''))[:120]}"):
                    return ToolResult(False, "", "Cancelled by user", {})

        elif tool == "manage_server":
            action = args.get("action", "")
            command = args.get("command", "")
            if action in ("start", "restart") and command:
                is_safe, risk, reason = self.safety_validator.validate_command(command)
                if not is_safe or risk == "blocked":
                    print(f"\n🛡️  BLOCKED: {reason}")
                    return ToolResult(False, "", f"Blocked: {reason}", {"risk": "blocked"})
                if self.explain_cb:
                    print(f"\n{self.explain_cb(command)}")
                needs_confirm = (confidence < 90) or (risk in ("medium", "high"))
                if needs_confirm and confirm_cb:
                    risk_icon = {"medium": "🟡", "high": "🔴"}.get(risk, "⚠️ ")
                    print(f"\n{risk_icon} [{risk.upper()} RISK]  Confidence: {confidence}%")
                    if not confirm_cb(f"\n   Start server? (y/n): ", command):
                        return ToolResult(False, "", "Cancelled by user", {})
                elif needs_confirm:
                    return ToolResult(False, "", "Requires confirmation but no confirm_cb", {"risk": risk})

        handler_map = {
            "execute_command": self._handle_execute_command,
            "create_file":     self._handle_create_file,
            "patch_file":      self._handle_patch_file,
            "web_search":      self._handle_web_search,
            "read_file":       self._handle_read_file,
            "memory_lookup":   self._handle_memory_lookup,
            "finish":          self._handle_finish,
            "manage_server":   self._handle_manage_server,
            "validate_arch":   self._handle_validate_arch,
            "get_deps":        self._handle_get_deps,
            "write_plan":      self._handle_write_plan,
        }
        return handler_map[tool](args)

    # --------------------------------------------------- tool handlers ------

    # Package managers and other slow commands that need extended timeouts
    _LONG_RUNNING_PATTERNS = [
        "pacman", "yay", "paru",           # Arch package managers
        "apt", "apt-get", "dnf", "yum",    # Other Linux PMs
        "pip install", "pip3 install",     # Python packages
        "npm install", "yarn add", "npx",  # Node packages / npx runners
        "cargo build", "cargo install",    # Rust
        "make", "cmake",                   # Build systems
    ]
    _LONG_RUNNING_MIN_TIMEOUT = 300

    # (pm_name, detection_regex, absent_flag_checks, inject_after_regex, flag)
    # If detection_regex matches AND none of absent_flag_checks appear in the command,
    # inject `flag` immediately after the text matched by inject_after_regex.
    _PM_NOINTERACTIVE = [
        ("pacman",
         r"pacman\s+(-[A-Za-z]*[SRU]|--sync|--remove|--upgrade)",
         ["--noconfirm"],
         r"(pacman\s+)",
         "--noconfirm"),
    ]

    # (pm_token, error_pattern_in_stderr, fix_command)
    _PM_LOCK_FIXES = [
        ("pacman",
         r"unable to lock database",
         "sudo rm -f /var/lib/pacman/db.lck"),
    ]

    def _handle_execute_command(self, args: dict) -> ToolResult:
        command = args.get("command", "")
        timeout = int(args.get("timeout", 30))

        if not command:
            return ToolResult(False, "", "No command provided", {})

        # Auto-extend timeout for package managers and build tools
        if any(p in command for p in self._LONG_RUNNING_PATTERNS):
            if timeout < self._LONG_RUNNING_MIN_TIMEOUT:
                print(f"  ⏱️  Auto-extending timeout to {self._LONG_RUNNING_MIN_TIMEOUT}s for package/build command")
                timeout = self._LONG_RUNNING_MIN_TIMEOUT

        # Auto-inject non-interactive flags for any supported package manager
        for pm_name, detect_pat, absent_checks, inject_pat, flag in self._PM_NOINTERACTIVE:
            if re.search(detect_pat, command) and all(s not in command for s in absent_checks):
                new_cmd = re.sub(inject_pat, lambda m, f=flag: m.group(0) + f + " ", command, count=1)
                if new_cmd != command:
                    command = new_cmd
                    print(f"  🔒 Auto-injected {flag} for {pm_name}")
                    break

        print(f"\n🔧 Running: {command}")
        start = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                executable="/bin/bash",
            )
            duration_ms = int((time.time() - start) * 1000)
            success = result.returncode == 0

            # Auto-recover from package manager lock files on first failure
            if not success:
                for pm_token, err_pat, fix_cmd in self._PM_LOCK_FIXES:
                    if pm_token in command and re.search(err_pat, result.stderr):
                        print(f"  🔓 {pm_token} lock detected — running fix and retrying once...")
                        subprocess.run(fix_cmd, shell=True, timeout=15, capture_output=True)
                        time.sleep(0.5)
                        start = time.time()
                        result = subprocess.run(
                            command, shell=True, capture_output=True, text=True,
                            timeout=timeout, executable="/bin/bash",
                        )
                        duration_ms = int((time.time() - start) * 1000)
                        success = result.returncode == 0
                        break

            if success:
                print(f"✅ Done ({duration_ms}ms)")
                self._failed_commands.pop(command.strip(), None)   # clear on success
            else:
                print(f"❌ Failed — exit code {result.returncode} ({duration_ms}ms)")
                norm = command.strip()
                self._failed_commands[norm] = self._failed_commands.get(norm, 0) + 1
                if self._failed_commands[norm] >= 2:
                    return ToolResult(
                        False, "",
                        f"BLOCKED: this command has failed {self._failed_commands[norm]} times:\n"
                        f"  {norm[:200]}\n\n"
                        f"You MUST take a fundamentally different approach.\n"
                        f"• Fix the ROOT CAUSE from the error above\n"
                        f"• Try a different command\n"
                        f"• Use web_search or memory_lookup for alternatives",
                        {"blocked_repeated_failure": True}
                    )

            if result.stdout:
                lines = result.stdout.split("\n")
                preview = "\n".join(lines[:20])
                print(f"\nOutput:\n{preview}")
                if len(lines) > 20:
                    print(f"  ... ({len(lines) - 20} more lines)")

            if result.stderr and not success:
                err_lines = result.stderr.strip().split("\n")
                preview = "\n".join(err_lines[:15])
                print(f"\nError:\n{preview}")
                if len(err_lines) > 15:
                    print(f"  ... ({len(err_lines) - 15} more lines)")

            return ToolResult(
                success=success,
                output=result.stdout,
                error=result.stderr,
                metadata={"exit_code": result.returncode, "duration_ms": duration_ms},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Timeout after {timeout}s", {"exit_code": -1})
        except Exception as e:
            return ToolResult(False, "", str(e), {"exit_code": -1})

    def _handle_create_file(self, args: dict) -> ToolResult:
        path = os.path.expanduser(args.get("path", ""))
        content = args.get("content", "")

        if not path:
            return ToolResult(False, "", "No path provided", {})

        protected_dirs = [
            "/bin", "/boot", "/dev", "/etc", "/lib", "/lib64",
            "/proc", "/root", "/sbin", "/sys", "/usr/bin", "/usr/sbin",
        ]
        for p in protected_dirs:
            if path.startswith(p):
                return ToolResult(False, "", f"Blocked: cannot write to {p}", {})

        # Cap at 3 successful writes per path
        self._file_write_counts[path] = self._file_write_counts.get(path, 0) + 1
        if self._file_write_counts[path] > 3:
            return ToolResult(
                False, "",
                f"BLOCKED: '{path}' has been written {self._file_write_counts[path]-1} times.\n"
                f"You MUST provide concrete failure evidence before rewriting.\n"
                f"Options: run the file and show the error, OR use patch_file for targeted edits,\n"
                f"OR call finish(success=false) if genuinely unresolvable.",
                {"blocked_write_cap": True}
            )

        print(f"\n📝 Creating file: {path}")
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            print(f"✅ Created: {path}")

            # Mechanical syntax check
            syntax_note = ""
            ext = os.path.splitext(path)[1].lower()
            if ext == ".py":
                try:
                    chk = subprocess.run(["python3", "-m", "py_compile", path],
                                         capture_output=True, text=True, timeout=10)
                    if chk.returncode != 0:
                        syntax_note = f"\n\n⚠️  SYNTAX ERROR in {path}:\n{chk.stderr.strip()[:500]}"
                        print(f"⚠️  Syntax error detected in {path}")
                except Exception as e:
                    syntax_note = f"\n\n⚠️  Syntax check failed: {e}"
            elif ext in (".js", ".ts"):
                try:
                    chk = subprocess.run(["node", "--check", path],
                                         capture_output=True, text=True, timeout=10)
                    if chk.returncode != 0:
                        syntax_note = f"\n\n⚠️  SYNTAX ERROR in {path}:\n{chk.stderr.strip()[:500]}"
                        print(f"⚠️  Syntax error detected in {path}")
                except Exception as e:
                    syntax_note = f"\n\n⚠️  Syntax check failed: {e}"

            return ToolResult(True, f"Created {path}{syntax_note}", "", {"path": path})
        except PermissionError:
            home = os.path.expanduser("~")
            return ToolResult(False, "",
                f"Permission denied writing to '{path}'. "
                f"This tool cannot use sudo. "
                f"Use a path under your home directory instead: {home}/projects/ "
                f"(NOT /home/user/ — the actual home is {home}). "
                f"For system paths, use execute_command with: "
                f"sudo tee {path} < /dev/stdin  (pipe content via heredoc).",
                {"permission_denied": True, "suggested_home": home})
        except Exception as e:
            return ToolResult(False, "", f"Failed to create {path}: {e}", {})

    @staticmethod
    def _fuzzy_patch(original: str, search: str, replace: str):
        """Try whitespace-flexible regex match. Returns patched content or None."""
        escaped = re.escape(search.strip())
        # Make any run of escaped whitespace flexible
        flexible = re.sub(r'((?:\\ |\\\t|\\\n)+)', r'\\s+', escaped)
        try:
            m = re.search(flexible, original)
        except re.error:
            return None
        if m:
            return original[: m.start()] + replace + original[m.end() :]
        return None

    def _handle_patch_file(self, args: dict) -> ToolResult:
        path = os.path.expanduser(args.get("path", ""))
        search = args.get("search", "")
        replace = args.get("replace", "")

        if not path:
            return ToolResult(False, "", "No path provided", {})
        if not search:
            return ToolResult(False, "", "No search string provided", {})

        try:
            with open(path, "r") as f:
                original = f.read()
        except FileNotFoundError:
            return ToolResult(False, "", f"File not found: {path}", {})
        except Exception as e:
            return ToolResult(False, "", f"Cannot read {path}: {e}", {})

        if search not in original:
            # Try whitespace-flexible match before counting as a failure
            fuzzy_result = self._fuzzy_patch(original, search, replace)
            if fuzzy_result is not None:
                print(f"  🔍 Fuzzy patch matched in {path}")
                new_content = fuzzy_result
            else:
                # True failure — track count and give context
                self._patch_not_found_counts[path] = self._patch_not_found_counts.get(path, 0) + 1
                # Context hint: find lines containing first token of search
                first_token = search.strip().split()[0][:30] if search.strip() else ""
                ctx_lines = [
                    f"  L{i+1}: {line.rstrip()[:100]}"
                    for i, line in enumerate(original.split("\n"))
                    if first_token and first_token in line
                ][:5]
                ctx_str = ("\nNearest matching lines:\n" + "\n".join(ctx_lines)) if ctx_lines else ""
                if self._patch_not_found_counts[path] >= 2:
                    return ToolResult(
                        False, "",
                        f"Search string not found in {path} (2nd failure). "
                        f"Triggering automatic full-file rewrite.{ctx_str}",
                        {"trigger_rewrite": True, "path": path},
                    )
                # Small-file hint on first failure
                file_size = len(original)
                size_hint = (
                    f" File is only {file_size:,} bytes — consider using create_file to rewrite it entirely."
                    if file_size < 10_000 else ""
                )
                return ToolResult(
                    False, "",
                    f"Search string not found in {path}.{size_hint}{ctx_str}",
                    {},
                )
        else:
            new_content = original.replace(search, replace, 1)

        # Per-file patch counter — block after 5 patches to prevent loops
        self._patch_counts[path] = self._patch_counts.get(path, 0) + 1
        if self._patch_counts[path] > 5:
            return ToolResult(
                False, "",
                f"BLOCKED: '{path}' has been patched {self._patch_counts[path] - 1} times this phase. "
                "Use create_file to rewrite the whole file instead.",
                {"force_rewrite": True},
            )

        # No-op detection — catches hallucination loops where search==replace
        if new_content == original:
            return ToolResult(
                False, "",
                "Error: search string found but replacement produced identical content. "
                "You are in a no-op loop. Read the file again and provide a different search string.",
                {},
            )

        # Create timestamped backup
        backup_dir = os.path.expanduser("~/.agent_bin/backups")
        os.makedirs(backup_dir, exist_ok=True)
        safe_name = path.lstrip("/").replace("/", "_")
        backup_path = os.path.join(backup_dir, f"{safe_name}.{int(time.time())}")

        try:
            with open(backup_path, "w") as f:
                f.write(original)
        except Exception as e:
            return ToolResult(False, "", f"Could not create backup: {e}", {})

        try:
            with open(path, "w") as f:
                f.write(new_content)
            print(f"✅ Patched: {path}  (backup: {backup_path})")

            # Mechanical syntax check (same gate as create_file)
            syntax_note = ""
            ext = os.path.splitext(path)[1].lower()
            if ext == ".py":
                try:
                    chk = subprocess.run(["python3", "-m", "py_compile", path],
                                         capture_output=True, text=True, timeout=10)
                    if chk.returncode != 0:
                        syntax_note = f"\n\n⚠️  SYNTAX ERROR in {path}:\n{chk.stderr.strip()[:500]}"
                        print(f"⚠️  Syntax error detected in {path}")
                except Exception as e:
                    syntax_note = f"\n\n⚠️  Syntax check failed: {e}"
            elif ext in (".js", ".ts"):
                try:
                    chk = subprocess.run(["node", "--check", path],
                                         capture_output=True, text=True, timeout=10)
                    if chk.returncode != 0:
                        syntax_note = f"\n\n⚠️  SYNTAX ERROR in {path}:\n{chk.stderr.strip()[:500]}"
                        print(f"⚠️  Syntax error detected in {path}")
                except Exception as e:
                    syntax_note = f"\n\n⚠️  Syntax check failed: {e}"

            return ToolResult(True, f"Patched {path}{syntax_note}", "", {"backup_path": backup_path})
        except Exception as e:
            # Attempt to restore from backup
            try:
                with open(backup_path, "r") as f:
                    restored = f.read()
                with open(path, "w") as f:
                    f.write(restored)
            except Exception:
                pass
            return ToolResult(False, "", f"Write failed: {e}", {})

    def _handle_web_search(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(False, "", "No query provided", {})
        try:
            results = self.search_agent.search(query)
            return ToolResult(True, results, "", {})
        except Exception as e:
            return ToolResult(False, "", str(e), {})

    def _handle_read_file(self, args: dict) -> ToolResult:
        path = os.path.expanduser(args.get("path", ""))
        if not path:
            return ToolResult(False, "", "No path provided", {})
        offset = int(args.get("offset", 0))
        limit  = int(args.get("limit", 200))

        # Track ALL reads (any offset) per path — catches pagination loops too
        self._read_file_counts[path] = self._read_file_counts.get(path, 0) + 1
        total_reads = self._read_file_counts[path]

        # Hard block at 8 total reads of the same file
        if total_reads > 8:
            fname = os.path.basename(path)
            return ToolResult(
                False, "",
                f"BLOCKED: {fname} has been read {total_reads} times this phase. "
                f"Re-reading is NOT helping. You MUST take a different action:\n"
                f"• Use execute_command: grep -n 'symbol' {path}\n"
                f"• Use execute_command: python3 -c \"import ast; ...\" to inspect structure\n"
                f"• Use patch_file or create_file to fix what you already know is wrong\n"
                f"• If imports are missing, ADD them — don't keep reading the file.",
                {"read_blocked": True, "total_reads": total_reads}
            )

        try:
            with open(path, "r") as f:
                lines = f.readlines()
            total = len(lines)
            slice_ = lines[offset : offset + limit]
            content = "".join(slice_)

            # Header so model always knows exactly where it is
            end_line = min(offset + limit, total)
            header = f"[FILE: {path} | TOTAL: {total} lines | SHOWING: lines {offset+1}-{end_line}]\n"
            content = header + content

            remaining = total - (offset + limit)
            if remaining > 0:
                content += (
                    f"\n[... {remaining} more lines. "
                    f"Use offset={offset + limit} to continue, "
                    f"or grep to find specific content instead of paginating.]"
                )

            # Warn at 4+ reads regardless of offset
            if total_reads >= 4:
                content += (
                    f"\n\n⚠️  READ #{total_reads} of {path}. "
                    f"You have {8 - total_reads} reads left before this file is blocked.\n"
                    f"STOP re-reading. Use grep or just patch what you know is wrong."
                )

            return ToolResult(True, content, "", {
                "path": path, "total_lines": total,
                "shown_lines": len(slice_), "offset": offset,
                "total_reads": total_reads,
            })
        except FileNotFoundError:
            return ToolResult(False, "", f"File not found: {path}", {})
        except Exception as e:
            return ToolResult(False, "", str(e), {})

    def _handle_memory_lookup(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(False, "", "No query provided", {})
        try:
            results = self.memory.lookup(query)
            if not results:
                return ToolResult(True, "No matching records found in memory.", "", {})
            lines = []
            for r in results:
                lines.append(
                    f"Command: {r['command']}\n"
                    f"  Task: {r.get('task', 'N/A')}\n"
                    f"  Success count: {r.get('success_count', 1)}\n"
                    f"  Last used: {r.get('used_at', 'N/A')}"
                )
            return ToolResult(True, "\n\n".join(lines), "", {"count": len(results)})
        except Exception as e:
            return ToolResult(False, "", str(e), {})

    def _handle_validate_arch(self, args: dict) -> ToolResult:
        """Validate a DOCS/ARCH.json file against arch_schema. No LLM call.
        Returns {ok, errors, summary}. Missing file is a warning, not a hard fail.
        """
        path = os.path.expanduser(args.get("path") or "DOCS/ARCH.json")
        try:
            import arch_schema  # flat import — added to sys.path by server.py
        except ImportError as e:
            return ToolResult(False, "", f"arch_schema module unavailable: {e}", {})

        if not os.path.exists(path):
            return ToolResult(
                True,
                f"ARCH.json not found at {path} — treating as empty contract (warning).",
                "",
                {"ok": True, "warn": "no ARCH.json", "errors": []},
            )

        try:
            data = arch_schema.load_arch(path)
        except Exception as e:
            return ToolResult(
                False, "", f"ARCH.json parse failed: {e}",
                {"ok": False, "errors": [str(e)]},
            )

        ok, errors = arch_schema.validate_arch(data)
        summary = arch_schema.extract_summary(data, max_chars=400)
        hard = [e for e in errors if not e.startswith("warning:")]
        warn = [e for e in errors if e.startswith("warning:")]

        head = f"ARCH.json @ {path}\n{summary}\n"
        body_parts = []
        if hard:
            body_parts.append("ERRORS:\n" + "\n".join(f"  ✗ {e}" for e in hard))
        if warn:
            body_parts.append("WARNINGS:\n" + "\n".join(f"  ⚠ {e}" for e in warn))
        if not body_parts:
            body_parts.append("✅ No violations.")
        output = head + "\n".join(body_parts)

        return ToolResult(
            ok, output, "" if ok else "arch validation failed",
            {"ok": ok, "errors": errors, "summary": summary},
        )

    def _handle_get_deps(self, args: dict) -> ToolResult:
        """Return the import dependency graph for a set of files (or current workspace).

        args:
          paths: list[str] — files to analyse (required)
        Response metadata: {graph, reverse_graph, dependents_of_paths}.
        """
        try:
            import dep_graph as _dep
        except ImportError as e:
            return ToolResult(False, "", f"dep_graph module unavailable: {e}", {})
        paths = args.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        paths = [os.path.expanduser(p) for p in paths if isinstance(p, str)]
        if not paths:
            return ToolResult(False, "", "get_deps requires paths:[...]", {})
        g = _dep.build_graph(paths)
        rev = _dep.reverse(g)
        dependents_of = {p: _dep.files_affected_by({p}, g) for p in paths if p in g}

        # Human-readable output
        lines = ["Dep graph:"]
        for f, ds in sorted(g.items()):
            short = os.path.basename(f)
            lines.append(f"  {short} -> {[os.path.basename(d) for d in ds]}")
        for p, affected in dependents_of.items():
            if affected:
                lines.append(f"  Dependents of {os.path.basename(p)}: "
                             f"{sorted(os.path.basename(x) for x in affected)}")
        output = "\n".join(lines)[:3000]

        meta = {
            "graph": {k: list(v) for k, v in g.items()},
            "reverse_graph": rev,
            "dependents_of_paths": {p: sorted(s) for p, s in dependents_of.items()},
        }
        return ToolResult(True, output, "", meta)

    def _handle_write_plan(self, args: dict) -> ToolResult:
        """Persist a markdown task plan and refresh the agent's pinned PLAN slot.

        Args: {"plan": str} — markdown body. Convention: the planner/builder writes
        ## Architecture / ## Files / ## Dependencies sections with `- [ ]` checkboxes
        and re-calls write_plan with `- [x]` after each file is written.

        Idempotent: rewrites the same path on every call. Returns checkbox totals
        so the agent (and the pinned slot) can see plan completion at a glance.
        """
        plan = args.get("plan", "")
        if not isinstance(plan, str) or not plan.strip():
            return ToolResult(False, "", "write_plan requires a non-empty 'plan' string", {})

        agent_id = "agent"
        try:
            if self.memory is not None and getattr(self.memory, "agent_id", None):
                agent_id = self.memory.agent_id
        except Exception:
            pass
        # Sanitize for filename
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(agent_id))[:40] or "agent"

        plans_dir = os.path.expanduser("~/.agent_bin/plans")
        try:
            os.makedirs(plans_dir, exist_ok=True)
        except Exception as e:
            return ToolResult(False, "", f"Could not create plans dir: {e}", {})
        path = os.path.join(plans_dir, f"{safe_id}_plan.md")

        try:
            with open(path, "w") as f:
                f.write(plan)
        except Exception as e:
            return ToolResult(False, "", f"Failed to write plan to {path}: {e}", {})

        total = len(re.findall(r"^\s*- \[[ xX]\]", plan, re.M))
        checked = len(re.findall(r"^\s*- \[[xX]\]", plan, re.M))

        # Refresh pinned PLAN slot via callback (mirrors _save_context_cb pattern)
        if callable(self._save_plan_cb):
            try:
                self._save_plan_cb(plan)
            except Exception as e:
                # Non-fatal — plan is still on disk. Log and continue.
                print(f"⚠️  _save_plan_cb failed (non-fatal): {e}")

        summary = f"plan saved ({checked}/{total} items checked)"
        print(f"\n📋 {summary} → {path}")
        return ToolResult(
            True, summary, "",
            {"plan_path": path, "checked": checked, "total": total},
        )

    def _handle_finish(self, args: dict) -> ToolResult:
        summary  = args.get("summary", "Task completed.")
        success  = bool(args.get("success", True))
        declared = args.get("files_created", [])

        # --- Guard 1: verify all declared files exist on disk ---
        missing = [f for f in declared if not os.path.exists(os.path.expanduser(f))]
        if missing:
            msg = (
                f"finish() REJECTED — {len(missing)} declared file(s) not found on disk:\n"
                + "\n".join(f"  ✗ {f}" for f in missing)
                + "\nCreate the missing files first, then call finish() again."
            )
            print(f"\n{'=' * 70}")
            print(f"🚫 FINISH REJECTED (missing files):")
            for f in missing:
                print(f"   ✗ {f}")
            print(f"{'=' * 70}")
            return ToolResult(False, "", msg, {"missing_files": missing})

        # --- Guard 2: reject no-op early exits (coder roles only) ---
        # Skip for commander/tester style roles that don't have create_file/patch_file.
        is_coder = True
        try:
            if self.allowed_tools is not None:
                is_coder = ("create_file" in self.allowed_tools) or ("patch_file" in self.allowed_tools)
        except AttributeError:
            is_coder = True
        if is_coder and success and not declared and self._max_iterations > 0:
            budget_pct = self._iteration_count / self._max_iterations
            if budget_pct < 0.5:
                msg = (
                    f"finish() REJECTED — no files_created declared and only "
                    f"{self._iteration_count}/{self._max_iterations} iterations used "
                    f"({budget_pct:.0%} of budget). "
                    "Complete the task or list all files you wrote in files_created."
                )
                print(f"\n{'=' * 70}")
                print(f"🚫 FINISH REJECTED (no files, too early): {self._iteration_count}/{self._max_iterations} iters")
                print(f"{'=' * 70}")
                return ToolResult(False, "", msg, {})

        # --- Accepted ---
        icon = "✅" if success else "⚠️ "
        print(f"\n{'=' * 70}")
        print(f"{icon} FINISH: {summary}")
        if declared:
            print(f"   Files verified: {len(declared)}")
        print(f"{'=' * 70}")
        return ToolResult(success, summary, "", {"finished": True, "files_created": declared})

    def _handle_manage_server(self, args: dict) -> ToolResult:
        action = (args.get("action") or "").lower()
        name = args.get("name", "")
        command = args.get("command", "")
        if not name:
            return ToolResult(False, "", "manage_server requires 'name'", {})
        if action not in ("start", "stop", "status", "restart"):
            return ToolResult(False, "", f"Unknown action '{action}'. Use: start|stop|status|restart", {})

        if action == "status":
            proc = self._server_procs.get(name)
            if proc is None:
                return ToolResult(True, f"'{name}': not tracked this session", "", {})
            running = proc.poll() is None
            return ToolResult(True, f"'{name}': {'running' if running else 'stopped'} (pid={proc.pid})", "", {"running": running})

        if action in ("stop", "restart"):
            proc = self._server_procs.get(name)
            if proc and proc.poll() is None:
                print(f"\n🛑 Stopping '{name}' (pid={proc.pid})")
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                except Exception as e:
                    return ToolResult(False, "", f"Stop failed: {e}", {})
                del self._server_procs[name]
                print(f"✅ Stopped '{name}'")
            elif action == "stop":
                return ToolResult(True, f"'{name}' was not running", "", {})

        if action in ("start", "restart"):
            if not command:
                return ToolResult(False, "", "manage_server start requires 'command'", {})
            existing = self._server_procs.get(name)
            if existing and existing.poll() is None:
                return ToolResult(False, "", f"'{name}' already running (pid={existing.pid}). Use restart.", {"pid": existing.pid})
            print(f"\n🚀 Starting '{name}': {command}")
            try:
                proc = subprocess.Popen(command, shell=True, executable="/bin/bash",
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                self._server_procs[name] = proc
                time.sleep(1)
                if proc.poll() is not None:
                    return ToolResult(False, "", f"'{name}' exited immediately (code={proc.returncode}). Check the command.", {})
                print(f"✅ '{name}' started (pid={proc.pid})")
                return ToolResult(True, f"'{name}' running (pid={proc.pid}). Redirect server logs in command: e.g. 'uvicorn ... >/tmp/{name}.log 2>&1'", "", {"pid": proc.pid})
            except Exception as e:
                return ToolResult(False, "", f"Start failed: {e}", {})
