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
        "gui_task",
        "save_context",
        "publish_context",
        "read_context",
    }

    def __init__(self, safety_validator, search_agent, memory, explain_cb=None,
                 allowed_tools=None):
        self.safety_validator = safety_validator
        self.search_agent = search_agent
        self.memory = memory
        # AgentMemory reference for shared context board
        self.agent_memory = memory
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
        call_key = (tool, json.dumps(args, sort_keys=True))
        self._recent_calls.append(call_key)
        if (
            len(self._recent_calls) == 4
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
            "gui_task":        self._handle_gui_task,
            "save_context":    self._handle_save_context,
            "publish_context": self._handle_publish_context,
            "read_context":    self._handle_read_context,
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
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            total = len(lines)
            slice_ = lines[offset : offset + limit]
            content = "".join(slice_)
            remaining = total - (offset + limit)
            if remaining > 0:
                content += (
                    f"\n\n[... {remaining} more lines not shown. "
                    f"Call read_file with offset={offset + limit} to continue reading.]"
                )
            # Track no-offset re-reads and inject redirect note at 3rd read
            if offset == 0:
                self._read_file_counts[path] = self._read_file_counts.get(path, 0) + 1
                count = self._read_file_counts[path]
                if count >= 3:
                    content += (
                        f"\n\n⚠️  You have read {path} from the top {count} times. "
                        f"If you haven't found what you need:\n"
                        f"• Use offset= to read a different section\n"
                        f"• Use execute_command with grep/sed to locate specific content\n"
                        f"• Proceed with what you already know — re-reading rarely helps"
                    )
            return ToolResult(True, content, "", {
                "path": path, "total_lines": total,
                "shown_lines": len(slice_), "offset": offset,
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

    def _handle_finish(self, args: dict) -> ToolResult:
        summary = args.get("summary", "")
        success = bool(args.get("success", True))
        icon = "✅" if success else "⚠️ "
        print(f"\n{'=' * 70}")
        print(f"{icon} FINISH: {summary}")
        print(f"{'=' * 70}")
        return ToolResult(success, summary, "", {"finished": True})

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

    def _handle_gui_task(self, args: dict) -> ToolResult:
        task = args.get("task", "")
        max_iter = int(args.get("max_iterations", 20))
        if not task:
            return ToolResult(False, "", "No task provided", {})
        # Auto-save CMD context before running the GUI agent (safety net)
        snapshot_path = None
        if hasattr(self, "_save_context_cb") and self._save_context_cb:
            try:
                snapshot_path = self._save_context_cb(f"pre_gui_{task[:20]}")
            except Exception:
                pass
        try:
            from gui_agent import GUIAgent
            agent = GUIAgent()
            result = agent.run(task=task, max_iterations=max_iter)
            summary = result.get("summary", result.get("finish_summary", ""))
            success = result.get("success", False)
            files = []
            try:
                for e in agent.agent.react_trace:
                    if e.get("tool") == "create_file" and getattr(e.get("result"), "success", False):
                        p = e.get("args", {}).get("path", "")
                        if p:
                            files.append(p)
            except Exception:
                pass
            output = f"GUI task {'succeeded' if success else 'failed'}.\nSummary: {summary}"
            if files:
                output += f"\nFiles created: {', '.join(files)}"
            meta = {"gui_iterations": result.get("iterations_used", 0),
                    "files_created": files}
            if snapshot_path:
                meta["context_snapshot"] = snapshot_path
            return ToolResult(success, output, "" if success else summary, meta)
        except ImportError:
            return ToolResult(False, "", "GUI agent not available on this machine", {})
        except Exception as e:
            return ToolResult(False, "", f"GUI task failed: {e}", {})

    def _handle_save_context(self, args: dict) -> ToolResult:
        label = args.get("label", "manual")
        # Delegate to the agent's save_context if the callback is set
        if hasattr(self, "_save_context_cb") and self._save_context_cb:
            try:
                path = self._save_context_cb(label)
                return ToolResult(True, f"Context saved → {path}", "", {"path": path})
            except Exception as e:
                return ToolResult(False, "", f"Save failed: {e}", {})
        return ToolResult(False, "", "save_context not wired up (no callback)", {})

    def _handle_publish_context(self, args: dict) -> ToolResult:
        key = args.get("key", "")
        value = args.get("value", "")
        ttl_hours = int(args.get("ttl_hours", 24))
        if not key or not value:
            return ToolResult(False, "", "key and value are required", {})
        try:
            self.agent_memory.set_context(key, value, agent_id="cmd", ttl=ttl_hours * 3600)
            return ToolResult(True, f"Published: {key} = {value}", "", {"key": key})
        except Exception as e:
            return ToolResult(False, "", f"publish_context failed: {e}", {})

    def _handle_read_context(self, args: dict) -> ToolResult:
        key = args.get("key", "")
        if not key:
            return ToolResult(False, "", "key is required", {})
        try:
            value = self.agent_memory.get_context(key)
            if value is None:
                return ToolResult(False, "", f"Key '{key}' not found or expired", {})
            return ToolResult(True, f"{key} = {value}", "", {"key": key, "value": value})
        except Exception as e:
            return ToolResult(False, "", f"read_context failed: {e}", {})
