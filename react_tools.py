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
    }

    def __init__(self, safety_validator, search_agent, memory, explain_cb=None):
        self.safety_validator = safety_validator
        self.search_agent = search_agent
        self.memory = memory
        # Optional callable(command: str) -> str that returns a human-readable
        # breakdown of the command segments, shown before execution / confirmation.
        self.explain_cb = explain_cb
        # Tracks the last 4 (tool, args_json) calls for stuck-loop detection
        self._recent_calls: deque = deque(maxlen=4)
        # How many times we've already warned about a stuck loop this session
        self._stuck_warn_count: int = 0

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
                    if not confirm_cb(prompt):
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
                if not confirm_cb(prompt):
                    return ToolResult(False, "", "Cancelled by user", {})

        elif tool == "patch_file":
            if confidence < 90 and confirm_cb:
                prompt = (
                    f"\n🔧 Patch file: {args.get('path', '?')}\n"
                    f"   Search: {str(args.get('search', ''))[:80]}\n"
                    f"   Confidence: {confidence}%\n"
                    f"   Proceed? (y/n): "
                )
                if not confirm_cb(prompt):
                    return ToolResult(False, "", "Cancelled by user", {})

        handler_map = {
            "execute_command": self._handle_execute_command,
            "create_file":     self._handle_create_file,
            "patch_file":      self._handle_patch_file,
            "web_search":      self._handle_web_search,
            "read_file":       self._handle_read_file,
            "memory_lookup":   self._handle_memory_lookup,
            "finish":          self._handle_finish,
        }
        return handler_map[tool](args)

    # --------------------------------------------------- tool handlers ------

    # Package managers and other slow commands that need extended timeouts
    _LONG_RUNNING_PATTERNS = [
        "pacman", "yay", "paru",           # Arch package managers
        "apt", "apt-get", "dnf", "yum",    # Other Linux PMs
        "pip install", "pip3 install",     # Python packages
        "npm install", "yarn add",         # Node packages
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
            else:
                print(f"❌ Failed — exit code {result.returncode} ({duration_ms}ms)")

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

        print(f"\n📝 Creating file: {path}")
        try:
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            print(f"✅ Created: {path}")
            return ToolResult(True, f"Created {path}", "", {"path": path})
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
            return ToolResult(False, "", f"Search string not found in {path}", {})

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

        # Apply patch (first occurrence only)
        new_content = original.replace(search, replace, 1)

        try:
            with open(path, "w") as f:
                f.write(new_content)
            print(f"✅ Patched: {path}  (backup: {backup_path})")
            return ToolResult(True, f"Patched {path}", "", {"backup_path": backup_path})
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
        try:
            with open(path, "r") as f:
                content = f.read()
            return ToolResult(True, content, "", {"path": path, "size": len(content)})
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
