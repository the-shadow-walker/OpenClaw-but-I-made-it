#!/usr/bin/env python3
"""
chat.py — Terminal chat client for JARVIS
=========================================
Usage:
    python3 chat.py
    python3 chat.py --url http://10.0.0.58:5003
    python3 chat.py --token <existing-session-token>

Supports local Mac execution: JARVIS can embed [LOCAL: shell_cmd] tags in
responses that this client intercepts and runs on your Mac (volume, apps, etc.)

Requires: requests  (pip install requests)
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys

import requests

# ── ANSI colours ──────────────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

DEFAULT_URL = "http://10.0.0.58:5003"
LOCAL_TAG_RE = re.compile(r'\[LOCAL:\s*(.*?)\]', re.DOTALL)


# ── Platform context ───────────────────────────────────────────────────────────

def _platform_ctx() -> str:
    """One-liner context appended to every message so JARVIS knows the client OS."""
    parts = [f"os={platform.system()}"]
    if platform.system() == "Darwin":
        ver = platform.mac_ver()[0]
        if ver:
            parts.append(f"macOS={ver}")
        parts.append("osascript=available")
        parts.append("local_exec=enabled")
    parts.append(f"hostname={platform.node()}")
    try:
        parts.append(f"user={os.getlogin()}")
    except Exception:
        pass
    return "CLIENT_PLATFORM: " + ", ".join(parts)

PLATFORM_CTX = _platform_ctx()


# ── API helpers ────────────────────────────────────────────────────────────────

def create_session(base_url: str) -> str:
    r = requests.post(f"{base_url}/api/session", timeout=10)
    r.raise_for_status()
    return r.json()["session_token"]


def stream_response(base_url: str, token: str, message: str) -> list[str]:
    """
    Stream /api/chat, printing chunks as they arrive.
    Intercepts [LOCAL: cmd] tags: hides them from display, returns them as a list.
    Returns list of local shell commands to execute.
    """
    full_msg = f"{message}\n\n[{PLATFORM_CTX}]"
    r = requests.post(
        f"{base_url}/api/chat",
        json={"message": full_msg, "session_token": token},
        stream=True,
        timeout=300,
    )
    r.raise_for_status()

    print(f"\n{GREEN}JARVIS{RESET}  ", end="", flush=True)

    local_cmds: list[str] = []
    # Sliding buffer for tag interception across chunk boundaries
    buf = ""
    HOLD = len("[LOCAL:")  # chars to hold back while watching for a tag start

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line).get("content", "")
        except json.JSONDecodeError:
            continue
        if not chunk:
            continue

        buf += chunk

        # Drain the buffer, intercepting [LOCAL: ...] tags
        while True:
            idx = buf.find("[LOCAL:")
            if idx == -1:
                # No tag starting — print everything except the tail we hold back
                safe = max(0, len(buf) - HOLD)
                print(buf[:safe], end="", flush=True)
                buf = buf[safe:]
                break
            else:
                # Print everything before the tag
                print(buf[:idx], end="", flush=True)
                tag_tail = buf[idx:]
                end_idx = tag_tail.find("]")
                if end_idx == -1:
                    # Tag not yet complete — hold and wait for next chunk
                    buf = tag_tail
                    break
                # Full tag found — extract command, skip it in output
                tag = tag_tail[: end_idx + 1]
                m = LOCAL_TAG_RE.match(tag)
                if m:
                    local_cmds.append(m.group(1).strip())
                buf = tag_tail[end_idx + 1 :]

    # Flush remaining buffer (no tags can still be hiding — tag would be incomplete)
    if buf:
        print(buf, end="", flush=True)

    print("\n")
    return local_cmds


# ── Local execution ────────────────────────────────────────────────────────────

def run_local(cmd: str) -> None:
    """Execute a shell command on this Mac and print the result."""
    short = cmd[:72] + ("…" if len(cmd) > 72 else "")
    print(f"{DIM}  → {short}{RESET}", flush=True)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if out:
            print(f"{DIM}  {out}{RESET}")
        if result.returncode == 0:
            print(f"{GREEN}  ✓{RESET}")
        else:
            print(f"{YELLOW}  ⚠ exit {result.returncode}: {err[:80]}{RESET}")
    except subprocess.TimeoutExpired:
        print(f"{YELLOW}  ⚠ command timed out{RESET}")
    except Exception as e:
        print(f"{YELLOW}  ✗ {e}{RESET}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JARVIS terminal chat client")
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"JARVIS server URL (default: {DEFAULT_URL})")
    parser.add_argument("--token", default=None,
                        help="Resume an existing session token")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print(f"\n{BOLD}{CYAN}  J A R V I S{RESET}  {DIM}{base_url}{RESET}")
    print(f"{DIM}  ────────────────────────────────────────{RESET}")
    print(f"{DIM}  {PLATFORM_CTX[:60]}{RESET}\n")

    if args.token:
        token = args.token
        print(f"{DIM}  Resuming session {token[:12]}…{RESET}\n")
    else:
        try:
            token = create_session(base_url)
            print(f"{DIM}  Session {token[:12]}  ·  Ctrl+C or /exit to quit{RESET}\n")
        except Exception as e:
            print(f"  ✗  Could not connect to JARVIS: {e}")
            sys.exit(1)

    # Chat loop
    while True:
        try:
            user_input = input(f"{YELLOW}You  {RESET}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{DIM}  Signing off.{RESET}\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"\n{DIM}  Signing off.{RESET}\n")
            break

        try:
            local_cmds = stream_response(base_url, token, user_input)
            for cmd in local_cmds:
                run_local(cmd)
        except requests.exceptions.ConnectionError:
            print(f"\n  {YELLOW}⚠  Connection lost — is JARVIS up?{RESET}\n")
        except requests.exceptions.Timeout:
            print(f"\n  {YELLOW}⚠  Request timed out (model may be slow).{RESET}\n")
        except KeyboardInterrupt:
            print(f"\n  {DIM}(interrupted){RESET}\n")
        except Exception as e:
            print(f"\n  ✗  {e}\n")


if __name__ == "__main__":
    main()
