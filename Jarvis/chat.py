#!/usr/bin/env python3
"""
chat.py — Terminal chat client for JARVIS
=========================================
Usage:
    python3 chat.py
    python3 chat.py --url http://10.0.0.58:5003
    python3 chat.py --url http://10.0.0.58:5003 --token <existing-session-token>

Requires only: requests  (pip install requests)
"""

import argparse
import json
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


# ── API helpers ───────────────────────────────────────────────────────────────

def create_session(base_url: str) -> str:
    r = requests.post(f"{base_url}/api/session", timeout=10)
    r.raise_for_status()
    return r.json()["session_token"]


def stream_chat(base_url: str, token: str, message: str) -> None:
    """POST /api/chat and print streamed NDJSON chunks as they arrive."""
    r = requests.post(
        f"{base_url}/api/chat",
        json={"message": message, "session_token": token},
        stream=True,
        timeout=300,
    )
    r.raise_for_status()

    print(f"\n{GREEN}JARVIS{RESET}  ", end="", flush=True)
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            chunk = json.loads(line)
            content = chunk.get("content", "")
            if content:
                print(content, end="", flush=True)
        except json.JSONDecodeError:
            continue
    print("\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JARVIS terminal chat client")
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"JARVIS server base URL  (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--token", default=None,
        help="Resume an existing session by token"
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Header
    print(f"\n{BOLD}{CYAN}  J A R V I S{RESET}  {DIM}{base_url}{RESET}")
    print(f"{DIM}  ────────────────────────────────────────{RESET}\n")

    # Session
    if args.token:
        token = args.token
        print(f"{DIM}  Resuming session {token[:12]}…{RESET}\n")
    else:
        try:
            token = create_session(base_url)
            print(f"{DIM}  Session {token[:12]}  ·  Ctrl+C or /exit to quit{RESET}\n")
        except Exception as e:
            print(f"  ✗  Could not connect: {e}")
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
            stream_chat(base_url, token, user_input)
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
