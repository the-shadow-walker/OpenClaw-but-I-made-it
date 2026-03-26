#!/usr/bin/env python3
"""
Swarm 3.0 — Client Entry Point
Connects to a remote Swarm API server (default: http://10.0.0.58:5002).

Usage:
  python3 run_me.py "question"           # single sync question
  python3 run_me.py -i                   # interactive REPL
  python3 run_me.py health               # server health check
  python3 run_me.py status               # job queue status
  python3 run_me.py jobs                 # list recent jobs
  python3 run_me.py result <job_id>      # fetch async result by ID
  python3 run_me.py ask "question"       # async submit → poll → print result

Environment:
  SWARM_SERVER   Base URL of the Swarm API server (default: http://10.0.0.58:5002)
  SWARM_API_KEY  Optional bearer token for authenticated servers
"""

import sys
import os
import json
import time
import argparse
import textwrap
import urllib.request
import urllib.error
import urllib.parse

DEFAULT_SERVER = os.environ.get("SWARM_SERVER", "http://10.0.0.58:5002")
API_KEY = os.environ.get("SWARM_API_KEY", "")
POLL_INTERVAL = 3   # seconds between status polls
POLL_TIMEOUT  = 300 # max seconds to wait for async result


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _get(server: str, path: str) -> dict:
    url = server.rstrip("/") + path
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[HTTP {e.code}] {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[Connection error] {e.reason}\n  Server: {server}", file=sys.stderr)
        sys.exit(1)


def _post(server: str, path: str, payload: dict) -> dict:
    url = server.rstrip("/") + path
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"[HTTP {e.code}] {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"[Connection error] {e.reason}\n  Server: {server}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_answer(result: dict) -> None:
    answer = result.get("answer") or result.get("result") or result.get("response", "")
    status = result.get("status", "")
    job_id = result.get("job_id", "")
    elapsed = result.get("elapsed_seconds")

    if job_id:
        print(f"\n─── Job {job_id} ───")
    if status and status not in ("completed", "done"):
        print(f"Status : {status}")
    if elapsed is not None:
        print(f"Time   : {elapsed:.1f}s")
    print()

    if answer:
        # Wrap long lines for terminal readability
        width = min(100, os.get_terminal_size(fallback=(100, 40)).columns)
        for line in answer.splitlines():
            if len(line) > width:
                print(textwrap.fill(line, width=width))
            else:
                print(line)
    else:
        print(json.dumps(result, indent=2))
    print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_health(server: str) -> None:
    data = _get(server, "/health")
    print(json.dumps(data, indent=2))


def cmd_status(server: str) -> None:
    data = _get(server, "/status")
    print(json.dumps(data, indent=2))


def cmd_jobs(server: str) -> None:
    data = _get(server, "/jobs")
    jobs = data.get("jobs", data)
    if isinstance(jobs, list):
        for j in jobs:
            jid = j.get("job_id", "?")
            st  = j.get("status", "?")
            q   = j.get("question", "")[:60]
            print(f"  {jid}  [{st:10s}]  {q}")
    else:
        print(json.dumps(jobs, indent=2))


def cmd_result(server: str, job_id: str) -> None:
    data = _get(server, f"/result/{job_id}")
    _print_answer(data)


def cmd_ask(server: str, question: str) -> None:
    """Submit async and poll until done."""
    print(f"Submitting: {question[:80]}…")
    resp = _post(server, "/query_async", {"question": question})
    job_id = resp.get("job_id")
    if not job_id:
        print("[Error] No job_id in response:", resp, file=sys.stderr)
        sys.exit(1)

    print(f"Job ID: {job_id}  — polling every {POLL_INTERVAL}s …")
    deadline = time.time() + POLL_TIMEOUT
    spinner = ["|", "/", "—", "\\"]
    tick = 0
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        tick += 1
        data = _get(server, f"/result/{job_id}")
        st = data.get("status", "unknown")
        print(f"  {spinner[tick % 4]}  [{st}]", end="\r", flush=True)
        if st in ("completed", "done", "error", "failed"):
            print()
            _print_answer(data)
            return

    print(f"\n[Timeout] Job {job_id} not complete after {POLL_TIMEOUT}s", file=sys.stderr)
    sys.exit(1)


def cmd_query(server: str, question: str) -> None:
    """Synchronous query — blocks until answer returned."""
    print(f"Asking: {question[:80]}…", flush=True)
    resp = _post(server, "/query", {"question": question})
    _print_answer(resp)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def cmd_repl(server: str) -> None:
    print(f"Swarm 3.0 REPL  (server: {server})")
    print("Type your question and press Enter. Commands: :health :status :jobs :quit\n")
    while True:
        try:
            line = input("swarm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue
        if line in (":quit", ":exit", "q", "quit", "exit"):
            print("Bye.")
            break
        elif line == ":health":
            cmd_health(server)
        elif line == ":status":
            cmd_status(server)
        elif line == ":jobs":
            cmd_jobs(server)
        elif line.startswith(":ask "):
            cmd_ask(server, line[5:].strip())
        else:
            cmd_query(server, line)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_me.py",
        description="Swarm 3.0 client — connect to the Swarm API server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 run_me.py "What is the escape velocity of Mars?"
              python3 run_me.py -i
              python3 run_me.py health
              python3 run_me.py ask "Design a 500N thrust rocket motor"
              python3 run_me.py result abc123
        """),
    )
    parser.add_argument("--server", "-s", default=DEFAULT_SERVER,
                        help=f"Swarm API base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="Launch interactive REPL")
    parser.add_argument("args", nargs="*", help="Command and arguments")

    opts = parser.parse_args()
    server = opts.server.rstrip("/")

    if opts.interactive:
        cmd_repl(server)
        return

    args = opts.args
    if not args:
        parser.print_help()
        return

    cmd = args[0].lower()

    if cmd == "health":
        cmd_health(server)
    elif cmd == "status":
        cmd_status(server)
    elif cmd == "jobs":
        cmd_jobs(server)
    elif cmd == "result":
        if len(args) < 2:
            print("Usage: run_me.py result <job_id>", file=sys.stderr)
            sys.exit(1)
        cmd_result(server, args[1])
    elif cmd == "ask":
        question = " ".join(args[1:]) if len(args) > 1 else ""
        if not question:
            print("Usage: run_me.py ask \"your question\"", file=sys.stderr)
            sys.exit(1)
        cmd_ask(server, question)
    else:
        # Treat the whole args list as the question
        question = " ".join(args)
        cmd_query(server, question)


if __name__ == "__main__":
    main()
