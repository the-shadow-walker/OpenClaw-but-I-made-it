"""
Interactive project Q&A client
================================
Drives the /project/start → /project/respond API loop from the terminal.

Usage:
    python3 project_chat.py "a GPS weather station with solar charging"
    python3 project_chat.py                   # prompts for description
    python3 project_chat.py --url http://10.0.0.58:5001
    python3 project_chat.py --key mytoken "my project"
"""

import argparse
import json
import os
import sys
import textwrap
import requests


def fmt_question(q: dict, qa_count: int) -> str:
    lines = [f"\n  Q{qa_count + 1}: {q['question']}"]
    if q.get("recommendation"):
        lines.append(f"      ℹ  {q['recommendation']}")
    opts = q.get("options", [])
    if opts:
        for i, opt in enumerate(opts, 1):
            lines.append(f"      [{i}] {opt}")
        lines.append("      (enter a number or type your own answer)")
    return "\n".join(lines)


def resolve_choice(raw: str, options: list) -> str:
    """If user typed a number and there are options, resolve it to the option text."""
    if options and raw.strip().isdigit():
        idx = int(raw.strip()) - 1
        if 0 <= idx < len(options):
            return options[idx]
    return raw


def run(base_url: str, api_key: str, description: str):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # ── Start session ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  🛠  Project Assistant")
    print(f"{'═'*60}")
    print(f"  Description: {description}")
    print(f"{'─'*60}")

    try:
        r = requests.post(f"{base_url}/project/start",
                          headers=headers,
                          json={"description": description},
                          timeout=60)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"\n  ✖  Cannot connect to {base_url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n  ✖  Server error: {e}\n  {r.text}")
        sys.exit(1)

    session = r.json()

    # ── Q&A loop ───────────────────────────────────────────────────────────────
    while session.get("state") == "qa":
        q = session.get("question", "")
        if not q:
            print("  ⚠  Empty question received — finalising...")
            break

        print(fmt_question(session, session.get("qa_count", 0)))

        try:
            raw = input("\n  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  (interrupted)")
            sys.exit(0)

        answer = resolve_choice(raw, session.get("options", []))

        try:
            r = requests.post(f"{base_url}/project/respond",
                              headers=headers,
                              json={"session_id": session["session_id"], "answer": answer},
                              timeout=60)
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"\n  ✖  Server error: {e}\n  {r.text}")
            sys.exit(1)

        session = r.json()

    # ── Done ───────────────────────────────────────────────────────────────────
    if session.get("state") != "done":
        print(f"\n  ⚠  Unexpected final state: {session.get('state')}")
        print(json.dumps(session, indent=2))
        sys.exit(1)

    print(f"\n{'═'*60}")
    print("  ✅  Project brief generated")
    print(f"{'═'*60}\n")

    md = session.get("result_markdown", "")
    # Indent markdown for terminal readability
    for line in md.splitlines():
        print(f"  {line}")

    req = session.get("requirements", {})
    cats = req.get("component_categories", [])
    if cats:
        print(f"\n{'─'*60}")
        print("  📦  Component categories to source:")
        for c in cats:
            print(f"      • {c}")

    # Optionally save markdown to file
    out_file = f"project_{session['session_id']}.md"
    with open(out_file, "w") as f:
        f.write(md)
    print(f"\n  💾  Saved to {out_file}")
    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive project Q&A client")
    parser.add_argument("description", nargs="?", help="Project description")
    parser.add_argument("--url", default=os.getenv("SWARM_SERVER", "http://localhost:5001"),
                        help="API base URL (default: http://localhost:5001)")
    parser.add_argument("--key", default=os.getenv("SWARM_API_KEY", ""),
                        help="API key (or set SWARM_API_KEY env var)")
    args = parser.parse_args()

    description = args.description
    if not description:
        try:
            description = input("  Describe your project: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
    if not description:
        print("No description provided.")
        sys.exit(1)

    run(args.url.rstrip("/"), args.key, description)


if __name__ == "__main__":
    main()
