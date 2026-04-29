"""``jarvis-chat`` — Mac-side terminal client for the daemon.

Talks to the daemon's ``/api/session`` + ``/api/chat`` endpoints over HTTPS
(via Cloudflare Access) or directly over HTTP for local dev. Renders
assistant turns as Markdown via ``rich``; tool calls / results show as
single-line status indicators.

Slash commands handled client-side before posting:
  * ``/new`` — POST /api/session again, replace the active ``conv_id``.
  * ``/onboard`` — flip on the get-to-know-you mode. Sets
    ``active_project_slug=onboarding`` for subsequent turns and primes
    Jarvis with a kickoff message. ``/onboard off`` clears it.
  * ``/quit`` — exit.

Any tool argument printed in the trace is ``repr()``-truncated to keep
single events under 120 chars total — same discipline that future Telegram
/ Slack adapters (P14) will need.

Install: ``pip install jarvis[client]`` (pulls in ``rich``). The daemon
install on arch01 doesn't need ``rich``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

import httpx
from rich.console import Console
from rich.markdown import Markdown

__all__ = ["main"]


def _truncate(value: object, n: int = 80) -> str:
    """Stringify + cap. Strings render bare; everything else via ``repr``."""
    s = value if isinstance(value, str) else repr(value)
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _render(console: Console, evt: dict) -> None:
    """Translate one stream event into a console line."""
    kind = evt.get("type")
    if kind == "delta":
        text = evt.get("text") or ""
        if text:
            console.print(Markdown(text))
    elif kind == "tool_call":
        args = evt.get("arguments") or {}
        args_str = ", ".join(f"{k}={_truncate(v)}" for k, v in args.items())
        line = f"→ {evt.get('name')}({args_str})"
        if len(line) > 120:
            line = line[:119] + "…"
        console.print(f"[dim]{line}[/dim]")
    elif kind == "tool_result":
        name = evt.get("name", "?")
        if evt.get("error"):
            console.print(f"[red]← {name}: error: {evt['error']}[/red]")
        else:
            result = evt.get("result")
            if isinstance(result, list):
                summary = f"{len(result)} results"
            elif isinstance(result, dict):
                summary = "ok"
            else:
                summary = "ok"
            console.print(f"[dim]← {name}: {summary}[/dim]")
    elif kind == "system_prompt":
        console.print(f"[dim](system_prompt: {evt.get('size', 0)} bytes)[/dim]")
    elif kind == "error":
        console.print(f"[red]error: {evt.get('message', '?')}[/red]")
    elif kind == "done":
        reason = evt.get("stop_reason")
        if reason and reason != "stop":
            console.print(f"[dim](done: {reason})[/dim]")
    # Unknown event types: silently ignore so future server-side additions
    # don't break old clients.


def _post_session(client: httpx.Client, channel_kind: str, channel_id: str) -> str:
    resp = client.post(
        "/api/session",
        json={"channel_kind": channel_kind, "channel_id": channel_id},
    )
    resp.raise_for_status()
    return resp.json()["conv_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis-chat")
    parser.add_argument(
        "--base",
        default=os.environ.get("JARVIS_BASE_URL", "https://jarvis.atomos.network"),
        help="Daemon base URL (env: JARVIS_BASE_URL).",
    )
    parser.add_argument("--channel-kind", default="cli")
    parser.add_argument(
        "--channel-id",
        default=f"cli-{socket.gethostname()}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout for chat turns (seconds).",
    )
    args = parser.parse_args(argv)

    console = Console()
    try:
        cli = httpx.Client(base_url=args.base, timeout=args.timeout)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]failed to construct httpx client: {e}[/red]")
        return 2

    with cli:
        try:
            conv_id = _post_session(cli, args.channel_kind, args.channel_id)
        except httpx.HTTPError as e:
            console.print(f"[red]session failed: {e}[/red]")
            return 1
        console.print(f"[dim]conv_id={conv_id}[/dim]")
        console.print(
            "[dim]/new = new conversation, /onboard [off] = "
            "get-to-know-you mode, /quit = exit[/dim]"
        )

        active_project: str | None = None

        while True:
            try:
                user = console.input("[bold cyan]> [/bold cyan]")
            except (KeyboardInterrupt, EOFError):
                console.print()
                return 0

            text = user.strip()
            if not text:
                continue
            if text == "/quit":
                return 0
            if text == "/new":
                try:
                    conv_id = _post_session(cli, args.channel_kind, args.channel_id)
                except httpx.HTTPError as e:
                    console.print(f"[red]session failed: {e}[/red]")
                    continue
                console.print(f"[dim]conv_id={conv_id}[/dim]")
                active_project = None
                continue
            if text in ("/onboard off", "/onboard stop"):
                active_project = None
                console.print("[dim]onboarding mode off[/dim]")
                continue
            if text == "/onboard":
                active_project = "onboarding"
                console.print("[dim]onboarding mode on — projects/onboarding.md is now in the prompt[/dim]")
                # Replace the user input with a priming message so Jarvis
                # opens the conversation rather than waiting for a question.
                text = (
                    "Begin the onboarding pass. Read USER.md to see what "
                    "you already know about me, then pick the section "
                    "with the most gaps and ask me 2-3 questions from "
                    "projects/onboarding.md. Save each answer with "
                    "user_profile_append. Stay conversational — this is "
                    "a chat, not a survey."
                )
                # Fall through to the normal POST path with text replaced.

            payload = {
                "conv_id": conv_id,
                "text": text,
                "channel_kind": args.channel_kind,
                "channel_id": args.channel_id,
            }
            if active_project:
                payload["active_project_slug"] = active_project
            try:
                with cli.stream("POST", "/api/chat", json=payload) as resp:
                    if resp.status_code != 200:
                        body = resp.read().decode("utf-8", errors="replace")
                        console.print(
                            f"[red]chat HTTP {resp.status_code}: {body}[/red]"
                        )
                        continue
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            evt = json.loads(line)
                        except json.JSONDecodeError:
                            console.print(f"[red]bad NDJSON line: {line!r}[/red]")
                            continue
                        _render(console, evt)
            except httpx.HTTPError as e:
                console.print(f"[red]chat failed: {e}[/red]")
                continue


if __name__ == "__main__":
    sys.exit(main())
