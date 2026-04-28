"""``memory_write`` LLM tool — append to today's daily log or MEMORY.md.

Two modes:

  * ``where="daily"`` (default) — appends a timestamped line to today's
    ``memory/<date>.md``. The Dreaming pipeline (P11+) is the *normal*
    promotion path from daily → MEMORY.md.

  * ``where="memory"`` — appends a bullet to ``MEMORY.md`` directly.
    Reserved for explicit user "remember this" requests; the model
    shouldn't auto-promote without the user asking.

The watcher picks up either change in <1s — same plumbing as P4.
"""

from __future__ import annotations

from typing import Literal

from jarvis.memory.files import (
    append_to_daily_log,
    read_markdown,
    write_markdown_atomic,
)
from jarvis.memory.workspace import WorkspacePaths

__all__ = ["memory_write_tool"]


def memory_write_tool(
    *,
    content: str,
    where: Literal["daily", "memory"] = "daily",
    tags: list[str] | None = None,
    paths: WorkspacePaths,
) -> dict:
    """Persist ``content`` to either today's daily log or MEMORY.md."""
    if where == "daily":
        # ``append_to_daily_log`` is strict — raises if today's log is missing
        # (bootstrap creates it on first run; the daily-rollover cron
        # creates each subsequent day's file).
        target = paths.daily_log()
        append_to_daily_log(target, content, tags=tags)
        return {"where": "daily", "file_path": str(target.relative_to(paths.root))}

    if where == "memory":
        existing = read_markdown(paths.memory_md) if paths.memory_md.exists() else ""
        # Build the bullet — strip leading "- " if the LLM already added one.
        bullet = content.strip()
        if bullet.startswith("- "):
            bullet = bullet[2:]
        new_text = existing
        if not new_text.endswith("\n"):
            new_text += "\n"
        new_text += f"- {bullet}\n"
        write_markdown_atomic(paths.memory_md, new_text, tmp_dir=paths.tmp_dir)
        return {"where": "memory", "file_path": "MEMORY.md"}

    raise ValueError(f"unknown 'where' value: {where!r}")
