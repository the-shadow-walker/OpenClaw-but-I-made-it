"""System prompt assembly per BUILD_SPEC §3.1 loading rules.

P1 implements whole-file loading only — no retrieval, no chunking, no search.
Each section is fenced with a ``# ===== <NAME> =====`` delimiter so the LLM
can tell sources apart.

A size guard emits a single ``logger.warning`` when the assembled prompt
exceeds ``PROMPT_SIZE_WARN_BYTES``. P3 hybrid retrieval will replace
whole-file loads.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from jarvis.memory.files import read_markdown
from jarvis.memory.workspace import WorkspacePaths

__all__ = ["ChannelKind", "PROMPT_SIZE_WARN_BYTES", "assemble_system_prompt"]

logger = logging.getLogger(__name__)

ChannelKind = Literal["dm", "group", "heartbeat", "cli"]

# ~5K tokens. P3 retrieval will replace whole-file loads with hybrid search;
# this is an early-warning trip so we notice ballooning before P5 hits a
# context limit.
PROMPT_SIZE_WARN_BYTES = 20_000


def _section(title: str, body: str) -> str:
    """Render a fenced section: header + body, ensuring trailing newline."""
    body = body.rstrip("\n")
    return f"# ===== {title} =====\n{body}\n"


def _maybe_section(title: str, path: Path) -> str | None:
    """Read ``path`` and return a fenced section, or None if it does not exist."""
    if not path.exists():
        return None
    return _section(title, read_markdown(path))


def assemble_system_prompt(
    paths: WorkspacePaths,
    channel_kind: ChannelKind,
    active_project_slug: str | None = None,
    *,
    today: date | None = None,
) -> str:
    """Build the system prompt for a conversation turn.

    Loading rules (BUILD_SPEC §3.1):
      * **dm**: USER.md + MEMORY.md + today's daily + yesterday's daily +
        SOUL.md + AGENTS.md + TOOLS.md
      * **group**: USER.md ONLY (MEMORY.md never loaded — multi-party leak guard)
      * **heartbeat**: HEARTBEAT.md + USER.md + today's daily + AGENTS.md

    ``active_project_slug`` (DM only): if provided and ``projects/<slug>.md``
    exists, append it.

    ``today`` is computed at call time (default ``date.today()``) — passing it
    is allowed for tests.
    """
    # IMPORTANT: compute today at call time, not function-definition time.
    today = today or date.today()
    yesterday = today - timedelta(days=1)

    sections: list[str] = []

    # CLI channel uses the same prompt scaffold as DM — single user, full memory
    # access. Branch comparison stays explicit so the assertion-free narrowing
    # that Literal provides isn't lost.
    if channel_kind in ("dm", "cli"):
        sections.append(_section("USER.md", read_markdown(paths.user_md)))
        sections.append(_section("MEMORY.md", read_markdown(paths.memory_md)))

        today_log = _maybe_section("daily/today", paths.daily_log(today))
        if today_log is not None:
            sections.append(today_log)
        yesterday_log = _maybe_section("daily/yesterday", paths.daily_log(yesterday))
        if yesterday_log is not None:
            sections.append(yesterday_log)

        sections.append(_section("SOUL.md", read_markdown(paths.soul_md)))
        sections.append(_section("AGENTS.md", read_markdown(paths.agents_md)))
        sections.append(_section("TOOLS.md", read_markdown(paths.tools_md)))

        if active_project_slug:
            project_path = paths.project(active_project_slug)
            project_section = _maybe_section(
                f"projects/{active_project_slug}.md", project_path
            )
            if project_section is not None:
                sections.append(project_section)

    elif channel_kind == "group":
        # USER.md ONLY — MEMORY.md never loaded in groups. Multi-party leak guard.
        sections.append(_section("USER.md", read_markdown(paths.user_md)))

    elif channel_kind == "heartbeat":
        sections.append(_section("HEARTBEAT.md", read_markdown(paths.heartbeat_md)))
        sections.append(_section("USER.md", read_markdown(paths.user_md)))
        today_log = _maybe_section("daily/today", paths.daily_log(today))
        if today_log is not None:
            sections.append(today_log)
        sections.append(_section("AGENTS.md", read_markdown(paths.agents_md)))

    else:  # pragma: no cover — Literal narrows this away at type-check time.
        raise ValueError(f"unknown channel_kind: {channel_kind!r}")

    prompt = "\n".join(sections)

    size = len(prompt.encode("utf-8"))
    if size > PROMPT_SIZE_WARN_BYTES:
        logger.warning(
            "system prompt is %d bytes (threshold %d) — P3 retrieval will replace "
            "whole-file loads with hybrid search",
            size,
            PROMPT_SIZE_WARN_BYTES,
        )

    return prompt
