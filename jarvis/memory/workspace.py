"""Workspace path resolution and bootstrap.

``WorkspacePaths`` is the single source of truth for every well-known file or
directory under ``workspace/``. ``bootstrap_workspace()`` ensures that every
expected directory exists and that every well-known Markdown file is present
(starter templates, idempotent — never overwrites existing files).

See BUILD_SPEC §3.1 for the canonical layout and starter-template wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from jarvis.config import JarvisConfig
from jarvis.memory.files import write_markdown_atomic

__all__ = ["WorkspacePaths", "bootstrap_workspace"]


# ---------------------------------------------------------------------------
# Starter templates — short, follow §3.1 verbatim where the spec gives wording.
# ---------------------------------------------------------------------------

_MEMORY_TEMPLATE = """\
# MEMORY.md

Curated long-term facts. Loaded at start of every DM conversation.

This file is the source of truth for stable facts about the user, their
projects, and their preferences. Entries are promoted here only after the
Dreaming pipeline gates them (see BUILD_SPEC §9). Never edit by hand unless
you mean it — Jarvis will read this verbatim into the system prompt.
"""

_USER_TEMPLATE = """\
# USER.md

User-modeling facts (name, locations, work, preferences). Always loaded.

This is the only memory file shared into group conversations, so keep it
free of anything you wouldn't say in front of a third party.
"""

_SOUL_TEMPLATE = """\
# SOUL.md

Personality, communication style, voice. Injected into the system prompt.

Describes how Jarvis talks: tone, register, defaults for terseness vs.
elaboration, humor preferences, formatting norms.
"""

_AGENTS_TEMPLATE = """\
# AGENTS.md

Rules for delegation (when to use CMD vs Swarm vs answer directly).

Lists the available specialist agents and the heuristics for routing turns.
The router consults this file (rendered into the prompt) to decide whether to
answer directly or emit a `delegate` tool call.
"""

_TOOLS_TEMPLATE = """\
# TOOLS.md

Auto-generated — do not edit by hand.

Tool documentation is regenerated on every tool-registry change (see
BUILD_SPEC §3.1). Manual edits here will be overwritten.
"""

_HEARTBEAT_TEMPLATE = """\
# HEARTBEAT.md

Optional checklist for the autonomous heartbeat loop (proactive tasks).

Each line item is a recurring check Jarvis runs on the heartbeat interval
(see config.heartbeat). Items can be plain prose; the loop will reason about
which are due based on conversation history.
"""

_DREAMS_TEMPLATE = """\
# DREAMS.md

Human-readable diary of consolidation passes. Never a promotion source.

Each Dreaming run appends a dated section summarizing what was promoted,
demoted, or considered. This file is for the user — Jarvis does not read it
back into context.
"""


# ---------------------------------------------------------------------------
# Path resolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolves every well-known path inside the Jarvis workspace.

    Construct via ``WorkspacePaths.from_config(cfg)``; the dataclass itself is
    a pure value type with no I/O.
    """

    root: Path
    memory_md: Path
    user_md: Path
    soul_md: Path
    agents_md: Path
    tools_md: Path
    heartbeat_md: Path
    dreams_md: Path
    projects_dir: Path
    memory_dir: Path
    conversations_dir: Path
    index_dir: Path
    dreams_staging_dir: Path
    tmp_dir: Path

    def daily_log(self, day: date | None = None) -> Path:
        """Path to the daily log for ``day`` (defaults to today)."""
        d = day or date.today()
        return self.memory_dir / f"{d.isoformat()}.md"

    def project(self, slug: str) -> Path:
        """Path to ``projects/<slug>.md``."""
        return self.projects_dir / f"{slug}.md"

    @classmethod
    def from_config(cls, cfg: JarvisConfig) -> WorkspacePaths:
        root = cfg.paths.workspace
        # Cheap defense against config bugs: validators expand ~ and $VARS, but
        # a malformed YAML could still give us a relative path. Catch it early.
        assert root.is_absolute(), f"workspace must be absolute, got {root!r}"

        return cls(
            root=root,
            memory_md=root / "MEMORY.md",
            user_md=root / "USER.md",
            soul_md=root / "SOUL.md",
            agents_md=root / "AGENTS.md",
            tools_md=root / "TOOLS.md",
            heartbeat_md=root / "HEARTBEAT.md",
            dreams_md=root / "DREAMS.md",
            projects_dir=root / "projects",
            memory_dir=root / "memory",
            conversations_dir=root / ".conversations",
            index_dir=root / ".index",
            dreams_staging_dir=root / "memory" / ".dreams",
            tmp_dir=root / ".tmp",
        )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


_FILE_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("memory_md", _MEMORY_TEMPLATE),
    ("user_md", _USER_TEMPLATE),
    ("soul_md", _SOUL_TEMPLATE),
    ("agents_md", _AGENTS_TEMPLATE),
    ("tools_md", _TOOLS_TEMPLATE),
    ("heartbeat_md", _HEARTBEAT_TEMPLATE),
    ("dreams_md", _DREAMS_TEMPLATE),
)


def bootstrap_workspace(paths: WorkspacePaths) -> None:
    """Create every workspace directory + starter Markdown files (idempotent).

    Behavior:
        * ``mkdir(parents=True, exist_ok=True)`` for every directory, including
          ``.tmp/``.
        * Writes ``.tmp/.gitignore`` (``*\\n!.gitignore\\n``) so atomic-write
          staging files never get committed.
        * For each well-known ``.md``, if missing, write a starter template.
        * Today's daily log gets ``# YYYY-MM-DD`` if missing — the only place
          daily logs are auto-created. After bootstrap, ``append_to_daily_log``
          is strict.
        * Existing files are NEVER overwritten (running twice is a no-op).
    """
    # Directories first.
    for d in (
        paths.root,
        paths.projects_dir,
        paths.memory_dir,
        paths.conversations_dir,
        paths.index_dir,
        paths.dreams_staging_dir,
        paths.tmp_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)

    # .tmp/.gitignore — never commit atomic-write staging files.
    tmp_gitignore = paths.tmp_dir / ".gitignore"
    if not tmp_gitignore.exists():
        write_markdown_atomic(tmp_gitignore, "*\n!.gitignore\n", tmp_dir=paths.tmp_dir)

    # Well-known top-level Markdown files.
    for attr, template in _FILE_TEMPLATES:
        target: Path = getattr(paths, attr)
        if not target.exists():
            write_markdown_atomic(target, template, tmp_dir=paths.tmp_dir)

    # Today's daily log — the only auto-created daily log.
    today_log = paths.daily_log()
    if not today_log.exists():
        write_markdown_atomic(
            today_log,
            f"# {date.today().isoformat()}\n",
            tmp_dir=paths.tmp_dir,
        )
