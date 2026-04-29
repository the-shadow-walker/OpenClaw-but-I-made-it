"""``user_profile_append`` LLM tool — write a bullet to USER.md under a
named section heading.

Why a separate tool from ``memory_write``: USER.md is the canonical
identity file. It needs structured, section-keyed appends (not a flat
timestamped log, not a single "remember this" bullet list). The
onboarding flow (``projects/onboarding.md``) drives 40-ish questions
and pipes each answer here.

Behaviour:
  * If a ``## <section>`` heading exists, append a bullet beneath the
    *last* line under that section (above the next heading or EOF).
  * If the heading does not exist, append it at the bottom of the file
    along with the bullet.
  * Writes are atomic via ``write_markdown_atomic`` — the watcher will
    reconcile <1s later.
"""

from __future__ import annotations

import re

from jarvis.memory.files import read_markdown, write_markdown_atomic
from jarvis.memory.workspace import WorkspacePaths

__all__ = ["user_profile_append_tool"]


def user_profile_append_tool(
    *,
    section: str,
    content: str,
    paths: WorkspacePaths,
) -> dict:
    """Append a bullet under ``## section`` in USER.md.

    Args:
        section: The H2 heading to append under (case-insensitive match).
            Whitespace is trimmed; '#' prefixes are stripped.
        content: Bullet text. Leading "- " is stripped if present.
    """
    section_clean = section.strip().lstrip("#").strip()
    if not section_clean:
        raise ValueError("section must be non-empty")
    bullet = content.strip()
    if bullet.startswith("- "):
        bullet = bullet[2:]
    if not bullet:
        raise ValueError("content must be non-empty")

    existing = read_markdown(paths.user_md) if paths.user_md.exists() else ""
    new_text = _insert_bullet(existing, section_clean, bullet)
    write_markdown_atomic(paths.user_md, new_text, tmp_dir=paths.tmp_dir)
    return {"section": section_clean, "file_path": "USER.md"}


_H2_RE = re.compile(r"^##\s+(.+?)\s*$")


def _insert_bullet(text: str, section: str, bullet: str) -> str:
    """Return ``text`` with ``- bullet`` inserted under ``## section``.

    If the heading is missing, append both heading and bullet at EOF.
    Heading match is case-insensitive on the trimmed title.
    """
    lines = text.splitlines()
    target_lower = section.lower()

    # Find the index of the matching ## heading (if any).
    heading_idx: int | None = None
    for i, line in enumerate(lines):
        m = _H2_RE.match(line)
        if m and m.group(1).strip().lower() == target_lower:
            heading_idx = i
            break

    if heading_idx is None:
        # No matching heading — append a new section at EOF.
        out = text.rstrip("\n")
        if out:
            out += "\n\n"
        out += f"## {section}\n- {bullet}\n"
        return out

    # Find the end of this section: next heading line (any level), or EOF.
    end_idx = len(lines)
    for j in range(heading_idx + 1, len(lines)):
        if lines[j].lstrip().startswith("#"):
            end_idx = j
            break

    # Walk back from end_idx to skip trailing blank lines so the bullet
    # joins the section's existing list cleanly instead of after a gap.
    insert_idx = end_idx
    while insert_idx > heading_idx + 1 and not lines[insert_idx - 1].strip():
        insert_idx -= 1

    new_lines = lines[:insert_idx] + [f"- {bullet}"] + lines[insert_idx:]
    out = "\n".join(new_lines)
    if not out.endswith("\n"):
        out += "\n"
    return out
