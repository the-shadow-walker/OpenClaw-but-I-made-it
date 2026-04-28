"""``memory_get`` LLM tool — read precise line range from a workspace .md.

Path safety: the tool refuses absolute inputs and any relative path that
resolves outside the workspace root. Resolved paths are checked with
``Path.is_relative_to(workspace_root.resolve())`` after symlink resolution
so a ``..`` traversal or a symlink escape both fail closed.
"""

from __future__ import annotations

from pathlib import Path

from jarvis.memory.files import read_lines

__all__ = ["memory_get_tool"]


def _safe_resolve(file_path: str, workspace_root: Path) -> Path:
    p = Path(file_path)
    if p.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {file_path!r}")
    resolved = (workspace_root / p).resolve()
    root_resolved = workspace_root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"path escapes workspace root: {file_path!r} → {resolved}"
        )
    return resolved


def memory_get_tool(
    *,
    file_path: str,
    start_line: int,
    end_line: int,
    workspace_root: Path,
) -> dict:
    """Return the lines [start, end] (1-indexed, inclusive) of a workspace file."""
    target = _safe_resolve(file_path, workspace_root)
    if not target.exists():
        raise FileNotFoundError(f"file not found: {file_path}")
    if not target.is_file():
        raise IsADirectoryError(f"not a file: {file_path}")

    content = read_lines(target, start_line, end_line)
    return {
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
    }
