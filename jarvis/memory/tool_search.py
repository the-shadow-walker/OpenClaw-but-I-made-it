"""``memory_search`` LLM tool wrapper (BUILD_SPEC §6.5, §21.3).

Thin shim over :func:`jarvis.memory.search.memory_search` with two
LLM-shaped concerns:

  1. **Group-chat MEMORY filter** (§21.3). In ``channel_kind="group"``
     conversations the ``memory`` file_kind is forced out of the result
     set — silently — even if the model passed ``file_kinds=["memory"]``.
     MEMORY.md never leaks across multi-party rooms; the model can't
     opt out.
  2. **Result serialization** — only the columns the LLM uses are
     returned (``score_components`` is debug-only and stripped).
"""

from __future__ import annotations

from typing import Any

from jarvis.core.conversation import ChannelKind
from jarvis.memory.embeddings import EmbeddingPipeline
from jarvis.memory.index import ALL_FILE_KINDS
from jarvis.memory.search import SearchOptions, memory_search

__all__ = ["memory_search_tool"]


def _apply_group_filter(file_kinds: list[str] | None) -> list[str]:
    """Force ``"memory"`` out of file_kinds for group conversations.

    ``None`` becomes "every kind except memory" (so the search is still
    broad). An explicit list keeps everything except ``"memory"``.
    """
    if file_kinds is None:
        return [k for k in ALL_FILE_KINDS if k != "memory"]
    return [k for k in file_kinds if k != "memory"]


def memory_search_tool(
    *,
    query: str,
    k: int = 5,
    file_kinds: list[str] | None = None,
    conn,
    embedder: EmbeddingPipeline | None,
    channel_kind: ChannelKind,
) -> list[dict[str, Any]]:
    """LLM-callable wrapper around the hybrid retriever.

    Returns a list of dicts (LLM-friendly). Empty list on no results /
    empty query / sanitization stripping all tokens. Errors from the
    embedder degrade to BM25-only inside ``memory_search``; this wrapper
    never raises for normal "nothing found" outcomes.
    """
    effective_kinds: list[str] | None = file_kinds
    if channel_kind == "group":
        effective_kinds = _apply_group_filter(file_kinds)

    results = memory_search(
        conn,
        query,
        embedder=embedder,
        options=SearchOptions(k=k, file_kinds=effective_kinds),
    )

    return [
        {
            "chunk_id": r.chunk_id,
            "file_path": r.file_path,
            "content": r.content,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "heading_path": r.heading_path,
            "score": r.score,
        }
        for r in results
    ]
