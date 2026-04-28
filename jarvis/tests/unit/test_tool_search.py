"""Group-chat MEMORY filter for memory_search_tool (BUILD_SPEC §21.3)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.index import ALL_FILE_KINDS, get_connection, init_schema
from jarvis.memory.tool_search import _apply_group_filter, memory_search_tool
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


@pytest.fixture
def conn(tmp_path: Path):
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    c = get_connection(paths.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# _apply_group_filter — pure function, the load-bearing rule.
# ---------------------------------------------------------------------------


def test_group_filter_none_becomes_all_minus_memory():
    out = _apply_group_filter(None)
    assert "memory" not in out
    assert set(out) == set(ALL_FILE_KINDS) - {"memory"}


def test_group_filter_empty_list_stays_empty():
    # Explicit empty list: nothing to keep, nothing to drop.
    assert _apply_group_filter([]) == []


def test_group_filter_strips_memory():
    assert _apply_group_filter(["memory"]) == []


def test_group_filter_strips_memory_keeps_others():
    assert _apply_group_filter(["memory", "user"]) == ["user"]


def test_group_filter_no_op_when_already_safe():
    assert _apply_group_filter(["user", "soul"]) == ["user", "soul"]


# ---------------------------------------------------------------------------
# memory_search_tool wraps memory_search, applies filter, serializes.
# ---------------------------------------------------------------------------


def test_dm_passes_filter_through_unchanged(conn):
    captured: dict = {}

    def fake_memory_search(_conn, _query, *, embedder, options):
        captured["file_kinds"] = options.file_kinds
        return []

    with patch("jarvis.memory.tool_search.memory_search", side_effect=fake_memory_search):
        memory_search_tool(
            query="x", k=3, file_kinds=["memory", "user"],
            conn=conn, embedder=None, channel_kind="dm",
        )
    assert captured["file_kinds"] == ["memory", "user"]


def test_group_strips_memory_from_explicit_kinds(conn):
    captured: dict = {}

    def fake_memory_search(_conn, _query, *, embedder, options):
        captured["file_kinds"] = options.file_kinds
        return []

    with patch("jarvis.memory.tool_search.memory_search", side_effect=fake_memory_search):
        memory_search_tool(
            query="x", k=3, file_kinds=["memory", "user"],
            conn=conn, embedder=None, channel_kind="group",
        )
    assert captured["file_kinds"] == ["user"]


def test_group_with_none_kinds_becomes_all_but_memory(conn):
    captured: dict = {}

    def fake_memory_search(_conn, _query, *, embedder, options):
        captured["file_kinds"] = options.file_kinds
        return []

    with patch("jarvis.memory.tool_search.memory_search", side_effect=fake_memory_search):
        memory_search_tool(
            query="x", k=3, file_kinds=None,
            conn=conn, embedder=None, channel_kind="group",
        )
    assert captured["file_kinds"] is not None
    assert "memory" not in captured["file_kinds"]
    assert set(captured["file_kinds"]) == set(ALL_FILE_KINDS) - {"memory"}


def test_serializes_to_minimal_dicts(conn):
    from jarvis.memory.search import SearchResult

    fake_results = [
        SearchResult(
            chunk_id=1, file_path="MEMORY.md", content="hello",
            start_line=1, end_line=1, heading_path="x",
            score=0.99, score_components={"bm25": 1.2, "fused": 0.99},
        ),
    ]
    with patch("jarvis.memory.tool_search.memory_search", return_value=fake_results):
        out = memory_search_tool(
            query="hi", conn=conn, embedder=None, channel_kind="dm",
        )
    assert out == [{
        "chunk_id": 1, "file_path": "MEMORY.md", "content": "hello",
        "start_line": 1, "end_line": 1, "heading_path": "x", "score": 0.99,
    }]
