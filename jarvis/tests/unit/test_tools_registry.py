"""ToolRegistry shape + dispatch + delegate stub behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.core.tools import ToolRegistry, ToolSpec, build_default_registry
from jarvis.memory.index import get_connection, init_schema
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


@pytest.fixture
def paths(tmp_path: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    p = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(p)
    return p


@pytest.fixture
def conn(paths: WorkspacePaths):
    c = get_connection(paths.index_dir / "memory.sqlite")
    init_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# ToolRegistry primitives
# ---------------------------------------------------------------------------


def test_register_then_get():
    r = ToolRegistry()
    r.register(ToolSpec(
        name="foo", description="d", parameters={"type": "object"},
        handler=lambda **_: "ok",
    ))
    assert r.get("foo").name == "foo"


def test_register_duplicate_raises():
    r = ToolRegistry()
    r.register(ToolSpec(name="foo", description="d", parameters={}, handler=lambda **_: 1))
    with pytest.raises(ValueError, match="already registered"):
        r.register(ToolSpec(name="foo", description="d", parameters={}, handler=lambda **_: 2))


def test_get_unknown_raises():
    r = ToolRegistry()
    with pytest.raises(KeyError):
        r.get("nope")


def test_execute_unpacks_kwargs():
    r = ToolRegistry()
    r.register(ToolSpec(
        name="add", description="", parameters={},
        handler=lambda a, b: a + b,
    ))
    assert r.execute("add", {"a": 1, "b": 2}) == 3


def test_execute_unknown_raises_keyerror():
    r = ToolRegistry()
    with pytest.raises(KeyError):
        r.execute("missing", {})


def test_schemas_shape():
    r = ToolRegistry()
    r.register(ToolSpec(
        name="foo", description="d",
        parameters={"type": "object", "properties": {}},
        handler=lambda **_: None,
    ))
    schemas = r.schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    assert s["function"]["name"] == "foo"
    assert s["function"]["description"] == "d"
    assert s["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# build_default_registry
# ---------------------------------------------------------------------------


def test_default_registry_has_all_four_tools(conn, paths):
    r = build_default_registry(conn=conn, embedder=None, paths=paths, channel_kind="dm")
    assert set(r.names()) == {
        "memory_search", "memory_get", "memory_write",
        "user_profile_append", "delegate",
    }


def test_delegate_stub_returns_error(conn, paths):
    r = build_default_registry(conn=conn, embedder=None, paths=paths, channel_kind="dm")
    out = r.execute("delegate", {"target": "cmd:quick", "task": "hi"})
    assert out["error"].startswith("delegate not implemented")
    assert out["target"] == "cmd:quick"


def test_default_memory_search_routes_through_filter_in_group(conn, paths):
    """End-to-end: build_default_registry with channel_kind='group' should
    give us a memory_search whose handler enforces the group MEMORY filter.
    """
    r = build_default_registry(conn=conn, embedder=None, paths=paths, channel_kind="group")
    # No content yet, so the result is an empty list — but the call must not
    # raise (proves the wiring is correct end-to-end).
    out = r.execute("memory_search", {"query": "anything", "k": 3, "file_kinds": ["memory"]})
    assert out == []
