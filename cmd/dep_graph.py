#!/usr/bin/env python3
"""
dep_graph.py — AST-based Python import dependency graph.

Used by the reconciler's active reverify: when a file changes, we enqueue its
dependents for re-check on the next reconciler gate.

Failures are silent — bad syntax in one file can't break the whole check.
"""

from __future__ import annotations

import ast
import os
from typing import Dict, List, Set


def _module_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def build_graph(files: List[str]) -> Dict[str, List[str]]:
    """Build {file_path: [imported_module_name, ...]} restricted to modules in the set.

    Imports that don't resolve to any file in the input list are filtered out — this
    keeps the graph focused on the project's own files.
    """
    files = [f for f in files if f.endswith(".py") and os.path.exists(f)]
    mod_to_path: Dict[str, str] = {_module_name(f): f for f in files}
    graph: Dict[str, List[str]] = {}

    for f in files:
        imports: Set[str] = set()
        try:
            with open(f, "r") as src:
                tree = ast.parse(src.read(), filename=f)
        except Exception:
            graph[f] = []
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    head = n.name.split(".")[0]
                    if head in mod_to_path and mod_to_path[head] != f:
                        imports.add(mod_to_path[head])
            elif isinstance(node, ast.ImportFrom) and node.module:
                head = node.module.split(".")[0]
                if head in mod_to_path and mod_to_path[head] != f:
                    imports.add(mod_to_path[head])
        graph[f] = sorted(imports)
    return graph


def reverse(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """{file: [files that depend on it]}."""
    rev: Dict[str, List[str]] = {f: [] for f in graph}
    for f, deps in graph.items():
        for d in deps:
            rev.setdefault(d, []).append(f)
    return {k: sorted(v) for k, v in rev.items()}


def files_affected_by(changed: Set[str], graph: Dict[str, List[str]]) -> Set[str]:
    """Transitive closure of dependents. Given a set of changed files, return every
    file that (transitively) imports any of them.

    Does NOT include the changed files themselves.
    """
    rev = reverse(graph)
    seen: Set[str] = set()
    frontier: Set[str] = set(changed)
    while frontier:
        nxt: Set[str] = set()
        for f in frontier:
            for dep in rev.get(f, []):
                if dep in seen or dep in changed:
                    continue
                seen.add(dep)
                nxt.add(dep)
        frontier = nxt
    return seen
