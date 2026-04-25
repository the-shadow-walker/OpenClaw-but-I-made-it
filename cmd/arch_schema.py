#!/usr/bin/env python3
"""
arch_schema.py — machine-readable architecture contract.

The planner emits DOCS/ARCH.json at phase 0. All downstream roles reconcile
against this contract. Pure stdlib; no LLM calls.

Schema (all top-level keys optional — absence is permitted, malformed shape is not):
  {
    "routes": [{"method": str, "path": str, "handler": str,
                "request_schema": str|dict|None,
                "response_schema": str|dict|None}],
    "models": [{"name": str, "fields": [...], "relationships": [...]}],
    "ports":  [{"service": str, "port": int}],
    "files":  [{"path": str, "purpose": str}],
    "dependencies": [str]
  }
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

TOP_LEVEL_KEYS = ("routes", "models", "ports", "files", "dependencies")
VALID_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}


def load_arch(path: str) -> Dict[str, Any]:
    """Load an ARCH.json file. Raises FileNotFoundError / json.JSONDecodeError on failure."""
    p = os.path.expanduser(path)
    with open(p) as f:
        return json.load(f)


def validate_arch(data: Any) -> Tuple[bool, List[str]]:
    """Return (ok, errors). ok=False if any REQUIRED structural constraint fails.
    Warnings (duplicate paths, unknown methods) are included in errors but ok stays True
    unless caller inspects the prefix — keep it simple: warnings prefixed "warning:".
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return False, ["root must be a JSON object"]

    # Every declared top-level key must be the correct type if present.
    if "routes" in data and not isinstance(data["routes"], list):
        errors.append("routes must be a list")
    if "models" in data and not isinstance(data["models"], list):
        errors.append("models must be a list")
    if "ports" in data and not isinstance(data["ports"], list):
        errors.append("ports must be a list")
    if "files" in data and not isinstance(data["files"], list):
        errors.append("files must be a list")
    if "dependencies" in data and not isinstance(data["dependencies"], list):
        errors.append("dependencies must be a list")

    # Per-route validation
    seen_route_keys: set = set()
    for i, r in enumerate(data.get("routes") or []):
        if not isinstance(r, dict):
            errors.append(f"routes[{i}] must be an object")
            continue
        method = (r.get("method") or "").upper()
        path = r.get("path") or ""
        if not method:
            errors.append(f"routes[{i}] missing method")
        elif method not in VALID_HTTP_METHODS:
            errors.append(f"warning: routes[{i}] unknown method {method!r}")
        if not path:
            errors.append(f"routes[{i}] missing path")
        key = (method, path)
        if key in seen_route_keys:
            errors.append(f"warning: routes[{i}] duplicate route {method} {path}")
        seen_route_keys.add(key)

    # Per-model validation
    seen_model_names: set = set()
    for i, m in enumerate(data.get("models") or []):
        if not isinstance(m, dict):
            errors.append(f"models[{i}] must be an object")
            continue
        name = m.get("name") or ""
        if not name:
            errors.append(f"models[{i}] missing name")
        if name in seen_model_names:
            errors.append(f"warning: models[{i}] duplicate name {name!r}")
        seen_model_names.add(name)

    # Per-port validation
    for i, p in enumerate(data.get("ports") or []):
        if not isinstance(p, dict):
            errors.append(f"ports[{i}] must be an object")
            continue
        port = p.get("port")
        if not isinstance(port, int):
            errors.append(f"ports[{i}] port must be int")

    hard_errors = [e for e in errors if not e.startswith("warning:")]
    return (len(hard_errors) == 0), errors


def extract_summary(data: Dict[str, Any], max_chars: int = 400) -> str:
    """Render a ≤max_chars compact pin-friendly summary of the contract."""
    if not isinstance(data, dict):
        return "(invalid ARCH)"

    parts: List[str] = []
    routes = data.get("routes") or []
    models = data.get("models") or []
    ports = data.get("ports") or []

    if routes:
        rbits = []
        for r in routes[:6]:
            m = (r.get("method") or "").upper()
            p = r.get("path") or ""
            rbits.append(f"{m} {p}")
        more = f" (+{len(routes) - 6} more)" if len(routes) > 6 else ""
        parts.append("Routes: " + ", ".join(rbits) + more)

    if models:
        mbits = [m.get("name", "?") for m in models[:6]]
        more = f" (+{len(models) - 6} more)" if len(models) > 6 else ""
        parts.append("Models: " + ", ".join(mbits) + more)

    if ports:
        pbits = [f"{p.get('service', '?')}:{p.get('port', '?')}" for p in ports[:4]]
        parts.append("Ports: " + ", ".join(pbits))

    out = " | ".join(parts) if parts else "(empty ARCH)"
    return out[:max_chars]
