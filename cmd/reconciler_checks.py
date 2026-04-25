#!/usr/bin/env python3
"""
reconciler_checks.py — deterministic static + semantic checks used by the
reconciler role. No LLM calls; pure AST + regex.

Two flavours of check:

  Syntactic
    find_undefined_symbols   — names referenced but never defined/imported
    find_import_cycles       — strongly-connected import components
    find_fk_ambiguity        — SQLAlchemy relationship() with multi-FK and no foreign_keys=

  Semantic (against ARCH.json)
    compare_routes_to_arch       — method/path/handler diff
    compare_handler_signatures   — handler param names/types vs arch.request_schema
    compare_pydantic_models_to_arch — Pydantic fields vs arch.models[].fields
    detect_schema_naming_drift   — imports symbol not in arch but a similar one exists

Bidirectional classifier
    classify_violations(findings) → {"patch_code": [...], "update_arch": [...],
                                     "report_only": [...]}
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Cache: file_path -> (mtime, parsed_ast | None)
_AST_CACHE: Dict[str, Tuple[float, Optional[ast.Module]]] = {}

# Configurable via env
ARCH_UPDATE_THRESHOLD = int(os.getenv("ARCH_UPDATE_THRESHOLD", "3"))


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return None


def _parse(path: str) -> Optional[ast.Module]:
    """Parse and cache an AST keyed on mtime. Returns None on syntax error."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _AST_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    src = _read(path)
    if src is None:
        _AST_CACHE[path] = (mtime, None)
        return None
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError:
        tree = None
    _AST_CACHE[path] = (mtime, tree)
    return tree


def _py_files_only(files: List[str]) -> List[str]:
    return [f for f in files if f.endswith(".py") and os.path.exists(f)]


# ---------------------------------------------------------------------------
# Syntactic checks
# ---------------------------------------------------------------------------

_PY_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(vars(__builtins__))


def _collect_definitions(tree: ast.Module) -> Set[str]:
    """Names bound at module scope — defs, classes, imports, assignments."""
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for n in node.names:
                names.add((n.asname or n.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                names.add(n.asname or n.name)
    return names


def find_undefined_symbols(files: List[str]) -> List[Dict]:
    """Flag Name references at module scope that are not defined, imported, or builtin.
    Intentionally conservative — skips names inside function bodies (local-scope is hard).
    """
    results: List[Dict] = []
    for path in _py_files_only(files):
        tree = _parse(path)
        if tree is None:
            continue
        defined = _collect_definitions(tree) | _PY_BUILTINS
        # Walk only module-level expressions/decorators/defaults
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name):
                continue
            if isinstance(node.ctx, ast.Store):
                continue
            if node.id in defined:
                continue
            # Skip names that appear anywhere as a function/class arg or local binding.
            # Cheap heuristic: only flag names referenced at module body depth.
            # We approximate by checking if the name appears as a def/arg anywhere.
            results.append({
                "file": path,
                "name": node.id,
                "line": getattr(node, "lineno", 0),
            })
    # Collapse: an undefined name may recur many times; keep first occurrence per (file,name)
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Dict] = []
    for r in results:
        key = (r["file"], r["name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    # Heavy filter: only return names that look like symbols we'd expect to be top-level
    # (capitalized class-like or snake_case fn-like). Leaves out e.g. `self`, `cls`.
    return [r for r in deduped if r["name"] not in {"self", "cls", "args", "kwargs"}]


def find_import_cycles(files: List[str]) -> List[List[str]]:
    """Return cycles in the import graph restricted to the given files."""
    files = _py_files_only(files)
    if not files:
        return []
    # Map module name → path
    mod_to_path: Dict[str, str] = {}
    for f in files:
        mod = os.path.splitext(os.path.basename(f))[0]
        mod_to_path[mod] = f

    # Build edges
    graph: Dict[str, Set[str]] = {m: set() for m in mod_to_path}
    for mod, path in mod_to_path.items():
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    head = n.name.split(".")[0]
                    if head in mod_to_path and head != mod:
                        graph[mod].add(head)
            elif isinstance(node, ast.ImportFrom) and node.module:
                head = node.module.split(".")[0]
                if head in mod_to_path and head != mod:
                    graph[mod].add(head)

    # Tarjan-ish cycle detection (iterative DFS, sufficient for small graphs)
    cycles: List[List[str]] = []
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}

    def strong(v: str):
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)
        for w in graph.get(v, ()):
            if w not in indices:
                strong(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            component: List[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    for v in list(graph):
        if v not in indices:
            try:
                strong(v)
            except RecursionError:
                return cycles
    return cycles


_FK_RE = re.compile(r'ForeignKey\s*\(\s*["\']([^"\']+)["\']', re.IGNORECASE)
_REL_RE = re.compile(
    r'(\w+)\s*[:=]\s*relationship\s*\(\s*["\']?(\w+)["\']?'
    r'(?P<tail>[^)]*)\)',
    re.IGNORECASE | re.DOTALL,
)


def find_fk_ambiguity(model_files: List[str]) -> List[Dict]:
    """Detect SQLAlchemy models with multiple FKs to the same table and a
    relationship() without foreign_keys= disambiguation.
    """
    out: List[Dict] = []
    for path in _py_files_only(model_files):
        src = _read(path) or ""
        # Group FK targets
        fk_targets = _FK_RE.findall(src)
        dup_targets = {t.split(".")[0] for t in fk_targets if fk_targets.count(t) >= 1
                       and sum(1 for x in fk_targets if x.split(".")[0] == t.split(".")[0]) > 1}
        if not dup_targets:
            continue
        for m in _REL_RE.finditer(src):
            attr, target = m.group(1), m.group(2)
            tail = m.group("tail") or ""
            # Relationship target matches an ambiguous-FK table class (case-insensitive)
            if target.lower() in {t.lower() for t in dup_targets}:
                if "foreign_keys" not in tail:
                    out.append({
                        "file": path,
                        "relationship": attr,
                        "target": target,
                        "fix": "add foreign_keys=[...] to disambiguate",
                    })
    return out


# ---------------------------------------------------------------------------
# Semantic checks — against ARCH.json
# ---------------------------------------------------------------------------

# Regex for detecting route decorators: @app.post("/path") / @router.get("/x")
_ROUTE_DEC_RE = re.compile(
    r'@(?P<obj>\w+)\.(?P<method>get|post|put|patch|delete|head|options)\s*\(\s*'
    r'["\'](?P<path>/[^"\']*)["\']',
    re.IGNORECASE,
)


def _extract_routes_from_source(files: List[str]) -> List[Dict]:
    """Scan .py files for FastAPI/Flask-style route decorators."""
    found: List[Dict] = []
    for path in _py_files_only(files):
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                # Flatten decorator source text safely via ast.unparse where available
                try:
                    dec_src = ast.unparse(dec)
                except Exception:
                    continue
                m = _ROUTE_DEC_RE.search("@" + dec_src)
                if not m:
                    continue
                found.append({
                    "file": path,
                    "handler": node.name,
                    "method": m.group("method").upper(),
                    "path": m.group("path"),
                    "params": [a.arg for a in node.args.args],
                    "annotations": {
                        a.arg: (ast.unparse(a.annotation) if a.annotation else None)
                        for a in node.args.args
                    },
                })
    return found


def compare_routes_to_arch(source_files: List[str], arch: Dict) -> Dict:
    """Diff declared vs implemented routes. Returns dict with missing / extra / method_drift."""
    declared = {(r.get("method", "").upper(), r.get("path", "")): r
                for r in (arch.get("routes") or [])}
    implemented = {(r["method"], r["path"]): r for r in _extract_routes_from_source(source_files)}

    missing = [dict(r, reason="declared in ARCH but no handler found")
               for key, r in declared.items() if key not in implemented]
    extra = [dict(r, reason="handler exists but not declared in ARCH")
             for key, r in implemented.items() if key not in declared]

    # method_drift: same path but different methods
    decl_paths = {k[1]: k[0] for k in declared}
    impl_paths = {k[1]: k[0] for k in implemented}
    method_drift = []
    for p, decl_method in decl_paths.items():
        impl_method = impl_paths.get(p)
        if impl_method and impl_method != decl_method:
            method_drift.append({
                "path": p,
                "declared_method": decl_method,
                "implemented_method": impl_method,
            })

    return {"missing": missing, "extra": extra, "method_drift": method_drift}


def compare_handler_signatures(source_files: List[str], arch: Dict) -> List[Dict]:
    """For each ARCH route with a declared request_schema, diff handler params against
    the fields of the matching model (if the schema name matches a model in arch.models).
    """
    out: List[Dict] = []
    models_by_name = {m.get("name"): m for m in (arch.get("models") or []) if isinstance(m, dict)}
    implemented = {(r["method"], r["path"]): r for r in _extract_routes_from_source(source_files)}

    for r in (arch.get("routes") or []):
        key = ((r.get("method") or "").upper(), r.get("path") or "")
        impl = implemented.get(key)
        if not impl:
            continue
        req_schema = r.get("request_schema")
        if not isinstance(req_schema, str):
            continue
        model = models_by_name.get(req_schema)
        if not model:
            continue
        expected_fields = {f.get("name") for f in (model.get("fields") or []) if f.get("name")}
        if not expected_fields:
            continue
        # Handler params less the usual FastAPI-injected deps
        ignore = {"self", "request", "db", "session", "current_user", "background_tasks"}
        handler_params = set(impl.get("params", [])) - ignore
        missing = expected_fields - handler_params
        # Signature drift only flags if handler takes the schema as a model (then it's OK),
        # OR if the handler takes raw fields that don't match. Heuristic: annotations mention schema?
        annot_values = [v for v in impl.get("annotations", {}).values() if v]
        mentions_schema = any(req_schema in a for a in annot_values)
        if mentions_schema:
            continue
        if missing:
            out.append({
                "file": impl["file"],
                "handler": impl["handler"],
                "method": impl["method"],
                "path": impl["path"],
                "expected_from": req_schema,
                "missing_fields": sorted(missing),
            })
    return out


_PYDANTIC_BASES = {"BaseModel", "SQLModel"}


def _extract_pydantic_models(files: List[str]) -> Dict[str, Dict]:
    """Return {ClassName: {"fields": [{"name","type"}], "file": path}}."""
    out: Dict[str, Dict] = {}
    for path in _py_files_only(files):
        tree = _parse(path)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = []
            for b in node.bases:
                try:
                    base_names.append(ast.unparse(b).split(".")[-1])
                except Exception:
                    pass
            if not any(b in _PYDANTIC_BASES for b in base_names):
                continue
            fields = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    try:
                        type_str = ast.unparse(stmt.annotation)
                    except Exception:
                        type_str = ""
                    fields.append({"name": stmt.target.id, "type": type_str})
            out[node.name] = {"fields": fields, "file": path}
    return out


def compare_pydantic_models_to_arch(source_files: List[str], arch: Dict) -> List[Dict]:
    """For each arch.model, find matching Pydantic class and diff fields. Flag renamed
    fields via fuzzy match so we catch e.g. `email` vs `email_address`.
    """
    out: List[Dict] = []
    impl = _extract_pydantic_models(source_files)
    for m in (arch.get("models") or []):
        name = m.get("name")
        if not name:
            continue
        declared_fields = {f.get("name"): f.get("type", "") for f in (m.get("fields") or [])
                           if f.get("name")}
        if not declared_fields:
            continue
        model_impl = impl.get(name)
        if not model_impl:
            # If a class with a very similar name exists, flag as rename candidate
            similar = difflib.get_close_matches(name, list(impl.keys()), n=1, cutoff=0.7)
            if similar:
                out.append({
                    "model": name,
                    "kind": "rename_candidate",
                    "implemented_as": similar[0],
                    "file": impl[similar[0]]["file"],
                })
            continue
        impl_field_names = {f["name"] for f in model_impl["fields"]}
        missing = set(declared_fields) - impl_field_names
        extra = impl_field_names - set(declared_fields)
        # Fuzzy rename detection within this model
        renamed_candidates = []
        for miss in list(missing):
            candidates = difflib.get_close_matches(miss, list(extra), n=1, cutoff=0.7)
            if candidates:
                renamed_candidates.append({"declared": miss, "implemented": candidates[0]})
                missing.discard(miss)
                extra.discard(candidates[0])
        if missing or extra or renamed_candidates:
            out.append({
                "model": name,
                "file": model_impl["file"],
                "missing_fields": sorted(missing),
                "extra_fields": sorted(extra),
                "renamed_candidates": renamed_candidates,
            })
    return out


def detect_schema_naming_drift(source_files: List[str], arch: Dict) -> List[Dict]:
    """Catch the RegisterRequest vs UserCreate pattern: a file imports a name that
    doesn't exist in ARCH.models but a similar one does.
    """
    arch_model_names = {m.get("name") for m in (arch.get("models") or []) if m.get("name")}
    if not arch_model_names:
        return []
    imported: Dict[str, List[Tuple[str, int]]] = {}
    for path in _py_files_only(source_files):
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for n in node.names:
                    imported.setdefault(n.name, []).append((path, node.lineno))
    out: List[Dict] = []
    for name, sites in imported.items():
        if name in arch_model_names:
            continue
        # Only consider PascalCase names likely to be schemas
        if not name or not name[0].isupper():
            continue
        close = difflib.get_close_matches(name, list(arch_model_names), n=1, cutoff=0.6)
        if close:
            for file, line in sites:
                out.append({
                    "file": file,
                    "line": line,
                    "imported_as": name,
                    "likely_meant": close[0],
                })
    return out


# ---------------------------------------------------------------------------
# Aggregator + bidirectional classifier
# ---------------------------------------------------------------------------

def run_all(files: List[str], arch: Optional[Dict]) -> Dict:
    """Run the full suite. ARCH may be None or empty; semantic checks short-circuit."""
    files = _py_files_only(files)
    arch = arch or {}
    return {
        "undefined_symbols": find_undefined_symbols(files),
        "import_cycles": find_import_cycles(files),
        "fk_ambiguity": find_fk_ambiguity(files),
        "route_drift": compare_routes_to_arch(files, arch),
        "signature_drift": compare_handler_signatures(files, arch),
        "pydantic_drift": compare_pydantic_models_to_arch(files, arch),
        "naming_drift": detect_schema_naming_drift(files, arch),
        "timestamp": int(time.time()),
    }


def has_any_issue(findings: Dict) -> bool:
    rd = findings.get("route_drift") or {}
    return bool(
        findings.get("undefined_symbols")
        or findings.get("import_cycles")
        or findings.get("fk_ambiguity")
        or rd.get("missing") or rd.get("extra") or rd.get("method_drift")
        or findings.get("signature_drift")
        or findings.get("pydantic_drift")
        or findings.get("naming_drift")
    )


def classify_violations(findings: Dict) -> Dict[str, List]:
    """Decide which way the reconciler should reconcile: patch code vs update ARCH.

    Heuristic:
      - If >ARCH_UPDATE_THRESHOLD code files have routes NOT in ARCH → ARCH lags. Update ARCH.
      - If >ARCH_UPDATE_THRESHOLD ARCH entries have no matching code → code lags. Patch code.
      - Else report_only.

    Always returns three buckets (possibly empty):
      patch_code   — findings that indicate code should be fixed
      update_arch  — findings that indicate ARCH.json should be rewritten
      report_only  — ambiguous; surface as warning
    """
    rd = findings.get("route_drift") or {}
    missing = rd.get("missing") or []       # in ARCH, no code
    extra = rd.get("extra") or []           # in code, not in ARCH
    method_drift = rd.get("method_drift") or []

    patch_code: List = []
    update_arch: List = []
    report_only: List = []

    # Code-lags-ARCH: ARCH has routes that aren't implemented
    if len(missing) >= ARCH_UPDATE_THRESHOLD:
        patch_code.extend(("missing_route", r) for r in missing)
    else:
        report_only.extend(("missing_route", r) for r in missing)

    # ARCH-lags-code: code has routes not in ARCH
    if len(extra) >= ARCH_UPDATE_THRESHOLD:
        update_arch.extend(("extra_route", r) for r in extra)
    else:
        report_only.extend(("extra_route", r) for r in extra)

    # Method drift always needs code alignment (safer to keep ARCH as source of truth for methods)
    patch_code.extend(("method_drift", r) for r in method_drift)

    # Pydantic model drift — patch code (schema names are authoritative to keep)
    for r in findings.get("pydantic_drift") or []:
        if r.get("kind") == "rename_candidate":
            patch_code.append(("pydantic_rename", r))
        else:
            patch_code.append(("pydantic_fields", r))

    # Naming drift and signature drift always patch_code
    for r in findings.get("signature_drift") or []:
        patch_code.append(("signature_drift", r))
    for r in findings.get("naming_drift") or []:
        patch_code.append(("naming_drift", r))

    # Static issues always patch_code
    for r in findings.get("undefined_symbols") or []:
        patch_code.append(("undefined_symbol", r))
    for r in findings.get("fk_ambiguity") or []:
        patch_code.append(("fk_ambiguity", r))
    for r in findings.get("import_cycles") or []:
        patch_code.append(("import_cycle", r))

    return {
        "patch_code": patch_code,
        "update_arch": update_arch,
        "report_only": report_only,
    }


def format_findings_for_prompt(findings: Dict, classification: Dict, max_items: int = 12) -> str:
    """Produce a compact prompt block for the reconciler minion."""
    lines: List[str] = []
    if classification.get("patch_code"):
        lines.append("CODE CHANGES NEEDED (patch code to match ARCH):")
        for tag, item in classification["patch_code"][:max_items]:
            lines.append(f"  - [{tag}] {str(item)[:180]}")
    if classification.get("update_arch"):
        lines.append("\nARCH CHANGES NEEDED (code has evolved past contract):")
        for tag, item in classification["update_arch"][:max_items]:
            lines.append(f"  - [{tag}] {str(item)[:180]}")
    if classification.get("report_only"):
        lines.append("\nAMBIGUOUS (surface as warning, no auto-fix):")
        for tag, item in classification["report_only"][:max_items]:
            lines.append(f"  - [{tag}] {str(item)[:180]}")
    if not lines:
        return "No reconciler findings."
    return "\n".join(lines)[:3500]
