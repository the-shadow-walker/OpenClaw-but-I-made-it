# ARCH.json Schema

Machine-readable architecture contract emitted by the **planner** at phase 0.
Consumed by the **reconciler** and **tester** roles for drift detection.

Location: `<workspace>/DOCS/ARCH.json` (or `DOCS/ARCH.json` if no workspace).

## Top-level shape

```json
{
  "routes": [ ... ],
  "models": [ ... ],
  "ports":  [ ... ],
  "files":  [ ... ],
  "dependencies": [ ... ]
}
```

All keys optional. If a key is present, its value MUST be a list.

## routes[]

```json
{
  "method": "POST",
  "path": "/users",
  "handler": "create_user",
  "request_schema": "UserCreate",
  "response_schema": "UserRead"
}
```

| Field | Type | Notes |
|---|---|---|
| method | str | One of `GET,POST,PUT,PATCH,DELETE,HEAD,OPTIONS` |
| path | str | Starts with `/` |
| handler | str | Qualified name: `module.function` or bare function |
| request_schema | str \| object \| null | Pydantic model name or inline `{field: type}` dict |
| response_schema | str \| object \| null | Same |

Duplicate `(method, path)` pairs are flagged as warnings.

## models[]

```json
{
  "name": "User",
  "fields": [{"name": "email", "type": "str"},
             {"name": "id", "type": "int"}],
  "relationships": [{"target": "Team", "kind": "many-to-many"}]
}
```

| Field | Type | Notes |
|---|---|---|
| name | str | Unique. Must match Python class name |
| fields | list[{name,type}] | |
| relationships | list[{target,kind,foreign_keys?}] | `foreign_keys` needed when multiple FKs target same table (SQLAlchemy ambiguity) |

## ports[]

```json
{"service": "api", "port": 8000}
```

## files[]

```json
{"path": "src/auth/router.py", "purpose": "HTTP endpoints for /auth/*"}
```

## dependencies[]

```json
["fastapi>=0.115", "sqlalchemy>=2", "bcrypt>=5"]
```

## Validation

Loaded via `arch_schema.load_arch(path)`, validated via `arch_schema.validate_arch(data)`.
The `validate_arch` tool (available to planner, builder, reconciler) wraps both.

Warnings (prefixed `warning:`) do not fail validation.
Hard errors (missing required field, wrong type) set `ok=False`.

## Bidirectional reconciliation

`reconciler_checks.classify_violations(findings)` returns one of:

- `patch_code`: code lags contract — reconciler edits code to match ARCH
- `update_arch`: code has evolved past contract — reconciler rewrites ARCH.json
- `report_only`: insufficient evidence either way, surface as warning

Threshold: if >`ARCH_UPDATE_THRESHOLD` (default 3) files deviate in one direction,
reconciler picks that direction. See `cmd/reconciler_checks.py`.
