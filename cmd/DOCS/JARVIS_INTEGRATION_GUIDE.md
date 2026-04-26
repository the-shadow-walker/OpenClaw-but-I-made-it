# Jarvis ↔ CMD Integration Guide

**Audience:** Jarvis-Claude (the instance building the new Jarvis/Noha).
**Scope:** how Jarvis uses CMD's tools, session continuity, and central context as a first-class participant — not a flaky bolt-on.
**Companion specs (read these too, in this order):**
1. [`/mnt/storage/NAS/Jarvis/INTEGRATION_CONTRACT.md`](../../INTEGRATION_CONTRACT.md) — the cross-agent API surface (memory paths, session-continuity protocol, tool naming). The contract is binding; this guide is its concrete CMD-side projection.
2. [`/mnt/storage/NAS/Jarvis/cmd/DOCS/CLAUDE_COORDINATION.md`](CLAUDE_COORDINATION.md) — protocol for CMD/Swarm/Jarvis Claudes to coordinate without stepping on each other.
3. [`/mnt/storage/NAS/Jarvis/cmd/DOCS/unification_progress.md`](unification_progress.md) — what CMD has shipped against the unification plan.

If anything in this guide disagrees with the integration contract, the contract wins. File a stanza in `CLAUDE_HANDOFF.md` to flag the drift.

---

## 0. Mental Model

Jarvis is **the chat surface and the central context curator**. CMD is a **specialist worker** that does:
- Reading/writing files on the host
- Running shell commands (safety-validated)
- ReAct-loop coding (multi-iteration with bounded inference)
- Multi-phase task chains (decomposed → planner → builders → testers → commander)
- Defensive cybersecurity (SENTINEL/blueteam mode)
- Quick one-shot shell answers (`/api/v1/quick`)

The relationship Jarvis → CMD is **the same** as CMD → GUI: peer delegation with snapshot/sidechain/merge. Whoever invokes first owns the parent context. Jarvis will usually be the parent because it owns the user-facing chat session.

CMD's full ReAct trace (every iteration, every tool call, every model response) is **never** going to fit in Jarvis's transcript and **never** should. CMD writes its inner loop to a sidechain JSONL on disk; Jarvis only sees a clean summary + deliverables list + context keys.

---

## 1. Memory Architecture (single source of truth)

| Layer | Path | Owned by | Notes |
|---|---|---|---|
| SQLite primary | `~/.agent_bin/memory.db` (= `/home/Grindlewalt/.agent_bin/memory.db`) | shared, write-anywhere | WAL mode on; concurrent readers + writers safe |
| Schema table | `shared_context` | shared | columns: `key TEXT PK, value TEXT, agent_id TEXT, created_at REAL, expires_at REAL` |
| Markdown mirror | `~/.agent_bin/central_context.md` | **Jarvis (going forward)** — CMD currently rewrites it; Jarvis will take over | Human-readable, grouped by `## agent_id` |
| Sessions dir | `~/.agent_bin/sessions/` | anyone | one JSON snapshot per save_context call |
| Sidechain dir | `~/.agent_bin/sidechains/` | anyone | one JSONL per delegated subagent run |
| Plans dir | `~/.agent_bin/plans/` | anyone | one MD per active plan (`{agent_id}_plan.md`) |

### Key-naming convention (binding)

`<scope>_<topic>_<detail>`

- `scope`: `chain`, `gui`, `cmd`, `swarm`, `jarvis`, `project`, `session`, `user`, `convo`
- `topic`: short noun (`rocket_sim`, `bookmarks`, `gmail_thread`, `morning_brief`)
- `detail`: optional (`verify`, `error`, `arch`, `summary`)

Avoid spaces, slashes, colons. TTL default 24h (set explicitly for long-lived project artifacts).

### Jarvis's central-context responsibility

The old CMD-side mirror is a placeholder. **Jarvis takes ownership** of `central_context.md`. CMD will keep writing to the SQLite table via `set_context`, but Jarvis is the one that:

1. Renders the markdown mirror with whatever structure it wants (per-user, per-project, per-day, etc.)
2. Curates: prunes stale entries, summarizes verbose ones, promotes important ones to durable Jarvis memory (the `facts.db` / `daily_logs/` analog from the old Jarvis)
3. Acts as **the** read API for human-facing context — Jarvis exposes `read_context` to the user and to CMD/GUI/Swarm callers

To take over the mirror cleanly:
- Subscribe to a SQLite trigger or poll `shared_context` for new rows (`created_at > last_seen`)
- Disable the CMD-side monkey-patch by setting env `AGENT_CENTRAL_MIRROR_OWNER=jarvis` on `ollama-cmd.service` — CMD will stop rewriting the file when this is set (you'll add that env-flag check in `cmd/core/ollama_agent_core.py:_wire_central_context_mirror`; coordinate via handoff stanza first)
- Until Jarvis is ready to own it, CMD keeps its mirror and you read from it

---

## 2. CMD Tool Inventory

Every tool CMD's ReAct agent exposes, with signature, purpose, and when Jarvis would care.

### Safe-by-default (Jarvis can fire freely)

| Tool | Args | Purpose | Returns |
|---|---|---|---|
| `read_file` | `{path, offset?, limit?}` | Read any file on host | `{lines, total_lines, more?}` |
| `web_search` | `{query}` | DuckDuckGo via the embedded search agent | `{results: [...]}` |
| `memory_lookup` | `{query}` | Search CMD's knowledge memory | `{matches: [...]}` |
| `read_context` | `{key}` OR `{prefix, limit?}` | Read shared context board | `{value}` or `{entries: [...]}` |
| `validate_arch` | `{path?}` (default `DOCS/ARCH.json`) | Schema-validate ARCH spec | `{valid, errors}` |
| `get_deps` | `{paths: [str]}` | Static import graph for Python files | `{nodes, edges}` |
| `save_context` | `{label?}` | Snapshot CMD's pinned state | `{path, summary}` |

### State-changing (use with care)

| Tool | Args | Purpose | Notes |
|---|---|---|---|
| `execute_command` | `{command, timeout?}` | Shell exec via safety validator | Risk levels: `safe/low/medium/high/blocked`. `medium`/`high` need `confirm_cb` |
| `create_file` | `{path, content, description}` | Write file (overwrites) | No diff; use `patch_file` for edits |
| `patch_file` | `{path, search, replace, description}` | Search-and-replace in file | Must match exactly once |
| `manage_server` | `{action: start/stop/status/restart, name, command?}` | Start/stop a long-lived process | Process tracked in CMD's `_server_procs` |
| `restore_context` | `{path}` | Reload a snapshot into CMD | Use after delegating to other agents |
| `write_plan` | `{plan}` (markdown with `## Architecture`, `## Files`, `## Dependencies`) | Persist task plan with checkboxes | Updates `📋 PLAN` pinned slot, `~/.agent_bin/plans/{agent_id}_plan.md` |
| `publish_context` | `{key, value, ttl_hours?, agent_id?}` | Write shared context | TTL default 24h, agent_id default `"cmd"` (override to `"jarvis"` when Jarvis writes) |
| `gui_task` | `{task, max_iterations?, context_keys?}` | Delegate to GUI agent (CMD parent) | GUI handles screenshots, OCR, xdotool — visual verification |
| `swarm_task` | `{mode: engineer/math/search, task, max_iterations?, timeout?, context_keys?}` | Delegate to swarm | Calls `POST {swarm_base}/query` or `/api/search` |
| `finish` | `{summary, success, files_created: [str]}` | End ReAct loop | Validates declared files exist on disk |

### Tool whitelisting

When Jarvis spawns a CMD job, it can pass `tool_whitelist` to limit what CMD is allowed to call. The system prompt then advertises **only** the whitelisted tools (single source of truth: `TOOL_SCHEMAS` filtered by whitelist). Use this to keep delegations focused — e.g. a "read these files and summarize" task should whitelist `{read_file, finish}` only.

```python
# Example whitelist sets
READ_ONLY = {"read_file", "memory_lookup", "read_context", "finish"}
CODER     = {"read_file", "create_file", "patch_file", "write_plan", "finish"}
SHELLER   = {"execute_command", "read_file", "manage_server", "finish"}
FULL      = None   # no whitelist; all tools available
```

---

## 3. Three Ways Jarvis Invokes CMD

Pick the right interface for the job. **Default to A (REST job) for any non-trivial task.** Use B (REST quick) for one-shot shell. Use C (in-process) only if Jarvis runs in the same Python process as `server.py` (it shouldn't).

### A. REST job — `POST /api/v1/execute` (the main path)

**Use when:** Jarvis wants CMD to run a ReAct loop with multiple tool calls. Asynchronous. This is the canonical integration.

```http
POST http://localhost:5000/api/v1/execute
Content-Type: application/json

{
  "instruction": "Read /home/Grindlewalt/projects/recipevault/server.py and tell me what routes it exposes",
  "max_iterations": 25,
  "tool_whitelist": ["read_file", "finish"],
  "system_prompt_override": null,
  "context_keys": ["project_recipevault_brief"],
  "agent_id": "jarvis-conv-abc123"
}
```

Response (immediate):
```json
{"job_id": "9e4a0cc7-6e66-4c35-99fd-d24394c99c01", "status": "queued"}
```

Then poll `GET /api/v1/jobs/<job_id>` until `status == "complete"`:
```json
{
  "job_id": "9e4a0cc7-...",
  "status": "complete",
  "success": true,
  "summary": "server.py exposes 12 routes...",
  "files_created": [],
  "execution_log": [...],
  "iterations": 7,
  "elapsed_seconds": 42.3
}
```

Or stream with `GET /api/v1/jobs/<job_id>/stream` (SSE) for live event flow.

Cancel with `DELETE /api/v1/jobs/<job_id>`.

### B. REST quick — `POST /api/v1/quick` (one-shot shell)

**Use when:** Jarvis wants to ask "what's the disk usage?" or "is X process running?" and doesn't need a ReAct loop.

```http
POST http://localhost:5000/api/v1/quick
{"question": "is nginx running"}
```

CMD's quick model translates the question to a shell command, runs it through the safety validator, executes, returns:
```json
{"command": "systemctl is-active nginx",
 "stdout": "active\n", "stderr": "", "returncode": 0,
 "success": true, "elapsed_ms": 220, "risk": "safe"}
```

Or skip the LLM with `{"command": "uptime"}`. Options: `timeout` (default 15s), `allow_risk` (`safe|low|medium`).

**Caveat:** quick endpoint shares the Ollama queue with everything else. If a long ReAct job is running, the 30s timeout will fire. Not for latency-critical paths.

### C. Task chains — `POST /api/v1/chains` (multi-phase decomposition)

**Use when:** Jarvis wants CMD to build something non-trivial — a real project, a multi-file change, anything that benefits from planner → builder → tester phasing.

```http
POST http://localhost:5000/api/v1/chains
{
  "goal": "Build a FastAPI todo app at /home/Grindlewalt/projects/todo with auth, sqlite, and a pytest suite. Workspace: /home/Grindlewalt/projects/todo",
  "max_iterations_per_subtask": 80
}
```

Returns `{"chain_id": "edad521b-..."}`. Poll `GET /api/v1/chains/<chain_id>` for `status` (`running` / `complete` / `failed`) and `subtasks[]`. Cancel with `DELETE /api/v1/chains/<chain_id>`.

Chains do their own decomposition — Jarvis just gives the goal. The decomposer extracts `Workspace: <path>` from the goal and inherits it into every subtask's `cwd` for AC checks.

### Don't do this

- Don't reach into `~/.agent_bin/memory.db` directly to set context **and then expect the mirror to update** — go through the REST `POST /api/v1/context` endpoint or the `publish_context` tool. Direct SQLite writes bypass the mirror rebuild trigger. (When Jarvis takes over the mirror, this caveat reverses — but you have to actually own the mirror first.)
- Don't fire `gui_task` / `swarm_task` from outside CMD as if they were Jarvis tools. They're CMD tools that Jarvis triggers indirectly by giving CMD a job that contains them in its whitelist. If Jarvis wants to delegate to GUI directly, hit `POST /api/v1/gui` (CMD's GUI proxy) or talk to the GUI server (port 5005) itself.

---

## 4. Session Continuity Protocol — Jarvis as Parent

This is binding (contract §2). When Jarvis invokes CMD:

1. **Jarvis snapshots its own state** before the call. (Whatever Jarvis's snapshot primitive looks like — file-first markdown export, JSON dump of pinned facts, etc. The contract just requires that Jarvis can recover if anything fails downstream.)

2. **Jarvis publishes context keys** that CMD will need. Don't pass CMD the entire conversation transcript — pass keys.
   ```python
   POST /api/v1/context {"key": "project_todo_spec", "value": "<detailed spec markdown>", "agent_id": "jarvis", "ttl_hours": 24}
   ```

3. **Jarvis dispatches the job** with `context_keys: ["project_todo_spec", "user_pref_python_style"]` in the body. CMD will inject those keys into the agent's pinned context at run start so the agent sees them even after compaction.

4. **CMD runs in a fresh ReAct loop** in its own process. Its full transcript goes to `~/.agent_bin/sidechains/<job_id>_cmd.jsonl`. **Jarvis never sees that file** unless it explicitly opens it for debugging.

5. **CMD publishes results.** Every non-trivial CMD job MUST drop at least one of:
   - A markdown deliverable file at `~/.agent_bin/results/<topic>_<id>.md` (path returned in `deliverables[]`)
   - One or more context keys with the substance (`<scope>_<topic>_result`)
   Raw stdout-only is not acceptable per contract §2.

6. **Jarvis merges back.** Read the result envelope:
   ```json
   {
     "success": true,
     "summary": "one-line outcome",
     "deliverables": ["/home/Grindlewalt/.agent_bin/results/todo_spec_review.md"],
     "context_keys_written": ["project_todo_review_result"],
     "sidechain_path": "/home/Grindlewalt/.agent_bin/sidechains/<id>_cmd.jsonl",
     "error": null
   }
   ```
   Jarvis pulls in the markdown deliverable (or the context key value) — that's what enters the user-facing transcript. **Not** the execution_log.

### Why this matters

Without snapshot/sidechain/merge, every CMD call would dump a 200-iter ReAct trace into Jarvis's context window. Three calls in and Jarvis is at 100% context with no headroom for actual conversation. The protocol is the **only** thing keeping Jarvis stable under multi-delegation load.

### Failure semantics

If CMD's job fails (`success: false`), the result envelope has `error` set and `deliverables` may be empty. Jarvis decides:
- Retry with a refined task? (one shot, max)
- Surface to user? ("CMD said: <error>")
- Abandon and try a different tool/agent?

Don't auto-retry indefinitely. A failed CMD job is a real signal — Jarvis should think before re-firing.

---

## 5. Central Context — Jarvis as Curator

The shared context board is **the** integration spine. Every cross-agent handoff goes through it. Jarvis is the curator.

### Reads — three options

```python
# A. Direct SQLite (in-tree code, same host) — fastest
import sqlite3
conn = sqlite3.connect("/home/Grindlewalt/.agent_bin/memory.db")
row = conn.execute(
    "SELECT value, agent_id, expires_at FROM shared_context WHERE key=?",
    (key,),
).fetchone()

# B. AgentMemory class wrapper (preferred for Python in-tree)
from cmd.core.react_memory import AgentMemory
mem = AgentMemory(agent_id="jarvis")
val = mem.get_context("project_todo_spec")
entries = mem.list_context(prefix="project_")

# C. REST (any language, any host)
GET  http://localhost:5000/api/v1/context?prefix=project_   # list
GET  http://localhost:5000/api/v1/context?key=project_todo  # single (use prefix=key, limit=1)
```

### Writes — two options

```python
# A. AgentMemory (in-tree)
mem.set_context(
    key="user_pref_morning_brief",
    value="<markdown>...</markdown>",
    agent_id="jarvis",
    ttl=3600 * 24 * 7,   # 7 days
)
# This call also kicks the central_context.md rebuild.

# B. REST
POST http://localhost:5000/api/v1/context
{"key": "user_pref_morning_brief", "value": "...", "agent_id": "jarvis", "ttl_hours": 168}

# Both paths hit the same SQLite + trigger the same mirror rebuild.
```

### Deletes

```http
DELETE http://localhost:5000/api/v1/context/<key>
```

Or set TTL to 0 / past timestamp.

### Markdown mirror — `~/.agent_bin/central_context.md`

CMD currently auto-rebuilds this on every write. **It's a flat append-only `## agent_id` grouping, capped at 200 entries.** That's a placeholder, not the design. Jarvis should:

1. Subscribe (poll or WAL hook) to `shared_context` writes
2. Render its **own** `central_context.md` structure — Jarvis can carve it into `## Conversations`, `## Active Projects`, `## User Profile`, `## Today`, `## Pending Tasks`, etc.
3. Curate: ephemeral entries (TTL < 1h) shouldn't pollute the mirror; verbose entries (>2KB) should be summarized + linked to the full text in the SQLite row
4. Write it atomically (`.tmp` + `os.replace`) — current CMD-side write is non-atomic, race condition flagged in contract §8

To take ownership: set `AGENT_CENTRAL_MIRROR_OWNER=jarvis` on `ollama-cmd.service` (you'll add the env check in `cmd/core/ollama_agent_core.py:_wire_central_context_mirror`; coordinate the patch via handoff first so I add it on CMD side).

### Compaction floor (contract §6 — binding)

When Jarvis compacts its own conversation, **never** compress:
- Fenced code blocks (` ``` … ``` `)
- Absolute file paths (anything starting with `/`)
- URLs
- The system prompt
- The last 3 user/assistant messages

These survive verbatim. Everything else can be summarized. CMD's hifi compression already does this; Jarvis must follow the same floor or cross-agent handoffs lose their reference paths.

---

## 6. Sidechains — what Jarvis sees vs doesn't

Every delegated subagent run dumps a JSONL trace at `~/.agent_bin/sidechains/<id>_<target>.jsonl`. One JSON object per line:
```json
{"iter": 1, "tool": "read_file", "args": {"path": "..."}, "result": {...}, "ts": 1714137600.123}
{"iter": 2, "tool": "create_file", "args": {...}, "result": {...}, "ts": 1714137605.456}
...
```

**Jarvis's transcript should never include this.** The sidechain path goes in the result envelope (`sidechain_path`); Jarvis can read it for post-mortem debugging when the user asks "what happened in that CMD job?" but it does NOT enter the live conversation.

Sidechain sweep policy (TBD — flagged in contract §8): probably "delete on success after 7 days, keep on failure indefinitely". Jarvis can run a daily cleanup job once it owns memory hygiene.

---

## 7. Concrete Jarvis-side Code Patterns

These are sketches — implement in whatever stack Jarvis ends up using (Python recommended for ecosystem alignment).

### 7.1 CMD client (sync wrapper around the REST API)

```python
import requests
import time
import os

class CMDClient:
    def __init__(self, base="http://localhost:5000", agent_id="jarvis"):
        self.base = base
        self.agent_id = agent_id

    def quick(self, question=None, command=None, timeout=15, allow_risk="low"):
        """One-shot shell — Q: 'is nginx running' → executes, returns dict."""
        body = {"timeout": timeout, "allow_risk": allow_risk}
        if question: body["question"] = question
        if command:  body["command"]  = command
        r = requests.post(f"{self.base}/api/v1/quick", json=body, timeout=timeout + 5)
        r.raise_for_status()
        return r.json()

    def execute(self, instruction, max_iterations=25, tool_whitelist=None,
                context_keys=None, poll_interval=2.0, max_wait=1800):
        """Submit ReAct job, poll until done, return result envelope."""
        body = {
            "instruction": instruction,
            "max_iterations": max_iterations,
            "agent_id": self.agent_id,
        }
        if tool_whitelist is not None: body["tool_whitelist"] = list(tool_whitelist)
        if context_keys:               body["context_keys"]  = list(context_keys)
        r = requests.post(f"{self.base}/api/v1/execute", json=body)
        r.raise_for_status()
        job_id = r.json()["job_id"]

        deadline = time.time() + max_wait
        while time.time() < deadline:
            status = requests.get(f"{self.base}/api/v1/jobs/{job_id}").json()
            if status.get("status") in ("complete", "failed"):
                return self._envelope(status)
            time.sleep(poll_interval)
        # Timeout — cancel
        requests.delete(f"{self.base}/api/v1/jobs/{job_id}")
        return {"success": False, "error": "timeout", "summary": "Job exceeded max_wait", "deliverables": [], "context_keys_written": [], "sidechain_path": None}

    def _envelope(self, status):
        return {
            "success": status.get("success", False),
            "summary": status.get("summary", ""),
            "deliverables": status.get("deliverables", []),
            "context_keys_written": status.get("context_keys_written", []),
            "sidechain_path": status.get("sidechain_path"),
            "error": status.get("error"),
        }

    def chain(self, goal, max_iterations_per_subtask=80, poll_interval=5.0, max_wait=7200):
        """Multi-phase task chain. Same envelope shape on completion."""
        r = requests.post(f"{self.base}/api/v1/chains",
                          json={"goal": goal, "max_iterations_per_subtask": max_iterations_per_subtask})
        r.raise_for_status()
        chain_id = r.json()["chain_id"]
        deadline = time.time() + max_wait
        while time.time() < deadline:
            status = requests.get(f"{self.base}/api/v1/chains/{chain_id}").json()
            if status.get("status") in ("complete", "failed"):
                return self._envelope(status)
            time.sleep(poll_interval)
        requests.delete(f"{self.base}/api/v1/chains/{chain_id}")
        return {"success": False, "error": "timeout", "summary": "Chain exceeded max_wait", "deliverables": [], "context_keys_written": [], "sidechain_path": None}

    # ---- context board ----
    def publish(self, key, value, ttl_hours=24, agent_id=None):
        body = {"key": key, "value": value, "ttl_hours": ttl_hours,
                "agent_id": agent_id or self.agent_id}
        r = requests.post(f"{self.base}/api/v1/context", json=body)
        r.raise_for_status()

    def read(self, key=None, prefix=None, limit=20):
        params = {}
        if key:    params["key"]    = key
        if prefix: params["prefix"] = prefix
        params["limit"] = limit
        r = requests.get(f"{self.base}/api/v1/context", params=params)
        r.raise_for_status()
        return r.json()
```

### 7.2 Delegation pattern (Jarvis tool implementation)

```python
def jarvis_tool_code_task(task: str, context_keys: list = None) -> str:
    """Jarvis tool: delegate a coding task to CMD. Returns the markdown deliverable."""
    cmd = CMDClient()

    # Step 1: snapshot Jarvis state (file-first MD dump, JSON, whatever)
    jarvis.snapshot(label=f"pre_code_task_{slug(task)}")

    # Step 2: ensure context keys exist (Jarvis may have set them earlier in the convo)
    # No-op if they're already published.

    # Step 3: dispatch
    result = cmd.execute(
        instruction=task,
        tool_whitelist=["read_file", "create_file", "patch_file", "write_plan", "execute_command", "finish"],
        context_keys=context_keys or [],
        max_iterations=80,
    )

    # Step 4: merge — pull markdown deliverables, NOT execution_log
    if not result["success"]:
        return f"❌ CMD failed: {result['error']}\n\nSummary: {result['summary']}"

    md_chunks = []
    for path in result["deliverables"]:
        if path.endswith(".md") and os.path.exists(path):
            md_chunks.append(open(path).read())

    # Step 5: optionally pull context keys CMD wrote
    for key in result["context_keys_written"]:
        entry = cmd.read(key=key)
        if entry.get("context"):
            md_chunks.append(f"### {key}\n\n{entry['context'][0]['value']}")

    summary = result["summary"]
    body = "\n\n---\n\n".join(md_chunks) if md_chunks else "(no markdown deliverable returned)"
    return f"**CMD result:** {summary}\n\n{body}"
```

### 7.3 Background watch (Jarvis monitoring CMD)

```python
def watch_running_jobs():
    """Fire periodically; surface in-flight CMD jobs in central_context.md."""
    jobs = requests.get("http://localhost:5000/api/v1/jobs").json()
    in_flight = [j for j in jobs if j["status"] == "running"]
    if not in_flight:
        cmd_client.publish("jarvis_cmd_jobs_inflight", "(none)", ttl_hours=1)
        return
    summary = "\n".join(
        f"- `{j['job_id'][:8]}` iter {j.get('iteration', '?')} — {j.get('current_action', '?')[:80]}"
        for j in in_flight
    )
    cmd_client.publish("jarvis_cmd_jobs_inflight", summary, ttl_hours=1)
```

---

## 8. Failure Modes & Recovery

| Symptom | Likely cause | Fix |
|---|---|---|
| `POST /api/v1/execute` returns 503 / connection refused | `ollama-cmd.service` is down | `mcssh "sudo systemctl restart ollama-cmd"` — flag in handoff if persistent |
| Job hangs at iter 1 for >2min | Cold model load (qwen3.6:35b) — first call after `OLLAMA_KEEP_ALIVE=5m` expiry | Wait it out; Jarvis should set `max_wait>=300` for cold-start tolerance |
| Job completes with `success: false, summary: "..."` and `iterations: 0` | Wrong model name in CMD config (KeyError: 'message' silent fail) | Check `journalctl -u ollama-cmd -n 50`; coordinate fix via handoff |
| `/api/v1/context POST` 500s | SQLite WAL contention or disk full | Quick fix: `df -h ~/.agent_bin/`; `sqlite3 memory.db ".recover"` if corrupt |
| `central_context.md` shows stale entries Jarvis wrote | CMD-side mirror rebuild lost a write (non-atomic write race) | After Jarvis takes ownership of the mirror, this goes away. Until then: re-publish |
| CMD job's deliverable path 404s | CMD declared a file that doesn't exist on disk (the `finish()` guard *should* catch this but doesn't always when the path is outside CMD's tracked files-created list) | Read `sidechain_path` to debug; flag agent bug in handoff |
| Two CMD jobs start writing the same file | Jarvis dispatched concurrent jobs without coordination | Serialize via Jarvis-side lock or use chain endpoint (chains serialize internally) |

---

## 9. Verification Recipe (run before going to prod)

A 6-step smoke test for the Jarvis↔CMD spine. Run each from Jarvis's environment.

1. **Quick endpoint reachability**
   ```python
   r = cmd_client.quick(command="uptime")
   assert r["success"] and "load average" in r["stdout"]
   ```

2. **Context round-trip**
   ```python
   cmd_client.publish("jarvis_smoke_test", "hello world", ttl_hours=1, agent_id="jarvis")
   r = cmd_client.read(key="jarvis_smoke_test")
   assert r["context"][0]["value"] == "hello world"
   ```

3. **ReAct job — read-only**
   ```python
   res = cmd_client.execute(
       "Read /etc/hostname and return its contents in your finish summary",
       tool_whitelist=["read_file", "finish"],
       max_iterations=5,
   )
   assert res["success"] and len(res["sidechain_path"]) > 0
   ```

4. **ReAct job — context injection**
   - Publish `project_smoke_brief = "Build a single-file Python script that prints 'hello smoke test'"`
   - Execute with `context_keys=["project_smoke_brief"]` and a coder whitelist
   - Verify the deliverable exists on disk

5. **Chain — multi-phase**
   - Submit a tiny chain: `"Create DOCS/ARCH.json at /tmp/jarvis-smoke/ describing a single GET /health route. Workspace: /tmp/jarvis-smoke"`
   - Poll to completion; assert `subtask_count >= 2` and at least one phase passed AC

6. **Failure surfacing**
   - Submit a deliberately impossible job: `"Run rm -rf /"` (will hit the safety validator)
   - Assert `success: false`, `error` mentions safety/blocked, and Jarvis surfaces this clearly to the user instead of swallowing it

If all 6 pass, the Jarvis↔CMD spine is healthy. Add to a CI/cron loop so degradation is caught early.

---

## 10. Migration Checklist (taking over central_context.md)

When Jarvis is ready to own the markdown mirror:

1. [ ] Implement Jarvis-side mirror renderer with the structure Jarvis wants
2. [ ] Test atomic write (`.tmp` + `os.replace`)
3. [ ] Implement subscription mechanism (poll `shared_context.created_at` every 5s, OR SQLite WAL trigger)
4. [ ] File handoff stanza: `need from CMD: add AGENT_CENTRAL_MIRROR_OWNER env check`
5. [ ] CMD ships the env check (this is a 5-line patch — `if os.getenv("AGENT_CENTRAL_MIRROR_OWNER", "cmd") != "cmd": return`)
6. [ ] Set the env on `ollama-cmd.service`, restart
7. [ ] Verify CMD stops touching the file (timestamp doesn't update on `set_context` from CMD)
8. [ ] Verify Jarvis-side renderer takes over (timestamp updates on Jarvis's poll cycle)
9. [ ] Run the verification recipe (§9) again

Until step 4, leave the CMD-side mirror alone — no reason to break a working placeholder.

---

## 11. Known Gaps (be aware, file in handoff if blocking)

- **No backpressure on /api/v1/execute** — Jarvis can submit 100 jobs in a loop and saturate Ollama's serial queue. Use Jarvis-side semaphore (recommend max 2 concurrent CMD jobs).
- **`/api/v1/jobs/<id>/stream` SSE buffering** — some HTTP clients buffer SSE incorrectly; if streaming hangs, fall back to polling.
- **Chain subtask roles** are hardcoded (planner/builder/tester/commander) — Jarvis can't currently inject a custom role. If needed, file in handoff and CMD adds it.
- **Sidechain log size** — busy chains can produce 50MB+ JSONL files. No automatic rotation yet. If disk pressure becomes real, Jarvis runs `find ~/.agent_bin/sidechains/ -mtime +7 -delete` daily.
- **Swarm AgentMemory mirror not yet shipped** (per `CLAUDE_HANDOFF.md`) — until Swarm's side lands, `swarm_task` calls go to swarm via REST and don't share Jarvis's session-continuity flow as cleanly. Pull progress from handoff stanzas; coordinate via Jarvis-Swarm handoff messages.

---

## 12. Closing — Don't Build a Flaky Connector

The point of this guide is precisely so Jarvis doesn't end up as another barely-works integration that breaks the moment a job times out or a context key gets overwritten. The protocols here are not optional:

- **Always** snapshot before delegating.
- **Always** pass context_keys, never the full transcript.
- **Always** read the result envelope, never the execution_log.
- **Always** consume markdown deliverables, never raw stdout for non-trivial work.
- **Always** respect the compaction floor.
- **Always** post a handoff stanza when you ship something the other side needs to know about.

If something in CMD's behavior makes any of the above hard, that's a CMD bug — file it in `CLAUDE_HANDOFF.md` and CMD Claude fixes it. Don't work around it on Jarvis's side; that's how the connector goes flaky.

---

**Contract version:** v0.1
**Guide version:** v0.1
**Last updated:** 2026-04-26
