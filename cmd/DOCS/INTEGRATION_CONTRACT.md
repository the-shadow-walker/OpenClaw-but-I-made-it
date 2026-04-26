# Integration Contract — CMD ⇄ Swarm ⇄ GUI

**Audience:** any agent, any Claude instance, working on either side of the stack.
**Canonical location on remote:** `/mnt/storage/NAS/Jarvis/INTEGRATION_CONTRACT.md` (repo root, both subprojects descend from it).
**Status:** v0.1 — CMD/GUI side has shipped pieces; swarm side still picks up.

This is the **single source of truth** for how the three agent stacks talk. If something here disagrees with code, the code is wrong, fix the code.

---

## 1. Universal Memory

### 1.1 Storage

| Layer | Path | Owner | Notes |
|---|---|---|---|
| SQLite primary | `~/.agent_bin/memory.db` (= `/home/Grindlewalt/.agent_bin/memory.db`) | CMD's `AgentMemory` | WAL mode on; safe for concurrent readers + writers |
| Markdown mirror | `~/.agent_bin/central_context.md` | CMD's `AgentMemory.set_context` rewrites it on every write | Human-readable; agents may **read** it, must not write directly |
| Sessions dir | `~/.agent_bin/sessions/` | Anyone | One JSON per session-snapshot |
| Sidechain dir | `~/.agent_bin/sidechains/` | Anyone | One JSONL per delegated subagent run |
| Plans dir | `~/.agent_bin/plans/` | Anyone | One markdown per active plan |

### 1.2 Schema (table `shared_context`)

```
key         TEXT PRIMARY KEY
value       TEXT
agent_id    TEXT      -- "cmd", "gui", "swarm.math", "swarm.engineer", "swarm.search", etc.
created_at  REAL      -- unix epoch
expires_at  REAL      -- unix epoch, 0 = never
```

### 1.3 Access path — pick ONE per process

**A. Direct SQLite (preferred for in-tree code on the same host)**
```python
from cmd.core.react_memory import AgentMemory   # CMD-side
mem = AgentMemory(agent_id="swarm.math")
mem.set_context("rocket_sim_inertial_tensor", "{...}", ttl=3600)
val = mem.get_context("rocket_sim_problem_brief")
```
Swarm should grow a thin wrapper that imports `AgentMemory` directly (it's pure stdlib + sqlite3, no CMD-only deps). If we want to keep them decoupled, copy the class into `swarm/core/agent_memory.py` — both must point at the same `~/.agent_bin/memory.db` path.

**B. REST (preferred for remote clients, run_me.py, dashboards)**
```
POST   http://localhost:5000/api/v1/context        {key, value, ttl?, agent_id?}
GET    http://localhost:5000/api/v1/context?prefix=rocket_sim
DELETE http://localhost:5000/api/v1/context/<key>
```
Hosted by `cmd/server.py`. Read-only mirror could be hosted by swarm later, but writes always land in the same SQLite via this endpoint or AgentMemory.

### 1.4 Key-naming convention

`<scope>_<topic>_<detail>`

- `scope`: `chain`, `gui`, `cmd`, `swarm`, `project`, `session` — who/what it's about
- `topic`: short noun — `rocket_sim`, `bookmarks`, `deploy`
- `detail`: optional — `verify`, `error`, `arch`

Examples:
- `project_rocket_sim_problem_brief`
- `swarm_search_results_three_js_physics`
- `chain_edad521b_phase3_ac_result`

Avoid spaces, slashes, colons in keys. TTL default 24h unless the value is a long-lived project artifact.

---

## 2. Session Continuity Protocol

When agent **A** invokes tool/agent **B**, the contract is:

1. **A snapshots** its current pinned slots + files-created list to `~/.agent_bin/sessions/<A_id>_<timestamp>.json`. (Helper: `agent.save_context(label)`.)
2. **A passes** to B only the **context keys** B needs (list of strings), **not** A's transcript. B reads them via `read_context`.
3. **B runs in a fresh ReAct loop** with its own tool whitelist. Its full ReAct trace lands in `~/.agent_bin/sidechains/<B_id>_<timestamp>.jsonl`.
4. **B publishes results** via `publish_context` — at minimum one key like `<scope>_<topic>_result` containing the markdown deliverable path.
5. **A merges back**: parent sees one summary line + the result keys; A's own transcript is **not** polluted by B's iterations.

### Required result fields (B → A)

Every subordinate call resolves to a JSON dict the parent can consume:

```json
{
  "success": true,
  "summary": "one-line outcome",
  "deliverables": ["abs/path/to/report.md", "abs/path/to/code.py"],
  "context_keys_written": ["swarm_search_rocket_physics_result"],
  "sidechain_path": "/home/Grindlewalt/.agent_bin/sidechains/swarm_search_2026-04-26T12-45-00.jsonl",
  "error": null
}
```

If `success: false`, `error` MUST be set and `deliverables` MAY be empty. Parent decides whether to retry or surface to the user.

### Markdown deliverable rule

Any non-trivial subagent (math solver, deep search, engineer, GUI verify) **must** drop at least one `.md` file at a path it returns in `deliverables`. Raw stdout / JSON only is not acceptable for end-of-task — it forces parents to summarize twice.

---

## 3. Tool Naming Across Boundaries

Cross-boundary delegation tools, all available on the parent side:

| Tool name | From | To | Purpose |
|---|---|---|---|
| `gui_task` | CMD | GUI | Visual/interactive verification |
| `code_task` | GUI | CMD | Heavy code work |
| `math_task` | CMD or swarm.engineer | swarm.math | Symbolic / numeric problem |
| `deep_search_task` | any | swarm.search | Multi-step web research → markdown |
| `engineer_task` | CMD | swarm.engineer | Hardware/circuit/component sourcing |

Implementation pattern is identical: each handler is ~20 lines that calls `SubAgentInvoker.run(target=…, task=…, context_keys=[…])`. The only delta between handlers is the `target` string and (for swarm targets) the swarm REST endpoint that gets hit.

**Adding a new cross-boundary tool** is mechanical:
1. Add a TOOL_SCHEMAS entry on the side that calls it.
2. Add a `_handle_<name>` method that delegates through SubAgentInvoker.
3. Make sure the target side actually exposes the corresponding agent.
4. Update this contract's table.

---

## 4. Swarm-Side Implementation Checklist

What CMD has shipped that swarm needs to mirror:

| CMD has | Swarm needs |
|---|---|
| `cmd/core/subagent.py` (SubAgentInvoker) | `swarm/core/subagent.py` — same shape, target="cmd"/"gui" instead of swarm |
| `AgentMemory` pointed at `~/.agent_bin/memory.db` | Either import CMD's class or duplicate it with same schema and path |
| `publish_context` / `read_context` ReAct tools | Same tools registered in swarm's tool registry, same signatures |
| `save_context` / `restore_context` on agent | Same primitive on swarm's base agent class |
| ReAct trace → sidechain when delegated | Swarm should write its own jobs' deep-loop transcripts to `~/.agent_bin/sidechains/swarm_<job_id>.jsonl` when called as a subagent |
| Markdown deliverable for every job | Swarm `deep_search`, `swarm_solver`, `engineer` modes must each produce `~/.agent_bin/results/<topic>_<id>.md` |

Until swarm ships its half, CMD-side `math_task`/`deep_search_task`/`engineer_task` tools will **fail loudly with a clear error** rather than silently degrade. Better than fake success.

---

## 5. Cross-Agent Project Lifecycle (the rocket simulator example)

User asks Jarvis to build a guided-rocket simulator. The expected flow:

1. **CMD** receives the request, calls `deep_search_task` for "real-world rocket physics + Three.js patterns".
   - CMD `save_context("rocket_pre_research")` → snapshot.
   - Swarm.search runs, dumps `~/.agent_bin/results/rocket_physics_research.md`, publishes `project_rocket_sim_research = "<path>"`.
   - CMD merges back: only the markdown path enters CMD's transcript, not the 200-iter search ReAct loop.
2. **CMD** calls `math_task` for "RK4 integrator + drag coefficient model".
   - Swarm.math returns `~/.agent_bin/results/rocket_math.md` with derivations.
3. **CMD** writes the actual code (its specialty), reading the two markdown files via `read_context`.
4. **CMD** calls `gui_task` to launch headless browser, verify WebGL canvas renders.
5. **CMD** publishes `project_rocket_sim_status = "shipped"` and the user sees a clean summary.

At no point does any one agent's transcript exceed its context budget, because every cross-agent hop offloads its inner loop to a sidechain and only summary + markdown paths flow.

---

## 6. Compaction Floor

The 90% high-fidelity compression in CMD does NOT compress:

- Fenced code blocks (` ``` … ``` `)
- Absolute file paths
- URLs
- The system prompt
- The last 3 messages

Markdown deliverables (`.md` files referenced by path) are **never** in the transcript itself — they're files on disk, the transcript only carries the path. So compaction has no effect on them.

Swarm's compaction (when it lands) MUST follow the same floor rules so deliverable references survive.

---

## 7. Versioning

Contract version: `0.1` (top of file). Bump minor for additions, major for breaking changes. Both sides should print the contract version they were built against on startup; mismatched majors should refuse to delegate.

---

## 8. Open Questions

- Should we add a separate `~/.agent_bin/locks/` for cross-process coordination, or rely on SQLite's locking? (Currently: rely on SQLite WAL.)
- `central_context.md` rewrite is non-atomic. If two writers race, last-write wins on the markdown but SQLite is consistent. Probably fine, file it under "monitor".
- Sidechain JSONL grows forever. Need a sweep policy — delete on success after N days? Keep on failure? TBD.
