# Jarvis Implementation Guide — Unification of Agents

The architecture for combining Swarm (engineer / math / deep_search), CMD
(coder + GUI + blue-team + ReAct), and Jarvis-the-orchestrator into one
seamless assistant whose context never gets polluted by tool internals.

This is **Jarvis**: my personal assistant, with memory, that delegates to
every advanced tool (AT) below it without inheriting their ReAct clutter.

---

## 1. Headline Architecture — Session Continuity

The single most important rule in the whole system:

> **Before any AT runs, Jarvis snapshots its session. The AT runs in its
> own context. When the AT finishes, only its clean deliverables are
> merged back into Jarvis's session — never the ReAct loops, never the
> intermediate failures, never the tool-internal chatter.**

```
                    ┌──────────────── Jarvis session (abc-123) ───────────────┐
                    │  user: "build me a guided rocket sim"                   │
                    │  jarvis: "ok, planning..."                              │
                    │  [SAVE abc-123.context] ───────────────────┐            │
                    └────────────────────────────────────────────┼────────────┘
                                                                 │
                                                                 ▼
                            ┌──── swarm:deep_search (own context) ────┐
                            │   500 ReAct turns, 40 searches,         │
                            │   12 retries, full transcript           │
                            │   → writes: Rocket-science-for-sim.md   │
                            └────────────┬────────────────────────────┘
                                         │ (clean .md only)
                    ┌────────────────────┼────────────────────────────────────┐
                    │  [RESTORE abc-123]  │                                   │
                    │  + insert reference to Rocket-science-for-sim.md       │
                    │  jarvis: "research done, now math..."                  │
                    │  [SAVE abc-123.context] ──────────────────┐            │
                    └───────────────────────────────────────────┼────────────┘
                                                                ▼
                            ┌──── swarm:math (own context) ─────────┐
                            │   200 ReAct turns, 8 SP retries       │
                            │   → writes: Rocket-math-burnout.md    │
                            └────────────┬──────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────────────────────┐
                    │  [RESTORE abc-123]                                      │
                    │  + insert Rocket-math-burnout.md                        │
                    │  → cmd writes ~/Rocket-sim/                             │
                    │  → final reply: 3 deliverables, no ReAct in transcript  │
                    └─────────────────────────────────────────────────────────┘
```

Every layer — Jarvis→AT, AT→sub-AT (e.g. CMD→swarm:math), AT→sub-tool
(e.g. solver→search) — uses the same pattern. **Save before, restore +
merge after.** ReAct chatter never crosses a layer boundary.

---

## 2. Unification Goals (recap)

Combine the strengths of three existing systems into one assistant:

- **Swarm** — engineer mode, math solver, deep research
- **CMD** — coding agent, GUI agent, ReAct loop, blue-team
- **Jarvis** — long-term memory, conversational identity, master orchestrator

Outcome: Jarvis behaves like a single assistant. Underneath, every AT
runs with its own clean context, returns markdown deliverables, and
merges into Jarvis's main session without polluting it.

---

## 3. The Single Context File

There is one canonical session file per Jarvis conversation:

```
~/.agent_bin/sessions/jarvis_<session_id>.context
```

Format: append-only JSONL. Every entry is one of:
- `user_message`, `jarvis_message`, `tool_invocation`, `deliverable_ref`,
  `compression_summary`, `restore_point`.

When an AT runs, it gets a **derived** context file:
```
~/.agent_bin/sessions/<at_role>_<job_id>.context
```
…which contains the relevant slice of the parent (the task + necessary
prior deliverables, but **not** prior ReAct logs).

When the AT finishes:
1. Its deliverables (`<topic>.md` + any files) land in `~/.agent_bin/results/`.
2. A `deliverable_ref` entry is appended to the parent `.context`.
3. The AT's derived context file is archived (not deleted — debugging).
4. Jarvis resumes with a transcript that is `prior + tool_invocation +
   deliverable_ref`. **No ReAct content.**

This is what the spec calls *"agents each work off of [the main context],
making their own sub context and then rebuilding back into the main
context."*

---

## 4. Tool Surface

| Tool | Endpoint | Best for | Latency |
|---|---|---|---|
| `swarm:math` | `POST :5002/subagent/math` | Equations, ODE/PDE, solve, units | 5–25 min |
| `swarm:engineer` | `POST :5002/subagent/engineer` | BOM, circuits, firmware, datasheets | 8–30 min |
| `swarm:deep_search` | `POST :5002/subagent/deep_search` | Multi-round research with citations | 3–10 min |
| `cmd:code` | `POST :5000/api/v1/execute` (mode=code) | Coding agent (master/sub) | varies |
| `cmd:gui` | `POST :5000/api/v1/execute` (mode=gui) | GUI agent (master/sub) | varies |
| `cmd:blue` | `POST :5000/api/v1/execute` (mode=blue) | Blue-team / security work | varies |
| `publish_context` | `POST :5000/api/v1/context` | Hand off a value/path | <100 ms |
| `read_context` | `GET  :5000/api/v1/context` | Pull a previously published value | <100 ms |
| `save_context(label)` | (per-agent) | Snapshot session before delegating | <50 ms |
| `restore_context(path)` | (per-agent) | Restore + merge after AT returns | <50 ms |

Every AT MUST implement `save_context` and `restore_context`. Already
stubbed in `core/base_agent.py`. Subclasses with richer state override.

---

## 5. The Refinements — Implementation Mapping

Each bullet from the spec, mapped to concrete code changes.

### 5.1 Deep Search rewrite
*"writes in md, actually good research with stronger searching agents
with bigger context AND has context of what we are doing with a smart
model (qwen3.6:35b-A3B-Grindlewalt)"*

- File: `compute/deep_search_agent.py` (already stubbed in Chunk 8).
- Model: default `qwen3.6:35b-A3B-Grindlewalt` (override
  `SWARM_DEEP_SEARCH_MODEL`). Currently flipped to 27b-IQ4 — revert to
  35b-A3B once GPU fit confirmed.
- Context: take parent session digest (last N user goals + active
  deliverable refs) as input — not just the bare query. Implement via
  `extra.parent_context_keys` parameter.
- Output: structured md with `## Summary / ## Findings / ## Sources /
  ## Methodology` (already specced).
- Round budget: 4 rounds default, configurable; bigger `num_ctx` (32K+)
  so each round sees the full prior synthesis.

### 5.2 Swarm-Solver rewrite
*"actually work, write in-depth with markdown, full context"*

- File: `compute/react_solver.py`.
- Add: full markdown report writer (currently writes one-line summaries).
  After all SPs solve, generate `Math-<topic>.md` with derivation,
  intermediate steps, units, dimensional checks, cited sources.
- Full context: solver receives parent context digest the same way
  deep_search does — knows *why* the math is being asked.
- Currently the solver returns `summary` strings; promote the deliverable
  to first-class output.

### 5.3 Engineer mode upgrades
*"context, model, research, md writing, ALL its tools — circuit gen,
Amazon, Adafruit, other site APIs for descriptions/price/reviews/etc.
+ session continuity"*

- File: `engineer/engineer_mode.py`.
- New tool modules under `engineer/tools/`:
  - `circuit_gen.py` — schematic generation (deferred in current PR)
  - `adafruit_api.py` — product search + datasheet fetch
  - `amazon_api.py` — product search via RapidAPI (`RAPIDAPI_KEY` env)
  - `octopart_api.py` — distributor search, lifecycle status
  - `digikey_api.py` — alternative distributor
  - `datasheet_fetch.py` — already partial in `project_mode.py`; extract
- Each tool registered in `engineer_mode`'s tool dispatch, gated by
  domain keywords (e.g. don't query Amazon for steel-beam design).
- Markdown deliverables: `Project-<name>.md` (overall),
  `BOM-<name>.md`, `Schematic-<name>.md`, `Firmware-<name>/`.
- Session continuity: engineer is just another AT — same save/restore
  contract.

### 5.4 Session continuity (the rocket-sim flow)
*"before each tool use, save session + context (abc-123.context), tool
runs, results produced, results thrown into old context, ReAct gone."*

Implementation:
1. **`SessionManager`** (new file: `core/session_manager.py`)
   - `snapshot(session_id, label) -> path` — writes
     `~/.agent_bin/sessions/<session_id>_<label>_<ts>.context`
   - `restore(path) -> dict` — loads, returns the session state
   - `merge_deliverables(session_path, deliverables: List[str])` —
     appends `deliverable_ref` entries
2. **`SubAgentInvoker.run()`** wraps every call:
   ```python
   snapshot = session.snapshot(label=f"pre_{target}")
   result = invoke(target, task, ...)
   session.merge_deliverables(snapshot, result.deliverables)
   return result   # caller sees ONLY summary + deliverable paths
   ```
3. **AT-to-AT** uses identical pattern. CMD calls `swarm:math`?
   CMD's session is snapshotted, swarm runs clean, math returns,
   CMD restores + merges. Swarm never sees CMD's ReAct trace; CMD
   never sees swarm's.
4. **Sub-tool inside AT** (e.g. solver doing 8 searches): the AT keeps
   its own internal manifest, but writes one `search-results-<query>.md`
   per search. The AT's final deliverable references those .md files;
   the parent never sees raw search dumps.

### 5.5 Auto-compression at 90% context
*"when 90% used, write summaries of everything so far, but not compressing
.md's, just talking, ReAct loops that are stale/failed/old."*

Policy implemented in two places:

**Solver-side** (existing, in `react_solver.py`): three-stage compression
already shipped — Stage 2 prune, Stage 2.5 verbatim extraction
(code/paths/URLs preserved byte-identical), Stage 3 LLM summarize.

**Jarvis-side** (new):
- Trigger: when session token estimate ≥ 0.9 × context window.
- Action: scan session entries; classify each as:
  - `user_message` / `jarvis_message` → keep last 6, summarize older
  - `tool_invocation` → keep envelope, summarize transcript
  - `deliverable_ref` → **never compress** (md path is canonical)
  - `react_log` → if associated AT call has a deliverable, drop
    entirely; otherwise summarize
- Output: replace summarized region with one
  `compression_summary` entry. Verbatim block (paths, URLs, code) is
  extracted and prepended (same as solver Stage 2.5).
- All deliverable .md files remain referenced by path — they live on
  disk, never inlined, never compressed.

File: `core/auto_compress.py` (new, ~200 LOC). Hook in jarvis main loop
before every LLM turn.

### 5.6 CMD coder + GUI unison
*"neither superior by default; whichever is invoked first is master,
the other becomes subordinate."*

Implementation in CMD:
- `cmd/core/role_arbiter.py` (new): tracks which mode entered first in
  the current session.
- When `cmd:code` is invoked first: subsequent `gui_task` calls run as
  subordinates (return clean deliverable to coder, coder integrates).
- Inverse for `cmd:gui` first.
- A "subordinate" call is just a normal subagent invocation — the
  master AT does the save/restore. The arbiter only governs which AT's
  context is the trunk.
- Session continuity contract still holds: subordinate runs in its own
  context, returns deliverable, master merges.

### 5.7 Seamless cross-tool calls
*"all tools can seamlessly call each other, can seamlessly use each's
context."*

- **`SubAgentInvoker.SUPPORTED_TARGETS`** must include every
  combination — already partial:
  ```
  cmd:code, cmd:gui, cmd:blue,
  swarm:math, swarm:engineer, swarm:deep_search,
  jarvis  (for re-entry / sub-conversation)
  ```
- Each invoker wires the session continuity contract identically.
- Context sharing: every AT can `read_context(key)` from the central
  board (`~/.agent_bin/central_context.md` mirrored from SQLite).
  This is the **horizontal** channel — independent of session
  snapshots, used for live cross-AT data passing.

### 5.8 One context file with sub-contexts that rebuild
*"context all in one file, agents make their own sub-context, then
rebuild back into the main."*

Already covered in §3. Concretely:
- Main: `jarvis_<session_id>.context`
- Sub: `<at_role>_<job_id>.context` (created by AT, archived after)
- Merge: deliverables appended to main; ReAct dropped; sub archived
  for forensics under `~/.agent_bin/sessions/archive/`.

---

## 6. The Rocket-Sim Acceptance Test

Verbatim from the spec — this is what a correct Jarvis must deliver.

```
user: "code me a physics-based rocket sim so I can make my own guided rockets"

jarvis (session abc-123):
  1. classify: project (multi-AT chain)
  2. plan DAG:
     W1: swarm:deep_search "rocket physics + Three.js patterns"
     W2: swarm:math       "RK4 with drag, guidance equations"
                           depends_on: W1
     W3: cmd:code         "scaffold ~/Rocket-sim/, write index.html"
                           depends_on: W1, W2

  3. execute:
     [SAVE abc-123_pre_W1.context]
     swarm:deep_search runs → Rocket-science-for-the-sim.md
     [RESTORE abc-123 + merge deliverable_ref]

     [SAVE abc-123_pre_W2.context]
     swarm:math runs → Rocket-math-burnout.md, Rocket-math-guidance.md
     [RESTORE abc-123 + merge]

     [SAVE abc-123_pre_W3.context]
     cmd:code runs → ~/Rocket-sim/, Project-guided-rocket.md
     [RESTORE abc-123 + merge]

  4. stitch final reply:
     - 1 paragraph summary
     - ## Files: ~/Rocket-sim/, Project-guided-rocket.md, 3 .md refs
     - ## Sources: pulled from deep_search deliverable

  5. session abc-123 ends with:
     - 1 user message
     - ~5 jarvis messages
     - 3 deliverable_refs
     - 0 ReAct loop entries  ← critical
```

Must pass: **`grep -c "react_log\|sub_thought\|tool_internal"
jarvis_abc-123.context` returns 0.**

---

## 7. Failure Modes & Recovery

| Symptom | Cause | Jarvis response |
|---|---|---|
| AT returns `success=false` | Specialist gave up | Surface error verbatim, ask user. **Do not retry.** |
| 504 timeout | Long-poll exceeded | Show partial deliverables if any; ask if user wants longer budget. |
| 429 from `/subagent` | MAX_CONCURRENT exceeded | Wait `retry_after`, retry once. |
| Empty deliverables | Job done but wrote nothing | Use `summary` only if useful; else treat as failure. |
| AT crashed mid-run | Process died | Sub-context still on disk; offer to resume from snapshot. |
| Mirror desync | CMD restarted mid-write | Don't auto-fix. Tell user to run `cmd rebuild_mirror`. |
| Auto-compression ate something useful | over-aggressive summarize | All deliverable .md still on disk; rewind via `restore_context(path)`. |

---

## 8. Cost & Latency Budget

- **<2 s of LLM thinking** → answer directly, no AT.
- **2 s – 30 s** → consider single AT call.
- **>30 s** → must stream (`/query_stream` SSE) so user sees progress.
- **>10 min** → confirm with user before launching.
- `extra.timeout_s` default 1800 s, hard cap 3600 s.
- Parallelize: every independent leaf in the DAG runs in
  `asyncio.gather`. Wall-clock = deepest path, not sum.

---

## 9. Model Routing

Defaults (post-3.16, with the new IQ4 quant):

```
SWARM_MODEL_DEFAULT       = batiai/qwen3.6-27b:iq4   (fits 2× GPU)
SWARM_DEEP_SEARCH_MODEL   = qwen3.6:35b-A3B-Grindlewalt   (when GPU fits)
SWARM_MODEL_SOLVER        = batiai/qwen3.6-27b:iq4
SWARM_MODEL_PLANNER       = batiai/qwen3.6-27b:iq4
SWARM_DIAGNOSTICIAN_MODEL = batiai/qwen3.6-27b:iq4
SWARM_MODEL_ENGINEER      = batiai/qwen3.6-27b:iq4
SWARM_DEFAULT_CHAT_MODEL  = batiai/qwen3.6-27b:iq4
SWARM_NUM_CTX             = 15360                   (single knob)
```

Jarvis itself does not pick models — it lets each AT run its configured
stack. If user asks for a faster/slower variant, set env via the `extra`
field rather than hardcoding.

---

## 10. Anti-Patterns (do not do)

1. **Inline ReAct transcripts in user-facing replies.** Reference
   sidechain paths or deliverable paths only.
2. **Compressing deliverables.** .md files are sacred — they're the
   horizontal data channel.
3. **Re-deriving values that are in the manifest.** If a value is
   already published via `read_context`, use it.
4. **Catching subagent errors silently.** Always surface to user.
5. **Restarting `ollama-swarm` mid-job.** Always check `:5002/status`.
6. **Mixing roles in one invocation.** `swarm:math` does math, period.
   For prose, follow with a write step.
7. **Inlining 50 KB of context in `task`.** Publish first, pass keys.
8. **Letting a sub-AT see its grandparent's ReAct.** The save/restore
   contract guarantees this — never bypass.

---

## 11. Minimum Viable Jarvis (v0)

Smallest implementation that satisfies the unification contract:

```
~/jarvis/
├── jarvis.py              # main loop, session manager, ~400 LOC
├── planner.py             # decompose → DAG, ~150 LOC
├── invoker.py             # SubAgentInvoker wrapper w/ save/restore, ~150 LOC
├── stitcher.py            # final-answer composer, ~120 LOC
├── auto_compress.py       # 90%-trigger compaction, ~200 LOC
├── manifest.py            # carry-forward dict, ~80 LOC
└── prompts/
    ├── planner.txt
    ├── stitcher.txt
    ├── compressor.txt
    └── system.txt
```

Reuse, do not duplicate:
- `cmd/core/subagent.py` — base SubAgentInvoker
- `swarm/core/agent_memory.py` — central context board client
- `swarm/core/sidechain.py` — JSONL trace writer
- `swarm/core/base_agent.py` — `save_context` / `restore_context` stubs
- `cmd/server.py /api/v1/context` — mirror-aware writer

---

## 12. Acceptance Criteria

A correct Jarvis MUST pass:

1. **Rocket-sim chain** (§6) — 3 ATs, 3 deliverables, 0 ReAct entries
   in `jarvis_<id>.context`.
2. **AT-to-AT cleanliness** — invoke `cmd:code` that internally calls
   `swarm:math`. After completion, neither session shows the other's
   ReAct.
3. **90% compression test** — run a 50K-token session; verify
   compression triggers, deliverable paths preserved byte-identical,
   stale ReAct gone.
4. **Master/subordinate switch** — invoke `cmd:gui` first, then
   `cmd:code` should run as subordinate; verify role_arbiter records
   correct master.
5. **Verbatim preservation** — every URL, fenced code block, and
   absolute path in any final reply matches a source byte-for-byte.
6. **Failure surface** — kill ollama-swarm mid-job; Jarvis surfaces
   504 verbatim, does not retry, offers to resume from snapshot.
7. **Memory persistence** — after Jarvis restart, `restore_session(id)`
   loads the full history minus all ReAct, ready to continue.
8. **No-numbers-in-prose** — when user requests `no_formulas`, Lock C
   strips formulas in deliverables, and Jarvis stitcher does not
   reintroduce them.

---

## 13. Build Order

Phased so each phase is independently shippable:

| Phase | Deliverable | Blocks on |
|---|---|---|
| P1 | `core/session_manager.py` + save/restore in BaseAgent override | – |
| P2 | `core/auto_compress.py` + 90% trigger hook | P1 |
| P3 | Deep search rewrite (35b-A3B model + parent_context_keys) | – |
| P4 | Solver markdown deliverable writer | – |
| P5 | Engineer tool plugins (Adafruit, Amazon, Octopart, Digi-Key) | – |
| P6 | Engineer circuit-gen module | P5 |
| P7 | CMD role_arbiter (master/subordinate switch) | – |
| P8 | `~/jarvis/` v0 — planner, invoker, stitcher, manifest | P1, P2 |
| P9 | Rocket-sim acceptance run | P1–P8 |

P3, P4, P5, P7 parallelize once P1+P2 land. P6 and P8 are independent
from each other.

---

*Document version: v0.2 — 2026-04-26.
 Companion to: INTEGRATION_CONTRACT.md, CLAUDE_HANDOFF.md.*
