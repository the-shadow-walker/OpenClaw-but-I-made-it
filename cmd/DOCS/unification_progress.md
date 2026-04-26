# Unification of Agents — Progress Doc

Status as of commits `e71ad40` (unification baseline) and `eabec17` (qwen3 thinking disabled).

**Companion docs:**
- [`INTEGRATION_CONTRACT.md`](INTEGRATION_CONTRACT.md) — the cross-agent API surface (memory paths, session-continuity protocol, tool naming) that both CMD and Swarm build to. Read this first if you're picking up swarm-side work.
- [`CLAUDE_COORDINATION.md`](CLAUDE_COORDINATION.md) — protocol for the two Claude instances (CMD-side + Swarm-side) to coordinate without stepping on each other. Includes the handoff-board convention.

This doc maps each goal/refinement in the original plan to **what's shipped**, **what's partial**, and **what's still open**, so future-me with a goldfish memory can pick up cleanly.

---

## Plan Goal Recap

> Combine swarm (engineer/math/research) + Jarvis (memory) + CMD (blue team, GUI, ReAct coding) into one stack. Each advanced tool (AT) needs:
> - Session continuity — save context before tool fires, merge clean result back
> - Smart model (`qwen3.6:35b-A3B-Grindlewalt`) with the problem's full context
> - Markdown deliverables, not just raw transcripts
> - Tool-to-tool delegation (CMD ↔ GUI peers; either can subordinate the other)
> - Single shared context file that sub-agents read/write
> - Strong auto-compression at ~90% so we don't lose work to context overflow

---

## What's Shipped

### 1. Bidirectional CMD ↔ GUI delegation (peer model)

**Files:** `cmd/core/subagent.py` (new), `cmd/core/react_tools.py`, `cmd/guiagent/gui_tools.py`

- `SubAgentInvoker` class — generic snapshot → run → merge primitive used by both directions.
- CMD's `gui_task` tool: CMD agent saves its session, fires GUI agent, GUI does its thing, GUI's ReAct trace goes to a sidechain file, only the clean result merges back into CMD's context.
- GUI's new `code_task` tool: mirrors the above in reverse — GUI subordinates CMD when GUI is the parent.
- `_ParentShim` in `gui_tools.py` so the GUI handler can re-use SubAgentInvoker without restructuring GUIAgent.
- Neither agent is "default superior". Whoever is invoked first owns the parent context; the other is the subordinate for that call.

**Maps to plan goals:** "CMD coding agent and GUI agent in perfect unison" + "all tools can seamlessly call each other, use each's context".

### 2. Central shared context file

**Files:** `cmd/core/react_memory.py`, `cmd/server.py`

- `~/.agent_bin/central_context.md` — single markdown mirror of the SQLite `shared_context` table.
- Auto-rendered on every `set_context` / `publish_context` call. Sectioned by key prefix (e.g. `## deploy`, `## rocket_sim`).
- New REST endpoints:
  - `POST /api/v1/context` — write
  - `GET /api/v1/context?prefix=…` — read
  - `DELETE /api/v1/context/:key` — clear
- Existing `publish_context` / `read_context` ReAct tools now write through the same code path, so the markdown file and SQLite stay in sync.

**Maps to plan goals:** "all in one file… agents make sub context and rebuild back into main context."

### 3. Session continuity primitive (snapshot/merge)

**Files:** `cmd/core/ollama_agent_core.py`, `cmd/core/subagent.py`

- `OllamaCommandAgent.save_context(label)` and `restore_context(path)` — serialize/restore pinned slots + files-created list to `~/.agent_bin/sessions/<label>.json`.
- SubAgentInvoker uses these around every cross-agent call: parent snapshots before delegating, subordinate runs in fresh context with only the relevant context keys passed in, parent merges the artifact list and a one-line summary back.
- Subordinate's full ReAct trace is written to `~/.agent_bin/sidechains/<id>.jsonl` and **stripped from the parent transcript** before resume — so parent never sees the cluttering inner loop.

**Maps to plan goals:** the entire "abc-123.context save → tool runs → results back into abc-123" workflow described in the plan, exactly as specified.

### 4. High-fidelity auto-compression at 90% context

**Files:** `cmd/core/ollama_agent_core.py`

- New Stage 2.5 in the staged compaction pipeline (between the 70% snip and 95% LLM compress).
- Threshold: `_HIFI_THRESHOLD = NUM_CTX * 0.90` (≈ 29.5K of 32K).
- Method: `_compress_high_fidelity()` — pure regex, no LLM call (~5ms vs ~20s for the LLM-driven Stage 3).
- Preserves verbatim:
  - Fenced code blocks (` ``` … ``` `)
  - Absolute file paths (e.g. `/home/Grindlewalt/projects/…/main.py`)
  - URLs
- Collapses everything else (stale ReAct deliberations, retry chatter, repeated tool errors) into one-line synopses. Caps: 40 preserved blocks, 30 synopsis lines.
- Keeps the original system prompt + last 3 messages verbatim regardless.

**Maps to plan goals:** "when 90% of context is used, just goes through everything and writes summaries of everything that has happened so far, but not compressing things like .md's, just talking, react loops that are stale."

### 5. Performance: qwen3 thinking disabled by default

**Files:** `cmd/core/ollama_agent_core.py`

- New class constant `THINK_DEFAULT = os.getenv("AGENT_THINK", "0") == "1"` (defaults off).
- `"think": self.THINK_DEFAULT` wired into `_call_model_oneshot`, `call_ollama`, and `call_ollama_react` request bodies.
- Confirmed live: ReAct iterations dropped from 3–5 min/iter to 27–60s/iter. No quality regression on agent code output observed yet.
- Re-enable per call: set `AGENT_THINK=1` in the systemd environment (or temporarily for an experiment).
- Modelfile route was a dead-end: `PARAMETER think false` rejected by Ollama 0.20.2 and `SYSTEM "/no_think"` ignored by qwen3.5 renderer. API-level parameter is the only working knob.

**Maps to plan goals:** infrastructure for "smart model with full context" — fast enough that we can afford bigger contexts and more chained calls.

### 6. GUI agent context symmetry

**Files:** `cmd/guiagent/gui_tools.py`

- New tools on the GUI side mirroring CMD:
  - `gui_publish_context` — writes to the same shared SQLite + central_context.md
  - `gui_read_context` — reads either a single key or a prefix list
- Both wired to `self.memory.set_context` / `get_context` / `list_context` (same `AgentMemory` instance CMD uses).
- Means: when CMD calls `gui_task` and writes a context key first, the GUI agent sees it immediately. Reverse direction works the same.

**Maps to plan goals:** "for the context system, agents each work off that one [file]."

---

## What's Partial

### Swarm integration (engineer / math / deep search)

**Status:** primitive exists, swarm-side not wired.

What's done:
- SubAgentInvoker accepts a `target` arg (`"cmd"`, `"gui"`, or `"swarm:<role>"`).
- Central context is reachable from anything that can hit the SQLite path or the REST endpoints.

What's still missing:
- Swarm subagents (math/engineer/deep_search) don't yet have a `_handle_*_task` registered in the swarm side — they have to be invoked via the swarm's own server (`swarm_api_server.py` on its own port).
- No CMD-side `math_task`, `engineer_task`, or `deep_search_task` tool yet. Adding them is mechanical (copy `_handle_gui_task`, swap target string and HTTP endpoint), but it hasn't been done.
- Deep search "writes a real .md report" requirement: not yet implemented. Current deep search returns a JSON blob, not a markdown deliverable.

### Engineer mode external APIs

**Status:** not started.

The plan calls for engineer mode to hit Adafruit, Amazon, etc. for component data (price/reviews/specs). None of that is wired. The swarm engineer agent currently only has its own tool registry; no external API integrations have been added in this round.

### Acceptance criteria for swarm deliverables

Swarm side has no equivalent of `AcceptanceCriteriaRunner`. Math/engineer/research output isn't checked for "did you actually write the markdown report". This is the swarm-side analog of the agent-hardening AC retry/soft-reverify work.

---

## What's Still Open

| Item | Plan quote | Owner |
|---|---|---|
| Deep search writes real `.md` reports with bigger-context smart model | "writes in md, does actually good research… stronger searching agents with bigger context" | swarm/search/ |
| Swarm-Solver works end-to-end with markdown output | "swarm-solver needs to actually work… write in-depth with markdown" | swarm/math/ |
| Engineer mode: circuit gen + Adafruit/Amazon APIs | "amazon, adafruit and any other sites api access… price, reviews… technical overview" | swarm/engineer/ |
| AT-to-AT delegation (CMD ↔ Swarm) with session continuity | "If command needs to make a website that's math heavy… asks for advanced math, swarm has full context of the past but session is saved, math is done, then cmd gets it back" | new bridge in cmd/core/ |
| Per-tool deliverable convention enforced | All ATs must produce a markdown artifact, not just a stdout blob | shared |
| Swarm's own session save/merge primitive | Mirror of SubAgentInvoker on the swarm side so swarm jobs can also delegate without context bloat | swarm/server/ |

---

## File Map (for quick orientation)

| Layer | File | What changed |
|---|---|---|
| Sub-agent primitive | `cmd/core/subagent.py` | NEW — SubAgentInvoker class |
| CMD ReAct tools | `cmd/core/react_tools.py` | + `gui_task`, `code_task` peer hooks |
| CMD core | `cmd/core/ollama_agent_core.py` | + `THINK_DEFAULT`, `_compress_high_fidelity`, save/restore_context, hifi pipeline wiring |
| GUI tools | `cmd/guiagent/gui_tools.py` | + `_handle_code_task`, `_handle_gui_publish_context`, `_handle_gui_read_context` |
| Memory | `cmd/core/react_memory.py` | + central_context.md mirror, set/get/list_context |
| Server | `cmd/server.py` | + `POST/GET/DELETE /api/v1/context` |

---

## Verification Receipts

- `e71ad40` deploy smoke test: POST/GET/DELETE `/api/v1/context` round-trip green; central_context.md auto-rendered with `## deploy` section.
- `eabec17` deploy smoke test: ReAct iter wall-clock measured 27–60s on bookmarks chain phase 3 builder, down from 3–5 min before the patch.
- High-fidelity compression: regex extractor smoke-tested on a 32K-line synthetic transcript; preserved 14/14 fenced code blocks and all 7 absolute paths; reduced total transcript tokens by ~62%.

---

## Test Run In Flight

Rocket simulator chain `edad521b-884c-4f73-a83c-69e228984ab5` is the first non-trivial integration test of:
- think:false speed
- gui_task end-of-chain verification
- workspace-aware AC

Outcome will be the next datapoint for what to fix in round 2.

Caveat: the chain came back as single-subtask fallback (`subtask_count: 1`, budget 100) instead of the 9-phase decomp the goal would normally produce. Likely TaskDecomposer race when the prior cancelled worker was still mid-iteration during submit. Logged as a separate diagnostic; not patching live per "this is just a test."
