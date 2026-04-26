# CLAUDE.md

Guidance for Claude Code when working in this repository.

## 📋 SESSION START — READ THESE FIRST, EVERY SESSION

This repo has a sibling Claude instance ("Swarm Claude") working on `/mnt/storage/NAS/Jarvis/swarm/`. We coordinate through three files. **Read all three at the start of every session before doing anything else** — they're the only way to know what the other side has shipped, what they need, and what's pending.

| Order | File | Why |
|---|---|---|
| 1 | `/mnt/storage/NAS/Jarvis/CLAUDE_HANDOFF.md` | Live append-only status board — most recent stanzas tell you what Swarm shipped/needs/blocking. Latest stanza is freshest signal. |
| 2 | `/mnt/storage/NAS/Jarvis/INTEGRATION_CONTRACT.md` | The cross-agent API spec — memory paths, session-continuity protocol, tool naming. Don't break it without coordinating. |
| 3 | `/mnt/storage/NAS/Jarvis/cmd/DOCS/CLAUDE_COORDINATION.md` | The protocol itself — domain split, conflict avoidance, who owns what. |

After reading, **before ending the session, append your own stanza** to `CLAUDE_HANDOFF.md` with the format:
```
## YYYY-MM-DDTHH:MM — CMD Claude
- shipped: <one-liner + commit hash>
- need from Swarm: <ask, or "nothing">
- blocking: <or "nothing">
- next: <intent>
```

If Swarm asked for something in their last stanza and you delivered it, mention it in your `shipped`. If you couldn't, say why in `blocking`. Append-only — never delete old stanzas.

## Key Rules
- **Always edit locally on Mac** (`/Users/grant/cmd/`), then SCP to remote. Never edit on remote directly.
- **Never push to `~/swarm3`** — swarm lives at `/mnt/storage/NAS/Jarvis/swarm/`
- **Git root** is `/mnt/storage/NAS/Jarvis/` — commit from there
- **Service name** is `ollama-cmd` (systemd unit: `ollama-cmd.service`)
- **Don't touch `swarm/` files** — that's Swarm Claude's domain. Cross-side asks go in the handoff board.

## ⛔ CRITICAL — NEVER DO THIS
**NEVER manually fix bugs in agent-generated test projects** (teamcollab, space-mission, or any other app the agent built as a chain test).
- These apps are **diagnostic artifacts** — their only purpose is to reveal agent weaknesses
- If the app is broken, that is the **signal**: fix the agent code, not the app
- Delete the broken app, improve the agent, rerun the chain
- Any time spent patching agent output directly is wasted effort and masks the real problem
- This rule applies even if the fix looks trivial — stop, delete, fix the agent instead

## Directory Structure

```
/mnt/storage/NAS/Jarvis/
  server.py          ← root-level entry point (root-owned, deploy via /tmp/)
  cmd/
    core/            ollama_agent_core.py, react_tools.py, react_memory.py
    chain/           task_chain.py
    blueteam/        blueteam_agent.py
    infra/           debug_logger.py, webhook_dispatcher.py, watchdog, sysinfo
    DOCS/            integration_guide.md, EVENT_MESH.md
    run_me.py        single client entry point
    README.md
  swarm/             AI agent swarm (moved from ~/swarm3)
  agent_inbox/       drop JSON files here for inbox watcher
  logs/
```

## Running the Service

```bash
# Service management
sudo systemctl status ollama-cmd
sudo systemctl restart ollama-cmd
sudo journalctl -u ollama-cmd -f

# Run directly (dev)
source .venv/bin/activate
python server.py
```

## SSH / Deployment

```bash
# SSH alias
mcssh   =   ssh -i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network' Grindlewalt@mcshell.atomos.network

SCP_OPTS="-i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network'"

# server.py — root-owned, must stage via /tmp/
scp $SCP_OPTS server.py Grindlewalt@mcshell.atomos.network:/tmp/
mcssh "sudo cp /tmp/server.py /mnt/storage/NAS/Jarvis/server.py && sudo cp /tmp/server.py /mnt/storage/NAS/Jarvis/cmd/server.py"

# All other cmd/ files — SCP directly to subpackage paths
scp $SCP_OPTS cmd/core/ollama_agent_core.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/core/
scp $SCP_OPTS cmd/core/react_tools.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/core/
scp $SCP_OPTS cmd/chain/task_chain.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/chain/
scp $SCP_OPTS cmd/blueteam/blueteam_agent.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/blueteam/
```

## Git Workflow

```bash
# Commit from the Jarvis root (git root is there)
mcssh "cd /mnt/storage/NAS/Jarvis && git add -A && git commit -m 'description' && git push"

# server.py is now user-owned after sudo cp — git add works without sudo
```

Git remote: `http://10.0.0.58:3000/Grindlewalt/Jarvis.git`

## Architecture Notes

- `server.py` adds `cmd/`, `cmd/core/`, `cmd/chain/`, `cmd/blueteam/`, `cmd/infra/` to sys.path — all flat imports inside modules continue working without changes
- Model: `qwen3.6:35b-Grindlewalt` (everything — ReAct loop + code generation)
- Service port: `5000`, no auth (local network)
- SENTINEL auto-watch: disabled at boot, triggered by cron at 3 AM daily
- Daily report: `~/.agent_bin/sentinel_report.md`, archives in `~/.agent_bin/sentinel_archive/`

## Agent Hardening (commit 191db6e — bulletproof tooling)

8 fixes shipped together to make the agent self-recover under load:

1. **`write_plan` is a real tool** (not just promised in prompts) — registered in TOOL_NAMES, TOOL_SCHEMAS, handler_map. Persists markdown plans with `- [ ]`/`- [x]` checkboxes to `~/.agent_bin/plans/{agent_id}_plan.md`. Updates pinned `📋 PLAN` slot. Stuck-loop exempt (idempotent updates).
2. **Tool advertisement = single source of truth** — `run_react` now builds the tool list dynamically from `TOOL_SCHEMAS.keys()`. New tools auto-advertise when whitelisted. `validate_arch`, `get_deps`, `write_plan` now visible to planners/builders.
3. **Dynamic role-tool block** — `_build_tool_restrictions_block(role_tools, first_action)` in task_chain.py replaces hardcoded "YOUR ONLY TOOLS" prose. Role identity drives the block, not `mt_type`.
4. **Bounded inference** — `_generate_file_content` `num_predict=8192 timeout=600`; `_generate_patch_replacement` same; `_generate_patch_search_and_replace` `num_predict=4096 timeout=180`. No more unbounded qwen3-thinking hangs. Class constants: `FILE_GEN_NUM_PREDICT`, `PATCH_NUM_PREDICT`.
5. **progress.md rate-limited + quiet** — `AGENT_PROGRESS_EVERY` env (default 3, 0=disable). Timeout bumped to 60s. Failures log a one-line warning, no curl-payload spam.
6. **Ollama retry loop** — `_call_model_oneshot` retries 2× on `TimeoutExpired`/connection errors with 1s,1.5s backoff. `call_ollama_react` retries 1× (long calls). Env: `AGENT_OLLAMA_RETRIES`.
7. **AC runner cwd-aware + soft re-verify** — `AcceptanceCriteriaRunner.run(cmd, cwd=...)` resolves relative paths against workspace. New `soft_re_verify(cmd, cwd)` parses `test -f X` tokens, checks disk + parses JSON; returns `{passed:True, soft_pass:True, recovered_via_disk:True}` if files exist on disk despite subprocess fail.
8. **Chain advancement: tight retry** — on AC fail: sleep 1.5s → re-run AC → soft_re_verify → only then fail. Env: `AGENT_AC_REVERIFY` (default 1). `chain.data["workspace"]` extracted from goal/decomposer and inherited by subtasks.

## Agent Unification (commit e71ad40 — tool fusion + central context)

Cross-agent delegation, shared memory, bidirectional CMD↔GUI subordination:

### New module: `cmd/core/subagent.py`
- `SubAgentInvoker(parent_agent, memory)` — uniform delegation primitive
- `run(target, task, max_iterations, context_keys, extra)` → `SubAgentResult`
- Targets: `gui`, `cmd`, `swarm:engineer`, `swarm:math`, `swarm:search`
- Pattern: snapshot parent (pinned state to `~/.agent_bin/sessions/`) → gather context from shared board → bridge into sub-task prompt → dispatch → merge `files_created` back into parent → pin `📥 SUBAGENT RESULT` slot for visibility across compactions
- ReAct trace dumped to `~/.agent_bin/sidechains/{sid}_{target}.jsonl` — never leaks into parent conversation
- Swarm calls: POST `{base}/query` (engineer/math), POST `{base}/api/search` (search). Default base `http://localhost:5002`. Config via `extra.swarm_url`.

### Tools added (CMD agent)
| Tool | Purpose |
|---|---|
| `save_context(label)` | Snapshot pinned slots + files + instruction to `~/.agent_bin/sessions/{ts}_{slug}.json` |
| `restore_context(path)` | Restore from a snapshot |
| `publish_context(key, value, ttl_hours, agent_id)` | Write to shared_context table (persists across runs/agents) |
| `read_context(key OR prefix, limit)` | Read from shared_context |
| `gui_task(task, max_iterations, context_keys)` | Delegate to GUI agent (CMD parent) |
| `swarm_task(mode, task, ...)` | Delegate to swarm (engineer/math/search) |

### GUI symmetry (`cmd/guiagent/gui_tools.py`)
| Tool | Purpose |
|---|---|
| `code_task(task, max_iterations, context_keys)` | Delegate to CMD agent (GUI parent) |
| `publish_context` / `read_context` | Same shared board, agent_id="gui" |

User's vision: *"Neither is superior to each other by default — when the coding agent is run then the gui agent is the subordinate, and vice versa."*

### Central context mirror — `~/.agent_bin/central_context.md`
- Auto-rendered from shared_context SQLite table on every `set_context` call
- Monkey-patched in `OllamaCommandAgent._wire_central_context_mirror()` (decoupled from react_memory.py)
- Grouped by `## agent_id` sections; capped at `CENTRAL_CONTEXT_MAX_ENTRIES = 200`
- Human-readable; sub-contexts from gui/swarm/cmd all merge into the one file

### REST API additions
- `GET /api/v1/context?prefix=...` (existed)
- `POST /api/v1/context` `{key, value, agent_id, ttl_hours}` (new)
- `DELETE /api/v1/context/<key>` (new)

### Stage 2.5 high-fidelity compression at 90% NUM_CTX
- New method: `OllamaCommandAgent._compress_high_fidelity()`
- Fires between Stage 2 (70% snip) and Stage 3 (~95% LLM compress)
- Regex-preserves verbatim: all ` ``` fenced blocks ``` `, abs file paths, URLs, head + last 3 messages
- Collapses middle history to one-line role-tagged synopses
- **No LLM call** — ~5ms vs ~20s for Stage 3
- Idempotent (running twice yields same result)
- Caps: 40 preserved blocks, 30 synopsis lines

### Snapshot/sidechain paths
- `~/.agent_bin/sessions/{ts}_{slug}.json` — full pinned state JSON
- `~/.agent_bin/sidechains/{sid}_{target}.jsonl` — sub-agent ReAct trace (one JSON per line)
- `~/.agent_bin/central_context.md` — human-readable mirror
- `~/.agent_bin/plans/{agent_id}_plan.md` — write_plan output
