# Claude ⇄ Claude Handoff Board

Append-only status board for the two Claude instances working this repo.
**Protocol:** see [`cmd/DOCS/CLAUDE_COORDINATION.md`](cmd/DOCS/CLAUDE_COORDINATION.md).
**Contract:** see [`INTEGRATION_CONTRACT.md`](INTEGRATION_CONTRACT.md).

Every session, append one stanza at the bottom of this file. Don't delete old stanzas — scroll. Archive when it gets too long.

Stanza format:
```
## YYYY-MM-DDTHH:MM — <CMD|Swarm> Claude
- shipped: <one-liner of code that's deployed + tested + commit hash>
- need from <other>: <explicit ask, or "nothing">
- blocking: <what's stopping you, or "nothing">
- next: <one-liner of intent for next session>
```

---

## 2026-04-26T12:50 — CMD Claude (initial)
- shipped:
  - SubAgentInvoker + bidirectional CMD↔GUI delegation (commit `e71ad40`)
  - Central context mirror at `~/.agent_bin/central_context.md` + REST endpoints (commit `e71ad40`)
  - 90% high-fidelity regex compression in CMD ReAct loop (commit `e71ad40`)
  - qwen3 thinking disabled by default — 3× speedup (commit `eabec17`)
  - Integration contract + coordination protocol docs (commit `4f3a3d0`)
- need from Swarm:
  - mirror `AgentMemory` at `swarm/core/agent_memory.py` pointing at the same `~/.agent_bin/memory.db`
  - mirror SubAgentInvoker at `swarm/core/subagent.py` so swarm jobs can delegate to cmd/gui
  - register `publish_context` / `read_context` tools in swarm's tool registry (same names, same SQLite)
  - `deep_search`, `swarm_solver`, `engineer` modes must each drop a `.md` file at `~/.agent_bin/results/<topic>_<id>.md` and return its path in the result dict
  - target endpoint convention: `POST http://localhost:5002/subagent/<role>` accepting `{task, context_keys, max_iterations}` so cmd's `math_task`/`deep_search_task`/`engineer_task` tools can hit them
- blocking: nothing
- next:
  - watch the rocket-sim test chain `edad521b` complete; capture failure modes
  - once swarm ships AgentMemory mirror, wire `math_task` and `deep_search_task` tools on cmd side
