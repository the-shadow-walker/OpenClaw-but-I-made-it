# Claude ⇄ Claude Coordination Protocol

**Two Claude instances, one repo, one user.** This doc keeps us out of each other's way and makes handoffs cheap. If you're a Claude Code instance reading this, **read it before touching files in the other side's domain.**

---

## 1. Domain Split

| Side | Local Mac path | Remote path | Owner |
|---|---|---|---|
| CMD | `/Users/grant/cmd/` | `/mnt/storage/NAS/Jarvis/cmd/` + `/mnt/storage/NAS/Jarvis/server.py` | "CMD Claude" |
| Swarm | `/Users/grant/swarm3/` (legacy editing) | `/mnt/storage/NAS/Jarvis/swarm/` | "Swarm Claude" |
| Shared | `/Users/grant/cmd/DOCS/INTEGRATION_CONTRACT.md` etc. | `/mnt/storage/NAS/Jarvis/INTEGRATION_CONTRACT.md` | both |

**Rule:** edit only your own side unless you have explicit handoff. If you need to change the contract, drop a proposal in the handoff board (§3) — don't unilaterally rewrite shared files.

---

## 2. Shared Truth Files (read these first every session)

In priority order:

1. `/mnt/storage/NAS/Jarvis/INTEGRATION_CONTRACT.md` — the API contract. Do not break it without coordination.
2. `/mnt/storage/NAS/Jarvis/cmd/DOCS/unification_progress.md` — what CMD has shipped against the master plan.
3. `/mnt/storage/NAS/Jarvis/CLAUDE_HANDOFF.md` — live status board (§3 below).
4. `/Users/grant/cmd/CLAUDE.md` and `/mnt/storage/NAS/Jarvis/swarm/CLAUDE.md` — per-side conventions.
5. `~/.claude/projects/-Users-grant-cmd/memory/MEMORY.md` — CMD Claude's persistent memory. Swarm Claude has its own.

Read order at session start: contract → handoff board → progress doc → memory. Five minutes total, saves hours of stepping on each other.

---

## 3. Handoff Board (`CLAUDE_HANDOFF.md`)

Both Claudes append-only to this single markdown file. Format:

```markdown
## 2026-04-26T13:05 — CMD Claude
- shipped: 90% hifi compression (commit e71ad40)
- need from swarm: AgentMemory mirror at swarm/core/agent_memory.py pointing at same SQLite path
- blocking: nothing
- next: wiring math_task tool

## 2026-04-26T13:40 — Swarm Claude
- shipped: AgentMemory wrapper at swarm/core/agent_memory.py
- need from cmd: confirm WAL mode is on for memory.db
- blocking: nothing
- next: subagent.py mirror
```

**Rules:**
- One stanza per session, timestamped. Timezone: server local.
- "shipped" = code on disk, deployed, smoke-tested.
- "need from <other>" = explicit ask. Other side picks it up next session.
- "blocking" = something stopping you. Be specific or it's noise.
- "next" = one-liner of intent so the other side knows what to expect.

Keep it append-only — don't delete old stanzas, just scroll. If the file gets too long, archive the top half to `CLAUDE_HANDOFF_archive_<date>.md` and link from the top.

---

## 4. Conflict Avoidance

### File-edit overlap

Risk: both Claudes edit the same file, last writer wins, work lost.

Mitigation:
- **Contract files** (`INTEGRATION_CONTRACT.md`): always commit before edit. If `git pull` shows a fresh contract commit you didn't make, **read it before editing**. Use the handoff board to flag intent: "I'm about to bump contract to v0.2 to add `engineer_task`."
- **Code files**: clearly owned by domain. CMD doesn't touch `swarm/`, swarm doesn't touch `cmd/`. The shared spots are only the contract docs and this file.
- **Server.py at root**: CMD-owned. If swarm needs a route there (e.g. `/api/v1/context` mirror), file the request via handoff and CMD adds it.

### Git workflow

- Both Claudes commit and push from `/mnt/storage/NAS/Jarvis/`.
- Pull before commit. If there's a conflict, do **not** force push. Resolve on remote, commit, push.
- Tag commits with side prefix: `cmd: …`, `swarm: …`, `contract: …`. Easier to scan history.

### Service restarts

- `ollama-cmd.service` is CMD's. Swarm Claude should not restart it without flagging.
- `ollama-swarm.service` is swarm's. CMD Claude should not restart it without flagging.
- Both share the underlying `ollama.service` (single Ollama instance). Restarting that affects everyone — coordinate via handoff first.

---

## 5. Live Coordination Channels

We can't message each other directly, so:

| Channel | Latency | Use for |
|---|---|---|
| `CLAUDE_HANDOFF.md` (git) | minutes (next pull) | All async coordination, shipped work, asks |
| `~/.agent_bin/central_context.md` | seconds (live) | Runtime state visible to *agents*, but Claudes can read it too |
| `~/.agent_bin/memory.db` `shared_context` table | seconds | Same, structured |
| Git commit history | persistent | What actually shipped, with diffs |

If you need real-time signal that the other side might be active right now, check `git log --since="1 hour ago"` and the handoff board.

---

## 6. The User's Role

The user (Grant) is the only one who can give cross-side instructions verbally. If the user tells one Claude to do something that depends on the other side, that Claude:

1. Checks the contract — is it already supported?
2. If not, files the ask in the handoff board with `need from <other>`.
3. Tells the user "filed in handoff, swarm Claude will pick it up next session" — don't try to do the other side's work.

Exception: trivial cross-side fixes (typo, one-line config). Make it, commit with `cross: <description>`, note in handoff.

---

## 7. First Sync (todo for next session of either Claude)

Initial handoff board needs to be created. Whichever Claude reads this first:

1. Create `/mnt/storage/NAS/Jarvis/CLAUDE_HANDOFF.md` with the header:
   ```markdown
   # Claude ⇄ Claude Handoff Board
   See [CLAUDE_COORDINATION.md](cmd/DOCS/CLAUDE_COORDINATION.md) for protocol.

   ## 2026-04-26T<time> — <CMD|Swarm> Claude (initial)
   - shipped: <whatever you just shipped>
   - need from <other>: <ask>
   - blocking: nothing
   - next: <intent>
   ```
2. Commit + push.
3. Other Claude picks it up on next session start.

---

## 8. Versioning

This protocol is `v0.1`. Either side can propose a bump via handoff. Mutual agreement (= other side acknowledges in their next stanza) makes it stick.
