# Jarvis Workspace

This directory holds Jarvis's runtime memory state. **Per-deployment, gitignored.**
See `../docs/BUILD_SPEC.md` §3.1 for the full layout and loading rules.

Bootstrapped in **P1**. Do not commit anything here except `.gitignore` and this README.

Expected contents once bootstrapped:

```
workspace/
├── MEMORY.md          # curated long-term facts (loaded every DM)
├── USER.md            # user-modeling facts (always loaded)
├── SOUL.md            # personality / voice (system-prompt-injected)
├── AGENTS.md          # delegation rules
├── TOOLS.md           # tool docs (auto-generated)
├── HEARTBEAT.md       # optional proactive checklist
├── DREAMS.md          # human-readable consolidation diary (NEVER a promotion source)
├── projects/<slug>.md # one file per active project
├── memory/YYYY-MM-DD.md          # daily logs (append-only)
├── memory/.dreams/candidates.sqlite  # Dream candidate staging
├── conversations/<conv_id>.jsonl     # per-conversation transcripts
└── .index/memory.sqlite              # derived index (FTS5 + sqlite-vec) — disposable
```

If you delete `.index/memory.sqlite`, search must still work after a rebuild from the
Markdown files alone (BUILD_SPEC §2 invariant 1, acceptance test §16 second test).
