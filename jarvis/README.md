# Jarvis

Personal AI assistant. File-first memory, snapshot/merge delegation to specialist services
(CMD on :5000 for shell/code/GUI; Swarm on :5002 for math/engineer/deep-search), and a
three-gate Dreaming consolidation pass that promotes durable facts from daily notes into
long-term `MEMORY.md`.

**Spec:** `docs/BUILD_SPEC.md` (v1.1) is the binding source of truth for every architectural
decision. Read it before changing anything material.

## Status

Phase **P0 — scaffolding**. Repo skeleton, config loader, deployable systemd unit (disabled).
No memory yet, no LLM client yet, no FastAPI server yet. See `docs/BUILD_SPEC.md` §18 for the
full phased build plan.

## Layout

See `docs/BUILD_SPEC.md` §4 for the full tree. Top-level:

```
jarvis/                   # package root + project root (= /mnt/storage/NAS/Jarvis/jarvis/)
├── pyproject.toml        # deps, ruff, pytest config
├── README.md             # this file
├── jarvis.service        # systemd unit (deploy to /etc/systemd/system/)
├── run.py                # CLI entry — `python -m jarvis.run`
├── config.py             # YAML config loader (~/.config/jarvis/config.yaml)
├── docs/BUILD_SPEC.md    # spec v1.1
├── core/                 # conversation, prompt, router, invoker, compaction, arbiter, tools
├── memory/               # files, chunker, index, search, embeddings, watcher
├── dreaming/             # candidate_db, light/rem/deep sleep, score, cli
├── clients/              # ollama, cmd, swarm, tts
├── adapters/             # cli, http, telegram, slack
├── workers/              # heartbeat, mirror_curator, archiver
├── tests/                # unit, integration, acceptance
└── workspace/            # gitignored — runtime memory state
```

## Dependency manager

**`uv`** is the chosen tool. Install with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the venv and install dev deps:

```bash
cd /Users/grant/Jarvis-Mk3
uv venv jarvis/.venv --python 3.12
uv pip install --python jarvis/.venv/bin/python -e jarvis[dev]
```

(On the server the venv lives at `/mnt/storage/NAS/Jarvis/.venv/` — already present, Python
3.14.4. `uv pip install -e jarvis` from `/mnt/storage/NAS/Jarvis/` registers the package
in-place.)

## Local dev loop

```bash
cd /Users/grant/Jarvis-Mk3
ruff check jarvis
pytest jarvis/tests
python -m jarvis.run                # P0: prints "jarvis stub — phase P0" and exits 0
```

## Deploy (Mac → server)

```bash
SCP_OPTS="-i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network'"

# Normal package files
scp $SCP_OPTS jarvis/<path>/<file> Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/jarvis/<path>/

# Root-owned files (jarvis.service, server.py once it exists)
scp $SCP_OPTS jarvis/jarvis.service Grindlewalt@mcshell.atomos.network:/tmp/
mcssh "sudo cp /tmp/jarvis.service /etc/systemd/system/jarvis.service && sudo systemctl daemon-reload"

# Commit + push (always from server)
mcssh "cd /mnt/storage/NAS/Jarvis && git add jarvis/<path> && git commit -m 'jarvis: <msg>' && git push"
```

**Never edit on the remote directly.** Edit on Mac, SCP, commit + push from the server. Tag
all jarvis commits with `jarvis: ...`.

## Service control (post-P5 once enabled)

```bash
mcssh "sudo systemctl status jarvis"
mcssh "sudo systemctl restart jarvis"
mcssh "sudo journalctl -u jarvis -f"
```

The unit is **deployed but disabled** during P0–P4. Enable in P5 when the FastAPI server lands.

## Architectural invariants (BUILD_SPEC §2)

Non-negotiable. Every PR review checks against this list:

1. **File-first memory.** Markdown in `workspace/` is the source of truth; SQLite is a
   rebuildable cache.
2. **ReAct never crosses a layer boundary.** Specialists run in their own context; only
   result envelopes return.
3. **Auto-flush before compaction.** Silent agentic turn extracts durable facts before truncation.
4. **Three-gate Dreaming promotion.** Score ≥ 0.8 AND recall_count ≥ 3 AND unique_query_count ≥ 3.
5. **Compaction floor.** Code blocks, absolute paths, URLs, system prompt, last 6 turns
   survive verbatim.

Push back on the user before deviating from any of these.

## Embedding model lock (BUILD_SPEC §6.3)

Default embedding provider is **Ollama `nomic-embed-text` (768d)**. Swapping to a different
provider (e.g. OpenAI `text-embedding-3-small` at 1536d) **requires reindex** because the
`chunks_vec` virtual table is dimension-locked. Don't panic when search quality dips during
the transition window — that's the documented graceful-degradation path (BM25 fallback for
unembedded rows).
