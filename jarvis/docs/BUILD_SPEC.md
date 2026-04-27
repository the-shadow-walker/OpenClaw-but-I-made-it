# Jarvis — Full Build Specification

**Audience:** the engineering agent building Jarvis from scratch.
**Authority:** this document is the single source of truth for the build. Where it conflicts with `INTEGRATION_CONTRACT.md` on cross-agent API, the contract wins; everything else is binding here.
**Constraint:** assume zero prior context. Every architectural decision is explained in this document. Do not consult the archived `Jarvis-2026-04-26/` codebase — it is reference-only.

---

## 0. What you are building, in one paragraph

A personal AI assistant called Jarvis that runs as a long-lived FastAPI daemon on `arch01` (port 5003). The user chats with it through a Mac terminal client, a web UI, and messaging adapters. Jarvis owns three things and only three things: **(1)** the user-facing conversation, **(2)** durable personal memory in the form of plain Markdown files plus a derived SQLite index, **(3)** orchestration of two specialist services it does *not* own — CMD (port 5000) for shell/coding/GUI and Swarm (port 5002) for math/engineering/deep research. When Jarvis delegates to either of those, it snapshots itself first, lets the specialist run in its own context, and merges only the clean Markdown deliverable back into the conversation. The hundreds of ReAct loop iterations the specialist generates never touch Jarvis's transcript. Memory is **file-first**: every fact Jarvis "remembers" lives as readable, editable Markdown on disk. The SQLite index is disposable and rebuilt from those files. A nightly background process called **Dreaming** consolidates short-term notes into long-term memory with three quality gates and full rollback. Auto-compaction runs an LLM-driven flush before truncating context so chat-only rules ("never deploy on Fridays") survive long conversations.

---

## 1. Operating environment

### Hosts and paths

| What | Path |
|---|---|
| Local edit (Mac) | `/Users/grant/cmd/jarvis/` |
| Remote deploy root | `/mnt/storage/NAS/Jarvis/jarvis/` |
| Git root | `/mnt/storage/NAS/Jarvis/.git/` |
| Gitea remote | `http://10.0.0.58:3000/Grindlewalt/Jarvis.git` |
| Jarvis service port | `5003` |
| systemd unit | `/etc/systemd/system/jarvis.service` |
| Python venv | `/mnt/storage/NAS/Jarvis/.venv/` |
| Jarvis personal workspace | `/mnt/storage/NAS/Jarvis/jarvis/workspace/` |
| Jarvis SQLite index | `jarvis/workspace/.index/memory.sqlite` (regenerable) |
| Shared cross-agent board | `~/.agent_bin/` (= `/home/Grindlewalt/.agent_bin/`) |
| Shared SQLite (read/write via REST) | `~/.agent_bin/memory.db` |
| Shared markdown mirror | `~/.agent_bin/central_context.md` (Jarvis owns this) |
| Snapshot store | `~/.agent_bin/sessions/jarvis_<conv_id>_<label>_<ts>.context` |
| Sidechain logs (read-only to Jarvis) | `~/.agent_bin/sidechains/<job_id>_<target>.jsonl` |
| Specialist deliverables | `~/.agent_bin/results/<topic>_<id>.md` |

### Sibling services Jarvis talks to

| Service | URL | Purpose |
|---|---|---|
| Ollama | `http://localhost:11434` | Local LLMs (chat, embeddings, dreaming) |
| CMD | `http://10.0.0.58:5000` | Shell exec, file I/O, ReAct coding, GUI agent, blue-team |
| Swarm | `http://10.0.0.58:5002` | Math solver, engineer mode, deep research |
| Piper TTS | local binary | Text-to-speech for the Mac client |

Jarvis **never** edits files in `cmd/` or `swarm/` directories. Cross-agent asks go through `CLAUDE_HANDOFF.md`.

### SSH and deploy

```bash
# alias 'mcssh' resolves to:
ssh -i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network' Grindlewalt@mcshell.atomos.network

SCP_OPTS="-i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network'"

# Deploy a normal file
scp $SCP_OPTS jarvis/core/foo.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/jarvis/core/

# Deploy a root-owned file (server.py, jarvis.service)
scp $SCP_OPTS server.py Grindlewalt@mcshell.atomos.network:/tmp/
mcssh "sudo cp /tmp/server.py /mnt/storage/NAS/Jarvis/jarvis/server.py"

# Commit + push from the server
mcssh "cd /mnt/storage/NAS/Jarvis && git add jarvis/path/to/file && git commit -m 'jarvis: <description>' && git push"

# Service control
mcssh "sudo systemctl restart jarvis"
mcssh "sudo journalctl -u jarvis -f"
```

Always edit on Mac, SCP, then commit + push from the server. **Never edit on the remote directly.** Tag commits with `jarvis: ...`.

### CLAUDE_HANDOFF.md

Append-only coordination board at `/mnt/storage/NAS/Jarvis/CLAUDE_HANDOFF.md`. Read the bottom (newest) at session start. Append one stanza at session end:

```markdown
## 2026-04-26T15:30 — Jarvis Claude
- shipped: <what's deployed + smoke-tested + commit hash>
- need from CMD: <explicit ask, or "nothing">
- need from Swarm: <explicit ask, or "nothing">
- blocking: <specific blocker, or "nothing">
- next: <intent for next session>
```

Append-only. Commit and push so the other Claudes can see it. `shipped` means deployed and smoke-tested, not "wrote a plan."

---

## 2. Architectural invariants (non-negotiable)

These five rules are the spine of the design. Violating any of them defeats the whole point. Every PR review checks against this list.

1. **File-first memory.** The Markdown files in `workspace/` are the source of truth. The SQLite index is a derived cache, rebuildable from filesystem alone. If you delete the index, every search must still work after a rebuild from the workspace files.
2. **ReAct never crosses a layer boundary.** Jarvis delegates to CMD or Swarm; the specialist runs in its own process with its own context. Jarvis sees the result envelope (summary + deliverable paths + context keys), never the execution log. The specialist's full transcript goes to a sidechain JSONL on disk, which Jarvis can open for debugging only.
3. **Auto-flush before compaction.** When the conversation approaches the model's context limit, Jarvis runs a silent agentic turn that asks the model to extract any durable facts from the about-to-be-truncated history and write them to the daily log. *Then* it truncates. Skipping this loses chat-only rules.
4. **Three-gate Dreaming promotion.** A short-term note never gets promoted to long-term `MEMORY.md` unless it clears all three gates: composite score ≥ 0.8, recall count ≥ 3, unique-query count ≥ 3. Before writing, Jarvis re-reads the source range and skips if it has drifted.
5. **Compaction floor (binding via INTEGRATION_CONTRACT §6).** Never compress: fenced code blocks, absolute file paths (anything starting with `/`), URLs, the system prompt, the last 3 user/assistant messages. These survive verbatim through every compaction pass. *Note: §8 applies a stricter floor (last 6 turns) — this is a deliberate Jarvis-side superset of the 3-message contract minimum, not drift. Reviewers: do not "fix" this back to 3.*

A violation of any of these is a real bug, not a tradeoff. Push back on the user before deviating.

---

## 3. The two-layer memory model

Jarvis has two distinct memory stores. Do not conflate them.

### 3.1 Personal memory (Jarvis-owned)

Lives in `jarvis/workspace/`. This is where the user's identity, preferences, projects, and conversational history are stored. Jarvis is the only writer and the only reader. **Plaintext Markdown files** make this readable and editable by the user with any text editor.

```
jarvis/workspace/
├── MEMORY.md                  Curated long-term facts. Loaded at start of every DM conversation.
├── USER.md                    User-modeling facts (name, locations, work, preferences). Always loaded.
├── SOUL.md                    Personality, communication style, voice. Injected into system prompt.
├── AGENTS.md                  Rules for delegation (when to use CMD vs Swarm vs answer directly).
├── TOOLS.md                   Tool documentation (auto-generated, regenerated on tool registry change).
├── HEARTBEAT.md               Optional checklist for the autonomous heartbeat loop (proactive tasks).
├── DREAMS.md                  Human-readable diary of consolidation passes. Never a promotion source.
├── projects/                  One file per active project: <slug>.md
│   ├── rocket-sim.md
│   └── jarvis-rebuild.md
├── memory/
│   ├── 2026-04-26.md          Today's append-only daily log (running notes, observations).
│   ├── 2026-04-25.md
│   └── .dreams/               Machine-facing staging for Dreaming. Not human-readable.
│       ├── candidates.sqlite  Dream candidate table with scoring signals.
│       └── runs/<run_id>.json Audit trail of each Dreaming pass for rollback.
├── conversations/
│   └── <conv_id>.jsonl        One JSONL per conversation. The transcript.
└── .index/
    └── memory.sqlite          Derived index. Disposable. FTS5 + sqlite-vec.
```

**Loading rules at conversation start:**
- DM: `USER.md` + `MEMORY.md` + today's daily log + yesterday's daily log + `SOUL.md` + `AGENTS.md` + `TOOLS.md` (system-prompt-injected).
- Group chat: `USER.md` only (no personal long-term memory leaks into multi-party context). `MEMORY.md` is **never** loaded in groups.
- Heartbeat: `HEARTBEAT.md` + `USER.md` + recent daily log + `AGENTS.md`.

**Active project loading:** when the user mentions a project by name (slug match) or the conversation router classifies the turn as project-scoped, the matching `projects/<slug>.md` is appended to context for that turn only.

### 3.2 Cross-agent context board (shared, Jarvis curates)

This is the integration spine across Jarvis, CMD, Swarm. It already exists per `INTEGRATION_CONTRACT`. Jarvis does not own the SQLite — it lives at `~/.agent_bin/memory.db` and any agent can read/write it. Jarvis **does** own the human-readable mirror at `~/.agent_bin/central_context.md`.

| Layer | Path | Owner |
|---|---|---|
| SQLite primary | `~/.agent_bin/memory.db` (table `shared_context`) | shared |
| Markdown mirror | `~/.agent_bin/central_context.md` | **Jarvis (going forward)** |
| Sessions / snapshots dir | `~/.agent_bin/sessions/` | shared |
| Sidechains dir | `~/.agent_bin/sidechains/` | shared (Jarvis read-only post-delegation) |
| Plans dir | `~/.agent_bin/plans/` | shared |
| Results dir | `~/.agent_bin/results/` | shared |

Schema for the SQLite (do not modify; CMD owns the table):
```sql
CREATE TABLE shared_context (
  key TEXT PRIMARY KEY,
  value TEXT,
  agent_id TEXT,
  created_at REAL,
  expires_at REAL
);
```

**Key naming convention (binding):** `<scope>_<topic>_<detail>` where scope ∈ `{chain, gui, cmd, swarm, jarvis, project, session, user, convo}`. No spaces, slashes, colons. TTL default 24h; set explicitly for long-lived artifacts.

**Jarvis writes to the board** when handing off context to a specialist (publish before dispatch) and when surfacing in-flight job state for the user. **Jarvis reads the board** in two cases: (a) pulling context the user asked about in chat, (b) the curator polling to rebuild the markdown mirror.

**Mirror takeover plan:** CMD currently rewrites `central_context.md` on every shared write. Jarvis takes over by setting `AGENT_CENTRAL_MIRROR_OWNER=jarvis` on `ollama-cmd.service` (this requires a 5-line patch from CMD Claude — file via handoff). Until the patch lands, leave the existing mirror alone and read from it; do not write the mirror yet.

---

## 4. Repository layout

Single Python package, no monorepo. Use `uv` or `poetry` for dep management — pick one and document.

```
jarvis/                              # = /mnt/storage/NAS/Jarvis/jarvis/
├── README.md
├── pyproject.toml                   # deps, ruff config, pytest config
├── jarvis.service                   # systemd unit (deploy to /etc/systemd/system/)
├── run.py                           # entry point: imports server.app, runs uvicorn
├── server.py                        # FastAPI app (chat endpoint, tools API, TTS, health)
├── config.py                        # ALL constants and config loading
├── core/
│   ├── __init__.py
│   ├── conversation.py              # Conversation lifecycle (renamed from "session")
│   ├── prompt.py                    # System prompt assembly
│   ├── router.py                    # Classifies turns: direct / cmd-delegate / swarm-delegate / chain
│   ├── invoker.py                   # Snapshot-delegate-merge wrapper around CMD/Swarm calls
│   ├── compaction.py                # Auto-compaction with auto-flush
│   ├── arbiter.py                   # Master/subordinate switch tracking
│   └── tools.py                     # Tool definitions exposed to the LLM
├── memory/
│   ├── __init__.py
│   ├── files.py                     # Atomic Markdown read/write
│   ├── chunker.py                   # Markdown → chunks (heading-aware, 400tok/80overlap)
│   ├── index.py                     # SQLite schema, FTS5, sqlite-vec, reconciliation
│   ├── search.py                    # Hybrid search (BM25 + vector, MMR, decay)
│   ├── embeddings.py                # Provider abstraction (ollama, openai, local-gguf)
│   ├── watcher.py                   # chokidar-equivalent (watchdog) → reconcile
│   ├── tool_search.py               # memory_search tool implementation
│   └── tool_get.py                  # memory_get tool implementation
├── dreaming/
│   ├── __init__.py
│   ├── candidate_db.py              # Dream candidate SQLite schema and CRUD
│   ├── light_sleep.py               # Ingestion + dedup + redaction
│   ├── rem_sleep.py                 # Reflection + theme extraction + signal updates
│   ├── deep_sleep.py                # Promotion gate + rehydration + MEMORY.md writes
│   ├── score.py                     # Six-signal weighted scoring
│   └── cli.py                       # `jarvis dreaming on|off|status`, promote, rem-backfill
├── clients/
│   ├── __init__.py
│   ├── ollama.py                    # Local LLM streaming client
│   ├── cmd.py                       # CMD REST client with envelope unwrapping
│   ├── swarm.py                     # Swarm REST client (engineer/math/deep_search)
│   └── tts.py                       # Piper TTS wrapper
├── adapters/
│   ├── __init__.py
│   ├── cli.py                       # `jarvis chat` terminal client (Mac satellite)
│   ├── http.py                      # POST /api/chat NDJSON streaming
│   ├── telegram.py                  # (optional, P12)
│   └── slack.py                     # (optional, P12)
├── workers/
│   ├── __init__.py
│   ├── heartbeat.py                 # Cron loop walking HEARTBEAT.md
│   ├── mirror_curator.py            # Polls shared SQLite, rebuilds central_context.md
│   └── archiver.py                  # Daily log → archive after 90 days
├── tests/
│   ├── unit/
│   ├── integration/
│   └── acceptance/
│       ├── test_rocket_sim.py       # The DAG acceptance test from §16
│       └── test_index_rebuild.py    # Delete index, search must still work
└── workspace/                       # gitignored, per-deployment
    ├── .gitignore
    └── README.md                    # Tells humans this is regenerated
```

`workspace/` is gitignored; the live deployment's workspace persists across deploys via the systemd `WorkingDirectory` not being wiped on update.

---

## 5. SQLite schema (full DDL)

The Jarvis-owned index at `jarvis/workspace/.index/memory.sqlite`. Run `PRAGMA journal_mode=WAL` on first connect.

**`sqlite-vec` extension load (mandatory, every connection).** The `vec0` virtual table will not exist until the extension is loaded. Install with `pip install sqlite-vec` (it ships the loadable shared object alongside the Python package). On every new SQLite connection, before any DDL or query touches `chunks_vec`:

```python
import sqlite3, sqlite_vec
conn = sqlite3.connect(db_path)
conn.enable_load_extension(True)
sqlite_vec.load(conn)         # equivalent to conn.load_extension(sqlite_vec.loadable_path())
conn.enable_load_extension(False)
conn.execute("PRAGMA journal_mode=WAL")
```

Wrap connection creation in a single `get_connection()` factory in `memory/index.py` so this incantation lives in exactly one place. Forgetting this is the most common P3 footgun.

```sql
-- Track every indexed file with content hash for change detection
CREATE TABLE IF NOT EXISTS files (
    path           TEXT PRIMARY KEY,
    content_hash   TEXT NOT NULL,
    modified_at    INTEGER NOT NULL,
    evergreen      INTEGER NOT NULL DEFAULT 0,    -- 1 for MEMORY.md / USER.md / SOUL.md
    file_kind      TEXT NOT NULL                  -- 'memory'|'user'|'soul'|'agents'|'tools'|'daily'|'project'|'dreams'
);

CREATE TABLE IF NOT EXISTS chunks (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    content          TEXT NOT NULL,
    start_line       INTEGER NOT NULL,
    end_line         INTEGER NOT NULL,
    token_count      INTEGER NOT NULL,
    heading_path     TEXT,                         -- "## Projects > ### Rocket-sim"
    created_at       INTEGER NOT NULL,
    embedding_model  TEXT                          -- NULL until embedded; e.g. "ollama:nomic-embed-text:768"
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_model ON chunks(embedding_model);

-- FTS5 mirror for BM25
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Vector index (sqlite-vec virtual table)
-- Dimensions are per-provider; default ollama nomic-embed-text = 768.
-- If you swap to openai text-embedding-3-small, this becomes 1536 and you must drop+recreate.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id   INTEGER PRIMARY KEY,
    embedding  FLOAT[768]
);

-- Embedding cache (LRU-evicted at 50k rows)
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash         TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    embedding         BLOB NOT NULL,
    accessed_at       INTEGER NOT NULL,
    PRIMARY KEY (text_hash, model_fingerprint)
);
CREATE INDEX IF NOT EXISTS idx_emb_cache_accessed ON embedding_cache(accessed_at);

-- Conversation transcripts (also written as JSONL files; this table is for cross-conversation search)
CREATE TABLE IF NOT EXISTS conversations (
    id             TEXT PRIMARY KEY,
    started_at     INTEGER NOT NULL,
    ended_at       INTEGER,
    channel_kind   TEXT NOT NULL,                  -- 'dm'|'group'|'cli'|'heartbeat'
    channel_id     TEXT,
    slug           TEXT,                           -- LLM-generated when conversation closes
    summary        TEXT,                           -- 2-3 sentence summary, generated at close
    transcript_path TEXT NOT NULL                   -- relative to workspace
);
CREATE INDEX IF NOT EXISTS idx_conv_started ON conversations(started_at DESC);

-- Recent search queries — used by REM Sleep to compute query diversity
CREATE TABLE IF NOT EXISTS search_queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT NOT NULL,
    query_hash  TEXT NOT NULL,
    queried_at  INTEGER NOT NULL,
    result_count INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_query_hash ON search_queries(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_time ON search_queries(queried_at DESC);
```

A second SQLite file at `jarvis/workspace/memory/.dreams/candidates.sqlite` for the Dreaming candidate table — kept separate so wiping/rebuilding Dreaming state doesn't touch the main index:

```sql
CREATE TABLE IF NOT EXISTS dream_candidates (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash          TEXT UNIQUE NOT NULL,
    content               TEXT NOT NULL,
    source_file           TEXT NOT NULL,
    source_line_start     INTEGER NOT NULL,
    source_line_end       INTEGER NOT NULL,
    recall_count          INTEGER NOT NULL DEFAULT 0,
    unique_query_count    INTEGER NOT NULL DEFAULT 0,
    query_set             TEXT NOT NULL DEFAULT '[]',  -- JSON array of query hashes
    relevance_avg         REAL NOT NULL DEFAULT 0.0,   -- avg search-score when matched
    consolidation_signal  REAL NOT NULL DEFAULT 0.0,   -- bumped by REM if candidate appears in a theme cluster
    conceptual_richness   REAL NOT NULL DEFAULT 0.0,   -- length+entity-density heuristic, computed at ingest
    first_seen_at         INTEGER NOT NULL,
    last_seen_at          INTEGER NOT NULL,
    promoted              INTEGER NOT NULL DEFAULT 0,
    promoted_at           INTEGER,
    promoted_in_run       TEXT                          -- run_id for rollback
);
CREATE INDEX IF NOT EXISTS idx_dream_promoted ON dream_candidates(promoted, last_seen_at);
CREATE INDEX IF NOT EXISTS idx_dream_run ON dream_candidates(promoted_in_run);

CREATE TABLE IF NOT EXISTS dream_runs (
    run_id          TEXT PRIMARY KEY,
    started_at      INTEGER NOT NULL,
    finished_at     INTEGER,
    light_count     INTEGER DEFAULT 0,
    rem_count       INTEGER DEFAULT 0,
    deep_promoted   INTEGER DEFAULT 0,
    rolled_back     INTEGER NOT NULL DEFAULT 0,
    rolled_back_at  INTEGER,
    notes           TEXT
);
```

---

## 6. Memory pipeline

### 6.1 File watcher → reconciliation

Use `watchdog` (Python's chokidar-equivalent). On startup, walk `workspace/` and reconcile every file. Then watch for changes.

```python
# memory/watcher.py
class WorkspaceWatcher:
    def __init__(self, workspace: Path, indexer: Indexer):
        self.workspace = workspace
        self.indexer = indexer
        self.observer = Observer()
        # Debounce: editor saves can fire multiple events
        self._pending: dict[Path, float] = {}
        self._debounce_ms = 250

    def start(self):
        # Initial reconcile
        for f in self._scan_workspace():
            self.indexer.reconcile(f)
        # Watch
        handler = _Handler(self._on_event)
        self.observer.schedule(handler, str(self.workspace), recursive=True)
        self.observer.start()
        # Debounce drainer thread
        threading.Thread(target=self._drain, daemon=True).start()

    def _on_event(self, path: Path, kind: str):
        if path.suffix != ".md":
            return
        if any(p.startswith(".") for p in path.relative_to(self.workspace).parts):
            return  # skip .index, .dreams, .git
        self._pending[path] = time.time()

    def _drain(self):
        while True:
            now = time.time()
            ready = [(p, t) for p, t in self._pending.items() if (now - t) * 1000 > self._debounce_ms]
            for p, _ in ready:
                self._pending.pop(p, None)
                if p.exists():
                    self.indexer.reconcile(p)
                else:
                    self.indexer.remove_file(p)
            time.sleep(0.1)
```

Reconciliation diffs by content hash. If the hash matches `files.content_hash`, no work. If different, delete all `chunks` for that file (cascade handles FTS and vec via triggers + foreign-key) and re-chunk + re-embed.

### 6.2 Chunking

Markdown chunking is heading-aware. Within a heading section, if the section exceeds 400 tokens, split with 80-token overlap. Preserve `start_line` / `end_line` so `memory_get` can do precise range reads.

```python
# memory/chunker.py
@dataclass
class Chunk:
    content: str
    start_line: int
    end_line: int
    heading_path: str  # "## Projects > ### Rocket-sim"
    token_count: int

def chunk_markdown(text: str, target_tokens: int = 400, overlap_tokens: int = 80) -> list[Chunk]:
    sections = split_by_heading(text)  # returns [(heading_path, body, line_range), ...]
    chunks = []
    for heading_path, body, (line_start, line_end) in sections:
        toks = count_tokens(body)
        if toks <= target_tokens:
            chunks.append(Chunk(body, line_start, line_end, heading_path, toks))
        else:
            chunks.extend(_sliding_window(body, line_start, heading_path, target_tokens, overlap_tokens))
    return chunks
```

`evergreen=1` is set on chunks from `MEMORY.md`, `USER.md`, `SOUL.md`. They are exempt from temporal decay during ranking.

### 6.3 Embeddings

Provider abstraction. Default to **Ollama with `nomic-embed-text` (768d)** because it runs locally and matches the local-LLM ethos. Fallbacks in order: OpenAI `text-embedding-3-small` (1536d, requires `OPENAI_API_KEY`), then degraded mode (BM25-only, vector search returns empty).

**Critical: dimension changes require re-indexing.** If the active fingerprint changes (e.g. user switches from Ollama-768 to OpenAI-1536), all rows in `chunks_vec` are now wrong. The change handler:

1. Detect via comparing `embedding_model` of newest chunks vs. current pipeline fingerprint.
2. Schedule a background reindex job that re-embeds chunks lazily as queries hit them — OR rebuilds the vec table fully if `--full-reindex` is requested.
3. Until reindex completes, search degrades gracefully: query against rows matching the current fingerprint, fall back to BM25 for unmatched rows.

Document this clearly in the README so the user doesn't panic when search quality dips temporarily after a config change.

### 6.4 Hybrid search

```python
# memory/search.py
@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    content: str
    start_line: int
    end_line: int
    heading_path: str
    score: float
    score_components: dict  # {bm25, vector, decay} for debugging

@dataclass
class SearchOptions:
    k: int = 10
    candidate_multiplier: int = 4
    vector_weight: float = 0.7
    text_weight: float = 0.3
    mmr_lambda: float = 0.7
    half_life_days: float = 30.0
    file_kinds: list[str] | None = None        # filter, e.g. ["memory", "user", "daily"]
    include_conversations: bool = False         # search past JSONL transcripts too
```

**Score normalization is the silent killer of hybrid search.** Cosine similarity is bounded; BM25 is not, and its scale depends on corpus size and term distribution. Always min-max normalize within each result set before weighted fusion. Do not use raw scores.

**MMR (Maximal Marginal Relevance)** prevents the top-k from being slight variants of the same chunk. With λ=0.7, results are 70% weighted toward query relevance and 30% toward dissimilarity from already-selected chunks. The vectors used in MMR come from a `vectors_by_id` dict pre-fetched during the search call — no re-embedding cost.

### 6.5 Tools exposed to the LLM

Two memory tools are exposed in the chat-loop tool list: `memory_search`, `memory_get`. A third tool `memory_write` handles `[REMEMBER: ...]`-style writes (where=daily default; where=memory only on explicit user "remember this"). Action-tag style from old Jarvis is **not** brought forward — proper tool calls only.

### Delegation tool

Routing to a specialist is also a tool call. The LLM emits `delegate` with a target enum; the runtime intercepts, runs the snapshot/dispatch/merge dance, and returns the merged result envelope as the tool result. Targets: `cmd:quick`, `cmd:react`, `cmd:chain`, `cmd:gui`, `cmd:blue`, `swarm:math`, `swarm:engineer`, `swarm:deep_search`. Single `delegate` tool with target enum is deliberate — gives the LLM a single decision point.

**Action-tag style (`[REMEMBER: ...]`, `[DEEP_SEARCH: ...]`) from the old Jarvis is intentionally not brought forward.** All actions are tool calls.

---

## 7. Conversation lifecycle

A **conversation** is a chat thread with a `conv_id`. Its transcript is one JSONL file at `workspace/conversations/<conv_id>.jsonl`. Each line is one event: `user_message`, `system_prompt`, `assistant_message`, `tool_call`, `tool_result`, `delegation_snapshot`, `delegation_envelope`, `compaction`.

### Lifecycle

- **Create:** new conversation on cold start, on `/new` command, on scheduled reset, or on idle timeout.
- **Reset schedules:** daily at 04:00 for DMs, 02:00 for groups; idle timeout 120 minutes; per-channel overrides allowed in config.
- **Close:** when reset triggers, generate slug + 2-3 sentence summary via LLM, write to `conversations` table, leave the JSONL on disk for cross-conversation search.
- **Resume:** if the same channel sends a message within the idle window, append to the existing JSONL.

The transcript is what the LLM sees as `messages[]` on each turn — except for the first system message, which is reassembled fresh from `USER.md` + `MEMORY.md` + daily logs every turn (so manual edits to memory take effect immediately).

---

## 8. Auto-compaction with auto-flush

**This is the single most important non-memory feature.** Without it, long conversations silently drop chat-only rules and the user thinks Jarvis is broken.

### Turns vs messages — definition lock

Used consistently throughout the spec:
- A **message** is one entry in the LLM's `messages[]` array — a `user`, `assistant`, `tool_call`, or `tool_result`.
- A **turn** is one user message plus the assistant's complete response, *including* any tool_call / tool_result pairs the assistant spawned. So one turn can be 1 user + 1 assistant + N×(tool_call, tool_result) + 1 final assistant — usually 3–8 messages.

Config keys say "turns" (`keep_recent_turns: 6`); internal compaction logic operates on turn boundaries (never split a tool_call from its tool_result during truncation). Token counters operate on messages.

### Trigger

When token estimate of `messages[]` ≥ 90% of the model's context window (configurable: `compaction.trigger_pct: 0.90`).

### Algorithm

1. **PHASE 1: Auto-flush (silent agentic turn).** With everything except last 6 turns, prompt the model: "Identify any durable facts, preferences, decisions, or rules the user stated that are not already in MEMORY.md. For each one, call memory_write with where='daily'. If nothing qualifies, respond 'NOTHING_TO_FLUSH'." Only `MEMORY_WRITE` tool available. Tool calls execute normally.
2. **PHASE 2: Truncate** with `keep_recent_turns=6`, summarize older tool results, reserve 2000 tokens floor. Append a `compaction` event to the transcript.

### Compaction floor (binding via INTEGRATION_CONTRACT §6)

Inside `compact_history`, **never compress**:
- Fenced code blocks (` ``` … ``` `).
- Absolute paths matching `^/`.
- URLs matching `https?://`.
- The system prompt.
- The last 6 user/assistant messages (the contract minimum is 3 — we use 6 for headroom).

These survive verbatim. Tool results from older turns can be summarized to "tool_call to X returned 2KB; summary: …" but their critical artifacts (paths, URLs) remain referenced.

### Test this rigorously

Acceptance test: write a 50-turn synthetic conversation where turn 3 contains "never deploy on Fridays — this is a hard rule from my last gig." Run compaction. Assert that after compaction, the rule is in either (a) the daily log via auto-flush, or (b) preserved verbatim in the truncated `messages[]`. If neither, the test fails and you fix compaction.

---

## 9. Dreaming

Opt-in, off by default. Enable with `jarvis dreaming on`. Runs as a cron job (default 03:00) — one full sweep of three sequential phases. The whole pass should complete in under 5 minutes for a workspace with a year of activity.

### 9.1 Light Sleep — ingestion

Reads recent daily logs and conversation transcripts (default lookback: 7 days). Extracts snippet-level candidates (sentence or paragraph), redacts PII (emails, phones, ID numbers), dedupes via Jaccard ≥ 0.9, stages into `dream_candidates`. `INSERT OR IGNORE` on `content_hash` makes re-ingestion idempotent. `_richness()` is a simple heuristic: token count × entity-count multiplier (proper nouns, numbers, specific dates), normalized to [0, 1].

### 9.2 REM Sleep — reflection

Updates the candidate scoring signals. **Never writes to MEMORY.md.** Writes a managed `## REM Sleep` block to `DREAMS.md` with extracted themes (human-readable diary, separate from operational state).

1. Replay recent search queries against candidates → bump `recall_count`, `unique_query_count`, `relevance_avg`.
2. Cluster candidates into themes (cheap K-means or HDBSCAN over candidate embeddings, `min_cluster_size=3`).
3. Bump `consolidation_signal` for candidates appearing in dense clusters.
4. Append managed block to `DREAMS.md` (overwrites previous REM block).

### 9.3 Deep Sleep — promotion

The gate. Score each unpromoted candidate (six-signal weighted: relevance 0.30, frequency 0.24, query_diversity 0.15, recency 0.15, consolidation 0.10, conceptual_richness 0.06). Three threshold gates: `score ≥ 0.8 AND recall_count ≥ 3 AND unique_query_count ≥ 3`. **Rehydrate** before writing — re-read source range, skip if Jaccard vs current `content` < 0.85. Hard age cap: 30 days.

### 9.4 MEMORY.md write format

When promoted, append under a managed `## Auto-promoted` section with provenance:

```markdown
## Auto-promoted

- The user prefers TypeScript over JavaScript for new projects. *(promoted 2026-04-26 from memory/2026-04-19.md L42-L45, run dream-2026-04-26-03-00, score 0.84)*
```

The provenance comment is what enables rollback.

### 9.5 CLI

```
jarvis dreaming on            Enable cron + future runs.
jarvis dreaming off           Disable cron. Existing candidates remain.
jarvis dreaming status        Show: enabled?, last run, candidate count, last promotion count.
jarvis dreaming run           Force a manual run now.
jarvis dreaming promote       Dry-run Deep Sleep. Print what WOULD be promoted. No writes.
jarvis dreaming promote --apply   Actually promote.
jarvis dreaming rem-backfill --days 30   Replay old daily notes through Light+REM (idempotent).
jarvis dreaming rollback <run_id>   Remove all promotions from a specific run from MEMORY.md.
```

### 9.6 The two files separation

- `DREAMS.md` is a **human diary**. Themes, summaries, "this week the user worked a lot on rocket sims." Never read by the agent for operational decisions.
- `MEMORY.md` is **operational state**. The agent reads this on every DM conversation start. Only Deep Sleep (or the user via `memory_write where=memory`) writes here.

The agent never reads `DREAMS.md` automatically. Confusing the two — letting the diary become a promotion source — is exactly the failure mode that makes naive consolidation pollute long-term memory with noise.

---

## 10. Session continuity protocol — Jarvis as parent

Every delegation follows the same shape, regardless of target.

```
Jarvis conversation (conv_id = abc-123)
│
├── [SNAPSHOT] save jarvis_abc-123_pre_W1_<ts>.context
│   Publish context keys: project_rocket_brief, user_pref_python_style
│   Dispatch: swarm:deep_search "rocket physics + Three.js patterns"
│       └── Swarm runs in its own context (500 ReAct turns, 40 searches)
│           Returns envelope: {success, summary, deliverables, context_keys_written, sidechain_path}
│
├── [MERGE] Append to abc-123 transcript: delegation_envelope event.
│   Read deliverable .md from disk, inject content into next turn's context.
│   ReAct trace stays in sidechain JSONL — Jarvis NEVER reads it into the transcript.
│
└── jarvis: "Built ~/Rocket-sim/. Files: ... Sources: see Rocket-science-for-the-sim.md."
```

### 10.1 Snapshot

Snapshot before every delegation, regardless of target. Saves `messages`, `active_project_files`, `system_prompt_components`, label, timestamp to `~/.agent_bin/sessions/jarvis_<conv_id>_<label>_<ts>.context`. If everything downstream fails, `restore_from_snapshot(path)` to continue or retry.

### 10.2 Context-key publish before dispatch

Do not pass the entire conversation transcript to the specialist. Pass **keys**. Render a brief, publish `project_<slug>_brief` to the shared board, return the list of keys. Specialist reads them via `read_context`.

### 10.3 Dispatch

Always async (REST job pattern). Even quick CMD calls go through the polling client — uniformity beats latency optimizations. (Quick CMD shell — `/api/v1/quick` — is reserved for the special-case stateless one-shot.)

### 10.4 Merge

The result envelope is the canonical, contract-defined shape: `ResultEnvelope(success, summary, deliverables, context_keys_written, sidechain_path, error)`. Merge appends `delegation_envelope` event, reads deliverable `.md` files from disk, pulls written context keys, stashes `DeliverableRef` for next-turn injection. **The execution_log field on the specialist's status response is intentionally not consumed.**

### 10.5 Sidechain access

If the user asks "what did Swarm actually do during that?", Jarvis can open `env.sidechain_path` and summarize on demand. One-shot read returns a summary to the user; file contents do not enter the transcript.

---

## 11. CMD client (full implementation)

### 11.0 Reality check before coding

Two parts of the CMD REST surface do not match what this spec needs, as of 2026-04-26. **Do not write the client from this section alone — verify against `cmd/server.py` first.**

1. **CMD does not return contract-shaped envelopes today.** `GET /api/v1/jobs/<id>` returns `{status, success, react_trace, execution_log, files_created, ...}` — the fields `summary`, `deliverables`, `context_keys_written`, and `sidechain_path` (for top-level jobs) do not exist. This spec's `_envelope()` helper synthesizes the envelope from what's available; see §11.4.
2. **`context_keys` is not accepted at the REST submit boundary today.** Only `SubAgentInvoker` (in-process) pre-stages context keys. Until CMD accepts it server-side, the field is ignored — Jarvis must publish the keys to the shared board *and* mention the key names in the `instruction` so the agent knows to call `read_context` on them.
3. **Submit URL is `POST /api/v1/jobs`, not `/api/v1/execute`.** The chain submit body is `{goal, max_iterations, model?}`.

### 11.1 The client

`CMDClient` exposes: `publish/read_context/delete_context` (board), `quick` (one-shot shell), `execute` (ReAct job, async polling), `chain` (multi-phase, longer poll). `_envelope()` synthesizes contract envelope from CMD's actual response (best-effort `deliverables` from `~/.agent_bin/results/*.md`, fallback to scan of `files_created`; `summary` from `finish_summary` or last assistant content; `sidechain_path` synthesized from `~/.agent_bin/sidechains/<job_id>_cmd.jsonl` if exists).

### Tool whitelisting — when to use it

```python
READ_ONLY = ["read_file", "memory_lookup", "read_context", "finish"]
CODER     = ["read_file", "create_file", "patch_file", "write_plan", "execute_command", "finish"]
SHELLER   = ["execute_command", "read_file", "manage_server", "finish"]
FULL      = None
```

Default to `CODER` for "build/edit/refactor" tasks and `READ_ONLY` for "summarize/analyze/inspect" tasks.

### Concurrency limits

CMD has no backpressure. Use a Jarvis-side `asyncio.Semaphore(max_concurrent=2)`.

---

## 12. Swarm client

### 12.0 Reality check before coding

**Verify against `swarm/server.py` before writing this client.** The route shape (`POST /subagent/<role>`), sync vs async, polling URL pattern have not been fully confirmed. Follows CMD client pattern; same envelope-synthesis caveat applies.

### 12.1 The client

`SwarmClient.dispatch(target, task, context_keys, max_iterations, timeout_s, ...)` posts to `/subagent/<role>` with body `{task, agent_id, extra: {max_iterations, timeout_s, parent_context_keys}}`. Polls `/subagent/<role>/result/<job_id>` until `complete|failed`. `_envelope()` same shape as CMD — Swarm follows the contract.

`parent_context_keys` is how Swarm reads context the parent published. Pass relevant keys; do not inline the full conversation.

---

## 13. The router — when does Jarvis delegate?

The router classifies each user turn into one of:

| Class | Meaning | Action |
|---|---|---|
| `direct` | Conversational, factual recall, simple Q&A, memory ops | Answer directly with chat model + memory tools |
| `cmd_quick` | One-shot shell question | `cmd_client.quick(...)` |
| `cmd_react` | Coding/file/shell task with multiple steps | Snapshot → `cmd_client.execute(...)` → merge |
| `cmd_chain` | Multi-phase project | Snapshot → `cmd_client.chain(...)` → merge |
| `swarm_math` | Equations, ODEs, derivations | Snapshot → `swarm_client.dispatch("swarm:math", ...)` → merge |
| `swarm_engineer` | BOM, schematic, datasheet | Snapshot → `swarm_client.dispatch("swarm:engineer", ...)` → merge |
| `swarm_research` | "Research X" | Snapshot → `swarm_client.dispatch("swarm:deep_search", ...)` → merge |
| `multi_phase` | Mixes the above (rocket sim) | Build a DAG, dispatch in dependency order |

Start rule-based with delegation hints. **The router is NOT an action tag.** Keywords are hints. The actual decision is made by the LLM calling the `delegate` tool — the runtime intercepts and routes.

---

## 14. Master/subordinate arbitration

CMD has two modes that can each invoke the other: `cmd:code` and `cmd:gui`. Whichever Jarvis invokes first becomes the "master" for that conversation; the other runs as a subordinate. `RoleArbiter` tracks `_master_per_conv: dict[conv_id, "code" | "gui"]`. Arbiter resets on conversation close.

Jarvis sets `master_mode` in the dispatch body when invoking the subordinate — CMD reads this and runs with reduced autonomy.

---

## 15. The mirror curator

Jarvis takes ownership of `~/.agent_bin/central_context.md`. Once CMD ships the env-flag check, set `AGENT_CENTRAL_MIRROR_OWNER=jarvis` on `ollama-cmd.service` and Jarvis becomes the sole writer.

### Strategy: poll, render, atomic-write

`MirrorCurator` polls the shared SQLite (read-only mode) every 5s. If `max(created_at)` increased, render a structured Markdown grouped by purpose (active conversations, active projects, in-flight jobs, user context, recent handoffs), atomic-write via `.tmp` + replace.

### Curation rules

- Skip entries with `expires_at - created_at < 3600` (ephemera).
- For values > 2KB, render a 200-char excerpt + `(see SQLite key for full content)`.
- Always render full keys for `*_brief` and `*_result` entries — those are the integration handoffs.
- Atomic write only.

### Migration

Until CMD ships the env-flag check, don't write the mirror. Just read from CMD's existing version.

---

## 16. The rocket-sim acceptance test (binding)

The single end-to-end test that proves the architecture works. User: "build me a physics-based rocket sim so I can make my own guided rockets."

**Expected flow:**
1. Router classifies as `multi_phase`.
2. Planner produces DAG: W1 (`swarm:deep_search`), W2 (`swarm:math`, deps: W1), W3 (`cmd:code chain`, deps: W1+W2).
3. Execute: snapshot → dispatch → merge for each leaf.
4. Final assistant turn: 1-paragraph summary, lists files, cites sources.

**Assertions:**
- A. Conversation JSONL contains 3 `delegation_snapshot` + 3 `delegation_envelope`, ZERO entries with `react_log`/`sub_thought`/`tool_internal`.
- B. All four deliverable .md paths exist on disk and are non-empty.
- C. Final assistant message contains the absolute paths verbatim.
- D. Jarvis token usage during the entire conversation stays under 50K tokens.

A second mandatory test (`test_index_rebuild.py`): populate workspace with 50 facts, run reconciliation, delete index, restart, verify all queries return identical (or near-identical, allowing for MMR-tie nondeterminism) results. Proves the index is truly disposable.

---

## 17. Configuration

Single YAML at `~/.config/jarvis/config.yaml` (override via `JARVIS_CONFIG`):

```yaml
server:
  host: 0.0.0.0
  port: 5003
  workers: 1

paths:
  workspace: /mnt/storage/NAS/Jarvis/jarvis/workspace
  shared_board: ~/.agent_bin/

llm:
  # Default: qwen2.5:3b (~2 GB) — small enough to coexist with CMD's resident model
  # with minimal additional CPU offload. Heavy lifting is delegated.
  chat_model: qwen2.5:3b
  fast_model: qwen2.5:3b
  ollama_host: http://localhost:11434
  ollama_keep_alive: 30m
  tokenizer: qwen-native             # or "approximation"

embeddings:
  providers:
    - kind: ollama
      model: nomic-embed-text
      dimensions: 768
    - kind: openai
      model: text-embedding-3-small
      dimensions: 1536
      api_key_env: OPENAI_API_KEY
  cache:
    max_rows: 50000

search:
  hybrid:
    vector_weight: 0.7
    text_weight: 0.3
    candidate_multiplier: 4
  mmr:
    lambda: 0.7
  decay:
    half_life_days: 30

conversation:
  reset:
    daily_at: "04:00"
    group_daily_at: "02:00"
    idle_minutes: 120
  compaction:
    trigger_pct: 0.90
    keep_recent_turns: 6
    reserve_tokens_floor: 2000
    auto_flush: true

dreaming:
  enabled: false
  schedule: "0 3 * * *"
  light_sleep:
    lookback_days: 7
    dedup_jaccard: 0.9
    min_snippet_chars: 20
  rem_sleep:
    lookback_days: 7
    cluster_min_size: 3
  deep_sleep:
    weights:
      relevance: 0.30
      frequency: 0.24
      query_diversity: 0.15
      recency: 0.15
      consolidation: 0.10
      conceptual_richness: 0.06
    gates:
      min_score: 0.80
      min_recall_count: 3
      min_unique_queries: 3
    recency_half_life_days: 14
    max_age_days: 30

orchestration:
  cmd:
    base: http://10.0.0.58:5000
    max_concurrent: 2
    quick_timeout_s: 15
    react_max_wait_s: 1800
    chain_max_wait_s: 7200
  swarm:
    base: http://10.0.0.58:5002
    max_concurrent: 2
    dispatch_max_wait_s: 1800

mirror:
  central_context_md: ~/.agent_bin/central_context.md
  poll_interval_s: 5.0
  enabled: false              # flip to true after CMD ships env-flag check

heartbeat:
  enabled: false
  interval_minutes: 30
  checklist_path: workspace/HEARTBEAT.md
```

The config schema is also the deliverable spec — when you finish the build, this YAML must round-trip through your config loader without losing fidelity.

### 17.1 Runtime dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | HTTP surface |
| `pydantic` v2 | Config and tool-arg validation |
| `sqlite-vec` | The `vec0` extension. **Loaded per-connection** — see §5. |
| `watchdog` | Workspace file-watcher |
| `httpx` (or `requests`) | CMD/Swarm/Ollama clients |
| `transformers` | Qwen tokenizer (only if `tokenizer: qwen-native`) |
| `python-frontmatter` | YAML frontmatter parsing on Markdown files |
| `markdown-it-py` | Markdown chunking by heading |
| `pyyaml` | Config |
| `apscheduler` | Cron for Dreaming + conversation resets |
| `pytest`, `pytest-asyncio`, `ruff` | Dev |

`sqlite-vec` is the install that bites — verify before P3:

```bash
python -c "import sqlite3, sqlite_vec; c = sqlite3.connect(':memory:'); c.enable_load_extension(True); sqlite_vec.load(c); print(c.execute('SELECT vec_version()').fetchone())"
```

---

## 18. Phased build plan

Each phase is independently shippable. After each phase, smoke-test, deploy, post a handoff stanza, and stop.

- **P0** Scaffolding (½ day) — repo, pyproject, config loader, systemd unit (disabled), CLI stub.
- **P1** File-first primitives (1–2 days) — atomic Markdown I/O, workspace bootstrap, system-prompt assembly, stub LLM client.
- **P2** Chunking + SQLite + FTS5 (1–2 days) — heading-aware chunker, schema, BM25.
- **P3** Embeddings + sqlite-vec + hybrid search (2–3 days) — Ollama nomic-embed-text, MMR, decay, search-query logging.
- **P4** File watcher (1 day) — debounce, atomic reconcile.
- **P5** Conversation + tool loop + chat endpoint (3–4 days) — `Conversation`, JSONL, lifecycle, `memory_*` tools, `POST /api/chat` NDJSON streaming, CLI client, systemd enabled.
- **P6** Auto-compaction with auto-flush (2 days) — token estimation, auto-flush, binding floor, "never deploy on Fridays" test.
- **P7** CMD client + delegation (3–4 days) — full client, snapshot/dispatch/merge, `delegate` tool, semaphore, whitelist presets.
- **P8** Swarm client + multi-phase planner (3–4 days) — DAG planner, topo-sort, parallel leaves; rocket-sim acceptance test.
- **P9** Mirror curator (1 day; coordinate with CMD) — poll, render, atomic-write; flip after CMD env-flag patch.
- **P10** Master/subordinate arbiter (1 day).
- **P11–P13** Dreaming Light/REM/Deep (2+2+3 days) — opt-in, six-signal scoring, three-gate, rehydrate, rollback.
- **P14** Polish (open-ended) — Telegram/Slack adapters, heartbeat, TTS, web UI, backfill, archiver.

**Stop and re-evaluate after P9.** That's the minimum viable Jarvis. Dreaming and the rest are gravy.

---

## 19. Anti-patterns — do not do these

1. Inlining ReAct transcripts in the user-facing reply.
2. Compressing deliverable .md content during compaction.
3. Reading `execution_log` from a CMD/Swarm response into the transcript.
4. Re-embedding chunks on every search.
5. Direct SQLite writes to `~/.agent_bin/memory.db` and expecting the mirror to update.
6. Reading `DREAMS.md` for operational decisions.
7. Auto-retrying failed CMD/Swarm jobs more than once.
8. Letting one specialist's ReAct flow into another specialist's context.
9. Inlining 50KB of context in the `task` field. Publish to the board, pass keys.
10. Hardcoding ports, model names, or URLs anywhere except `config.py`.
11. Naming things "session" without disambiguating. Use `conversation` for the chat thread, `snapshot` for the pre-delegation save.
12. Promoting the entire daily log to MEMORY.md "to be safe."

---

## 20. Security model

- **Private use, single-user.** No multi-tenant concerns.
- **The Mac client connects via Cloudflare Access tunnel** — auth handled there.
- **Memory files contain personal info.** `chmod 700` on `workspace/`. Do not commit `workspace/` to git.
- **Tool injection via specialist deliverables.** Specialists never have direct access to Jarvis's tools. They publish to the shared board; Jarvis decides whether to act. For destructive actions, require explicit user confirmation in the next turn.
- **No shell-exec from Jarvis.** Shell-exec is CMD's job, sandboxed there.
- **Encryption at rest:** out of scope for v1.

---

## 21. Open questions and conflicts to resolve with the user

1. **Dreaming default:** off. Confirm before enabling on the live deployment.
2. **Daily log archiving:** files older than 90 days → `workspace/memory/archive/YYYY/`.
3. **Group-chat memory boundary:** `MEMORY.md` is intentionally never loaded in groups. **`memory_search` in groups is filtered to exclude `file_kinds=["memory"]`** — silently dropped at the search layer. Confirm this is the desired privacy model.
4. **Heartbeat:** disabled by default.
5. **Embedding model lock:** swapping requires reindex. Document in README.
6. **Web UI:** out of scope for P0–P9.
7. **The action-tag-style protocol from old Jarvis (`[REMEMBER: ...]`)** is dropped in favor of proper tool calls. Confirm break is acceptable.
8. **Conversation IDs:** UUID v4 by default.
9. **The mirror takeover requires CMD Claude to ship a 5-line env-flag patch.**
10. **Cross-side dependencies before P7 (CMD) and P8 (Swarm).** Three handoffs filed *before* P0 starts:
    - **CMD: envelope shaping at job-status boundary.** `GET /api/v1/jobs/<id>` and `GET /api/v1/chains/<id>` should return contract envelope.
    - **CMD: `context_keys` first-class at REST submit.** `POST /api/v1/jobs` should accept `context_keys: list[str]`.
    - **CMD: `AGENT_CENTRAL_MIRROR_OWNER` env-flag check** in `_wire_central_context_mirror`. 5-line patch.
    - **Swarm: verify (or land) envelope-shaped responses** at `/subagent/<role>/result/<job_id>`.

    None block P0–P6. P7 partially built against synthesis. P9 strictly blocks on env-flag patch.

---

## 22. Glossary

| Term | Meaning |
|---|---|
| **AT** | Advanced Tool — a specialist service Jarvis delegates to (CMD or Swarm). |
| **Conversation** | A chat thread with a `conv_id`. The user-facing transcript. |
| **Snapshot** | A saved copy of conversation state taken before delegation, used for rollback. |
| **Sidechain** | A specialist's full ReAct execution log, written to JSONL by the specialist, never read by Jarvis except for forensics. |
| **Deliverable** | A Markdown file (or directory) produced by a specialist, dropped at `~/.agent_bin/results/<topic>_<id>.md`, that Jarvis reads back into the conversation. |
| **Result envelope** | The contract-defined response shape: `{success, summary, deliverables, context_keys_written, sidechain_path, error}`. |
| **Context key** | A short string identifier for a value in the shared SQLite board. |
| **Master / subordinate** | Within CMD, whichever of `cmd:code` / `cmd:gui` is invoked first per conversation is master. |
| **Light / REM / Deep Sleep** | The three sequential phases of a Dreaming pass. Light = ingest, REM = reflect (no MEMORY.md writes), Deep = promote with three gates. |
| **Three-gate promotion** | A candidate is promoted to MEMORY.md only if score ≥ 0.8 AND recall_count ≥ 3 AND unique_query_count ≥ 3. |
| **Rehydration** | Re-reading a candidate's source range before promotion; if drifted, skip. |
| **Auto-flush** | Silent agentic turn before compaction that extracts durable facts into the daily log. |
| **Compaction floor** | Content types that survive compaction verbatim: fenced code, paths, URLs, system prompt, last 6 turns. |
| **Mirror** | Human-readable Markdown rendering of the shared SQLite board, at `~/.agent_bin/central_context.md`. Jarvis-curated. |
| **Heartbeat** | Optional autonomous loop where Jarvis wakes on a timer to walk a checklist (`HEARTBEAT.md`) and act proactively. |

---

## 23. First-session checklist (for the agent reading this)

When you start the build, do this in order:

1. Read `CLAUDE_HANDOFF.md` (bottom = newest). Look for any `need from Jarvis` items.
2. Confirm the open questions in §21 with the user.
3. Set up the local repo at `/Users/grant/cmd/jarvis/` with the structure from §4.
4. Ship P0 (scaffolding) end-to-end including a deployable systemd unit (disabled). Smoke-test the deploy flow: SCP a placeholder, restart, verify systemd sees the file.
5. Append a handoff stanza listing what's shipped and what's needed from CMD/Swarm (probably "nothing" at P0).
6. Move to P1.

Do not skip steps. Do not interleave phases. The architectural invariants in §2 apply from day one.

---

*Spec version: v1.1 — 2026-04-26*
*Author: Jarvis-Claude (project-management role)*
*Companion docs: `INTEGRATION_CONTRACT.md`, `cmd/DOCS/JARVIS_INTEGRATION_GUIDE.md`, `swarm/DOCS/INTEGRATION_GUIDE.md`*

### Changelog

**v1.1 (2026-04-26)** — review pass from CMD-Claude. Changes that are not architectural drift but bug-fixes against reality:
- §2 invariant 5: noted that 6-turn floor is intentional Jarvis-side superset of the 3-message contract floor.
- §5 / §17.1: added the mandatory `sqlite-vec` extension load incantation and dependency-verification snippet.
- §6.4: fixed MMR to use a `vectors_by_id` dict pre-fetched during search.
- §6.5: added the `delegate` tool definition and explicitly stated that legacy action tags are dropped.
- §8: added a turns-vs-messages definition lock.
- §11.0: added a "reality check" subsection. CMD does not return contract-shaped envelopes today, does not accept `context_keys` at the REST boundary, and the submit URL is `/api/v1/jobs`.
- §11 chain submit: corrected body shape to `{goal, max_iterations, model?}`.
- §12.0: added "verify against live Swarm server before coding" note.
- §17 chat_model: changed default from `qwen2.5:14b` to `qwen2.5:3b` with VRAM rationale.
- §17 tokenizer: specified (qwen-native via transformers, with 4-char approximation fallback).
- §17.1: added a runtime-dependencies table.
- §21.3: locked group-chat `memory_search` behavior — filters out `file_kinds=["memory"]`.
- §21.10: added the three cross-side handoffs that should be filed before P0.

**v1.0 (2026-04-26)** — initial spec.
