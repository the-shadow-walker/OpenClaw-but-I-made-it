# P3 — Embeddings + sqlite-vec + Hybrid Search (BUILT)

This is a retrospective plan-style doc: what actually shipped, in plan format,
so it can be scrutinized before deploy. Cross-reference against
`docs/BUILD_SPEC.md` §5/§6.3/§6.4/§19 and the original P3 prompt.

Local state: ruff clean, 71 passed / 1 skipped. Has not been deployed to arch01
or run against real Ollama.

---

## Files added

### 1. `jarvis/memory/embeddings.py` (NEW)

Provider stack + pipeline + cache. Public surface:

```python
class EmbeddingError(RuntimeError): ...

class EmbeddingProvider(Protocol):
    kind: str
    model: str
    dimensions: int
    @property
    def fingerprint(self) -> str: ...
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

@dataclass
class OllamaEmbeddings:
    host: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    dimensions: int = 768
    timeout_s: float = 60.0
    kind: str = "ollama"

# (OpenAI provider removed in scrutiny pass — see Open question #1.)

class EmbeddingPipeline:
    def __init__(self, provider, cache_conn=None): ...
    @property
    def fingerprint(self) -> str: ...
    @property
    def dimensions(self) -> int: ...
    def embed_one(self, text) -> list[float] | None: ...
    def embed_batch(self, texts) -> list[list[float] | None]: ...

def build_provider_from_config(cfg) -> EmbeddingProvider | None: ...
def l2_normalize(vec) -> list[float]: ...
def fingerprint_for(kind, model, dimensions) -> str: ...

EMBEDDING_CACHE_MAX_ROWS = 50_000

@dataclass
class _DeterministicEmbeddings:
    """Hash-based fake. Tests-only; NOT in __all__."""
```

**Decisions worth scrutinizing:**

1. **L2-normalize-on-write inside `EmbeddingPipeline`**, not in the provider.
   Every output of `embed_batch` is unit-norm, so vec0's L2 distance ranks
   identically to cosine. If anyone ever calls `provider.embed_batch` directly
   they get raw vectors — there's a docstring warning but no runtime check.

2. **Cache uses indexer's same SQLite connection** (passed in as `cache_conn`).
   Reads use `SELECT FROM embedding_cache`; touches `accessed_at` on every hit;
   writes use `INSERT OR REPLACE`. All wrapped in `with conn:` and any
   `sqlite3.Error` becomes a `logger.warning` — cache failures **never** block
   reconciliation.

3. **LRU eviction is opportunistic, not strict.** After every write batch
   `_maybe_evict()` runs `SELECT COUNT(*)` and, if over cap, deletes
   `n - cap + n // 20` oldest rows (5% slack so we don't sweep on every
   single write). `accessed_at` is updated on hit so the LRU has signal,
   but we don't sweep on hit.

4. **Provider failure semantics.** `OllamaEmbeddings.embed_batch` raises
   `EmbeddingError` for *any* batch-level failure (HTTP error, malformed
   payload, dimension mismatch). The pipeline catches it and returns
   `[None] * len(texts)` — per-item failures aren't modeled. The indexer
   then leaves `embedding_model` NULL on those chunks so the next sweep
   retries.

5. **OpenAI is lazy-imported** (`from openai import OpenAI` inside
   `embed_batch`) and skipped silently in `build_provider_from_config` if
   `OPENAI_API_KEY` is unset. This is the §6.3 fallback order:
   ollama → openai → degraded.

**Footguns I deliberately did NOT close:**

- The cache reads `accessed_at` on hit but doesn't `BEGIN IMMEDIATE` — under
  heavy concurrent load the touch-update could race with eviction. P3 only
  has one writer (the CLI), so this is fine for now. Note for P5 daemon.
- `_DeterministicEmbeddings` is in `embeddings.py` instead of `tests/`.
  Rationale: it's tightly coupled to provider-protocol shape and lives next
  to the real providers so test imports are simple. Underscored to signal
  internal-only; not in `__all__`.

### 2. `jarvis/memory/index.py` (EDITED)

Changes from P2:

- New import: `from jarvis.memory.embeddings import EmbeddingPipeline`.
- `ReconcileStats` gains two int fields: `chunks_embedded`, `chunks_embed_failed`.
- `Indexer.__init__` gains two kwargs: `embedder: EmbeddingPipeline | None = None`,
  `embed_batch_size: int = 32`.
- `Indexer.reconcile()` adds, **before** the cascade `DELETE FROM files`, an
  explicit `DELETE FROM chunks_vec WHERE chunk_id IN (SELECT id FROM chunks
  WHERE file_path = ?)`. This is the **vec0-doesn't-honor-FK-CASCADE** fix.
- `Indexer.remove_file()` gains the same vec0 cleanup before the cascade.
- `Indexer.reconcile_all()` calls `self.embed_pending()` at the end if an
  embedder is set; populates the new stats fields.
- New method `Indexer.embed_pending(limit=None) -> tuple[int, int]`. Selects
  chunks with `embedding_model IS NULL OR embedding_model != ?`, batches
  them, calls `embedder.embed_batch`, writes via `_write_vectors`. Returns
  `(embedded_count, failed_count)`.
- New method `Indexer._write_vectors(chunk_ids, vectors, fingerprint)`. For
  each non-None vector: `DELETE FROM chunks_vec WHERE chunk_id=?`,
  `INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)` with
  `sqlite_vec.serialize_float32(vec)`, then
  `UPDATE chunks SET embedding_model=? WHERE id=?`. None vectors are skipped
  so `embedding_model` stays NULL → picked up next sweep.

**Decisions worth scrutinizing:**

1. **Manual vec0 cleanup before CASCADE.** I confirmed via failing tests
   that vec0 virtual tables don't honor the parent FK CASCADE. Spec §5
   doesn't mention this; the workaround lives in `reconcile()` and
   `remove_file()` only. If anyone adds a third path that deletes from
   `chunks` (or `files`) without going through these methods, we'll get
   orphan vec0 rows. **No test covers this third-path case yet** —
   probably worth adding before P5.

2. **Lazy backfill happens at end of `reconcile_all`**, not per-file inside
   `reconcile()`. The reasoning: a single reconcile may add 200 chunks; we
   want them in one batched provider call (cheaper than 200 single calls).
   The cost is that mid-reconcile a file is BM25-searchable but not
   vector-searchable for the duration of the sweep. P3 search degrades
   gracefully — chunks without a current-fingerprint vector are just
   skipped from `_candidate_vector` results.

3. **`embed_batch_size=32` is a guess.** Ollama nomic-embed-text seems
   happy with batches of that size locally; no benchmarking. Configurable
   via the constructor; not exposed in `JarvisConfig`. Worth measuring on
   arch01.

4. **`embed_pending()` filter:** `embedding_model IS NULL OR != fingerprint`.
   This means switching providers (e.g. ollama → openai) re-embeds the
   entire workspace on the next reconcile. That's the §6.3 lazy-migration
   contract.

### 3. `jarvis/memory/search.py` (REWRITTEN)

P2's `search_bm25` is kept verbatim (used by tests + the `--bm25-only` CLI
flag + the degraded-mode fallback). Three new public exports:

```python
@dataclass
class SearchOptions:
    k: int = 10
    candidate_multiplier: int = 4    # pool = k * 4
    vector_weight: float = 0.7
    text_weight: float = 0.3
    mmr_lambda: float = 0.7
    half_life_days: float = 30.0
    file_kinds: list[str] | None = None
    include_conversations: bool = False  # P5+

def memory_search(conn, query, embedder=None, options=None) -> list[SearchResult]: ...
def min_max_normalize(scores: dict) -> dict: ...
def mmr_select(candidates, relevance, vectors, k, lambda_=0.7) -> list[int]: ...
```

**Algorithm (in order):**

1. Sanitize query (strip non-alphanumerics, collapse whitespace) — same
   regex as `search_bm25`.
2. Pull a BM25 candidate pool of size `k * candidate_multiplier` via
   `_candidate_bm25` → `{chunk_id: -bm25_score}`.
3. If embedder present: `embedder.embed_one(query)`, then
   `_candidate_vector` against vec0 with the resulting unit vector.
   `_candidate_vector` issues `MATCH ?` with `k=raw_limit` and a
   fingerprint filter (`c.embedding_model = ?`); when `file_kinds` is set,
   pulls 4× the requested pool size and post-filters by file_kind.
   Converts L2 distance to cosine via `cos = 1 - d²/2` (valid only because
   vectors are unit-norm).
4. Min-max normalize each pool independently. All-equal pools collapse to
   0.5 (neutral). **This is the spec §6.4 "BM25 is unbounded, cosine is
   bounded — always normalize" rule.**
5. Union the candidate IDs. For each: fetch metadata once via
   `_fetch_chunk_meta` (joined `chunks` + `files` so we have evergreen +
   modified_at). Compute fused = `0.7*v + 0.3*b`, then multiply by decay
   (1.0 if evergreen else `exp(-ln2 * age_days / 30)`).
6. Sort fused descending, slice to `pool_size`, batch-fetch their vectors
   via `_fetch_chunk_vectors` (struct-unpack the vec0 BLOB). Run
   `mmr_select` over the candidate IDs with the pre-fetched vectors.
7. Return `SearchResult` rows for the selected IDs in MMR order, with
   `score_components` populated for transparency.
8. Insert one row into `search_queries` (P9 dreaming signal source). Logged
   with `int(time.time())` and `result_count`. Failures are warnings; never
   raise.

**Decisions worth scrutinizing:**

1. **MMR's first pick is pure relevance** (no diversity term). I considered
   making the first pick aware of the query vector itself, but the
   relevance score is already query-aware (it's the fused score) so adding
   a query-similarity term at step 1 would just re-multiply the same
   signal. Spec §6.4 doesn't specify either way; this matches Carbonell's
   original MMR.

2. **Candidates lacking a current-fingerprint vector get
   `max_sim = 0` in MMR**, which makes their diversity term maximally
   generous. That's deliberate (§6.4 graceful-degradation: "BM25-flavored"
   results stay reachable when the index is partially embedded). Could
   silently bias toward unembedded chunks during a backfill window. Tested
   in `test_mmr_select_handles_missing_vectors` only as a smoke test.

3. **`_candidate_vector` over-pulls 4× when `file_kinds` is set**, then
   post-filters in Python via a kept_paths set. SQL-level filtering would
   require either a join in the vec0 MATCH (vec0 doesn't support that) or
   a CTE — the over-pull is simpler and the pool is small (typically
   4 × k × 4 = 160 rows max).

4. **Decay half-life is hardcoded to 30 days** in `SearchOptions`. The
   spec mentions 30 days; if we want to change it that's an option-knob,
   not a config-knob. Fine for now.

5. **`include_conversations=False` is dead code in P3** — there are no
   conversation transcripts indexed yet (P5 work). I shipped the option
   on the dataclass so P5 doesn't have to widen `SearchOptions` and break
   call sites.

### 4. `jarvis/cli.py` (EDITED)

P2's CLI structure preserved exactly. Changes:

- Imports `build_provider_from_config`, `EmbeddingPipeline`,
  `memory_search`, `SearchOptions`.
- `search` subparser gains `--bm25-only` (force degraded path) and
  `--show-components` (print bm25/vector/decay alongside fused score).
- After indexer creation, builds an `EmbeddingPipeline` from cfg via
  `build_provider_from_config`. **`provider is None` → degraded mode**:
  emits a `logger.warning` ("embeddings: degraded mode...") and
  `embedder=None`. **Provider built → `logger.info("embeddings: %s",
  fingerprint)`**.
- Embedder is attached to the indexer (`indexer.embedder = embedder`) so
  `reconcile_all` picks it up automatically.
- `search` branch routes to `memory_search` by default, falls back to
  `search_bm25` when `--bm25-only` or `embedder is None`.
- `--show-components` prints a comma-joined dict of score components per
  result.

**Decisions worth scrutinizing:**

1. **The CLI shares one connection between indexer and embedder cache.**
   This is fine for the CLI (single-threaded, single-process) but will
   bite if P5's FastAPI daemon spins up worker threads and they all share
   the same `sqlite3.Connection`. Marker for the daemon redesign.

2. **Embedder fingerprint logs INFO once per process, DEBUG after.**
   Module-level `_LOGGED_EMBEDDER_FINGERPRINTS` set in `cli.py` tracks
   which fingerprints we've already announced. CLI invocations are fresh
   processes so they always log INFO once (preserves the "visible at
   deploy time" property). The P5 daemon will log INFO at boot and DEBUG
   for subsequent requests, avoiding journalctl spam. The tokenizer-style
   "every invocation" rule was wrong here: tokenizer fallback affects
   correctness (P6 compaction triggers), embedder fingerprint affects
   deploy-time decisions only.

3. **`bm25_only` is a CLI flag, not a config knob.** Operators forcing
   BM25-only as a debug step shouldn't need to edit config. If we ever
   want a permanent BM25-only mode (no-network deploy?), promote to
   config.

---

## Tests added

### `tests/unit/test_embeddings.py` (9 tests)

- `test_l2_normalize_unit_norm` — `[3, 4]` → unit norm.
- `test_l2_normalize_zero_vector_passthrough` — zero stays zero (no /0).
- `test_fingerprint_format` — `"ollama:nomic-embed-text:768"`.
- `test_pipeline_normalizes_outputs` — every output is unit-norm.
- `test_pipeline_caches_hits` — second call doesn't bump provider counter;
  one new text adds one provider call.
- `test_pipeline_returns_none_on_provider_failure` — `EmbeddingError`
  → `[None, None]`.
- `test_pipeline_partial_cache_partial_provider` — primed cache + failing
  new provider: cached items still return, new items return None.
- `test_cache_lru_eviction_triggers_above_threshold` — monkeypatches
  `EMBEDDING_CACHE_MAX_ROWS` to 5, inserts 10, asserts `<= 5` rows.
- `test_cache_default_max_rows_is_50k` — sanity check on §5 constant.

### `tests/unit/test_search_hybrid.py` (11 tests)

- `test_min_max_normalize_basic` — 0/0.5/1.
- `test_min_max_normalize_all_equal_collapses_to_neutral` — all 0.5.
- `test_min_max_normalize_empty` — `{}` → `{}`.
- `test_decay_factor_evergreen_skip_via_caller` — half-life math: 1.0,
  0.5, 0.25 at 0/30/60 days.
- `test_mmr_select_picks_first_by_relevance`.
- `test_mmr_select_diverse_when_top_two_are_similar` — λ=0.5, the
  near-twin loses to the orthogonal lower-relevance candidate.
- `test_mmr_select_handles_missing_vectors`.
- `test_memory_search_roundtrip_with_deterministic_embedder` — full
  populate + reconcile + embed + search; uses `vector_weight=0.0,
  text_weight=1.0` because the deterministic embedder is hash-noise.
- `test_memory_search_empty_query_returns_empty`.
- `test_memory_search_evergreen_skips_decay` — backdate every file by
  180 days, run "standup" with text-only, assert evergreen
  (MEMORY/USER/SOUL) decay = 1.0 and non-evergreen decay < 1.0.
- `test_memory_search_degraded_mode_without_embedder` — no embedder,
  still gets results, no crash.

### `tests/acceptance/test_index_rebuild.py` (1 test)

- `test_top5_overlap_at_least_80_percent_across_rebuild` — populate +
  reconcile + run 10 queries → wipe DB (also `-wal`/`-shm`) + reconcile
  + same 10 queries → average per-query overlap of top-5 keys
  `(file_path, start_line, end_line)` ≥ 80%. Currently passes at 100%
  with the deterministic embedder.

**Decisions worth scrutinizing:**

1. **Test embedder is hash-based, not real.** Two tests
   (`test_memory_search_roundtrip` and `_evergreen_skips_decay`)
   explicitly pin `vector_weight=0.0, text_weight=1.0` because the
   hash-based embedder produces noise that drowns BM25 signal at default
   0.7/0.3 weights. The acceptance test uses default weights and still
   gets 100% overlap because both runs use the same hash function — the
   index is disposable but **the embedder's quality is not validated by
   any test** in this PR. That's the "real Ollama on arch01" job.

2. **Acceptance threshold is 80%.** The spec asks for ≥80% overlap.
   Currently at 100% locally, so we have headroom. If real Ollama runs
   produce different rankings between rebuilds (it shouldn't — same
   chunks, same embed text → same vectors → same scores) we'll find out
   on arch01.

3. **Top-5 keys use `(file_path, start_line, end_line)`** instead of
   `chunk_id` because chunk autoincrement re-counts after a wipe.

4. **No test for the search-degrades-when-embedder-fails path.** The
   pipeline tests cover provider failure → None vectors; the search
   tests cover no-embedder + degraded mode. There's a gap where the
   embedder is non-None but `embed_one(query)` returns None mid-search.
   Code path exists (search.py:438-446), not exercised. Could add a
   monkeypatched test if scrutiny reveals it.

---

## What did NOT change

- `pyproject.toml` — no dep changes; embeddings code uses only `httpx`
  (already a P0 dep) and the optional `openai` (lazy-imported, not
  declared).
- `jarvis/run.py` — still the P0 stub. `jarvis daemon` shim unchanged.
- `jarvis/config.py` — `cfg.embeddings.providers` already existed; we
  consume it as-is. No new fields needed.
- The §5 DDL — verbatim. `chunks_vec` is fixed at 768 dims; OpenAI's
  1536d won't fit without a schema migration. **Listed as Open question
  below.**
- `jarvis.service` systemd unit — still runs `python -m jarvis.run`.
  Will switch to `jarvis daemon` in P5.

---

## Open questions / known limitations

1. **(RESOLVED)** OpenAI dropped from `EmbeddingProviderConfig.kind`
   entirely. Config schema now `Literal["ollama"]`; YAML files listing
   `kind: openai` fail loudly at `load_config`. `OpenAIEmbeddings` class
   removed from `embeddings.py`. Defer multi-provider work to P14, where
   the dimension-migration path can be designed properly. The `chunks_vec
   FLOAT[768]` schema is now safe — every code path that produces an
   embedding does so at 768d.

2. **`chunks_vec` rows aren't audited against `chunks` rows.** A bug in
   `_write_vectors` could leave vec0 with stale rowids; we'd never know
   until search returned junk. A periodic
   `SELECT COUNT(*) FROM chunks_vec WHERE chunk_id NOT IN (SELECT id FROM chunks)`
   sanity check would catch it. Not added in P3. Pinned in
   `test_index.py::test_vec0_orphans_when_chunks_deleted_directly`
   so the footgun is documented and the test will flip if vec0 ever
   honors CASCADE.

3. **Concurrent-writer correctness.** The cache uses the indexer's
   connection. Two concurrent CLI invocations would race on
   `embedding_cache` writes. SQLite WAL mode handles read-write
   concurrency; the cache is best-effort so loss is fine. But P5's daemon
   needs its own thinking — deferred.

4. **Per-batch retry on provider failure is not implemented.** A 32-item
   batch where one item is "too long for nomic-embed-text" tanks the
   whole batch. We could chunk into single-item retries on failure;
   adding complexity for a real failure mode that may not occur in
   practice. Punt until we see it.

5. **(RESOLVED)** MMR tiebreakers now use chunk_id asc as the explicit
   secondary sort key in both the candidate-id sort
   (`memory_search`) and the inner-loop iteration (`mmr_select`). No
   more relying on CPython dict insertion order for ranking determinism.

---

## Verification — done locally

```
ruff check .                                        # clean
pytest tests/                                       # 71 passed, 1 skipped
pytest tests/acceptance/test_index_rebuild.py -v    # acceptance green
```

**The CI acceptance test is necessary but not sufficient.** It uses a
hash-noise fake embedder, so the 100% top-5 overlap proves only that the
indexer + reconciler + hybrid scorer + MMR are deterministic across
delete+rebuild *for the same embedder*. It says nothing about real
embedding quality. The actual P3 exit criterion is the arch01 deploy
verification below, with real Ollama nomic-embed-text. The spec accepts
≥80% top-5 overlap; if the deploy run comes back at 92% with the same
files appearing in different orders, that's a pass. If it comes back
showing different files than the pre-rebuild run, that's a real bug —
most likely in candidate-pool sizing or the `_candidate_vector`
file_kind post-filter over-pull.

## Verification — NOT done yet

Pending an explicit go-ahead:

```bash
# Deploy P3 to arch01 + reinstall (console-script unchanged from P2 but
# no harm in re-running):
ssh mcssh "cd /mnt/storage/NAS/Jarvis/jarvis && \
  /mnt/storage/NAS/Jarvis/.venv/bin/pip install -e . --no-deps --quiet"
ssh mcssh "cd /mnt/storage/NAS/Jarvis/jarvis && \
  /mnt/storage/NAS/Jarvis/.venv/bin/python -m pytest tests/ -v"

# Real-Ollama exit-criterion: populate, reconcile, eyeball top-3 on the
# 10-query eval set, then prove rebuild stability:
ssh mcssh "cd /mnt/storage/NAS/Jarvis/jarvis && \
  /mnt/storage/NAS/Jarvis/.venv/bin/python -m tests.fixtures.populate_workspace --root /tmp/p3-eval && \
  JARVIS_WORKSPACE=/tmp/p3-eval /mnt/storage/NAS/Jarvis/.venv/bin/jarvis reconcile && \
  for q in 'rocket fin' 'typescript' 'daily log' 'jarvis-rebuild' 'garden' 'standup' 'preferences' 'fast model' 'grant' 'reconcile'; do \
    echo == \"\$q\" ==; \
    JARVIS_WORKSPACE=/tmp/p3-eval /mnt/storage/NAS/Jarvis/.venv/bin/jarvis search -k 3 \"\$q\"; \
  done > /tmp/p3-search-before.txt"

# Hybrid disposability — run the same eval after wipe:
ssh mcssh "rm /tmp/p3-eval/.index/memory.sqlite* && \
  JARVIS_WORKSPACE=/tmp/p3-eval /mnt/storage/NAS/Jarvis/.venv/bin/jarvis reconcile && \
  for q in 'rocket fin' 'typescript' 'daily log' 'jarvis-rebuild' 'garden' 'standup' 'preferences' 'fast model' 'grant' 'reconcile'; do \
    echo == \"\$q\" ==; \
    JARVIS_WORKSPACE=/tmp/p3-eval /mnt/storage/NAS/Jarvis/.venv/bin/jarvis search -k 3 \"\$q\"; \
  done > /tmp/p3-search-after.txt && \
  diff /tmp/p3-search-before.txt /tmp/p3-search-after.txt"
# diff should be byte-empty IF the embedder is deterministic across calls.
# (Real Ollama IS deterministic for the same input + model.)
```

---

## Handoff stanza (draft, not yet committed)

```
## 2026-04-27THH:MM — Jarvis Claude
- shipped: P3 — embeddings (Ollama-only; OpenAI dropped from config
  schema until P14 to avoid the 768/1536 vec0 dimension trap) +
  sqlite-vec writes + hybrid search (BM25 + vector w/ min-max
  normalize per pool, 0.7v/0.3t fusion, MMR λ=0.7 with prefetched
  vectors and chunk_id-asc tiebreakers, 30-day decay sparing
  evergreen). Indexer gained lazy backfill (embedding_model
  fingerprint diff) + vec0-pre-cascade manual cleanup (vec0 doesn't
  honor parent FK CASCADE — pinned in
  test_vec0_orphans_when_chunks_deleted_directly). CLI search
  defaults to hybrid; --bm25-only forces degraded path;
  --show-components prints bm25/vector/decay. Embedder fingerprint
  logs INFO once per process then DEBUG (CLI: once per invocation;
  daemon: once at boot — no per-query journalctl spam in P5).
  *Real exit criterion is the arch01 deploy verification with live
  Ollama; the CI acceptance test (top-5 overlap 100% with hash-noise
  fake embedder) only proves the index is disposable, not that
  embedding quality is real. Spec accepts ≥80% overlap on real
  queries — don't panic if real Ollama lands below 100%.*
- need from CMD: greenlight to deploy + run /tmp/p3-eval on arch01.
- need from Swarm: nothing.
- blocking: nothing locally; arch01 verification not yet performed.
- next: P4 — watcher (debounced reindex on save) + Dreaming gate
  scaffolding. Followups deferred from P3: BUILD_SPEC.md §5 note that
  vec0 virtual tables don't honor parent FK CASCADE (deletes must
  explicitly clean chunks_vec first); chunks/vec0 audit query for
  P5 health probe.
```
