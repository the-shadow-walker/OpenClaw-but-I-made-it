"""SQLite index for the workspace — schema, connection factory, reconciler.

Scope (BUILD_SPEC §5, §6.1, §6.3):
  * ``get_connection()`` — the mandatory ``sqlite-vec`` load incantation,
    centralized so we can never forget it. Also sets WAL + foreign keys
    (CASCADE on chunks needs them) + Row factory.
  * ``init_schema()`` — verbatim DDL from §5: ``files``, ``chunks`` (+ idx),
    ``chunks_fts`` (porter unicode61) + ai/ad/au triggers, ``chunks_vec``
    (vec0 768d), ``embedding_cache``, ``conversations``, ``search_queries``.
    All ``IF NOT EXISTS``; idempotent.
  * ``Indexer`` — wraps a connection + workspace_root, reconciles individual
    files by content-hash diff, walks the workspace skipping dotfile dirs,
    removes vanished files. P3 adds: an optional ``EmbeddingPipeline`` so
    new chunks get embedded into ``chunks_vec`` and the chunk row's
    ``embedding_model`` fingerprint is stamped on success.

Lazy backfill (§6.3): chunks whose ``embedding_model`` is NULL or differs
from the active pipeline's fingerprint are re-embedded on the next sweep
(``reconcile_all``). A failed batch never blocks FTS — chunks stay queryable
via BM25, and search degrades gracefully to BM25-only for any chunk lacking
a current-fingerprint vector.

Path representation: paths are stored as POSIX-style relative-to-workspace
strings. Lets the workspace move without invalidating rows; consistent on
macOS / Linux. The ``Indexer`` accepts absolute ``Path`` inputs and
normalizes internally.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from jarvis.memory.chunker import chunk_markdown
from jarvis.memory.embeddings import EmbeddingPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "ReconcileStats",
    "Indexer",
    "ALL_FILE_KINDS",
    "get_connection",
    "init_schema",
    "insert_conversation",
    "close_conversation",
    "get_open_conversation",
    "list_open_conversations",
]


# Single source of truth for the file_kind enum produced by ``_classify`` —
# imported by tools that need to filter by kind (e.g. memory_search_tool's
# group-chat MEMORY filter). Adding a new kind: extend both this tuple and
# ``_classify`` together.
ALL_FILE_KINDS: tuple[str, ...] = (
    "memory",
    "user",
    "soul",
    "agents",
    "tools",
    "heartbeat",
    "dreams",
    "daily",
    "project",
)


# ---------------------------------------------------------------------------
# Connection factory + DDL
# ---------------------------------------------------------------------------


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sqlite-vec loaded, WAL on, FKs on.

    The mandatory incantation from BUILD_SPEC §5. Loaded on EVERY connection.
    Forgetting this is the most common P3 footgun — keeping it in one place
    guarantees we never ship a code path that opens a raw ``sqlite3.connect``.

    ``check_same_thread=False`` only relaxes Python's per-connection thread
    check; SQLite itself serializes via its own mutex. The P4 watcher
    creates the ``Indexer`` on the caller thread and then drives it from
    the drainer thread; that pattern is single-writer-at-a-time but the
    creator and writer threads differ, which is precisely what this flag
    permits. Concurrent multi-thread writes are still the caller's
    responsibility to prevent (the watcher's drainer is the sole caller).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")  # ON DELETE CASCADE on chunks needs this.
    conn.row_factory = sqlite3.Row
    return conn


# Verbatim from BUILD_SPEC §5. Keep one DDL block per logical object so future
# diffs are reviewable.
_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS files (
        path           TEXT PRIMARY KEY,
        content_hash   TEXT NOT NULL,
        modified_at    INTEGER NOT NULL,
        evergreen      INTEGER NOT NULL DEFAULT 0,
        file_kind      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path        TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
        chunk_index      INTEGER NOT NULL,
        content          TEXT NOT NULL,
        start_line       INTEGER NOT NULL,
        end_line         INTEGER NOT NULL,
        token_count      INTEGER NOT NULL,
        heading_path     TEXT,
        created_at       INTEGER NOT NULL,
        embedding_model  TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_model ON chunks(embedding_model)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        content,
        content='chunks',
        content_rowid='id',
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE rowid = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE rowid = old.id;
        INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
    END
    """,
    # Vector index — P3 will write to this; P2 ships the DDL only.
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
        chunk_id   INTEGER PRIMARY KEY,
        embedding  FLOAT[768]
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        text_hash         TEXT NOT NULL,
        model_fingerprint TEXT NOT NULL,
        embedding         BLOB NOT NULL,
        accessed_at       INTEGER NOT NULL,
        PRIMARY KEY (text_hash, model_fingerprint)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_emb_cache_accessed ON embedding_cache(accessed_at)",
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id              TEXT PRIMARY KEY,
        started_at      INTEGER NOT NULL,
        ended_at        INTEGER,
        channel_kind    TEXT NOT NULL,
        channel_id      TEXT,
        slug            TEXT,
        summary         TEXT,
        transcript_path TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_conv_started ON conversations(started_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS search_queries (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        query        TEXT NOT NULL,
        query_hash   TEXT NOT NULL,
        queried_at   INTEGER NOT NULL,
        result_count INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_query_hash ON search_queries(query_hash)",
    "CREATE INDEX IF NOT EXISTS idx_query_time ON search_queries(queried_at DESC)",
)


def init_schema(conn: sqlite3.Connection) -> None:
    """Apply every DDL statement (idempotent — all use IF NOT EXISTS)."""
    with conn:
        for ddl in _DDL_STATEMENTS:
            conn.execute(ddl)


# ---------------------------------------------------------------------------
# File-kind classifier
# ---------------------------------------------------------------------------


def _classify(rel_path: Path) -> tuple[str, bool]:
    """Map a workspace-relative path to ``(file_kind, evergreen)``.

    AGENTS.md is operational config (delegation rules), not durable user
    content — not evergreen. §6.2 reserves evergreen=1 for MEMORY.md /
    USER.md / SOUL.md only. Do not "fix" AGENTS.md back to evergreen=1.

    Note on 'heartbeat': §5's file_kind enum doesn't list it, but §3.1 ships
    HEARTBEAT.md as a real file we'll want to search. We expand the enum (a
    strict superset — no schema change, just the string we write into
    file_kind). Documented in the handoff stanza.
    """
    name = rel_path.name
    parts = rel_path.parts

    if name == "MEMORY.md" and len(parts) == 1:
        return ("memory", True)
    if name == "USER.md" and len(parts) == 1:
        return ("user", True)
    if name == "SOUL.md" and len(parts) == 1:
        return ("soul", True)
    if name == "AGENTS.md" and len(parts) == 1:
        return ("agents", False)
    if name == "TOOLS.md" and len(parts) == 1:
        return ("tools", False)
    if name == "HEARTBEAT.md" and len(parts) == 1:
        return ("heartbeat", False)
    if name == "DREAMS.md" and len(parts) == 1:
        return ("dreams", False)
    if len(parts) == 2 and parts[0] == "memory":
        return ("daily", False)
    if len(parts) == 2 and parts[0] == "projects":
        return ("project", False)

    # Catch-all: treat as a generic project-ish file. Evergreen stays False.
    # We still index it (so "where did I write about X?" works) but it lives
    # outside the curated kinds.
    return ("project", False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_rel_posix(workspace_root: Path, file_path: Path) -> str:
    """POSIX-style relative-to-workspace string for DB storage."""
    abs_file = file_path.resolve() if not file_path.is_absolute() else file_path
    abs_root = workspace_root.resolve()
    return abs_file.relative_to(abs_root).as_posix()


def _is_dotfile_path(rel_path: Path) -> bool:
    """§6.1 dotfile-skip rule: any path component starting with '.'."""
    return any(p.startswith(".") for p in rel_path.parts)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


@dataclass
class ReconcileStats:
    files_seen: int = 0
    files_changed: int = 0
    files_added: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    chunks_embedded: int = 0
    chunks_embed_failed: int = 0


class Indexer:
    """Reconciles workspace .md files into the SQLite index.

    With ``embedder`` set, ``reconcile`` and ``reconcile_all`` also embed
    new/stale chunks and write to ``chunks_vec``. With ``embedder=None``,
    the indexer is BM25-only — useful for tests and the degraded-mode
    fallback when the configured provider is unreachable.
    """

    def __init__(
        self,
        db_path: Path,
        workspace_root: Path,
        embedder: EmbeddingPipeline | None = None,
        embed_batch_size: int = 32,
    ) -> None:
        self.db_path = Path(db_path)
        self.workspace_root = Path(workspace_root).resolve()
        self.conn = get_connection(self.db_path)
        init_schema(self.conn)
        self.embedder = embedder
        self.embed_batch_size = embed_batch_size

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()

    # -- reconcile a single file -------------------------------------------

    def reconcile(self, file_path: Path) -> bool:
        """Reconcile one file. Returns True if anything changed.

        - Missing in DB → INSERT files row + chunks. Returns True.
        - Hash matches  → no-op. Returns False.
        - Hash differs  → DELETE files row (CASCADE drops chunks; FTS triggers
                          drop chunks_fts rows) + re-insert. Returns True.

        All transactional in a single ``with self.conn:`` block per file so
        a crash mid-reconcile can never leave orphan FTS rows.
        """
        abs_path = file_path.resolve() if not Path(file_path).is_absolute() else Path(file_path)
        rel_str = _to_rel_posix(self.workspace_root, abs_path)
        rel_path = Path(rel_str)

        raw_bytes = abs_path.read_bytes()
        content_hash = _sha256(raw_bytes)

        row = self.conn.execute(
            "SELECT content_hash FROM files WHERE path = ?", (rel_str,)
        ).fetchone()

        if row is not None and row["content_hash"] == content_hash:
            return False

        text = raw_bytes.decode("utf-8", errors="replace")
        chunks = chunk_markdown(text)
        file_kind, evergreen = _classify(rel_path)
        modified_at = int(abs_path.stat().st_mtime)
        now = int(time.time())

        with self.conn:
            if row is not None:
                # Hash differs: clear old vec rows BEFORE the cascade drops
                # the chunks they belong to (vec0 doesn't honor FK CASCADE).
                self.conn.execute(
                    "DELETE FROM chunks_vec WHERE chunk_id IN ("
                    "  SELECT id FROM chunks WHERE file_path = ?"
                    ")",
                    (rel_str,),
                )
                # Now drop the file row — cascade clears chunks + FTS triggers.
                self.conn.execute("DELETE FROM files WHERE path = ?", (rel_str,))

            self.conn.execute(
                "INSERT INTO files (path, content_hash, modified_at, evergreen, file_kind) "
                "VALUES (?, ?, ?, ?, ?)",
                (rel_str, content_hash, modified_at, 1 if evergreen else 0, file_kind),
            )

            for idx, ch in enumerate(chunks):
                self.conn.execute(
                    "INSERT INTO chunks "
                    "(file_path, chunk_index, content, start_line, end_line, "
                    " token_count, heading_path, created_at, embedding_model) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                    (
                        rel_str,
                        idx,
                        ch.content,
                        ch.start_line,
                        ch.end_line,
                        ch.token_count,
                        ch.heading_path or None,
                        now,
                    ),
                )
        return True

    # -- remove a file (delete cascades) -----------------------------------

    def remove_file(self, file_path: Path) -> None:
        """Drop a file's row + cascading chunks/FTS rows. No-op if absent."""
        p = Path(file_path)
        # Already-relative input is accepted as-is (POSIX-style normalize).
        rel_str = _to_rel_posix(self.workspace_root, p) if p.is_absolute() else p.as_posix()
        with self.conn:
            # vec0 virtual tables don't honor FK CASCADE; drop manually first.
            self.conn.execute(
                "DELETE FROM chunks_vec WHERE chunk_id IN ("
                "  SELECT id FROM chunks WHERE file_path = ?"
                ")",
                (rel_str,),
            )
            self.conn.execute("DELETE FROM files WHERE path = ?", (rel_str,))

    # -- full sweep --------------------------------------------------------

    def reconcile_all(self) -> ReconcileStats:
        """Walk the workspace, reconcile every .md, drop vanished rows."""
        stats = ReconcileStats()

        seen_rel: set[str] = set()
        for abs_path in sorted(self.workspace_root.rglob("*.md")):
            try:
                rel = abs_path.relative_to(self.workspace_root)
            except ValueError:
                # Symlink escape or similar — skip.
                continue
            if _is_dotfile_path(rel):
                continue
            if not abs_path.is_file():
                continue

            stats.files_seen += 1
            existed_before = self.conn.execute(
                "SELECT 1 FROM files WHERE path = ?", (rel.as_posix(),)
            ).fetchone() is not None

            changed = self.reconcile(abs_path)
            if changed:
                if existed_before:
                    stats.files_changed += 1
                else:
                    stats.files_added += 1
            seen_rel.add(rel.as_posix())

        # Remove DB rows for files that no longer exist on disk.
        rows = self.conn.execute("SELECT path FROM files").fetchall()
        for r in rows:
            if r["path"] not in seen_rel:
                self.remove_file(Path(r["path"]))
                stats.files_removed += 1

        stats.chunks_written = self.conn.execute(
            "SELECT COUNT(*) AS c FROM chunks"
        ).fetchone()["c"]

        # Lazy backfill: any chunk whose embedding_model is NULL or differs
        # from the active pipeline fingerprint gets re-embedded. Failed
        # batches don't block — search degrades to BM25 for those rows.
        if self.embedder is not None:
            embedded, failed = self.embed_pending()
            stats.chunks_embedded = embedded
            stats.chunks_embed_failed = failed

        return stats

    # -- embedding backfill ------------------------------------------------

    def embed_pending(self, limit: int | None = None) -> tuple[int, int]:
        """Embed chunks whose fingerprint differs from the active pipeline.

        Returns ``(embedded_count, failed_count)``. No-op without an
        ``embedder``. Rebuilds also go through this path: after wipe +
        ``reconcile_all``, every fresh row has ``embedding_model=NULL``
        and gets picked up here.
        """
        if self.embedder is None:
            return (0, 0)

        fingerprint = self.embedder.fingerprint
        sql = (
            "SELECT id, content FROM chunks "
            "WHERE embedding_model IS NULL OR embedding_model != ? "
            "ORDER BY id"
        )
        params: tuple = (fingerprint,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (fingerprint, int(limit))

        pending = self.conn.execute(sql, params).fetchall()
        if not pending:
            return (0, 0)

        embedded = 0
        failed = 0
        bs = max(1, self.embed_batch_size)
        for start in range(0, len(pending), bs):
            batch = pending[start : start + bs]
            texts = [r["content"] for r in batch]
            ids = [int(r["id"]) for r in batch]
            vecs = self.embedder.embed_batch(texts)
            self._write_vectors(ids, vecs, fingerprint)
            for v in vecs:
                if v is None:
                    failed += 1
                else:
                    embedded += 1

        return (embedded, failed)

    def _write_vectors(
        self,
        chunk_ids: list[int],
        vectors: list[list[float] | None],
        fingerprint: str,
    ) -> None:
        """Write a batch of vectors to ``chunks_vec`` + stamp fingerprint."""
        with self.conn:
            for chunk_id, vec in zip(chunk_ids, vectors, strict=True):
                if vec is None:
                    # Leave embedding_model NULL — picked up next sweep.
                    continue
                # Replace any stale row (vec0 enforces PK uniqueness).
                self.conn.execute(
                    "DELETE FROM chunks_vec WHERE chunk_id = ?", (chunk_id,)
                )
                self.conn.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vec)),
                )
                self.conn.execute(
                    "UPDATE chunks SET embedding_model = ? WHERE id = ?",
                    (fingerprint, chunk_id),
                )


# ---------------------------------------------------------------------------
# Conversation row helpers (BUILD_SPEC §7) — used by jarvis.core.conversation
# ---------------------------------------------------------------------------
#
# Sit next to the schema so a future migration sees both the DDL and the
# helpers that depend on its columns. Times are stored as Unix epoch ints to
# match `started_at` / `ended_at` in the existing DDL above.


def insert_conversation(
    conn: sqlite3.Connection,
    conv_id: str,
    started_at: int,
    channel_kind: str,
    channel_id: str | None,
    transcript_path: str,
) -> None:
    """Insert a fresh conversations row. ``transcript_path`` is workspace-relative POSIX."""
    with conn:
        conn.execute(
            "INSERT INTO conversations "
            "(id, started_at, ended_at, channel_kind, channel_id, slug, summary, transcript_path) "
            "VALUES (?, ?, NULL, ?, ?, NULL, NULL, ?)",
            (conv_id, int(started_at), channel_kind, channel_id, transcript_path),
        )


def close_conversation(
    conn: sqlite3.Connection,
    conv_id: str,
    ended_at: int,
    slug: str | None,
    summary: str | None,
) -> None:
    """Stamp ``ended_at``, ``slug``, ``summary`` onto an existing conversation row."""
    with conn:
        conn.execute(
            "UPDATE conversations SET ended_at = ?, slug = ?, summary = ? WHERE id = ?",
            (int(ended_at), slug, summary, conv_id),
        )


def get_open_conversation(
    conn: sqlite3.Connection, channel_kind: str, channel_id: str | None
) -> sqlite3.Row | None:
    """Most-recently-started open conversation for ``(channel_kind, channel_id)``.

    "Open" means ``ended_at IS NULL``. SQLite treats two NULL ``channel_id``
    values as not-equal in normal comparisons, so the query uses ``IS`` for
    the channel_id slot to handle that edge case correctly.
    """
    if channel_id is None:
        return conn.execute(
            "SELECT * FROM conversations "
            "WHERE channel_kind = ? AND channel_id IS NULL AND ended_at IS NULL "
            "ORDER BY started_at DESC LIMIT 1",
            (channel_kind,),
        ).fetchone()
    return conn.execute(
        "SELECT * FROM conversations "
        "WHERE channel_kind = ? AND channel_id = ? AND ended_at IS NULL "
        "ORDER BY started_at DESC LIMIT 1",
        (channel_kind, channel_id),
    ).fetchone()


def list_open_conversations(
    conn: sqlite3.Connection, channel_kinds: Sequence[str]
) -> list[sqlite3.Row]:
    """All open conversations whose ``channel_kind`` is in ``channel_kinds``."""
    kinds = list(channel_kinds)
    if not kinds:
        return []
    placeholders = ",".join("?" for _ in kinds)
    rows = conn.execute(
        f"SELECT * FROM conversations "
        f"WHERE ended_at IS NULL AND channel_kind IN ({placeholders}) "
        f"ORDER BY started_at ASC",
        tuple(kinds),
    ).fetchall()
    return list(rows)
