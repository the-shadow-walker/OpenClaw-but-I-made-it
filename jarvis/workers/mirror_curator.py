"""Mirror curator (BUILD_SPEC §15).

Polls the shared SQLite at ``cfg.mirror.shared_db_path`` every
``cfg.mirror.poll_interval_s`` seconds. When ``MAX(created_at)`` has
increased since the last cycle, fetches all rows, renders curated
Markdown, and atomically writes to ``cfg.mirror.central_context_md``.

Jarvis is the SOLE writer of central_context.md. CMD/Swarm specialists
read it but never touch it (they back off when AGENT_CENTRAL_MIRROR_OWNER=
jarvis is set, shipped by CMD-Claude prior to P9).

One-way output channel — never feeds central_context.md back into
Jarvis's own context. §19#3 (no ReAct internals), §19#11 (no 'session').

Failure policy: every cycle wraps try/except logging at WARNING; three
consecutive failures escalate to ERROR. Curator never crashes the daemon.
``_consecutive_failures`` resets to 0 on every successful cycle. It does
NOT reset on shutdown — if the curator survives 3 failures, escalates,
then succeeds once, the next failure starts the counter back at 1.

Excerpt rules (in this order):
    1. Exact suffix match — ``key.endswith("_brief")`` or
       ``key.endswith("_result")`` is the always-full path. ``_brief_v2``
       does NOT qualify.
    2. Always-full keys: 8KB cap. If value exceeds 8192 bytes the
       curator emits a 200-char excerpt + a publisher-bug warning line
       and logs at WARNING. The 500KB ``_brief`` is a publisher-side
       mistake; the mirror surfaces it visibly rather than silently
       inlining.
    3. Standard keys: 2KB cap. If value exceeds 2048 bytes the curator
       emits a 200-char excerpt + a "see SQLite key" pointer.

Skip rule applies BEFORE excerpt rule. Order: (a) drop expired
(``expires_at < now and expires_at != 0``), (b) drop ephemera
(``expires_at - created_at < 3600``), (c) excerpt long values, (d) group
by namespace, (e) render.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from jarvis.config import MirrorConfig
from jarvis.memory.files import write_markdown_atomic

logger = logging.getLogger(__name__)

__all__ = ["MirrorRow", "MirrorCurator", "render_mirror"]


# Module-level constants — spec §15 verbatim.
_EPHEMERA_TTL_S = 3600
_EXCERPT_THRESHOLD_BYTES = 2048
_ALWAYS_FULL_THRESHOLD_BYTES = 8192
_EXCERPT_LEN = 200
_RECENT_HANDOFFS_N = 20

# Ordered: (prefix, section title). Iteration order = render order.
_NAMESPACE_GROUPS: tuple[tuple[str, str], ...] = (
    ("convo_", "Active conversations"),
    ("project_", "Active projects"),
    ("user_", "User context"),
)
_INFLIGHT_PREFIXES: tuple[str, ...] = (
    "chain_", "gui_", "cmd_", "swarm_", "session_",
)
_ALWAYS_FULL_SUFFIXES: tuple[str, ...] = ("_brief", "_result")

# Set of all prefixes that get their own section. Recent handoffs is the
# bucket for everything NOT in this set, so unknown future namespaces
# (e.g. ``dream_*``) gracefully degrade into Recent handoffs by default.
_GROUPED_PREFIXES: frozenset[str] = frozenset(
    [p for p, _ in _NAMESPACE_GROUPS] + list(_INFLIGHT_PREFIXES)
)


@dataclass(frozen=True)
class MirrorRow:
    """One row from ``shared_context``. ``expires_at`` uses 0.0 as the
    NULL sentinel — SQLite gives us NULL for "no TTL" and we coerce at
    fetch time so downstream code can always do float arithmetic.
    """

    key: str
    value: str
    agent_id: str
    created_at: float
    expires_at: float    # 0.0 sentinel when NULL

    def is_expired(self, now: float) -> bool:
        return self.expires_at != 0.0 and self.expires_at < now

    def is_ephemera(self) -> bool:
        # Skip rule: TTL < 1 hour means in-flight tool plumbing. Only
        # meaningful when expires_at is set (non-zero).
        return self.expires_at != 0.0 and (self.expires_at - self.created_at) < _EPHEMERA_TTL_S


# ---------------------------------------------------------------------------
# Pure renderer
# ---------------------------------------------------------------------------


def _excerpt(row: MirrorRow) -> str:
    """Apply the excerpt rules to a row's value. Returns the rendered
    string for the entry body.

    Rules in order:
      1. Exact suffix match for ``_brief``/``_result`` is always-full.
         ``project_x_brief_v2`` is NOT always-full (publisher convention
         is "use ``_brief`` exactly, or pick a different suffix").
      2. Always-full keys: render full up to 8KB; otherwise excerpt with
         a publisher-bug warning + logger.warning.
      3. Standard keys: render full up to 2KB; otherwise excerpt with
         "see SQLite key" pointer.
    """
    value = row.value
    nbytes = len(value.encode("utf-8"))

    is_always_full = any(row.key.endswith(s) for s in _ALWAYS_FULL_SUFFIXES)

    if is_always_full:
        if nbytes <= _ALWAYS_FULL_THRESHOLD_BYTES:
            return value
        logger.warning(
            "mirror-curator: %s exceeded 8KB always-full cap (%d bytes)",
            row.key, nbytes,
        )
        return (
            f"{value[:_EXCERPT_LEN]}…\n\n"
            f"*(value exceeded 8KB promoted-summary cap; this is a "
            f"publisher bug for key `{row.key}`. See SQLite for full "
            f"content.)*"
        )

    if nbytes <= _EXCERPT_THRESHOLD_BYTES:
        return value
    return (
        f"{value[:_EXCERPT_LEN]}…\n\n"
        f"*(see SQLite key `{row.key}` for full content)*"
    )


def _filter_live(rows: list[MirrorRow], *, now: float) -> list[MirrorRow]:
    """Drop expired rows and ephemera. Order matters: spec §15."""
    return [r for r in rows if not r.is_expired(now) and not r.is_ephemera()]


def _format_entry(row: MirrorRow) -> str:
    """One entry block. Header line + body. Header carries key,
    agent_id, and created_at as a UTC ISO-ish timestamp so the mirror
    is self-describing without the consumer needing to query SQLite.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(row.created_at))
    body = _excerpt(row)
    return f"### `{row.key}`\n*agent: {row.agent_id} · created: {ts}*\n\n{body}\n"


def render_mirror(rows: list[MirrorRow], *, now: float) -> str:
    """Pure renderer. Sections in order:

        # Jarvis Central Context Mirror
        ## Active conversations    (convo_*)
        ## Active projects         (project_*)
        ## In-flight jobs          (chain_*/gui_*/cmd_*/swarm_*/session_*)
        ## User context            (user_*)
        ## Recent handoffs         (last 20, deduped against grouped above)

    Recent handoffs shows the 20 most recent rows whose key prefix is
    NOT in the grouped+in-flight prefix set. A future namespace lands in
    Recent handoffs by default — graceful degradation. Empty board
    emits a marker line so consumers know the curator ran.
    """
    live = _filter_live(rows, now=now)
    # Stable sort: created_at desc primary, key asc as tiebreaker.
    live_sorted = sorted(live, key=lambda r: (-r.created_at, r.key))

    ts_header = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    parts: list[str] = [
        "# Jarvis Central Context Mirror",
        "",
        f"*generated: {ts_header} — sole writer: jarvis (BUILD_SPEC §15)*",
        "",
    ]

    if not live_sorted:
        parts.append("_(shared board empty — curator ran but found no live entries)_")
        parts.append("")
        return "\n".join(parts)

    # --- Active conversations & Active projects (namespace groups, in order)
    for prefix, title in _NAMESPACE_GROUPS:
        if prefix == "user_":
            continue  # rendered below, between in-flight and recent handoffs
        section_rows = [r for r in live_sorted if r.key.startswith(prefix)]
        parts.append(f"## {title}")
        parts.append("")
        if section_rows:
            for r in section_rows:
                parts.append(_format_entry(r))
        else:
            parts.append("_(none)_")
            parts.append("")

    # --- In-flight jobs (any of the in-flight prefixes)
    inflight_rows = [
        r for r in live_sorted if any(r.key.startswith(p) for p in _INFLIGHT_PREFIXES)
    ]
    parts.append("## In-flight jobs")
    parts.append("")
    if inflight_rows:
        for r in inflight_rows:
            parts.append(_format_entry(r))
    else:
        parts.append("_(none)_")
        parts.append("")

    # --- User context
    user_rows = [r for r in live_sorted if r.key.startswith("user_")]
    parts.append("## User context")
    parts.append("")
    if user_rows:
        for r in user_rows:
            parts.append(_format_entry(r))
    else:
        parts.append("_(none)_")
        parts.append("")

    # --- Recent handoffs: anything NOT in a grouped/in-flight prefix.
    handoff_rows = [
        r for r in live_sorted
        if not any(r.key.startswith(p) for p in _GROUPED_PREFIXES)
    ][:_RECENT_HANDOFFS_N]
    parts.append("## Recent handoffs")
    parts.append("")
    if handoff_rows:
        for r in handoff_rows:
            parts.append(_format_entry(r))
    else:
        parts.append("_(none)_")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------


class MirrorCurator:
    """Polls the shared SQLite, renders curated Markdown, atomically
    writes to ``central_context_md``. Thread lifecycle mirrors
    ``WorkspaceWatcher``: ``start()`` is idempotent, ``stop()`` joins.

    Failure policy: every cycle wraps try/except logging at WARNING.
    Three consecutive failures escalate to ERROR. The curator never
    crashes the daemon. ``_consecutive_failures`` resets to 0 on every
    successful cycle.
    """

    def __init__(
        self,
        cfg: MirrorConfig,
        *,
        shared_db_path: Path,
        central_context_md: Path,
        tmp_dir: Path | None = None,
    ) -> None:
        self._cfg = cfg
        self._shared_db_path = Path(shared_db_path)
        self._central_context_md = Path(central_context_md)
        # Default tmp_dir lives next to the mirror file. Same filesystem
        # is critical for the os.replace atomicity guarantee.
        self._tmp_dir = Path(tmp_dir) if tmp_dir is not None else self._central_context_md.parent

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False

        # Per-cycle short-circuit state.
        self._last_seen_max_created: float = -1.0
        self._consecutive_failures: int = 0

    # -- public API --------------------------------------------------------

    def start(self) -> None:
        """Spawn the daemon thread. Idempotent — second call is a noop."""
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-mirror-curator", daemon=True
        )
        self._thread.start()
        self._started = True
        logger.info("mirror-curator: started (poll=%.1fs)", self._cfg.poll_interval_s)

    def stop(self, *, timeout: float = 5.0) -> None:
        """Idempotent shutdown. Sets the stop event and joins."""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._started = False
        logger.info("mirror-curator: stopped")

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        """Loop: cycle then sleep on stop_event for poll_interval_s."""
        while not self._stop_event.is_set():
            try:
                self._cycle()
                self._consecutive_failures = 0
            except Exception:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    logger.error(
                        "mirror-curator: cycle failed (%d consecutive); curator unhealthy",
                        self._consecutive_failures,
                        exc_info=True,
                    )
                else:
                    logger.warning(
                        "mirror-curator: cycle failed (%d consecutive)",
                        self._consecutive_failures,
                        exc_info=True,
                    )
            # Sleep on the stop event so stop() wakes us promptly.
            self._stop_event.wait(self._cfg.poll_interval_s)

    def _cycle(self) -> None:
        """One poll cycle. Reads MAX(created_at); short-circuits if
        unchanged; else fetches all rows, renders, atomic-writes.
        """
        conn = self._connect_ro()
        try:
            cur = conn.execute("SELECT MAX(created_at) FROM shared_context")
            row = cur.fetchone()
            current_max = row[0] if row and row[0] is not None else 0.0

            if current_max <= self._last_seen_max_created:
                return  # nothing new

            rows = self._read_all_rows(conn)
        finally:
            conn.close()

        now = time.time()
        content = render_mirror(rows, now=now)
        write_markdown_atomic(
            self._central_context_md, content, tmp_dir=self._tmp_dir
        )
        self._last_seen_max_created = current_max

    def _connect_ro(self) -> sqlite3.Connection:
        """Open a read-only URI connection. Write attempts raise
        ``OperationalError`` even if the file is rwx — the read-only
        contract is enforced at the driver level, not by convention.
        """
        uri = f"file:{self._shared_db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True, timeout=2.0)

    def _read_all_rows(self, conn: sqlite3.Connection) -> list[MirrorRow]:
        cur = conn.execute(
            "SELECT key, value, agent_id, created_at, expires_at "
            "FROM shared_context"
        )
        out: list[MirrorRow] = []
        for key, value, agent_id, created_at, expires_at in cur.fetchall():
            out.append(MirrorRow(
                key=key,
                value=value if value is not None else "",
                agent_id=agent_id if agent_id is not None else "",
                created_at=float(created_at) if created_at is not None else 0.0,
                expires_at=float(expires_at) if expires_at is not None else 0.0,
            ))
        return out
