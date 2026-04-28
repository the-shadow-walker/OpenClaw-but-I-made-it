"""Canonical ``jarvis`` console-script entry point.

Subcommands (P2 + P3):
  * ``jarvis reconcile``  — full reindex + embed-pending of the workspace.
  * ``jarvis search Q``   — hybrid BM25 + vector search (``--bm25-only`` to
                             force the degraded path; ``--kind`` repeatable
                             to filter by file_kind).
  * ``jarvis daemon``     — thin shim into ``jarvis.run:main`` (the P0 stub).
                             P5 grows this into the FastAPI server without
                             changing the entry point or any other subcommand.

Two entry points planning to merge later is two entry points planning to
break each other later — converge upfront. ``pyproject.toml`` flips the
console script to ``jarvis.cli:main`` in P2; the systemd unit will switch
to ``jarvis daemon`` in P5 when we actually need the daemon to run.

Env: ``JARVIS_WORKSPACE`` overrides ``cfg.paths.workspace`` post-load (paired
with ``JARVIS_CONFIG`` from P0). Used by the exit-criterion eval against
``/tmp/p2-eval`` / ``/tmp/p3-eval``.
"""

from __future__ import annotations

import argparse
import logging
import sys

from jarvis.bootstrap import apply_workspace_override, setup_logging
from jarvis.config import load_config
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.embeddings import EmbeddingPipeline, build_provider_from_config
from jarvis.memory.index import Indexer
from jarvis.memory.search import SearchOptions, memory_search, search_bm25
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

logger = logging.getLogger(__name__)

# Per-process tracking so we emit the embedder fingerprint at INFO exactly
# once per fingerprint, then DEBUG for subsequent uses. CLI invocations are
# fresh processes so they always log INFO once (current behavior preserved);
# the P5 daemon will log INFO at boot then DEBUG per request, avoiding
# journalctl spam.
_LOGGED_EMBEDDER_FINGERPRINTS: set[str] = set()


def _log_embedder_fingerprint(fingerprint: str) -> None:
    if fingerprint in _LOGGED_EMBEDDER_FINGERPRINTS:
        logger.debug("embeddings: %s", fingerprint)
    else:
        logger.info("embeddings: %s", fingerprint)
        _LOGGED_EMBEDDER_FINGERPRINTS.add(fingerprint)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("reconcile", help="full reindex + embed-pending of workspace/")

    sp = sub.add_parser("search", help="hybrid BM25 + vector search over the index")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=10)
    sp.add_argument("--kind", action="append", help="filter by file_kind; repeatable")
    sp.add_argument(
        "--bm25-only",
        action="store_true",
        help="force BM25-only path (skip embeddings even if available)",
    )
    sp.add_argument(
        "--show-components",
        action="store_true",
        help="print bm25/vector/decay components alongside the fused score",
    )

    sub.add_parser(
        "daemon",
        help="run the Jarvis server (P0 stub today, real FastAPI in P5)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "daemon":
        # Thin shim — the P0 stub does its own thing and exits 0.
        # P5 grows the daemon subcommand without touching the entry point.
        from jarvis.run import main as daemon_main
        return daemon_main([])

    # All non-daemon subcommands share the workspace + indexer setup.
    cfg = load_config()
    cfg = apply_workspace_override(cfg)

    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)

    resolved_kind = configure_tokenizer(cfg.llm.tokenizer)
    if resolved_kind == "qwen-native":
        logger.info("using tokenizer: qwen-native (Qwen/Qwen2.5-3B)")
    elif cfg.llm.tokenizer == "qwen-native":
        # Already ERROR-logged inside configure_tokenizer; no redundant emit.
        pass
    else:
        logger.info("using tokenizer: approximation (4-char rule)")

    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)

    # Build the embedder once. The pipeline uses the indexer connection as
    # its cache so embedding_cache lives in the same DB. None → degraded
    # mode (BM25-only); we log it loudly so operators can see it.
    bm25_only = bool(getattr(args, "bm25_only", False))
    embedder: EmbeddingPipeline | None = None
    if not bm25_only:
        provider = build_provider_from_config(cfg)
        if provider is None:
            logger.warning(
                "embeddings: degraded mode (no provider buildable); "
                "search falls back to BM25-only and reconcile skips vector writes"
            )
        else:
            embedder = EmbeddingPipeline(provider, cache_conn=indexer.conn)
            _log_embedder_fingerprint(embedder.fingerprint)

    indexer.embedder = embedder

    try:
        if args.cmd == "reconcile":
            stats = indexer.reconcile_all()
            print(f"reconciled: {stats}")
        elif args.cmd == "search":
            if embedder is None or bm25_only:
                results = search_bm25(
                    indexer.conn, args.query, k=args.k, file_kinds=args.kind
                )
            else:
                results = memory_search(
                    indexer.conn,
                    args.query,
                    embedder=embedder,
                    options=SearchOptions(k=args.k, file_kinds=args.kind),
                )
            for r in results:
                print(
                    f"[{r.score:7.3f}] {r.file_path}:{r.start_line}-{r.end_line}"
                    f"  {r.heading_path or ''}"
                )
                snippet = r.content.strip().replace("\n", " ")
                if len(snippet) > 200:
                    snippet = snippet[:197] + "..."
                print(f"           {snippet}")
                if args.show_components and r.score_components:
                    parts = ", ".join(
                        f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in r.score_components.items()
                    )
                    print(f"           components: {parts}")
    finally:
        indexer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
