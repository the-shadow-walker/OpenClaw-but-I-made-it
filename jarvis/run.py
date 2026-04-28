"""Jarvis daemon entry point.

P4 brought up the workspace watcher; P5 mounts the FastAPI chat server on top.
The watcher still starts first (so the index is fresh by the time the first
request arrives), then uvicorn takes over the main thread and handles
SIGINT/SIGTERM itself — we drop the threading.Event signal block that P4 used.

Token-counting / context overflow is **not** handled here: the conversation
can overflow the model's context window in P5. Auto-compaction is P6.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from jarvis.bootstrap import apply_workspace_override, setup_logging
from jarvis.clients.cmd import CMDClient
from jarvis.clients.ollama import OllamaClient
from jarvis.clients.swarm import SwarmClient
from jarvis.config import load_config
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.conversation import ConversationConfig
from jarvis.core.scheduler import ResetScheduler
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.embeddings import EmbeddingPipeline, build_provider_from_config
from jarvis.memory.index import Indexer, close_conversation, list_open_conversations
from jarvis.memory.watcher import WorkspaceWatcher
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from jarvis.server import create_app
from jarvis.workers.mirror_curator import MirrorCurator

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    try:
        cfg = load_config()
    except Exception as e:  # noqa: BLE001 — surface any config error and bail.
        print(f"config: FAILED to load: {e}", file=sys.stderr)
        return 1
    cfg = apply_workspace_override(cfg)

    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)

    configure_tokenizer(cfg.llm.tokenizer)

    db_path = paths.index_dir / "memory.sqlite"
    indexer = Indexer(db_path, paths.root)

    provider = build_provider_from_config(cfg)
    if provider is not None:
        indexer.embedder = EmbeddingPipeline(provider, cache_conn=indexer.conn)
        logger.info("embeddings: %s", indexer.embedder.fingerprint)
    else:
        logger.warning(
            "embeddings: degraded mode — watcher will reconcile without vectors"
        )

    watcher = WorkspaceWatcher(paths.root, indexer)
    watcher.start()
    logger.info("daemon: watcher running on %s", paths.root)

    ollama = OllamaClient(cfg.llm.ollama_host)
    cmd_client = CMDClient(
        cfg.orchestration.cmd.base,
        max_concurrent=cfg.orchestration.cmd.max_concurrent,
        quick_timeout_s=cfg.orchestration.cmd.quick_timeout_s,
        react_max_wait_s=cfg.orchestration.cmd.react_max_wait_s,
        chain_max_wait_s=cfg.orchestration.cmd.chain_max_wait_s,
    )
    swarm_client = SwarmClient(
        cfg.orchestration.swarm.base,
        max_concurrent=cfg.orchestration.swarm.max_concurrent,
        dispatch_max_wait_s=cfg.orchestration.swarm.dispatch_max_wait_s,
    )
    arbiter = RoleArbiter()

    convo_cfg = ConversationConfig.from_jarvis_config(cfg)

    def _close_no_summary(channel_kinds: list[str]) -> None:
        """Daily reset callback. P5 stamps ended_at without slug/summary;
        the LLM-generated slug+summary lands in P6 alongside compaction.
        Also drops the per-conversation arbiter master-mode entry so the
        in-memory dict stays bounded."""
        import time as _time
        rows = list_open_conversations(indexer.conn, channel_kinds)
        for row in rows:
            close_conversation(
                indexer.conn, row["id"], int(_time.time()),
                slug=None, summary=None,
            )
            arbiter.reset(row["id"])
        if rows:
            logger.info(
                "scheduler: closed %d open %s conversation(s) at daily boundary",
                len(rows), "/".join(channel_kinds),
            )

    scheduler = ResetScheduler(convo_cfg, on_reset=_close_no_summary)
    scheduler.start()

    mirror_curator: MirrorCurator | None = None
    if cfg.mirror.enabled:
        mirror_curator = MirrorCurator(
            cfg.mirror,
            shared_db_path=cfg.mirror.shared_db_path,
            central_context_md=cfg.mirror.central_context_md,
        )
        mirror_curator.start()

    app = create_app(
        paths=paths, cfg=cfg, indexer=indexer,
        ollama=ollama, embedder=indexer.embedder,
        cmd_client=cmd_client, swarm_client=swarm_client,
        arbiter=arbiter,
    )

    uv_config = uvicorn.Config(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(uv_config)

    # The "server listening" log is emitted from a FastAPI startup hook
    # inside ``create_app`` — the subprocess test (and ``journalctl -u
    # jarvis``) reads it as the all-clear signal.

    try:
        # uvicorn owns SIGINT/SIGTERM here — installs its own handlers and
        # blocks until shutdown completes.
        server.run()
    finally:
        try:
            scheduler.stop()
        finally:
            try:
                watcher.stop()
            finally:
                try:
                    ollama.close()
                finally:
                    try:
                        cmd_client.close()
                    finally:
                        try:
                            swarm_client.close()
                        finally:
                            try:
                                if mirror_curator is not None:
                                    mirror_curator.stop()
                            except Exception:
                                logger.exception("mirror-curator stop failed")
                            indexer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
