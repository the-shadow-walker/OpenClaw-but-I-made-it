"""FastAPI app for the Jarvis daemon (BUILD_SPEC §6.5, §7, §17, §20).

Three endpoints:

  * ``GET  /api/health``  — liveness probe + degraded-mode visibility.
  * ``POST /api/session`` — returns a fresh ``conv_id`` for a channel. The
                            "session token" from the spec is just a
                            conversation handle; the network is trusted
                            (Cloudflare Access fronts the daemon, §20). The
                            response key is ``conv_id``, not ``session_id``
                            — anti-pattern §19 #11.
  * ``POST /api/chat``    — drives one user turn through ``run_turn`` and
                            streams events as ``application/x-ndjson`` (one
                            JSON object per line). Errors during the stream
                            yield a final ``{"type": "error", ...}`` line
                            and close cleanly — never raise out of the
                            generator (a 500 mid-stream is unparseable).

Auth: none at this layer. Cloudflare Access fronts the daemon. Adding a
shared-secret check creates a "is auth working?" surface area that CF
already owns; if a future deployment sits without a tunnel, that's when we
add a header check — not now.

Token streaming inside the model response is **not** implemented in P5 —
each assistant turn lands as one ``delta`` event. NDJSON streams *events*,
not characters. P5+ optimization.
"""

from __future__ import annotations

import json
import logging
import socket
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jarvis.clients.cmd import CMDClient
from jarvis.clients.ollama import OllamaClient
from jarvis.clients.swarm import SwarmClient
from jarvis.config import JarvisConfig
from jarvis.core.arbiter import RoleArbiter
from jarvis.core.chat import run_turn
from jarvis.core.conversation import (
    Conversation,
    ConversationConfig,
)
from jarvis.core.tools import build_default_registry
from jarvis.memory.embeddings import EmbeddingPipeline
from jarvis.memory.index import Indexer
from jarvis.memory.workspace import WorkspacePaths

logger = logging.getLogger(__name__)

__all__ = ["create_app", "ChatRequest", "SessionRequest", "SessionResponse"]


_CHANNEL_KIND = Literal["dm", "group", "heartbeat", "cli"]


def _default_channel_id() -> str:
    return f"cli-{socket.gethostname()}"


class SessionRequest(BaseModel):
    channel_kind: _CHANNEL_KIND = "cli"
    channel_id: str = Field(default_factory=_default_channel_id)


class SessionResponse(BaseModel):
    conv_id: str
    channel_kind: str


class ChatRequest(BaseModel):
    conv_id: str
    text: str
    channel_kind: _CHANNEL_KIND = "cli"
    channel_id: str = Field(default_factory=_default_channel_id)
    active_project_slug: str | None = None


def create_app(
    *,
    paths: WorkspacePaths,
    cfg: JarvisConfig,
    indexer: Indexer,
    ollama: OllamaClient,
    embedder: EmbeddingPipeline | None,
    cmd_client: CMDClient | None = None,
    swarm_client: SwarmClient | None = None,
    arbiter: RoleArbiter | None = None,
) -> FastAPI:
    """Build the FastAPI app with all dependencies pre-wired.

    The app instance carries a single, long-lived sqlite connection
    (``indexer.conn``); chat handlers run on FastAPI's worker threads and
    do read-only work on that connection (memory_search) plus disk writes
    (memory_write) — the watcher reconciles the writes back into the DB.
    No new threads touch the indexer's connection.

    ``cmd_client`` and ``arbiter`` are optional — when both are wired,
    the chat tool registry binds the real ``delegate`` handler. When
    either is ``None``, the registry registers the legacy stub (preserves
    backward compatibility for tests that don't bring up CMD).

    ``swarm_client`` is optional — when wired together with ``ollama``
    and ``cfg``, the chat tool registry registers the P8 ``plan_and_execute``
    multi-phase planner tool. Without it, swarm:* delegation lands in a
    degraded-mode envelope (`error="Swarm client not wired"`).
    """
    app = FastAPI(title="Jarvis", version="0.1.0")
    convo_cfg = ConversationConfig.from_jarvis_config(cfg)
    # Process-scoped: one arbiter for the lifetime of the app. P7 never
    # calls set_master (no cmd:code/cmd:gui targets routed); P10 wires it.
    arbiter = arbiter if arbiter is not None else RoleArbiter()

    # Log a single, predictable "server listening" line once uvicorn's lifespan
    # startup completes — used by the subprocess test (and `journalctl -u
    # jarvis`) to confirm the daemon is fully up. Lifespan startup runs after
    # uvicorn binds the socket but before it accepts requests.
    @app.on_event("startup")
    def _log_server_listening() -> None:
        logger.info(
            "daemon: server listening on %s:%d", cfg.server.host, cfg.server.port
        )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "embedder": embedder.fingerprint if embedder is not None else "degraded",
            "watcher": "running",
        }

    @app.post("/api/session", response_model=SessionResponse)
    def session(req: SessionRequest) -> SessionResponse:
        conv = Conversation.open(
            channel_kind=req.channel_kind,
            channel_id=req.channel_id,
            paths=paths,
            conn=indexer.conn,
            cfg=convo_cfg,
        )
        # Release the file descriptor; the chat handler reopens for append.
        # We don't `close()` the row — that would stamp ended_at.
        conv.__exit__(None, None, None)
        return SessionResponse(conv_id=conv.conv_id, channel_kind=conv.channel_kind)

    @app.post("/api/chat")
    def chat(req: ChatRequest) -> StreamingResponse:
        return StreamingResponse(
            _ndjson_stream(req, paths=paths, cfg=cfg, indexer=indexer,
                           ollama=ollama, embedder=embedder, convo_cfg=convo_cfg,
                           cmd_client=cmd_client, swarm_client=swarm_client,
                           arbiter=arbiter),
            media_type="application/x-ndjson",
        )

    return app


def _ndjson_stream(
    req: ChatRequest,
    *,
    paths: WorkspacePaths,
    cfg: JarvisConfig,
    indexer: Indexer,
    ollama: OllamaClient,
    embedder: EmbeddingPipeline | None,
    convo_cfg: ConversationConfig,
    cmd_client: CMDClient | None,
    swarm_client: SwarmClient | None,
    arbiter: RoleArbiter,
) -> Iterator[bytes]:
    """Wrap ``run_turn`` and emit one NDJSON line per yielded event.

    Errors during the stream emit a final ``{"type": "error", ...}`` plus a
    ``done`` line and close cleanly — they never propagate out of the
    generator (which would surface as a 500 mid-stream the client can't
    reliably parse).
    """
    try:
        conv = Conversation.open(
            channel_kind=req.channel_kind,
            channel_id=req.channel_id,
            paths=paths,
            conn=indexer.conn,
            cfg=convo_cfg,
        )
        # Resume might have produced a different conv_id than the client
        # asked for (idle-reset between session() and chat()). The spec
        # accepts that — the session/chat handle is advisory, not binding.
    except Exception as e:  # noqa: BLE001
        logger.exception("chat: failed to open conversation")
        yield _line({"type": "error", "message": f"open conversation: {e}"})
        yield _line({"type": "done", "stop_reason": "error"})
        return

    registry = build_default_registry(
        conn=indexer.conn,
        embedder=embedder,
        paths=paths,
        channel_kind=req.channel_kind,
        cmd_client=cmd_client,
        swarm_client=swarm_client,
        arbiter=arbiter,
        conversation=conv,
        shared_board=cfg.paths.shared_board,
        ollama=ollama,
        cfg=cfg,
    )

    try:
        for evt in run_turn(
            user_text=req.text,
            conversation=conv,
            paths=paths,
            cfg=cfg,
            ollama=ollama,
            registry=registry,
            channel_kind=req.channel_kind,
            active_project_slug=req.active_project_slug,
            embedder=embedder,
        ):
            # Tag the conversation id onto every event so multi-conv
            # clients can demux a future shared stream. No-op for the
            # single-conv CLI.
            evt = {"conv_id": conv.conv_id, **evt}
            yield _line(evt)
    except Exception as e:  # noqa: BLE001
        logger.exception("chat: run_turn raised mid-stream")
        yield _line({"type": "error", "message": f"chat error: {e}",
                     "conv_id": conv.conv_id})
        yield _line({"type": "done", "stop_reason": "error",
                     "conv_id": conv.conv_id})
    finally:
        # Release the FD without stamping ended_at; the row stays open for
        # the next turn.
        conv.__exit__(None, None, None)


def _line(evt: dict) -> bytes:
    return (json.dumps(evt, ensure_ascii=False) + "\n").encode("utf-8")
