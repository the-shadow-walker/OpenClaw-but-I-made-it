"""Tool registry — JSON-Schema specs + handler dispatch (BUILD_SPEC §6.5).

Wires the four P5 tools into a single registry that the chat loop uses to
present schemas to Ollama and dispatch tool-calls back.

Tools registered by :func:`build_default_registry`:
  * ``memory_search`` — hybrid retriever (group-MEMORY filter applied here).
  * ``memory_get``    — read precise line range from a workspace .md.
  * ``memory_write``  — append to daily log or MEMORY.md.
  * ``delegate``      — stub returning ``{"error": "not implemented"}`` (P7).

A non-existent tool name raises ``KeyError`` — the chat loop catches this
and emits a structured tool_result with the error string, so the model
learns and can recover on the next iteration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.core.conversation import ChannelKind
from jarvis.memory.embeddings import EmbeddingPipeline
from jarvis.memory.tool_get import memory_get_tool
from jarvis.memory.tool_search import memory_search_tool
from jarvis.memory.tool_write import memory_write_tool
from jarvis.memory.workspace import WorkspacePaths

if TYPE_CHECKING:
    from jarvis.clients.cmd import CMDClient
    from jarvis.clients.gmail import GmailClient
    from jarvis.clients.ollama import OllamaClient
    from jarvis.clients.swarm import SwarmClient
    from jarvis.config import JarvisConfig
    from jarvis.core.arbiter import RoleArbiter
    from jarvis.core.conversation import Conversation

__all__ = ["ToolSpec", "ToolRegistry", "build_default_registry"]


@dataclass(frozen=True)
class ToolSpec:
    """One tool: its schema (sent to Ollama) + handler (called locally)."""
    name: str
    description: str
    parameters: dict          # JSON Schema for the arguments
    handler: Callable[..., Any]


class ToolRegistry:
    """Holds ToolSpecs by name. Provides Ollama-shaped schemas + dispatch.

    The schema list is what gets passed to Ollama's ``/api/chat`` ``tools``
    field. Ollama wraps each entry in ``{"type": "function", "function":
    {...}}`` — see :meth:`schemas` for the exact shape.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name!r}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict]:
        """Return the Ollama tool schema list (one dict per tool)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._tools.values()
        ]

    def execute(self, name: str, arguments: dict) -> Any:
        """Look up ``name`` and call ``handler(**arguments)``. Raises KeyError
        for an unknown tool (chat loop handles by emitting a tool_result
        error event); other exceptions propagate so the chat loop can wrap
        them with the ``call_id``.
        """
        spec = self.get(name)  # KeyError on miss
        return spec.handler(**arguments)


# ---------------------------------------------------------------------------
# Default registry — the four P5 tools.
# ---------------------------------------------------------------------------


_MEMORY_SEARCH_PARAMS: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language search query over the workspace.",
        },
        "k": {
            "type": "integer",
            "description": "Max results to return (default 5).",
            "default": 5,
        },
        "file_kinds": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Optional file_kind filter. In group conversations, "
                "'memory' is forced out regardless of input."
            ),
        },
    },
    "required": ["query"],
}

_MEMORY_GET_PARAMS: dict = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Workspace-relative path (no .. traversal, no absolute).",
        },
        "start_line": {"type": "integer", "description": "1-indexed inclusive start line."},
        "end_line":   {"type": "integer", "description": "1-indexed inclusive end line."},
    },
    "required": ["file_path", "start_line", "end_line"],
}

_MEMORY_WRITE_PARAMS: dict = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "Text to append."},
        "where": {
            "type": "string",
            "enum": ["daily", "memory"],
            "default": "daily",
            "description": (
                "'daily' (default) appends a timestamped line to today's daily log. "
                "'memory' appends a bullet to MEMORY.md (explicit-remember only)."
            ),
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional hashtag list (daily only).",
        },
    },
    "required": ["content"],
}

_DELEGATE_PARAMS: dict = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "enum": [
                "cmd:quick", "cmd:react", "cmd:chain", "cmd:gui", "cmd:blue",
                "swarm:math", "swarm:engineer",
                "swarm:research", "swarm:deep_search",
            ],
            "description": "Which specialist to route to.",
        },
        "task": {"type": "string", "description": "Task description for the specialist."},
    },
    "required": ["target", "task"],
}


_EMAIL_SEARCH_PARAMS: dict = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Gmail search syntax. Empty returns most recent. Examples: "
                "'is:unread', 'from:foo@bar.com', 'subject:invoice', "
                "'newer_than:2d', 'has:attachment'. Combine: "
                "'from:school is:unread newer_than:7d'."
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "Max messages to return (default 10).",
            "default": 10,
        },
    },
    "required": [],
}

_EMAIL_READ_PARAMS: dict = {
    "type": "object",
    "properties": {
        "email_id": {
            "type": "string",
            "description": "Gmail message id (from email_search results).",
        },
    },
    "required": ["email_id"],
}

_EMAIL_SEND_PARAMS: dict = {
    "type": "object",
    "properties": {
        "to": {"type": "string", "description": "Primary recipient(s), comma-separated."},
        "subject": {"type": "string", "description": "Subject line."},
        "body": {"type": "string", "description": "Plain-text body."},
        "cc": {"type": "string", "description": "Optional CC recipient(s)."},
        "bcc": {"type": "string", "description": "Optional BCC recipient(s)."},
    },
    "required": ["to", "subject", "body"],
}

_EMAIL_DRAFT_PARAMS: dict = _EMAIL_SEND_PARAMS  # same shape, different verb


_PLAN_AND_EXECUTE_PARAMS: dict = {
    "type": "object",
    "properties": {
        "user_request": {
            "type": "string",
            "description": (
                "The user's full request. The planner decomposes it into "
                "a DAG of specialist sub-tasks, then dispatches each in "
                "dependency order."
            ),
        },
    },
    "required": ["user_request"],
}


def _delegate_stub(**kwargs) -> dict:  # type: ignore[no-untyped-def]
    """Pre-P7 stub — returns a structured 'not implemented' result.

    Registered when ``build_default_registry`` is called without all four
    delegation dependencies (cmd_client / arbiter / conversation /
    shared_board) wired — preserves backward compatibility for tests
    that don't bring up the daemon-scoped infra. Returning rather than
    raising lets the LLM learn and decide whether to apologize / answer
    directly, instead of crashing the tool loop.
    """
    return {
        "error": "delegate not implemented (lands in P7)",
        "target": kwargs.get("target"),
    }


def _build_real_delegate_handler(
    *,
    cmd_client: CMDClient,
    arbiter: RoleArbiter,
    conversation: Conversation,
    paths: WorkspacePaths,
    shared_board: Path,
    swarm_client: SwarmClient | None = None,
):
    """Bind delegation deps into a handler callable for the registry."""
    # Local import to avoid a cycle at module import time.
    from jarvis.core.invoker import dispatch as _dispatch

    def _handler(**kwargs) -> dict:  # type: ignore[no-untyped-def]
        target = kwargs.get("target")
        task = kwargs.get("task", "")
        context_keys = kwargs.get("context_keys")
        return _dispatch(
            target=target,
            task=task,
            conversation=conversation,
            paths=paths,
            shared_board=shared_board,
            cmd_client=cmd_client,
            arbiter=arbiter,
            swarm_client=swarm_client,
            context_keys=context_keys,
        )

    return _handler


def _build_plan_and_execute_handler(
    *,
    ollama: OllamaClient,
    cfg: JarvisConfig,
    cmd_client: CMDClient,
    swarm_client: SwarmClient,
    arbiter: RoleArbiter,
    conversation: Conversation,
    paths: WorkspacePaths,
    shared_board: Path,
):
    """Bind planner+orchestrator deps into a single tool handler.

    The handler returns the canonical contract envelope shape so the
    LLM sees an identical result whether the request was a 1-node plan
    (fast-path through ``invoker.dispatch``) or a multi-node DAG (through
    the orchestrator). PlanError is surfaced as ``error`` on the
    envelope; the chat loop never sees a raw exception (§16-F).
    """
    from jarvis.core.invoker import dispatch as _dispatch
    from jarvis.core.orchestrator import execute as _orchestrator_execute
    from jarvis.core.planner import PlanError
    from jarvis.core.planner import plan as _plan

    def _handler(**kwargs) -> dict:  # type: ignore[no-untyped-def]
        user_request = kwargs.get("user_request") or ""
        if not isinstance(user_request, str) or not user_request.strip():
            return _err_envelope("plan_and_execute: empty user_request")

        try:
            plan_obj = _plan(
                user_request,
                ollama=ollama,
                model=cfg.llm.fast_model,
                num_ctx=cfg.llm.context_window,
            )
        except PlanError as e:
            return _err_envelope(f"plan failed: {e}")
        except Exception as e:  # noqa: BLE001
            return _err_envelope(f"planner: unexpected error: {e}")

        # Single-node fast-path (plan pre-decision #8).
        if len(plan_obj.nodes) == 1:
            node = plan_obj.nodes[0]
            envelope = _dispatch(
                target=node.target,
                task=node.task,
                conversation=conversation,
                paths=paths,
                shared_board=shared_board,
                cmd_client=cmd_client,
                arbiter=arbiter,
                swarm_client=swarm_client,
                context_keys=list(node.consume_keys) if node.consume_keys else None,
                snapshot_label=f"pre_{node.id}_{node.target.replace(':', '_')}",
            )
            return {
                "success": bool(envelope.get("success")),
                "summary": envelope.get("summary"),
                "deliverables": list(envelope.get("deliverables") or []),
                "context_keys_written": list(
                    envelope.get("context_keys_written") or []
                ),
                "sidechain_path": envelope.get("sidechain_path"),
                "error": envelope.get("error"),
            }

        # Multi-node path.
        result = _orchestrator_execute(
            plan_obj,
            conversation=conversation,
            paths=paths,
            shared_board=shared_board,
            cmd_client=cmd_client,
            swarm_client=swarm_client,
            arbiter=arbiter,
            cfg=cfg,
        )
        success = result.failed_node_id is None
        # Aggregate context_keys_written across nodes.
        keys_written: list[str] = []
        seen_keys: set[str] = set()
        for nr in result.results.values():
            for k in nr.envelope.get("context_keys_written") or []:
                if k not in seen_keys:
                    seen_keys.add(k)
                    keys_written.append(k)
        err_msg: str | None = None
        if not success:
            failed = result.results.get(result.failed_node_id or "")
            if failed is not None:
                err_msg = (
                    f"node {result.failed_node_id!r} failed: "
                    f"{failed.envelope.get('error') or '(no error)'}"
                )
            else:
                err_msg = f"node {result.failed_node_id!r} failed"
        return {
            "success": success,
            "summary": result.summary,
            "deliverables": list(result.deliverable_paths),
            "context_keys_written": keys_written,
            "sidechain_path": None,
            "error": err_msg,
        }

    return _handler


def _err_envelope(message: str) -> dict:
    """Local helper — contract envelope shape with ``error=message``."""
    return {
        "success": False,
        "summary": None,
        "deliverables": [],
        "context_keys_written": [],
        "sidechain_path": None,
        "error": message,
    }


def build_default_registry(
    *,
    conn,
    embedder: EmbeddingPipeline | None,
    paths: WorkspacePaths,
    channel_kind: ChannelKind,
    cmd_client: CMDClient | None = None,
    arbiter: RoleArbiter | None = None,
    conversation: Conversation | None = None,
    shared_board: Path | None = None,
    swarm_client: SwarmClient | None = None,
    ollama: OllamaClient | None = None,
    cfg: JarvisConfig | None = None,
    gmail_client: GmailClient | None = None,
) -> ToolRegistry:
    """Wire the four P5 tools into a fresh registry.

    ``channel_kind`` is captured here so the group-MEMORY filter is applied
    inside ``memory_search`` regardless of what the model passes.
    """
    workspace_root: Path = paths.root
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="memory_search",
            description=(
                "Hybrid BM25 + vector search over the workspace memory. "
                "Returns up to k chunks ranked by relevance. Use this first "
                "when the user references something they may have told you "
                "before."
            ),
            parameters=_MEMORY_SEARCH_PARAMS,
            handler=lambda **kw: memory_search_tool(
                **kw, conn=conn, embedder=embedder, channel_kind=channel_kind
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="memory_get",
            description=(
                "Read a precise line range from a workspace markdown file. "
                "Use this after memory_search returns a chunk and you want "
                "more surrounding context."
            ),
            parameters=_MEMORY_GET_PARAMS,
            handler=lambda **kw: memory_get_tool(**kw, workspace_root=workspace_root),
        )
    )
    registry.register(
        ToolSpec(
            name="memory_write",
            description=(
                "Persist a snippet to the workspace. Default 'daily' appends "
                "to today's daily log; 'memory' (explicit user request only) "
                "adds a bullet to MEMORY.md."
            ),
            parameters=_MEMORY_WRITE_PARAMS,
            handler=lambda **kw: memory_write_tool(**kw, paths=paths),
        )
    )
    delegation_ready = (
        cmd_client is not None
        and arbiter is not None
        and conversation is not None
        and shared_board is not None
    )
    if delegation_ready:
        delegate_handler = _build_real_delegate_handler(
            cmd_client=cmd_client,    # type: ignore[arg-type]
            arbiter=arbiter,          # type: ignore[arg-type]
            conversation=conversation,  # type: ignore[arg-type]
            paths=paths,
            shared_board=shared_board,  # type: ignore[arg-type]
            swarm_client=swarm_client,
        )
        delegate_description = (
            "Delegate this turn to a specialist. Supports 'cmd:quick' "
            "(one-shot shell question), 'cmd:react' (multi-step coding/"
            "file/shell task), and 'swarm:math'/'swarm:engineer'/"
            "'swarm:research' (Swarm specialists). 'cmd:chain', "
            "'cmd:gui', 'cmd:blue' currently return 'not implemented'."
        )
    else:
        delegate_handler = _delegate_stub
        delegate_description = (
            "Route a sub-task to a specialist (CMD or Swarm). Returns an "
            "'error: not implemented' object in P5; routes to specialists "
            "in P7+."
        )

    registry.register(
        ToolSpec(
            name="delegate",
            description=delegate_description,
            parameters=_DELEGATE_PARAMS,
            handler=delegate_handler,
        )
    )

    # P8: register ``plan_and_execute`` only when every dependency is
    # wired. The planner needs ollama + cfg; the orchestrator needs the
    # full delegation surface (cmd + swarm + arbiter + conversation).
    plan_ready = (
        delegation_ready
        and swarm_client is not None
        and ollama is not None
        and cfg is not None
    )
    # Email tools (P11): only when gmail_client is wired.
    if gmail_client is not None:
        from jarvis.mail.tool_email import (
            email_draft_tool,
            email_read_tool,
            email_search_tool,
            email_send_tool,
        )

        registry.register(
            ToolSpec(
                name="email_search",
                description=(
                    "List or search Gmail messages. Returns compact "
                    "summaries (id, from, subject, date, snippet, "
                    "unread). Use Gmail search syntax in 'query'."
                ),
                parameters=_EMAIL_SEARCH_PARAMS,
                handler=lambda **kw: email_search_tool(**kw, gmail=gmail_client),
            )
        )
        registry.register(
            ToolSpec(
                name="email_read",
                description=(
                    "Fetch one Gmail message in full (headers + body). "
                    "Body is truncated at 8000 chars; check "
                    "'body_truncated' if you need to know."
                ),
                parameters=_EMAIL_READ_PARAMS,
                handler=lambda **kw: email_read_tool(**kw, gmail=gmail_client),
            )
        )
        registry.register(
            ToolSpec(
                name="email_send",
                description=(
                    "Send a plain-text email. ALWAYS confirm with the "
                    "user (recipient, subject, body) before calling — "
                    "this leaves the system. Prefer email_draft if the "
                    "user hasn't explicitly said 'send it'."
                ),
                parameters=_EMAIL_SEND_PARAMS,
                handler=lambda **kw: email_send_tool(**kw, gmail=gmail_client),
            )
        )
        registry.register(
            ToolSpec(
                name="email_draft",
                description=(
                    "Save a Gmail draft (does NOT send). Safer default "
                    "than email_send — user can review in Gmail UI "
                    "before they hit send themselves."
                ),
                parameters=_EMAIL_DRAFT_PARAMS,
                handler=lambda **kw: email_draft_tool(**kw, gmail=gmail_client),
            )
        )

    if plan_ready:
        plan_handler = _build_plan_and_execute_handler(
            ollama=ollama,            # type: ignore[arg-type]
            cfg=cfg,                  # type: ignore[arg-type]
            cmd_client=cmd_client,    # type: ignore[arg-type]
            swarm_client=swarm_client,  # type: ignore[arg-type]
            arbiter=arbiter,          # type: ignore[arg-type]
            conversation=conversation,  # type: ignore[arg-type]
            paths=paths,
            shared_board=shared_board,  # type: ignore[arg-type]
        )
        registry.register(
            ToolSpec(
                name="plan_and_execute",
                description=(
                    "Decompose a multi-phase request into a DAG of "
                    "specialist sub-tasks (math / engineer / research / "
                    "code) and execute them in dependency order. For "
                    "requests bundling research + math + implementation, "
                    "prefer this over chaining 'delegate' calls yourself."
                ),
                parameters=_PLAN_AND_EXECUTE_PARAMS,
                handler=plan_handler,
            )
        )
    return registry
