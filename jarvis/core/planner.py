"""Multi-phase planner — one LLM call → validated DAG (BUILD_SPEC §13, P8).

The planner is invoked by the ``plan_and_execute`` tool when the LLM
decides a request should be decomposed across specialists (math /
engineer / research / cmd). It runs **one** ``ollama.chat()`` call on
``cfg.llm.fast_model`` with a single tool ``submit_plan`` whose schema
forces the structured DAG shape; the planner parses the tool call,
validates the DAG (target enum, dependency closure, acyclicity, task
length), and returns a frozen :class:`Plan`.

Token budget (§16-D)
--------------------
One LLM call per plan. The system prompt is concise (~150 tokens), no
in-prompt examples, no plan-then-critique loop. A small DAG (3–4 nodes)
fits easily inside the planner's per-call budget; the orchestrator's
per-node summary stays short by reading only ``envelope.summary``.

Anti-patterns explicitly avoided
--------------------------------
* **§13 — router is not an action tag.** The planner is a *tool* the
  LLM invokes; it doesn't bypass tool-calling. The rule-based router
  remains hint-only.
* **§19 #9 — no inline 50KB context.** ``task`` is capped at 1000 chars;
  context flows via ``consume_keys`` references resolved at dispatch
  time.
* **§19 #11 — naming.** ``Plan`` / ``PlanNode`` / ``topo_order``;
  no "session".
* No retry on a malformed plan (§19 #7). One call, one validation pass;
  the tool handler surfaces ``PlanError`` as an envelope error.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field

from jarvis.clients.ollama import OllamaClient

logger = logging.getLogger(__name__)

__all__ = ["Plan", "PlanNode", "PlanError", "plan"]


_VALID_TARGETS = frozenset(
    {"swarm:math", "swarm:engineer", "swarm:research", "cmd:react", "cmd:quick"}
)

_MAX_TASK_CHARS = 1000


class PlanError(ValueError):
    """The LLM emitted a plan that failed validation."""


@dataclass(frozen=True)
class PlanNode:
    """One node in the DAG.

    ``id`` is short, e.g. ``"W1"``; ``target`` ∈ ``_VALID_TARGETS``;
    ``task`` is imperative and ≤ 1000 chars; ``depends_on`` references
    other node ids; ``publish_keys`` lists context keys the node will
    write to the shared board; ``consume_keys`` lists planner-declared
    inputs the orchestrator merges with transitive ancestor publishes.
    """

    id: str
    target: str
    task: str
    depends_on: tuple[str, ...] = ()
    publish_keys: tuple[str, ...] = ()
    consume_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """A validated DAG. ``rationale`` is optional planner prose."""

    nodes: tuple[PlanNode, ...]
    rationale: str = ""

    def topo_order(self) -> list[list[PlanNode]]:
        """Kahn-layered topological sort.

        Returns a list of layers; each layer is a list of nodes whose
        unsatisfied dependencies have just been satisfied. Sibling nodes
        in a layer can run in parallel. Raises :class:`PlanError` if a
        cycle is present (some indegrees never reach zero).
        """
        by_id: dict[str, PlanNode] = {n.id: n for n in self.nodes}
        indegree: dict[str, int] = {n.id: len(n.depends_on) for n in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for n in self.nodes:
            for dep in n.depends_on:
                children[dep].append(n.id)

        ready = deque([nid for nid, deg in indegree.items() if deg == 0])
        layers: list[list[PlanNode]] = []
        seen = 0
        while ready:
            layer_size = len(ready)
            layer: list[PlanNode] = []
            for _ in range(layer_size):
                nid = ready.popleft()
                layer.append(by_id[nid])
                seen += 1
                for c in children[nid]:
                    indegree[c] -= 1
                    if indegree[c] == 0:
                        ready.append(c)
            layers.append(layer)
        if seen != len(self.nodes):
            raise PlanError("plan contains a cycle (indegrees did not reach zero)")
        return layers


# ---------------------------------------------------------------------------
# Planner LLM call
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are the Jarvis multi-phase planner. The user has issued a request \
that benefits from decomposition across specialists. Produce a directed \
acyclic graph of nodes by calling submit_plan exactly once.

Targets you may choose from:
  - swarm:math      — derivations, equations, numerical models
  - swarm:engineer  — implementations: code, scripts, configs
  - swarm:research  — background literature / concept summaries
  - cmd:react       — multi-step shell / file / coding work
  - cmd:quick       — one-shot factual or status questions

Rules:
  - 1 to 6 nodes total. Keep the graph minimal.
  - id is short and unique (e.g. W1, W2, W3).
  - task is imperative, <= 1000 characters, no inlined large context.
  - depends_on lists ids that must complete first; leaves use [].
  - publish_keys lists shared-board keys the node will write.
  - consume_keys lists shared-board keys the node needs to read.
  - Do NOT inline any results yourself; that is the orchestrator's job.

Return one tool call to submit_plan with the entire DAG. No prose.
"""

_SUBMIT_PLAN_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": (
            "Submit the multi-phase plan as a DAG. One call only; the "
            "planner returns immediately after validating the structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rationale": {
                    "type": "string",
                    "description": "One short sentence on why this DAG.",
                },
                "nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "target": {
                                "type": "string",
                                "enum": sorted(_VALID_TARGETS),
                            },
                            "task": {"type": "string"},
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "publish_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "consume_keys": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["id", "target", "task"],
                    },
                },
            },
            "required": ["nodes"],
        },
    },
}


def plan(
    user_text: str,
    *,
    ollama: OllamaClient,
    model: str,
    num_ctx: int,
) -> Plan:
    """Run one LLM call to produce a validated DAG.

    Raises :class:`PlanError` on malformed output (no tool call, unknown
    target, dangling dep, cycle, oversize task).
    """
    if not (user_text or "").strip():
        raise PlanError("planner: empty user_text")

    user_msg = {"role": "user", "content": user_text.strip()}
    try:
        resp = ollama.chat(
            model,
            [user_msg],
            tools=[_SUBMIT_PLAN_TOOL],
            system=_PLANNER_SYSTEM_PROMPT,
            num_ctx=num_ctx,
        )
    except Exception as e:  # noqa: BLE001
        raise PlanError(f"planner: LLM call raised: {e}") from e

    if not resp.tool_calls:
        raise PlanError("planner: model did not emit a submit_plan tool call")

    # Take the first submit_plan call; ignore any others (small models
    # occasionally double-emit). Anything that's not submit_plan is also
    # a planner failure.
    chosen = None
    for tc in resp.tool_calls:
        if tc.name == "submit_plan":
            chosen = tc
            break
    if chosen is None:
        raise PlanError(
            f"planner: expected submit_plan, got {[tc.name for tc in resp.tool_calls]!r}"
        )

    args = chosen.arguments or {}
    if isinstance(args, str):
        # Some models emit JSON strings; coerce.
        try:
            args = json.loads(args)
        except Exception as e:  # noqa: BLE001
            raise PlanError(f"planner: arguments are non-JSON string: {e}") from e
    if not isinstance(args, dict):
        raise PlanError(f"planner: arguments must be an object (got {type(args).__name__})")

    return _validate_plan(args)


def _validate_plan(args: dict) -> Plan:
    """Validate the planner output and produce a frozen :class:`Plan`."""
    raw_nodes = args.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise PlanError("planner: 'nodes' must be a non-empty list")

    nodes: list[PlanNode] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            raise PlanError(f"planner: node[{i}] must be an object")
        nid = raw.get("id")
        if not isinstance(nid, str) or not nid:
            raise PlanError(f"planner: node[{i}].id must be a non-empty string")
        if nid in seen_ids:
            raise PlanError(f"planner: duplicate node id {nid!r}")
        seen_ids.add(nid)

        target = raw.get("target")
        if target not in _VALID_TARGETS:
            raise PlanError(
                f"planner: node {nid!r} has unknown target {target!r}; "
                f"must be one of {sorted(_VALID_TARGETS)}"
            )

        task = raw.get("task")
        if not isinstance(task, str) or not task.strip():
            raise PlanError(f"planner: node {nid!r}.task must be a non-empty string")
        if len(task) > _MAX_TASK_CHARS:
            raise PlanError(
                f"planner: node {nid!r}.task is {len(task)} chars "
                f"(>{_MAX_TASK_CHARS}); §19 #9 — keep tasks short"
            )

        deps = tuple(_coerce_str_list(raw.get("depends_on") or [], f"node {nid!r}.depends_on"))
        pubs = tuple(_coerce_str_list(raw.get("publish_keys") or [], f"node {nid!r}.publish_keys"))
        cons = tuple(_coerce_str_list(raw.get("consume_keys") or [], f"node {nid!r}.consume_keys"))

        nodes.append(PlanNode(
            id=nid,
            target=target,
            task=task,
            depends_on=deps,
            publish_keys=pubs,
            consume_keys=cons,
        ))

    # Validate dep closure (every depends_on id exists).
    ids = {n.id for n in nodes}
    for n in nodes:
        for dep in n.depends_on:
            if dep not in ids:
                raise PlanError(
                    f"planner: node {n.id!r} depends on unknown id {dep!r}"
                )

    rationale_raw = args.get("rationale", "")
    rationale = rationale_raw if isinstance(rationale_raw, str) else ""

    p = Plan(nodes=tuple(nodes), rationale=rationale)
    # Cycle check (raises PlanError on cycle).
    p.topo_order()
    return p


def _coerce_str_list(value, label: str) -> list[str]:
    """Validate that ``value`` is a list of non-empty strings; return a list."""
    if not isinstance(value, list):
        raise PlanError(f"planner: {label} must be a list of strings")
    out: list[str] = []
    for i, v in enumerate(value):
        if not isinstance(v, str) or not v:
            raise PlanError(f"planner: {label}[{i}] must be a non-empty string")
        out.append(v)
    return out


# Re-export for tests that want to inspect the field default.
_field_for_tests = field  # noqa: F841
