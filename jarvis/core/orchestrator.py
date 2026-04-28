"""DAG orchestrator — topological execution of a :class:`Plan` (BUILD_SPEC §10, P8).

Given a validated :class:`Plan`, the orchestrator dispatches each node
through ``invoker.dispatch`` in topological order. Sibling nodes within
a layer run in parallel via a :class:`concurrent.futures.ThreadPoolExecutor`;
the per-client semaphores in :class:`SwarmClient` and :class:`CMDClient`
provide per-specialist concurrency caps.

Failure policy — first-failure-stops, "wait-then-abort"
-------------------------------------------------------
``concurrent.futures.ThreadPoolExecutor`` cannot cancel a *running*
task — :meth:`Future.cancel` returns ``False`` once execution has
begun. So the precise semantics here are:

1. Within a layer, when one node returns ``envelope.success=False``,
   we **wait for all other in-flight siblings to complete normally**
   (their deliverables may land on disk; we ignore them when building
   the result).
2. **Unstarted children at deeper levels are then skipped** — the
   topo loop checks the failure flag before scheduling the next layer.
3. The orchestrator returns an :class:`OrchestrationResult` with
   ``failed_node_id`` set; the LLM-facing summary surfaces the failing
   node's ``envelope.error`` plus whatever earlier deliverables landed.

This is "wait-then-abort," not "best-effort" and not "kill-in-flight."
A future ``policy=`` knob (best-effort) is plan pre-decision #1 — out
of scope for P8.

Anti-patterns explicitly avoided
--------------------------------
* **§19 #1** — final reply built from envelope fields only; we never
  open deliverable files or read sidechain content.
* **§19 #3 / §10.4** — we never bind ``execution_log`` /
  ``sidechain_path`` content into the orchestrator's summary.
* **§19 #7** — no auto-retry. First failure stops the DAG.
* **§19 #11** — names: ``node`` / ``plan`` / ``result``; never "session".

``consume_keys`` merge semantics (per node)
-------------------------------------------
For each node, dispatch receives ``context_keys`` = the dedup-by-first
union of:

    [pub_key for anc in transitive_ancestors_in_DAG_order
            for pub_key in anc.publish_keys]
    + node.consume_keys

Ancestor publish_keys appear first; the node's own consume_keys come
last. The shared board is the value store; same key from same source =
same value, so dedup is safe. Missing keys at dispatch time are NOT a
failure — the orchestrator logs a warning and continues. A planner
declaring an optional key the specialist can survive without is more
common than a hard pre-condition.

Deliverable de-duplication
--------------------------
If two nodes report the same path in their ``deliverables``, the
orchestrator dedups at the path level and logs a warning — likely
indicates planner overlap (two nodes producing the same artifact).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jarvis.core.invoker import dispatch as invoker_dispatch
from jarvis.core.planner import Plan, PlanNode

if TYPE_CHECKING:
    from jarvis.clients.cmd import CMDClient
    from jarvis.clients.swarm import SwarmClient
    from jarvis.config import JarvisConfig
    from jarvis.core.arbiter import RoleArbiter
    from jarvis.core.conversation import Conversation
    from jarvis.memory.workspace import WorkspacePaths

logger = logging.getLogger(__name__)

__all__ = ["NodeResult", "OrchestrationResult", "execute"]


@dataclass
class NodeResult:
    node: PlanNode
    envelope: dict
    elapsed_ms: int


@dataclass
class OrchestrationResult:
    plan: Plan
    results: dict[str, NodeResult] = field(default_factory=dict)
    failed_node_id: str | None = None
    summary: str = ""
    deliverable_paths: list[str] = field(default_factory=list)


def execute(
    plan: Plan,
    *,
    conversation: Conversation,
    paths: WorkspacePaths,
    shared_board: Path,
    cmd_client: CMDClient | None,
    swarm_client: SwarmClient | None,
    arbiter: RoleArbiter,
    cfg: JarvisConfig | None = None,
    max_workers: int | None = None,
) -> OrchestrationResult:
    """Run ``plan`` through ``invoker.dispatch`` topologically.

    Sibling nodes within a layer run in parallel. First failure stops
    the DAG (wait-then-abort). Returns an :class:`OrchestrationResult`
    with partial results on failure.
    """
    del cfg  # reserved — currently unused; passed by callers for forward-compat.

    layers = plan.topo_order()
    # Default worker pool size: at most the largest layer width, capped
    # at 8 to avoid spinning excess threads for small DAGs.
    widest = max((len(layer) for layer in layers), default=1)
    workers = min(max_workers or widest, 8)
    workers = max(1, workers)

    results: dict[str, NodeResult] = {}
    failed_node_id: str | None = None

    # Pre-compute ancestor closure (DAG order — depends_on is acyclic).
    ancestors = _ancestor_closure(plan)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for layer in layers:
            if failed_node_id is not None:
                # First-failure-stop: skip every deeper layer.
                break

            futures: dict[Future, PlanNode] = {}
            for node in layer:
                merged_keys = _merge_consume_keys(node, ancestors[node.id], plan)
                fut = pool.submit(
                    _dispatch_one,
                    node=node,
                    context_keys=merged_keys,
                    conversation=conversation,
                    paths=paths,
                    shared_board=shared_board,
                    cmd_client=cmd_client,
                    swarm_client=swarm_client,
                    arbiter=arbiter,
                )
                futures[fut] = node

            # Wait for ALL siblings in this layer to complete (wait-then-
            # abort). We do not return early on first failure — that
            # would leave in-flight tasks running with no result handler.
            for fut in list(futures.keys()):
                try:
                    nr = fut.result()
                except Exception as exc:  # noqa: BLE001
                    # invoker.dispatch never raises; if we get here,
                    # something exotic happened. Synthesize an err
                    # envelope.
                    node = futures[fut]
                    logger.exception(
                        "orchestrator: node %s raised unexpectedly", node.id
                    )
                    nr = NodeResult(
                        node=node,
                        envelope={
                            "success": False, "summary": None,
                            "deliverables": [], "context_keys_written": [],
                            "sidechain_path": None,
                            "error": f"orchestrator: {type(exc).__name__}: {exc}",
                        },
                        elapsed_ms=0,
                    )
                results[nr.node.id] = nr
                if not nr.envelope.get("success") and failed_node_id is None:
                    failed_node_id = nr.node.id

    summary, deliverables = _build_result(plan, results, failed_node_id)
    return OrchestrationResult(
        plan=plan,
        results=results,
        failed_node_id=failed_node_id,
        summary=summary,
        deliverable_paths=deliverables,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dispatch_one(
    *,
    node: PlanNode,
    context_keys: list[str],
    conversation: Conversation,
    paths: WorkspacePaths,
    shared_board: Path,
    cmd_client: CMDClient | None,
    swarm_client: SwarmClient | None,
    arbiter: RoleArbiter,
) -> NodeResult:
    """Single-node dispatch wrapper for the thread pool."""
    started = time.monotonic()
    envelope = invoker_dispatch(
        target=node.target,
        task=node.task,
        conversation=conversation,
        paths=paths,
        shared_board=shared_board,
        cmd_client=cmd_client,
        arbiter=arbiter,
        swarm_client=swarm_client,
        context_keys=list(context_keys) if context_keys else None,
        snapshot_label=f"pre_{node.id}_{node.target.replace(':', '_')}",
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return NodeResult(node=node, envelope=envelope, elapsed_ms=elapsed_ms)


def _ancestor_closure(plan: Plan) -> dict[str, list[str]]:
    """Map node id → ordered list of transitive ancestor ids (DAG order)."""
    by_id = {n.id: n for n in plan.nodes}
    out: dict[str, list[str]] = {}
    for n in plan.nodes:
        seen: set[str] = set()
        ordered: list[str] = []
        # BFS over depends_on; preserve first-seen order for ancestor publish merge.
        frontier: list[str] = list(n.depends_on)
        while frontier:
            anc = frontier.pop(0)
            if anc in seen or anc not in by_id:
                continue
            seen.add(anc)
            ordered.append(anc)
            frontier.extend(by_id[anc].depends_on)
        # Reverse so the deepest ancestors come first (DAG-natural order:
        # the root the node depends on is published before the immediate
        # parent). For dedup-by-first-seen this gives the published-first
        # ordering the spec wants.
        ordered.reverse()
        out[n.id] = ordered
    return out


def _merge_consume_keys(
    node: PlanNode, ancestor_ids: list[str], plan: Plan
) -> list[str]:
    """Compute dispatch ``context_keys`` for ``node``.

    Ancestor publish_keys (DAG order) + node.consume_keys, dedup
    preserving first-seen order.
    """
    by_id = {n.id: n for n in plan.nodes}
    seen: set[str] = set()
    out: list[str] = []
    for anc_id in ancestor_ids:
        anc = by_id.get(anc_id)
        if anc is None:
            continue
        for k in anc.publish_keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
    for k in node.consume_keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
        else:
            # Already provided by an ancestor — silently dedup.
            pass
    # Surface a warning for planner-declared keys that no ancestor
    # publishes — a missing optional key is acceptable, but it's worth
    # logging so an operator can see "the planner expected key X."
    for k in node.consume_keys:
        ancestor_publishes = {
            pk for aid in ancestor_ids for pk in by_id.get(aid, node).publish_keys
        }
        if k not in ancestor_publishes:
            logger.warning(
                "orchestrator: node %s consume_key %r not published by any ancestor",
                node.id, k,
            )
    return out


def _build_result(
    plan: Plan,
    results: dict[str, NodeResult],
    failed_node_id: str | None,
) -> tuple[str, list[str]]:
    """Produce ``(summary, deliverable_paths)`` from envelope fields only.

    Never opens deliverable files; never reads ``sidechain_path``
    content (anti-patterns §19 #1, §19 #3).
    """
    lines: list[str] = []
    deliverables_seen: set[str] = set()
    deliverables_ordered: list[str] = []

    # Iterate in plan node order so the summary mirrors the DAG.
    for node in plan.nodes:
        nr = results.get(node.id)
        if nr is None:
            # Node never ran (deeper level, skipped on failure).
            continue
        env = nr.envelope
        ok = bool(env.get("success"))
        head = f"[{node.id}/{node.target}]"
        if ok:
            summary = env.get("summary") or "(no summary)"
            lines.append(f"{head} ok: {summary}")
        else:
            err = env.get("error") or "(no error message)"
            lines.append(f"{head} failed: {err}")

        for path in env.get("deliverables") or []:
            if not isinstance(path, str) or not path:
                continue
            if path in deliverables_seen:
                logger.warning(
                    "orchestrator: deliverable %r reported by multiple nodes "
                    "(planner overlap?)",
                    path,
                )
                continue
            deliverables_seen.add(path)
            deliverables_ordered.append(path)

    if failed_node_id is not None:
        lines.append(
            f"DAG halted at node {failed_node_id!r}; deeper nodes were skipped."
        )

    summary = "\n".join(lines) if lines else "(orchestrator: no nodes ran)"
    return summary, deliverables_ordered


# Re-export for tests.
def _public_dispatch_one(*args: Any, **kwargs: Any) -> NodeResult:  # noqa: D401
    """Test hook — wraps :func:`_dispatch_one`."""
    return _dispatch_one(*args, **kwargs)
