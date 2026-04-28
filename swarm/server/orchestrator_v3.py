"""
Swarm 3.11 — OrchestratorV3

Drop-in replacement for OrchestratorV2_1.  Same constructor signature and
process_question() interface.

Routing:
  THEORETICAL        → OrchestratorV2_1 (unchanged delegation)
  ENGINEERING_DESIGN → engineer_mode.run_engineer_mode (unchanged delegation)
  UNKNOWN            → OrchestratorV2_1 (safe fallback)
  MATHEMATICAL       → NEW: PlannerV2 + ReAct wave executor + synthesis
  HYBRID             → NEW: PlannerV2 + targeted research + ReAct waves + synthesis
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    import _paths
except ImportError:
    pass

import asyncio
import json
import re
import requests
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from base_agent import BaseAgent
from core import AgentType

# ── Classifer ────────────────────────────────────────────────────────────────
try:
    from question_classifier import QuestionClassifier, QuestionType
    _HAS_CLASSIFIER = True
except ImportError:
    _HAS_CLASSIFIER = False

# ── Planner V2 ───────────────────────────────────────────────────────────────
try:
    from planner_v2 import PlannerV2, SolvePlan, SubProblem, Requirement
    _HAS_PLANNER_V2 = True
except ImportError:
    _HAS_PLANNER_V2 = False
    Requirement = None
    print("⚠️  OrchestratorV3: planner_v2 not available")

# ── ReAct solver ─────────────────────────────────────────────────────────────
try:
    from react_solver import ReactSolver, SolverResult
    _HAS_REACT = True
except ImportError:
    _HAS_REACT = False
    print("⚠️  OrchestratorV3: react_solver not available")

# ── Sidechain JSONL (Chunk 6) — only writes when SWARM_AS_SUBAGENT=1 ────────
try:
    from sidechain import make_sidechain as _make_sidechain  # type: ignore
    _HAS_SIDECHAIN = True
except ImportError:
    _HAS_SIDECHAIN = False
    _make_sidechain = None  # type: ignore

# ── Search (for targeted HYBRID research) ────────────────────────────────────
try:
    from flexible_search_agent import FlexibleSearchAgent
    _HAS_SEARCH = True
except ImportError:
    _HAS_SEARCH = False

# ── Fallback: delegate to V2_1 ───────────────────────────────────────────────
try:
    from orchestrator_v2_1 import OrchestratorV2_1
    _HAS_V2 = True
except ImportError:
    _HAS_V2 = False
    print("⚠️  OrchestratorV3: orchestrator_v2_1 not available — fallback disabled")

# ── Engineer mode ────────────────────────────────────────────────────────────
try:
    from engineer_mode import run_engineer_mode
    _HAS_ENGINEER = True
except ImportError:
    _HAS_ENGINEER = False


# ── Ollama constants ─────────────────────────────────────────────────────────
_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Swarm 3.17 — Model flipped back to qwen3.6:35b-Grindlewalt (2026-04-26).
# 23 GB on disk, ~27 GB loaded with KV cache — exceeds 20 GB combined VRAM,
# so ~33% CPU spill is expected. MoE (~3-4B active params/tok) keeps
# inference at ~17 tok/s though, and raw coding quality beats the IQ4.
# Env-overridable; SWARM_NUM_CTX caps KV cache.
_MODEL_DEFAULT       = os.getenv("SWARM_MODEL_DEFAULT", "qwen3.6:35b-Grindlewalt")
_MODEL_REASONER      = os.getenv("SWARM_MODEL_REASONER",      _MODEL_DEFAULT)
_MODEL_FALLBACK      = os.getenv("SWARM_MODEL_FALLBACK",      _MODEL_DEFAULT)
_MODEL_CODER         = os.getenv("SWARM_MODEL_CODER",         _MODEL_DEFAULT)
# Swarm 3.18 — Classifier + planner stay on the unified default. With
# format:"json" + retry-on-failure, these tiny structured-JSON tasks are now
# reliable on the big model AND avoid the 100-200s VRAM swap cost of loading
# a smaller secondary (OLLAMA_MAX_LOADED_MODELS=1). Override for benchmarking.
_MODEL_PLANNER       = os.getenv("SWARM_MODEL_PLANNER",       _MODEL_DEFAULT)
_MODEL_SMART_PLANNER = os.getenv("SWARM_MODEL_SMART_PLANNER", _MODEL_DEFAULT)
_NUM_CTX             = int(os.getenv("SWARM_NUM_CTX", "32768"))

# ── Swarm 3.13 kill switches ─────────────────────────────────────────────────
RESIDUAL_LOCK_ORCH = os.getenv("SWARM_RESIDUAL_LOCK_ORCH", "1") != "0"
RESIDUAL_TOLERANCE_REL_ORCH = float(os.getenv("SWARM_RESIDUAL_TOL_ORCH", "1e-6"))
# Swarm 3.14 — Stability Gate (orchestrator side)
STABILITY_GATE_ORCH = os.getenv("SWARM_STABILITY_GATE_ORCH", "1") != "0"
# Swarm 3.14.1 — Plausibility Gate (Lock D): auditor peer-review for suspicious values
PLAUSIBILITY_GATE_ENABLED = os.getenv("SWARM_PLAUSIBILITY_GATE", "1") != "0"
PLAUSIBILITY_AUDITOR_MODEL = os.getenv("SWARM_AUDITOR_MODEL", _MODEL_DEFAULT)


# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorV3:
    """
    Swarm 3.11 top-level dispatcher.
    Drop-in replacement for OrchestratorV2_1.
    """

    def __init__(
        self,
        max_search_concurrent: int = 3,
        enable_verification: bool = True,
        debug: bool = False,
        searxng_url: Optional[str] = None,
        deep_research: bool = False,
        context_window_size: int = 8000,
        date_filter: Optional[str] = None,
        save_markdown: bool = False,
        **kwargs,          # absorb any future/unknown kwargs gracefully
    ):
        self.debug = debug
        self.searxng_url = searxng_url or os.getenv("SEARXNG_URL", "http://10.0.0.58:8080")
        self.date_filter = date_filter
        self.save_markdown = save_markdown
        self.max_search_concurrent = max_search_concurrent
        self.enable_verification = enable_verification
        self.deep_research = deep_research
        self.context_window_size = context_window_size

        self.status = None

        _think_state = "on" if os.getenv("SWARM_THINK", "0") == "1" else "off"
        print(f"🚀 Swarm 3.18 OrchestratorV3  —  default:{_MODEL_DEFAULT} | think:{_think_state}")
        print(f"   planner:{_MODEL_SMART_PLANNER} (json-mode) | reasoner:{_MODEL_REASONER} | coder:{_MODEL_CODER}")
        print("   ✅ ReAct solver pipeline (MATHEMATICAL/HYBRID)")
        print("   ✅ Delegation to V2_1 (THEORETICAL/UNKNOWN)")
        print("   ✅ Delegation to engineer_mode (ENGINEERING_DESIGN)")

    # ── Entry point ───────────────────────────────────────────────────────────

    async def process_question(self, question: str, status=None, job_id: str = "") -> str:
        """Answer any question. Drop-in interface for OrchestratorV2_1."""
        self.status = status
        self._job_id = job_id
        t0 = datetime.now()
        _qtype_val: str = "unknown"   # track before try so except can read it

        print("\n" + "="*70)
        print("🚀 SWARM 3.6 — OrchestratorV3")
        print("="*70)
        print(f"Q: {question[:100]}")
        print("="*70)

        try:
            # ── Phase 0A: Classify ────────────────────────────────────────
            t_classify = time.time()
            print(f"\n{'─'*62}")
            print(f"Phase 0A  Classification    │ 0.0s")
            if self.status:
                self.status.set_phase(1, "Classification")
            classification = await self._classify(question)
            qtype = classification.question_type if classification else None
            _qtype_val = qtype.value if qtype else "unknown"
            print(f"  → {qtype.value.upper() if qtype else 'UNKNOWN'} "
                  f"({time.time()-t_classify:.1f}s)")

            # ── Safety override: upgrade THEORETICAL/UNKNOWN → HYBRID when the
            #    question contains specific numerical assignments AND computation verbs.
            #    Guards against phi4 misclassifying multi-part HYBRID questions.
            if qtype is not None and qtype.value in ("theoretical", "unknown") and classification:
                _num_assign  = bool(re.search(r'[A-Za-z_]\w*\s*=\s*[\d.]+', question))
                _num_units   = bool(re.search(
                    r'\b\d+\.?\d*\s*(kg|m/s|rad/s|N\b|J\b|W\b|Hz|mol|K\b|m\b|s\b)',
                    question, re.IGNORECASE))
                _compute_verb = bool(re.search(
                    r'\b(compute|calculate|solve|evaluate|find|determine)\b',
                    question, re.IGNORECASE))
                _explicitly_numerical = bool(re.search(
                    r'\bnumerically\b|\bsolve\s+for\b|\bevaluate.{0,50}\d',
                    question, re.IGNORECASE))
                if (_num_assign or _num_units) and (_compute_verb or _explicitly_numerical):
                    print(f"  ⚠️  Override: {qtype.value.upper()} → HYBRID "
                          f"(numerical constants + computation verbs detected)")
                    if _HAS_CLASSIFIER:
                        classification.question_type = QuestionType.HYBRID
                    qtype = QuestionType.HYBRID if _HAS_CLASSIFIER else type(
                        'Q', (), {'value': 'hybrid'})()

            # ── Route ─────────────────────────────────────────────────────

            # Engineering design → delegate unchanged
            if qtype and qtype.value == "engineering_design":
                return await self._delegate_engineer(question)

            # Theoretical / Unknown → delegate to V2_1
            if qtype is None or qtype.value in ("theoretical", "unknown"):
                return await self._delegate_v2(question)

            # MATHEMATICAL or HYBRID → new ReAct pipeline
            if qtype.value in ("mathematical", "hybrid"):
                return await self._solve_react(question, classification, qtype.value, job_id=self._job_id)

            # Fallback for any other classification
            return await self._delegate_v2(question)

        except Exception as e:
            import traceback as _tb
            print(f"\n❌ OrchestratorV3 error: {e}")
            print(_tb.format_exc())   # print full traceback to stdout → captured in job log
            # For MATHEMATICAL/HYBRID questions do NOT fall back to V2_1 — it will
            # hallucinate convincing-looking but wrong numbers.  Return an honest
            # failure so the user knows to retry rather than trust fabricated output.
            if _qtype_val in ("mathematical", "hybrid"):
                print("⛔  Refusing V2_1 fallback for math question — returning failure")
                return (
                    "⚠️ The ReAct solver encountered an error and could not compute "
                    "a verified answer for this mathematical question.\n\n"
                    f"Error: {e}\n\n"
                    "Please retry — transient LLM timeouts or planning failures are "
                    "common on first attempt."
                )
            # Theoretical / unknown → V2_1 is safe (no math to hallucinate)
            if _HAS_V2:
                print("↩️  Falling back to V2_1 (non-math question)")
                return await self._delegate_v2(question)
            return f"Unable to answer: {e}"

    # ── ReAct pipeline ────────────────────────────────────────────────────────

    async def _solve_react(
        self,
        question: str,
        classification,
        qtype_value: str,
        job_id: str = "",
    ) -> str:
        t0 = time.time()
        _sep = "─" * 62

        def _elapsed() -> str:
            return f"{time.time() - t0:.1f}s"

        # Chunk 6 — open one sidechain JSONL for this whole solve.
        # make_sidechain() returns None unless SWARM_AS_SUBAGENT=1, so direct
        # /query callers don't pay the disk cost.
        _sc = None
        if _HAS_SIDECHAIN and _make_sidechain is not None:
            try:
                _sc = _make_sidechain(role="react", job_id=job_id or "no_job")
            except Exception as _e:
                _sc = None
                print(f"⚠️  sidechain make failed: {_e}")

        # ── Phase 0B: Generate SolvePlan ─────────────────────────────────
        print(f"\n{_sep}")
        print(f"Phase 0B  Planning          │ {_elapsed()}")
        if self.status:
            self.status.set_phase(2, "Planning")

        requirements: List = []
        if _HAS_PLANNER_V2:
            plan, requirements = await PlannerV2.create_plan(
                question, classification, self._llm_query_planner
            )
        else:
            print("⚠️  PlannerV2 not available — single-SP fallback")
            from planner_v2 import SolvePlan, SubProblem  # might still work
            plan = SolvePlan(
                problem=question,
                domain=classification.domain if classification else "physics",
                given_values={},
                coordinate_system="N/A",
                sub_problems=[SubProblem(
                    id="SP1", description="Solve the full problem",
                    domain="physics", inputs={}, expected_outputs=[],
                    approach="", lookup_queries=[], depends_on=[],
                )],
                dependency_order=["SP1"],
                notes="",
            )

        print(f"  → {len(plan.sub_problems)} SP(s), "
              f"{len(requirements)} requirement(s), "
              f"order: {plan.dependency_order}")
        print(plan.to_markdown()[:600])

        # Dashboard event: emit a single-line JSON summary of the dependency
        # graph + wave grouping so the UI can render lanes/rows up-front.
        try:
            _waves_for_event = self._topological_waves(plan)
            _graph = {
                "sps": [
                    {
                        "id": sp.id,
                        "desc": (sp.description or "")[:200],
                        "deps": list(getattr(sp, "depends_on", []) or []),
                    }
                    for sp in plan.sub_problems
                ],
                "waves": _waves_for_event,
            }
            print(f"DEPENDENCY_GRAPH: {json.dumps(_graph, ensure_ascii=False)}")
        except Exception as _dg_err:
            print(f"  (dep-graph emit skipped: {_dg_err})")

        # ── Phase 0C: Targeted research (HYBRID only) ─────────────────────
        research_contexts: Dict[str, str] = {}
        if qtype_value == "hybrid":
            print(f"\n{_sep}")
            print(f"Phase 0C  Research          │ {_elapsed()}")
            if self.status:
                self.status.set_phase(3, "Research")
            research_contexts = await self._targeted_research(plan)

        # ── VRAM handoff: evict phi4, pre-warm solver ─────────────────────
        solver_model = os.getenv("SWARM_MODEL_DEFAULT", "batiai/qwen3.6-27b:iq4")
        try:
            from react_solver import ReactSolver as _RS_tmp
            solver_model = _RS_tmp.MODEL
        except Exception:
            pass
        if _MODEL_PLANNER != solver_model:
            print(f"\n  🔄 VRAM handoff: {_MODEL_PLANNER} → {solver_model}")
            await self._unload_model(_MODEL_PLANNER)
        else:
            print(f"\n  🔒 VRAM: planner == solver ({solver_model}), model stays loaded")
        # Pre-warm solver in background while we start Phase 1 setup
        asyncio.ensure_future(self._prewarm_model(solver_model))

        # ── Phase 1: Run ReactSolvers in topological waves ─────────────────
        print(f"\n{_sep}")
        print(f"Phase 1   Solving           │ {_elapsed()}")
        if self.status:
            self.status.set_phase(4, "Solving")

        sp_map = {sp.id: sp for sp in plan.sub_problems}
        solver_results: Dict[str, "SolverResult"] = {}
        waves = self._topological_waves(plan)
        print(f"  ⚡ {len(waves)} wave(s): {waves}")

        for wave_idx, wave in enumerate(waves):
            print(f"\n  Wave {wave_idx+1}/{len(waves)}: {wave}  │ {_elapsed()}")
            # Dashboard event: clean breakpoint for animating wave transitions.
            try:
                _solved_so_far = [
                    pid for pid, pr in solver_results.items() if pr.status == "solved"
                ]
                _reason = (
                    f"deps met from solved: {','.join(_solved_so_far)}"
                    if _solved_so_far else "no deps"
                )
                print(
                    f"WAVE_DISPATCH: Wave {wave_idx+1}/{len(waves)} → "
                    f"[{', '.join(wave)}] reason='{_reason}'"
                )
            except Exception:
                pass

            # ── Build global manifest from ALL solved SPs so far ────────────────
            global_manifest: Dict[str, str] = {}  # var → "value unit"
            for prior_id, prior_res in solver_results.items():
                if prior_res.status == "solved":
                    if getattr(prior_res, "results_with_units", None):
                        for var, meta in prior_res.results_with_units.items():
                            v = meta.get("value", "")
                            u = meta.get("unit", "")
                            global_manifest[var] = f"{v} {u}".strip()
                    elif prior_res.results:
                        for var, val in prior_res.results.items():
                            global_manifest[var] = str(val)

            # ── Build manifest text block and inject numeric values into SP inputs ─
            manifest_block = ""
            # Float version of manifest for ReactSolver anchor splitting
            global_manifest_float: Dict[str, float] = {}
            for _var, _val_str in global_manifest.items():
                try:
                    global_manifest_float[_var] = float(_val_str.split()[0])
                except (ValueError, IndexError):
                    pass
            if global_manifest:
                mlines = [f"  {k} = {v}" for k, v in global_manifest.items()]
                manifest_block = (
                    "\n🔒 COMPUTED FACTS FROM PRIOR STEPS — USE THESE DIRECTLY, DO NOT RECOMPUTE:\n"
                    + "\n".join(mlines) + "\n"
                )
                print(f"  📋 Manifest: {len(global_manifest)} computed value(s) → all wave SPs")

            # ── Soft salvage: scan failed SP logs for any RESULT: lines ─────────
            # Swarm 3.13 — skip variables that were rejected by the residual lock
            soft_manifest: Dict[str, str] = {}  # var → "value unit" (unverified)
            for prior_id, prior_res in solver_results.items():
                if prior_res.status != "solved" and getattr(prior_res, "raw_log", ""):
                    # Build per-SP skip set from RESIDUAL_LOCK_REJECTED: sentinel lines
                    rejected_vars: set = set()
                    for log_line in prior_res.raw_log.splitlines():
                        rej_m = re.match(
                            r'RESIDUAL_LOCK_REJECTED:\s*(.+)',
                            log_line.strip(),
                        )
                        if rej_m:
                            for _v in rej_m.group(1).split(","):
                                _v = _v.strip()
                                if _v:
                                    rejected_vars.add(_v)
                        # Swarm 3.14 — also skip vars flagged by stability gate
                        sg_m = re.match(
                            r'STABILITY_GATE_REJECTED:\s*(\w+)',
                            log_line.strip(),
                        )
                        if sg_m:
                            rejected_vars.add(sg_m.group(1).strip())
                        # Swarm 3.14.1 — skip vars flagged by plausibility audit
                        pg_m = re.match(
                            r'PLAUSIBILITY_GATE_REJECTED:\s*(.+)',
                            log_line.strip(),
                        )
                        if pg_m:
                            for _v in pg_m.group(1).split(","):
                                _v = _v.strip()
                                if _v:
                                    rejected_vars.add(_v)
                    for log_line in prior_res.raw_log.splitlines():
                        rm = re.match(
                            r'RESULT:\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d[\d.e+\-]*)\s*(.*)',
                            log_line.strip(),
                            re.IGNORECASE,
                        )
                        if rm:
                            var, val, unit = rm.group(1), rm.group(2), rm.group(3).strip()
                            if var in rejected_vars:
                                continue   # residual lock said this is poisoned
                            if var not in global_manifest and var not in soft_manifest:
                                soft_manifest[var] = f"{val} {unit}".strip()

            # Append soft results to manifest block (separate section, labeled unverified)
            if soft_manifest:
                soft_lines = [f"  {k} = {v}  (unverified)" for k, v in soft_manifest.items()]
                manifest_block += (
                    "\n💡 SOFT RESULTS (partial computation from failed SPs — use if nothing better):\n"
                    + "\n".join(soft_lines) + "\n"
                )
                print(f"  💡 Soft salvage: {len(soft_manifest)} value(s) from failed SP logs")

            for sp_id in wave:
                sp = sp_map.get(sp_id)
                if sp is None:
                    continue
                # Inject numeric values into sp.inputs (for code constant injection)
                for var, val_str in global_manifest.items():
                    if var not in sp.inputs:
                        try:
                            sp.inputs[var] = float(val_str.split()[0])
                        except (ValueError, IndexError):
                            pass  # non-numeric (string results) — skip
                # Also inject soft results at lower priority (don't override confirmed values)
                for var, val_str in soft_manifest.items():
                    if var not in sp.inputs:
                        try:
                            sp.inputs[var] = float(val_str.split()[0])
                        except (ValueError, IndexError):
                            pass

                # Swarm 3.14.2 — Hard-Lock Consensus: scrub locked vars from
                # this SP's expected_outputs. If a variable is already in the
                # global_manifest (solved + residual-locked), downstream SPs
                # must NOT re-derive it. Prevents Numerical Schizophrenia
                # (e.g. SP2 r0=1.445 vs SP3 r0=1.259 in the same report).
                if sp.expected_outputs and global_manifest:
                    _scrubbed: List[Any] = []
                    _dropped: List[str] = []
                    for _eo in sp.expected_outputs:
                        if isinstance(_eo, dict):
                            _name = _eo.get("name") or _eo.get("var") or ""
                        else:
                            _name = str(_eo)
                        if _name and _name in global_manifest:
                            _dropped.append(_name)
                        else:
                            _scrubbed.append(_eo)
                    if _dropped:
                        print(f"  🔒 HARD-LOCK: {sp.id} scrubbing locked outputs "
                              f"{_dropped} (use manifest value, do not recompute)")
                        sp.expected_outputs = _scrubbed

            # Run this wave in parallel
            tasks = []
            task_sp_ids = []
            for sp_id in wave:
                sp = sp_map.get(sp_id)
                if sp is None:
                    print(f"  ⚠️  Skipping unknown SP id '{sp_id}' (not in plan)")
                    continue
                ctx = research_contexts.get(sp_id, "")
                ctx_with_manifest = manifest_block + (ctx or "")
                if _HAS_REACT:
                    solver = ReactSolver(
                        sub_problem=sp,
                        plan=plan,
                        research_context=ctx_with_manifest,
                        searxng_url=self.searxng_url,
                        manifest_values=global_manifest_float,
                        sidechain=_sc,
                    )
                    tasks.append(solver.solve())
                else:
                    tasks.append(self._stub_solve(sp))
                task_sp_ids.append(sp_id)

            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            wave_results = []
            for sp_id, res in zip(task_sp_ids, raw_results):
                if isinstance(res, Exception):
                    print(f"  ⚠️  SP '{sp_id}' raised exception: {res}")
                    from react_solver import SolverResult as _SR
                    wave_results.append(_SR(
                        sub_problem_id=sp_id, status="failed",
                        verification_note=str(res),
                    ))
                else:
                    wave_results.append(res)
            # ── Manifest consensus: reject values that conflict with prior waves ─
            # If a new result differs by >1% from an already-solved manifest value,
            # pin the new SP's value to the prior manifest (trust earlier waves).
            for result in wave_results:
                if result.status != "solved":
                    continue
                for var in list(result.results.keys()):
                    if var not in global_manifest:
                        continue
                    try:
                        prior_val = float(global_manifest[var].split()[0])
                        new_val = float(str(result.results[var]))
                        if prior_val == 0 or new_val == 0:
                            continue
                        pct_diff = abs(new_val - prior_val) / abs(prior_val)
                        if pct_diff > 0.01:  # >1% threshold
                            print(
                                f"  🚨 CONSENSUS: {var} manifest={prior_val:.6g} "
                                f"vs {result.sub_problem_id}={new_val:.6g} "
                                f"({pct_diff*100:.1f}% diff) — pinned to manifest value"
                            )
                            result.results[var] = prior_val
                            if var in result.results_with_units:
                                result.results_with_units[var]["value"] = prior_val
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass

            # ── Swarm 3.13: Orchestrator-side Pydantic Residual Lock ─────────
            # Independently re-execute each solved SP's CHECK expression(s).
            # If any fail, flip the SP to "failed" and stamp the raw_log so
            # soft-salvage skips the poisoned vars downstream.
            if RESIDUAL_LOCK_ORCH:
                for result in wave_results:
                    if result.status != "solved":
                        continue
                    sp = sp_map.get(result.sub_problem_id)
                    if sp is None or not sp.expected_outputs:
                        continue
                    ok, reason, worst, failed_vars = await self._residual_lock_check(
                        result, sp
                    )
                    if ok:
                        print(f"  🔬 RESIDUAL LOCK: {sp.id} → PASS (worst={worst:.2e})")
                    else:
                        print(
                            f"  🔬 RESIDUAL LOCK: {sp.id} → FAIL "
                            f"(worst={worst:.2e}) — {reason[:140]}"
                        )
                        result.status = "failed"
                        result.verification_note = (
                            (result.verification_note + "; " if result.verification_note else "")
                            + f"residual lock rejected: {reason[:200]}"
                        )
                        if failed_vars:
                            # Stamp the raw_log so soft-salvage knows which vars
                            # to skip when scavenging RESULT: lines from failed SPs.
                            result.raw_log = (result.raw_log or "") + (
                                f"\nRESIDUAL_LOCK_REJECTED: {','.join(failed_vars)}\n"
                            )

            # Swarm 3.14 — Stability Gate (orchestrator side)
            # Catches the "omega_r = 0.0 for an unstable orbit" cheat: if the
            # raw_log contains CHECK: Vpp_value = <negative> AND any RESULT:
            # omega_r ≈ 0 (not NaN), reject it and force a retry.
            if STABILITY_GATE_ORCH:
                for result in wave_results:
                    if result.status != "solved":
                        continue
                    raw = result.raw_log or ""
                    # Find the most recent Vpp_value CHECK line
                    vpp_matches = re.findall(
                        r"CHECK:\s*Vpp_value\s*=\s*([+-]?[\d.eE+\-]+)", raw
                    )
                    if not vpp_matches:
                        continue
                    try:
                        vpp = float(vpp_matches[-1])
                    except ValueError:
                        continue
                    # Look at reported omega_r in results dict
                    omega_r = None
                    for k, v in (result.results or {}).items():
                        if k.lower() in ("omega_r", "omega", "omega_osc", "omega_small"):
                            omega_r = v
                            break
                    if omega_r is None:
                        continue
                    # If Vpp < 0 AND omega_r is finite and ≈ 0, that's the lie
                    try:
                        import math as _m
                        if vpp < 0 and _m.isfinite(omega_r) and abs(omega_r) < 1e-9:
                            print(
                                f"  ⚖️  STABILITY GATE: {result.sub_problem_id} → FAIL "
                                f"(Vpp={vpp:.3e} < 0 but omega_r={omega_r:.3e}; "
                                f"unstable orbit needs NaN, not 0)"
                            )
                            result.status = "failed"
                            result.verification_note = (
                                (result.verification_note + "; " if result.verification_note else "")
                                + "stability gate: unstable equilibrium (Vpp<0) reported "
                                  "omega_r≈0 instead of nan"
                            )
                            result.raw_log = raw + (
                                f"\nSTABILITY_GATE_REJECTED: omega_r (Vpp={vpp:.3e})\n"
                            )
                    except Exception:
                        pass

            # Swarm 3.14.1 — Plausibility Gate (Lock D)
            # Catches "Lazy Scientist" cheats: when the solver exits with a
            # suspiciously round value (0.0, 1.0, -1.0) or a value equal to an
            # input, a fresh Auditor LLM peer-reviews whether that value is
            # physically plausible for this potential. Rejects are marked
            # PLAUSIBILITY_GATE_REJECTED: so soft-salvage skips poisoned vars.
            if PLAUSIBILITY_GATE_ENABLED:
                for result in wave_results:
                    if result.status != "solved":
                        continue
                    sp = sp_map.get(result.sub_problem_id)
                    if sp is None or not result.results:
                        continue
                    ok_p, reason_p, rejected_p = await self._plausibility_audit(result, sp)
                    if ok_p:
                        if reason_p:
                            print(f"  🕵️  PLAUSIBILITY: {sp.id} → PASS ({reason_p[:80]})")
                    else:
                        print(
                            f"  🕵️  PLAUSIBILITY: {sp.id} → REJECT "
                            f"({','.join(rejected_p)}) — {reason_p[:140]}"
                        )
                        result.status = "failed"
                        result.verification_note = (
                            (result.verification_note + "; " if result.verification_note else "")
                            + f"plausibility audit rejected: {reason_p[:200]}"
                        )
                        result.raw_log = (result.raw_log or "") + (
                            f"\nPLAUSIBILITY_GATE_REJECTED: {','.join(rejected_p)}\n"
                        )

            for result in wave_results:
                solver_results[result.sub_problem_id] = result

            # ── Post-wave: retry SPs with expected_outputs but 0 results ─
            retry_tasks = []
            retry_sp_ids = []
            for result in wave_results:
                sp_id = result.sub_problem_id
                sp = sp_map.get(sp_id)
                if sp and result.status != "solved" and sp.expected_outputs and _HAS_REACT:
                    # Swarm 3.13 — craft a targeted retry seed if the reason
                    # this SP failed was a residual-lock rejection.
                    residual_rejected = (
                        "residual lock rejected" in (result.verification_note or "").lower()
                        or "RESIDUAL_LOCK_REJECTED" in (result.raw_log or "")
                    )
                    if residual_rejected:
                        rej_vars: List[str] = []
                        for _ln in (result.raw_log or "").splitlines():
                            _rm = re.match(r'RESIDUAL_LOCK_REJECTED:\s*(.+)', _ln.strip())
                            if _rm:
                                for _v in _rm.group(1).split(","):
                                    _v = _v.strip()
                                    if _v:
                                        rej_vars.append(_v)
                        worst = 0.0
                        for _v, _r in (getattr(result, "check_eval_values", {}) or {}).items():
                            try:
                                _rv = float(_r)
                                if _rv > worst:
                                    worst = _rv
                            except (TypeError, ValueError):
                                pass
                        retry_msg = (
                            f"⛔ PRIOR ATTEMPT FAILED THE RESIDUAL LOCK.\n"
                            f"Rejected variable(s): {', '.join(rej_vars) or 'unknown'}.\n"
                            f"Worst relative residual: {worst:.3e} "
                            f"(tolerance {RESIDUAL_TOLERANCE_REL_ORCH:.0e}).\n"
                            f"This means the value(s) you printed did NOT satisfy the "
                            f"equation — you likely guessed a round number to escape a "
                            f"quartic / transcendental that sympy.solve couldn't handle.\n\n"
                            f"MANDATORY this retry:\n"
                            f"  1. Derive the equation symbolically first.\n"
                            f"  2. Solve it with scipy.optimize.brentq — NOT sympy.solve.\n"
                            f"  3. Verify the CHECK: residual < 1e-6 before FINAL_ANSWER.\n"
                            f"  4. Emit a paired `CHECK: <var>_residual = <expr>` line for "
                            f"EVERY RESULT: line, per Rule 21.\n"
                            f"Do not fabricate. If no real root exists, output STATUS: failed."
                        )
                        print(f"  ⚡ Retrying {sp_id} (residual-rejected, worst={worst:.2e})…")
                    else:
                        retry_msg = (
                            "Previous attempt returned 0 results. "
                            "You MUST write Python code and use ACTION: run_code. "
                            "Do not describe the calculation in prose — execute it."
                        )
                        print(f"  ⚡ Retrying {sp_id} (status={result.status}, "
                              f"0 computed results, outputs expected)…")

                    retry_solver = ReactSolver(
                        sub_problem=sp,
                        plan=plan,
                        research_context=manifest_block + research_contexts.get(sp_id, ""),
                        searxng_url=self.searxng_url,
                        manifest_values=global_manifest_float,
                        sidechain=_sc,
                    )
                    retry_solver._history.append({
                        "role": "user",
                        "content": retry_msg,
                    })
                    retry_tasks.append(retry_solver.solve())
                    retry_sp_ids.append(sp_id)

            if retry_tasks:
                retry_results = await asyncio.gather(*retry_tasks)
                for sp_id, rr in zip(retry_sp_ids, retry_results):
                    solver_results[sp_id] = rr
                    vals_str = (
                        ", ".join(
                            f"{v}={d['value']:.4g}{' '+d['unit'] if d.get('unit') else ''}"
                            for v, d in rr.results_with_units.items()
                        ) if rr.results_with_units else "no results"
                    )
                    print(f"  {sp_id} RETRY → {rr.status.upper()} | {vals_str} "
                          f"| {rr.turn_count} turns")

        # ── Requirement Audit: one-pass re-solve for completely dropped SPs ──
        # Catches SPs that were skipped due to timeouts, OOM, or wave-level
        # failures — ensures every requirement from the planner gets a result.
        dropped_sps = [
            sp for sp in plan.sub_problems
            if (
                solver_results.get(sp.id) is None
                or (solver_results[sp.id].status != "solved" and sp.expected_outputs)
            )
        ]
        if dropped_sps:
            print(f"\n{'─'*62}")
            print(f"  🚨 Requirement Audit: {len(dropped_sps)} SP(s) with no results — "
                  f"re-solving: {[sp.id for sp in dropped_sps]}")
            # Rebuild full manifest from all solved SPs so far
            audit_manifest: Dict[str, str] = {}
            for prior_id, prior_res in solver_results.items():
                if prior_res.status == "solved":
                    if getattr(prior_res, "results_with_units", None):
                        for var, meta in prior_res.results_with_units.items():
                            audit_manifest[var] = (
                                f"{meta.get('value','')} {meta.get('unit','')}".strip()
                            )
                    elif prior_res.results:
                        for var, val in prior_res.results.items():
                            audit_manifest[var] = str(val)
            audit_block = ""
            if audit_manifest:
                mlines = [f"  {k} = {v}" for k, v in audit_manifest.items()]
                audit_block = (
                    "\n🔒 COMPUTED FACTS FROM PRIOR STEPS — USE THESE DIRECTLY, DO NOT RECOMPUTE:\n"
                    + "\n".join(mlines) + "\n"
                )
            audit_tasks = []
            audit_sp_ids = []
            for sp in dropped_sps:
                # Inject manifest into SP inputs (makes them available as LOCKED GIVEN VALUES)
                for var, val_str in audit_manifest.items():
                    if var not in sp.inputs:
                        try:
                            sp.inputs[var] = float(val_str.split()[0])
                        except (ValueError, IndexError):
                            pass
                ctx = research_contexts.get(sp.id, "")
                # Rebuild audit manifest float for anchor splitting
                audit_manifest_float: Dict[str, float] = {}
                for _var, _val_str in audit_manifest.items():
                    try:
                        audit_manifest_float[_var] = float(_val_str.split()[0])
                    except (ValueError, IndexError):
                        pass
                audit_solver = ReactSolver(
                    sub_problem=sp,
                    plan=plan,
                    research_context=audit_block + (ctx or ""),
                    searxng_url=self.searxng_url,
                    manifest_values=audit_manifest_float,
                    sidechain=_sc,
                )
                audit_solver._history.append({
                    "role": "user",
                    "content": (
                        "⚠️  REQUIREMENT AUDIT: This sub-problem produced no results in "
                        "the first pass. You MUST solve it now. Use ACTION: run_code "
                        "immediately. Do NOT skip, approximate, or write prose — compute "
                        "the exact numerical result."
                    ),
                })
                audit_tasks.append(audit_solver.solve())
                audit_sp_ids.append(sp.id)
            if audit_tasks:
                audit_results = await asyncio.gather(*audit_tasks)
                for sp_id, ar in zip(audit_sp_ids, audit_results):
                    solver_results[sp_id] = ar
                    vals_str = (
                        ", ".join(
                            f"{v}={d['value']:.4g}{' '+d['unit'] if d.get('unit') else ''}"
                            for v, d in ar.results_with_units.items()
                        ) if ar.results_with_units else "no results"
                    ) if ar.status == "solved" else "no results"
                    print(f"  🚨 AUDIT {sp_id} → {ar.status.upper()} | {vals_str}")

        # ── Phase 1X: Domain Wave Gate ────────────────────────────────────
        # Block the writer until EVERY domain (MATHEMATICS, CHEMISTRY) has
        # at least one solved SP.  Catches the "Megaprompt Dropout" where 10+
        # physics SPs exhaust the planner's attention and math/chem get skipped.
        _gate_domains = ("MATHEMATICS", "CHEMISTRY")
        domain_missed = [
            sp for sp in plan.sub_problems
            if self._domain_category(sp.domain, sp.description) in _gate_domains
            and (
                solver_results.get(sp.id) is None
                or (solver_results[sp.id].status != "solved" and sp.expected_outputs)
            )
        ]
        if domain_missed:
            print(f"\n{_sep}")
            print(f"Phase 1X  Domain Gate       │ {_elapsed()}")
            print(f"  ⛔ Writer BLOCKED — {len(domain_missed)} MATH/CHEM SP(s) unsolved:")
            for _sp in domain_missed:
                _cat  = self._domain_category(_sp.domain, _sp.description)
                _st   = solver_results[_sp.id].status if _sp.id in solver_results else "MISSING"
                print(f"    {_sp.id} [{_cat}] {_st}: {_sp.description[:60]}")

            # Rebuild manifest from all solved SPs so far
            gate_manifest: Dict[str, str] = {}
            for _pid, _pr in solver_results.items():
                if _pr.status == "solved":
                    if getattr(_pr, "results_with_units", None):
                        for _var, _meta in _pr.results_with_units.items():
                            gate_manifest[_var] = (
                                f"{_meta.get('value','')} {_meta.get('unit','')}".strip()
                            )
                    elif _pr.results:
                        for _var, _val in _pr.results.items():
                            gate_manifest[_var] = str(_val)
            gate_manifest_float: Dict[str, float] = {}
            for _var, _vs in gate_manifest.items():
                try:
                    gate_manifest_float[_var] = float(_vs.split()[0])
                except (ValueError, IndexError):
                    pass
            gate_block = ""
            if gate_manifest:
                _glines = [f"  {k} = {v}" for k, v in gate_manifest.items()]
                gate_block = (
                    "\n🔒 COMPUTED FACTS FROM PRIOR STEPS — USE THESE DIRECTLY:\n"
                    + "\n".join(_glines) + "\n"
                )

            gate_tasks, gate_ids = [], []
            for _sp in domain_missed:
                _cat = self._domain_category(_sp.domain, _sp.description)
                for _var, _vs in gate_manifest.items():
                    if _var not in _sp.inputs:
                        try:
                            _sp.inputs[_var] = float(_vs.split()[0])
                        except (ValueError, IndexError):
                            pass
                _ctx = research_contexts.get(_sp.id, "")
                _gs = ReactSolver(
                    sub_problem=_sp, plan=plan,
                    research_context=gate_block + (_ctx or ""),
                    searxng_url=self.searxng_url,
                    manifest_values=gate_manifest_float,
                    sidechain=_sc,
                )
                _gs._history.append({"role": "user", "content": (
                    f"⛔ DOMAIN GATE: This {_cat} sub-problem was NOT solved in the "
                    f"physics wave. The writer is BLOCKED until all domains are complete. "
                    f"You MUST solve {_sp.id} now: {_sp.description}\n"
                    f"Begin IMMEDIATELY with ACTION: run_code. No prose."
                )})
                gate_tasks.append(_gs.solve())
                gate_ids.append(_sp.id)

            if gate_tasks:
                gate_res = await asyncio.gather(*gate_tasks)
                for _gid, _gr in zip(gate_ids, gate_res):
                    solver_results[_gid] = _gr
                    _gvals = (
                        ", ".join(
                            f"{v}={d['value']:.4g}{' '+d['unit'] if d.get('unit') else ''}"
                            for v, d in _gr.results_with_units.items()
                        ) if _gr.results_with_units else "no results"
                    )
                    print(f"  ⛔→{'✅' if _gr.status=='solved' else '❌'} "
                          f"Gate {_gid} → {_gr.status.upper()} | {_gvals} | {_gr.turn_count} turns")

        # ── Free VRAM before synthesis — only if switching models ─────────
        try:
            from react_solver import ReactSolver as _RS_w
            _solver_model_now = _RS_w.MODEL
        except Exception:
            _solver_model_now = ""
        if _solver_model_now != _MODEL_CODER:
            await self._unload_solver_model()
        else:
            print(f"  🔒 VRAM: solver == writer ({_MODEL_CODER}), model stays loaded")

        # ── Phase 2: Synthesis ─────────────────────────────────────────────
        print(f"\n{_sep}")
        print(f"Phase 2   Synthesis         │ {_elapsed()}")
        if self.status:
            self.status.set_phase(5, "Synthesis")
        synthesis = await self._synthesize(question, plan, solver_results)

        # ── Phase 3: Writer ────────────────────────────────────────────────
        print(f"\n{_sep}")
        print(f"Phase 3   Writing           │ {_elapsed()}")
        if self.status:
            self.status.set_phase(6, "Writing")
        answer = await self._write_final_answer(
            question, synthesis, plan, solver_results, requirements
        )

        # ── Phase 3D: Domain completeness gate ────────────────────────────
        print(f"\n{'─'*62}")
        print(f"Phase 3D  DomainGate")
        answer = await self._domain_completeness_gate(answer, plan, solver_results)

        n_solved = sum(1 for r in solver_results.values() if r.status == "solved")
        elapsed = time.time() - t0
        print(f"\n{_sep}")
        print(f"✅ Done │ {n_solved}/{len(solver_results)} SP(s) solved │ Total: {elapsed:.1f}s")

        # ── Debug: SP status summary + persist raw logs ───────────────────
        print(f"\n{'─'*62}")
        print("📊 SP Debug Summary")
        for sp in plan.sub_problems:
            sr = solver_results.get(sp.id)
            if sr is None:
                print(f"  {sp.id}  MISSING   {sp.description[:60]}")
                continue
            vals = ", ".join(
                f"{v}={d['value']:.4g}{' '+d['unit'] if d.get('unit') else ''}"
                for v, d in (sr.results_with_units or {}).items()
            ) or "(none)"
            print(f"  {sp.id}  {sr.status.upper():8s}  turns={sr.turn_count}  "
                  f"results={vals}  │  {sp.description[:50]}")

        if job_id:
            import pathlib
            debug_dir = pathlib.Path("./swarm_results/debug") / job_id
            debug_dir.mkdir(parents=True, exist_ok=True)
            for sp_id, sr in solver_results.items():
                if sr.raw_log:
                    (debug_dir / f"{sp_id}.log").write_text(sr.raw_log, encoding="utf-8")
            # Write plan summary
            try:
                (debug_dir / "_plan.md").write_text(plan.to_markdown(), encoding="utf-8")
            except Exception:
                pass
            print(f"  📁 SP logs → swarm_results/debug/{job_id}/")

        return answer

    # ── Swarm 3.13: Orchestrator-side Residual Lock ──────────────────────────

    async def _plausibility_audit(
        self,
        result: "SolverResult",
        sp: "SubProblem",
    ) -> tuple:
        """
        Swarm 3.14.1 — Plausibility Gate (Lock D)

        When the solver returns a "suspicious" value (0.0, 1.0, -1.0,
        or exactly equal to one of sp.inputs), spin up a fresh Auditor
        LLM to peer-review whether that value is physically plausible
        for this sub-problem's domain + potential.

        Returns (ok: bool, reason: str, rejected_vars: List[str]).
        Non-suspicious results return (True, "", []) immediately without
        calling the LLM — this is cheap by design.
        """
        import math as _m
        # Step 1 — triage: only audit suspicious values
        _SUSPICIOUS_CONSTANTS = {0.0, 1.0, -1.0, 0.5, -0.5}
        input_values = set()
        for v in (sp.inputs or {}).values():
            try:
                input_values.add(float(v))
            except (TypeError, ValueError):
                pass

        suspicious: List[Tuple[str, float, str]] = []  # (var, val, why)
        for var, val in (result.results or {}).items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not _m.isfinite(fval):
                continue  # NaN/inf handled by residual lock
            why = None
            if fval in _SUSPICIOUS_CONSTANTS:
                why = f"suspiciously round ({fval})"
            elif fval in input_values:
                why = f"equals an input value ({fval})"
            if why is None:
                # Also flag "marginally stable" or "trivially" language paired with 0
                raw = (result.raw_log or "").lower()
                if (
                    abs(fval) < 1e-12
                    and any(kw in raw for kw in (
                        "marginally stable", "marginal stability",
                        "trivially zero", "trivial solution",
                        "reduces to zero",
                    ))
                ):
                    why = f"zero-value with 'marginally stable' escape-hatch language"
            if why:
                suspicious.append((var, fval, why))

        if not suspicious:
            return True, "", []

        # Step 2 — ask the Auditor LLM whether each suspicious value is plausible
        findings_str = "\n".join(f"  • {v} = {x} ({w})" for v, x, w in suspicious)
        sp_inputs_str = ", ".join(
            f"{k}={v}" for k, v in (sp.inputs or {}).items()
        ) or "(none)"
        prompt = (
            "You are an INDEPENDENT PHYSICS AUDITOR reviewing another agent's\n"
            "sub-problem result. The solver reported values that look suspicious\n"
            "(round numbers or equal to an input). Decide if these are physically\n"
            "plausible OR if the solver cheated to exit a hard problem.\n\n"
            f"SUB-PROBLEM: {sp.id} — {sp.description}\n"
            f"DOMAIN: {getattr(sp, 'domain', '?')}\n"
            f"APPROACH: {getattr(sp, 'approach', '(none)')}\n"
            f"INPUTS: {sp_inputs_str}\n\n"
            f"SUSPICIOUS RESULTS:\n{findings_str}\n\n"
            "Rules:\n"
            "  • A radial oscillation frequency ω_r = 0 is NEVER physically\n"
            "    plausible for a generic non-linear potential (1/r^3, r^4, etc.)\n"
            "    unless V''(r0) is proven identically zero.\n"
            "  • A result == input usually means the solver skipped the math.\n"
            "  • 'Marginally stable' requires an exact proof, not an escape hatch.\n\n"
            "OUTPUT EXACTLY ONE OF:\n"
            "  PLAUSIBLE: <short justification>\n"
            "  IMPLAUSIBLE: <var1,var2,...> — <1-sentence physical reason>\n"
        )
        system = (
            "You are a rigorous physics auditor. Be skeptical. If a value looks\n"
            "like a lazy fabrication to exit a hard computation, reject it.\n"
            "Accept only values that have a genuine physical justification."
        )
        try:
            # Import locally to avoid circular import
            from react_solver import ReactSolver as _RS
            reply = await _RS._ollama_chat(
                model=PLAUSIBILITY_AUDITOR_MODEL,
                prompt=prompt,
                system=system,
                timeout=300,
                num_predict=300,
                keep_alive=300,
            )
        except Exception as e:
            # Auditor unavailable — fail OPEN (don't block the solve on infra
            # problems) but record it
            return True, f"auditor-unavailable ({e})", []

        reply_u = (reply or "").strip()
        if not reply_u:
            return True, "auditor-empty", []

        # Parse reply
        if re.search(r"\bIMPLAUSIBLE\b", reply_u, re.IGNORECASE):
            m = re.search(
                r"IMPLAUSIBLE:\s*([A-Za-z_][\w, ]*?)\s*[—\-:]\s*(.+)",
                reply_u,
                re.IGNORECASE | re.DOTALL,
            )
            if m:
                rejected = [
                    v.strip() for v in m.group(1).split(",") if v.strip()
                ]
                reason = m.group(2).strip()[:300]
            else:
                # Couldn't parse specific vars — reject ALL suspicious ones
                rejected = [v for v, _, _ in suspicious]
                reason = reply_u[:300]
            # Only reject vars that actually appear in our suspicious list
            suspicious_names = {v for v, _, _ in suspicious}
            rejected = [v for v in rejected if v in suspicious_names] or list(suspicious_names)
            return False, reason, rejected

        # PLAUSIBLE branch
        return True, reply_u[:200], []

    async def _residual_lock_check(
        self,
        result: "SolverResult",
        sp: "SubProblem",
    ) -> tuple:
        """
        Belt-and-suspenders residual verification: the solver already ran its
        own gate, but the orchestrator re-executes each CHECK expression
        independently so a compromised solver cannot fake passing residuals.

        Returns (ok: bool, reason: str, max_residual: float, failed_vars: List[str]).
        """
        if not RESIDUAL_LOCK_ORCH:
            return True, "", 0.0, []
        try:
            from equation_validator import EquationExecutor
        except ImportError:
            return True, "executor unavailable — orch gate skipped", 0.0, []

        check_residuals = getattr(result, "check_residuals", None) or {}
        if not result.results:
            return True, "", 0.0, []
        if not check_residuals:
            return True, "no CHECK lines provided", 0.0, []

        # Build numeric bindings: sp.inputs (locked + given) + result.results.
        locals_lines: List[str] = []
        for k, v in sp.inputs.items():
            try:
                fv = float(v)
                locals_lines.append(f"{k} = {fv!r}")
            except (TypeError, ValueError):
                continue
        for k, v in result.results.items():
            if k in sp.inputs:
                continue
            try:
                fv = float(v)
                locals_lines.append(f"{k} = {fv!r}")
            except (TypeError, ValueError):
                continue

        max_res = 0.0
        failed_vars: List[str] = []
        diagnostics: List[str] = []

        for var, expr in check_residuals.items():
            # Absolute tolerance fallback for CHECK expressions that already
            # represent an absolute residual (large-magnitude inputs).
            try:
                var_val = float(result.results.get(var, 0.0))
            except (TypeError, ValueError):
                var_val = 0.0
            abs_tol = max(1e-9, min(1e-3 * abs(var_val), 1e-3))

            script = (
                "import math\n"
                "try:\n"
                "    import numpy as np\n"
                "except Exception:\n"
                "    np = None\n"
                "try:\n"
                "    import mpmath\n"
                "except Exception:\n"
                "    mpmath = None\n"
                + "\n".join(locals_lines) + "\n"
                f"_res = {expr}\n"
                "try:\n"
                "    _res = float(_res)\n"
                "except Exception:\n"
                "    _res = float('inf')\n"
                "print(f'RESIDUAL: {_res}')\n"
            )
            try:
                exec_result = await EquationExecutor.execute(
                    script, given_values={}, timeout=30
                )
            except Exception as e:
                failed_vars.append(var)
                diagnostics.append(f"{var}: gate exception {e}")
                max_res = float("inf")
                continue

            if not exec_result.success:
                failed_vars.append(var)
                diagnostics.append(
                    f"{var}: CHECK did not execute ({(exec_result.error or '')[:80]})"
                )
                max_res = float("inf")
                continue

            m = re.search(r"RESIDUAL:\s*([+-]?[\d.eE+\-]+)", exec_result.output or "")
            if not m:
                failed_vars.append(var)
                diagnostics.append(f"{var}: no RESIDUAL printed")
                max_res = float("inf")
                continue

            try:
                res_val = abs(float(m.group(1)))
            except ValueError:
                failed_vars.append(var)
                diagnostics.append(f"{var}: unparseable residual '{m.group(1)}'")
                max_res = float("inf")
                continue

            # Record for debugging
            try:
                result.check_eval_values[var] = res_val
            except AttributeError:
                pass

            if res_val > max_res:
                max_res = res_val
            if res_val >= RESIDUAL_TOLERANCE_REL_ORCH and res_val >= abs_tol:
                failed_vars.append(var)
                diagnostics.append(
                    f"{var}: residual {res_val:.3e} exceeds tol "
                    f"(rel={RESIDUAL_TOLERANCE_REL_ORCH:.0e}, abs={abs_tol:.0e})"
                )

        if failed_vars:
            return False, "; ".join(diagnostics), max_res, failed_vars
        return True, "", max_res, []

    # ── Domain categorisation ─────────────────────────────────────────────────

    @staticmethod
    def _domain_category(domain: str, description: str = "") -> str:
        """
        Map an SP domain string + description to a broad category.
        Description keywords take priority over the planner's domain label
        (planner often over-labels everything as 'physics').
        """
        desc = description.lower()
        # ── Description-first overrides (planner often mislabels these) ─────
        # Mathematics / Calculus
        if any(kw in desc for kw in [
            "stirling", "infinite series", "series expansion", "series analysis",
            "series sum", "convergence", "sigma notation", "factorial approximat",
            "improper integral", "evaluate integral", "antiderivative",
            "integrate f", "integrate the", "compute the integral", "definite integral",
        ]):
            return "MATHEMATICS"
        # Chemistry
        if any(kw in desc for kw in [
            "gibbs-helmholtz", "gibbs helmholtz", "gibbs free energy",
            "electrochemical", "electrochemistry", "cell potential",
            "faraday", "nernst", "oxidation", "reduction", "half-cell",
            "molar enthalpy", "molar entropy", "thermochemistry",
        ]):
            return "CHEMISTRY"

        # ── Fall back to domain label ─────────────────────────────────────────
        d = domain.lower().replace("_", "").replace(" ", "")
        if any(x in d for x in ["electrochemist", "thermochemist", "biochem",
                                 "analyticalchem", "physicalchem", "gibbs", "faraday"]):
            return "CHEMISTRY"
        if any(x in d for x in ["chem", "reaction", "molec", "enthalpy",
                                 "stoichi", "molar"]):
            return "CHEMISTRY"
        if any(x in d for x in ["calc", "algebra", "numbertheory", "series",
                                 "statistic", "geomet", "combinat", "puremath",
                                 "stirling", "integral", "differentiat"]):
            return "MATHEMATICS"
        if any(x in d for x in ["math", "analysis"]):
            return "MATHEMATICS"
        return "PHYSICS"

    # ── Domain completeness gate ──────────────────────────────────────────────

    async def _domain_completeness_gate(
        self,
        answer: str,
        plan: "SolvePlan",
        solver_results: Dict[str, "SolverResult"],
    ) -> str:
        """
        Phase 3D: scan writer output for missing domain sections.
        If a domain has solved SPs whose results don't appear in the answer,
        generate a targeted append section via llm_query_coder and add it.
        """
        domain_groups: Dict[str, List[str]] = {}
        for sp in plan.sub_problems:
            # Pass description so mislabeled "physics" SPs are correctly categorised
            cat = self._domain_category(sp.domain, sp.description)
            domain_groups.setdefault(cat, []).append(sp.id)

        # Always run the gate — even "single-domain" labels may hide math/chem SPs
        # that were mislabeled by the planner
        if not any(
            solver_results.get(sp.id) and solver_results[sp.id].status == "solved"
            for sp in plan.sub_problems
        ):
            print("  → no solved SPs — gate skipped")
            return answer

        answer_lower = answer.lower()
        append_sections: List[str] = []

        for cat in ("PHYSICS", "MATHEMATICS", "CHEMISTRY", "OTHER"):
            sp_ids = domain_groups.get(cat, [])
            if not sp_ids:
                continue

            # Collect solved results for this domain
            solved_lines: List[str] = []
            for sp_id in sp_ids:
                sr = solver_results.get(sp_id)
                sp_obj = next((s for s in plan.sub_problems if s.id == sp_id), None)
                desc = sp_obj.description[:80] if sp_obj else sp_id
                if sr and sr.status == "solved" and sr.results_with_units:
                    for var, info in sr.results_with_units.items():
                        val = info.get("value", "")
                        unit = info.get("unit", "")
                        solved_lines.append(f"  {sp_id} | {var} = {val} {unit}".rstrip())
                elif sr and sr.status == "solved" and sr.results:
                    for var, val in sr.results.items():
                        solved_lines.append(f"  {sp_id} | {var} = {val}")

            if not solved_lines:
                continue  # no solved results to check

            # Check if this domain's results are represented in the answer
            # (look for at least one numeric value match OR domain header keyword)
            cat_keyword = cat.lower()  # "physics", "mathematics", "chemistry"
            has_header = any(kw in answer_lower for kw in
                             [f"## {cat_keyword}", f"# {cat_keyword}",
                              cat_keyword + " result", cat_keyword + " section"])

            if not has_header:
                # Also check if any computed values appear verbatim (first 6 chars)
                val_found = False
                for sp_id in sp_ids:
                    sr = solver_results.get(sp_id)
                    if sr and sr.results_with_units:
                        for info in sr.results_with_units.values():
                            val_str = str(info.get("value", ""))
                            if len(val_str) >= 4 and val_str[:4] in answer:
                                val_found = True
                                break
                if val_found:
                    has_header = True  # values are there, just no header

            if has_header:
                print(f"  ✅ {cat}: present in answer")
                continue

            # Domain is genuinely absent — generate targeted section
            print(f"  ⚠️  {cat}: MISSING from answer — generating section")
            results_str = "\n".join(solved_lines)
            prompt = (
                f"The {cat} section is missing from the answer below. "
                f"Write a complete '## {cat.title()} Results' section.\n\n"
                f"COMPUTED {cat} RESULTS:\n{results_str}\n\n"
                f"Write 3-5 sentences per result: method used, numerical value, "
                f"physical/mathematical meaning. Use markdown. "
                f"Start IMMEDIATELY with '## {cat.title()} Results'.\n"
            )
            section = await self._llm_query_coder(prompt)
            if section.strip():
                append_sections.append(section.strip())

        if not append_sections:
            return answer

        print(f"  ✍️  Appending {len(append_sections)} missing domain section(s)")
        return answer + "\n\n---\n\n" + "\n\n".join(append_sections)

    # ── Topological waves ─────────────────────────────────────────────────────

    @staticmethod
    def _topological_waves(plan: "SolvePlan") -> List[List[str]]:
        """
        Group sub_problems into parallel execution waves via BFS.
        Wave 0 = no dependencies; Wave 1 = depends only on wave 0; etc.
        """
        sp_map = {sp.id: sp for sp in plan.sub_problems}
        # Use dependency_order to determine sequence if available
        remaining = list(plan.dependency_order)
        completed: set = set()
        waves: List[List[str]] = []

        while remaining:
            wave = [
                sp_id for sp_id in remaining
                if sp_id in sp_map
                and all(dep in completed for dep in sp_map[sp_id].depends_on)
            ]
            if not wave:
                # Break cycles — dump remaining, but only IDs that exist in sp_map.
                # Stray IDs (e.g. "result_R5") from LLM hallucination would cause
                # KeyError when sp_map[sp_id] is accessed in the wave executor.
                wave = [x for x in remaining if x in sp_map]
                if not wave:
                    break   # truly nothing runnable — stop rather than spin
            waves.append(wave)
            completed.update(wave)
            remaining = [sp_id for sp_id in remaining if sp_id not in completed]

        return waves if waves else [[sp.id for sp in plan.sub_problems]]

    # ── Targeted research ─────────────────────────────────────────────────────

    async def _targeted_research(self, plan: "SolvePlan") -> Dict[str, str]:
        """
        For HYBRID questions: run lookup_queries per sub-problem and return
        a dict mapping sp_id → concatenated search snippets.
        """
        if not _HAS_SEARCH:
            return {}

        results: Dict[str, str] = {}
        all_queries: List[tuple] = []  # (sp_id, query)
        for sp in plan.sub_problems:
            for q in sp.lookup_queries:
                all_queries.append((sp.id, q))

        if not all_queries:
            return {}

        print(f"\n🔍 Targeted research: {len(all_queries)} queries across "
              f"{len(plan.sub_problems)} sub-problems")

        agent = FlexibleSearchAgent(
            searxng_url=self.searxng_url,
            timeout=30,
            max_results=3,
        )

        async def _one_search(sp_id: str, query: str) -> tuple:
            try:
                loop = asyncio.get_event_loop()
                sr = await loop.run_in_executor(
                    None,
                    lambda: agent.search_and_fetch(query, num_sources=2, fetch_content=False)
                )
                snippets = "\n".join(
                    f"[{r.source}] {r.title}: {r.snippet}" for r in sr[:3]
                )
                return sp_id, f"Query: {query}\n{snippets}"
            except Exception as e:
                return sp_id, f"Query: {query}\n(search error: {e})"

        tasks = [_one_search(sp_id, q) for sp_id, q in all_queries]
        pairs = await asyncio.gather(*tasks)

        for sp_id, text in pairs:
            results[sp_id] = results.get(sp_id, "") + "\n\n" + text

        return results

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def _synthesize(
        self,
        question: str,
        plan: "SolvePlan",
        solver_results: Dict[str, "SolverResult"],
    ) -> str:
        """
        Phase 2: Ask qwq:32b to chain results, check units, run a final
        verification if needed, and output a clean answer_data block.
        """
        plan_md = plan.to_markdown()[:2000]

        # Build structured result blocks (no raw logs)
        result_blocks = []
        for sp_id in plan.dependency_order:
            sr = solver_results.get(sp_id)
            if not sr:
                result_blocks.append(f"### {sp_id}\nStatus: NOT RUN\n")
                continue
            lines = [f"### {sp_id}", f"Status: {sr.status.upper()}"]
            if sr.results:
                lines.append("Results:")
                for var, val in sr.results.items():
                    unit = sr.results_with_units.get(var, {}).get("unit", "")
                    lines.append(f"  {var} = {val:.6g} {unit}".rstrip())
            if sr.verification_note:
                lines.append(f"Verification: {sr.verification_note}")
            result_blocks.append("\n".join(lines))

        results_section = "\n\n".join(result_blocks)

        prompt = f"""\
You are synthesising the results of a multi-step scientific computation.

ORIGINAL QUESTION:
{question}

SOLVE PLAN:
{plan_md}

SOLVER RESULTS:
{results_section}

Your tasks:
1. Chain the results in dependency order.
2. Check for unit mismatches (e.g. km vs m, degrees vs radians).
3. If a key result is missing or implausible, note it explicitly.
4. Output a concise ANSWER_DATA block:

ANSWER_DATA:
FINAL_RESULT: <main variable> = <value> <unit>
FINAL_RESULT: <other key variable> = <value> <unit>
UNIT_CHECK: <any mismatches or "all consistent">
PLAUSIBILITY: <brief physical sanity check>
CHAIN_SUMMARY: <one paragraph explaining how the sub-results connect>
END_ANSWER_DATA
"""
        print(f"\n🧠 Synthesising with {_MODEL_CODER} …")
        raw = await self._llm_query_coder(prompt)
        # If qwen fails, try fallback
        if not raw.strip():
            print("⚠️  qwq:32b empty — trying deepseek-r1 fallback")
            raw = await self._llm_query_fallback(prompt)
        return raw

    # ── Writer ────────────────────────────────────────────────────────────────

    async def _write_final_answer(
        self,
        question: str,
        synthesis: str,
        plan: "SolvePlan",
        solver_results: Dict[str, "SolverResult"],
        requirements: Optional[List] = None,
    ) -> str:
        """
        Phase 3: qwen2.5:14b writes a full-page report.
        Phase 3C: Lock C enforces negative constraints from requirements.
        """
        # Collect final codes for appendix
        code_appendix = []
        for sp_id in plan.dependency_order:
            sr = solver_results.get(sp_id)
            if sr and sr.final_code:
                code_appendix.append(f"#### {sp_id} — Final Code\n```python\n{sr.final_code}\n```")

        # Build verified-only results block — Writer must NOT invent numbers
        verified_results = []
        failed_sps = []
        for sp_id in plan.dependency_order:
            sr = solver_results.get(sp_id)
            sp = next((s for s in plan.sub_problems if s.id == sp_id), None)
            sp_desc = sp.description[:120] if sp else sp_id
            if sr and sr.status == "solved" and sr.results_with_units:
                for var, info in sr.results_with_units.items():
                    val = info.get("value", "")
                    unit = info.get("unit", "")
                    verified_results.append(f"  {sp_id} | {var} = {val} {unit}".strip())
            elif sr and sr.status == "solved" and sr.results:
                for var, val in sr.results.items():
                    verified_results.append(f"  {sp_id} | {var} = {val}")
            else:
                failed_sps.append(f"  {sp_id} | FAILED — {sp_desc}")

        # ── Domain isolation: group results by domain category ───────────────
        domain_groups: Dict[str, List[str]] = {}  # category → [sp_id, ...]
        for sp_id in plan.dependency_order:
            sp_obj = next((s for s in plan.sub_problems if s.id == sp_id), None)
            cat = self._domain_category(
                sp_obj.domain if sp_obj else "physics",
                sp_obj.description if sp_obj else "",
            )
            domain_groups.setdefault(cat, []).append(sp_id)

        # Build domain-structured results block
        domain_results_lines = []
        for cat in ("PHYSICS", "MATHEMATICS", "CHEMISTRY", "OTHER"):
            sp_ids_in_cat = domain_groups.get(cat, [])
            if not sp_ids_in_cat:
                continue
            domain_results_lines.append(f"\n── {cat} ──")
            for sp_id in sp_ids_in_cat:
                sr = solver_results.get(sp_id)
                sp_obj = next((s for s in plan.sub_problems if s.id == sp_id), None)
                sp_desc = sp_obj.description[:80] if sp_obj else sp_id
                if sr and sr.status == "solved" and sr.results_with_units:
                    for var, info in sr.results_with_units.items():
                        val = info.get("value", "")
                        unit = info.get("unit", "")
                        domain_results_lines.append(f"  {sp_id} | {var} = {val} {unit}".rstrip())
                elif sr and sr.status == "solved" and sr.results:
                    for var, val in sr.results.items():
                        domain_results_lines.append(f"  {sp_id} | {var} = {val}")
                else:
                    domain_results_lines.append(f"  {sp_id} | FAILED — {sp_desc}")

        domain_results_block = "\n".join(domain_results_lines) or "  (no results)"

        # Build coverage checklist
        checklist_lines = []
        for cat in ("PHYSICS", "MATHEMATICS", "CHEMISTRY", "OTHER"):
            sp_ids_in_cat = domain_groups.get(cat, [])
            if sp_ids_in_cat:
                checklist_lines.append(f"  • {cat}: {', '.join(sp_ids_in_cat)}")
        coverage_checklist = "\n".join(checklist_lines)

        failed_block = ("\nFAILED SUB-PROBLEMS (do NOT invent values for these):\n" +
                        "\n".join(failed_sps)) if failed_sps else ""

        prompt = f"""\
Write a complete, well-structured technical answer based ONLY on the verified computed results below.

VERIFIED COMPUTED RESULTS — grouped by domain:
{domain_results_block}
{failed_block}

DOMAIN COVERAGE CHECKLIST — your answer MUST contain a dedicated section for EACH item:
{coverage_checklist}
Do NOT skip any domain. If a domain has only failed SPs, still include that section
with "**[description]: Numerical solution failed.**" — no invented values.

CONTEXT (for framing only — do NOT use any numbers from here):
{question[:400]}

RULES — you MUST follow these:
1. Every number in your answer MUST come from the VERIFIED COMPUTED RESULTS above.
2. Structure your answer with one H2 section per domain category (## Physics Results,
   ## Mathematics Results, ## Chemistry Results, etc.).
3. Explain the physical/mathematical meaning and method for each result.
4. List all computed values in a summary table with units at the end.
5. Use markdown formatting (headers, bold, tables).
6. Minimum 400 words.

{"CODE APPENDIX:" + chr(10) + chr(10).join(code_appendix[:3]) if code_appendix else ""}
"""
        system = (
            "You are a technical writer. You ONLY report numbers that were explicitly "
            "computed and provided to you. If a value was not computed, you state it failed. "
            "You NEVER guess, estimate, or invent numerical results."
        )
        print(f"\n✍️  Writing final answer with {_MODEL_CODER} …")
        answer = await self._ollama_chat(
            model=_MODEL_CODER,
            prompt=prompt,
            system=system,
            timeout=1500,      # 25 min — 35b MoE safety margin (Swarm 3.17)
            num_predict=4096,
            emit_chunks=True,  # Swarm 3.20 — stream ORCH_THINK to dashboard
        )
        if not answer.strip():
            answer = f"## Result\n\n{synthesis}"

        # ── Phase 3C: Lock C — enforce negative constraints ───────────────
        print(f"\n{'─'*62}")
        elapsed_str = ""  # timing handled in _solve_react
        print(f"Phase 3C  ConstraintCheck")
        answer = await self._enforce_negative_constraints(answer, requirements or [])

        # ── Phase 3D: Append "## Solver Code" appendix (Swarm 3.20) ───────
        # Runs AFTER constraint filter so python keywords don't get rewritten
        # as prose. Deterministic — never trusts the writer to embed code.
        try:
            code_blocks: List[str] = []
            for sp_id in plan.dependency_order:
                sr = solver_results.get(sp_id)
                if not sr:
                    continue
                final_code = (getattr(sr, 'final_code', '') or '').strip()
                if not final_code:
                    continue
                sp_obj = next(
                    (s for s in plan.sub_problems if s.id == sp_id),
                    None,
                )
                desc = (sp_obj.description if sp_obj else '').strip()[:200]
                heading = f"### {sp_id} — {desc}" if desc else f"### {sp_id}"
                code_blocks.append(
                    f"{heading}\n\n```python\n{final_code}\n```\n"
                )
            if code_blocks:
                answer = (
                    answer.rstrip()
                    + "\n\n---\n\n## Solver Code\n\n"
                    + "Each sub-problem produced an executable Python script. "
                    + "Click **Copy** or **Download** in the dashboard to grab "
                    + "any block and run it locally.\n\n"
                    + "\n".join(code_blocks)
                )
                print(f"  📎 Appended {len(code_blocks)} solver code block(s)")
        except Exception as _ce:
            print(f"  (code appendix skipped: {_ce})")

        return answer

    # ── Lock C: Negative constraint enforcement ───────────────────────────────

    async def _enforce_negative_constraints(
        self,
        answer: str,
        requirements: List,
    ) -> str:
        """
        Scan answer for violations of negative constraints extracted from
        requirements (e.g. no_formulas, no_math).  Calls phi4:14b to detect
        violations, then qwen2.5:14b to rewrite only the offending sections.
        Retries rewrite up to 2 times with a rescan between attempts.
        Prepends a warning if both attempts fail.
        """
        if not requirements:
            print("  → OK (no requirements)")
            return answer

        # Collect all negative constraints across requirements
        constraint_items = []
        for r in requirements:
            nc = getattr(r, "negative_constraints", [])
            for c in nc:
                constraint_items.append((r.id, c, r.text))

        if not constraint_items:
            print("  → OK (no negative constraints)")
            return answer

        constraints_str = "\n".join(
            f"  - {rid} ({rtxt[:60]}): {c}"
            for rid, c, rtxt in constraint_items
        )

        # Initial violation scan
        check_result = await self._scan_violations_text(answer, constraints_str)

        if "VIOLATION:" not in check_result:
            print("  → OK (no violations detected)")
            return answer

        violations = re.findall(r"VIOLATION:", check_result)
        print(f"  → {len(violations)} violation(s) found — rewriting (up to 2 attempts)…")

        # 2-retry rewrite loop with rescan between attempts
        current = answer
        last_check = check_result
        for attempt in range(2):
            corrected = await self._rewrite_violations_text(current, last_check, constraints_str)
            if corrected and len(corrected.strip()) > 100 and not corrected.strip().startswith("Error:"):
                rescan = await self._scan_violations_text(corrected, constraints_str)
                if "VIOLATION:" not in rescan:
                    print(f"  → Constraint rewrite complete (attempt {attempt+1}, {len(corrected)} chars)")
                    return corrected
                print(f"  ⚠️  Attempt {attempt+1}: violations remain — retrying…")
                current = corrected
                last_check = rescan
            else:
                print(f"  ⚠️  Attempt {attempt+1}: rewrite failed or too short")

        # Both rewrites failed — flag answer with warning
        warning = (
            "⚠️ **Note**: This answer may contain content that violates a stated "
            "constraint (e.g., symbolic formulas where plain language was requested). "
            "The constraint enforcement system was unable to fully correct it.\n\n"
        )
        print(f"  ⚠️  Both rewrite attempts failed — prepending warning to original")
        return warning + answer

    async def _scan_violations_text(self, answer: str, constraints_str: str) -> str:
        """phi4:14b scan for constraint violations. Returns raw LLM output."""
        check_prompt = f"""\
Scan this answer for violations of these constraints:
{constraints_str}

Constraint meanings:
- no_formulas: answer must NOT use symbolic math (no ΔG, Σ, ∫, nF, dE, ΔH,
  Ksp, dT, etc.) in the relevant section
- no_equations: no mathematical equations
- no_math: conceptual explanation only, no numbers or formulas
- no_calculus: no integrals or derivatives

ANSWER (first 4000 chars):
{answer[:4000]}

For each violation output exactly:
VIOLATION: <section_title> | <constraint> | <offending_text_snippet>

If no violations, output only: OK
"""
        return await self._llm_query_coder(
            check_prompt,
            "You are a constraint checker. Be precise. Only report genuine violations."
        )

    async def _rewrite_violations_text(
        self, answer: str, check_result: str, constraints_str: str
    ) -> str:
        """qwen2.5:14b rewrite of offending sections only."""
        rewrite_prompt = f"""\
Rewrite the answer below to fix these constraint violations:

VIOLATIONS DETECTED:
{check_result[:1500]}

CONSTRAINTS TO ENFORCE:
{constraints_str}

ORIGINAL ANSWER:
{answer[:5000]}

Rules:
- Fix ONLY the sections mentioned in violations.
- Replace symbolic math with plain English descriptions of the same concept.
- Keep all other sections, headers, tables, and numerical results unchanged.
- Return the complete corrected answer in full.
"""
        return await self._llm_query_coder(
            rewrite_prompt,
            "You are a technical editor. Enforce the listed constraints while preserving meaning.",
        )

    # ── Delegation helpers ────────────────────────────────────────────────────

    async def _delegate_v2(self, question: str) -> str:
        if not _HAS_V2:
            return f"OrchestratorV2_1 not available. Question: {question}"
        print("↩️  Delegating to OrchestratorV2_1 …")
        v2 = OrchestratorV2_1(
            debug=self.debug,
            searxng_url=self.searxng_url,
            date_filter=self.date_filter,
            save_markdown=self.save_markdown,
        )
        return await v2.process_question(question, status=self.status)

    async def _delegate_engineer(self, question: str) -> str:
        if not _HAS_ENGINEER:
            print("⚠️  engineer_mode not available — falling back to V2_1")
            return await self._delegate_v2(question)
        print("🔧 Delegating to engineer_mode …")
        return await run_engineer_mode(
            problem=question,
            searxng_url=self.searxng_url,
            debug=self.debug,
            save_markdown=self.save_markdown,
        )

    # ── Classification ────────────────────────────────────────────────────────

    async def _classify(self, question: str):
        if not _HAS_CLASSIFIER:
            return None
        try:
            return await QuestionClassifier.classify(question, self._llm_query)
        except Exception as e:
            print(f"⚠️  Classification failed: {e}")
            return None

    # ── Stub solve (when ReactSolver is unavailable) ──────────────────────────

    @staticmethod
    async def _stub_solve(sp) -> "SolverResult":
        from react_solver import SolverResult as SR
        return SR(
            sub_problem_id=sp.id,
            status="failed",
            verification_note="ReactSolver not available",
        )

    # ── LLM helpers ───────────────────────────────────────────────────────────

    async def _llm_query(self, prompt: str, system_prompt: str = "") -> str:
        """Classifier + lightweight structured-output reasoning. Uses Ollama
        format:"json" so the model can never emit prose-only thinking replies.
        4096-token cap prevents JSON truncation on big classifier prompts.
        """
        return await OrchestratorV3._ollama_chat(
            model=_MODEL_PLANNER,
            prompt=prompt,
            system=system_prompt or "You are a helpful assistant.",
            timeout=300,
            num_predict=4096,
            format="json",
        )

    async def _llm_query_coder(self, prompt: str, system_prompt: str = "") -> str:
        """qwen2.5:14b — code generation and technical writing."""
        try:
            agent = BaseAgent(
                agent_id="v3_coder",
                agent_type=AgentType.WORKER,
                model_name=_MODEL_CODER,
                system_prompt=system_prompt or (
                    "You are an expert Python programmer and physicist. "
                    "Write complete, correct, directly executable code."
                ),
            )
            return await agent.query_llm(prompt, stream=False)
        except Exception as e:
            print(f"⚠️  _llm_query_coder error: {e}")
            return ""

    async def _llm_query_reasoner(self, prompt: str, system_prompt: str = "") -> str:
        """qwq:32b — synthesis + verification. Uses THINKING_ENABLED from ReactSolver."""
        from react_solver import ReactSolver as _RS
        return await self._ollama_chat(
            model=_MODEL_REASONER,
            prompt=prompt,
            system=system_prompt or (
                "You are an expert scientist and mathematician. "
                "Think step by step. Be precise with units and numerical values."
            ),
            timeout=1800,
            num_predict=_RS.NUM_PREDICT,
            think=_RS.THINKING_ENABLED,
        )

    async def _llm_query_planner(self, prompt: str, system_prompt: str = "") -> str:
        """qwq:32b — Lead Architect: deep CoT for requirement shredding + SP decomposition.
        keep_alive uses default (600s) — explicit unload happens before solver if needed."""
        system = system_prompt or (
            "You are the Lead Systems Architect. You are being evaluated on COMPLETENESS. "
            "Every distinct mathematical operation, integral, series, chemical reaction, "
            "or conceptual explanation in the prompt MUST have its own Sub-Problem. "
            "If you omit a single task, the entire mission fails. "
            "Output ONLY valid JSON matching the schema exactly."
        )
        return await OrchestratorV3._ollama_chat(
            model=_MODEL_SMART_PLANNER,
            prompt=prompt,
            system=system,
            timeout=600,        # 10 min ceiling for planning
            num_predict=4096,   # JSON plan fits well within 4k tokens
            format="json",      # Swarm 3.18 — force structured JSON output
        )

    async def _llm_query_fallback(self, prompt: str, system_prompt: str = "") -> str:
        """deepseek-r1:32b — fallback when qwq fails."""
        return await self._ollama_chat(
            model=_MODEL_FALLBACK,
            prompt=prompt,
            system=system_prompt or (
                "You are an expert scientist and mathematician. "
                "Think step by step. Be precise with units and numerical values."
            ),
            timeout=600,
            num_predict=6144,
        )

    @staticmethod
    async def _ollama_chat(
        model: str,
        prompt: str,
        system: str = "",
        timeout: int = 1800,
        num_predict: int = 2048,
        think: bool = True,
        keep_alive: int = 600,
        format: Optional[str] = None,   # Swarm 3.18 — set "json" for structured output
        emit_chunks: bool = False,      # Swarm 3.20 — emit ORCH_THINK per delta for dashboard
    ) -> str:
        """Direct Ollama /api/chat call using streaming (avoids HTTP timeout on large models)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive,
            # NOTE: do NOT pass "think" — Ollama 0.17+ rejects it for qwq/deepseek models.
            "options": {
                "temperature": 0.6,
                "num_predict": num_predict,
                "num_ctx": _NUM_CTX,
            },
        }
        if format:
            # Ollama API: top-level "format":"json" forces strict JSON-mode decode.
            payload["format"] = format

        def _stream() -> str:
            resp = requests.post(
                f"{_OLLAMA_URL}/api/chat",
                json=payload,
                stream=True,
                timeout=timeout,
            )
            resp.raise_for_status()
            parts = []
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        parts.append(delta)
                        if emit_chunks:
                            # ORCH_THINK: dashboard live-streams the writer composing
                            # the final answer (mirrors react_solver's [LLMTOK]).
                            try:
                                _esc = (
                                    delta.replace('\\', '\\\\').replace('\n', '\\n')
                                )
                                print(f"ORCH_THINK: {_esc}")
                            except Exception:
                                pass
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
            return "".join(parts)

        try:
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, _stream)
            # Strip <think>…</think> from reasoning models
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            print(f"⚠️  _ollama_chat({model}) error: {e}")
            return ""

    @staticmethod
    async def _unload_model(model: str) -> None:
        """Evict any model from VRAM (keep_alive=0 signal to Ollama)."""
        def _do():
            import requests as _req
            try:
                _req.post(
                    f"{_OLLAMA_URL}/api/chat",
                    json={"model": model, "messages": [], "keep_alive": 0},
                    timeout=10,
                )
            except Exception:
                pass
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do)
            print(f"  🧹 Unloaded {model}")
        except Exception as e:
            print(f"  ⚠️  Unload {model} skipped: {e}")

    @staticmethod
    async def _prewarm_model(model: str) -> None:
        """Pre-load model into VRAM with a minimal 1-token generation."""
        def _do():
            import requests as _req
            try:
                _req.post(
                    f"{_OLLAMA_URL}/api/generate",
                    json={"model": model, "prompt": " ", "stream": False,
                          "keep_alive": 600, "options": {"num_predict": 1}},
                    timeout=600,
                )
            except Exception:
                pass
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _do)
            print(f"  🔥 Pre-warmed {model}")
        except Exception as e:
            print(f"  ⚠️  Pre-warm {model} skipped: {e}")

    @staticmethod
    async def _unload_solver_model() -> None:
        """Evict the ReactSolver model. Called after Phase 1 before synthesis."""
        try:
            from react_solver import ReactSolver as _RS
            await OrchestratorV3._unload_model(_RS.MODEL)
        except Exception as e:
            print(f"  ⚠️  VRAM unload skipped: {e}")
