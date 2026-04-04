"""
Swarm 3.1 — OrchestratorV3

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
from typing import Dict, List, Optional, Any
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
_MODEL_REASONER = "qwq:32b"
_MODEL_FALLBACK  = "deepseek-r1:32b"
_MODEL_CODER     = "qwen2.5:14b"
_MODEL_PLANNER   = "phi4:14b"


# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorV3:
    """
    Swarm 3.1 top-level dispatcher.
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

        print("🚀 Swarm 3.1 OrchestratorV3 initialized")
        print("   ✅ ReAct solver pipeline (MATHEMATICAL/HYBRID)")
        print("   ✅ Delegation to V2_1 (THEORETICAL/UNKNOWN)")
        print("   ✅ Delegation to engineer_mode (ENGINEERING_DESIGN)")

    # ── Entry point ───────────────────────────────────────────────────────────

    async def process_question(self, question: str, status=None) -> str:
        """Answer any question. Drop-in interface for OrchestratorV2_1."""
        self.status = status
        t0 = datetime.now()
        _qtype_val: str = "unknown"   # track before try so except can read it

        print("\n" + "="*70)
        print("🚀 SWARM 3.1 — OrchestratorV3")
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
                return await self._solve_react(question, classification, qtype.value)

            # Fallback for any other classification
            return await self._delegate_v2(question)

        except Exception as e:
            print(f"\n❌ OrchestratorV3 error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
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
    ) -> str:
        t0 = time.time()
        _sep = "─" * 62

        def _elapsed() -> str:
            return f"{time.time() - t0:.1f}s"

        # ── Phase 0B: Generate SolvePlan ─────────────────────────────────
        print(f"\n{_sep}")
        print(f"Phase 0B  Planning          │ {_elapsed()}")
        if self.status:
            self.status.set_phase(2, "Planning")

        requirements: List = []
        if _HAS_PLANNER_V2:
            plan, requirements = await PlannerV2.create_plan(
                question, classification, self._llm_query
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

        # ── Phase 0C: Targeted research (HYBRID only) ─────────────────────
        research_contexts: Dict[str, str] = {}
        if qtype_value == "hybrid":
            print(f"\n{_sep}")
            print(f"Phase 0C  Research          │ {_elapsed()}")
            if self.status:
                self.status.set_phase(3, "Research")
            research_contexts = await self._targeted_research(plan)

        # ── VRAM handoff: evict phi4, pre-warm solver ─────────────────────
        solver_model = "deepseek-r1:14b"
        try:
            from react_solver import ReactSolver as _RS_tmp
            solver_model = _RS_tmp.MODEL
        except Exception:
            pass
        print(f"\n  🔄 VRAM handoff: {_MODEL_PLANNER} → {solver_model}")
        await self._unload_model(_MODEL_PLANNER)
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

            # Inject outputs from previous waves into inputs for this wave
            for sp_id in wave:
                sp = sp_map[sp_id]
                for dep_id in sp.depends_on:
                    dep_result = solver_results.get(dep_id)
                    if dep_result and dep_result.results:
                        for var, val in dep_result.results.items():
                            if var not in sp.inputs:
                                sp.inputs[var] = val

            # Run this wave in parallel
            tasks = []
            for sp_id in wave:
                sp = sp_map[sp_id]
                ctx = research_contexts.get(sp_id, "")
                if _HAS_REACT:
                    solver = ReactSolver(
                        sub_problem=sp,
                        plan=plan,
                        research_context=ctx,
                        searxng_url=self.searxng_url,
                    )
                    tasks.append(solver.solve())
                else:
                    tasks.append(self._stub_solve(sp))

            wave_results = await asyncio.gather(*tasks)
            for result in wave_results:
                solver_results[result.sub_problem_id] = result

            # ── Post-wave: retry SPs with expected_outputs but 0 results ─
            retry_tasks = []
            retry_sp_ids = []
            for result in wave_results:
                sp_id = result.sub_problem_id
                sp = sp_map.get(sp_id)
                if sp and result.status != "solved" and sp.expected_outputs and _HAS_REACT:
                    print(f"  ⚡ Retrying {sp_id} (status={result.status}, "
                          f"0 computed results, outputs expected)…")
                    retry_solver = ReactSolver(
                        sub_problem=sp,
                        plan=plan,
                        research_context=research_contexts.get(sp_id, ""),
                        searxng_url=self.searxng_url,
                    )
                    # Inject a mandatory-code message at the top of history
                    retry_solver._history.append({
                        "role": "user",
                        "content": (
                            "Previous attempt returned 0 results. "
                            "You MUST write Python code and use ACTION: run_code. "
                            "Do not describe the calculation in prose — execute it."
                        ),
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

        # ── Free VRAM before synthesis (evict reactor model) ───────────────
        await self._unload_solver_model()

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

        n_solved = sum(1 for r in solver_results.values() if r.status == "solved")
        elapsed = time.time() - t0
        print(f"\n{_sep}")
        print(f"✅ Done │ {n_solved}/{len(solver_results)} SP(s) solved │ Total: {elapsed:.1f}s")
        return answer

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
                # Break cycles — just dump everything left
                wave = remaining[:]
            waves.append(wave)
            completed.update(wave)
            remaining = [sp_id for sp_id in remaining if sp_id not in completed]

        return waves if waves else [plan.dependency_order]

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
        print("\n🧠 Synthesising with qwen2.5:14b …")
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

        results_block = "\n".join(verified_results) if verified_results else "  (no numerical results computed)"
        failed_block  = ("\nFAILED SUB-PROBLEMS (do NOT invent values for these):\n" +
                         "\n".join(failed_sps)) if failed_sps else ""

        prompt = f"""\
Write a complete, well-structured technical answer based ONLY on the verified computed results below.

VERIFIED COMPUTED RESULTS (these are the ONLY numbers you may use):
{results_block}
{failed_block}

CONTEXT (for framing only — do NOT use any numbers from here):
{question[:400]}

RULES — you MUST follow these:
1. Every number in your answer MUST come from the VERIFIED COMPUTED RESULTS above.
2. If a sub-problem FAILED, write exactly: "**[SP description]: Numerical solution failed.**"
   Do NOT guess, estimate, or invent a value for failed sub-problems.
3. Explain the physical meaning and method for each result.
4. List all computed values in a summary table with units.
5. Use markdown formatting (headers, bold, tables).
6. Minimum 400 words.

{"CODE APPENDIX:" + chr(10) + chr(10).join(code_appendix[:3]) if code_appendix else ""}
"""
        system = (
            "You are a technical writer. You ONLY report numbers that were explicitly "
            "computed and provided to you. If a value was not computed, you state it failed. "
            "You NEVER guess, estimate, or invent numerical results."
        )
        print("\n✍️  Writing final answer with qwen2.5:14b …")
        answer = await self._ollama_chat(
            model=_MODEL_CODER,
            prompt=prompt,
            system=system,
            timeout=900,       # CPU-only fallback needs up to ~10 min
            num_predict=4096,
        )
        if not answer.strip():
            answer = f"## Result\n\n{synthesis}"

        # ── Phase 3C: Lock C — enforce negative constraints ───────────────
        print(f"\n{'─'*62}")
        elapsed_str = ""  # timing handled in _solve_react
        print(f"Phase 3C  ConstraintCheck")
        answer = await self._enforce_negative_constraints(answer, requirements or [])

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
        Returns the answer unchanged if no constraints are defined or no
        violations are found.
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
        check_result = await self._llm_query_coder(
            check_prompt,
            "You are a constraint checker. Be precise. Only report genuine violations."
        )

        if "VIOLATION:" not in check_result:
            print("  → OK (no violations detected)")
            return answer

        violations = re.findall(r"VIOLATION:", check_result)
        print(f"  → {len(violations)} violation(s) found — rewriting offending sections…")

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
        corrected = await self._llm_query_coder(
            rewrite_prompt,
            "You are a technical editor. Enforce the listed constraints while preserving meaning.",
        )

        is_error = (not corrected.strip() or
                    corrected.strip().startswith("Error:") or
                    len(corrected.strip()) < 50)
        if is_error:
            print(f"  ⚠️  Constraint rewrite failed — returning original answer")
            return answer
        print(f"  → Constraint rewrite complete ({len(corrected)} chars)")
        return corrected

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
        """phi4:14b — planning, classification, lightweight reasoning."""
        try:
            agent = BaseAgent(
                agent_id="v3_phi4",
                agent_type=AgentType.WORKER,
                model_name=_MODEL_PLANNER,
                system_prompt=system_prompt or "You are a helpful assistant.",
            )
            return await agent.query_llm(prompt, stream=False)
        except Exception as e:
            print(f"⚠️  _llm_query error: {e}")
            return ""

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
            "keep_alive": 600,  # keep loaded 10 min — explicit unload at phase transitions
            # NOTE: do NOT pass "think" — Ollama 0.17+ rejects it for qwq/deepseek models.
            "options": {
                "temperature": 0.6,
                "num_predict": num_predict,
            },
        }

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
