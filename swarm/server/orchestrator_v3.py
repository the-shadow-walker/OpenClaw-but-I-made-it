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
    from planner_v2 import PlannerV2, SolvePlan, SubProblem
    _HAS_PLANNER_V2 = True
except ImportError:
    _HAS_PLANNER_V2 = False
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

        print("\n" + "="*70)
        print("🚀 SWARM 3.1 — OrchestratorV3")
        print("="*70)
        print(f"Q: {question[:100]}")
        print("="*70)

        try:
            # ── Phase 0A: Classify ────────────────────────────────────────
            if self.status:
                self.status.set_phase(1, "Classification")
            classification = await self._classify(question)
            qtype = classification.question_type if classification else None
            print(f"🎯 Type: {qtype.value.upper() if qtype else 'UNKNOWN'}")

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
            # Last-resort fallback
            if _HAS_V2:
                print("↩️  Falling back to V2_1 due to error")
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

        # ── Phase 0B: Generate SolvePlan ─────────────────────────────────
        if self.status:
            self.status.set_phase(2, "Planning")
        if _HAS_PLANNER_V2:
            plan = await PlannerV2.create_plan(question, classification, self._llm_query)
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

        print(f"\n📋 Plan: {len(plan.sub_problems)} SP(s), order: {plan.dependency_order}")
        print(plan.to_markdown()[:800])

        # ── Phase 0C: Targeted research (HYBRID only) ─────────────────────
        research_contexts: Dict[str, str] = {}
        if qtype_value == "hybrid":
            if self.status:
                self.status.set_phase(3, "Research")
            research_contexts = await self._targeted_research(plan)

        # ── Phase 1: Run ReactSolvers in topological waves ─────────────────
        if self.status:
            self.status.set_phase(4, "Solving")

        solver_results: Dict[str, SolverResult] = {}
        waves = self._topological_waves(plan)
        print(f"\n⚡ Executing {len(waves)} wave(s): {waves}")

        for wave_idx, wave in enumerate(waves):
            print(f"\n  Wave {wave_idx+1}/{len(waves)}: {wave}")

            # Inject outputs from previous waves into inputs for this wave
            for sp_id in wave:
                sp = next(s for s in plan.sub_problems if s.id == sp_id)
                for dep_id in sp.depends_on:
                    dep_result = solver_results.get(dep_id)
                    if dep_result and dep_result.results:
                        for var, val in dep_result.results.items():
                            if var not in sp.inputs:
                                sp.inputs[var] = val

            # Run this wave in parallel
            tasks = []
            for sp_id in wave:
                sp = next(s for s in plan.sub_problems if s.id == sp_id)
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
                    # No ReactSolver — return a stub failure
                    tasks.append(self._stub_solve(sp))

            wave_results = await asyncio.gather(*tasks)
            for result in wave_results:
                solver_results[result.sub_problem_id] = result
                print(f"  {result.sub_problem_id}: {result.status.upper()} "
                      f"({result.turn_count} turns, {len(result.results)} results)")

        # ── Phase 2: Synthesis ─────────────────────────────────────────────
        if self.status:
            self.status.set_phase(5, "Synthesis")
        synthesis = await self._synthesize(question, plan, solver_results)

        # ── Phase 3: Writer ────────────────────────────────────────────────
        if self.status:
            self.status.set_phase(6, "Writing")
        answer = await self._write_final_answer(question, synthesis, plan, solver_results)

        elapsed = time.time() - t0
        print(f"\n✅ Done in {elapsed:.1f}s | "
              f"{sum(1 for r in solver_results.values() if r.status=='solved')}/"
              f"{len(solver_results)} SP(s) solved")
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
        print("\n🧠 Synthesising with qwq:32b …")
        raw = await self._llm_query_reasoner(prompt)
        # If qwq fails, try fallback
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
    ) -> str:
        """
        Phase 3: qwen2.5:14b writes a full-page report.
        """
        # Collect sources from search tool calls (best-effort)
        all_sources: List[str] = []

        # Collect final codes for appendix
        code_appendix = []
        for sp_id in plan.dependency_order:
            sr = solver_results.get(sp_id)
            if sr and sr.final_code:
                code_appendix.append(f"#### {sp_id} — Final Code\n```python\n{sr.final_code}\n```")

        prompt = f"""\
Write a complete, well-structured technical answer to this question.

QUESTION:
{question}

SYNTHESIS (chain of results from the solver):
{synthesis[:3000]}

REQUIREMENTS:
- Start with a clearly highlighted final answer (bold or header).
- Explain step-by-step reasoning in plain language.
- List all assumptions and their physical basis.
- Include a table or list of all computed values with units.
- Mention what the user needs to understand the result.
- Minimum length: one full page (500+ words).
- Use markdown formatting (headers, bold, tables, code blocks).

{"CODE APPENDIX:" + chr(10) + chr(10).join(code_appendix[:3]) if code_appendix else ""}
"""
        system = (
            "You are a technical writer and scientist. "
            "Write precise, well-structured explanations with correct units. "
            "Always present the key numerical result prominently at the top."
        )
        print("\n✍️  Writing final answer with qwen2.5:14b …")
        answer = await self._llm_query_coder(prompt, system)
        if not answer.strip():
            # Fallback: return the synthesis as-is
            answer = f"## Result\n\n{synthesis}"
        return answer

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
        """qwq:32b — deep reasoning, synthesis, verification. Long timeout."""
        return await self._ollama_chat(
            model=_MODEL_REASONER,
            prompt=prompt,
            system=system_prompt or (
                "You are an expert scientist and mathematician. "
                "Think step by step. Be precise with units and numerical values."
            ),
            timeout=600,
            num_predict=6144,
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
        num_predict: int = 4096,
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
