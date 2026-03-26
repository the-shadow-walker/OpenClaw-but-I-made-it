"""
Swarm 3.0 Orchestrator

Handles:
1. Better value extraction (finds "5000 kg" not "A=3, a=100")
2. Non-math questions properly (just answer from search)
3. Equation generator only called for math problems
4. Graceful degradation on all failure types
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
from typing import Dict, Optional, List
from datetime import datetime

from shared_memory import SharedMemory, Source, FactType
from base_agent import BaseAgent
from core import AgentType

# New imports
try:
    from question_classifier import QuestionClassifier, QuestionType
    HAS_CLASSIFIER = True
except:
    HAS_CLASSIFIER = False

try:
    from equation_generator import EquationGenerator
    HAS_GENERATOR = True
except:
    HAS_GENERATOR = False

try:
    from equation_validator import EquationValidator, EquationExecutor
    HAS_EXECUTOR = True
except:
    HAS_EXECUTOR = False

try:
    from value_extractor import ValueExtractor
    HAS_EXTRACTOR = True
except:
    HAS_EXTRACTOR = False

try:
    from flexible_search_agent import FlexibleSearchAgent
    HAS_SEARCH = True
except:
    HAS_SEARCH = False

try:
    from checklist_system import ChecklistGenerator, AgentTaskAssigner, Phase1Checklist, Phase2Checklist, TodoStatus
    HAS_CHECKLISTS = True
except:
    HAS_CHECKLISTS = False

try:
    from planner_agent import PlannerAgent
    HAS_PLANNER = True
except:
    HAS_PLANNER = False

try:
    from writer_agent import WriterAgent
    HAS_WRITER = True
except:
    HAS_WRITER = False

try:
    from consensus_agent import ConsensusAgent
    HAS_CONSENSUS = True
except:
    HAS_CONSENSUS = False

try:
    from search_parallel import ParallelSearchCoordinator, DeepSearchCoordinator
    HAS_PARALLEL = True
except:
    HAS_PARALLEL = False

try:
    from formal_calculator import FormalCalculator
    _HAS_FORMAL_CALC = True
except ImportError:
    _HAS_FORMAL_CALC = False

try:
    from math_verifier import MathVerifier
    _HAS_MATH_VERIFIER = True
except ImportError:
    _HAS_MATH_VERIFIER = False

try:
    from physics_supervisor import PhysicsSupervisor, PhysicsEquationPlan
    HAS_SUPERVISOR = True
except ImportError:
    HAS_SUPERVISOR = False


class OrchestratorV2_1:
    """
    Swarm 3.0 - Enhanced Universal Question Answerer

    Handles:
    - Mathematical questions (extract values, generate equations, compute)
    - Theoretical questions (research + explain)
    - Hybrid questions (both)
    - Mega-problems (decompose into sub-problems)
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
    ):
        self.memory = SharedMemory()
        self.max_search_concurrent = max_search_concurrent
        self.enable_verification = enable_verification
        self.debug = debug
        self.searxng_url = searxng_url or os.getenv(
            "SEARXNG_URL",
            "http://10.0.0.58:8080"
        )

        self.deep_research = deep_research
        self.context_window_size = context_window_size
        self.date_filter = date_filter
        self.save_markdown = save_markdown
        self.markdown_path: Optional[str] = None
        self.status = None   # set by process_question when a StatusDisplay is passed
        
        self.classification = None
        self.question_type = QuestionType.UNKNOWN if HAS_CLASSIFIER else None
        self.sub_problems = []
        self.plan_result = None
        
        self.is_math_question = False
        self.is_engineering_design = False
        self.raw_content: List[str] = []
        self.results = {}
        
        self.phase_1_checklist = None
        self.phase_2_checklist = None
        
        self.start_time = None
        self.phase_times = {}
        
        print("🚀 Swarm 3.0 Orchestrator initialized")
        print("   ✅ Search + Summarize for ANY question")
        print("   ✅ Better value extraction")
        print("   ✅ Non-math question handling")
        print("   ✅ Problem decomposition")
    
    async def process_question(self, question: str, status=None) -> str:
        """Answer any question"""
        self.start_time = datetime.now()
        self.memory.original_question = question
        self.status = status   # optional StatusDisplay passed from the UI layer

        print("\n" + "="*70)
        print(f"🚀 SWARM 3.0 - ANSWERING")
        print("="*70)
        print(f"Q: {question[:80]}...")
        print("="*70)

        try:
            # PHASE 0A: Classify
            if HAS_CLASSIFIER:
                if self.status:
                    self.status.set_phase(1, "Classification")
                await self._phase_0a_classify(question)

            # EARLY EXIT: Engineering Design → delegate to engineer_mode
            if self.is_engineering_design:
                try:
                    from engineer_mode import run_engineer_mode
                    return await run_engineer_mode(
                        problem=question,
                        searxng_url=self.searxng_url,
                        debug=self.debug,
                        save_markdown=self.save_markdown,
                    )
                except ImportError:
                    print("⚠️  engineer_mode.py not available — falling back to HYBRID")
                    self.question_type = QuestionType.HYBRID
                    self.is_math_question = True
                    self.is_engineering_design = False

            # PHASE 1: Plan
            if self.status:
                self.status.set_phase(2, "Planning")
            await self._phase_1_plan_and_decompose(question)

            # PHASE 1A: Create information checklist
            await self._phase_1a_create_checklist(question)

            # PHASE 2: Search
            if self.status:
                self.status.set_phase(3, "Search")
            await self._phase_2_search(question)

            # PHASE 2A: Create solution checklist (if math problem)
            if self.is_math_question:
                await self._phase_2a_create_solution_checklist(question)

            # PHASE 3-4: Math (only if applicable)
            if self.question_type in [QuestionType.MATHEMATICAL, QuestionType.HYBRID]:
                if self.status:
                    self.status.set_phase(4, "Math")
                await self._phase_3_4_solve_math()

            # PHASE 5: Summary
            if self.status:
                self.status.set_phase(5, "Summary")
            answer = await self._phase_5_summarize(question)

            # Optional: deep Markdown report
            if self.save_markdown:
                if self.status:
                    self.status.set_phase(6, "Markdown Report")
                await self._save_markdown_report(question)

            self._print_summary()
            return answer
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return f"Unable to answer: {e}"
    
    async def _phase_0a_classify(self, question: str) -> None:
        """PHASE 0A: Classify the question"""
        print("\n" + "="*70)
        print("PHASE 0A: QUESTION CLASSIFICATION")
        print("="*70)
        
        phase_start = datetime.now()
        
        try:
            self.classification = await QuestionClassifier.classify(
                question,
                self._llm_query
            )
            
            self.question_type = self.classification.question_type
            self.is_math_question = self.question_type in [
                QuestionType.MATHEMATICAL,
                QuestionType.HYBRID
            ]
            if HAS_CLASSIFIER and self.question_type == QuestionType.ENGINEERING_DESIGN:
                self.is_engineering_design = True

            print(f"\n🎯 Type: {self.question_type.value.upper()}")
            print(f"   Solvable: {'✅ YES' if self.classification.is_solvable else '⚠️  PARTIAL'}")
        
        except Exception as e:
            print(f"⚠️  Classification failed: {e}")
            self.question_type = QuestionType.UNKNOWN
            self.is_math_question = False
        
        self.phase_times['classification'] = (datetime.now() - phase_start).total_seconds()
    
    async def _phase_1_plan_and_decompose(self, question: str) -> None:
        """PHASE 1: Plan and decompose"""
        print("\n" + "="*70)
        print("PHASE 1: PLANNING & DECOMPOSITION")
        print("="*70)
        
        phase_start = datetime.now()
        
        try:
            # Check for mega-problems
            indicators = ['while', 'simultaneously', 'meanwhile', 'also', 'and also a']
            has_multiple = sum(1 for ind in indicators if ind in question.lower()) > 2
            
            if has_multiple and self.is_math_question:
                print(f"\n🔀 Multiple scenarios detected, decomposing...")
                self.sub_problems = await self._decompose_problem(question)
                
                if self.sub_problems:
                    print(f"✓ Decomposed into {len(self.sub_problems)} sub-problems")
                else:
                    print(f"✓ Decomposition failed, treating as single problem")
                    self.sub_problems = [{
                        'title': 'Main Problem',
                        'question': question,
                    }]
            else:
                print(f"\n✓ Single problem")
                self.sub_problems = [{
                    'title': 'Main Problem',
                    'question': question,
                }]
        
        except Exception as e:
            print(f"⚠️  Planning error: {e}")
            self.sub_problems = [{
                'title': 'Main Problem',
                'question': question,
            }]

        # Always run PlannerAgent to generate targeted sub-queries
        if HAS_PLANNER:
            try:
                planner = PlannerAgent(self.memory)
                self.plan_result = await planner.plan(question)
                print(f"✓ Planner produced {len(self.plan_result.get('search_needed', []))} search queries")
            except Exception as e:
                print(f"⚠️  PlannerAgent error: {e}")
                self.plan_result = None

        self.phase_times['planning'] = (datetime.now() - phase_start).total_seconds()
    
    async def _phase_1a_create_checklist(self, question: str) -> None:
        """PHASE 1A: Create Phase 1 TODO checklist"""
        print("\n" + "="*70)
        print("PHASE 1A: CREATE INFORMATION CHECKLIST")
        print("="*70)
        
        if not HAS_CHECKLISTS:
            print("⏭️  Checklist system not available")
            return
        
        try:
            self.phase_1_checklist = await ChecklistGenerator.generate_phase_1(
                question,
                self._llm_query
            )
            
            print(self.phase_1_checklist.format())
            
            # Assign tasks to agents
            tasks = AgentTaskAssigner.assign_phase_1_tasks(self.phase_1_checklist)
            
            print(f"\n📋 Assigning {sum(len(t) for t in tasks.values())} tasks to agents:")
            for agent_name, task_list in tasks.items():
                if task_list:
                    print(f"   → {agent_name}: {len(task_list)} tasks")
        
        except Exception as e:
            print(f"⚠️  Checklist generation failed: {e}")
            self.phase_1_checklist = None
    
    async def _decompose_problem(self, question: str) -> List[Dict]:
        """Decompose mega-problem"""
        try:
            response = await self._llm_query(
                f"""Identify independent sub-problems in this question:

{question}

For EACH sub-problem provide JSON (no markdown):
{{
  "sub_problems": [
    {{"title": "Problem 1", "description": "what it is"}},
    {{"title": "Problem 2", "description": "what it is"}}
  ]
}}
""",
                system_prompt="Extract ONLY JSON, nothing else."
            )
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [{'title': p['title'], 'question': question} for p in data.get('sub_problems', [])]
        except:
            pass
        
        return []
    
    async def _phase_2_search(self, question: str) -> None:
        """PHASE 2: Deep parallel search"""
        print("\n" + "="*70)
        print("PHASE 2: SEARCH")
        print("="*70)

        phase_start = datetime.now()

        if not HAS_SEARCH:
            print("⚠️  Search not available")
            self.phase_times['search'] = 0
            return

        try:
            search_agent = FlexibleSearchAgent(
                searxng_url=self.searxng_url,
                max_results=5,
                date_filter=self.date_filter,
            )

            # Build sub-query list from planner output (up to 4 queries)
            sub_queries = []
            if self.plan_result:
                for q in self.plan_result.get('search_needed', [])[:4]:
                    if q and q != question:
                        sub_queries.append(q)

            # Run coordinator on sub-queries (writes facts to SharedMemory)
            # THEORETICAL/HYBRID → iterative deep search with reflection loop
            # MATHEMATICAL       → simple parallel search (values must be precise)
            if HAS_PARALLEL and sub_queries:
                if self.question_type in (QuestionType.THEORETICAL, QuestionType.HYBRID):
                    coordinator = DeepSearchCoordinator(
                        memory=self.memory,
                        max_concurrent=self.max_search_concurrent,
                        searxng_url=self.searxng_url,
                        search_budget=12,
                        date_filter=self.date_filter,
                    )
                    await coordinator.research(question, sub_queries)
                else:
                    coordinator = ParallelSearchCoordinator(
                        memory=self.memory,
                        max_concurrent=self.max_search_concurrent,
                        searxng_url=self.searxng_url,
                        date_filter=self.date_filter,
                    )
                    await coordinator.search_all(sub_queries)

            # Fetch full content for top-level question (5 sources, 5000 chars each)
            results = search_agent.search_and_fetch(question, num_sources=5)

            if results:
                print(f"✅ Found {len(results)} sources for main query")
                for i, result in enumerate(results, 1):
                    if result.content and len(result.content) > 100:
                        content = result.content[:5000]
                        self.raw_content.append(content)
                        source = Source(
                            url=result.url,
                            title=result.title,
                            search_query=question
                        )
                        self.memory.add_fact(
                            content=content[:500],
                            fact_type=FactType.SEARCH_RESULT,
                            source=source
                        )
                        print(f"   [{i}] {result.title[:50]}...")

        except Exception as e:
            print(f"⚠️  Search error: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

        fact_count = len(self.memory.get_facts(fact_type=FactType.SEARCH_RESULT))
        self.phase_times['search'] = (datetime.now() - phase_start).total_seconds()
        print(f"✅ Search complete - {len(self.raw_content)} sources fetched, {fact_count} facts in memory")

        if self.status:
            self.status.set_stat("Sources",  len(self.raw_content))
            self.status.set_stat("Facts",    fact_count)

    async def _phase_2b_expand_search(self, question: str, search_agent) -> None:
        """PHASE 2B: Expand search when coverage is thin (THEORETICAL only)"""
        facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)
        if len(facts) >= 5:
            print(f"\n✅ Coverage sufficient ({len(facts)} facts) - skipping expand")
            return

        print(f"\n🔍 Coverage thin ({len(facts)} facts) - expanding search...")

        try:
            context = "\n".join(self.raw_content[:2])[:2000] if self.raw_content else ""
            prompt = f"""Based on what we know about: {question}

Current research summary:
{context}

List 2-3 specific follow-up search queries to fill gaps. Output one query per line, no numbering."""

            follow_up_text = await self._llm_query(
                prompt,
                system_prompt="Output only search queries, one per line."
            )
            follow_up_queries = [
                q.strip() for q in follow_up_text.strip().split('\n') if q.strip()
            ][:3]

            for query in follow_up_queries:
                print(f"   🔎 Follow-up: {query}")
                results = search_agent.search_and_fetch(query, num_sources=3)
                for result in results:
                    if result.content and len(result.content) > 100:
                        content = result.content[:5000]
                        self.raw_content.append(content)
                        source = Source(
                            url=result.url,
                            title=result.title,
                            search_query=query
                        )
                        self.memory.add_fact(
                            content=content[:500],
                            fact_type=FactType.SEARCH_RESULT,
                            source=source
                        )

        except Exception as e:
            print(f"⚠️  Expand search error: {e}")
    
    async def _phase_2a_create_solution_checklist(self, question: str) -> None:
        """PHASE 2A: Create Phase 2 TODO checklist"""
        print("\n" + "="*70)
        print("PHASE 2A: CREATE SOLUTION CHECKLIST")
        print("="*70)
        
        if not HAS_CHECKLISTS:
            print("⏭️  Checklist system not available")
            return
        
        if not self.is_math_question:
            print("⏭️  Skipping solution checklist (not a math problem)")
            return
        
        try:
            # Create available info summary
            available_info = f"Found {len(self.raw_content)} sources"
            if self.classification:
                available_info += f"\nGiven variables: {', '.join(self.classification.given_variables[:5])}"
                available_info += f"\nUnknown variables: {', '.join(self.classification.unknown_variables[:5])}"
            
            self.phase_2_checklist = await ChecklistGenerator.generate_phase_2(
                question,
                available_info,
                self._llm_query
            )
            
            print(self.phase_2_checklist.format())
            
            # Assign tasks to agents
            tasks = AgentTaskAssigner.assign_phase_2_tasks(self.phase_2_checklist)
            
            print(f"\n📋 Assigning {sum(len(t) for t in tasks.values())} computational tasks:")
            for agent_name, task_list in tasks.items():
                if task_list:
                    print(f"   → {agent_name}: {len(task_list)} tasks")
        
        except Exception as e:
            print(f"⚠️  Solution checklist failed: {e}")
            self.phase_2_checklist = None
    
    async def _phase_3_4_solve_math(self) -> None:
        """PHASE 3 & 4: Generate and execute (only for math problems)"""
        
        if not self.is_math_question:
            print("\n⏭️  Skipping math solving (not a math problem)")
            return
        
        print("\n" + "="*70)
        print("PHASE 3: EQUATION GENERATION")
        print("="*70)
        
        print("\n" + "="*70)
        print("PHASE 4: VALIDATION & EXECUTION")
        print("="*70)
        
        phase_start = datetime.now()
        
        for i, problem in enumerate(self.sub_problems, 1):
            print(f"\n{'─'*70}")
            print(f"Sub-Problem {i}: {problem['title']}")
            print(f"{'─'*70}")
            
            try:
                # ── 1. Extract explicitly given numeric values ─────────────
                if HAS_EXTRACTOR:
                    given_values = ValueExtractor.extract_all(problem['question'])
                else:
                    given_values = self._extract_values_simple(problem['question'])

                # Drop obviously-wrong zero values from comma-stripping artefacts
                _zero_keys = [k for k, v in given_values.items()
                              if v == 0.0 and k not in ('initial_velocity', 'v0', 'u', 'theta')]
                for k in _zero_keys:
                    del given_values[k]

                if given_values:
                    print(f"\n📋 Extracted {len(given_values)} explicit value(s):")
                    for key, val in given_values.items():
                        print(f"   {key} = {val}")
                else:
                    print("\n📋 No explicit numeric values in question — LLM will use physics defaults.")

                # ── 2. Build search-context string for the generator ───────
                search_context = self._build_math_context(for_code_gen=True)

                # ── Phase 3-pre: Physics Supervisor ───────────────────────────────────
                physics_plan = None
                if HAS_SUPERVISOR:
                    print("\n  🔬 Physics Supervisor deriving equations (phi4)...")
                    try:
                        physics_plan = await PhysicsSupervisor.derive_equations(
                            problem=problem['question'],
                            given_variables=given_values,
                            llm_query_func=self._llm_query,   # phi4:14b (reasoning model)
                        )
                        if physics_plan:
                            print(f"  ✅ Supervisor: {len(physics_plan.symbolic_equations)} equations, "
                                  f"strategy: {physics_plan.solution_strategy[:80]}")
                            for p in physics_plan.known_pitfalls:
                                print(f"     ⚠️  {p}")
                        else:
                            print("  ⚠️  Supervisor returned no plan — proceeding without")
                    except Exception as sv_err:
                        print(f"  ⚠️  Supervisor error: {sv_err}")

                # ── 3. Generate complete, directly-executable Python code ──
                if not HAS_GENERATOR:
                    print("⚠️  Generator not available")
                    self.results[problem['title']] = {"status": "no_generator"}
                    continue

                print("\n  Generating complete solution script...")
                equation = await EquationGenerator.generate(
                    problem=problem['question'],
                    given_variables=given_values,        # floats, not strings
                    unknown_variables=(
                        self.classification.unknown_variables
                        if self.classification else []
                    ),
                    equations_to_use=(
                        self.classification.equations_needed
                        if self.classification else []
                    ),
                    llm_query_func=self._llm_query_coder,  # qwen2.5:14b for code
                    context=search_context,
                    variable_schema=(
                        self.classification.variable_schema
                        if self.classification else None
                    ),
                    physics_plan=physics_plan,
                )

                if not equation.is_valid_syntax:
                    print(f"  ⚠️  Syntax error — retrying with minimal context...")
                    equation = await EquationGenerator.generate(
                        problem=problem['question'],
                        given_variables=given_values,
                        unknown_variables=(
                            self.classification.unknown_variables
                            if self.classification else []
                        ),
                        equations_to_use=(
                            self.classification.equations_needed
                            if self.classification else []
                        ),
                        llm_query_func=self._llm_query_coder,
                        context="",   # no search context — self-contained retry
                        variable_schema=(
                            self.classification.variable_schema
                            if self.classification else None
                        ),
                        physics_plan=physics_plan,
                    )
                    if not equation.is_valid_syntax:
                        print(f"❌ Invalid syntax after retry: {equation.error}")
                        self.results[problem['title']] = {"status": "syntax_error",
                                                          "error": equation.error}
                        continue
                    print(f"  ✅ Retry succeeded ({len(equation.python_code)} chars)")

                print(f"✓ Solution script generated ({len(equation.python_code)} chars, valid syntax)")

                # ── 4. Check for unfilled placeholders (should be none) ────
                if not HAS_EXECUTOR:
                    print("⚠️  Executor not available")
                    self.results[problem['title']] = {"status": "no_executor"}
                    continue

                is_solvable, reason = EquationValidator.check_solvable(
                    equation.python_code,
                    given_values,
                    equation.variables,
                )

                if not is_solvable:
                    print(f"⚠️  Solvability check: {reason}")
                    # Do NOT abort — the check is now advisory; attempt execution anyway
                    print("   Attempting execution despite warning...")

                # ── 5. Execute the script (with up to 2 self-correction retries) ──
                print("  Executing solution script...")
                execution = await EquationExecutor.execute(
                    equation.python_code,
                    given_values,
                    timeout=90,
                )

                # ── 5b. Self-correction loop (up to 2 retries) ────────────
                current_code = equation.python_code
                for retry_num in range(1, 3):
                    if execution.success:
                        break
                    err_msg = execution.error or "unknown error"
                    print(f"⚠️  Execution failed: {err_msg[:200]}")
                    print(f"  Attempting self-correction (retry {retry_num})...")

                    fix_prompt = f"""The Python script below failed to execute. Fix ONLY the error — do not change the physics logic.

ORIGINAL PROBLEM:
{problem['question'][:400]}

FAILING SCRIPT:
```python
{current_code[:3000]}
```

ERROR:
{err_msg[:500]}

Return the corrected complete Python script in ```python``` fences. No prose."""

                    fixed_response = await self._llm_query_coder(fix_prompt)
                    fixed_match = re.search(r'```python\n(.*?)\n```', fixed_response, re.DOTALL)
                    if not fixed_match:
                        fixed_match = re.search(r'```\n(.*?)\n```', fixed_response, re.DOTALL)

                    if fixed_match and EquationGenerator.validate_syntax(fixed_match.group(1)):
                        current_code = fixed_match.group(1).strip()
                        print(f"  Running corrected script (retry {retry_num})...")
                        execution = await EquationExecutor.execute(
                            current_code, given_values, timeout=90
                        )
                        if execution.success:
                            equation.python_code = current_code
                            print(f"  ✅ Self-correction succeeded on retry {retry_num}.")
                        else:
                            print(f"  ❌ Retry {retry_num} failed: {(execution.error or '')[:150]}")
                    else:
                        print(f"  ⚠️  Retry {retry_num} returned invalid Python — stopping retries.")
                        break

                if execution.success:
                    print(f"\n✅ Execution successful!")
                    print("\nOutput:")
                    for line in execution.output.strip().split("\n")[-40:]:
                        print(f"   {line}")
                    if execution.computed_values:
                        print(f"\n📊 Extracted key-value pairs:")
                        for k, v in execution.computed_values.items():
                            print(f"   {k} = {v:.6g}")

                    # ── Phase 3-4b: Multi-layer verification ──────────────
                    verif_notes = []
                    verif = None   # will be set by independent-verification sub-block

                    # a) Expanded physics bounds
                    if HAS_EXECUTOR and execution.computed_values:
                        bounds_ok, bounds_violations = EquationValidator.validate_results(
                            execution.computed_values, problem['question'])
                        if bounds_violations:
                            print(f"\n⚠️  Physics bounds check:")
                            for v in bounds_violations:
                                print(f"   ⚠️  Bounds: {v}")
                                verif_notes.append(f"BOUNDS: {v}")
                        else:
                            print(f"  ✅ Physics bounds: all clear")

                    # b) Unit parsing from RESULT: lines
                    with_units = {}
                    if _HAS_FORMAL_CALC:
                        with_units = FormalCalculator.parse_output_units(execution.output)
                        if with_units:
                            print(f"\n📐 RESULT: lines parsed ({len(with_units)} vars with units):")
                            print(FormalCalculator.summarize(with_units))
                        unit_issues = FormalCalculator.check_dimensional_consistency(with_units)
                        for issue in unit_issues:
                            print(f"  ⚠️  Units: {issue}")
                            verif_notes.append(f"UNITS: {issue}")

                    # c) Independent second computation
                    if execution.computed_values and _HAS_MATH_VERIFIER:
                        print(f"\n🔁 Running independent verification...")
                        try:
                            verif = await MathVerifier.verify_independent(
                                problem['question'], equation.python_code,
                                execution.computed_values, self._llm_query_coder)
                            if verif.overall_agreement:
                                print(f"  ✅ Independent verify: PASS ({verif.note})")
                            else:
                                print(f"  ⚠️  Independent verify: DISCREPANCIES ({verif.note})")
                                for d in verif.discrepancies:
                                    print(f"     {d}")
                                verif_notes.extend(verif.discrepancies)
                        except Exception as ve:
                            print(f"  ⚠️  Independent verify error: {ve}")

                    # d) Numeric consensus vs search facts
                    if _HAS_MATH_VERIFIER and execution.computed_values:
                        try:
                            numeric_check = await MathVerifier.cross_check_with_search(
                                execution.computed_values,
                                self._build_math_context(),
                                self._llm_query)
                            for w in numeric_check.get("warnings", []):
                                print(f"  ⚠️  Fact-check: {w}")
                                verif_notes.append(f"FACT: {w}")
                            if not numeric_check.get("warnings"):
                                print(f"  ✅ Fact-check: {numeric_check.get('notes', 'no issues')}")
                        except Exception as fe:
                            print(f"  ⚠️  Fact-check error: {fe}")

                    # ── Phase 3-4c: Physics-correction retry (if verification disagreed) ──
                    if (_HAS_MATH_VERIFIER
                            and execution.computed_values
                            and verif is not None
                            and not verif.overall_agreement
                            and verif.discrepancies):
                        print(f"\n🔧 Discrepancy detected — attempting physics-correction...")
                        try:
                            reconcile = await MathVerifier.debug_reconcile(
                                problem['question'],
                                equation.python_code,
                                verif.secondary_code,
                                verif.discrepancies,
                                self._llm_query_coder,
                            )
                            if reconcile["success"] and reconcile["values"]:
                                print(f"  ✅ Reconciliation succeeded: {reconcile['diagnosis'][:120]}")
                                # Replace execution values with reconciled values
                                execution.computed_values.update(reconcile["values"])
                                verif_notes.append(f"RECONCILED: {reconcile['diagnosis'][:100]}")
                            else:
                                print(f"  ⚠️  Reconciliation failed — keeping primary result (flagged unverified)")
                                verif_notes.append("UNRESOLVED_DISCREPANCY")
                        except Exception as re_err:
                            print(f"  ⚠️  Reconcile error: {re_err}")

                    self.results[problem['title']] = {
                        "status":             "solved",
                        "values":             execution.computed_values,
                        "values_with_units":  with_units,
                        "output":             execution.output,
                        "verified":           len(verif_notes) == 0,
                        "verification_notes": verif_notes,
                    }
                else:
                    print(f"❌ Execution error (after all retries): {(execution.error or '')[:200]}")
                    self.results[problem['title']] = {
                        "status": "exec_error",
                        "error":  execution.error,
                        "code":   equation.python_code,
                    }

            except Exception as e:
                print(f"❌ Error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
                self.results[problem['title']] = {"status": "error", "message": str(e)}
        
        self.phase_times['math'] = (datetime.now() - phase_start).total_seconds()
    
    async def _phase_5_summarize(self, question: str) -> str:
        """PHASE 5: Summarize"""
        print("\n" + "="*70)
        print("PHASE 5: SUMMARY")
        print("="*70)
        
        phase_start = datetime.now()
        
        try:
            # For non-math questions (THEORETICAL or UNKNOWN), use consensus + WriterAgent
            if self.question_type not in [QuestionType.MATHEMATICAL, QuestionType.HYBRID]:
                print("\n📚 Answering theoretical question from research...")

                # 1. Run consensus validation on gathered facts
                if HAS_CONSENSUS:
                    try:
                        consensus = ConsensusAgent(self.memory)
                        await consensus.validate_all_facts()
                    except Exception as e:
                        print(f"⚠️  Consensus error: {e}")

                # 2. Use WriterAgent's research answer method
                search_facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)
                if HAS_WRITER and search_facts:
                    try:
                        writer = WriterAgent(self.memory)
                        answer = await writer.write_research_answer(question)
                        self.phase_times['summary'] = (datetime.now() - phase_start).total_seconds()
                        return answer
                    except Exception as e:
                        print(f"⚠️  WriterAgent error: {e}")

                # Fallback: structured prompt with raw_content
                if self.raw_content:
                    content_block = "\n\n---\n\n".join(self.raw_content[:5])
                    prompt = f"QUESTION: {question}\n\nRESEARCH:\n{content_block}\n\nWrite a comprehensive answer with key findings and source attribution."
                else:
                    prompt = question

                answer = await self._llm_query(prompt)
                self.phase_times['summary'] = (datetime.now() - phase_start).total_seconds()
                return answer
            
            # ── Build a rich block from every math result ─────────────
            results_sections = []
            any_solved = False
            full_outputs = []   # raw execution stdout for solved problems

            for title, result in self.results.items():
                status = result.get("status", "unknown")
                section = [f"=== {title} ===", f"Status: {status}"]

                if status == "solved":
                    any_solved = True
                    output = result.get("output", "").strip()
                    if output:
                        full_outputs.append(output)
                        section.append("\nCOMPUTED OUTPUT:\n" + output)
                    values = result.get("values", {})
                    if values:
                        section.append("\nEXTRACTED KEY-VALUE PAIRS:")
                        for k, v in values.items():
                            section.append(f"  {k} = {v:.6g}")

                elif status == "exec_error":
                    section.append(f"\nExecution error: {result.get('error', '')}")
                    # Still include generated code so writer can reference equations
                    code = result.get("code", "")
                    if code:
                        section.append(f"\nGenerated (but failed) script:\n{code[:1500]}")

                elif status in ("syntax_error", "no_generator", "no_executor"):
                    section.append(f"\nPipeline error: {result.get('error', status)}")

                results_sections.append("\n".join(section))

            results_text = "\n\n".join(results_sections) if results_sections \
                else "(No math results — computation was not reached)"

            if any_solved:
                # ── Use qwen2.5:14b to write a precise technical answer ────
                # from the actual computed output — NOT from search facts.
                combined_output = "\n\n---\n\n".join(full_outputs)
                print(f"\n  Writing answer from {len(full_outputs)} computed result(s)...")

                prompt = f"""You are a world-class technical writer and physicist.

A Python computation script was executed to solve the following engineering problem.
Present the solution clearly and precisely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM:
{question}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PYTHON EXECUTION OUTPUT (the actual computed values):
{combined_output[:5000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Write a structured technical answer that:
1. Lists every computed numerical result with its value, unit, and equation used
2. Explains the physical meaning of each result in one sentence
3. States every engineering assumption made (Isp, structural fraction, etc.)
4. Notes which sub-problems could not be solved numerically and why

CRITICAL: Use the EXACT numbers from the execution output above.
Do NOT substitute vague estimates like "approximately" or "around".
If the script printed "delta_v = 9450.23 m/s", write "9450.23 m/s", not "~9.5 km/s"."""

                answer = await self._llm_query_coder(
                    prompt,
                    system_prompt="You are a technical writer. Present computed engineering results precisely. Use exact numbers from the computation output."
                )

            else:
                # ── Math ran but nothing solved — fall back to research ────
                print("\n  Math phase produced no solved results — falling back to research answer.")

                # Gather search facts for context
                search_facts = self.memory.get_facts(fact_type=FactType.SEARCH_RESULT)
                facts_text = "\n".join(
                    f"• {f.content[:200]}" for f in search_facts[:30]
                )

                prompt = f"""You are a technical writer. Answer the following engineering question using the research gathered.

PROBLEM:
{question}

MATH PIPELINE STATUS:
{results_text}

RESEARCH FINDINGS:
{facts_text}

Provide the best possible answer from the research. Where the math pipeline failed,
explain what equations and approach would be needed and state typical values from
the literature. Be explicit about what is a computed result vs. a literature value."""

                if HAS_WRITER:
                    try:
                        writer = WriterAgent(self.memory)
                        answer = await writer.write_research_answer(question)
                    except Exception as e:
                        print(f"⚠️  WriterAgent error: {e}")
                        answer = await self._llm_query_coder(prompt)
                else:
                    answer = await self._llm_query_coder(prompt)

            self.phase_times['summary'] = (datetime.now() - phase_start).total_seconds()
            return answer
        
        except Exception as e:
            print(f"⚠️  Summary error: {e}")
            return f"Problem processed but summary generation failed: {e}"
    
    async def _save_markdown_report(self, question: str) -> None:
        """Generate and save a detailed Markdown research report to /tmp/."""
        if not HAS_WRITER:
            print("⚠️  WriterAgent not available — skipping Markdown report")
            return
        try:
            writer = WriterAgent(self.memory)
            md_content = await writer.write_deep_markdown(question)

            slug = re.sub(r'[^a-z0-9]+', '_', question.lower())[:40].strip('_')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.markdown_path = f"/tmp/swarm2_{timestamp}_{slug}.md"

            os.makedirs(os.path.dirname(self.markdown_path), exist_ok=True)
            with open(self.markdown_path, 'w', encoding='utf-8') as fh:
                fh.write(md_content)

            print(f"\n📄 Markdown report saved: {self.markdown_path}")
        except Exception as e:
            print(f"⚠️  Markdown report generation failed: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def _build_math_context(self, max_chars: int = 3000, for_code_gen: bool = False) -> str:
        """
        Collect the most relevant facts from search memory to pass as
        context to the equation generator.  Prioritises VALIDATED_FACT
        and COMPUTED_RESULT entries; falls back to SEARCH_RESULT.
        Returns a plain-text string of bullet points.

        When for_code_gen=True, caps at 1200 chars and skips raw SEARCH_RESULT
        entries (generic background text adds noise for code generation).  If
        fewer than 3 high-quality facts exist it falls back to SEARCH_RESULT
        entries that contain equation-relevant keywords.
        """
        try:
            facts = self.memory.get_facts()   # returns list of all Fact objects
            # Sort by type priority
            priority = {"VALIDATED_FACT": 0, "COMPUTED_RESULT": 1,
                        "SEARCH_RESULT": 2, "SEARCH": 3}
            facts = sorted(
                facts,
                key=lambda f: priority.get(str(f.fact_type).split(".")[-1], 9)
            )

            if for_code_gen:
                max_chars = 1200
                # Only include high-quality facts for code gen
                high_quality_types = {"VALIDATED_FACT", "COMPUTED_RESULT"}
                hq_facts = [
                    f for f in facts
                    if str(f.fact_type).split(".")[-1] in high_quality_types
                ]
                # If fewer than 3 HQ facts, fall back to equation-relevant search results
                if len(hq_facts) < 3:
                    eq_keywords = ('=', 'equation', 'formula', 'law')
                    hq_facts = hq_facts + [
                        f for f in facts
                        if str(f.fact_type).split(".")[-1] not in high_quality_types
                        and any(kw in (f.content if isinstance(f.content, str)
                                       else str(f.content)).lower()
                                for kw in eq_keywords)
                    ]
                facts = hq_facts

            lines = []
            total = 0
            for f in facts:
                content = f.content if isinstance(f.content, str) else str(f.content)
                line = f"• {content}"
                if total + len(line) > max_chars:
                    break
                lines.append(line)
                total += len(line)
            return "\n".join(lines) if lines else ""
        except Exception:
            return ""

    def _extract_values_simple(self, text: str) -> Dict[str, float]:
        """Simple fallback value extraction"""
        values = {}
        
        # "5000 kg", "150 m/s", "7000 km"
        pattern = r'(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Z/²³]*)'
        
        for match in re.finditer(pattern, text):
            try:
                val = float(match.group(1))
                unit = match.group(2).strip() or "value"
                key = f"{unit}_{len(values)}" if unit != "value" else f"value_{len(values)}"
                values[key] = val
            except:
                pass
        
        return values
    
    async def _llm_query(self, prompt: str, system_prompt: str = "") -> str:
        """Query LLM (phi4:14b — reasoning/research)"""
        try:
            agent = BaseAgent(
                agent_id="temp",
                agent_type=AgentType.WORKER,
                model_name="phi4:14b",
                system_prompt=system_prompt or "You are a helpful assistant."
            )
            return await agent.query_llm(prompt, stream=False)
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return ""

    async def _llm_query_coder(self, prompt: str, system_prompt: str = "") -> str:
        """Query LLM (qwen2.5:14b — code generation & technical writing)"""
        try:
            agent = BaseAgent(
                agent_id="coder",
                agent_type=AgentType.WORKER,
                model_name="qwen2.5:14b",
                system_prompt=system_prompt or (
                    "You are an expert Python programmer and physicist. "
                    "Write complete, correct, directly executable code. "
                    "Never use placeholder syntax like {variable}."
                )
            )
            return await agent.query_llm(prompt, stream=False)
        except Exception as e:
            print(f"⚠️  Coder LLM error: {e}")
            return ""
    
    def _print_summary(self) -> None:
        """Print summary"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("📊 EXECUTION SUMMARY")
        print("="*70)
        
        print(f"\nTotal Time: {total_time:.1f}s")
        print(f"Type: {self.question_type.value if self.question_type else 'Unknown'}")
        print(f"Math: {'✅ YES' if self.is_math_question else '❌ NO'}")
        
        solved = sum(1 for r in self.results.values() if r.get('status') == 'solved')
        if self.results:
            print(f"Sub-problems: {len(self.results)} (Solved: {solved})")
        
        print("\nTiming:")
        for phase, duration in sorted(self.phase_times.items()):
            pct = (duration / total_time * 100) if total_time > 0 else 0
            print(f"   {phase:20s}: {duration:6.1f}s ({pct:5.1f}%)")
    
    def save_session(self, filepath: str) -> None:
        """Save session"""
        try:
            data = {
                'question': self.memory.original_question[:100],
                'type': self.question_type.value if self.question_type else 'unknown',
                'results': self.results,
                'times': self.phase_times
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"\n💾 Session saved")
        except Exception as e:
            print(f"⚠️  Save failed: {e}")
