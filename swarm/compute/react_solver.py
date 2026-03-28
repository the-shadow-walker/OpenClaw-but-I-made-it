"""
ReAct Solver — per-sub-problem reasoning agent (qwq:32b)

Runs a Reason→Act→Observe loop to solve a single SubProblem.
Tools are invoked via text-parsed markers (no native function-calling needed —
works with any Ollama model including qwq:32b and deepseek-r1:32b).

Tool format the model must follow:
──────────────────────────────────────────────────────────────────
  THOUGHT: <reasoning>
  ACTION: run_code | search | rag
  INPUT:
  ```python
  <script>
  ```
  END_INPUT

  OR:

  THOUGHT: <final reasoning>
  FINAL_ANSWER:
  STATUS: solved | failed
  RESULT: <var> = <value> <unit>
  VERIFICATION: <residual/check info>
  CODE: <final working script, trimmed to 3000 chars>
  END_ANSWER
──────────────────────────────────────────────────────────────────
"""

import sys
import os
import re
import json
import asyncio
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
try:
    import _paths
except ImportError:
    pass

# ── Optional imports (graceful degradation) ──────────────────────────────────
try:
    from equation_validator import EquationExecutor
    _HAS_EXECUTOR = True
except ImportError:
    _HAS_EXECUTOR = False
    print("⚠️  react_solver: EquationExecutor not available")

try:
    from flexible_search_agent import FlexibleSearchAgent
    _HAS_SEARCH = True
except ImportError:
    _HAS_SEARCH = False
    print("⚠️  react_solver: FlexibleSearchAgent not available")

try:
    from rag_tool import rag_search
    _HAS_RAG = True
except ImportError:
    _HAS_RAG = False
    print("⚠️  react_solver: rag_tool not available")

try:
    from planner_v2 import SubProblem, SolvePlan
    _HAS_PLANNER_TYPES = True
except ImportError:
    _HAS_PLANNER_TYPES = False
    SubProblem = Any
    SolvePlan = Any


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    sub_problem_id: str
    status: str                        # "solved" | "failed" | "timeout"
    results: Dict[str, float] = field(default_factory=dict)
    results_with_units: Dict[str, Dict] = field(default_factory=dict)
    final_code: str = ""
    verification_note: str = ""
    turn_count: int = 0
    raw_log: str = ""                  # full ReAct transcript (archived)


# ── System prompt template ────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert scientific problem solver working inside a ReAct loop.
You MUST follow the exact tool format below — no deviations.

═══════════════════════════════════════════
SUB-PROBLEM: {sp_id}
{sp_description}

DOMAIN: {sp_domain}
APPROACH: {sp_approach}

INPUTS (already known):
{sp_inputs}

EXPECTED OUTPUTS:
{sp_outputs}

COORDINATE SYSTEM: {coord_system}

PLAN NOTES: {plan_notes}
═══════════════════════════════════════════

{research_context_block}

═══════════════════════════════════════════
TOOL FORMAT (copy EXACTLY — every marker on its own line):

To execute code:
  THOUGHT: <your reasoning>
  ACTION: run_code
  INPUT:
  ```python
  <complete self-contained Python script>
  ```
  END_INPUT

To search the web:
  THOUGHT: <your reasoning>
  ACTION: search
  INPUT: <one specific query string>
  END_INPUT

To query the physics reference database:
  THOUGHT: <your reasoning>
  ACTION: rag
  INPUT: <one specific query string>
  END_INPUT

When you have the final answer:
  THOUGHT: <final reasoning>
  FINAL_ANSWER:
  STATUS: solved
  RESULT: var_name = numeric_value unit
  RESULT: another_var = numeric_value unit
  VERIFICATION: <brief check, e.g. residual < 1e-6 or cross-check value>
  CODE: <the final working Python script, max 3000 chars>
  END_ANSWER

If you cannot solve it after trying:
  THOUGHT: <why it failed>
  FINAL_ANSWER:
  STATUS: failed
  VERIFICATION: <what went wrong>
  CODE:
  END_ANSWER

═══════════════════════════════════════════
RULES:
1. Each RESULT line: one variable, one numeric value, one unit (no text).
2. Code must be complete and self-contained (imports + values inlined).
3. Print each computed result as: print(f"RESULT: var_name = {{value:.6g}} unit")
4. Never skip the END_INPUT or END_ANSWER marker.
5. Use SymPy or scipy for solving; numpy for arrays.
6. Verify your answer numerically before declaring STATUS: solved.
7. Think step by step inside THOUGHT blocks.
8. CRITICAL: After your <think> block, you MUST output EITHER a valid
   ACTION block OR a FINAL_ANSWER block — NOTHING ELSE. No prose summary,
   no markdown, just the structured block. The parser only reads those markers.
"""


# ── ReactSolver ───────────────────────────────────────────────────────────────

class ReactSolver:
    MAX_TURNS = 15
    MODEL = "qwq:32b"
    OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_TIMEOUT = 1800  # seconds — streaming keep-alive; actual cap is hard timeout below

    # ── Thinking controls (experiment by changing these) ──────────────────
    # NUM_PREDICT: max tokens for the ENTIRE response (think block + answer).
    #   4096 → ~19 min/turn on this hardware (too slow)
    #   2048 → ~8-10 min/turn  (default — halves think time)
    #   1024 → ~4-5 min/turn   (may truncate complex reasoning)
    #    512 → ~2 min/turn     (minimal thinking; good for simple steps)
    NUM_PREDICT: int = 2048

    # THINKING_ENABLED: set False to pass think=False to Ollama.
    #   True  → qwq reasons fully before answering (slower, more accurate)
    #   False → qwq skips <think> block entirely (fast, less accurate)
    #   Tip: try False first for simple arithmetic SPs, True for derivations.
    THINKING_ENABLED: bool = True

    def __init__(
        self,
        sub_problem: "SubProblem",
        plan: "SolvePlan",
        research_context: str = "",
        searxng_url: Optional[str] = None,
    ):
        self.sub_problem = sub_problem
        self.plan = plan
        self.research_context = research_context
        self.searxng_url = searxng_url or os.getenv("SEARXNG_URL", "http://localhost:8080")

        # Build conversation history: system + alternating user/assistant
        self._system = self._build_system_prompt()
        self._history: List[Dict[str, str]] = []   # {"role": ..., "content": ...}
        self._turn = 0
        self._log_parts: List[str] = []

    # ── Public ────────────────────────────────────────────────────────────────

    async def solve(self) -> SolverResult:
        sp = self.sub_problem
        print(f"\n{'─'*60}")
        print(f"🤖 ReactSolver: {sp.id} — {sp.description[:60]}")
        print(f"   Model: {self.MODEL}  |  Max turns: {self.MAX_TURNS}")
        print(f"{'─'*60}")

        # Seed with the sub-problem statement
        seed_msg = (
            f"Solve {sp.id}: {sp.description}\n"
            f"Domain: {sp.domain}\n"
            f"Given: {json.dumps(sp.inputs)}\n"
            f"Expected outputs: {sp.expected_outputs}\n"
            f"Begin your ReAct reasoning now."
        )
        self._history.append({"role": "user", "content": seed_msg})
        self._log(f"[USER SEED]\n{seed_msg}")

        t0 = time.time()

        for turn in range(1, self.MAX_TURNS + 1):
            self._turn = turn
            print(f"  Turn {turn}/{self.MAX_TURNS}", end=" ", flush=True)

            # Check timeout — 60 min hard cap per sub-problem
            # (qwq:32b takes ~19 min per turn on this hardware)
            if time.time() - t0 > 3600:
                print("⏱️  TIMEOUT")
                return SolverResult(
                    sub_problem_id=sp.id,
                    status="timeout",
                    turn_count=turn,
                    raw_log="\n".join(self._log_parts),
                    verification_note="Hard timeout (3600s) reached",
                )

            # Query the model
            response = await self._llm_call()
            if not response:
                print("❌ empty response")
                continue

            self._log(f"\n[TURN {turn} — ASSISTANT]\n{response}")
            self._history.append({"role": "assistant", "content": response})

            # Search the FULL response first (qwq sometimes puts FINAL_ANSWER
            # inside <think> blocks), then fall back to the stripped version.
            clean = self._strip_thinking(response)
            search_text = response if (
                "FINAL_ANSWER:" in response or "ACTION:" in response
            ) else clean
            print(f"({len(clean)} chars post-think, {len(response)} total)")

            # Check for FINAL_ANSWER in either surface
            if "FINAL_ANSWER:" in search_text:
                result = self._parse_final_answer(search_text, turn)
                # If no RESULT: lines found, try extracting from think block
                if not result.results:
                    result = self._extract_from_think(response, result, turn)
                print(f"  → {result.status.upper()} | "
                      f"{len(result.results)} result(s) | "
                      f"{turn} turns | {time.time()-t0:.0f}s")
                return result

            # Parse and dispatch tool
            parsed = self._parse_action(search_text)
            if parsed is None:
                # Last resort: scan the think block for any RESULT: lines
                fallback = self._try_extract_results_from_think(response)
                if fallback:
                    print(f"  → SOLVED (think-block fallback) | "
                          f"{len(fallback)} result(s) | {turn} turns")
                    return SolverResult(
                        sub_problem_id=sp.id,
                        status="solved",
                        results={v: d["value"] for v, d in fallback.items()},
                        results_with_units=fallback,
                        turn_count=turn,
                        raw_log="\n".join(self._log_parts),
                        verification_note="Extracted from think block",
                    )
                obs = (
                    "Your response did not contain a valid ACTION or FINAL_ANSWER. "
                    "You MUST end with one of:\n"
                    "  ACTION: run_code / search / rag  (followed by INPUT: ... END_INPUT)\n"
                    "  FINAL_ANSWER: ... END_ANSWER\n"
                    "Do not continue reasoning — pick an action NOW."
                )
            else:
                action, inp = parsed
                obs = await self._run_tool(action, inp)

            self._log(f"\n[OBSERVATION turn {turn}]\n{obs}")
            self._history.append({"role": "user", "content": f"OBSERVATION:\n{obs}"})

        # Exhausted turns
        print(f"  → FAILED (turn limit)")
        return SolverResult(
            sub_problem_id=sp.id,
            status="failed",
            turn_count=self.MAX_TURNS,
            raw_log="\n".join(self._log_parts),
            verification_note=f"Did not converge in {self.MAX_TURNS} turns",
        )

    # ── LLM call ─────────────────────────────────────────────────────────────

    async def _llm_call(self) -> str:
        """Call qwq:32b via Ollama /api/chat (streaming to avoid HTTP timeout)."""
        messages = [{"role": "system", "content": self._system}] + self._history
        payload = {
            "model": self.MODEL,
            "messages": messages,
            "stream": True,   # streaming keeps the connection alive for slow models
            "think": self.THINKING_ENABLED,   # False = skip <think> block entirely
            "options": {
                "temperature": 0.6,
                "num_predict": self.NUM_PREDICT,
            },
        }
        try:
            loop = asyncio.get_event_loop()

            def _stream_call() -> str:
                resp = requests.post(
                    f"{self.OLLAMA_URL}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=self.LLM_TIMEOUT,
                )
                resp.raise_for_status()
                accumulated = []
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        delta = chunk.get("message", {}).get("content", "")
                        if delta:
                            accumulated.append(delta)
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                return "".join(accumulated)

            content = await loop.run_in_executor(None, _stream_call)
            return content
        except Exception as e:
            print(f"\n⚠️  LLM error: {e}")
            return ""

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    async def _run_tool(self, action: str, input_text: str) -> str:
        action = action.strip().lower()

        if action == "run_code":
            return await self._tool_run_code(input_text)
        elif action == "search":
            return await self._tool_search(input_text.strip())
        elif action == "rag":
            return await self._tool_rag(input_text.strip())
        else:
            return f"Unknown action '{action}'. Use run_code, search, or rag."

    async def _tool_run_code(self, code_text: str) -> str:
        """Execute a Python code block and return stdout / error."""
        # Extract code from markdown fences if present
        m = re.search(r"```(?:python)?\s*\n(.*?)\n```", code_text, re.DOTALL)
        code = m.group(1) if m else code_text.strip()

        if not _HAS_EXECUTOR:
            return "ERROR: EquationExecutor not available — cannot run code."

        print("    [run_code]", end=" ", flush=True)
        try:
            result = await EquationExecutor.execute(code, given_values={}, timeout=90)
            if result.success:
                print(f"OK ({len(result.output)} chars)")
                return result.output[:4000] or "(no output)"
            else:
                print(f"FAILED")
                return f"EXECUTION ERROR:\n{result.error}\n\nOUTPUT:\n{result.output[:2000]}"
        except Exception as e:
            print(f"EXC: {e}")
            return f"EXCEPTION running code: {e}"

    async def _tool_search(self, query: str) -> str:
        """Run a web search and return snippet text."""
        if not _HAS_SEARCH:
            return "Search not available."

        print(f"    [search] {query[:60]}", end=" ", flush=True)
        try:
            agent = FlexibleSearchAgent(
                searxng_url=self.searxng_url,
                timeout=30,
                max_results=4,
            )
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: agent.search_and_fetch(query, num_sources=3, fetch_content=False)
            )
            if not results:
                print("0 results")
                return "No results found."
            chunks = []
            for r in results[:4]:
                chunks.append(f"[{r.source}] {r.title}\n{r.snippet}")
            print(f"{len(chunks)} results")
            return "\n---\n".join(chunks)
        except Exception as e:
            print(f"ERR: {e}")
            return f"Search error: {e}"

    async def _tool_rag(self, query: str) -> str:
        """Query the physics RAG database."""
        if not _HAS_RAG:
            return "RAG not available."

        print(f"    [rag] {query[:60]}", end=" ", flush=True)
        domain = self.sub_problem.domain if self.sub_problem.domain in ("physics",) else "physics"
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: rag_search(query, domain=domain, n=4))
        if result:
            print(f"OK ({len(result)} chars)")
            return result[:3000]
        print("no results")
        return "No RAG results found."

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>…</think> blocks produced by qwq/deepseek reasoning models."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _build_system_prompt(self) -> str:
        sp = self.sub_problem
        inputs_str = "\n".join(f"  {k} = {v}" for k, v in sp.inputs.items()) or "  (none)"
        outputs_str = "\n".join(
            f"  {o.get('name','?')} [{o.get('unit','?')}]"
            for o in sp.expected_outputs
        ) or "  (derive as needed)"

        if self.research_context:
            ctx_block = (
                "═══════════════════════════════════════════\n"
                "PRE-FETCHED RESEARCH CONTEXT:\n"
                + self.research_context[:3000]
                + "\n═══════════════════════════════════════════"
            )
        else:
            ctx_block = "(no pre-fetched research — use search/rag tools as needed)"

        return _SYSTEM_PROMPT_TEMPLATE.format(
            sp_id=sp.id,
            sp_description=sp.description,
            sp_domain=sp.domain,
            sp_approach=sp.approach or "derive from first principles",
            sp_inputs=inputs_str,
            sp_outputs=outputs_str,
            coord_system=self.plan.coordinate_system if self.plan else "N/A",
            plan_notes=self.plan.notes[:500] if self.plan else "",
            research_context_block=ctx_block,
        )

    def _parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract (action, input) from a model response.
        Returns None if no valid ACTION block is found.
        """
        # Match ACTION: <name>\nINPUT:\n...\nEND_INPUT
        m = re.search(
            r"ACTION:\s*(\w+)\s*\nINPUT:\s*\n(.*?)\nEND_INPUT",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # Looser fallback: ACTION: <name> followed by INPUT on same or next line
        m2 = re.search(r"ACTION:\s*(\w+)[^\n]*\nINPUT:\s*(.*?)(?:\nEND_INPUT|$)", text, re.DOTALL)
        if m2:
            return m2.group(1).strip(), m2.group(2).strip()
        return None

    def _parse_final_answer(self, text: str, turn: int) -> SolverResult:
        """Extract STATUS, RESULT lines, VERIFICATION, CODE from a FINAL_ANSWER block."""
        # Find the FINAL_ANSWER block
        fa_match = re.search(r"FINAL_ANSWER:(.*?)(?:END_ANSWER|$)", text, re.DOTALL)
        block = fa_match.group(1) if fa_match else text

        status_m = re.search(r"STATUS:\s*(\w+)", block, re.IGNORECASE)
        status = status_m.group(1).lower() if status_m else "failed"

        # Parse RESULT lines: "RESULT: var_name = value unit"
        results: Dict[str, float] = {}
        results_with_units: Dict[str, Dict] = {}
        for rm in re.finditer(
            r"RESULT:\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([^\n]*)",
            block,
            re.IGNORECASE,
        ):
            var = rm.group(1).strip()
            try:
                val = float(rm.group(2))
            except ValueError:
                continue
            unit = rm.group(3).strip()
            results[var] = val
            results_with_units[var] = {"value": val, "unit": unit}

        verif_m = re.search(r"VERIFICATION:\s*([^\n]+)", block, re.IGNORECASE)
        verification = verif_m.group(1).strip() if verif_m else ""

        # CODE: everything from "CODE:" to END_ANSWER (or end of block)
        code_m = re.search(r"CODE:\s*(.*?)(?:END_ANSWER|$)", block, re.DOTALL)
        code = code_m.group(1).strip()[:3000] if code_m else ""

        return SolverResult(
            sub_problem_id=self.sub_problem.id,
            status=status,
            results=results,
            results_with_units=results_with_units,
            final_code=code,
            verification_note=verification,
            turn_count=turn,
            raw_log="\n".join(self._log_parts),
        )

    def _extract_from_think(self, full_response: str, existing: SolverResult, turn: int) -> SolverResult:
        """
        If FINAL_ANSWER parsing found no RESULT lines, scan inside <think> blocks
        for any "RESULT: var = value unit" lines and merge them in.
        """
        think_results = self._try_extract_results_from_think(full_response)
        if think_results and not existing.results:
            return SolverResult(
                sub_problem_id=existing.sub_problem_id,
                status="solved",
                results={v: d["value"] for v, d in think_results.items()},
                results_with_units=think_results,
                final_code=existing.final_code,
                verification_note=existing.verification_note or "Results from think block",
                turn_count=turn,
                raw_log="\n".join(self._log_parts),
            )
        return existing

    @staticmethod
    def _try_extract_results_from_think(response: str) -> Dict[str, Dict]:
        """
        Scan the full response (including <think> content) for RESULT: lines.
        Returns {var: {"value": float, "unit": str}} or {} if nothing found.
        """
        results = {}
        for rm in re.finditer(
            r"RESULT:\s*([A-Za-z_]\w*)\s*=\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([^\n]*)",
            response,
            re.IGNORECASE,
        ):
            var = rm.group(1).strip()
            try:
                val = float(rm.group(2))
            except ValueError:
                continue
            unit = rm.group(3).strip()
            results[var] = {"value": val, "unit": unit}
        return results

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, text: str) -> None:
        self._log_parts.append(text)
