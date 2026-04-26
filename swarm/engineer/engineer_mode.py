"""
Engineer Mode — Swarm 3

Handles complex multi-variable engineering design problems, e.g.:
  "Design a two-stage rocket to carry 15,000 kg to LEO using LOX/RP-1"

Pipeline:
  E1 — Classify + build dependency graph
  E2 — Extract given values + targeted web hunt for missing parameters
  E3 — Assumption engine (fill gaps from engineering_defaults)
  E4 — Generate complete iterative Python simulation
  E5 — Execute + bounds-check + self-correct (max 2 rounds)
  E6 — Synthesize 6-section Technical Data Sheet

Entry point:
  await run_engineer_mode(problem, searxng_url=..., debug=False, save_markdown=False)
"""

import asyncio
import json
import re
import os
import tempfile
import subprocess
import requests as _requests
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# ─── Project imports ──────────────────────────────────────────────────────────
try:
    from base_agent import BaseAgent
    from core import AgentType
    _HAS_BASE = True
except ImportError:
    _HAS_BASE = False

try:
    from value_extractor import ValueExtractor
    _HAS_EXTRACTOR = True
except ImportError:
    _HAS_EXTRACTOR = False

try:
    from flexible_search_agent import FlexibleSearchAgent
    _HAS_SEARCH = True
except ImportError:
    _HAS_SEARCH = False

try:
    from equation_generator import EquationGenerator
    _HAS_GENERATOR = True
except ImportError:
    _HAS_GENERATOR = False

try:
    from equation_validator import EquationExecutor
    _HAS_EXECUTOR = True
except ImportError:
    _HAS_EXECUTOR = False

try:
    from engineering_defaults import (
        get_defaults_for_domain,
        format_defaults_for_prompt,
        check_physical_bounds,
        BoundsViolation,
        DOMAIN_ALIASES,
    )
    _HAS_DEFAULTS = True
except ImportError:
    _HAS_DEFAULTS = False


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class DesignNode:
    """A single node in the engineering dependency graph."""
    name: str
    description: str
    depends_on: List[str]
    equation: str
    is_computed: bool = False


@dataclass
class AssumptionRecord:
    """Records a design assumption / default used during the analysis."""
    parameter: str
    value: float
    unit: str
    basis: str          # "Given", "Engineering default", "Search result"
    source: str = ""    # URL or citation if from search


class DependencyGraph:
    """Directed acyclic graph of design variables/equations."""

    def __init__(self):
        self.nodes: Dict[str, DesignNode] = {}

    def add_node(self, node: DesignNode) -> None:
        self.nodes[node.name] = node

    def topological_order(self) -> List[str]:
        """Kahn's algorithm — returns nodes in dependency order."""
        in_degree: Dict[str, int] = {n: 0 for n in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep in in_degree:
                    in_degree[node.name] = in_degree[node.name] + 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for candidate in self.nodes.values():
                if n in candidate.depends_on:
                    in_degree[candidate.name] -= 1
                    if in_degree[candidate.name] == 0:
                        queue.append(candidate.name)

        # Append any nodes not reached (cycles / disconnected)
        for n in self.nodes:
            if n not in order:
                order.append(n)
        return order

    def to_prompt_context(self) -> str:
        """Format graph as an ordered list of computation steps for a prompt."""
        order = self.topological_order()
        lines = ["Dependency-ordered computation steps:"]
        for i, name in enumerate(order, 1):
            node = self.nodes[name]
            deps = ", ".join(node.depends_on) if node.depends_on else "none"
            lines.append(
                f"  {i:2d}. {name:<30} depends on: [{deps}]\n"
                f"       equation: {node.equation}\n"
                f"       ({node.description})"
            )
        return "\n".join(lines)


# ─── Fallback graph (rockets) ─────────────────────────────────────────────────

ROCKET_DEFAULT_GRAPH = DependencyGraph()
for _n in [
    DesignNode("payload_mass",    "Payload to orbit (given)",          [],                         "m_PL = given"),
    DesignNode("delta_v_total",   "Total delta-v inc. losses",         ["payload_mass"],            "dV = dV_ideal + dV_gravity + dV_drag + dV_steering"),
    DesignNode("Isp_stage1",      "Stage 1 specific impulse",          [],                         "Isp1 = propellant_choice"),
    DesignNode("Isp_stage2",      "Stage 2 specific impulse",          [],                         "Isp2 = propellant_choice"),
    DesignNode("mass_ratio_s2",   "Stage 2 mass ratio (Tsiolkovsky)",  ["Isp_stage2","delta_v_total"], "MR2 = exp(dV2 / (Isp2*g0))"),
    DesignNode("mass_ratio_s1",   "Stage 1 mass ratio (Tsiolkovsky)",  ["Isp_stage1","delta_v_total"], "MR1 = exp(dV1 / (Isp1*g0))"),
    DesignNode("propellant_mass_s2","Stage 2 propellant mass",         ["mass_ratio_s2","payload_mass"],  "mp2 = m0_s2*(1 - 1/MR2)"),
    DesignNode("structural_mass_s2","Stage 2 structural mass",         ["propellant_mass_s2"],     "ms2 = eps2 * mp2 / (1 - eps2)"),
    DesignNode("m0_s2",           "Stage 2 gross mass",                ["propellant_mass_s2","structural_mass_s2","payload_mass"], "m0_s2 = mp2 + ms2 + m_PL"),
    DesignNode("propellant_mass_s1","Stage 1 propellant mass",         ["mass_ratio_s1","m0_s2"],  "mp1 = m0_s1*(1 - 1/MR1)"),
    DesignNode("structural_mass_s1","Stage 1 structural mass",         ["propellant_mass_s1"],     "ms1 = eps1 * mp1 / (1 - eps1)"),
    DesignNode("GTOW",            "Gross take-off weight",             ["propellant_mass_s1","structural_mass_s1","m0_s2"], "GTOW = mp1 + ms1 + m0_s2"),
    DesignNode("thrust_liftoff",  "Liftoff thrust",                    ["GTOW"],                   "F = TW * GTOW * g0"),
    DesignNode("payload_fraction","Payload fraction",                  ["GTOW","payload_mass"],    "lambda = m_PL / GTOW"),
]:
    ROCKET_DEFAULT_GRAPH.add_node(_n)


# ─── Search query templates ───────────────────────────────────────────────────

SEARCH_QUERY_TEMPLATES: Dict[str, List[str]] = {
    "Isp":                    ["{propellant} rocket engine Isp specific impulse seconds typical value"],
    "structural_fraction":    ["structural mass fraction {stage_type} rocket stage typical expendable"],
    "delta_v":                ["delta-v budget LEO launch gravity drag loss m/s ascent trajectory"],
    "efficiency":             ["typical {component} efficiency percent engineering"],
    "yield_strength":         ["{material} yield strength MPa material properties"],
    "thermal_resistance":     ["{component} thermal resistance junction case typical datasheet"],
    "generic":                ["typical {parameter} value {domain} engineering design"],
}


# ─── Prompt templates ─────────────────────────────────────────────────────────

E1_GRAPH_PROMPT = """You are an expert systems engineer performing a top-down design analysis.

PROBLEM:
{problem}

Your task: decompose this into a dependency graph of engineering design variables.

Respond with ONLY valid JSON (no markdown, no prose):
{{
  "domain": "rocket | motor | structure | thermal | power | fluid | other",
  "problem_summary": "one sentence",
  "dependency_order": ["var1", "var2", ...],
  "nodes": [
    {{
      "name": "short_var_name",
      "description": "what this parameter represents",
      "depends_on": ["var_names_it_needs"],
      "equation": "symbolic equation or formula"
    }},
    ...
  ]
}}

Rules:
- Include 8–15 nodes (neither too coarse nor too fine-grained)
- List them in topological order (inputs first, outputs last)
- Use short_snake_case names
- Write concrete equations (Tsiolkovsky, Ohm, Fourier, etc.)
"""

E2_MISSING_PROMPT = """Given this engineering design problem and dependency graph, identify which parameters
are missing (not given in the problem statement) and must be looked up or assumed.

PROBLEM: {problem}
GIVEN VALUES: {given_values}
DEPENDENCY GRAPH:
{graph_context}

List ONLY the parameters that are missing. For each, suggest the single best search query.
Respond with ONLY valid JSON:
{{
  "missing_parameters": [
    {{"name": "Isp_stage1", "query": "LOX RP-1 rocket engine Isp specific impulse seconds"}},
    ...
  ]
}}
Limit to 6 most important missing parameters.
"""

ENGINEER_GENERATION_PROMPT = """You are an expert aerospace/mechanical engineer and Python programmer.

Write a COMPLETE, directly-executable Python script for the following design problem.
The script must run with `python3 script.py` with ZERO modifications.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN PROBLEM:
{problem}

GIVEN VALUES (inline these exactly):
{given_vars_block}

DEPENDENCY GRAPH (compute in this order):
{graph_context}

ENGINEERING DEFAULTS AVAILABLE (use these for any missing parameters):
{defaults_block}

SEARCH CONTEXT (use numeric values found here when applicable):
{search_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANDATORY RULES:
1. NO {{placeholder}} syntax. Every variable must be a concrete float/int.
2. Imports allowed: math, numpy as np, scipy (scipy.optimize for optimization).
3. For any parameter NOT given: use the engineering default above; add a comment:
     # assumed: <value>  — <rationale>
4. Physics constants block:
     g0   = 9.80665          # m/s² standard gravity
     mu_E = 3.986004418e14   # m³/s² Earth GM
     R_E  = 6.3781e6         # m Earth mean radius
5. Follow the dependency graph order for intermediate computations.
6. For optimization (e.g. staging split): use a sweep loop or scipy.optimize.minimize_scalar.
7. At the END of the script, print a block that starts with the exact line:
     print("TECHNICAL DATA SHEET")
   followed by:
     print(f"  {{var:<35}} {{value:>15.4g}}  {{unit}}")
   for EVERY key result variable (mass, Isp, delta-v, efficiency, etc.).
8. Add assert statements for critical physical constraints:
     assert GTOW > 0, f"GTOW must be positive, got {{GTOW}}"
     assert 0 < eps1 < 0.5, f"Structural fraction out of range: {{eps1}}"
9. Compute EVERY intermediate step and print it with label + unit.
10. Handle edge cases: no division by zero, no log of non-positive.

Output ONLY the Python code inside ```python ... ``` fences.
No prose before or after.
"""

ENGINEER_SYNTHESIS_PROMPT = """You are a senior systems engineer writing a Technical Data Sheet.

DESIGN PROBLEM:
{problem}

ASSUMPTIONS USED:
{assumptions_table}

COMPUTATION OUTPUT (EXACT stdout — use ONLY these numbers, do not invent values):
{execution_output}

Write a Technical Data Sheet in Markdown with EXACTLY these 6 sections,
using the EXACT numbers from the computation output above.
Bold all key numeric values.

## Executive Summary
3–5 bullet points, each with a **bolded number and unit**.

## Design Table
| Parameter | Symbol | Value | Unit | Source |
| --- | --- | --- | --- | --- |
(one row per key result variable from the computation output)

## Governing Equations
Numbered list. For each equation show the symbolic form AND numeric substitution:
  1. Tsiolkovsky: MR = exp(dV / (Isp·g₀)) = exp(**9500** / (**358**·9.807)) = **13.2**

## Assumptions & Defaults
| Parameter | Value | Unit | Basis |
| --- | --- | --- | --- |
(one row per AssumptionRecord)

## Sensitivity Analysis
For the 3 most impactful parameters, estimate ±10% change in input → % change in primary output.
Use the format:
- **Isp +10%** → GTOW −8.5% (qualitative estimate)

## Qualitative Assessment
3–5 sentences on: technical feasibility, key risks, recommended next steps.

IMPORTANT: Use ONLY numbers that appear in the computation output. Do not invent figures.
"""

E6_FALLBACK_PROMPT = """You are a senior systems engineer writing a Technical Data Sheet.

DESIGN PROBLEM:
{problem}

No numerical computation was successfully completed. Write a qualitative Technical Data Sheet
that covers the design approach, key governing equations (symbolic only), typical parameter
ranges from engineering literature, and recommended next steps.

Use the 6-section format:
## Executive Summary
## Design Table (with typical ranges, not computed values)
## Governing Equations
## Assumptions & Defaults
## Sensitivity Analysis (qualitative)
## Qualitative Assessment
"""


# ─── Main Orchestrator ────────────────────────────────────────────────────────

class EngineerModeOrchestrator:
    """
    Six-phase engineering design pipeline.

    Usage:
        emo = EngineerModeOrchestrator(searxng_url=..., debug=True)
        tds = await emo.run("Design a two-stage rocket to put 15,000 kg in LEO")
    """

    def __init__(
        self,
        searxng_url: Optional[str] = None,
        debug: bool = False,
        save_markdown: bool = False,
        sidechain: Optional[Any] = None,
        job_id: str = "",
    ):
        self.searxng_url = searxng_url or os.getenv("SEARXNG_URL", "http://localhost:8080")
        self.debug = debug
        self.save_markdown = save_markdown
        self.sidechain = sidechain     # Swarm 3.15 / Chunk 8: optional JSONL trace
        self.job_id = job_id

        # State set during phases
        self.problem: str = ""
        self.domain: str = "rocket"
        self.dep_graph: DependencyGraph = DependencyGraph()
        self.given_values: Dict[str, float] = {}
        self.assumptions: List[AssumptionRecord] = []
        self.search_context: str = ""
        self.generated_code: str = ""
        self.execution_output: str = ""
        self.execution_success: bool = False
        self.bounds_violations: List = []
        self.tds_markdown: str = ""
        self.markdown_path: Optional[str] = None

        self._phase_times: Dict[str, float] = {}

    # ── Sidechain ────────────────────────────────────────────────────────────

    def _emit_phase(self, phase: str, status: str = "start", **fields: Any) -> None:
        """
        Swarm 3.15 / Chunk 8: write a JSONL event for the active phase.
        No-op when sidechain is None (i.e. not invoked as subagent).
        """
        sc = self.sidechain
        if sc is None:
            return
        try:
            sc.write_event(
                "engineer_phase",
                phase=phase,
                status=status,
                job_id=self.job_id,
                **fields,
            )
        except Exception:
            # Sidechain failures must never break the run
            pass

    # ── LLM helpers ───────────────────────────────────────────────────────────

    async def _llm_query(self, prompt: str, system_prompt: str = "") -> str:
        """Unified-model reasoning / classification / graph building (Swarm 3.15)."""
        if not _HAS_BASE:
            return ""
        try:
            agent = BaseAgent(
                agent_id="eng_phi4",
                agent_type=AgentType.WORKER,
                model_name=os.getenv("SWARM_MODEL_ENGINEER", os.getenv("SWARM_MODEL_DEFAULT", "qwen3-coder:30b")),
                system_prompt=system_prompt or "You are an expert systems engineer.",
            )
            return await agent.query_llm(prompt, stream=False)
        except Exception as e:
            print(f"⚠️  _llm_query error: {e}")
            return ""

    async def _llm_query_coder(self, prompt: str, system_prompt: str = "") -> str:
        """Qwen3-coder:30b — streaming code generation with 1800s timeout."""
        _CODER_MODEL = "Qwen3-coder:30b"
        sys = system_prompt or (
            "You are an expert Python programmer and engineer. "
            "Write complete, correct, directly executable code. "
            "Never use placeholder syntax like {variable}. /no_think"
        )
        full_prompt = f"{sys}\n\n{prompt}"
        payload = {
            "model": _CODER_MODEL,
            "prompt": full_prompt,
            "stream": True,
            "keep_alive": 0,
            "options": {"temperature": 0.1, "num_predict": 4096},
        }
        print(f"🤖 Initialized eng_coder (worker) using {_CODER_MODEL}")
        try:
            resp = await asyncio.to_thread(
                lambda: _requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload, stream=True, timeout=1800,
                )
            )
            resp.raise_for_status()
            result = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        d = json.loads(line)
                        result += d.get("response", "")
                        if d.get("done"):
                            break
                    except Exception:
                        continue
            # Strip <think>…</think> blocks (Qwen3 reasoning output)
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            return result
        except _requests.exceptions.Timeout:
            print(f"      ❌ Timeout after 1800s")
            return ""
        except Exception as e:
            print(f"      ❌ Coder error: {e}")
            return ""

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self, problem: str) -> str:
        """Run the full 6-phase engineering design pipeline. Returns TDS markdown."""
        self.problem = problem
        start = datetime.now()

        print("\n" + "=" * 70)
        print("🔧 ENGINEER MODE — SWARM 3")
        print("=" * 70)
        print(f"Problem: {problem[:80]}...")
        print("=" * 70)

        self._emit_phase("run", "start", problem=problem[:200])

        await self._phase_e1_classify_and_graph()
        await self._phase_e2_extract_and_hunt()
        await self._phase_e3_assumption_engine()
        await self._phase_e4_generate_code()
        await self._phase_e5_execute_and_validate()
        await self._phase_e6_synthesize()

        elapsed = (datetime.now() - start).total_seconds()
        print(f"\n✅ Engineer Mode complete in {elapsed:.1f}s")

        if self.save_markdown and self.tds_markdown:
            await self._save_markdown()

        # Swarm 3.15 / Chunk 8: write agent-bin deliverable for subagent dispatch
        if self.tds_markdown:
            self._write_agent_bin_deliverable()

        self._emit_phase(
            "run", "done",
            elapsed_s=elapsed,
            tds_chars=len(self.tds_markdown),
            deliverable=self.markdown_path or "",
        )

        return self.tds_markdown

    def _write_agent_bin_deliverable(self) -> None:
        """
        Write the TDS to ~/.agent_bin/results/<topic>_engineer_<job_id>.md
        so the /subagent/engineer endpoint can return a path-based deliverable.
        """
        try:
            base = os.path.expanduser(
                os.getenv("AGENT_BIN_RESULTS", "~/.agent_bin/results")
            )
            os.makedirs(base, exist_ok=True)
            slug_src = re.sub(r"[^a-z0-9]+", "_", self.problem[:60].lower()).strip("_") or "engineer"
            jid = self.job_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(base, f"{slug_src}_engineer_{jid}.md")
            with open(path, "w") as f:
                f.write(f"# Engineer Mode Deliverable\n\n")
                f.write(f"**Problem:** {self.problem}\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
                f.write(f"**Domain:** {self.domain}\n\n")
                f.write("---\n\n")
                f.write(self.tds_markdown)
                if self.generated_code:
                    f.write("\n\n---\n\n## Generated Simulation Code\n\n")
                    f.write(f"```python\n{self.generated_code}\n```\n")
            # If the user didn't already populate markdown_path via _save_markdown,
            # surface this path as the canonical deliverable.
            if not self.markdown_path:
                self.markdown_path = path
            print(f"📄 Engineer deliverable: {path}")
        except Exception as e:
            print(f"⚠️  Could not write agent-bin deliverable: {e}")

    # ── Phase E1 ──────────────────────────────────────────────────────────────

    async def _phase_e1_classify_and_graph(self) -> None:
        """Classify domain and build dependency graph via phi4."""
        t0 = datetime.now()
        print("\n── Phase E1: Dependency Graph ──")
        self._emit_phase("E1", "start")

        prompt = E1_GRAPH_PROMPT.format(problem=self.problem)
        response = await self._llm_query(
            prompt,
            system_prompt="You are an expert systems engineer. Respond ONLY with valid JSON.",
        )

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(json_match.group()) if json_match else json.loads(response)

            self.domain = data.get("domain", "rocket")

            nodes_data = data.get("nodes", [])
            if nodes_data:
                for nd in nodes_data:
                    node = DesignNode(
                        name=nd.get("name", "unknown"),
                        description=nd.get("description", ""),
                        depends_on=nd.get("depends_on", []),
                        equation=nd.get("equation", ""),
                    )
                    self.dep_graph.add_node(node)
                print(f"   Domain: {self.domain}")
                print(f"   Graph nodes: {len(self.dep_graph.nodes)}")
                order = self.dep_graph.topological_order()
                print(f"   Order: {' → '.join(order[:6])}{'...' if len(order)>6 else ''}")
            else:
                raise ValueError("No nodes in LLM response")

        except Exception as e:
            print(f"   ⚠️  Graph parse failed ({e}) — using ROCKET_DEFAULT_GRAPH")
            self.dep_graph = ROCKET_DEFAULT_GRAPH
            self.domain = "rocket"

        self._phase_times["E1"] = (datetime.now() - t0).total_seconds()
        self._emit_phase("E1", "done", elapsed_s=self._phase_times["E1"], domain=self.domain, n_nodes=len(self.dep_graph.nodes))

    # ── Phase E2 ──────────────────────────────────────────────────────────────

    async def _phase_e2_extract_and_hunt(self) -> None:
        """Extract given values from problem, then hunt for missing params."""
        t0 = datetime.now()
        print("\n── Phase E2: Value Extraction + Search ──")
        self._emit_phase("E2", "start")

        # Extract explicit values
        if _HAS_EXTRACTOR:
            self.given_values = ValueExtractor.extract_all(self.problem)
            print(f"   Extracted {len(self.given_values)} values from problem statement")
            if self.debug:
                for k, v in self.given_values.items():
                    print(f"     {k} = {v}")

        # Ask LLM which parameters are missing
        graph_ctx = self.dep_graph.to_prompt_context() if self.dep_graph.nodes else "(no graph)"
        given_str = json.dumps(self.given_values, indent=2) if self.given_values else "{}"

        missing_response = await self._llm_query(
            E2_MISSING_PROMPT.format(
                problem=self.problem,
                given_values=given_str,
                graph_context=graph_ctx,
            ),
            system_prompt="You are an expert engineer. Respond ONLY with valid JSON.",
        )

        # Build targeted search queries
        search_queries: List[str] = []
        try:
            json_match = re.search(r'\{.*\}', missing_response, re.DOTALL)
            if json_match:
                miss_data = json.loads(json_match.group())
                for item in miss_data.get("missing_parameters", []):
                    q = item.get("query", "")
                    if q:
                        search_queries.append(q)
        except Exception as e:
            if self.debug:
                print(f"   ⚠️  Missing-params parse failed: {e}")

        # Always add a broad domain fallback query
        if not search_queries:
            search_queries = [
                f"typical {self.domain} design parameters engineering defaults",
                f"{self.problem[:60]} engineering analysis parameters",
            ]

        print(f"   Firing {len(search_queries)} targeted searches…")
        if self.debug:
            for q in search_queries:
                print(f"     → {q}")

        # Parallel searches
        if _HAS_SEARCH and search_queries:
            agent = FlexibleSearchAgent(
                searxng_url=self.searxng_url,
                max_results=3,
            )
            tasks = [
                asyncio.to_thread(agent.search, q)
                for q in search_queries
            ]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            snippets: List[str] = []
            for results in results_list:
                if isinstance(results, Exception):
                    continue
                for r in (results or []):
                    if r.snippet:
                        snippets.append(f"[{r.url}] {r.snippet}")
            self.search_context = "\n".join(snippets[:8])
            print(f"   Collected {min(len(snippets), 8)}/{len(snippets)} search snippets (capped at 8)")
        else:
            self.search_context = "(search unavailable)"

        self._phase_times["E2"] = (datetime.now() - t0).total_seconds()
        self._emit_phase("E2", "done", elapsed_s=self._phase_times["E2"], n_given=len(self.given_values), search_chars=len(self.search_context))

    # ── Phase E3 ──────────────────────────────────────────────────────────────

    async def _phase_e3_assumption_engine(self) -> None:
        """Fill missing parameters from engineering_defaults (or search results)."""
        t0 = datetime.now()
        print("\n── Phase E3: Assumption Engine ──")
        self._emit_phase("E3", "start")

        if not _HAS_DEFAULTS:
            print("   ⚠️  engineering_defaults.py not found; skipping")
            self._phase_times["E3"] = 0.0
            return

        defaults = get_defaults_for_domain(self.domain)
        if not defaults:
            print(f"   No defaults for domain '{self.domain}'")
            self._phase_times["E3"] = 0.0
            return

        # Record the relevant defaults as assumptions
        for key, (value, unit, rationale) in defaults.items():
            # Check if this default was already given or found in search
            if key in self.given_values:
                self.assumptions.append(AssumptionRecord(
                    parameter=key, value=self.given_values[key],
                    unit=unit, basis="Given in problem", source="problem statement",
                ))
                continue

            # See if search context contains a value for this parameter
            search_value = self._extract_from_search_context(key)
            if search_value is not None:
                self.assumptions.append(AssumptionRecord(
                    parameter=key, value=search_value,
                    unit=unit, basis="Search result", source="web search",
                ))
            else:
                self.assumptions.append(AssumptionRecord(
                    parameter=key, value=value,
                    unit=unit, basis="Engineering default", source=rationale,
                ))

        print(f"   Recorded {len(self.assumptions)} assumptions")
        if self.debug:
            for a in self.assumptions[:5]:
                print(f"     {a.parameter} = {a.value} {a.unit}  [{a.basis}]")

        self._phase_times["E3"] = (datetime.now() - t0).total_seconds()
        self._emit_phase("E3", "done", elapsed_s=self._phase_times["E3"], n_assumptions=len(self.assumptions))

    def _extract_from_search_context(self, param_name: str) -> Optional[float]:
        """Very simple numeric extraction for a named parameter from search snippets."""
        if not self.search_context:
            return None
        # Look for "param_name = 311" or "311 s" near param_name text
        pattern = rf'{re.escape(param_name.replace("_"," "))}.*?(\d+(?:\.\d+)?)'
        m = re.search(pattern, self.search_context, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    # ── Phase E4 ──────────────────────────────────────────────────────────────

    async def _phase_e4_generate_code(self) -> None:
        """Generate a complete iterative Python simulation via qwen2.5."""
        t0 = datetime.now()
        print("\n── Phase E4: Code Generation ──")
        self._emit_phase("E4", "start")

        given_vars_block = "\n".join(
            f"  {k} = {v}  # from problem statement"
            for k, v in self.given_values.items()
        ) or "  # (no numeric values explicitly given — use engineering defaults)"

        graph_context = (
            self.dep_graph.to_prompt_context()
            if self.dep_graph.nodes else "(no dependency graph)"
        )

        defaults_block = (
            format_defaults_for_prompt(self.domain)
            if _HAS_DEFAULTS else "# (engineering_defaults.py not available)"
        )

        search_ctx = (self.search_context[:2000] + "\n# [truncated]") \
            if len(self.search_context) > 2000 else self.search_context or "# (no search context)"

        prompt = ENGINEER_GENERATION_PROMPT.format(
            problem=self.problem,
            given_vars_block=given_vars_block,
            graph_context=graph_context,
            defaults_block=defaults_block,
            search_context=search_ctx,
        )

        response = await self._llm_query_coder(prompt)

        # Extract code block
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            import_match = re.search(r'^(import |from )', response, re.MULTILINE)
            if import_match:
                code = response[import_match.start():].strip()
            else:
                code = ""
        else:
            code = code_match.group(1).strip()

        # Auto-fill any stray placeholders
        if code:
            leftovers = re.findall(r'(?<!\{)\{(\w+)\}(?!\})', code)
            leftovers = [p for p in leftovers if not re.search(r'f["\'].*\{' + p + r'\}', code)]
            for ph in leftovers:
                val = self.given_values.get(ph, 0.0)
                code = code.replace(f"{{{ph}}}", str(val))
                print(f"   ⚙️  Auto-filled placeholder {{{ph}}} = {val}")

        if code:
            valid = EquationGenerator.validate_syntax(code) if _HAS_GENERATOR else True
            if valid:
                print(f"   ✅ Generated {len(code.splitlines())} lines, syntax valid")
            else:
                print(f"   ⚠️  Generated code has syntax errors")
        else:
            print("   ⚠️  Code generation produced empty output")

        self.generated_code = code
        self._phase_times["E4"] = (datetime.now() - t0).total_seconds()
        self._emit_phase("E4", "done", elapsed_s=self._phase_times["E4"], code_lines=len(code.splitlines()) if code else 0)

    # ── Phase E5 ──────────────────────────────────────────────────────────────

    async def _phase_e5_execute_and_validate(self) -> None:
        """Execute the generated script, check bounds, self-correct up to 2 rounds."""
        t0 = datetime.now()
        print("\n── Phase E5: Execute + Validate ──")
        self._emit_phase("E5", "start")

        if not self.generated_code:
            print("   ⚠️  No code to execute")
            self._phase_times["E5"] = 0.0
            return

        for attempt in range(2):
            print(f"   Attempt {attempt + 1}/2…")
            result = await self._execute_code(self.generated_code, timeout=60)

            if result["success"]:
                self.execution_output = result["output"]
                self.execution_success = True
                print(f"   ✅ Execution succeeded ({len(self.execution_output)} chars output)")

                # Bounds check
                if _HAS_DEFAULTS:
                    computed = self._parse_output_values(self.execution_output)
                    self.bounds_violations = check_physical_bounds(computed)
                    fatals = [v for v in self.bounds_violations if v.is_fatal]
                    warnings = [v for v in self.bounds_violations if not v.is_fatal]
                    if warnings:
                        print(f"   ⚠️  Bounds warnings: {len(warnings)}")
                    if fatals:
                        print(f"   ❌ Fatal bounds violations: {len(fatals)}")
                        for bv in fatals:
                            print(f"      {bv.message}")
                        if attempt == 0:
                            corrected = await self._adjust_assumptions_and_retry(fatals)
                            if corrected:
                                self.generated_code = corrected
                                self.execution_success = False
                                continue   # retry
                break  # success, exit loop

            else:
                error_text = result.get("error", "unknown error")
                print(f"   ❌ Execution failed: {error_text[:200]}")
                if attempt == 0:
                    corrected = await self._self_correct_syntax(error_text)
                    if corrected:
                        self.generated_code = corrected
                        continue  # retry with corrected code
                break  # give up

        self._phase_times["E5"] = (datetime.now() - t0).total_seconds()
        self._emit_phase(
            "E5", "done",
            elapsed_s=self._phase_times["E5"],
            success=self.execution_success,
            n_violations=len(self.bounds_violations),
        )

    async def _execute_code(self, code: str, timeout: int = 60) -> Dict:
        """Execute Python code in a subprocess and return output."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                tmp = f.name
            try:
                proc = subprocess.run(
                    ["python3", tmp],
                    capture_output=True, text=True, timeout=timeout,
                )
                if proc.returncode != 0:
                    return {
                        "success": False,
                        "output": proc.stdout,
                        "error": f"Exit code {proc.returncode}\n{proc.stderr}",
                    }
                return {"success": True, "output": proc.stdout}
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Timeout after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _self_correct_syntax(self, error_text: str) -> Optional[str]:
        """Feed execution error back to qwen2.5 for a single correction attempt."""
        print("   🔄 Self-correcting syntax…")
        prompt = (
            f"The following Python script failed with this error:\n\n"
            f"ERROR:\n{error_text[:1000]}\n\n"
            f"SCRIPT:\n```python\n{self.generated_code[:3000]}\n```\n\n"
            f"Fix the error and return the complete corrected script in ```python``` fences. "
            f"Do not add placeholder syntax."
        )
        response = await self._llm_query_coder(prompt)
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            if _HAS_GENERATOR and EquationGenerator.validate_syntax(code):
                return code
        return None

    async def _adjust_assumptions_and_retry(
        self, violations: List
    ) -> Optional[str]:
        """Ask qwen2.5 to adjust violated assumptions and return corrected script."""
        print("   🔄 Adjusting assumptions for bounds violations…")
        viol_text = "\n".join(f"  - {v.message}" for v in violations)
        defaults_hint = format_defaults_for_prompt(self.domain) if _HAS_DEFAULTS else ""
        prompt = (
            f"This script produced physically impossible results:\n{viol_text}\n\n"
            f"Available engineering defaults to use instead:\n{defaults_hint[:500]}\n\n"
            f"SCRIPT:\n```python\n{self.generated_code[:3000]}\n```\n\n"
            f"Correct the assumption values so all results are physically plausible. "
            f"Return the complete corrected script in ```python``` fences."
        )
        response = await self._llm_query_coder(prompt)
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return None

    def _parse_output_values(self, output: str) -> Dict[str, float]:
        """Extract key=value pairs from script stdout for bounds checking."""
        values: Dict[str, float] = {}
        for m in re.finditer(
            r'([a-zA-Z_]\w*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', output
        ):
            try:
                values[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        return values

    # ── Phase E6 ──────────────────────────────────────────────────────────────

    async def _phase_e6_synthesize(self) -> None:
        """Synthesize the 6-section Technical Data Sheet via qwen2.5."""
        t0 = datetime.now()
        print("\n── Phase E6: TDS Synthesis ──")
        self._emit_phase("E6", "start")

        assumptions_table = self._format_assumptions_table()

        if self.execution_success and self.execution_output:
            prompt = ENGINEER_SYNTHESIS_PROMPT.format(
                problem=self.problem,
                assumptions_table=assumptions_table,
                execution_output=self.execution_output[:4000],
            )
        else:
            prompt = E6_FALLBACK_PROMPT.format(problem=self.problem)

        tds = await self._llm_query_coder(
            prompt,
            system_prompt=(
                "You are a senior systems engineer writing a concise but complete "
                "Technical Data Sheet. Use Markdown. Bold all key numeric values."
            ),
        )

        # Clean up any code fences the LLM might wrap around markdown
        tds = re.sub(r'^```(?:markdown)?\n?', '', tds.strip(), flags=re.MULTILINE)
        tds = re.sub(r'\n?```$', '', tds.strip(), flags=re.MULTILINE)

        self.tds_markdown = tds.strip()
        print(f"   TDS length: {len(self.tds_markdown)} chars")
        self._phase_times["E6"] = (datetime.now() - t0).total_seconds()
        self._emit_phase("E6", "done", elapsed_s=self._phase_times["E6"], tds_chars=len(self.tds_markdown))

    def _format_assumptions_table(self) -> str:
        """Format assumptions as a Markdown table string."""
        if not self.assumptions:
            return "| Parameter | Value | Unit | Basis |\n|---|---|---|---|\n| (none recorded) | | | |"
        lines = ["| Parameter | Value | Unit | Basis |", "|---|---|---|---|"]
        for a in self.assumptions:
            lines.append(
                f"| {a.parameter} | {a.value} | {a.unit} | {a.basis} |"
            )
        return "\n".join(lines)

    # ── Markdown save ─────────────────────────────────────────────────────────

    async def _save_markdown(self) -> None:
        """Save the TDS to a timestamped file in /tmp/."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = re.sub(r'[^a-z0-9]+', '_', self.problem[:40].lower()).strip('_')
            path = f"/tmp/engineer_{slug}_{ts}.md"
            with open(path, "w") as f:
                f.write(f"# Technical Data Sheet\n\n")
                f.write(f"**Problem:** {self.problem}\n\n")
                f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
                f.write("---\n\n")
                f.write(self.tds_markdown)
                if self.generated_code:
                    f.write("\n\n---\n\n## Generated Simulation Code\n\n")
                    f.write(f"```python\n{self.generated_code}\n```\n")
            self.markdown_path = path
            print(f"\n📄 Markdown saved: {path}")
        except Exception as e:
            print(f"⚠️  Could not save markdown: {e}")


# ─── Entry function ───────────────────────────────────────────────────────────

async def run_engineer_mode(
    problem: str,
    searxng_url: Optional[str] = None,
    debug: bool = False,
    save_markdown: bool = False,
    sidechain: Optional[Any] = None,
    job_id: str = "",
) -> str:
    """
    Run Engineer Mode for the given design problem.

    Returns the Technical Data Sheet as a Markdown string.
    """
    emo = EngineerModeOrchestrator(
        searxng_url=searxng_url,
        debug=debug,
        save_markdown=save_markdown,
        sidechain=sidechain,
        job_id=job_id,
    )
    return await emo.run(problem)


if __name__ == "__main__":
    import sys
    problem = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Design a two-stage rocket to put 15,000 kg in LEO using LOX/RP-1"
    )
    result = asyncio.run(run_engineer_mode(
        problem=problem,
        searxng_url=os.getenv("SEARXNG_URL"),
        debug=True,
        save_markdown=True,
    ))
    print("\n" + "=" * 70)
    print("TECHNICAL DATA SHEET")
    print("=" * 70)
    print(result)
