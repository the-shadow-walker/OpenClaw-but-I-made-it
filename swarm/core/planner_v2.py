"""
Planner V2 — Structured SolvePlan generator

Given a question and ClassificationResult, produces a SolvePlan that:
  • Decomposes the problem into ordered SubProblems
  • Specifies targeted lookup_queries per sub-problem (NOT the raw question)
  • Establishes a dependency graph so the solver can run in topological waves

Uses phi4:14b for speed (structured JSON output).
"""

import json
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    """One distinct task extracted from the user question."""
    id: str                          # "R1", "R2", ...
    text: str                        # "find radius of circular orbit"
    req_type: str                    # "compute" | "explain" | "compare" | "describe"
    negative_constraints: List[str]  # ["no_formulas", "no_equations"]


@dataclass
class SubProblem:
    id: str                          # "SP1", "SP2", …
    description: str                 # human-readable task
    domain: str                      # "physics", "electrical", etc.
    inputs: Dict[str, Any]           # given values + outputs from prior SPs
    expected_outputs: List[Dict]     # [{"name": "thrust", "unit": "N"}, …]
    approach: str                    # "Tsiolkovsky equation", "KVL", etc.
    lookup_queries: List[str]        # targeted search queries
    depends_on: List[str]            # SP IDs that must finish first


@dataclass
class SolvePlan:
    problem: str
    domain: str
    given_values: Dict[str, float]
    coordinate_system: str
    sub_problems: List[SubProblem]
    dependency_order: List[str]       # topological execution order
    notes: str

    def to_markdown(self) -> str:
        """Serialise plan to a clean reference doc string."""
        lines = [
            f"# Solve Plan\n",
            f"**Problem:** {self.problem}\n",
            f"**Domain:** {self.domain}",
            f"**Coordinate system:** {self.coordinate_system}\n",
            f"## Given Values",
        ]
        for var, val in self.given_values.items():
            lines.append(f"  - {var} = {val}")
        lines.append(f"\n## Sub-Problems (execution order: {', '.join(self.dependency_order)})")
        for sp in self.sub_problems:
            lines += [
                f"\n### {sp.id}: {sp.description}",
                f"- **Domain:** {sp.domain}",
                f"- **Approach:** {sp.approach}",
                f"- **Depends on:** {', '.join(sp.depends_on) or 'none'}",
                f"- **Inputs:** {sp.inputs}",
                f"- **Expected outputs:** {sp.expected_outputs}",
                f"- **Lookup queries:** {sp.lookup_queries}",
            ]
        lines.append(f"\n## Notes\n{self.notes}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "domain": self.domain,
            "given_values": self.given_values,
            "coordinate_system": self.coordinate_system,
            "sub_problems": [
                {
                    "id": sp.id,
                    "description": sp.description,
                    "domain": sp.domain,
                    "inputs": sp.inputs,
                    "expected_outputs": sp.expected_outputs,
                    "approach": sp.approach,
                    "lookup_queries": sp.lookup_queries,
                    "depends_on": sp.depends_on,
                }
                for sp in self.sub_problems
            ],
            "dependency_order": self.dependency_order,
            "notes": self.notes,
        }


# ── Prompts ───────────────────────────────────────────────────────────────────

_REQUIREMENT_PROMPT = """\
Extract all distinct tasks from this question. Each separate computation,
analysis, or explanation is ONE requirement.

QUESTION: {question}

Split on: semicolons, "then", "finally", "next", "also", "and" (when joining
distinct unrelated tasks).

Look for negative constraints:
- "without using symbolic formulas" / "without equations" → "no_formulas"
- "in plain language" / "conceptually" / "without math" → "no_math"
- "no calculus" → "no_calculus"

Respond ONLY with valid JSON (no markdown fences):
{{
  "requirements": [
    {{
      "id": "R1",
      "text": "find the circular orbit radius",
      "req_type": "compute",
      "negative_constraints": []
    }},
    {{
      "id": "R2",
      "text": "explain the electrochemistry process without symbolic formulas",
      "req_type": "explain",
      "negative_constraints": ["no_formulas"]
    }}
  ]
}}

req_type values: "compute" | "explain" | "compare" | "describe"
"""

_PLANNER_PROMPT = """\
You are a precise scientific problem planner. Decompose the question below into
sub-problems that can be solved in dependency order.

QUESTION: {question}

CLASSIFICATION INFO:
- Type: {qtype}
- Domain: {domain}
- Given variables: {given}
- Unknown variables: {unknown}
- Equations needed: {equations}
- Variable schema: {schema}

{requirements_block}

RULES:
0. Create EXACTLY ONE sub-problem per requirement listed above. Do NOT merge
   requirements. Do NOT skip requirements. Do NOT create extra SPs beyond the list.
1. Each sub-problem solves ONE clearly defined thing.
2. lookup_queries must be SPECIFIC (e.g. "molar enthalpy CO2 at 500K") NOT vague
   (e.g. "how does combustion work").  Max 2 queries per sub-problem.
3. depends_on lists SP ids (e.g. ["SP1"]) that must finish before this SP runs.
4. If the whole question is a single calculation, use exactly ONE sub-problem.
5. coordinate_system: choose a clear frame and origin (or "N/A" for non-spatial).
6. Create exactly ONE SP per DISTINCT task. Do NOT merge unrelated tasks into
   one SP. Do NOT omit any task. Do NOT invent extra SPs.
7. Unrelated tasks that share NO variables (e.g., a classical mechanics problem
   AND a number theory problem AND a chemistry problem) MUST be separate SPs with
   depends_on: []. They will run in PARALLEL, cutting total time. Only add a
   dependency if SP_B genuinely needs a RESULT value from SP_A.

Respond ONLY with valid JSON (no markdown fences, no explanation):
{{
  "domain": "physics",
  "coordinate_system": "+x east, +y up, origin at launch pad",
  "given_values": {{"mass_kg": 10.0, "velocity_ms": 5.0}},
  "sub_problems": [
    {{
      "id": "SP1",
      "description": "Calculate kinetic energy",
      "domain": "mechanics",
      "inputs": {{"mass_kg": 10.0, "velocity_ms": 5.0}},
      "expected_outputs": [{{"name": "KE", "unit": "J"}}],
      "approach": "KE = 0.5 * m * v^2",
      "lookup_queries": [],
      "depends_on": []
    }}
  ],
  "dependency_order": ["SP1"],
  "notes": "Straightforward single-step calculation."
}}
"""


class PlannerV2:
    """
    Generates a SolvePlan from a question + ClassificationResult.
    Uses phi4:14b (fast, structured JSON).
    """

    @staticmethod
    async def create_plan(
        question: str,
        classification,              # ClassificationResult or None
        llm_query_func: Callable,    # async (prompt, system) → str
    ) -> Tuple["SolvePlan", List["Requirement"]]:
        """
        Generate and return (SolvePlan, List[Requirement]).
        Falls back to a single-SP plan wrapping the whole question on any failure.
        """
        # ── Step 1: Extract requirements (Requirement Shredder) ───────────────
        requirements = await PlannerV2.extract_requirements(question, llm_query_func)
        print(f"📝 Requirements: {len(requirements)} extracted"
              + (f" [{', '.join(r.id for r in requirements)}]" if requirements else ""))

        try:
            # ── Step 2: Build planner prompt ──────────────────────────────────
            if classification:
                given = classification.given_variables[:10]
                unknown = classification.unknown_variables[:10]
                equations = classification.equations_needed[:6]
                domain = classification.domain or "unknown"
                qtype = classification.question_type.value
                schema_str = json.dumps(classification.variable_schema, indent=None)[:600]
            else:
                given = unknown = equations = []
                domain = "unknown"
                qtype = "unknown"
                schema_str = "{}"

            # Build requirements block for the planner
            if requirements:
                req_lines = []
                for r in requirements:
                    constraints = (
                        f" [{'|'.join(r.negative_constraints)}]"
                        if r.negative_constraints else ""
                    )
                    req_lines.append(f"  {r.id}: {r.text} [{r.req_type}]{constraints}")
                requirements_block = (
                    f"REQUIRED TASKS ({len(requirements)} total):\n"
                    + "\n".join(req_lines)
                    + "\n\nYou MUST create EXACTLY ONE SP per requirement. "
                    + "Do NOT merge, skip, or combine any requirements."
                )
            else:
                requirements_block = ""

            prompt = _PLANNER_PROMPT.format(
                question=question[:1000],
                qtype=qtype,
                domain=domain,
                given=given,
                unknown=unknown,
                equations=equations,
                schema=schema_str,
                requirements_block=requirements_block,
            )

            system = (
                "You are a scientific problem planner. "
                "Output ONLY valid JSON matching the schema exactly."
            )

            raw = await llm_query_func(prompt, system)
            plan = PlannerV2._parse_plan(question, raw, classification, requirements)
            print(f"📋 SolvePlan: {len(plan.sub_problems)} sub-problem(s), "
                  f"order: {plan.dependency_order}")
            return plan, requirements

        except Exception as e:
            print(f"⚠️  PlannerV2 failed ({e}), using single-SP fallback")
            return PlannerV2._fallback_plan(question, classification), requirements

    # ── Requirement extraction ────────────────────────────────────────────────

    @staticmethod
    async def extract_requirements(
        question: str,
        llm_query_func: Callable,
    ) -> List["Requirement"]:
        """
        Pre-planning step: extract all distinct tasks and negative constraints
        from the question. Falls back gracefully to a single R1 on any failure.
        """
        try:
            prompt = _REQUIREMENT_PROMPT.format(question=question[:1200])
            system = (
                "You are a requirement extractor. "
                "Output ONLY valid JSON matching the schema exactly."
            )
            raw = await llm_query_func(prompt, system)
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
            data = json.loads(text)
            reqs = []
            for item in data.get("requirements", []):
                reqs.append(Requirement(
                    id=item.get("id", f"R{len(reqs)+1}"),
                    text=item.get("text", ""),
                    req_type=item.get("req_type", "compute"),
                    negative_constraints=item.get("negative_constraints", []),
                ))
            if reqs:
                return reqs
        except Exception as e:
            print(f"⚠️  extract_requirements failed ({e}), using R1 fallback")

        # Fallback: single R1 covering the whole question
        return [Requirement(id="R1", text=question[:200], req_type="compute",
                            negative_constraints=[])]

    # ── Parsing ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_plan(question: str, raw: str, classification,
                    requirements: Optional[List["Requirement"]] = None) -> "SolvePlan":
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        data = json.loads(text)

        # Build SubProblem list
        sub_problems = []
        for sp_data in data.get("sub_problems", []):
            sub_problems.append(SubProblem(
                id=sp_data.get("id", "SP1"),
                description=sp_data.get("description", "Solve the problem"),
                domain=sp_data.get("domain", data.get("domain", "physics")),
                inputs=sp_data.get("inputs", {}),
                expected_outputs=sp_data.get("expected_outputs", []),
                approach=sp_data.get("approach", ""),
                lookup_queries=sp_data.get("lookup_queries", [])[:3],
                depends_on=sp_data.get("depends_on", []),
            ))

        if not sub_problems:
            raise ValueError("No sub_problems in LLM response")

        # Topological order — use LLM's if provided, else derive
        kept_ids = {sp.id for sp in sub_problems}
        dep_order = [x for x in data.get("dependency_order", []) if x in kept_ids]
        if not dep_order:
            dep_order = [sp.id for sp in sub_problems]

        # Given values — merge from classification.variable_schema if available
        given_vals = data.get("given_values", {})
        if classification and classification.variable_schema:
            for vname, vmeta in classification.variable_schema.items():
                if vmeta.get("known") and "value" in vmeta and vname not in given_vals:
                    try:
                        given_vals[vname] = float(vmeta["value"])
                    except (TypeError, ValueError):
                        pass

        # Validate: warn if planner dropped any requirements
        if requirements and len(sub_problems) < len(requirements):
            missing_count = len(requirements) - len(sub_problems)
            print(f"⚠️  WARNING: Planner created {len(sub_problems)} SP(s) for "
                  f"{len(requirements)} requirement(s) — "
                  f"{missing_count} requirement(s) may have been dropped!")
            print(f"   Requirements: {[r.id for r in requirements]}")
            print(f"   SPs created:  {[sp.id for sp in sub_problems]}")

        return SolvePlan(
            problem=question,
            domain=data.get("domain", "physics"),
            given_values=given_vals,
            coordinate_system=data.get("coordinate_system", "N/A"),
            sub_problems=sub_problems,
            dependency_order=dep_order,
            notes=data.get("notes", ""),
        )

    @staticmethod
    def _fallback_plan(question: str, classification) -> SolvePlan:
        """Single-SP plan used when the LLM fails or returns bad JSON."""
        given_vals = {}
        domain = "physics"
        if classification:
            domain = classification.domain or "physics"
            if classification.variable_schema:
                for vname, vmeta in classification.variable_schema.items():
                    if vmeta.get("known") and "value" in vmeta:
                        try:
                            given_vals[vname] = float(vmeta["value"])
                        except (TypeError, ValueError):
                            pass

        sp = SubProblem(
            id="SP1",
            description="Solve the full problem",
            domain=domain,
            inputs=given_vals,
            expected_outputs=[],
            approach="derive from first principles",
            lookup_queries=[],
            depends_on=[],
        )
        return SolvePlan(
            problem=question,
            domain=domain,
            given_values=given_vals,
            coordinate_system="N/A",
            sub_problems=[sp],
            dependency_order=["SP1"],
            notes="Single-SP fallback (planner returned no valid JSON).",
        )
