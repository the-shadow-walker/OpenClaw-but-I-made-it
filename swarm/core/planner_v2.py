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


# ── JSON helpers ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robustly extract and parse a JSON object from LLM output.
    Handles: markdown fences, <think>…</think> blocks, prose preambles,
    single-quote JSON (from qwq/phi4 lazy outputs).
    """
    import ast

    # Strip <think>…</think> reasoning blocks (qwq/deepseek-r1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    # Direct parse first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Scan for first {...} block (handles prose/think prefixes)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        blob = m.group()
        # Try strict JSON parse on the extracted blob
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
        # Fallback: ast.literal_eval handles single-quote dicts from lazy models
        try:
            result = ast.literal_eval(blob)
            if isinstance(result, dict):
                return result
        except (ValueError, SyntaxError):
            pass
    raise ValueError("No JSON object found in LLM response")


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
8. VARIABLE NAME CONTINUITY: If SP_B depends on SP_A, the variable names in
   SP_B's inputs MUST exactly match the expected_output names from SP_A.
   WRONG: SP_A outputs "r_circular_orbit", SP_B inputs has "r0" or "radius"
   RIGHT: SP_A outputs "r_circular_orbit", SP_B inputs has "r_circular_orbit"
9. NO DUPLICATE COMPUTATIONS: Never create two SPs that compute the same physical
   quantity. If SP_A computes the circular orbit radius, no later SP should also
   compute the radius. Use SP_A's result as an input to SP_B, not a re-derivation.
   If a requirement says "verify the radius", the SP should take r from prior SP and
   verify it numerically — NOT solve a fresh equation to find a different r.
10. STRICT DOMAIN TAGGING — the following rules are NON-NEGOTIABLE. Apply each
    SP's description against these keyword lists and tag the domain accordingly.
    The orchestrator WILL post-process and override any violation — match it
    the first time to avoid retries.

    MATHEMATICS (use domain = "mathematics"):
      KEYWORDS: integral, integrate, ∫, series, sum, summation, Σ, converge,
      divergence, Stirling, Taylor series, Laurent, residue, contour, limit,
      improper integral, asymptotic, L'Hôpital, differentiate symbolically.
      EXAMPLES: "evaluate ∫_0^∞ x^3 e^{{-ax}} dx" → domain: "mathematics"
                "analyze convergence of Σ n!e^n/(n^n √n)" → domain: "mathematics"

    CHEMISTRY (use domain = "chemistry" or "electrochemistry"):
      KEYWORDS: Nernst, Gibbs, Gibbs-Helmholtz, ΔG, ΔH, ΔS, enthalpy, entropy,
      cell potential, EMF, half-cell, Faraday, activity coefficient, pH, pKa,
      reaction quotient, electrolyte, redox, equilibrium constant K.
      EXAMPLES: "derive cell potential via Gibbs-Helmholtz" → domain: "chemistry"
                "compute EMF vs temperature" → domain: "electrochemistry"

    PHYSICS (use domain = "physics", "mechanics", "orbital_mechanics", etc.):
      KEYWORDS: force, Newton's, Lagrangian, Hamiltonian, orbit, potential V(r),
      relativistic, wavefunction, Schrödinger, field, charge, momentum, energy
      conservation.

    ALGORITHM:
      For each SP, scan the description for the keyword lists above IN ORDER:
      1. If any MATHEMATICS keyword appears → domain MUST be "mathematics"
         (even if the SP is "part of" a bigger physics problem)
      2. Else if any CHEMISTRY keyword appears → domain MUST be "chemistry"
      3. Else use PHYSICS subdomain

    FORBIDDEN: defaulting all SPs to the overall question domain. A physics
    question with an integral sub-task MUST have the integral SP tagged
    "mathematics", not "physics". The DomainGate depends on this.

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
                "You are the Lead Systems Architect. "
                "You are being evaluated on COMPLETENESS — every distinct mathematical "
                "operation, integral, series, chemical step, or conceptual explanation "
                "MUST have its own Sub-Problem. Omitting a single task means mission failure. "
                "Output ONLY valid JSON matching the schema exactly. No prose, no fences."
            )

            raw = await llm_query_func(prompt, system)
            # Swarm 3.18 — one-shot retry if first reply isn't parseable JSON.
            try:
                plan = PlannerV2._parse_plan(question, raw, classification, requirements)
            except (ValueError, KeyError, TypeError) as parse_err:
                print(f"⚠️  PlannerV2 parse failed ({parse_err}) — retrying with stricter reminder")
                strict_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous reply was not valid JSON. "
                    + "Output ONLY the JSON object — no thinking, no prose, no fences. "
                    + "Begin with `{` and end with `}`."
                )
                raw = await llm_query_func(strict_prompt, system)
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
                "You are a precision requirement extractor. "
                "Your job: find EVERY distinct task in the question — computations, "
                "explanations, comparisons, derivations, each gets its own requirement. "
                "Never merge two different operations. Never skip fine-print tasks. "
                "Output ONLY valid JSON matching the schema exactly. No prose, no fences."
            )
            raw = await llm_query_func(prompt, system)
            # Swarm 3.18 — one-shot retry on JSON parse failure with stricter reminder.
            try:
                data = _extract_json(raw)
            except ValueError:
                print("⚠️  extract_requirements: no JSON in first reply — retrying")
                strict_prompt = (
                    prompt
                    + "\n\nIMPORTANT: Your previous reply was not valid JSON. "
                    + "Output ONLY the JSON object — no thinking, no prose, no fences. "
                    + "Begin with `{` and end with `}`."
                )
                raw = await llm_query_func(strict_prompt, system)
                data = _extract_json(raw)
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
        data = _extract_json(raw)

        # Build SubProblem list
        # Swarm 3.15: qwen3-coder:30b sometimes emits inputs as list-of-dicts;
        # coerce to {var: value} so downstream .items() never crashes.
        def _coerce_inputs(_raw):
            if isinstance(_raw, dict):
                return _raw
            if isinstance(_raw, list):
                _out = {}
                for _it in _raw:
                    if isinstance(_it, dict):
                        _n = _it.get("name") or _it.get("var") or _it.get("variable") or _it.get("symbol")
                        _v = _it.get("value", _it.get("val"))
                        if _n is not None and _v is not None:
                            _out[str(_n)] = _v
                return _out
            return {}

        sub_problems = []
        for sp_data in data.get("sub_problems", []):
            sub_problems.append(SubProblem(
                id=sp_data.get("id", "SP1"),
                description=sp_data.get("description", "Solve the problem"),
                domain=sp_data.get("domain", data.get("domain", "physics")),
                inputs=_coerce_inputs(sp_data.get("inputs", {})),
                expected_outputs=sp_data.get("expected_outputs", []),
                approach=sp_data.get("approach", ""),
                lookup_queries=sp_data.get("lookup_queries", [])[:3],
                depends_on=sp_data.get("depends_on", []),
            ))

        if not sub_problems:
            raise ValueError("No sub_problems in LLM response")

        # Swarm 3.14.2 — Authoritative Domain Partitions (doesn't trust the LLM).
        # Scan each SP description for strict keyword matches. If matched,
        # FORCE the domain to the matched category. Prevents the planner's
        # "everything is physics" drift from hiding math/chem SPs from the
        # DomainGate at the end.
        _MATH_KWS = re.compile(
            r"\b(integral|integrate|∫|series\b|summation|Σ|converge|diverge|"
            r"stirling|taylor\s+series|laurent|residue\s+theorem|contour|"
            r"l['’]?hôpital|l['’]?hopital|improper\s+integral|asymptotic|"
            r"differentiate\s+symbolically|symbolic\s+integration)",
            re.IGNORECASE,
        )
        _CHEM_KWS = re.compile(
            r"\b(nernst|gibbs[-\s]?helmholtz|gibbs\s+(free\s+)?energy|ΔG|Δ?H\b|"
            r"enthalpy|entropy|cell\s+potential|\bemf\b|half[-\s]?cell|faraday|"
            r"activity\s+coefficient|\bpH\b|pKa|reaction\s+quotient|electrolyte|"
            r"redox|equilibrium\s+constant|electrochemical)",
            re.IGNORECASE,
        )
        for sp in sub_problems:
            desc = sp.description or ""
            original = sp.domain or ""
            if _MATH_KWS.search(desc):
                if "math" not in original.lower() and "calculus" not in original.lower():
                    print(f"   🔖 Domain override: {sp.id} {original!r} → 'mathematics' "
                          f"(keyword match in description)")
                    sp.domain = "mathematics"
            elif _CHEM_KWS.search(desc):
                if not any(x in original.lower() for x in ("chem", "electro")):
                    print(f"   🔖 Domain override: {sp.id} {original!r} → 'chemistry' "
                          f"(keyword match in description)")
                    sp.domain = "chemistry"

        # Topological order — use LLM's if provided, else derive
        kept_ids = {sp.id for sp in sub_problems}
        dep_order = [x for x in data.get("dependency_order", []) if x in kept_ids]
        if not dep_order:
            dep_order = [sp.id for sp in sub_problems]

        # Given values — merge from classification.variable_schema if available.
        # Swarm 3.15: qwen3-coder:30b sometimes emits list-of-dicts shape; coerce
        # to {var: value} dict here so downstream .items() never crashes.
        _gv_raw = data.get("given_values", {})
        given_vals: Dict[str, float] = {}
        if isinstance(_gv_raw, dict):
            given_vals = dict(_gv_raw)
        elif isinstance(_gv_raw, list):
            for _item in _gv_raw:
                if not isinstance(_item, dict):
                    continue
                _name = _item.get("name") or _item.get("var") or _item.get("variable") or _item.get("symbol")
                _value = _item.get("value", _item.get("val"))
                if _name and _value is not None:
                    try:
                        given_vals[str(_name)] = float(_value)
                    except (TypeError, ValueError):
                        given_vals[str(_name)] = _value  # keep as-is
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
