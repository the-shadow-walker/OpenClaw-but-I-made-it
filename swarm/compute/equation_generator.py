"""
Equation Generator — v2 (Complete-Code approach)

Instead of generating code with {placeholder} variables that must be
substituted later (which fails when not all values are stated in the
question), the LLM is asked to write a COMPLETE, directly-executable
Python script that:

  1. Inlines all explicitly given values from the question
  2. Uses standard physics constants (g0, mu, c, etc.)
  3. Applies well-known engineering defaults for unspecified design
     variables, with clear comments explaining assumptions
  4. Computes ALL intermediate steps and prints them

This eliminates the "Not solvable: Missing values" failure and produces
rich, show-your-work output even for complex multi-variable problems.
"""

import re
import ast
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class GeneratedEquation:
    """Result of equation generation"""
    python_code:    str
    variables:      List[str]   # informational — extracted from problem, not blocking
    output_vars:    List[str]   # what the code aims to compute
    is_valid_syntax: bool
    error:          Optional[str]    = None
    equations_used: List[str]        = field(default_factory=list)


class EquationGenerator:
    """
    Generates complete, directly-executable Python code to solve math/physics
    problems.  No placeholder substitution required.
    """

    GENERATION_PROMPT = """You are an expert physicist and Python programmer.

Write a COMPLETE, directly-executable Python script to solve the following problem.
The script must run with `python3 script.py` with zero modifications.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM:
{problem}

EXPLICITLY GIVEN VALUES (inline these exactly):
{given_vars_block}

KNOWN EQUATIONS / PRINCIPLES TO APPLY:
{equations}

RESEARCH CONTEXT (use facts here to choose realistic parameters):
{context}

REQUIRED VARIABLE NAMES (use these exact names in code when present):
{variable_schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTHORITATIVE PHYSICS EQUATIONS (implement these — do not invent alternatives):

COORDINATE SYSTEM:
{physics_coordinate_system}

REQUIRED EQUATIONS (implement in order):
{physics_equations_block}

SOLUTION STRATEGY:
{physics_solution_strategy}

PITFALLS TO AVOID:
{physics_pitfalls_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MANDATORY RULES:
1. NO {{variable}} placeholder syntax.  Every variable must be assigned a
   concrete float/int value before it is used.
2. Import only: math, numpy, sympy, scipy (all available).
   uncertainties library available for uncertainty propagation (see Rule 12).
   CoolProp (via PropsSI) and mendeleev available for material properties.
   Use material_props.get_fluid_property() / get_element_property() if installed.
3. For values NOT stated in the problem (Isp, structural fraction,
   expansion ratio, drag coefficient, material properties, etc.):
   - Use the best-practice / textbook engineering default.
   - Add a comment: # assumed: <value> — <one-line rationale>
4. Use all standard physics constants explicitly:
     g0      = 9.80665        # m/s² standard gravity
     mu_E    = 3.986004418e14 # m³/s² Earth GM
     R_E     = 6.3781e6       # m Earth radius
     R_gas   = 8.31446        # J/(mol·K)
     ...etc as needed
5. Compute EVERY intermediate step; print it with a clear label and unit.
   For ALL final answers, also print a machine-readable line:
       RESULT: variable_name = value unit
   Example:  RESULT: delta_v = 9450.23 m/s
   Example:  RESULT: propellant_mass = 48300.0 kg
6. Structure:
     a) GIVEN VALUES block
     b) PHYSICS CONSTANTS block
     c) ASSUMPTIONS block (engineering defaults)
     d) COMPUTATION — one step at a time, with the governing equation
        shown as a comment above each line
     e) SUMMARY — print all final answers clearly
7. For design problems: solve the full chain
   (e.g. delta-v → mass ratio → propellant mass → GTOW → power budget).
8. Use sympy only when you genuinely need symbolic algebra; prefer
   direct numerical computation otherwise.
9. Handle edge cases (avoid division by zero, log of non-positive, etc.).
10. TIME-DEPENDENT PROBLEMS: If the problem involves changing forces, varying
    density, drag, or any quantity that evolves over time (trajectory, rocket
    ascent, thermal transient, orbital propagation), you MUST use
    scipy.integrate.solve_ivp.
    Structure:
      - Define state vector y (e.g. [x, vx, y, vy, mass])
      - Define dy_dt(t, y) right-hand-side function
      - Call solve_ivp(dy_dt, [t0, tf], y0, max_step=0.1, dense_output=True)
    Do NOT use single-formula approximations for these problems.
11. COORDINATE SYSTEM: At the top of the COMPUTATION section, add a block:
    # === COORDINATE SYSTEM ===
    # Origin: <state where origin is, e.g. "launch pad center">
    # +x: <direction, e.g. "East / downrange">
    # +y: <direction, e.g. "Up / altitude">
    # +z: <direction if 3D, else omit>
    # All vector components below follow this convention.
    Gravity is always negative in the +altitude direction: g = -9.80665 m/s²
12. UNCERTAINTY: For any input value that is an engineering assumption (not a
    stated exact figure), wrap it with the `uncertainties` library:
      try:
          from uncertainties import ufloat
          isp = ufloat(350, 17.5)   # ±5% assumed uncertainty
      except ImportError:
          isp = 350.0               # fallback to plain float
    Report final answers including their ± bounds where possible.
    Also print the central value as a plain RESULT: line for machine parsing
    (use  float(nominal_value(x))  or  x.nominal_value  to extract the float).
13. UNIT SAFETY (optional): For complex multi-step calculations, define
    physical quantities using sympy.physics.units to catch dimension errors
    before printing results. Example:
      from sympy.physics.units import kg, meter, second, newton
      F = 5000 * newton
      m = 500 * kg
      a = F / m   # gives 10 m/s² — unit-safe
    Convert to float for RESULT: lines: float(a.evalf())

Output ONLY the Python code inside ```python ... ``` fences.
No prose before or after.
"""

    @staticmethod
    async def generate(
        problem: str,
        given_variables: Dict[str, float],   # {"mass": 15000.0, "distance_km": 400.0}
        unknown_variables: List[str],
        equations_to_use: List[str],
        llm_query_func,
        context: str = "",
        variable_schema: Optional[Dict] = None,
        physics_plan=None,                   # Optional[PhysicsEquationPlan]
    ) -> "GeneratedEquation":
        """
        Generate a complete, directly-executable Python script.

        Args:
            problem:           Full problem statement
            given_variables:   Values extracted from the question
            unknown_variables: Things we want to compute (informational)
            equations_to_use:  Physics principles to apply
            llm_query_func:    Async LLM call (same signature as orchestrator)
            context:           Relevant text from web search results

        Returns:
            GeneratedEquation with directly-runnable python_code
        """

        # Format given values as a clean assignment block for the prompt
        given_vars_block = "\n".join(
            f"  {k} = {v}  # (from problem statement)"
            for k, v in given_variables.items()
        ) if given_variables else "  (No numeric values explicitly stated — use physics defaults)"

        equations_str = "\n".join(f"  • {e}" for e in equations_to_use) \
            if equations_to_use else "  (apply appropriate physics laws)"

        # Trim context to avoid prompt overflow
        context_trimmed = (context[:6000] + "\n  [... context truncated ...]") \
            if len(context) > 6000 else context or "  (no additional context)"

        # Format variable_schema block (canonical names + units for the code generator)
        schema_lines = []
        if variable_schema:
            for var, meta in variable_schema.items():
                if not isinstance(meta, dict):
                    continue
                known_str = f" = {meta['value']}" if meta.get('known') and 'value' in meta else " (unknown — to be computed)"
                unit_str = f" [{meta.get('unit', '')}]" if meta.get('unit') else ""
                desc_str = f" — {meta['description']}" if meta.get('description') else ""
                schema_lines.append(f"  {var}{unit_str}{known_str}{desc_str}")
        schema_block = "\n".join(schema_lines) if schema_lines else "  (no schema — use descriptive names)"

        # Format physics supervisor block (or placeholder if absent)
        if physics_plan is not None:
            physics_coordinate_system = physics_plan.coordinate_system or "(not specified)"
            physics_equations_block = "\n".join(
                f"  {i+1}. {eq}" for i, eq in enumerate(physics_plan.symbolic_equations)
            ) or "  (none provided)"
            physics_solution_strategy = physics_plan.solution_strategy or "(not specified)"
            physics_pitfalls_block = "\n".join(
                f"  • {p}" for p in physics_plan.known_pitfalls
            ) or "  (none identified)"
        else:
            _no_supervisor = "(no supervisor — derive from first principles)"
            physics_coordinate_system = _no_supervisor
            physics_equations_block   = _no_supervisor
            physics_solution_strategy = _no_supervisor
            physics_pitfalls_block    = _no_supervisor

        prompt = EquationGenerator.GENERATION_PROMPT.format(
            problem=problem,
            given_vars_block=given_vars_block,
            equations=equations_str,
            context=context_trimmed,
            variable_schema=schema_block,
            physics_coordinate_system=physics_coordinate_system,
            physics_equations_block=physics_equations_block,
            physics_solution_strategy=physics_solution_strategy,
            physics_pitfalls_block=physics_pitfalls_block,
        )

        try:
            response = await llm_query_func(
                prompt=prompt,
                system_prompt=(
                    "You are a Python code generator. "
                    "Respond ONLY with valid, complete Python code in ```python``` blocks. "
                    "Never use {placeholder} syntax — inline all values directly."
                )
            )

            # Extract code block
            code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if not code_match:
                code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)

            if not code_match:
                # Last resort: take everything after the first import
                import_match = re.search(r'^(import |from )', response, re.MULTILINE)
                if import_match:
                    code = response[import_match.start():].strip()
                else:
                    return GeneratedEquation(
                        python_code="",
                        variables=list(given_variables.keys()),
                        output_vars=unknown_variables,
                        is_valid_syntax=False,
                        error="LLM did not return Python code in a ```python``` block",
                    )
            else:
                code = code_match.group(1).strip()

            # Reject code that still has unfilled placeholders
            leftover = re.findall(r'\{(\w+)\}', code)
            # Filter out f-string expressions which are intentional
            leftover = [p for p in leftover
                        if not re.search(r'f["\'].*\{' + p + r'\}', code)]
            if leftover:
                # Try to auto-fill with known values or zero
                for ph in leftover:
                    val = given_variables.get(ph, 0.0)
                    code = code.replace(f"{{{ph}}}", str(val))
                    print(f"   ⚙️  Auto-filled placeholder {{{ph}}} = {val}")

            # Validate syntax
            is_valid = EquationGenerator.validate_syntax(code)
            if not is_valid:
                return GeneratedEquation(
                    python_code=code,
                    variables=list(given_variables.keys()),
                    output_vars=unknown_variables,
                    is_valid_syntax=False,
                    error="Generated code has invalid Python syntax",
                )

            return GeneratedEquation(
                python_code=code,
                variables=list(given_variables.keys()),
                output_vars=unknown_variables,
                is_valid_syntax=True,
                equations_used=equations_to_use,
            )

        except Exception as e:
            return GeneratedEquation(
                python_code="",
                variables=[],
                output_vars=unknown_variables,
                is_valid_syntax=False,
                error=f"Generation failed: {e}",
            )

    @staticmethod
    def validate_syntax(code: str) -> bool:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def format_code(equation: "GeneratedEquation") -> str:
        """Format equation for display."""
        lines = ["\n" + "=" * 70, "GENERATED EQUATION CODE", "=" * 70]

        if equation.is_valid_syntax:
            lines += [
                "\n✅ Syntax: VALID",
                f"\n📐 Equations used:",
            ]
            for eq in (equation.equations_used or []):
                lines.append(f"   • {eq}")
            lines += [
                f"\n💻 Code preview (first 25 lines):",
                "   " + "\n   ".join(equation.python_code.split("\n")[:25]),
            ]
        else:
            lines += [
                "\n❌ Syntax: INVALID",
                f"\n❌ Error: {equation.error}",
                "\n💻 Generated code:",
                "   " + "\n   ".join((equation.python_code or "N/A").split("\n")[:15]),
            ]

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("Equation Generator v2 — Complete-Code approach")
    print("=" * 70)
    print("\nCapabilities:")
    print("  • Generate complete, directly-executable Python scripts")
    print("  • Inlines all given values; uses physics constants + eng. defaults")
    print("  • No placeholder substitution — eliminates 'Missing values' failures")
    print("  • Rich show-your-work output with intermediate steps")
