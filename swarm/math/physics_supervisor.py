"""
Physics Supervisor — Swarm 3.0

Derives symbolic equations and a solver strategy (using phi4:14b) *before*
the coder (qwen2.5:14b) runs.  The coder is then required to implement the
supervisor's exact equations rather than inventing its own from scratch.

This prevents class-of-errors like:
  • Confusing robot-frame vs world-frame velocities
  • Finding where a parabola hits the ground instead of a target
  • Incorrect discriminant leading to negative sqrt (geometry already wrong)
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class PhysicsEquationPlan:
    """Structured output from the Physics Supervisor."""
    coordinate_system:  str               # "Origin at robot initial pos; +x forward..."
    symbolic_equations: List[str]         # ["x(t) = x_turret + Vx_world*t", ...]
    solution_strategy:  str               # "scipy.optimize.fsolve; 3 unknowns (az, el, t)"
    known_pitfalls:     List[str]         # ["do not mix robot-frame and world-frame velocity"]
    raw_response:       str = ""


SUPERVISOR_PROMPT = """You are a physics expert. Your job is to derive the CORRECT governing equations
for the problem below BEFORE any code is written.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROBLEM:
{problem}

EXPLICITLY GIVEN VALUES:
{given_vars_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSTRUCTIONS — work through each step in your reasoning, then output JSON:

STEP 1 — COORDINATE FRAMES
  • Define ALL coordinate frames (world frame, robot frame, turret frame, etc.)
  • For each physical quantity (velocity, position, force), state explicitly which frame it is in.
  • If the problem involves a moving platform: derive the world-frame velocity of the projectile as
      V_world = V_robot_linear + ω × r_offset + R_robot @ V_launch_body
    where R_robot is the rotation matrix from robot heading, ω is angular velocity,
    r_offset is the vector from robot CoM to the turret, and V_launch_body is the launch
    velocity in the robot (body) frame.

STEP 2 — GOVERNING EQUATIONS
  • Write EVERY governing equation as a named symbolic relationship.
  • For projectile motion: treat horizontal (x, y) and vertical (z) as INDEPENDENT.
    - x(t) = x0 + Vx_world * t
    - y(t) = y0 + Vy_world * t
    - z(t) = z0 + Vz_world * t - 0.5 * g * t²
  • Do NOT conflate 2D ground-plane range with the vertical trajectory.
  • For rotating-frame problems: ALWAYS compute the full 3-component velocity
    before splitting into horizontal and vertical.

STEP 3 — UNKNOWNS AND SOLVER STRATEGY
  • List every unknown variable.
  • If there are ≥ 3 unknowns with nonlinear coupling, recommend scipy.optimize.fsolve
    (or brentq for 1-D root finding).
  • If a quadratic/analytic approach is used, explicitly check the discriminant and
    flag if it can go negative (target geometrically unreachable — adjust approach).

STEP 4 — KNOWN PITFALLS
  • List any specific pitfalls for THIS problem (frame confusion, sign conventions,
    unit mismatch, negative discriminant risk, etc.)

OUTPUT — return ONLY this JSON object (no prose before or after):
{{
  "coordinate_system": "<single paragraph describing all frames and sign conventions>",
  "symbolic_equations": [
    "<equation 1 — e.g. Vx_world = V_robot*cos(theta) - omega*r_y + V_launch*cos(az)*cos(el)*cos(theta) - ...>",
    "<equation 2>",
    "..."
  ],
  "solution_strategy": "<one paragraph: which solver, which variables are unknowns, boundary conditions>",
  "known_pitfalls": [
    "<pitfall 1>",
    "<pitfall 2>"
  ]
}}
"""


class PhysicsSupervisor:
    """
    Calls phi4:14b to derive symbolic equations before the code generator runs.
    Returns a PhysicsEquationPlan or None on any failure (never raises).
    """

    @staticmethod
    async def derive_equations(
        problem: str,
        given_variables: Dict[str, Any],
        llm_query_func,          # phi4:14b _llm_query from orchestrator
    ) -> Optional[PhysicsEquationPlan]:
        """
        Ask the supervisor LLM to derive the correct physics equations.

        Args:
            problem:          Full problem statement string
            given_variables:  Dict of extracted numeric values {name: value}
            llm_query_func:   Async callable(prompt, system_prompt='') → str

        Returns:
            PhysicsEquationPlan, or None if the LLM/parse failed
        """
        given_vars_block = "\n".join(
            f"  {k} = {v}" for k, v in given_variables.items()
        ) if given_variables else "  (no explicit numeric values)"

        prompt = SUPERVISOR_PROMPT.format(
            problem=problem,
            given_vars_block=given_vars_block,
        )

        try:
            response = await llm_query_func(
                prompt=prompt,
                system_prompt=(
                    "You are a physics supervisor. "
                    "Output ONLY a valid JSON object matching the schema given. "
                    "No prose before or after the JSON."
                ),
            )
        except Exception as llm_err:
            print(f"  ⚠️  [Supervisor] LLM call failed: {llm_err}")
            return None

        return PhysicsSupervisor._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> Optional[PhysicsEquationPlan]:
        """Extract and parse the JSON from the LLM response."""
        if not response:
            return None

        # Strip ```json fences if present
        cleaned = re.sub(r'^```(?:json)?\s*', '', response.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*```$', '', cleaned.strip())

        # Try to grab the outermost { ... } block (handles leading prose)
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not json_match:
            print(f"  ⚠️  [Supervisor] No JSON object found in response")
            return None

        raw_json = json_match.group(0)

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as e:
            print(f"  ⚠️  [Supervisor] JSON parse error: {e}")
            return None

        # Validate required keys
        required = {"coordinate_system", "symbolic_equations", "solution_strategy", "known_pitfalls"}
        missing = required - set(data.keys())
        if missing:
            print(f"  ⚠️  [Supervisor] Missing keys in JSON: {missing}")
            return None

        return PhysicsEquationPlan(
            coordinate_system=str(data.get("coordinate_system", "")),
            symbolic_equations=[str(e) for e in data.get("symbolic_equations", [])],
            solution_strategy=str(data.get("solution_strategy", "")),
            known_pitfalls=[str(p) for p in data.get("known_pitfalls", [])],
            raw_response=response,
        )


if __name__ == "__main__":
    import asyncio

    async def _smoke_test():
        async def mock_llm(prompt, system_prompt=""):
            return json.dumps({
                "coordinate_system": "World frame: +x forward, +y left, +z up. Origin at robot start.",
                "symbolic_equations": [
                    "Vx_world = V_robot*cos(theta) - omega*r_y + V_launch*cos(az)*cos(el)*cos(theta)",
                    "x(t) = x_turret + Vx_world*t",
                    "z(t) = z0 + Vz_world*t - 0.5*g*t^2",
                ],
                "solution_strategy": "scipy.optimize.fsolve with 3 unknowns: azimuth, elevation, time of flight",
                "known_pitfalls": [
                    "Do not mix robot-frame and world-frame velocities",
                    "Horizontal range is sqrt(x^2+y^2), not just x — check 2-D geometry",
                ],
            })

        plan = await PhysicsSupervisor.derive_equations(
            problem="Test FRC turret problem",
            given_variables={"v_robot": 4.2, "omega": 1.3, "v_launch": 18.0},
            llm_query_func=mock_llm,
        )
        print("Plan:", plan)
        assert plan is not None
        assert len(plan.symbolic_equations) == 3
        print("✅ Smoke test passed")

    asyncio.run(_smoke_test())
