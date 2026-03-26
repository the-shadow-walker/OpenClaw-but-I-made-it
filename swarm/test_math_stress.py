"""
Math Pipeline Stress Tests
===========================

Three complex multi-variable problems that exercise the full orchestrator math pipeline:
  1. Two-stage rocket sizing  (delta-v, TWR, propellant flow, chamber pressure)
  2. Thermal resistance network  (cylindrical vessel, insulation, convection heat loss)
  3. Hohmann + plane-change transfer  (delta-v budget, propellant mass, GEO insertion)

Run:
    python3 test_math_stress.py [--debug] [--problem N]

Requires a running Ollama server and (optionally) a SearXNG instance.
"""

import asyncio
import argparse
import sys
import os
import time
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Problem definitions
# ---------------------------------------------------------------------------

PROBLEMS: List[Tuple[str, str]] = [
    (
        "Two-Stage Rocket Sizing",
        """
        A two-stage rocket must deliver a 500 kg payload to a 400 km circular LEO.
        The first stage uses RP-1/LOX with an Isp of 311 s (sea level) / 338 s (vacuum).
        The second stage uses LH2/LOX with an Isp of 450 s (vacuum).
        Assume a total delta-v requirement of 9300 m/s (including gravity and drag losses).
        Stage mass fractions: first stage structural fraction = 0.08,
                               second stage structural fraction = 0.12.
        Propellant split: allocate 5500 m/s delta-v to the first stage and 3800 m/s to the second.

        Compute:
        - Propellant mass and total wet mass for each stage
        - Mass ratio (Tsiolkovsky) for each stage
        - Thrust for each stage assuming thrust-to-weight ratio >= 1.3 at liftoff
        - Sea-level and vacuum thrust for stage 1 (assume 10% Isp reduction sea-level)
        - First stage chamber pressure assuming exit-area ratio of 16 and nozzle exit pressure
          matching sea-level ambient (101.325 kPa) for a first approximation
        - Mass flow rate for each stage
        - GLOW (gross lift-off weight)
        - Payload fraction (payload / GLOW)
        """,
    ),
    (
        "Cylindrical Vessel Thermal Analysis",
        """
        A steel cylindrical pressure vessel (inner radius 0.5 m, wall thickness 20 mm,
        length 3 m) stores LNG at -162 °C.  The outer surface is wrapped with 80 mm of
        polyurethane foam insulation (k = 0.025 W/m·K).  The ambient environment is 25 °C
        with a natural convection coefficient of 8 W/m²·K on the outer insulation surface.

        Steel thermal conductivity: 16 W/m·K.

        Compute:
        - Thermal resistance of the steel wall (cylindrical geometry)
        - Thermal resistance of the insulation layer (cylindrical geometry)
        - Convective thermal resistance on the outer surface
        - Total thermal resistance (series combination)
        - Steady-state heat ingress rate (W)
        - Boil-off rate of LNG (kg/hr) using latent heat of vaporisation = 510 kJ/kg
        - Temperature at the steel/insulation interface
        - Temperature at the insulation/air interface
        """,
    ),
    (
        "GEO Insertion via Hohmann + Plane Change",
        """
        A satellite is in a circular LEO at 300 km altitude with an orbital inclination of
        28.5 degrees.  It must be transferred to geostationary orbit (GEO, 35786 km altitude)
        at 0 degrees inclination.

        The transfer strategy is:
          1. Hohmann transfer from LEO to GTO (apogee at GEO altitude)
          2. Combined plane-change + circularisation burn at apogee

        The propulsion system has a specific impulse of 321 s.
        The spacecraft dry mass (no propellant) is 1200 kg.

        Compute:
        - LEO circular velocity
        - GTO perigee velocity (after Hohmann burn 1)
        - GTO apogee velocity (before combined burn 2)
        - GEO circular velocity
        - Delta-v for burn 1 (Hohmann departure)
        - Delta-v for burn 2 (combined plane change + circularisation)
        - Total mission delta-v
        - Required propellant mass (rocket equation)
        - Spacecraft wet mass at launch
        - Transfer time (half the GTO period)
        """,
    ),
]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

async def run_problem(title: str, question: str, debug: bool = False) -> dict:
    """Run a single problem through the full orchestrator pipeline."""
    from orchestrator_v2_1 import OrchestratorV2_1

    print(f"\n{'='*70}")
    print(f"  STRESS TEST: {title}")
    print(f"{'='*70}")

    orch = OrchestratorV2_1(
        max_search_concurrent=2,
        enable_verification=True,
        debug=debug,
        searxng_url=os.getenv('SEARXNG_URL'),
    )

    t0 = time.time()
    try:
        answer = await orch.process_question(question.strip())
        elapsed = time.time() - t0

        success = bool(answer and len(answer) > 100)
        has_numbers = any(c.isdigit() for c in answer)

        print(f"\n{'─'*70}")
        print(f"  Result ({elapsed:.1f}s):")
        print(f"{'─'*70}")
        # Print first 800 chars of answer
        preview = answer[:800] + ("..." if len(answer) > 800 else "")
        for line in preview.splitlines():
            print(f"  {line}")

        return {
            "title":      title,
            "success":    success and has_numbers,
            "elapsed_s":  round(elapsed, 1),
            "answer_len": len(answer),
        }

    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ Exception after {elapsed:.1f}s: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return {
            "title":     title,
            "success":   False,
            "elapsed_s": round(elapsed, 1),
            "error":     str(e),
        }


async def main(problem_num: int = None, debug: bool = False):
    problems = PROBLEMS if problem_num is None else [PROBLEMS[problem_num - 1]]

    results = []
    for title, question in problems:
        result = await run_problem(title, question, debug=debug)
        results.append(result)

    # Summary
    print(f"\n\n{'='*70}")
    print("  STRESS TEST SUMMARY")
    print(f"{'='*70}")
    passed = 0
    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        elapsed = f"{r['elapsed_s']}s"
        title = r["title"]
        if r.get("error"):
            print(f"  {status}  {title}  ({elapsed})  — {r['error'][:80]}")
        else:
            print(f"  {status}  {title}  ({elapsed})  — {r.get('answer_len', 0)} chars")
        if r["success"]:
            passed += 1

    print(f"{'─'*70}")
    print(f"  {passed}/{len(results)} passed")
    print(f"{'='*70}\n")

    return passed == len(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Math pipeline stress tests")
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--problem', type=int, choices=[1, 2, 3],
                        help='Run only problem N (1=rocket, 2=thermal, 3=orbital)')
    args = parser.parse_args()

    ok = asyncio.run(main(problem_num=args.problem, debug=args.debug))
    sys.exit(0 if ok else 1)
