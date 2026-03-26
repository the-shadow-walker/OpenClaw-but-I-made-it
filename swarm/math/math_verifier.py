"""
Math Verifier — independent second computation + numeric fact-check.

Two complementary verification strategies:

1. verify_independent()
   Asks the LLM to solve the SAME problem via a DIFFERENT mathematical method,
   executes the resulting script, then compares key numeric outputs within a
   configurable tolerance.  Agreement → high confidence; disagreement → flag.

2. cross_check_with_search()
   Asks the reasoning LLM to compare each computed number against the research
   facts gathered during Phase 2, returning a structured JSON plausibility verdict.
"""

import re
import json
import asyncio
import subprocess
import tempfile
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class MathVerificationResult:
    primary_values:    Dict[str, float]       # from first execution
    secondary_values:  Dict[str, float]       # from independent second solve
    agreements:        Dict[str, bool]        # var → within tolerance?
    discrepancies:     List[str]              # human-readable difference messages
    overall_agreement: bool
    tolerance_pct:     float
    note:              str = ""
    secondary_code:    str = ""              # Python code of the independent second solve


class MathVerifier:
    """
    Independent cross-verification of LLM-generated math results.
    All methods are static / async-static so no instantiation is needed.
    """

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_result_lines(stdout: str) -> Dict[str, float]:
        """
        Extract  RESULT: var = value [unit]  lines from script stdout.
        Returns a dict of  var_name → float.
        """
        pattern = re.compile(
            r'RESULT:\s*(\w+)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            re.IGNORECASE,
        )
        out: Dict[str, float] = {}
        for m in pattern.finditer(stdout):
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        return out

    @staticmethod
    def _extract_any_values(stdout: str) -> Dict[str, float]:
        """
        Fallback: extract  name = value  or  name: value  pairs from stdout.
        """
        pattern = re.compile(
            r'([a-zA-Z_]\w*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        )
        out: Dict[str, float] = {}
        for m in pattern.finditer(stdout):
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        return out

    @staticmethod
    async def _run_code(code: str, timeout: int = 90) -> str:
        """Write code to a temp file and execute it; return stdout."""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False
            ) as f:
                f.write(code)
                tmp = f.name
            try:
                proc = subprocess.run(
                    ['python3', tmp],
                    capture_output=True, text=True, timeout=timeout
                )
                return proc.stdout
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    # ── public API ─────────────────────────────────────────────────────────

    @staticmethod
    async def verify_independent(
        problem: str,
        first_code: str,
        first_values: Dict[str, float],
        llm_coder_func,
        tolerance_pct: float = 5.0,
    ) -> MathVerificationResult:
        """
        Generate and execute an independent second solution, then compare.

        Args:
            problem:         Original problem statement
            first_code:      Python code from the primary solution (shown for
                             context only — NOT to be copied)
            first_values:    Computed values from the primary execution
            llm_coder_func:  Async callable(prompt, system_prompt) → str
                             pointing to the coder LLM (qwen2.5:14b)
            tolerance_pct:   Max % deviation to count as "agreement" (default 5 %)

        Returns:
            MathVerificationResult
        """
        # Build the list of primary results for the prompt
        results_block = "\n".join(
            f"  {k} = {v:.6g}" for k, v in first_values.items()
        )

        # Build a block of "given" values inferred from the first code
        given_block = ""
        given_pattern = re.compile(
            r'^\s*([a-zA-Z_]\w*)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:#.*)?$',
            re.MULTILINE
        )
        given_candidates: Dict[str, str] = {}
        for m in given_pattern.finditer(first_code):
            given_candidates[m.group(1)] = m.group(2)
        if given_candidates:
            given_block = "\n".join(
                f"  {k} = {v}" for k, v in list(given_candidates.items())[:30]
            )
        else:
            given_block = "  (derive from problem statement)"

        prompt = f"""A physics/engineering problem was solved. The first solution produced:

{results_block}

Your task: solve the SAME problem using a COMPLETELY DIFFERENT mathematical approach.

Guidelines:
- If the first approach used a direct formula (e.g. Tsiolkovsky rocket equation),
  verify with an alternative (e.g. numerical mass-flow integration or energy methods).
- If the first derived from first principles, try an established closed-form formula.
- Do NOT reference or copy the first solution's code structure.
- This is an independent cross-check meant to reach the same numbers by a different path.
- Inline all given values; import only math / numpy / sympy / scipy.
- For every final answer, print a machine-readable line:
    RESULT: variable_name = value unit
  Example:  RESULT: delta_v = 9450.23 m/s

GIVEN VALUES (from problem):
{given_block}

PROBLEM:
{problem[:600]}

Output ONLY Python code in ```python ... ``` fences. No prose."""

        system_prompt = (
            "You are an expert physicist and Python programmer. "
            "Respond ONLY with valid, complete Python code in ```python``` blocks. "
            "Use a DIFFERENT method than was already used. Never use {placeholder} syntax."
        )

        try:
            response = await llm_coder_func(
                prompt=prompt, system_prompt=system_prompt
            )
        except Exception as e:
            return MathVerificationResult(
                primary_values=first_values,
                secondary_values={},
                agreements={},
                discrepancies=[f"LLM call failed: {e}"],
                overall_agreement=False,
                tolerance_pct=tolerance_pct,
                note="Independent verification skipped — LLM error",
            )

        # Extract code block
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)

        if not code_match:
            return MathVerificationResult(
                primary_values=first_values,
                secondary_values={},
                agreements={},
                discrepancies=["Independent solver returned no valid Python code"],
                overall_agreement=False,
                tolerance_pct=tolerance_pct,
                note="No code block found in LLM response",
            )

        second_code = code_match.group(1).strip()

        # Execute the second script
        stdout = await MathVerifier._run_code(second_code, timeout=90)

        # Extract second values (prefer RESULT: lines; fall back to generic)
        second_values = MathVerifier._extract_result_lines(stdout)
        if not second_values:
            second_values = MathVerifier._extract_any_values(stdout)

        if not second_values:
            return MathVerificationResult(
                primary_values=first_values,
                secondary_values={},
                agreements={},
                discrepancies=["Independent script produced no parseable numeric output"],
                overall_agreement=False,
                tolerance_pct=tolerance_pct,
                note="Second script may have crashed or printed no results",
            )

        # Compare overlapping keys
        agreements: Dict[str, bool] = {}
        discrepancies: List[str] = []

        primary_keys = set(first_values.keys())
        secondary_keys = set(second_values.keys())

        # Exact name match first
        common_exact = primary_keys & secondary_keys

        # 6-char prefix fuzzy match for remaining keys
        unmatched_primary = primary_keys - common_exact
        unmatched_secondary = secondary_keys - common_exact
        fuzzy_pairs: List[tuple] = []
        for pk in unmatched_primary:
            for sk in unmatched_secondary:
                if pk[:6].lower() == sk[:6].lower():
                    fuzzy_pairs.append((pk, sk))

        # Build comparison set
        compare_pairs = [(k, k) for k in common_exact] + fuzzy_pairs

        for pk, sk in compare_pairs:
            pv = first_values[pk]
            sv = second_values[sk]

            if pv == 0 and sv == 0:
                agreements[pk] = True
                continue

            denom = max(abs(pv), abs(sv), 1e-30)
            pct_diff = abs(pv - sv) / denom * 100.0

            ok = pct_diff <= tolerance_pct
            agreements[pk] = ok

            if not ok:
                discrepancies.append(
                    f"{pk}: primary={pv:.6g}, secondary={sv:.6g}, "
                    f"diff={pct_diff:.1f}% (tolerance={tolerance_pct}%)"
                )

        overall = len(discrepancies) == 0 and len(compare_pairs) > 0

        note = (
            f"Compared {len(compare_pairs)} variable(s) "
            f"({len(common_exact)} exact + {len(fuzzy_pairs)} fuzzy matches)"
        )
        if not compare_pairs:
            note = "No overlapping variable names found between primary and secondary results"
            overall = False

        return MathVerificationResult(
            primary_values=first_values,
            secondary_values=second_values,
            agreements=agreements,
            discrepancies=discrepancies,
            overall_agreement=overall,
            tolerance_pct=tolerance_pct,
            note=note,
            secondary_code=second_code,
        )

    @staticmethod
    async def cross_check_with_search(
        computed_values: Dict[str, float],
        search_context: str,
        llm_func,
    ) -> Dict[str, Any]:
        """
        Ask the reasoning LLM to judge whether the computed numbers are
        consistent with facts gathered from web search.

        Args:
            computed_values:  var_name → float from execution
            search_context:   String of search-result text (build_math_context output)
            llm_func:         Async callable(prompt, system_prompt) → str
                              pointing to the reasoning LLM (phi4:14b)

        Returns:
            dict with keys:
              "plausible": bool
              "warnings":  list of strings (one per suspicious value)
              "notes":     overall comment string
        """
        if not computed_values:
            return {"plausible": True, "warnings": [], "notes": "No computed values to check"}

        values_block = "\n".join(
            f"  {k} = {v:.6g}" for k, v in computed_values.items()
        )
        context_trimmed = search_context[:2000] if search_context else "(no search context)"

        prompt = f"""You are a physics fact-checker. Assess whether the computed results below
are consistent with the research context provided.

COMPUTED NUMERICAL RESULTS:
{values_block}

RESEARCH CONTEXT (web search facts):
{context_trimmed}

Instructions:
- Check each computed value against any relevant stated fact or typical textbook value.
- Flag any result that directly contradicts a stated fact, is orders-of-magnitude wrong,
  or is physically implausible (e.g. Isp of 9000 s, orbital velocity of 200 m/s, etc.).
- If the research context says nothing relevant about a variable, do not flag it.

Respond ONLY with valid JSON, no prose:
{{"plausible": true_or_false, "warnings": ["...", "..."], "notes": "brief overall comment"}}"""

        system_prompt = (
            "You are a concise scientific fact-checker. "
            "Respond ONLY with a single valid JSON object. No markdown fences."
        )

        try:
            response = await llm_func(prompt=prompt, system_prompt=system_prompt)

            # Strip markdown fences if present
            cleaned = response.strip()
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

            result = json.loads(cleaned)
            # Ensure expected keys exist
            result.setdefault("plausible", True)
            result.setdefault("warnings", [])
            result.setdefault("notes", "")
            return result

        except json.JSONDecodeError:
            # Best-effort: look for true/false and extract warnings
            plausible = "false" not in response.lower()
            return {
                "plausible": plausible,
                "warnings": ["(could not parse LLM JSON response)"],
                "notes": response[:300],
            }
        except Exception as e:
            return {
                "plausible": True,
                "warnings": [],
                "notes": f"Cross-check skipped: {e}",
            }


    @staticmethod
    async def debug_reconcile(
        problem: str,
        primary_code: str,
        secondary_code: str,
        discrepancies: List[str],
        llm_coder_func,
        timeout: int = 90,
    ) -> Dict[str, Any]:
        """
        Given two disagreeing solutions, ask the LLM to diagnose the disagreement
        and produce a corrected third solution.

        Args:
            problem:          Original problem statement
            primary_code:     Python code of the first (primary) solution
            secondary_code:   Python code of the independent second solution
            discrepancies:    Human-readable discrepancy strings from verify_independent()
            llm_coder_func:   Async callable(prompt, system_prompt) → str (qwen2.5:14b)
            timeout:          Execution timeout in seconds for the corrected script

        Returns:
            {
                "success":   bool,
                "values":    Dict[str, float],   # parsed RESULT: lines
                "diagnosis": str,
                "code":      str,
            }
        """
        disc_block = "\n".join(f"  {d}" for d in discrepancies)

        prompt = f"""Two independent solutions to the same problem disagree:

DISCREPANCIES:
{disc_block}

PRIMARY CODE:
```python
{primary_code[:2000]}
```

SECONDARY CODE:
```python
{secondary_code[:2000]}
```

PROBLEM: {problem[:400]}

Your task:
1. Diagnose the disagreement (wrong sign convention, missing term, wrong formula variant,
   unit mismatch, wrong physical constant, etc.) — explain in 1–3 sentences.
2. Write a CORRECTED third solution that resolves the conflict.
   - Use the most physically correct approach.
   - Inline all values; import only math / numpy / sympy / scipy.
   - For every final answer, print:  RESULT: variable_name = value unit

Start your response with a one-line diagnosis (prefix it with DIAGNOSIS:), then output
the corrected code in ```python ... ``` fences. No other prose."""

        system_prompt = (
            "You are an expert physicist and Python programmer. "
            "Diagnose disagreements between two physics solutions and write a corrected third. "
            "Never use {placeholder} syntax — inline all values directly."
        )

        try:
            response = await llm_coder_func(prompt=prompt, system_prompt=system_prompt)
        except Exception as e:
            return {"success": False, "values": {}, "diagnosis": f"LLM call failed: {e}", "code": ""}

        # Extract diagnosis line
        diagnosis = ""
        for line in response.splitlines():
            if line.strip().upper().startswith("DIAGNOSIS:"):
                diagnosis = line.split(":", 1)[1].strip()
                break
        if not diagnosis:
            # Grab first non-empty line as fallback
            for line in response.splitlines():
                if line.strip():
                    diagnosis = line.strip()[:200]
                    break

        # Extract code block
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)\n```', response, re.DOTALL)

        if not code_match:
            return {
                "success": False,
                "values": {},
                "diagnosis": diagnosis or "No code block found in reconciler response",
                "code": "",
            }

        corrected_code = code_match.group(1).strip()

        # Execute the corrected script
        stdout = await MathVerifier._run_code(corrected_code, timeout=timeout)

        # Parse RESULT: lines
        values = MathVerifier._extract_result_lines(stdout)
        if not values:
            values = MathVerifier._extract_any_values(stdout)

        success = len(values) > 0

        return {
            "success": success,
            "values": values,
            "diagnosis": diagnosis,
            "code": corrected_code,
        }


if __name__ == "__main__":
    print("math_verifier.py — module OK")
    print("MathVerificationResult and MathVerifier are importable.")
    print("debug_reconcile:", hasattr(MathVerifier, "debug_reconcile"))
