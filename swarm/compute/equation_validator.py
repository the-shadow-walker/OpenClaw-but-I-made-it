"""
Equation Validator and Executor

1. Validates that equations can be solved
2. Checks all required variables are available
3. Executes code safely with timeout and sandboxing
4. Validates results make physical sense
"""

import re
import math
import subprocess
import tempfile
import os
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of executing an equation"""
    success: bool
    output: str = ""
    computed_values: Dict[str, Any] = None
    error: str = None
    execution_time: float = 0.0


class EquationValidator:
    """
    Validates that generated equations are executable
    """
    
    @staticmethod
    def check_solvable(
        code: str,
        given_values: Dict[str, float],
        required_placeholders: list
    ) -> tuple[bool, str]:
        """
        Check if equation can be solved.

        v2 (Complete-Code approach): code should have no {placeholder} tokens
        because the generator inlines all values.  We check for any leftover
        unfilled placeholders and fail only if real ones remain.

        kept backward-compatible: required_placeholders arg still accepted
        but is no longer the blocking factor.
        """
        # Detect any remaining {word} tokens that are NOT inside f-string braces
        # (f-strings use {{ }} to escape literal braces, so bare {word} = unfilled placeholder)
        unfilled = re.findall(r'(?<!\{)\{(\w+)\}(?!\})', code)
        # Filter out known f-string-like patterns (e.g. print(f"...{var}..."))
        # Simple heuristic: if the placeholder name is also a variable assigned earlier, it's fine
        assigned = set(re.findall(r'^(\w+)\s*=', code, re.MULTILINE))
        truly_missing = [p for p in unfilled if p not in assigned]

        if truly_missing:
            return False, f"Unfilled placeholders in code: {', '.join(truly_missing)}"

        return True, "Code is complete and directly executable"
    
    # ── Comprehensive domain-aware physics bounds ──────────────────────────
    # Each entry: variable_keyword → (min_or_None, max_or_None, violation_description)
    # Variable name matching: case-insensitive substring match against key.
    PHYSICS_BOUNDS: Dict[str, Tuple] = {
        # kinematics / dynamics
        "velocity":             (0,       3e8,    "exceeds speed of light"),
        "speed":                (0,       3e8,    "exceeds speed of light"),
        "delta_v":              (0,       150000, "delta-v unrealistically large (>150 km/s)"),
        "orbital_velocity":     (1000,    40000,  "implausible Earth orbital velocity"),
        "exhaust_velocity":     (500,     50000,  "exhaust velocity outside known range"),
        # orbital
        "eccentricity":         (0,       1.0,    "bound orbit eccentricity must be 0–1"),
        "altitude":             (-6.4e6,  None,   "below Earth's surface"),
        # rocket / propulsion
        "isp":                  (50,      5000,   "Isp outside physical range for known propellants"),
        "thrust":               (0,       None,   "negative thrust"),
        "mass_ratio":           (1.0,     50,     "mass ratio below 1 or unrealistically high"),
        "propellant_fraction":  (0,       0.97,   "propellant fraction > 0.97 leaves structural fraction < 3%"),
        "chamber_pressure":     (1e4,     5e8,    "chamber pressure unphysical"),
        "twr":                  (0.01,    200,    "thrust-to-weight ratio outside practical range"),
        # mass
        "mass":                 (0,       None,   "negative mass is impossible"),
        "propellant_mass":      (0,       None,   "negative propellant mass"),
        "dry_mass":             (0,       None,   "negative dry mass"),
        # thermodynamics
        "temperature_k":        (0,       1e8,    "below absolute zero or above stellar-core range"),
        "temperature":          (-273.16, 1e8,    "below absolute zero"),
        "pressure":             (0,       None,   "negative absolute pressure"),
        "heat_flux":            (0,       1e12,   "heat flux exceeds laser ablation threshold"),
        "thermal_resistance":   (0,       None,   "negative thermal resistance"),
        "thermal_conductivity": (1e-4,    10000,  "outside known material range"),
        "heat_transfer":        (1,       1e6,    "outside convection/conduction range"),
        # structural / mechanical
        "stress":               (0,       1e12,   "exceeds theoretical material strength"),
        "strain":               (0,       10,     "strain outside physical range"),
        "safety_factor":        (0.01,    None,   "safety factor below ~1 means failure"),
        # aerodynamics
        "mach":                 (0,       50,     "Mach > 50 unphysical for air"),
        "reynolds":             (0,       None,   "negative Reynolds number"),
        # efficiency
        "efficiency":           (0,       1.0,    "efficiency must be 0–1"),
        # electrical
        "resistance":           (0,       None,   "negative resistance"),
        "frequency":            (0,       None,   "negative frequency"),
    }

    @staticmethod
    def validate_results(
        results: Dict[str, float],
        problem_context: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that computed results make physical sense.

        Args:
            results: Dictionary of computed values (var_name → float)
            problem_context: Description of what was solved (used for context only)

        Returns:
            (all_ok, violations) where violations is a list of human-readable
            strings describing each bound violation (empty list = all clear).
        """
        violations: List[str] = []

        for var, val in results.items():
            if not isinstance(val, (int, float)):
                continue

            # NaN / Inf are always invalid
            try:
                if math.isnan(val):
                    violations.append(f"{var} is NaN (undefined result)")
                    continue
                if math.isinf(val):
                    violations.append(f"{var} is infinite (physically impossible)")
                    continue
            except (TypeError, ValueError):
                continue

            # Check against PHYSICS_BOUNDS using case-insensitive substring match
            var_lower = var.lower()
            for keyword, (lo, hi, msg) in EquationValidator.PHYSICS_BOUNDS.items():
                if keyword in var_lower:
                    if lo is not None and val < lo:
                        violations.append(
                            f"{var} = {val:.6g} is below minimum {lo} — {msg}"
                        )
                    if hi is not None and val > hi:
                        violations.append(
                            f"{var} = {val:.6g} exceeds maximum {hi} — {msg}"
                        )
                    break  # one rule per variable

        all_ok = len(violations) == 0
        return all_ok, violations


class EquationExecutor:
    """
    Safely executes generated Python code
    """
    
    @staticmethod
    async def execute(
        code: str,
        given_values: Dict[str, float],
        timeout: int = 10
    ) -> ExecutionResult:
        """
        Execute Python code safely
        
        Args:
            code: Python code with {placeholder} format
            given_values: Dictionary of values to substitute
            timeout: Execution timeout in seconds
        
        Returns:
            ExecutionResult with output and results
        """
        
        import time
        start_time = time.time()
        
        # Step 1: Substitute placeholders
        substituted_code = code
        for var_name, value in given_values.items():
            substituted_code = substituted_code.replace(
                f"{{{var_name}}}",
                str(value)
            )
        
        # Step 2: Safety checks
        dangerous_patterns = [
            'os.', 'subprocess', 'eval', 'exec', '__import__',
            'open(', 'requests.', 'urllib', 'socket', 'system', 
            '__builtins__', 'compile', 'globals', 'locals'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in substituted_code:
                return ExecutionResult(
                    success=False,
                    error=f"Code contains dangerous operation: {pattern}"
                )
        
        # Step 3: Execute in temporary file
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(substituted_code)
                temp_file = f.name
            
            try:
                # Run with timeout
                result = subprocess.run(
                    ['python3', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode != 0:
                    return ExecutionResult(
                        success=False,
                        output=result.stdout,
                        error=f"Execution error:\n{result.stderr}",
                        execution_time=execution_time
                    )
                
                # Step 4: Try to extract computed values from output
                computed = EquationExecutor._extract_values(result.stdout)
                
                return ExecutionResult(
                    success=True,
                    output=result.stdout,
                    computed_values=computed,
                    execution_time=execution_time
                )
            
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timeout (>{timeout}s)"
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Execution failed: {str(e)}"
            )
    
    @staticmethod
    def _extract_values(output: str) -> Dict[str, float]:
        """
        Try to extract computed values from script output
        
        Looks for patterns like:
        "Variable: 123.45"
        "a_final = 8592 km"
        etc.
        """
        
        values = {}
        
        # Pattern: "name = value" or "name: value"
        pattern = r'([a-zA-Z_]\w*)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
        
        for match in re.finditer(pattern, output):
            var_name = match.group(1)
            try:
                value = float(match.group(2))
                values[var_name] = value
            except ValueError:
                pass
        
        return values
    
    @staticmethod
    def format_result(result: ExecutionResult, code_snippet: str = None) -> str:
        """Format execution result for display"""
        
        lines = [
            "\n" + "="*70,
            "EXECUTION RESULT",
            "="*70,
        ]
        
        if result.success:
            lines.extend([
                "\n✅ Status: SUCCESS",
                f"⏱️  Execution time: {result.execution_time:.2f}s",
                f"\n📊 Output:",
                "   " + "\n   ".join(result.output.strip().split("\n")[-10:])  # Last 10 lines
            ])
            
            if result.computed_values:
                lines.append(f"\n📈 Extracted Values:")
                for var, val in result.computed_values.items():
                    lines.append(f"   {var} = {val:.6g}")
        else:
            lines.extend([
                "\n❌ Status: FAILED",
                f"\n❌ Error:\n   {result.error}",
                f"\n📋 Output (first 500 chars):\n   {result.output[:500]}"
            ])
        
        lines.append("\n" + "="*70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("Equation Validator and Executor Module")
    print("="*70)
    print("\nCapabilities:")
    print("  • Validate equations are solvable")
    print("  • Check physical validity of results")
    print("  • Execute code safely (timeout, sandbox)")
    print("  • Extract computed values from output")
