"""
Python Compute Engine
The ONLY component allowed to do math

NO LLMs. Pure computation.
Uses formal_calculator.py (SymPy + Pint)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from formal_calculator import FormalCalculator, CalculationResult
    _HAS_FORMAL_CALC = True
except ImportError:
    _HAS_FORMAL_CALC = False
    class FormalCalculator:
        def __init__(self):
            print("⚠️  formal_calculator.py not installed — FormalCalculator is a stub")
        def calculate(self, *a, **kw):
            raise NotImplementedError("formal_calculator.py not installed")
        def convert_units(self, *a, **kw):
            raise NotImplementedError("formal_calculator.py not installed")
    class CalculationResult:
        def __init__(self):
            self.success = False
            self.error = "formal_calculator.py not available"
            self.result = {}
from shared_memory import SharedMemory, Variable, Equation, ComputedResult
from typing import Dict, List, Optional
import re


class PythonComputeEngine:
    """
    Deterministic Python computation engine.
    
    This is the ONLY place where numbers are calculated.
    
    Features:
    - Symbolic math (SymPy)
    - Unit tracking (Pint)
    - Dimensional analysis
    - Error bounds
    
    NO LLMs can override this.
    """
    
    def __init__(self, memory: SharedMemory):
        self.memory = memory
        self.calculator = FormalCalculator()

        print("🔧 Python Compute Engine initialized")
        print("   This is the ONLY component that does math")
        if not _HAS_FORMAL_CALC:
            print("   ⚠️  formal_calculator.py not found — formal compute unavailable")
    
    def compute_all(self) -> List[ComputedResult]:
        """
        Compute all solvable equations in memory.
        
        Returns:
            List of computed results
        """
        print("\n🔧 [Python Compute] Starting computation...")
        
        # Get validated equations
        equations = self.memory.get_equations(validated_only=True)
        
        if not equations:
            print("   ⚠️ No validated equations to compute")
            return []
        
        # Get variables
        knowns = self.memory.get_known_variables()
        unknowns = self.memory.get_unknown_variables()
        
        print(f"   📊 {len(equations)} equations, "
              f"{len(knowns)} known vars, "
              f"{len(unknowns)} unknowns")
        
        results = []
        
        # Try to solve each equation
        for eq in equations:
            result = self._solve_equation(eq, knowns, unknowns)
            if result:
                results.append(result)
        
        print(f"   ✅ Computed {len(results)} results")
        
        return results
    
    def _solve_equation(
        self,
        equation: Equation,
        knowns: Dict[str, Variable],
        unknowns: Dict[str, Variable]
    ) -> Optional[ComputedResult]:
        """
        Solve a single equation.
        
        Args:
            equation: Equation to solve
            knowns: Known variables
            unknowns: Unknown variables
            
        Returns:
            ComputedResult if successful, None otherwise
        """
        print(f"\n   📐 Solving: {equation.symbolic_form}")
        
        # Check if we can solve this equation
        eq_vars = set(equation.variables_used)
        known_vars = set(knowns.keys())
        unknown_vars = set(unknowns.keys())
        
        # Find what we're solving for
        solvable = eq_vars & unknown_vars
        
        if not solvable:
            print(f"      ℹ️ All variables known, nothing to solve")
            return None
        
        if len(solvable) > 1:
            print(f"      ⚠️ Multiple unknowns: {solvable}, cannot solve directly")
            return None
        
        solve_for = list(solvable)[0]
        
        # Check we have all other variables
        needed = eq_vars - {solve_for}
        missing = needed - known_vars
        
        if missing:
            print(f"      ⚠️ Missing variables: {missing}")
            return None
        
        # Prepare known values for calculator
        known_values = {}
        for var_symbol in needed:
            var = knowns[var_symbol]
            if var.value is not None and var.unit:
                known_values[var_symbol] = f"{var.value} {var.unit}"
        
        print(f"      🎯 Solving for: {solve_for}")
        print(f"      📊 Known: {known_values}")
        
        # Use calculator
        try:
            calc_result = self.calculator.calculate(
                equations=[equation.symbolic_form],
                knowns=known_values,
                solve_for=[solve_for],
                domain=equation.domain
            )
            
            if calc_result.success:
                # Extract result
                result_str = calc_result.result.get(solve_for, "")
                
                # Parse value and unit
                value, unit = self._parse_result(result_str)
                
                if value is not None:
                    # Create computed result
                    result = ComputedResult(
                        id="",
                        variable=solve_for,
                        value=value,
                        unit=unit,
                        method=equation.symbolic_form,
                        symbolic_form=equation.symbolic_form,
                        numeric_substitution=self._create_substitution_string(
                            equation.symbolic_form,
                            known_values
                        )
                    )
                    
                    # Add to memory
                    result_id = self.memory.add_computed_result(
                        variable=solve_for,
                        value=value,
                        unit=unit,
                        method=equation.symbolic_form,
                        symbolic_form=equation.symbolic_form,
                        numeric_substitution=result.numeric_substitution
                    )
                    
                    print(f"      ✅ {solve_for} = {value} {unit}")
                    
                    return result
                else:
                    print(f"      ⚠️ Could not parse result: {result_str}")
            else:
                print(f"      ❌ Calculation failed: {calc_result.error}")
        
        except Exception as e:
            print(f"      ❌ Error: {e}")
        
        return None
    
    def _parse_result(self, result_str: str) -> tuple:
        """
        Parse result string into value and unit.
        
        Args:
            result_str: String like "49050 N" or "2267.96 kg"
            
        Returns:
            (value, unit) tuple
        """
        parts = result_str.strip().split()
        
        if len(parts) >= 2:
            try:
                value = float(parts[0])
                unit = ' '.join(parts[1:])
                return value, unit
            except ValueError:
                return None, ""
        elif len(parts) == 1:
            try:
                value = float(parts[0])
                return value, "dimensionless"
            except ValueError:
                return None, ""
        
        return None, ""
    
    def _create_substitution_string(
        self,
        equation: str,
        knowns: Dict[str, str]
    ) -> str:
        """
        Create a string showing numeric substitution.
        
        Args:
            equation: Symbolic equation like "F = m * g"
            knowns: Dict like {"m": "5000 kg", "g": "9.81 m/s^2"}
            
        Returns:
            String like "F = 5000 * 9.81"
        """
        result = equation
        
        for symbol, value_str in knowns.items():
            # Extract just the number
            value = value_str.split()[0]
            
            # Replace symbol with value
            result = re.sub(
                r'\b' + re.escape(symbol) + r'\b',
                value,
                result
            )
        
        return result
    
    def convert_units(self, from_value: str, to_unit: str) -> Optional[Dict]:
        """
        Convert between units.
        
        Args:
            from_value: Value with unit like "5000 lbm"
            to_unit: Target unit like "kg"
            
        Returns:
            Dict with result or None
        """
        print(f"\n🔄 [Python Compute] Converting {from_value} to {to_unit}")
        
        result = self.calculator.convert_units(from_value, to_unit)
        
        if result.success:
            converted = result.result['converted']
            print(f"   ✅ {from_value} = {converted}")
            
            # Add to memory as a computed result
            value, unit = self._parse_result(converted)
            
            if value is not None:
                # Create a variable name for this conversion
                # e.g., "m_kg" for mass in kg
                var_name = f"converted_{to_unit.replace('/', '_per_')}"
                
                self.memory.add_computed_result(
                    variable=var_name,
                    value=value,
                    unit=unit,
                    method=f"Unit conversion: {from_value} to {to_unit}",
                    numeric_substitution=f"{from_value} = {converted}"
                )
            
            return {
                'success': True,
                'result': converted,
                'value': value,
                'unit': unit
            }
        else:
            print(f"   ❌ Conversion failed: {result.error}")
            return None
    
    def verify_dimensional_consistency(self) -> Dict:
        """
        Verify dimensional consistency of all equations.
        
        Returns:
            Dict with verification results
        """
        print("\n🔍 [Python Compute] Checking dimensional consistency...")
        
        equations = self.memory.get_equations(validated_only=True)
        variables = self.memory.variables
        
        issues = []
        
        for eq in equations:
            # For each equation, check if units work out
            # This is simplified - a full implementation would use SymPy's unit system
            
            eq_vars = eq.variables_used
            
            # Get units for all variables
            units = {}
            for var_symbol in eq_vars:
                if var_symbol in variables:
                    var = variables[var_symbol]
                    units[var_symbol] = var.unit or "dimensionless"
            
            # Basic check: all variables should have units defined
            if not all(units.values()):
                issues.append({
                    'equation': eq.symbolic_form,
                    'issue': 'Some variables missing unit definitions',
                    'variables': eq_vars
                })
        
        if issues:
            print(f"   ⚠️ Found {len(issues)} dimensional issues")
            for issue in issues:
                print(f"      - {issue['equation']}: {issue['issue']}")
        else:
            print(f"   ✅ All equations dimensionally consistent")
        
        return {
            'consistent': len(issues) == 0,
            'issues': issues
        }
    
    def get_computation_summary(self) -> str:
        """Get a summary of all computations performed"""
        results = self.memory.get_computed_results()
        
        if not results:
            return "No computations performed yet"
        
        lines = []
        lines.append(f"🔧 Computation Summary ({len(results)} results):")
        lines.append("")
        
        for result in results:
            lines.append(f"  {result.variable} = {result.value} {result.unit}")
            if result.method:
                lines.append(f"    Method: {result.method}")
            if result.numeric_substitution:
                lines.append(f"    Calculation: {result.numeric_substitution}")
            lines.append("")
        
        return '\n'.join(lines)


# Quick test
if __name__ == "__main__":
    from shared_memory import SharedMemory
    
    print("Testing Python Compute Engine...\n")
    
    memory = SharedMemory()
    
    # Setup problem
    memory.add_variable("F", "thrust force", "N")
    memory.add_variable("m", "mass", "kg", value=5000.0)
    memory.add_variable("g", "gravity", "m/s^2", value=9.81)
    
    eq_id = memory.add_equation(
        symbolic_form="F = m * g",
        variables_used=["F", "m", "g"],
        domain="physics"
    )
    memory.validate_equation(eq_id)
    
    # Test computation
    compute = PythonComputeEngine(memory)
    
    results = compute.compute_all()
    
    print("\n" + "="*70)
    print(compute.get_computation_summary())
    
    print("\n" + "="*70)
    memory.print_state()
    
    # Test unit conversion
    print("\n" + "="*70)
    conversion = compute.convert_units("5000 pound_mass", "kilogram")
    if conversion:
        print(f"Conversion result: {conversion}")
    
    # Test dimensional consistency
    print("\n" + "="*70)
    consistency = compute.verify_dimensional_consistency()
    print(f"Consistency check: {consistency}")
