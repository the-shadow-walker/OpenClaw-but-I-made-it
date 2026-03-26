"""
Formal Calculator — Unit parsing, dimensional consistency, and conversion.

Provides:
  - FormalCalculator.parse_output_units()   — extract RESULT: lines from stdout
  - FormalCalculator.check_dimensional_consistency() — flag wrong-domain units
  - FormalCalculator.convert_units()         — pint-based unit conversion
  - CalculationResult dataclass

The equation_generator is expected to emit machine-readable lines:
    RESULT: variable_name = value unit
e.g.
    RESULT: delta_v = 9450.23 m/s
    RESULT: propellant_mass = 48300.0 kg
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CalculationResult:
    success: bool
    result: float = 0.0
    unit: str = ""
    error: str = ""


# Maps a "domain keyword" → accepted unit strings for that physical quantity.
# Used by check_dimensional_consistency() to catch obvious mismatches.
UNIT_DOMAINS: Dict[str, List[str]] = {
    "velocity":             ["m/s", "km/s", "ft/s", "mph", "knot", "kph", "km/h"],
    "speed":                ["m/s", "km/s", "ft/s", "mph", "knot", "kph", "km/h"],
    "mass":                 ["kg", "g", "mg", "lb", "ton", "tonne", "slug"],
    "pressure":             ["Pa", "kPa", "MPa", "GPa", "psi", "bar", "atm", "mmHg", "torr"],
    "temperature":          ["K", "°C", "°F", "degC", "degF", "degK", "Celsius", "Kelvin"],
    "energy":               ["J", "kJ", "MJ", "GJ", "Wh", "kWh", "cal", "kcal", "BTU", "eV"],
    "power":                ["W", "kW", "MW", "GW", "hp", "BTU/s"],
    "force":                ["N", "kN", "MN", "lbf", "dyn"],
    "length":               ["m", "km", "cm", "mm", "ft", "in", "mile", "nm", "AU", "ly"],
    "time":                 ["s", "ms", "min", "hr", "h", "day", "year"],
    "frequency":            ["Hz", "kHz", "MHz", "GHz", "rpm", "rad/s"],
    "acceleration":         ["m/s²", "m/s^2", "g", "ft/s^2", "ft/s²"],
    "density":              ["kg/m³", "kg/m^3", "g/cm³", "g/cm^3", "kg/L", "lb/ft^3"],
    "isp":                  ["s"],
    "exhaust_velocity":     ["m/s", "km/s"],
    "delta_v":              ["m/s", "km/s", "ft/s"],
    "thrust":               ["N", "kN", "MN", "lbf"],
    "efficiency":           ["%", "fraction", "dimensionless", ""],
}

# Pairs that are definitively wrong (velocity measured in pressure units, etc.)
OBVIOUSLY_WRONG: List[Tuple[str, str, str]] = [
    # (domain_keyword_in_var_name, bad_unit_fragment, human_message)
    ("velocity",    "Pa",   "velocity expressed in pressure units (Pa)"),
    ("velocity",    "bar",  "velocity expressed in pressure units (bar)"),
    ("mass",        "m/s",  "mass expressed in velocity units (m/s)"),
    ("mass",        "Pa",   "mass expressed in pressure units (Pa)"),
    ("pressure",    "m/s",  "pressure expressed in velocity units (m/s)"),
    ("pressure",    "kg",   "pressure expressed in mass units (kg)"),
    ("temperature", "m/s",  "temperature expressed in velocity units (m/s)"),
    ("temperature", "kg",   "temperature expressed in mass units (kg)"),
    ("isp",         "m/s",  "Isp should be in seconds, not m/s"),
    ("isp",         "kg",   "Isp should be in seconds, not kg"),
]

# RESULT line regex — applied line-by-line to avoid cross-line unit captures.
# Uses [^\S\n]* (horizontal whitespace only) before the unit to prevent
# consuming the newline and capturing the next line's content.
# Captures: (var_name, numeric_value, optional_rest_of_line_for_unit)
_RESULT_RE = re.compile(
    r'^RESULT:[^\S\n]*(\w+)[^\S\n]*=[^\S\n]*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[^\S\n]*(.*)$',
    re.IGNORECASE | re.MULTILINE
)


class FormalCalculator:
    """
    Static utility class for unit-aware result handling.
    """

    @staticmethod
    def parse_output_units(stdout: str) -> Dict[str, Tuple[float, str]]:
        """
        Parse all  RESULT: var_name = value unit  lines from script stdout.

        Returns:
            dict mapping variable name → (float_value, unit_string)
            e.g. {"delta_v": (9450.23, "m/s"), "mass_ratio": (4.5, "")}
        """
        results: Dict[str, Tuple[float, str]] = {}
        for match in _RESULT_RE.finditer(stdout):
            var_name = match.group(1)
            try:
                value = float(match.group(2))
            except (ValueError, TypeError):
                continue
            # The third group is everything after the number on the same line.
            # Strip inline comments (# ...) and leading/trailing whitespace.
            raw_unit = (match.group(3) or "").strip()
            raw_unit = re.sub(r'\s*#.*$', '', raw_unit).strip()
            results[var_name] = (value, raw_unit)
        return results

    @staticmethod
    def check_dimensional_consistency(
        results_with_units: Dict[str, Tuple[float, str]]
    ) -> List[str]:
        """
        Flag variables whose reported unit is clearly wrong for the domain
        implied by the variable name.

        Args:
            results_with_units: output of parse_output_units()

        Returns:
            list of human-readable warning strings (empty = no issues)
        """
        warnings: List[str] = []
        for var_name, (value, unit) in results_with_units.items():
            if not unit:
                continue
            var_lower = var_name.lower()
            for domain_kw, bad_unit_frag, message in OBVIOUSLY_WRONG:
                if domain_kw in var_lower and bad_unit_frag in unit:
                    warnings.append(
                        f"{var_name} = {value} {unit} — {message}"
                    )
                    break  # one warning per variable is enough
        return warnings

    @staticmethod
    def convert_units(value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Convert a value between units using pint.

        Returns converted float, or None if pint is unavailable or
        the conversion fails (e.g. incompatible dimensions).
        """
        try:
            import pint
            ureg = pint.UnitRegistry()
            qty = value * ureg(from_unit)
            return qty.to(to_unit).magnitude
        except Exception:
            return None

    @staticmethod
    def summarize(results_with_units: Dict[str, Tuple[float, str]]) -> str:
        """Format parsed results as a readable block for logging."""
        if not results_with_units:
            return "  (no RESULT: lines found)"
        lines = []
        for var, (val, unit) in results_with_units.items():
            unit_str = f" {unit}" if unit else ""
            lines.append(f"  {var} = {val:.6g}{unit_str}")
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke tests
    print("formal_calculator.py — smoke tests")
    print("=" * 50)

    sample_stdout = (
        "Step 1: computing mass ratio...\n"
        "RESULT: mass_ratio = 4.5\n"
        "Step 2: delta-v from Tsiolkovsky...\n"
        "RESULT: delta_v = 9450.23 m/s\n"
        "RESULT: propellant_mass = 48320.0 kg\n"
        "RESULT: isp = 363.0 s\n"
    )

    parsed = FormalCalculator.parse_output_units(sample_stdout)
    print("Parsed results:")
    for k, v in parsed.items():
        print(f"  {k}: value={v[0]}, unit={v[1]!r}")

    issues = FormalCalculator.check_dimensional_consistency(parsed)
    print(f"\nDimensional issues: {issues or 'none'}")

    # Bad units
    bad = {"velocity": (9450.0, "Pa"), "isp": (363.0, "m/s")}
    bad_parsed = {k: v for k, v in bad.items()}
    issues2 = FormalCalculator.check_dimensional_consistency(bad_parsed)
    print(f"Bad-unit issues: {issues2}")

    # Unit conversion
    val_km = FormalCalculator.convert_units(9450.23, "m/s", "km/s")
    print(f"\n9450.23 m/s → km/s: {val_km}")
