"""
material_props.py — Temperature/pressure-dependent material properties.

Provides:
  get_fluid_property(fluid, prop, T_K, P_Pa) → float | None
  get_element_property(symbol, prop) → float | None
  list_available_fluids() → List[str]

Falls back gracefully if CoolProp or mendeleev are not installed.
"""

from typing import List, Optional


# ── CoolProp availability ────────────────────────────────────────────────────

try:
    from CoolProp.CoolProp import PropsSI, FluidsList  # type: ignore
    _HAS_COOLPROP = True
except ImportError:
    _HAS_COOLPROP = False


# ── mendeleev availability ───────────────────────────────────────────────────

try:
    import mendeleev as _mendeleev  # type: ignore
    _HAS_MENDELEEV = True
except ImportError:
    _HAS_MENDELEEV = False


# ── Public API ───────────────────────────────────────────────────────────────

def get_fluid_property(
    fluid: str,
    prop: str,
    T_K: float,
    P_Pa: float,
) -> Optional[float]:
    """
    Query CoolProp for a fluid thermodynamic property.

    Args:
        fluid: e.g. "Water", "Air", "R134a", "CO2", "Helium", "Nitrogen"
        prop:  CoolProp output key string:
                 "H"  — specific enthalpy  (J/kg)
                 "D"  — density            (kg/m³)
                 "C"  — isobaric heat cap. (J/kg/K)   [Cp]
                 "V"  — dynamic viscosity  (Pa·s)
                 "L"  — thermal cond.      (W/m/K)
                 "Q"  — vapour quality     (-)
                 "S"  — specific entropy   (J/kg/K)
                 "T"  — temperature        (K)  [useful for saturation queries]
                 "P"  — pressure           (Pa)
        T_K:   Temperature in Kelvin
        P_Pa:  Pressure in Pascals

    Returns:
        Float value, or None if CoolProp unavailable or lookup fails.
    """
    if not _HAS_COOLPROP:
        return None
    try:
        return float(PropsSI(prop, "T", T_K, "P", P_Pa, fluid))
    except Exception:
        return None


def get_element_property(
    symbol: str,
    prop: str,
) -> Optional[float]:
    """
    Query mendeleev for a periodic-table element property.

    Args:
        symbol: Element symbol, e.g. "Al", "Fe", "Ti", "Cu", "C"
        prop:   Attribute name on the mendeleev Element object:
                  "atomic_weight"         — g/mol
                  "density"               — g/cm³ (at STP)
                  "melting_point"         — K
                  "boiling_point"         — K
                  "thermal_conductivity"  — W/(m·K)
                  "specific_heat"         — J/(g·K)
                  "heat_of_fusion"        — kJ/mol
                  "heat_of_vaporization"  — kJ/mol
                  "electrical_resistivity"— Ω·m (×10⁻⁸ stored in mendeleev)
                  "youngs_modulus"        — GPa

    Returns:
        Float value of the requested property, or None if unavailable.
    """
    if not _HAS_MENDELEEV:
        return None
    try:
        el = _mendeleev.element(symbol)
        val = getattr(el, prop, None)
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def list_available_fluids() -> List[str]:
    """
    Return the list of CoolProp fluid names.
    Returns an empty list if CoolProp is not installed.
    """
    if not _HAS_COOLPROP:
        return []
    try:
        return FluidsList()
    except Exception:
        return []


# ── Module status ─────────────────────────────────────────────────────────────

def _status() -> str:
    parts = []
    parts.append(f"CoolProp: {'available' if _HAS_COOLPROP else 'NOT installed'}")
    parts.append(f"mendeleev: {'available' if _HAS_MENDELEEV else 'NOT installed'}")
    return ", ".join(parts)


if __name__ == "__main__":
    print("material_props.py —", _status())

    # Quick smoke test
    d = get_fluid_property("Water", "D", 373.15, 101325.0)
    if d is not None:
        print(f"  Water density at 100°C, 1 atm: {d:.3f} kg/m³  (expect ~958)")

    rho_al = get_element_property("Al", "density")
    if rho_al is not None:
        print(f"  Aluminium density (mendeleev): {rho_al} g/cm³  (expect 2.7)")
