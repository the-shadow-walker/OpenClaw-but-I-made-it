"""
Enhanced Value Extractor

Properly extracts physics values like:
- "5000 kg" → mass = 5000
- "150 m/s" → velocity = 150
- "μ = 3.986e14" → mu = 3.986e14
- "40 degrees" → angle = 40
"""

import re
from typing import Dict, Tuple


class ValueExtractor:
    """
    Extract numeric values and their meanings from natural language
    """

    # Common physics patterns (ordered most-specific → least-specific)
    # Within each category: compound/longer units before shorter to prevent partial matches.
    PATTERNS = [
        # ── Gravitational / orbital ──────────────────────────────────────────────
        # gravitational parameter: "μ=3.986e14" or "mu = 3.986e14"
        (r'(?:u|mu)\s*[=:]\s*(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)', 'mu'),
        (r'(\d+(?:\.\d+)?[eE][-+]?\d+)\s*m[³3]/s[²2]', 'mu'),

        # ── Specific impulse / exhaust ────────────────────────────────────────────
        # "Isp = 350 s" or "Isp: 350 s"
        (r'(?:Isp|specific[_ ]impulse)\s*[=:]\s*(\d+(?:\.\d+)?)', 'isp'),
        (r'(?:Isp|specific impulse)\s*=?\s*(\d+(?:\.\d+)?)\s*s', 'isp'),
        (r'(\d+(?:\.\d+)?)\s*s\s+(?:SL|vac|vacuum|sea.level)', 'isp'),
        # exhaust velocity: "3000 m/s exhaust" — before plain m/s
        (r'(\d+(?:\.\d+)?)\s*m/s\s+(?:exhaust|exhaust velocity)', 'exhaust_velocity'),

        # ── Angular velocity ──────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*rad/s', 'angular_velocity'),
        (r'(\d+(?:\.\d+)?)\s*rpm\b', 'rpm'),
        (r'(\d+(?:\.\d+)?)\s*rev/s\b', 'angular_velocity_revs'),

        # ── Velocity / speed ──────────────────────────────────────────────────────
        # Compound (longer) units first
        (r'(\d+(?:\.\d+)?)\s*km/s', 'velocity_kms'),
        (r'(\d+(?:\.\d+)?)\s*ft/s\b', 'velocity_fts'),
        (r'(\d+(?:\.\d+)?)\s*mph\b', 'velocity_mph'),
        (r'(\d+(?:\.\d+)?)\s*km/h\b', 'velocity_kmh'),
        (r'(\d+(?:\.\d+)?)\s*knots?\b', 'velocity_knots'),
        (r'(\d+(?:\.\d+)?)\s*Mach\b', 'velocity_mach'),
        (r'(\d+(?:\.\d+)?)\s*m/s', 'velocity'),

        # ── Acceleration ──────────────────────────────────────────────────────────
        # m/s² before m/s; ft/s²; G-load
        (r'(\d+(?:\.\d+)?)\s*m/s[²2]', 'acceleration'),
        (r'(\d+(?:\.\d+)?)\s*ft/s[²2]', 'acceleration_fts2'),
        (r'(\d+(?:\.\d+)?)\s*[Gg]-?load\b', 'gload'),

        # ── Frequency ────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*THz\b', 'frequency_thz'),
        (r'(\d+(?:\.\d+)?)\s*GHz\b', 'frequency_ghz'),
        (r'(\d+(?:\.\d+)?)\s*MHz\b', 'frequency_mhz'),
        (r'(\d+(?:\.\d+)?)\s*kHz\b', 'frequency_khz'),
        (r'(\d+(?:\.\d+)?)\s*Hz\b', 'frequency_hz'),

        # ── Energy ────────────────────────────────────────────────────────────────
        # Specific energy before plain energy; larger units before smaller
        (r'(\d+(?:\.\d+)?)\s*kJ/kg\b', 'specific_energy_kjkg'),
        (r'(\d+(?:\.\d+)?)\s*J/kg\b', 'specific_energy_jkg'),
        (r'(\d+(?:\.\d+)?)\s*J/kgK\b', 'specific_heat'),
        (r'(\d+(?:\.\d+)?)\s*kWh\b', 'energy_kwh'),
        (r'(\d+(?:\.\d+)?)\s*Wh\b', 'energy_wh'),
        (r'(\d+(?:\.\d+)?)\s*GJ\b', 'energy_gj'),
        (r'(\d+(?:\.\d+)?)\s*MJ\b', 'energy_mj'),
        (r'(\d+(?:\.\d+)?)\s*kJ\b', 'energy_kj'),
        (r'(\d+(?:\.\d+)?)\s*MeV\b', 'energy_mev'),
        (r'(\d+(?:\.\d+)?)\s*keV\b', 'energy_kev'),
        (r'(\d+(?:\.\d+)?)\s*eV\b', 'energy_ev'),
        (r'(\d+(?:\.\d+)?)\s*kcal\b', 'energy_kcal'),
        (r'(\d+(?:\.\d+)?)\s*cal\b', 'energy_cal'),
        (r'(\d+(?:\.\d+)?)\s*BTU\b', 'energy_btu'),
        (r'(\d+(?:\.\d+)?)\s*J(?![/a-zA-Z])', 'energy_j'),

        # ── Power ────────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*GW\b', 'power_gw'),
        (r'(\d+(?:\.\d+)?)\s*MW\b', 'power_mw'),
        (r'(\d+(?:\.\d+)?)\s*kW\b', 'power_kw'),
        (r'(\d+(?:\.\d+)?)\s*mW\b', 'power_mw_milli'),
        (r'(\d+(?:\.\d+)?)\s*W(?![/a-zA-Z\d])', 'power_w'),
        (r'(\d+(?:\.\d+)?)\s*hp\b', 'power_hp'),

        # ── Force ────────────────────────────────────────────────────────────────
        # N·m torque must come before N force — handled in torque section below
        (r'(\d+(?:\.\d+)?)\s*MN\b', 'force_mn'),
        (r'(\d+(?:\.\d+)?)\s*kN\b', 'force_kn'),
        (r'(\d+(?:\.\d+)?)\s*lbf\b', 'force_lbf'),
        # Plain N — negative lookahead prevents matching N·m or N/m etc.
        (r'(\d+(?:\.\d+)?)\s*N(?![·.\-/a-zA-Z\d])', 'force_n'),

        # ── Torque ───────────────────────────────────────────────────────────────
        # N·m / Nm  (dot or middle-dot or none)
        (r'(\d+(?:\.\d+)?)\s*N[·\u00b7]?m\b', 'torque_nm'),
        (r'(\d+(?:\.\d+)?)\s*lb[·\u00b7]?ft\b', 'torque_lbft'),
        (r'(\d+(?:\.\d+)?)\s*oz[·\u00b7]?in\b', 'torque_ozin'),

        # ── Pressure ─────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*GPa\b', 'pressure_gpa'),
        (r'(\d+(?:\.\d+)?)\s*MPa\b', 'pressure_mpa'),
        (r'(\d+(?:\.\d+)?)\s*kPa\b', 'pressure_kpa'),
        (r'(\d+(?:\.\d+)?)\s*mbar\b', 'pressure_mbar'),
        (r'(\d+(?:\.\d+)?)\s*(?:atm|bar)\b', 'pressure'),
        (r'(\d+(?:\.\d+)?)\s*psi\b', 'pressure_psi'),
        (r'(\d+(?:\.\d+)?)\s*(?:mmHg|torr)\b', 'pressure_mmhg'),
        (r'(\d+(?:\.\d+)?)\s*(?:Pa|pascal)\b', 'pressure_pa'),

        # ── Electrical — voltage ─────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*kV\b', 'voltage_kv'),
        (r'(\d+(?:\.\d+)?)\s*mV\b', 'voltage_mv'),
        (r'(\d+(?:\.\d+)?)\s*V(?![a-zA-Z\d])', 'voltage_v'),

        # ── Electrical — current ─────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*kA\b', 'current_ka'),
        (r'(\d+(?:\.\d+)?)\s*mA\b', 'current_ma'),
        (r'(\d+(?:\.\d+)?)\s*uA\b', 'current_ua'),
        # Plain A — word boundary on both sides; lookbehind prevents "123A" in part numbers
        (r'(?<!\w)(\d+(?:\.\d+)?)\s*A\b', 'current_a'),

        # ── Electrical — resistance ───────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*M[Ω\u03a9]\b', 'resistance_mohm'),
        (r'(\d+(?:\.\d+)?)\s*k[Ω\u03a9]\b', 'resistance_kohm'),
        (r'(\d+(?:\.\d+)?)\s*[Ω\u03a9]\b', 'resistance_ohm'),
        (r'(\d+(?:\.\d+)?)\s*(?:ohm|ohms)\b', 'resistance_ohm'),

        # ── Electrical — charge / capacitance / inductance ───────────────────────
        (r'(\d+(?:\.\d+)?)\s*mAh\b', 'charge_mah'),
        (r'(\d+(?:\.\d+)?)\s*Ah\b', 'charge_ah'),
        (r'(\d+(?:\.\d+)?)\s*uF\b', 'capacitance_uf'),
        (r'(\d+(?:\.\d+)?)\s*nF\b', 'capacitance_nf'),
        (r'(\d+(?:\.\d+)?)\s*pF\b', 'capacitance_pf'),
        (r'(\d+(?:\.\d+)?)\s*F(?![a-zA-Z\d])', 'capacitance_f'),
        (r'(\d+(?:\.\d+)?)\s*mH\b', 'inductance_mh'),
        (r'(\d+(?:\.\d+)?)\s*uH\b', 'inductance_uh'),
        (r'(\d+(?:\.\d+)?)\s*H(?![a-zA-Z\d])', 'inductance_h'),

        # ── Mass / density / flow ─────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*kg/m[³3]\b', 'density_kgm3'),
        (r'(\d+(?:\.\d+)?)\s*g/cm[³3]\b', 'density_gcm3'),
        (r'(\d+(?:\.\d+)?)\s*g/L\b', 'density_gl'),
        (r'(\d+(?:\.\d+)?)\s*lb/ft[³3]\b', 'density_lbft3'),
        (r'(\d+(?:\.\d+)?)\s*kg/s\b', 'mass_flow_kgs'),
        (r'(\d+(?:\.\d+)?)\s*L/s\b', 'vol_flow_ls'),
        (r'(\d+(?:\.\d+)?)\s*L/min\b', 'vol_flow_lmin'),
        (r'(\d+(?:\.\d+)?)\s*kg\b', 'mass'),
        (r'(\d+(?:\.\d+)?)\s*tonne\b', 'mass_tonne'),
        (r'(\d+(?:\.\d+)?)\s*lb\b', 'mass_lb'),
        (r'(\d+(?:\.\d+)?)\s*oz(?![·.\-])', 'mass_oz'),
        (r'(\d+(?:\.\d+)?)\s*mg\b', 'mass_mg'),
        # Plain g — word boundary prevents matching "mg" prefix
        (r'(?<!\w)(\d+(?:\.\d+)?)\s*g\b', 'mass_g'),

        # ── Temperature ──────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*°?R\b', 'temperature_rankine'),
        (r'(\d+(?:\.\d+)?)\s*°F\b', 'temperature_f'),
        (r'(\d+(?:\.\d+)?)\s*°C\b', 'temperature_c'),
        (r'(\d+(?:\.\d+)?)\s*(?:K\b|kelvin)', 'temperature'),

        # ── Thermal ───────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*W/m[²2]K\b', 'heat_transfer_coeff'),
        (r'(\d+(?:\.\d+)?)\s*W/mK\b', 'thermal_conductivity'),
        (r'(\d+(?:\.\d+)?)\s*J/kgK\b', 'specific_heat'),

        # ── Distance ─────────────────────────────────────────────────────────────
        # Astronomy first (AU, ly, pc)
        (r'(\d+(?:\.\d+)?)\s*AU\b', 'distance_au'),
        (r'(\d+(?:\.\d+)?)\s*ly\b', 'distance_ly'),
        (r'(\d+(?:\.\d+)?)\s*pc\b', 'distance_pc'),
        # Larger metric
        (r'(\d+(?:\.\d+)?)\s*km\b', 'distance_km'),
        # Imperial
        (r'(\d+(?:\.\d+)?)\s*mi\b', 'distance_mi'),
        (r'(\d+(?:\.\d+)?)\s*ft\b', 'distance_ft'),
        (r'(\d+(?:\.\d+)?)\s*yd\b', 'distance_yd'),
        (r'(\d+(?:\.\d+)?)\s*in\b', 'distance_in'),
        # Smaller metric (cm before mm before um/nm/Å)
        (r'(\d+(?:\.\d+)?)\s*cm\b', 'distance_cm'),
        (r'(\d+(?:\.\d+)?)\s*mm\b', 'distance_mm'),
        (r'(\d+(?:\.\d+)?)\s*um\b', 'distance_um'),
        (r'(\d+(?:\.\d+)?)\s*nm\b', 'distance_nm'),
        (r'(\d+(?:\.\d+)?)\s*[Åa]ngstrom', 'distance_angstrom'),
        # Plain metres: negative lookahead avoids m/s, m², m³, etc.
        (r'(\d+(?:\.\d+)?)\s*m(?![/a-zA-Z\d])', 'distance'),

        # ── Area ─────────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*km[²2]\b', 'area_km2'),
        (r'(\d+(?:\.\d+)?)\s*m[²2]\b', 'area_m2'),
        (r'(\d+(?:\.\d+)?)\s*cm[²2]\b', 'area_cm2'),
        (r'(\d+(?:\.\d+)?)\s*mm[²2]\b', 'area_mm2'),
        (r'(\d+(?:\.\d+)?)\s*ft[²2]\b', 'area_ft2'),
        (r'(\d+(?:\.\d+)?)\s*in[²2]\b', 'area_in2'),
        (r'(\d+(?:\.\d+)?)\s*ha\b', 'area_ha'),
        (r'(\d+(?:\.\d+)?)\s*acres?\b', 'area_acres'),

        # ── Volume ───────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*m[³3]\b', 'volume_m3'),
        (r'(\d+(?:\.\d+)?)\s*cm[³3]\b', 'volume_cm3'),
        (r'(\d+(?:\.\d+)?)\s*ft[³3]\b', 'volume_ft3'),
        (r'(\d+(?:\.\d+)?)\s*in[³3]\b', 'volume_in3'),
        (r'(\d+(?:\.\d+)?)\s*mL\b', 'volume_ml'),
        (r'(\d+(?:\.\d+)?)\s*(?:L\b|liter|litre)', 'volume_l'),
        (r'(\d+(?:\.\d+)?)\s*gal\b', 'volume_gal'),
        (r'(\d+(?:\.\d+)?)\s*fl\.?\s*oz\b', 'volume_floz'),

        # ── Angle ────────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*rad\b', 'angle_rad'),
        (r'(\d+(?:\.\d+)?)\s*(?:degree|deg|°)', 'angle'),

        # ── Time ─────────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*ms\b', 'time_ms'),
        (r'(\d+(?:\.\d+)?)\s*us\b', 'time_us'),
        (r'(\d+(?:\.\d+)?)\s*ns\b', 'time_ns'),
        (r'(\d+(?:\.\d+)?)\s*min\b', 'time_min'),
        (r'(\d+(?:\.\d+)?)\s*hr\b', 'time_hr'),
        (r'(\d+(?:\.\d+)?)\s*(?:days?)\b', 'time_days'),
        (r'(\d+(?:\.\d+)?)\s*(?:years?|yr)\b', 'time_years'),
        # Plain seconds — after Isp patterns so "300 s SL" is captured first
        (r'(\d+(?:\.\d+)?)\s*s\b', 'time_s'),

        # ── Structural mass fraction ──────────────────────────────────────────────
        (r'(?:structural mass fraction|mass fraction)\s+(\d+(?:\.\d+)?)', 'mass_fraction'),
        (r'(?:structural.*?fraction)\s+(?:of\s+)?(\d+(?:\.\d+)?)', 'mass_fraction'),

        # ── Moles / concentration ─────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*mol/L\b', 'concentration_molL'),
        (r'(\d+(?:\.\d+)?)\s*mol\b', 'moles'),
        (r'(\d+(?:\.\d+)?)\s*M(?![a-zA-Z\d/])', 'concentration_M'),

        # ── Magnetic ──────────────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*T(?![a-zA-Z\d])', 'magnetic_flux_t'),
        (r'(\d+(?:\.\d+)?)\s*mT\b', 'magnetic_flux_mt'),
        (r'(\d+(?:\.\d+)?)\s*uT\b', 'magnetic_flux_ut'),
        (r'(\d+(?:\.\d+)?)\s*G(?![a-zA-Z\d])', 'magnetic_flux_gauss'),

        # ── Data / bandwidth ──────────────────────────────────────────────────────
        (r'(\d+(?:\.\d+)?)\s*Gbps\b', 'bandwidth_gbps'),
        (r'(\d+(?:\.\d+)?)\s*Mbps\b', 'bandwidth_mbps'),
        (r'(\d+(?:\.\d+)?)\s*kbps\b', 'bandwidth_kbps'),
        (r'(\d+(?:\.\d+)?)\s*TB\b', 'storage_tb'),
        (r'(\d+(?:\.\d+)?)\s*GB\b', 'storage_gb'),
        (r'(\d+(?:\.\d+)?)\s*MB\b', 'storage_mb'),
        (r'(\d+(?:\.\d+)?)\s*KB\b', 'storage_kb'),

        # ── Generic scientific notation (before plain integer) ────────────────────
        (r'(\d+(?:\.\d+)?[eE][-+]?\d+)', 'value'),

        # ── Generic numbers with context (last resort) ────────────────────────────
        (r'(\d+(?:\.\d+)?)', 'value'),
    ]

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """
        Strip thousands-separator commas from numbers so '15,000' → '15000'.
        Also normalise Unicode superscripts used in units (m³ → m3).
        Also normalise micro-sign variants (µ → u) so patterns can use plain 'u'.
        """
        # Remove commas inside digit sequences: 15,000 → 15000
        text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)
        # Normalise superscript characters sometimes seen in copy-pasted text
        text = text.replace('\u00b2', '2').replace('\u00b3', '3')
        # Normalise micro-sign (µ U+00B5 and μ U+03BC) to plain 'u' for unit patterns
        text = text.replace('\u00b5', 'u').replace('\u03bc', 'u')
        return text

    @staticmethod
    def extract_all(text: str) -> Dict[str, float]:
        """Extract all values from text with intelligent naming"""

        text = ValueExtractor._normalize_numbers(text)
        values = {}

        # Track character spans already claimed by a specific pattern so that
        # the fall-through generic r'(\d+)' pattern doesn't re-emit duplicates.
        claimed_spans: set = set()

        # Try each pattern in order (most-specific first)
        for pattern, key_hint in ValueExtractor.PATTERNS:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))

            for i, match in enumerate(matches):
                # The captured group (the number) spans match.start(1)…match.end(1)
                num_start = match.start(1)
                num_end   = match.end(1)

                if key_hint == 'value':
                    # Generic pattern — skip if this position was already claimed
                    if num_start in claimed_spans:
                        continue
                    # Try to infer from context
                    context_label = ValueExtractor._get_context(text, match.start())
                    base_key = context_label or f"value_{i}"
                    key = base_key
                    if key in values:
                        suffix = 2
                        while f"{base_key}_{suffix}" in values:
                            suffix += 1
                        key = f"{base_key}_{suffix}"
                else:
                    # Specific pattern — skip if an earlier pattern already claimed
                    # this number position (prevents e.g. concentration_M stealing
                    # a velocity value that was already extracted as m/s)
                    if num_start in claimed_spans:
                        continue
                    claimed_spans.add(num_start)
                    key = key_hint
                    if key in values:
                        key = f"{key_hint}_{i+1}"

                values[key] = float(match.group(1))

        return values
    
    @staticmethod
    def _get_context(text: str, position: int) -> str:
        """Get context around a number to infer what it is"""
        
        # Get 50 chars before and after
        start = max(0, position - 50)
        end = min(len(text), position + 50)
        context = text[start:end].lower()
        
        # Check context for hints
        if any(w in context for w in ['satellite', 'spacecraft', 'probe', 'object']):
            return 'mass'
        elif any(w in context for w in ['velocity', 'speed', 'burn', 'm/s']):
            # Guard: if "angular" is nearby, this is angular velocity not linear
            if 'angular' in context:
                return 'angular_velocity'
            return 'velocity'
        elif any(w in context for w in ['orbit', 'km', 'radius', 'altitude']):
            return 'distance'
        elif any(w in context for w in ['temperature', 'kelvin', 'K']):
            return 'temperature'
        elif any(w in context for w in ['pressure', 'atm']):
            return 'pressure'
        elif any(w in context for w in ['angle', 'degree', 'incident']):
            return 'angle'
        elif any(w in context for w in ['current', 'ampere', 'amp']):
            return 'current'
        elif any(w in context for w in ['voltage', 'volt']):
            return 'voltage'
        elif any(w in context for w in ['power', 'watt']):
            return 'power'
        elif any(w in context for w in ['force', 'thrust', 'newton']):
            return 'force'
        elif any(w in context for w in ['frequency', 'hertz', 'cycle']):
            return 'frequency'
        elif any(w in context for w in ['time', 'duration', 'period', 'second']):
            return 'time'
        elif any(w in context for w in ['energy', 'joule', 'work']):
            return 'energy'

        return None
    
    @staticmethod
    def extract_specific(text: str, variable_names: list) -> Dict[str, float]:
        """
        Extract values for specific variables
        
        Args:
            text: Text to extract from
            variable_names: List like ["mass", "velocity", "distance"]
        
        Returns:
            Dict of {variable: value}
        """
        
        all_values = ValueExtractor.extract_all(text)
        result = {}
        
        for var_name in variable_names:
            # Try exact match
            if var_name in all_values:
                result[var_name] = all_values[var_name]
            # Try with _1, _2 suffix
            elif f"{var_name}_1" in all_values:
                result[var_name] = all_values[f"{var_name}_1"]
            # Try common aliases
            elif var_name == 'mu' and 'gravitational_parameter' in all_values:
                result[var_name] = all_values['gravitational_parameter']
        
        return result


# Example test
if __name__ == "__main__":
    test_text = """
    A 5000 kg spacecraft in a 7000 km circular orbit around Earth (μ = 3.986e14 m³/s²) 
    performs a tangential burn that reduces its mass to 4000 kg with exhaust velocity 3000 m/s 
    at periapsis.
    """
    
    values = ValueExtractor.extract_all(test_text)
    print("Extracted values:")
    for k, v in values.items():
        print(f"  {k} = {v}")
