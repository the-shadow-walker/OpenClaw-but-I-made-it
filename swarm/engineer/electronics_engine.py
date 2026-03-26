"""
Electronics Engine — Deterministic/LLM-hybrid validation for DIY electronics projects.

Three public functions:
  power_budget(bom, requirements) -> dict
  voltage_rail_map(bom) -> list[dict]
  pin_conflict_check(bom, requirements) -> dict

Reuses llm() / llm_json() from project_mode.py — import lazily to avoid circular deps.
"""

import re
import json
from typing import List, Dict, Any


# ── LLM helpers (imported from project_mode to avoid duplication) ─────────────
def _get_llm():
    from project_mode import llm, llm_json
    return llm, llm_json


# ── Current-draw heuristics ───────────────────────────────────────────────────
# Maps lowercase keyword → typical current draw in mA (peak/worst-case)
_CURRENT_HEURISTICS: Dict[str, float] = {
    "arduino uno":       50.0,
    "arduino nano":      40.0,
    "arduino mega":      80.0,
    "esp32":            240.0,
    "esp8266":          200.0,
    "raspberry pi pico": 100.0,
    "raspberry pi zero": 350.0,
    "raspberry pi 3":   2500.0,
    "raspberry pi 4":   3000.0,
    "atmega":            50.0,
    "stm32":             80.0,
    "servo":            1000.0,  # stall per servo
    "motor driver":      500.0,
    "l298n":             500.0,
    "l293":              500.0,
    "drv8825":           300.0,
    "a4988":             300.0,
    "stepper":           700.0,
    "dc motor":         1000.0,
    "brushless":        5000.0,
    "neopixel":          60.0,   # per pixel at full white
    "ws2812":            60.0,
    "led strip":        2000.0,
    "lcd":               60.0,
    "oled":              30.0,
    "tft":              200.0,
    "camera":           300.0,
    "bluetooth":        150.0,
    "wifi":             300.0,
    "gps":               30.0,
    "ultrasonic":        15.0,
    "sensor":            20.0,
    "mpu6050":           10.0,
    "imu":               10.0,
    "relay":             90.0,
    "buzzer":            35.0,
    "fan":              500.0,
    "solenoid":        1500.0,
}

# Voltage rail detection heuristics: keyword → typical voltage
_VOLTAGE_HEURISTICS: Dict[str, float] = {
    "3.3v":   3.3,
    "3.3 v":  3.3,
    "5v":     5.0,
    "5 v":    5.0,
    "12v":   12.0,
    "12 v":  12.0,
    "24v":   24.0,
    "24 v":  24.0,
    "9v":     9.0,
    "9 v":    9.0,
    "7.4v":   7.4,
    "7.4 v":  7.4,
    "11.1v": 11.1,
    "lipo":   7.4,   # default 2S
    "18650":  3.7,
    "li-ion": 3.7,
    "lithium": 3.7,
    "arduino":  5.0,
    "esp32":    3.3,
    "esp8266":  3.3,
    "raspberry pi": 5.0,
    "raspberry pi 3": 5.0,
    "raspberry pi 4": 5.0,
    "l298n":   12.0,
    "stepper": 12.0,
}


def _title(item: Dict) -> str:
    """Extract lowercased product title from a BOM item."""
    return item.get("product", {}).get("product_title", "").lower()


def _category(item: Dict) -> str:
    return item.get("category", "").lower()


def _item_text(item: Dict) -> str:
    return f"{_title(item)} {_category(item)}"


# ── 1. Power Budget ───────────────────────────────────────────────────────────

def power_budget(bom: List[Dict], requirements: Dict) -> Dict:
    """
    Estimate power draw from BOM heuristics + LLM extraction.
    Compare against battery capacity if one is found in the BOM.

    Returns:
        {
          "pass": bool,
          "total_mA": float,
          "margin_mA": float,      # battery_max - total  (None if no battery)
          "battery_mAh": float,    # None if not found
          "battery_max_mA": float, # based on 1C discharge
          "breakdown": {category: mA},
          "warnings": [str],
        }
    """
    llm, llm_json = _get_llm()

    breakdown: Dict[str, float] = {}
    warnings: List[str] = []
    battery_mAh: float = 0.0
    battery_max_mA: float = 0.0

    for item in bom:
        text = _item_text(item)
        cat  = item.get("category", "Unknown")

        # Battery detection
        mah_match = re.search(r"(\d[\d,.]+)\s*mah", text)
        c_match   = re.search(r"(\d+(?:\.\d+)?)\s*c\b", text)
        if mah_match:
            battery_mAh = float(mah_match.group(1).replace(",", ""))
            c_rate = float(c_match.group(1)) if c_match else 1.0
            battery_max_mA = battery_mAh * c_rate
            continue

        # Heuristic match
        matched_mA = 0.0
        for kw, draw in sorted(_CURRENT_HEURISTICS.items(), key=lambda x: -len(x[0])):
            if kw in text:
                matched_mA = draw
                break

        if matched_mA > 0:
            breakdown[cat] = max(breakdown.get(cat, 0.0), matched_mA)
        else:
            # LLM fallback for unrecognised components
            title_short = item.get("product", {}).get("product_title", "")[:80]
            prompt = (
                f'Component: "{title_short}"\n'
                f"Estimate the typical peak current draw in mA for this component in a DIY project.\n"
                f'Respond with JSON only: {{"current_mA": 50, "reasoning": "..."}}'
            )
            data = llm_json(prompt, max_tokens=100)
            est_mA = float(data.get("current_mA", 30))
            breakdown[cat] = max(breakdown.get(cat, 0.0), est_mA)

    total_mA = sum(breakdown.values())

    # Safety factor
    total_mA_safe = total_mA * 1.2

    margin_mA = None
    passed = True

    if battery_max_mA > 0:
        margin_mA = battery_max_mA - total_mA_safe
        passed = margin_mA >= 0
        if not passed:
            warnings.append(
                f"Power budget EXCEEDS battery discharge limit: "
                f"{total_mA_safe:.0f}mA needed vs {battery_max_mA:.0f}mA max (1C). "
                f"Use a higher-C battery or reduce load."
            )
        elif margin_mA < total_mA_safe * 0.15:
            warnings.append(
                f"Power margin is thin ({margin_mA:.0f}mA). Consider a higher-capacity or higher-C battery."
            )
    else:
        warnings.append(
            "No battery found in BOM — power budget check skipped. "
            f"Estimated total draw: {total_mA_safe:.0f}mA (with 1.2× safety factor)."
        )

    return {
        "pass":            passed,
        "total_mA":        round(total_mA, 1),
        "total_mA_safe":   round(total_mA_safe, 1),
        "margin_mA":       round(margin_mA, 1) if margin_mA is not None else None,
        "battery_mAh":     battery_mAh if battery_mAh else None,
        "battery_max_mA":  round(battery_max_mA, 1) if battery_max_mA else None,
        "breakdown":       {k: round(v, 1) for k, v in breakdown.items()},
        "warnings":        warnings,
    }


# ── 2. Voltage Rail Map ───────────────────────────────────────────────────────

# Auto-inject rules: if rail A and rail B coexist, inject a converter
_RAIL_CONVERTERS = [
    # (high_voltage, low_voltage, query)
    (12.0, 5.0,  "12V to 5V 3A DC-DC buck converter step-down module"),
    (12.0, 3.3,  "12V to 3.3V DC-DC buck converter step-down module"),
    (24.0, 5.0,  "24V to 5V 3A DC-DC buck converter step-down module"),
    (24.0, 12.0, "24V to 12V DC-DC buck converter step-down module"),
    (24.0, 3.3,  "24V to 3.3V DC-DC buck converter step-down module"),
    (9.0,  5.0,  "9V to 5V DC-DC buck converter step-down module"),
    (9.0,  3.3,  "9V to 3.3V DC-DC buck converter step-down module"),
    (7.4,  5.0,  "7.4V LiPo to 5V DC-DC buck converter step-down module"),
    (7.4,  3.3,  "7.4V to 3.3V DC-DC buck converter step-down module"),
    (5.0,  3.3,  "5V to 3.3V LDO voltage regulator AMS1117"),
]


def voltage_rail_map(bom: List[Dict]) -> List[Dict]:
    """
    Detect all voltage rails referenced in BOM items.
    If incompatible rails coexist, return a list of auto-add component dicts.

    Returns list of {"category": str, "query": str} dicts to be sourced.
    """
    rails: set = set()

    for item in bom:
        text = _item_text(item)
        for kw, v in sorted(_VOLTAGE_HEURISTICS.items(), key=lambda x: -len(x[0])):
            if kw in text:
                rails.add(v)
                break

    auto_adds = []
    rails_list = sorted(rails, reverse=True)

    for high_v, low_v, query in _RAIL_CONVERTERS:
        if high_v in rails and low_v in rails:
            # Check we don't already have a converter for this pair
            pair_key = f"{high_v:.0f}V to {low_v:.0f}V"
            already_have = any(
                pair_key.lower().replace(" ", "") in _item_text(i).replace(" ", "")
                for i in bom
            )
            if not already_have and not any(
                a["query"] == query for a in auto_adds
            ):
                auto_adds.append({
                    "category": f"DC-DC Buck Converter ({high_v:.0f}V→{low_v:.0f}V)",
                    "query":    query,
                    "must_meet": f"Input {high_v:.0f}V, Output {low_v:.0f}V, ≥1A",
                })

    return auto_adds


# ── 3. Pin Conflict Check ─────────────────────────────────────────────────────

def pin_conflict_check(bom: List[Dict], requirements: Dict) -> Dict:
    """
    Use LLM to identify pin requirements per component and map against MCU availability.

    Returns:
        {
          "conflicts": [str],
          "warnings": [str],
          "pin_map": {component: {interface: [pins]}},
        }
    """
    llm, llm_json = _get_llm()

    # Build component list for the LLM
    components = []
    mcu_type = "unknown MCU"
    for item in bom:
        text  = _item_text(item)
        title = item.get("product", {}).get("product_title", "")[:80]
        cat   = item.get("category", "Component")

        # Detect MCU
        for kw in ["arduino uno", "arduino nano", "arduino mega", "esp32", "esp8266",
                   "raspberry pi pico", "stm32", "atmega"]:
            if kw in text:
                mcu_type = kw.title()
                break

        components.append({"category": cat, "title": title})

    prompt = f"""You are an electronics engineer. The project uses a {mcu_type}.

Components:
{json.dumps(components, indent=2)}

For each non-MCU component, identify:
1. Which interface it uses (I2C, SPI, UART, PWM, analog, digital GPIO)
2. How many pins it needs

Then check for conflicts:
- Multiple I2C devices sharing the same address (conflict)
- More PWM channels needed than the MCU has
- More UART ports needed than available
- Too many SPI CS lines

Also note if the total pin count exceeds the MCU's available pins.

Respond ONLY with valid JSON:
{{
  "mcu": "{mcu_type}",
  "pin_map": {{
    "ComponentName": {{
      "interface": "I2C|SPI|UART|PWM|GPIO|analog",
      "pins_needed": 2,
      "i2c_address": "0x68"
    }}
  }},
  "conflicts": ["description of conflict"],
  "warnings": ["potential issue or recommendation"]
}}"""

    data = llm_json(prompt, max_tokens=600)

    return {
        "mcu":       data.get("mcu", mcu_type),
        "conflicts": data.get("conflicts", []),
        "warnings":  data.get("warnings", []),
        "pin_map":   data.get("pin_map", {}),
    }
