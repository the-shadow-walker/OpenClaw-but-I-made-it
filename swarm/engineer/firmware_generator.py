"""
Firmware Generator — Generates starter firmware for DIY MCU projects.

Uses qwen2.5:14b via direct Ollama HTTP API (same pattern as project_mode.py).
Single public function: generate_firmware(specs, bom, pin_map) -> dict
"""

import os
import re
import json
import requests
from typing import List, Dict, Any, Optional

# Swarm 3.16 — Unified default model (overridable via env)
_DEFAULT_MODEL = os.getenv("SWARM_MODEL_DEFAULT", "batiai/qwen3.6-27b:iq4")


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm(prompt: str, model: str = _DEFAULT_MODEL, max_tokens: int = 2500) -> str:
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  False,
        "keep_alive": 0,
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }
    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[firmware generation error: {e}]"


def _llm_json(prompt: str, model: str = _DEFAULT_MODEL, max_tokens: int = 500) -> Dict:
    raw = _llm(prompt, model=model, max_tokens=max_tokens)
    try:
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


# ── Library keyword mapping ───────────────────────────────────────────────────
# keyword in product title (lower) → (library_name, include_string)
_LIBRARY_MAP = [
    ("mpu6050",       ("Adafruit MPU6050",   "#include <Adafruit_MPU6050.h>\n#include <Adafruit_Sensor.h>")),
    ("mpu-6050",      ("Adafruit MPU6050",   "#include <Adafruit_MPU6050.h>\n#include <Adafruit_Sensor.h>")),
    ("bmp280",        ("Adafruit BMP280",    "#include <Adafruit_BMP280.h>")),
    ("bme280",        ("Adafruit BME280",    "#include <Adafruit_BME280.h>")),
    ("dht22",         ("DHT sensor library", "#include <DHT.h>")),
    ("dht11",         ("DHT sensor library", "#include <DHT.h>")),
    ("ds18b20",       ("OneWire + DallasTemperature", "#include <OneWire.h>\n#include <DallasTemperature.h>")),
    ("l298n",         ("AFMotor",            "#include <AFMotor.h>")),
    ("l293",          ("AFMotor",            "#include <AFMotor.h>")),
    ("drv8825",       ("AccelStepper",       "#include <AccelStepper.h>")),
    ("a4988",         ("AccelStepper",       "#include <AccelStepper.h>")),
    ("stepper",       ("AccelStepper",       "#include <AccelStepper.h>")),
    ("servo",         ("Servo",              "#include <Servo.h>")),
    ("neopixel",      ("Adafruit NeoPixel",  "#include <Adafruit_NeoPixel.h>")),
    ("ws2812",        ("Adafruit NeoPixel",  "#include <Adafruit_NeoPixel.h>")),
    ("oled",          ("Adafruit SSD1306",   "#include <Adafruit_SSD1306.h>\n#include <Adafruit_GFX.h>")),
    ("ssd1306",       ("Adafruit SSD1306",   "#include <Adafruit_SSD1306.h>\n#include <Adafruit_GFX.h>")),
    ("lcd",           ("LiquidCrystal_I2C",  "#include <LiquidCrystal_I2C.h>")),
    ("hcsr04",        ("NewPing",            "#include <NewPing.h>")),
    ("ultrasonic",    ("NewPing",            "#include <NewPing.h>")),
    ("rfid",          ("MFRC522",            "#include <MFRC522.h>")),
    ("mfrc522",       ("MFRC522",            "#include <MFRC522.h>")),
    ("hx711",         ("HX711",              "#include <HX711.h>")),
    ("load cell",     ("HX711",              "#include <HX711.h>")),
    ("pca9685",       ("Adafruit PWM Servo Driver", "#include <Adafruit_PWMServoDriver.h>")),
    ("blynk",         ("Blynk",              "#include <BlynkSimpleEsp32.h>")),
    ("mqtt",          ("PubSubClient",       "#include <PubSubClient.h>")),
]


# ── MCU detection ─────────────────────────────────────────────────────────────

def _detect_mcu(bom: List[Dict]) -> Dict[str, str]:
    """Return {"mcu": str, "language": "cpp"|"python", "board": str}"""
    mcu_map = [
        ("raspberry pi 4",   "python",  "Raspberry Pi 4",   "micropython"),
        ("raspberry pi 3",   "python",  "Raspberry Pi 3",   "micropython"),
        ("raspberry pi zero", "python", "Raspberry Pi Zero", "micropython"),
        ("raspberry pi pico", "python", "Raspberry Pi Pico", "micropython"),
        ("raspberry pi",     "python",  "Raspberry Pi",     "micropython"),
        ("esp32",            "cpp",     "ESP32",             "arduino"),
        ("esp8266",          "cpp",     "ESP8266",           "arduino"),
        ("arduino mega",     "cpp",     "Arduino Mega 2560", "arduino"),
        ("arduino uno",      "cpp",     "Arduino Uno",       "arduino"),
        ("arduino nano",     "cpp",     "Arduino Nano",      "arduino"),
        ("arduino",          "cpp",     "Arduino",           "arduino"),
        ("stm32",            "cpp",     "STM32",             "arduino"),
        ("atmega",           "cpp",     "ATmega",            "arduino"),
    ]
    for item in bom:
        text = f"{item.get('product', {}).get('product_title', '')} {item.get('category', '')}".lower()
        for kw, lang, board, fw_type in mcu_map:
            if kw in text:
                return {"mcu": kw, "language": lang, "board": board, "fw_type": fw_type}
    return {"mcu": "arduino", "language": "cpp", "board": "Arduino", "fw_type": "arduino"}


def _detect_libraries(bom: List[Dict]) -> tuple:
    """Return (library_names: list, include_lines: list)"""
    lib_names: List[str] = []
    includes:  List[str] = []
    seen: set = set()

    for item in bom:
        text = f"{item.get('product', {}).get('product_title', '')} {item.get('category', '')}".lower()
        for kw, (lib, inc) in _LIBRARY_MAP:
            if kw in text and lib not in seen:
                seen.add(lib)
                lib_names.append(lib)
                includes.append(inc)

    return lib_names, includes


def _build_pin_defs(pin_map: Dict) -> str:
    """Convert pin_map dict to C++ #define or Python constant lines."""
    lines = []
    pin_counter = 2  # start at D2 for Arduino
    used = set()

    for comp, info in pin_map.items():
        iface = info.get("interface", "GPIO").upper()
        n = info.get("pins_needed", 1)
        comp_clean = re.sub(r"[^a-zA-Z0-9]", "_", comp).upper()

        if iface == "I2C":
            lines.append(f"// {comp}: I2C (SDA/SCL — hardware pins, no defines needed)")
        elif iface == "SPI":
            lines.append(f"// {comp}: SPI (MOSI/MISO/SCK — hardware pins)")
            lines.append(f"#define {comp_clean}_CS {pin_counter}")
            pin_counter += 1
        elif iface == "UART":
            lines.append(f"// {comp}: UART (use Serial1 or SoftwareSerial)")
        elif iface in ("PWM", "GPIO"):
            for idx in range(min(n, 3)):
                while pin_counter in used:
                    pin_counter += 1
                suffix = f"_PIN_{idx}" if n > 1 else "_PIN"
                lines.append(f"#define {comp_clean}{suffix} {pin_counter}")
                used.add(pin_counter)
                pin_counter += 1
        elif iface == "ANALOG":
            lines.append(f"#define {comp_clean}_PIN A0  // adjust analog pin")

    return "\n".join(lines)


# ── Mermaid flowchart generation ──────────────────────────────────────────────

def _generate_mermaid(specs: Dict, bom: List[Dict], mcu: Dict) -> str:
    """Generate a simple Mermaid.js flowchart for the firmware logic."""
    llm, _ = _get_llm_helpers()

    components = [item.get("category", "") for item in bom
                  if item.get("category", "").lower() not in
                  ("microcontroller", "mcu", "development board", "arduino", "esp32")][:6]

    prompt = f"""Write a Mermaid.js flowchart (flowchart TD) for a {mcu['board']} firmware program.

Project: {specs.get('description', 'DIY electronics project')}
Components: {', '.join(components) if components else 'sensors and actuators'}

Include: setup/init node, main loop, read sensor(s), process/decision, control output(s).
Keep it simple — max 12 nodes. Use --> for transitions, {{{{condition}}}} for decisions.

Output ONLY the raw Mermaid code starting with 'flowchart TD', no markdown fences."""

    mermaid = llm(prompt, model=_DEFAULT_MODEL, max_tokens=400)
    # Strip any accidental backtick fences
    mermaid = re.sub(r"^```.*?\n?|```$", "", mermaid.strip(), flags=re.MULTILINE).strip()
    if not mermaid.startswith("flowchart"):
        mermaid = "flowchart TD\n    A[Start] --> B[Setup]\n    B --> C[Main Loop]\n    C --> D[Read Sensors]\n    D --> E[Process Data]\n    E --> F[Control Outputs]\n    F --> C"
    return mermaid


def _get_llm_helpers():
    from project_mode import llm, llm_json
    return llm, llm_json


# ── Main function ─────────────────────────────────────────────────────────────

def generate_firmware(specs: Dict, bom: List[Dict], pin_map: Dict) -> Dict:
    """
    Generate starter firmware for the detected MCU.

    Args:
        specs:   Project vision dict from project_mode
        bom:     List of sourced BOM items
        pin_map: Pin map from electronics_engine.pin_conflict_check()

    Returns:
        {
          "language":  "cpp" | "python",
          "board":     str,
          "code":      str,
          "libraries": list[str],
          "mermaid":   str,
          "skipped":   bool,   # True if no MCU detected
        }
    """
    mcu = _detect_mcu(bom)
    lib_names, includes = _detect_libraries(bom)
    pin_defs = _build_pin_defs(pin_map)
    mermaid  = _generate_mermaid(specs, bom, mcu)

    # Build component list summary for the LLM
    bom_summary = "\n".join(
        f"  - {item.get('category', '')}: {item.get('product', {}).get('product_title', '')[:60]}"
        for item in bom
    )

    include_block = "\n".join(dict.fromkeys(includes))  # deduped

    if mcu["language"] == "cpp":
        # Arduino-style C++
        prompt = f"""Write a complete, well-commented Arduino sketch (.ino) for the following project.

Board: {mcu['board']}
Project: {specs.get('description', 'DIY project')}
Name: {specs.get('name', 'MyProject')}

Components in the BOM:
{bom_summary}

Libraries detected (already #included below):
{chr(10).join(lib_names) if lib_names else "  (none — use built-in Arduino functions)"}

Pin definitions already declared:
{pin_defs if pin_defs else "  (use default Arduino pins)"}

Requirements from user:
{json.dumps(specs.get('requirements', specs), indent=2)[:400]}

Write a COMPLETE .ino file that:
1. Starts with the #include block and pin definitions provided
2. Declares global objects for each library
3. Implements setup() that initialises Serial, all sensors, and actuators
4. Implements loop() with realistic logic for this specific project
5. Includes helper functions as needed
6. Has clear comments on every non-obvious line

The code must compile without errors (use correct API calls for the detected libraries).
Output ONLY the raw .ino code, no markdown fences."""

        code = _llm(prompt, model=_DEFAULT_MODEL, max_tokens=2500)
        # Prepend includes if the LLM omitted them
        if include_block and "#include" not in code[:200]:
            code = include_block + "\n\n" + code
        if pin_defs and "#define" not in code[:400]:
            header, rest = (code.split("\n\n", 1) + [""])[:2]
            code = header + "\n\n" + pin_defs + "\n\n" + rest

    else:
        # MicroPython / Python
        prompt = f"""Write a complete, well-commented MicroPython program for the following project.

Board: {mcu['board']}
Project: {specs.get('description', 'DIY project')}
Name: {specs.get('name', 'MyProject')}

Components:
{bom_summary}

Requirements:
{json.dumps(specs.get('requirements', specs), indent=2)[:400]}

Write a COMPLETE main.py / boot.py that:
1. Imports necessary MicroPython modules (machine, utime, etc.)
2. Defines pin numbers and global objects
3. Has an init() function that sets up hardware
4. Has a main() loop with realistic logic for this project
5. Has clear comments throughout
6. Handles basic exceptions

Output ONLY the raw Python code, no markdown fences."""

        code = _llm(prompt, model=_DEFAULT_MODEL, max_tokens=2000)

    # Strip accidental fences from LLM output
    code = re.sub(r"^```[a-z]*\n?", "", code.strip(), flags=re.MULTILINE)
    code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE).strip()

    return {
        "language":  mcu["language"],
        "board":     mcu["board"],
        "code":      code,
        "libraries": lib_names,
        "mermaid":   mermaid,
        "skipped":   False,
    }
