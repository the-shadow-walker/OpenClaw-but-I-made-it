"""
Project Mode - Dynamic LLM-Driven Project Assistant
Integrated into Swarm 3.0

The LLM asks HIGH-LEVEL vision questions only.
Technical specs (battery capacity, DPI, refresh rate, etc.) are computed, not asked.
User can type "also ..." at any prompt to add freeform context.
"""

import asyncio
import requests
import json
import re
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

# ── Amazon RapidAPI ──────────────────────────────────────────────────────────
RAPIDAPI_KEY  = "364612b91cmshda611ad08d3cfd7p16790ejsnc906bc1fc5c9"
RAPIDAPI_HOST = "amazon-online-data-api.p.rapidapi.com"


def amazon_search(query: str, max_results: int = 6) -> List[Dict]:
    url = "https://amazon-online-data-api.p.rapidapi.com/search"
    params = {"query": query, "country": "US"}
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key":  RAPIDAPI_KEY,
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        products = r.json().get("products", [])
        return [p for p in products if p.get("product_price")][:max_results]
    except Exception as e:
        print(f"   ⚠️  Amazon search error: {e}")
        return []


# ── Ollama LLM helpers ───────────────────────────────────────────────────────
def llm(prompt: str, system: str = "", model: str = "phi4:14b", max_tokens: int = 1000) -> str:
    full = f"{system}\n\n{prompt}" if system else prompt
    payload = {
        "model": model,
        "prompt": full,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": max_tokens},
    }
    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=90)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def llm_json(prompt: str, system: str = "", model: str = "phi4:14b", max_tokens: int = 800) -> Dict:
    raw = llm(prompt, system=system, model=model, max_tokens=max_tokens)
    try:
        raw = re.sub(r"```json|```", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else {}
    except Exception:
        return {}


# ── I/O helpers ──────────────────────────────────────────────────────────────
def _header(title: str):
    print("\n" + "═" * 68)
    print(f"  {title}")
    print("═" * 68)

def _hr():
    print("─" * 68)

def _read_input(prompt_line: str) -> str:
    """
    Read a line of input. Handles 'also ...' prefix — returns the full string
    so the caller can detect it and merge context.
    """
    try:
        return input(prompt_line).strip()
    except (EOFError, KeyboardInterrupt):
        return ""

def _ask(question: str, default: str = "") -> str:
    suffix = f"  (default: {default})" if default else ""
    raw = _read_input(f"\n  JARVIS: {question}{suffix}\n  YOU  > ")
    return raw if raw else default

def _choose(question: str, options: List[str], recommendation: str = "") -> str:
    print(f"\n  JARVIS: {question}")
    if recommendation:
        print(f"  ℹ️  {recommendation}")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    print("  (Or type 'also <extra context>' to add something before answering)")
    while True:
        raw = _read_input("  YOU  > ")
        if not raw:
            return options[0]
        # "also ..." — return as-is for the caller to handle
        if raw.lower().startswith("also"):
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        matches = [o for o in options if raw.lower() in o.lower()]
        if len(matches) == 1:
            return matches[0]
        print(f"  (Enter a number 1–{len(options)}, or type your answer)")

def _choose_multi(question: str, options: List[str], recommendation: str = "") -> List[str]:
    print(f"\n  JARVIS: {question}")
    if recommendation:
        print(f"  ℹ️  {recommendation}")
    print("  (Numbers separated by spaces, 'all', 'none', or start with 'also' to add context)")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        raw = _read_input("  YOU  > ").lower()
        if not raw or raw == "none":
            return []
        if raw == "all":
            return options[:]
        if raw.startswith("also"):
            return [raw]   # caller handles
        parts = re.split(r"[,\s]+", raw)
        chosen = [options[int(p) - 1] for p in parts if p.isdigit() and 1 <= int(p) <= len(options)]
        if chosen:
            return chosen
        print(f"  (Enter numbers 1–{len(options)})")

def _yn(question: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    raw = _read_input(f"\n  JARVIS: {question} {hint}\n  YOU  > ").lower()
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    return default


# ── "also" handling ───────────────────────────────────────────────────────────

def _extract_also(raw: str) -> Optional[str]:
    """
    If the user typed 'also <something>', return the extra context string.
    Otherwise return None.
    """
    m = re.match(r"^also\s+(.+)$", raw.strip(), re.IGNORECASE)
    return m.group(1).strip() if m else None


def _handle_also(raw: str, specs: Dict, history: List[Dict]) -> Optional[str]:
    """
    If raw starts with 'also', store the extra note in specs['_notes'] and
    return the extra context so the caller knows to re-ask the question.
    Returns the note string if it was an 'also', else None.
    """
    note = _extract_also(raw)
    if note:
        existing = specs.get("_user_notes", [])
        existing.append(note)
        specs["_user_notes"] = existing
        history.append({"question": "(user added context)", "answer": f"also: {note}"})
        print(f"  ✅ Got it — added to project notes: \"{note}\"")
        return note
    return None


# ── Question engine ───────────────────────────────────────────────────────────

QUESTION_SYSTEM = """You are JARVIS, a technical project assistant. You ask HIGH-LEVEL VISION questions only.

Your goal is to understand WHAT the user wants to build and WHY — not the engineering details.
You will figure out battery capacity, DPI, refresh rates, resistance values, etc. yourself later.

GOOD questions (vision-level):
- "What features do you want it to have?"
- "How long should it run on a charge?"  (not "What mAh battery?")
- "Should it be portable or plugged in?"
- "What environment will it be used in?"
- "What materials do you want the outer shell to be?"
- "Does it need to connect to your phone or other devices?"

BAD questions (never ask these — figure them out yourself):
- "What DPI/PPI do you need?"
- "What refresh rate in Hz?"
- "What is the exact battery capacity in mAh?"
- "What voltage regulator do you need?"
- "What is the resistance of the motor winding?"
- Anything the user clearly cannot be expected to know without engineering knowledge

Stop asking (done=true) after 6-10 questions. You do NOT need exhaustive specs — you need the vision.
If the user has already described something specific (e.g. "night vision cameras"), do not ask about it again.

Respond ONLY with valid JSON, no markdown."""


def get_next_question(specs: Dict, history: List[Dict]) -> Optional[Dict]:
    specs_str   = json.dumps(specs, indent=2)
    history_str = "\n".join(
        f"  Q: {h['question']}\n  A: {h['answer']}" for h in history[-8:]
    ) or "(none yet)"

    prompt = f"""What we know so far:
{specs_str}

Recent Q&A:
{history_str}

Generate the next single high-level vision question, or done=true if you have enough.
Aim to stop after 6-10 total questions.

Return JSON only:
{{
  "done": false,
  "question": "...",
  "type": "text" | "choice" | "multi",
  "options": ["option1", "option2"],
  "recommendation": "optional brief suggestion",
  "key": "snake_case_key"
}}

Or: {{"done": true}}"""

    return llm_json(prompt, system=QUESTION_SYSTEM, max_tokens=400)


# ── Requirements computation ──────────────────────────────────────────────────

def compute_requirements(specs: Dict) -> Dict:
    notes = specs.get("_user_notes", [])
    notes_str = "\n".join(f"  - {n}" for n in notes) if notes else "  (none)"

    prompt = f"""You are a hardware engineer. Given the project vision below, compute ALL technical requirements.

PROJECT VISION:
{json.dumps({k: v for k, v in specs.items() if not k.startswith('_')}, indent=2)}

EXTRA USER NOTES:
{notes_str}

Determine all technical specs the user did NOT specify — battery capacity, display specs,
processor requirements, power budgets, connector types, etc. Make sensible engineering decisions.
Apply a 1.3–1.5x safety factor where relevant.

Respond ONLY with JSON:
{{
  "requirements": {{
    "descriptive_key": "value with unit"
  }},
  "component_categories": ["category1", "category2"],
  "engineering_decisions": {{
    "decision": "rationale"
  }},
  "safety_factor": 1.3,
  "notes": "any warnings or important constraints"
}}"""

    data = llm_json(prompt, max_tokens=900)
    return data if data else {"requirements": {}, "component_categories": [], "notes": ""}


# ── Search query generation ───────────────────────────────────────────────────

def build_search_queries(specs: Dict, req: Dict) -> List[Dict]:
    sourcing_mode = specs.get("sourcing_mode", "balanced")

    mode_hint = ""
    if sourcing_mode == "budget":
        mode_hint = "Append 'budget' or 'generic' or 'multipack' to each query to find affordable options."
    elif sourcing_mode == "quality":
        mode_hint = "Prefer well-known brand names (Adafruit, SparkFun, Pololu, Arduino, Raspberry Pi, etc.) in queries."
    else:
        mode_hint = "Balance quality and price — avoid the cheapest no-name parts but don't require premium brands."

    prompt = f"""You are a hardware procurement engineer.

PROJECT:
{json.dumps({k: v for k, v in specs.items() if not k.startswith('_')}, indent=2)}

REQUIREMENTS:
{json.dumps(req.get('requirements', {}), indent=2)}

CATEGORIES TO SOURCE:
{json.dumps(req.get('component_categories', []))}

SOURCING MODE: {sourcing_mode}
{mode_hint}

Generate specific Amazon search queries for each category.
Include key specs in queries (voltage, model numbers, capacity, etc.).

Respond ONLY with JSON:
{{
  "searches": [
    {{
      "category": "human readable name",
      "query": "amazon search string",
      "must_meet": "one-line validation requirement"
    }}
  ]
}}"""

    data = llm_json(prompt, max_tokens=700)
    return data.get("searches", [])


# ── Product validation ────────────────────────────────────────────────────────

def validate_product(product: Dict, must_meet: str, specs: Dict) -> Dict:
    title  = product.get("product_title", "")
    price  = product.get("product_price", "?")
    rating = product.get("product_star_rating", "?")

    prompt = f"""Product: {title}
Price: ${price}  Rating: {rating}★
Requirement: {must_meet}
Project: {specs.get('name', '')} — {specs.get('description', '')}

Extract specs from the title. Does it meet the requirement?

Respond ONLY with JSON:
{{
  "meets": true,
  "confidence": 80,
  "reason": "one sentence",
  "extracted_specs": {{"key": "value"}}
}}"""

    data = llm_json(prompt, max_tokens=200)
    return {
        "pass":       data.get("meets", True),
        "confidence": data.get("confidence", 50),
        "reason":     data.get("reason", ""),
        "extracted":  data.get("extracted_specs", {}),
    }


# ── Electronics trigger detection ─────────────────────────────────────────────

_ELECTRONICS_KW = [
    "arduino", "esp32", "esp8266", "raspberry", "motor", "battery", "lipo",
    "servo", "sensor", "microcontroller", "stm32", "atmega", "mcu", "stepper",
    "solenoid", "relay", "neopixel", "ws2812", "led strip",
]

_MCU_KW = [
    "arduino", "esp32", "esp8266", "raspberry pi", "microcontroller",
    "atmega", "stm32", "raspberry pi pico", "teensy",
]

_3DPRINT_KW = [
    "3d print", "filament", "fdm", "resin", "pla", "petg", "abs filament",
    "sla print",
]


def _bom_has(bom: List[Dict], keywords: List[str]) -> bool:
    for item in bom:
        text = f"{item.get('product', {}).get('product_title', '')} {item.get('category', '')}".lower()
        if any(kw in text for kw in keywords):
            return True
    return False


def _specs_has(specs: Dict, keywords: List[str]) -> bool:
    blob = json.dumps(specs).lower()
    return any(kw in blob for kw in keywords)


# ── Safety check ─────────────────────────────────────────────────────────────

def run_safety_check(bom: List[Dict]) -> List[Dict]:
    """
    Load safety_db.json and scan all BOM titles + categories for hazard keywords.
    Returns list of triggered warning dicts: [{keyword, level, warnings}]
    """
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safety_db.json")
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            safety_db = json.load(f)
    except Exception as e:
        print(f"  ⚠️  Could not load safety_db.json: {e}")
        return []

    triggered: List[Dict] = []
    seen_keys: set = set()

    for item in bom:
        text = f"{item.get('product', {}).get('product_title', '')} {item.get('category', '')}".lower()
        for keyword, entry in safety_db.items():
            if keyword in text and keyword not in seen_keys:
                seen_keys.add(keyword)
                triggered.append({
                    "keyword":  keyword,
                    "level":    entry.get("level", "LOW"),
                    "warnings": entry.get("warnings", []),
                })

    return triggered


# ── Phase 3.5: Electronics Validation ────────────────────────────────────────

def run_electronics_validation(bom: List[Dict], req: Dict, specs: Dict) -> Dict:
    """Run power_budget, voltage_rail_map, pin_conflict_check. Print results."""
    from electronics_engine import power_budget, voltage_rail_map, pin_conflict_check

    results = {
        "power":       None,
        "auto_adds":   [],
        "pin_check":   None,
    }

    # Power budget
    _header("PHASE 3.5a — POWER BUDGET")
    print("  Estimating current draw from BOM...")
    pb = power_budget(bom, req)
    results["power"] = pb

    status = "✅ PASS" if pb["pass"] else "❌ FAIL"
    print(f"\n  {status}  Total draw: {pb['total_mA_safe']:.0f}mA (with 1.2× safety factor)")
    if pb.get("battery_mAh"):
        print(f"  Battery:    {pb['battery_mAh']:.0f}mAh @ 1C = {pb['battery_max_mA']:.0f}mA max")
        if pb.get("margin_mA") is not None:
            print(f"  Margin:     {pb['margin_mA']:.0f}mA")
    print("\n  Breakdown:")
    for cat, mA in pb.get("breakdown", {}).items():
        print(f"    • {cat}: {mA:.0f}mA")
    for w in pb.get("warnings", []):
        print(f"  ⚠️  {w}")

    # Voltage rail map
    _header("PHASE 3.5b — VOLTAGE RAIL MAP")
    print("  Checking for rail mismatches...")
    auto_adds = voltage_rail_map(bom)
    results["auto_adds"] = auto_adds

    if auto_adds:
        print(f"\n  ⚡ Found {len(auto_adds)} missing power conversion component(s):")
        for a in auto_adds:
            print(f"    + {a['category']}: {a['query']}")
    else:
        print("  ✅ No missing converters detected.")

    # Pin conflict check
    _header("PHASE 3.5c — PIN CONFLICT CHECK")
    print("  Analysing pin requirements vs MCU availability...")
    pc = pin_conflict_check(bom, req)
    results["pin_check"] = pc

    mcu = pc.get("mcu", "unknown MCU")
    print(f"\n  MCU: {mcu}")
    pin_map = pc.get("pin_map", {})
    if pin_map:
        print("  Pin assignments:")
        for comp, info in pin_map.items():
            iface = info.get("interface", "?")
            n     = info.get("pins_needed", "?")
            addr  = f"  addr={info['i2c_address']}" if "i2c_address" in info else ""
            print(f"    • {comp}: {iface} ({n} pins){addr}")
    conflicts = pc.get("conflicts", [])
    if conflicts:
        print(f"\n  ❌ CONFLICTS ({len(conflicts)}):")
        for c in conflicts:
            print(f"    ✖  {c}")
    else:
        print("  ✅ No pin conflicts detected.")
    for w in pc.get("warnings", []):
        print(f"  ⚠️  {w}")

    return results


# ── Phase 3.6: Datasheet & Tutorial Fetch ─────────────────────────────────────

async def fetch_datasheets_and_tutorials(bom: List[Dict], orchestrator) -> List[Dict]:
    """
    For each BOM item, search for datasheet PDF and tutorial links.
    Stores results as bom[i]['datasheet_url'] and bom[i]['tutorial_urls'].
    Returns the annotated BOM.
    """
    if orchestrator is None:
        return bom

    _header("PHASE 3.6 — DATASHEET & TUTORIAL FETCH")

    for item in bom:
        title = item.get("product", {}).get("product_title", "")[:60]
        cat   = item.get("category", "component")

        # Detect MCU type for tutorials
        mcu_hint = ""
        for kw in ["arduino", "esp32", "raspberry pi", "stm32"]:
            if kw in title.lower():
                mcu_hint = kw
                break

        try:
            # Datasheet search
            ds_query = f"{title} datasheet filetype:pdf"
            print(f"  [datasheet] {title[:50]}...")
            ds_result = await orchestrator.process_question(ds_query)
            # Extract first URL from result
            urls = re.findall(r"https?://\S+\.pdf", ds_result)
            item["datasheet_url"] = urls[0] if urls else None
        except Exception:
            item["datasheet_url"] = None

        try:
            # Tutorial search
            tut_query = (
                f"{cat} {mcu_hint} tutorial site:instructables.com OR site:hackaday.com"
                if mcu_hint else
                f"{cat} DIY tutorial site:instructables.com OR site:hackaday.com"
            )
            print(f"  [tutorial]  {cat}...")
            tut_result = await orchestrator.process_question(tut_query)
            urls = re.findall(r"https?://\S+", tut_result)
            item["tutorial_urls"] = [u for u in urls if "instructables" in u or "hackaday" in u][:2]
        except Exception:
            item["tutorial_urls"] = []

    return bom


# ── Phase 3.7: Print safety warnings ─────────────────────────────────────────

def print_safety_warnings(safety_warnings: List[Dict]):
    if not safety_warnings:
        return
    high_crit = [w for w in safety_warnings if w["level"] in ("HIGH", "CRITICAL")]

    _header("⚠️  SAFETY NOTICES")
    for entry in safety_warnings:
        lvl = entry["level"]
        kw  = entry["keyword"].upper()
        icon = {"CRITICAL": "🚨", "HIGH": "⚠️ ", "MEDIUM": "⚡", "LOW": "ℹ️ "}.get(lvl, "⚠️ ")
        print(f"\n  {icon} [{lvl}] {kw}")
        for w in entry["warnings"]:
            print(f"     • {w}")


# ── Markdown generation ───────────────────────────────────────────────────────

def generate_markdown(
    specs: Dict,
    req: Dict,
    bom: List[Dict],
    safety_warnings: Optional[List[Dict]] = None,
    pin_map: Optional[Dict] = None,
    firmware_result: Optional[Dict] = None,
    power_budget_result: Optional[Dict] = None,
) -> str:
    bom_lines = []
    total = 0.0
    for item in bom:
        p   = item.get("product", {})
        cat = item.get("category", "")
        price_str = p.get("product_price", "0")
        try:
            total += float(re.sub(r"[^\d.]", "", price_str))
        except Exception:
            pass
        bom_lines.append({
            "category":     cat,
            "title":        p.get("product_title", "")[:80],
            "price":        f"${price_str}",
            "rating":       f"{p.get('product_star_rating','?')}★",
            "url":          p.get("product_url", ""),
            "note":         item.get("validation", {}).get("reason", ""),
            "datasheet":    item.get("datasheet_url"),
            "tutorials":    item.get("tutorial_urls", []),
        })

    user_notes = specs.get("_user_notes", [])
    notes_str = "\n".join(f"  - {n}" for n in user_notes) if user_notes else "  (none)"

    # Check for 3D printing in project
    has_3dprint = _specs_has(specs, _3DPRINT_KW) or _bom_has(bom, _3DPRINT_KW)

    prompt = f"""Write a complete professional project document in Markdown.

PROJECT VISION:
{json.dumps({k: v for k, v in specs.items() if not k.startswith('_')}, indent=2)}

EXTRA USER NOTES:
{notes_str}

COMPUTED REQUIREMENTS:
{json.dumps(req.get('requirements', {}), indent=2)}
Engineering decisions: {json.dumps(req.get('engineering_decisions', {}), indent=2)}
Notes: {req.get('notes', '')}

BILL OF MATERIALS ({len(bom_lines)} items, ~${total:.2f} total):
{json.dumps(bom_lines, indent=2)}

{"3D PRINTING NOTED — include fastener suggestions (M3/M4 screws, heat-set inserts) in Assembly section." if has_3dprint else ""}

Write a complete .md with:
1. # Project name + tagline
2. ## Overview — specs table
3. ## Computed Requirements — table of all technical specs
4. ## Engineering Decisions — explain choices made (battery size, display spec, etc.)
5. ## Bill of Materials — table: Category | Part (markdown link) | Price | Rating | Notes
   End with **Estimated Total: $X**
6. ## Assembly Overview — numbered steps for THIS project{"  Include M3/M4 fastener and heat-set insert suggestions where relevant." if has_3dprint else ""}
7. ## Wiring & Connections — specific to this hardware
8. ## Software Setup — firmware/OS/libraries for the chosen controller
9. ## Upgrade Paths — 4-5 specific next steps

Be specific. Reference actual parts chosen. Do not be generic."""

    print("\n  Generating project document...")
    md = llm(prompt, max_tokens=2000)

    if md.startswith("[LLM error"):
        rows = "\n".join(
            f"| {b['category']} | [{b['title']}]({b['url']}) | {b['price']} | {b['rating']} | {b['note']} |"
            for b in bom_lines
        )
        md = f"""# {specs.get('name', 'Project')}\n\n## BOM\n| Category | Part | Price | Rating | Notes |\n|---|---|---|---|---|\n{rows}\n\n**Total: ${total:.2f}**"""

    # ── Prepend safety section ─────────────────────────────────────────────
    if safety_warnings:
        safety_lines = ["## ⚠️ Safety & Hazards\n"]
        for entry in safety_warnings:
            lvl  = entry["level"]
            kw   = entry["keyword"].upper()
            icon = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "ℹ️"}.get(lvl, "⚠️")
            safety_lines.append(f"### {icon} {kw} [{lvl}]\n")
            for w in entry["warnings"]:
                safety_lines.append(f"- {w}")
            safety_lines.append("")
        md = "\n".join(safety_lines) + "\n---\n\n" + md

    # ── Before You Power On checklist ──────────────────────────────────────
    checklist = _build_power_on_checklist(safety_warnings or [], pin_map or {})
    if checklist:
        md += "\n\n## Before You Power On\n\n" + checklist

    # ── Source code section ────────────────────────────────────────────────
    if firmware_result and not firmware_result.get("skipped"):
        lang = firmware_result.get("language", "cpp")
        board = firmware_result.get("board", "")
        code  = firmware_result.get("code", "")
        libs  = firmware_result.get("libraries", [])

        md += f"\n\n## Source Code\n\n_Generated starter firmware for {board}_\n"
        if libs:
            md += "\n**Required libraries:**\n"
            for lib in libs:
                md += f"- {lib}\n"
        fence = "cpp" if lang == "cpp" else "python"
        md += f"\n```{fence}\n{code}\n```\n"

    # ── Mermaid flowchart ──────────────────────────────────────────────────
    if firmware_result and firmware_result.get("mermaid"):
        mermaid = firmware_result["mermaid"]
        md += f"\n\n## Firmware Logic\n\n```mermaid\n{mermaid}\n```\n"

    # ── Datasheets & Tutorials section ─────────────────────────────────────
    ds_lines = []
    for b in bom_lines:
        ds  = b.get("datasheet")
        tut = b.get("tutorials", [])
        if ds or tut:
            ds_lines.append(f"### {b['category']}: {b['title'][:50]}")
            if ds:
                ds_lines.append(f"- **Datasheet:** [{ds}]({ds})")
            for t in tut:
                ds_lines.append(f"- **Tutorial:** [{t}]({t})")
            ds_lines.append("")
    if ds_lines:
        md += "\n\n## Datasheets & Tutorials\n\n" + "\n".join(ds_lines)

    # ── Power budget table ─────────────────────────────────────────────────
    if power_budget_result and power_budget_result.get("breakdown"):
        pb = power_budget_result
        status = "✅ Pass" if pb["pass"] else "❌ Fail"
        pb_rows = "\n".join(
            f"| {cat} | {mA:.0f}mA |" for cat, mA in pb["breakdown"].items()
        )
        battery_row = ""
        if pb.get("battery_mAh"):
            battery_row = (
                f"\n\n| Battery capacity | {pb['battery_mAh']:.0f}mAh |\n"
                f"| Max discharge (1C) | {pb['battery_max_mA']:.0f}mA |\n"
                f"| Margin | {pb.get('margin_mA', 'N/A')}mA |"
            )
        md += (
            f"\n\n## Power Budget\n\n"
            f"**Status: {status}** — Total draw: {pb['total_mA_safe']:.0f}mA (1.2× safety factor)\n\n"
            f"| Component | Current Draw |\n|---|---|\n{pb_rows}{battery_row}\n"
        )
        for w in pb.get("warnings", []):
            md += f"\n> ⚠️ {w}\n"

    return md


def _build_power_on_checklist(safety_warnings: List[Dict], pin_map: Dict) -> str:
    """Generate a deterministic pre-power-on checklist from safety + pin map."""
    items = []

    # Universal basics
    items += [
        "- [ ] Double-check all power supply polarity before connecting",
        "- [ ] Verify supply voltage matches the rated voltage for every component",
        "- [ ] Inspect all solder joints — look for bridges, cold joints, and shorts",
        "- [ ] Confirm continuity between GND rails with a multimeter",
    ]

    # Safety-driven items
    for entry in safety_warnings:
        kw  = entry["keyword"].lower()
        lvl = entry["level"]
        if kw in ("lipo", "li-ion", "lithium", "18650"):
            items.append("- [ ] Check battery voltage before first charge — reject if below 2.5V/cell")
            items.append("- [ ] Confirm LVC (low-voltage cutoff) circuit is active")
        if kw == "mains":
            items.append("- [ ] Verify mains circuit is isolated and fully enclosed before energising")
            items.append("- [ ] Confirm GFCI/RCD is installed on the mains circuit")
        if kw == "laser":
            items.append("- [ ] Put on laser safety goggles before powering laser")
            items.append("- [ ] Ensure beam path is fully enclosed or aimed at a safe target")
        if kw in ("motor", "servo", "stepper"):
            items.append("- [ ] Clear the motion envelope — no fingers or objects near moving parts")
        if kw in ("mosfet", "relay"):
            items.append("- [ ] Confirm flyback/snubber diode is installed across inductive load")

    # Pin-map driven items
    i2c_devs = [c for c, i in pin_map.items() if i.get("interface", "").upper() == "I2C"]
    if len(i2c_devs) > 1:
        items.append("- [ ] Verify I2C addresses are unique (use I2C scanner sketch before main firmware)")
    if pin_map:
        items.append("- [ ] Upload firmware and test each subsystem individually before full integration")

    # Final
    items += [
        "- [ ] Upload firmware and verify serial output before connecting high-power loads",
        "- [ ] Keep a fire extinguisher accessible when powering on for the first time",
    ]

    return "\n".join(items)


# ── Save + display ────────────────────────────────────────────────────────────

def _save_and_display(name: str, md: str):
    safe  = re.sub(r"[^\w\- ]", "", name).replace(" ", "_")
    fname = f"{safe}.md"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(md)
        path = os.path.abspath(fname)
    except Exception:
        path = f"/tmp/{fname}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)

    _header(f"PROJECT FILE: {fname}")
    print(f"  Saved to: {path}\n")
    _hr()
    for line in md.splitlines()[:50]:
        print(f"  {line}")
    _hr()
    try:
        if _read_input("\n  JARVIS: Show full file? [y/N]\n  YOU  > ").lower() in ("y", "yes"):
            print("\n" + md)
    except (EOFError, KeyboardInterrupt):
        pass


# ── Source component list (used for auto-adds) ────────────────────────────────

def _source_components(searches: List[Dict], specs: Dict, bom: List[Dict]):
    """Run Amazon sourcing for a list of search dicts, appending to bom in-place."""
    for s in searches:
        cat       = s.get("category", "Component")
        query     = s.get("query", "")
        must_meet = s.get("must_meet", "")
        if not query:
            continue

        print(f"\n  [{cat}]")
        print(f"  Query : {query}")
        print(f"  Needs : {must_meet}")
        print("  Searching...", end=" ", flush=True)

        products = amazon_search(query, max_results=5)
        if not products:
            print("no results.")
            continue
        print(f"found {len(products)}.")

        print("  Validating...")
        validated = []
        for p in products:
            v = validate_product(p, must_meet, specs)
            validated.append((p, v))
            mark = "✅" if v["pass"] else "⚠️"
            print(f"    {mark} {p.get('product_title','')[:60]}  ${p.get('product_price','?')}  — {v['reason']}")

        passing    = [(p, v) for p, v in validated if v["pass"]]
        candidates = (passing if passing else validated)[:4]

        print(f"\n  JARVIS: Best matches for {cat}:")
        for i, (p, v) in enumerate(candidates, 1):
            mark  = "✅" if v["pass"] else "⚠️"
            title = p.get("product_title", "")[:63]
            price = p.get("product_price", "?")
            stars = p.get("product_star_rating", "?")
            print(f"    [{i}] {mark} {title}  ${price}  {stars}★")
        print(f"    [0] Skip")

        while True:
            try:
                sel = input("  YOU  > ").strip()
            except (EOFError, KeyboardInterrupt):
                sel = "0"
            if sel == "0":
                break
            if sel.isdigit() and 1 <= int(sel) <= len(candidates):
                cp, cv = candidates[int(sel) - 1]
                bom.append({"category": cat, "product": cp, "validation": cv})
                print(f"  ✅ Selected: {cp.get('product_title','')[:60]}")
                break
            print(f"  (1–{len(candidates)} or 0)")


# ── Main entry ────────────────────────────────────────────────────────────────

async def run_project_mode(orchestrator=None):
    """
    Dynamic project mode. LLM asks vision-level questions only.
    User can type 'also <context>' at any point to add freeform notes.
    """
    _header("JARVIS — PROJECT MODE")
    print("  Describe your vision and I'll handle the technical details.")
    print("  At any point, type 'also <something>' to add extra context.")
    print("  Type 'cancel' to exit.\n")

    specs: Dict[str, Any] = {}
    history: List[Dict] = []

    # ── Seed questions ────────────────────────────────────────────────────────
    desc = _ask("What are you building?")
    if desc.lower() == "cancel":
        return

    # Handle 'also' even on first answer
    if _handle_also(desc, specs, history):
        desc = _ask("And what are you building?")
    specs["description"] = desc

    name = _ask("What should we call this project?")
    if name.lower() == "cancel":
        return
    specs["name"] = name

    print(f"\n  Got it — {name}.")
    print("  Just answer what you know — I'll figure out the technical specs.\n")

    # ── Dynamic Q&A ──────────────────────────────────────────────────────────
    for i in range(15):  # hard cap — but LLM should stop at 6-10
        print("  [thinking...]", end="\r", flush=True)
        q_data = get_next_question(specs, history)
        print("               ", end="\r", flush=True)

        if not q_data or q_data.get("done"):
            print("  JARVIS: I have enough to work with.\n")
            break

        question   = q_data.get("question", "")
        q_type     = q_data.get("type", "text")
        options    = q_data.get("options", [])
        recommend  = q_data.get("recommendation", "")
        key        = q_data.get("key", f"answer_{i}")

        if not question:
            break

        # Collect answer — re-ask if user typed 'also'
        while True:
            if q_type == "choice" and options:
                raw = _choose(question, options, recommend)
            elif q_type == "multi" and options:
                result = _choose_multi(question, options, recommend)
                raw = result if isinstance(result, str) else (
                    ", ".join(result) if result else ""
                )
            else:
                raw = _ask(question + (f"\n  ℹ️  {recommend}" if recommend else ""))

            if str(raw).lower() == "cancel":
                print("  Cancelled.")
                return

            # Check for 'also' prefix
            note = _handle_also(str(raw), specs, history) if isinstance(raw, str) else None
            if note:
                # Re-ask the same question after noting the context
                print(f"  (Re-asking the original question with your note added)")
                continue

            # Normal answer
            answer = raw
            break

        specs[key] = answer
        history.append({"question": question, "answer": str(answer)})

    # Print what was collected
    _header("PROJECT VISION CAPTURED")
    clean = {k: v for k, v in specs.items() if not k.startswith("_")}
    for k, v in clean.items():
        print(f"  {k}: {v}")
    notes = specs.get("_user_notes", [])
    if notes:
        print(f"\n  Extra notes:")
        for n in notes:
            print(f"    • {n}")

    # ── Compute requirements ─────────────────────────────────────────────────
    _header("COMPUTING REQUIREMENTS")
    print("  Deriving technical specs from your vision...")
    req = compute_requirements(specs)

    if req.get("requirements"):
        print("\n  Computed requirements:")
        for k, v in req["requirements"].items():
            print(f"    • {k}: {v}")
    if req.get("engineering_decisions"):
        print("\n  Engineering decisions made:")
        for k, v in req["engineering_decisions"].items():
            print(f"    • {k}: {v}")
    if req.get("notes"):
        print(f"\n  ⚠️  {req['notes']}")

    # ── Budget/Quality toggle ─────────────────────────────────────────────
    _header("SOURCING MODE")
    mode_raw = _choose(
        "How should I prioritise component sourcing?",
        ["Balanced (quality + price)", "Best Quality (name brands)", "Budget / Generic (cheapest)"],
        recommendation="Balanced is best for most projects.",
    )
    if "quality" in mode_raw.lower() or mode_raw == "Best Quality (name brands)":
        specs["sourcing_mode"] = "quality"
    elif "budget" in mode_raw.lower() or "generic" in mode_raw.lower() or mode_raw == "Budget / Generic (cheapest)":
        specs["sourcing_mode"] = "budget"
    else:
        specs["sourcing_mode"] = "balanced"
    print(f"  Sourcing mode: {specs['sourcing_mode'].upper()}")

    # ── Amazon search ─────────────────────────────────────────────────────────
    if not _yn("Search Amazon for components?", True):
        _save_and_display(specs["name"], generate_markdown(specs, req, []))
        return

    _header("SEARCHING AMAZON")
    searches = build_search_queries(specs, req)

    if not searches:
        print("  Could not generate queries.")
        _save_and_display(specs["name"], generate_markdown(specs, req, []))
        return

    bom: List[Dict] = []
    _source_components(searches, specs, bom)

    # ── Phase 3.5: Electronics Validation ────────────────────────────────────
    elec_results = {"power": None, "auto_adds": [], "pin_check": None}
    pin_map = {}
    power_budget_result = None

    if _bom_has(bom, _ELECTRONICS_KW):
        _header("PHASE 3.5 — ELECTRONICS VALIDATION")
        elec_results = run_electronics_validation(bom, req, specs)
        power_budget_result = elec_results.get("power")
        if elec_results.get("pin_check"):
            pin_map = elec_results["pin_check"].get("pin_map", {})

        # Auto-source missing converters (max 1 extra round)
        auto_adds = elec_results.get("auto_adds", [])
        if auto_adds and _yn(
            f"Auto-source {len(auto_adds)} missing power converter(s)?", True
        ):
            _header("AUTO-SOURCING MISSING COMPONENTS")
            _source_components(auto_adds, specs, bom)
    else:
        print("\n  (Electronics validation skipped — no MCU/motor/battery detected)")

    # ── Phase 3.6: Datasheet & Tutorial Fetch ─────────────────────────────────
    if orchestrator is not None and bom and _yn(
        "Fetch datasheets and tutorials for each component?", False
    ):
        bom = await fetch_datasheets_and_tutorials(bom, orchestrator)

    # ── Phase 3.7: Safety Check ────────────────────────────────────────────────
    safety_warnings = run_safety_check(bom)
    if safety_warnings:
        print_safety_warnings(safety_warnings)
    else:
        print("\n  ✅ No safety hazards detected in BOM.")

    # ── Firmware Generation ────────────────────────────────────────────────────
    firmware_result = None
    if _bom_has(bom, _MCU_KW) or _specs_has(specs, _MCU_KW):
        if _yn("Generate starter firmware for this project?", True):
            from firmware_generator import generate_firmware
            _header("GENERATING FIRMWARE")
            print(f"  Detecting MCU and libraries...")
            firmware_result = generate_firmware(specs, bom, pin_map)
            board = firmware_result.get("board", "MCU")
            lang  = firmware_result.get("language", "?")
            libs  = firmware_result.get("libraries", [])
            print(f"  Board:     {board}")
            print(f"  Language:  {lang}")
            if libs:
                print(f"  Libraries: {', '.join(libs)}")
            print("  ✅ Firmware generated.")
    else:
        print("\n  (Firmware generation skipped — no MCU detected)")

    # ── Generate & save ───────────────────────────────────────────────────────
    _header("GENERATING PROJECT FILE")
    md = generate_markdown(
        specs,
        req,
        bom,
        safety_warnings=safety_warnings,
        pin_map=pin_map,
        firmware_result=firmware_result,
        power_budget_result=power_budget_result,
    )
    _save_and_display(specs["name"], md)


if __name__ == "__main__":
    asyncio.run(run_project_mode())
