"""
gui_macros.py — Named action-sequence store for the GUI agent.

Macros are JSON files in ~/.agent_bin/gui_macros/.
Each macro is a named list of tool calls (steps) that can be replayed
in a single agent iteration via the sequence tool — DuckyScript style.

Example macro:
  {
    "name": "LOGIN_DELTAMATH",
    "description": "Login to deltamath.com with cached credentials",
    "steps": [
      {"tool": "click",     "args": {"x": 12.57, "y": 2.5}},
      {"tool": "wait",      "args": {"seconds": 1.0}},
      {"tool": "click",     "args": {"x": 8.0, "y": 7.13}},
      {"tool": "type",      "args": {"text": "user@example.com"}},
      {"tool": "key",       "args": {"combo": "Tab"}},
      {"tool": "type",      "args": {"text": "password"}},
      {"tool": "click",     "args": {"x": 8.0, "y": 9.63}},
      {"tool": "wait",      "args": {"seconds": 2.0}},
      {"tool": "screenshot","args": {}}
    ]
  }
"""

import json
import os
import re
from typing import List, Dict, Optional

MACROS_DIR = os.path.expanduser("~/.agent_bin/gui_macros/")


class MacroStore:
    def __init__(self, macros_dir: str = MACROS_DIR):
        self.macros_dir = macros_dir
        os.makedirs(self.macros_dir, exist_ok=True)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _safe_name(self, name: str) -> str:
        safe = re.sub(r"[^A-Z0-9_]", "_", name.upper().strip())
        return re.sub(r"_+", "_", safe).strip("_") or "MACRO"

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_all(self) -> List[Dict]:
        """Return all macros as [{name, description, step_count, steps}]."""
        macros = []
        try:
            for fname in sorted(os.listdir(self.macros_dir)):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(self.macros_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    macros.append({
                        "name":        data.get("name", fname[:-5]),
                        "description": data.get("description", ""),
                        "step_count":  len(data.get("steps", [])),
                        "steps":       data.get("steps", []),
                    })
                except Exception:
                    continue
        except FileNotFoundError:
            pass
        return macros

    def load(self, name: str) -> Optional[Dict]:
        """Load a macro by name (case-insensitive). Returns None if not found."""
        safe = self._safe_name(name)
        # Try exact filename match, then safe-name match
        for candidate in (f"{name}.json", f"{safe}.json", f"{name.upper()}.json"):
            path = os.path.join(self.macros_dir, candidate)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
        return None

    def save(self, name: str, description: str, steps: list) -> str:
        """Save a macro. Overwrites if exists. Returns the safe name used."""
        safe = self._safe_name(name)
        data = {
            "name":        safe,
            "description": description.strip(),
            "steps":       steps,
        }
        path = os.path.join(self.macros_dir, f"{safe}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return safe

    def format_for_prompt(self) -> str:
        """Format macro catalog for system prompt injection."""
        macros = self.list_all()
        if not macros:
            return "(none — use save_macro after a successful flow to cache it for next time)"
        lines = []
        for m in macros:
            lines.append(f"  {m['name']}: {m['description']} ({m['step_count']} steps)")
        return "\n".join(lines)
