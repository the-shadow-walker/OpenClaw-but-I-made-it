"""
gui_profiles.py — Task profile store for the GUI agent.

Profiles are markdown files in ~/.agent_bin/gui_profiles/.
Each profile documents how to accomplish a specific task on a specific site/app —
login flows, form sequences, navigation patterns, known gotchas.

Before each run, ProfileStore.search() sends a compact catalog of all profiles to
the LLM and asks it to select the ones relevant to the current task. Matched profiles
are injected into the system prompt so the agent starts with site-specific knowledge.

After a successful run, the agent can call save_profile tool to persist what it learned.
"""

import json
import os
import re
import subprocess
from typing import List, Dict

PROFILES_DIR = os.path.expanduser("~/.agent_bin/gui_profiles/")
_OLLAMA_MODEL = "qwen3.6:35b-chain"


def _call_ollama(prompt: str, model: str = _OLLAMA_MODEL, timeout: int = 30) -> str:
    """One-shot text call to local Ollama. Returns '' on any failure."""
    request_data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0, "num_ctx": 2048},
    }
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/chat",
             "-d", json.dumps(request_data)],
            capture_output=True, text=True, timeout=timeout,
        )
        content = json.loads(result.stdout)["message"]["content"]
        # Strip <think> blocks (qwen3 reasoning models)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception:
        return ""


class ProfileStore:
    def __init__(self, profiles_dir: str = PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(self.profiles_dir, exist_ok=True)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _all_profiles(self) -> List[Dict]:
        """Load all profiles. Returns list of {name, content, header}."""
        profiles = []
        try:
            for fname in sorted(os.listdir(self.profiles_dir)):
                if not fname.endswith(".md"):
                    continue
                path = os.path.join(self.profiles_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Extract header: first 5 non-empty lines (name, site, task)
                    header_lines = [l for l in content.splitlines() if l.strip()][:5]
                    profiles.append({
                        "name":    fname[:-3],
                        "content": content,
                        "header":  "\n".join(header_lines),
                    })
                except Exception:
                    continue
        except FileNotFoundError:
            pass
        return profiles

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, task_description: str, top_n: int = 2) -> List[Dict]:
        """Find relevant profiles using LLM semantic matching.

        Sends a compact catalog (name + header) to the LLM and asks it to return
        a JSON list of the most relevant profile names for the task.
        Returns list of dicts: {name, content} for matched profiles.
        Falls back to [] if no profiles exist or Ollama is unreachable.
        """
        all_profiles = self._all_profiles()
        if not all_profiles:
            return []

        # Build compact catalog for the LLM — name + first 5 lines only
        catalog_lines = []
        for p in all_profiles:
            catalog_lines.append(f"PROFILE: {p['name']}\n{p['header']}\n")
        catalog = "\n".join(catalog_lines)

        prompt = (
            f"You are selecting task profiles to help an automation agent.\n\n"
            f"CURRENT TASK: {task_description}\n\n"
            f"AVAILABLE PROFILES:\n{catalog}\n"
            f"Which profiles (if any) are relevant to the current task? "
            f"Select at most {top_n}. Return ONLY a JSON array of profile name strings, "
            f'e.g. ["name1", "name2"] or [] if none are relevant. No explanation.'
        )

        response = _call_ollama(prompt)
        if not response:
            return []

        # Parse JSON array from LLM response
        try:
            # Find the first [...] block in the response
            m = re.search(r"\[.*?\]", response, re.DOTALL)
            if not m:
                return []
            selected_names = json.loads(m.group())
            if not isinstance(selected_names, list):
                return []
        except (json.JSONDecodeError, ValueError):
            return []

        # Look up full content for each selected name
        name_to_profile = {p["name"]: p for p in all_profiles}
        results = []
        for name in selected_names[:top_n]:
            if name in name_to_profile:
                p = name_to_profile[name]
                results.append({"name": p["name"], "content": p["content"]})

        return results

    def load(self, profile_name: str) -> str:
        """Load a profile by name (without .md extension). Returns '' if not found."""
        path = os.path.join(self.profiles_dir, f"{profile_name}.md")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def save(self, profile_name: str, site: str, task: str,
             steps: str, notes: str = "") -> str:
        """Save a task profile. Overwrites existing. Returns the saved filename stem."""
        safe = re.sub(r"[^a-z0-9_\-]", "_", profile_name.lower().strip())
        safe = re.sub(r"_+", "_", safe).strip("_") or "unnamed"
        content = (
            f"# Profile: {safe}\n"
            f"Site: {site}\n"
            f"Task: {task}\n\n"
            f"## Steps\n{steps.strip()}\n"
        )
        if notes.strip():
            content += f"\n## Notes\n{notes.strip()}\n"
        path = os.path.join(self.profiles_dir, f"{safe}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return safe

    def format_for_prompt(self, profiles: List[Dict], max_chars: int = 2500) -> str:
        """Format matched profiles as a compact block for system prompt injection."""
        if not profiles:
            return "(none found — document this task with save_profile after completion)"
        parts = []
        total = 0
        for p in profiles:
            header = f"--- Profile: {p['name']} ---"
            chunk = f"{header}\n{p['content']}"
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    chunk = chunk[:remaining] + "\n...(truncated)"
                else:
                    break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)
