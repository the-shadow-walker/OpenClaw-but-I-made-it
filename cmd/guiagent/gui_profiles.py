"""
gui_profiles.py — Task profile store for the GUI agent.

Profiles are markdown files in ~/.agent_bin/gui_profiles/.
Each profile documents how to accomplish a specific task on a specific site/app —
login flows, form sequences, navigation patterns, known gotchas.

Before each run, ProfileStore.search() finds profiles that match the task by keyword
overlap. Matched profiles are injected into the system prompt so the agent starts
with site-specific knowledge instead of exploring from scratch.

After a successful run, the agent can call save_profile tool to persist what it learned.
"""

import os
import re
from typing import List, Dict

PROFILES_DIR = os.path.expanduser("~/.agent_bin/gui_profiles/")


class ProfileStore:
    def __init__(self, profiles_dir: str = PROFILES_DIR):
        self.profiles_dir = profiles_dir
        os.makedirs(self.profiles_dir, exist_ok=True)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> set:
        """Lowercase word set, min 3 chars, skip stopwords."""
        _STOP = {"the", "and", "for", "with", "this", "that", "from",
                 "into", "then", "will", "can", "you", "are", "its"}
        return {w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in _STOP}

    def _score(self, task_tokens: set, name: str, content: str) -> int:
        """Score a profile against task tokens. Name + tags weighted higher."""
        name_tokens    = self._tokenize(name.replace('_', ' '))
        header_tokens  = self._tokenize(content[:400])   # tags/title/site
        body_tokens    = self._tokenize(content[400:1000])
        return (
            len(task_tokens & name_tokens)   * 4 +
            len(task_tokens & header_tokens) * 2 +
            len(task_tokens & body_tokens)   * 1
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(self, task_description: str, top_n: int = 2) -> List[Dict]:
        """Find the most relevant profiles for a task.

        Returns list of dicts: {name, content, score}
        Only returns profiles with score > 0.
        """
        task_tokens = self._tokenize(task_description)
        results = []
        try:
            for fname in sorted(os.listdir(self.profiles_dir)):
                if not fname.endswith('.md'):
                    continue
                path = os.path.join(self.profiles_dir, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    name = fname[:-3]
                    score = self._score(task_tokens, name, content)
                    if score > 0:
                        results.append({'name': name, 'content': content, 'score': score})
                except Exception:
                    continue
        except FileNotFoundError:
            pass
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def load(self, profile_name: str) -> str:
        """Load a profile by name (without .md extension). Returns '' if not found."""
        path = os.path.join(self.profiles_dir, f"{profile_name}.md")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return ''

    def save(self, profile_name: str, site: str, task: str,
             steps: str, notes: str = "") -> str:
        """Save a task profile. Overwrites existing. Returns the saved filename stem."""
        safe = re.sub(r'[^a-z0-9_\-]', '_', profile_name.lower().strip())
        safe = re.sub(r'_+', '_', safe).strip('_') or 'unnamed'
        content = (
            f"# Profile: {safe}\n"
            f"Site: {site}\n"
            f"Task: {task}\n\n"
            f"## Steps\n{steps.strip()}\n"
        )
        if notes.strip():
            content += f"\n## Notes\n{notes.strip()}\n"
        path = os.path.join(self.profiles_dir, f"{safe}.md")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return safe

    def format_for_prompt(self, profiles: List[Dict], max_chars: int = 2500) -> str:
        """Format matched profiles as a compact block for system prompt injection.

        Returns '' if no profiles matched.
        """
        if not profiles:
            return "(none found — document this task with save_profile after completion)"
        parts = []
        total = 0
        for p in profiles:
            header = f"--- Profile: {p['name']} (score {p['score']}) ---"
            body = p['content']
            chunk = f"{header}\n{body}"
            if total + len(chunk) > max_chars:
                remaining = max_chars - total
                if remaining > 100:
                    chunk = chunk[:remaining] + "\n...(truncated)"
                else:
                    break
            parts.append(chunk)
            total += len(chunk)
        return "\n\n".join(parts)
