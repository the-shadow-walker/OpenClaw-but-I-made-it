"""
Background Worker for JARVIS Distributed System
Autonomous background thread that processes projects during idle periods
Extracted and adapted from Jarvis.py BackgroundWorker class
"""

import threading
import time
import re
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from config.server_config import (
    OLLAMA_HOST, MODELS, PROJECTS_DIR, TASKS_DIR,
    IDLE_THRESHOLD_SECONDS, WORKER_CHECK_INTERVAL,
    USER_PROFILE_FILE
)


class BackgroundWorker(threading.Thread):
    """
    Autonomous background thread that wakes up during idle periods
    (>60s since last user interaction) and works on active projects.

    Logic:
      1. Find the first active project
      2. Read its .md file and look for unchecked todos (- [ ])
      3. If unchecked todos exist → execute the top one
      4. If NO todos exist → generate a detailed todo list
      5. If no projects exist → run email self-refinement
    """

    IDLE_THRESHOLD = IDLE_THRESHOLD_SECONDS
    POLL_INTERVAL = WORKER_CHECK_INTERVAL
    WORK_COOLDOWN = 120  # seconds to wait after completing a work cycle

    def __init__(self, server):
        super().__init__(daemon=True, name="background-worker")
        self.server = server  # Reference to JarvisServer instance
        self._last_work_at = 0.0
        self._last_refine_time = 0.0
        self._last_user_interaction = time.time()
        self._running = True

    # ── Helpers ────────────────────────────────────────────────────────────────

    def update_activity(self):
        """Called by server when user sends a message"""
        self._last_user_interaction = time.time()

    def _is_idle(self) -> bool:
        elapsed = time.time() - self._last_user_interaction
        return elapsed >= self.IDLE_THRESHOLD

    def _cooldown_done(self) -> bool:
        return (time.time() - self._last_work_at) >= self.WORK_COOLDOWN

    def _first_unchecked_todo(self, md_text: str) -> Optional[str]:
        """Return the first unchecked checkbox line, or None"""
        for line in md_text.splitlines():
            if line.strip().startswith("- [ ]"):
                return line.strip()
        return None

    def _list_projects(self):
        """Get list of active project .md files"""
        if not PROJECTS_DIR.exists():
            return []
        return [p.stem for p in PROJECTS_DIR.glob("*.md")]

    def _get_project_content(self, project_name: str) -> Optional[str]:
        """Read project markdown file"""
        path = PROJECTS_DIR / f"{project_name}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    # ── Autonomous Mode ────────────────────────────────────────────────────────

    def autonomous_mode(self):
        """Called once per idle cycle. Does one unit of work then returns"""
        projects = self._list_projects()

        if not projects:
            # No active projects — use idle time to learn about the user
            self._email_self_refine()
            return

        project_name = projects[0]
        md_text = self._get_project_content(project_name)
        if md_text is None:
            return

        todo = self._first_unchecked_todo(md_text)

        if todo:
            # Unchecked todo found → execute it
            print(f"\n🤖 [BG] Executing todo on '{project_name}': {todo}")
            result = self.execute_task(todo, project_name)
            self._mark_todo_done(project_name, todo, result)
        else:
            # No todos → generate a detailed plan
            print(f"\n🤖 [BG] No todos for '{project_name}'. Generating plan...")
            self._generate_todo_list(project_name, md_text)

        self._last_work_at = time.time()

    # ── Email Self-Refinement ──────────────────────────────────────────────────

    def _email_self_refine(self):
        """
        When no projects are active, fetch new emails and ask the Reasoning Model
        to analyze them (+ recent conversation history) to refine the User Profile
        """
        # 4-hour cooldown (14400 seconds) to prevent API spam
        if time.time() - self._last_refine_time < 14400:
            return
        self._last_refine_time = time.time()

        print("\n🤖 [BG] No active projects — running email self-refinement...")

        # 1. Fetch new emails (best-effort; fails silently if not configured)
        try:
            new_count = self.server.email_agent.fetch_and_process()
            if new_count:
                print(f"   📧 [BG] Processed {new_count} new emails")
        except Exception as e:
            print(f"   📧 [BG] Email fetch skipped: {e}")

        # 2. Gather recent email notes
        email_context = ""
        try:
            notes = self.server.email_agent.get_recent_notes(n=20)
            if notes:
                lines = []
                for n in notes:
                    imp_label = {3: "HIGH", 2: "MED", 1: "LOW"}.get(n.get("importance", 1), "LOW")
                    lines.append(f"[{imp_label}] {n.get('subject','')} — {n.get('note','')}")
                email_context = "\n".join(lines)
        except Exception:
            pass

        # 3. Gather recent conversation history from sessions
        # (For simplicity, we'll skip this in MVP - can be added later)
        conv_context = ""

        if not email_context and not conv_context:
            print("   🤖 [BG] No data to analyze — skipping self-refinement")
            return

        # 4. Build analysis prompt
        data_block = ""
        if email_context:
            data_block += f"RECENT EMAILS:\n{email_context}\n\n"
        if conv_context:
            data_block += f"RECENT CONVERSATIONS:\n{conv_context}"

        prompt = (
            f"{data_block}\n\n"
            "Analyze this data to refine the User Profile. "
            "What are their goals, tech stack preferences, work habits, and notable patterns?\n\n"
            "Respond with a JSON object with keys matching profile sections:\n"
            "{\n"
            '  "Work Style": ["insight 1", "insight 2"],\n'
            '  "Tech Stack": ["insight 1"],\n'
            '  "Interests & Hobbies": ["insight 1"],\n'
            '  "Observations": ["insight 1", "insight 2"]\n'
            "}\n"
            "Only include sections where you have genuine, specific insights. "
            "No filler. No preamble. JSON only."
        )

        # 5. Call chat model (keep VRAM free for interactive queries)
        try:
            raw = self._call_ollama(MODELS["chat"], prompt)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        except Exception as e:
            print(f"   🤖 [BG] Reasoning model error: {e}")
            return

        # 6. Parse JSON and append insights to user_profile.md
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                print("   🤖 [BG] Could not parse profile insights JSON")
                return
            insights = json.loads(match.group())

            # Append to user_profile.md
            written = self._append_profile_insights(insights)
            print(f"   🤖 [BG] Self-refinement complete — {written} insights added to user_profile.md")
        except Exception as e:
            print(f"   🤖 [BG] Failed to write profile insights: {e}")

    def _append_profile_insights(self, insights: dict) -> int:
        """Append insights to user_profile.md"""
        if not USER_PROFILE_FILE.exists():
            USER_PROFILE_FILE.write_text("# User Profile\n\n", encoding="utf-8")

        content = USER_PROFILE_FILE.read_text(encoding="utf-8")
        written = 0

        for section, items in insights.items():
            if isinstance(items, list):
                # Find or create section
                section_header = f"## {section}"
                if section_header not in content:
                    content += f"\n{section_header}\n\n"

                for item in items:
                    if isinstance(item, str) and item.strip():
                        # Append bullet point with timestamp
                        ts = datetime.now().strftime("%Y-%m-%d")
                        content += f"- {item.strip()} (discovered {ts})\n"
                        written += 1

        USER_PROFILE_FILE.write_text(content, encoding="utf-8")
        return written

    # ── Task Execution ─────────────────────────────────────────────────────────

    def execute_task(self, task_text: str, project_name: str) -> str:
        """
        Route a todo item to the appropriate agent/model and return the result

        Routing rules (checked in order):
          research/find/search/investigate → Reasoning model (Deep Search would go here)
          code/script/write/implement → Coding model → save file
          server/deploy/run/start/install → Draft command, log only
          anything else → Reasoning model
        """
        task_lower = task_text.lower()

        try:
            if any(kw in task_lower for kw in
                   ['research', 'find', 'search', 'investigate', 'look up', 'compare']):
                return self._execute_research(task_text, project_name)

            elif any(kw in task_lower for kw in
                     ['code', 'script', 'write', 'implement', 'function', 'program', 'class']):
                return self._execute_coding(task_text, project_name)

            elif any(kw in task_lower for kw in
                     ['server', 'deploy', 'run', 'start', 'install', 'configure', 'setup']):
                return self._execute_server_task(task_text)

            else:
                return self._execute_reasoning(task_text, project_name)

        except Exception as e:
            return f"Error during execution: {e}"

    def _execute_research(self, task_text: str, project_name: str) -> str:
        """Use reasoning model for research tasks"""
        query = re.sub(r'^-\s*\[.\]\s*', '', task_text).strip()
        print(f"   🔍 [BG] Researching: {query[:60]}...")

        prompt = f"Research task for project '{project_name}':\n\n{query}\n\nProvide detailed findings with sources where applicable."

        try:
            result = self._call_ollama(MODELS["chat"], prompt)
            self._save_research_file(project_name, query, result)
            preview = result[:500] + "..." if len(result) > 500 else result
            return f"Research complete. Full results saved to project files.\n\nSummary:\n{preview}"
        except Exception as e:
            return f"Research error: {e}"

    def _execute_coding(self, task_text: str, project_name: str) -> str:
        """Generate code with the coding model and save it to a file"""
        query = re.sub(r'^-\s*\[.\]\s*', '', task_text).strip()
        print(f"   💻 [BG] Coding: {query[:60]}...")

        prompt = f"Project context: {project_name}\n\nTask: {query}\n\nWrite clean, well-commented code. Output ONLY the code, no explanation."

        try:
            response = self._call_ollama(MODELS["chat"], prompt)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()

            saved_path = self._save_code_file(project_name, query, response)
            return f"Code generated and saved to `{saved_path}`.\n\n```\n{response[:400]}\n```"
        except Exception as e:
            return f"Coding error: {e}"

    def _execute_server_task(self, task_text: str) -> str:
        """Draft a server command but DO NOT execute — log it as ready"""
        query = re.sub(r'^-\s*\[.\]\s*', '', task_text).strip()
        print(f"   🖥️  [BG] Server task drafted (not auto-executed): {query[:60]}...")

        return (
            f"⚠️  SERVER TASK — Ready to execute (requires manual approval):\n\n"
            f"Task: {query}\n\n"
            f"To execute, tell JARVIS: \"run the server task: {query}\""
        )

    def _execute_reasoning(self, task_text: str, project_name: str) -> str:
        """Use the Reasoning Model for analysis, planning, or design tasks"""
        query = re.sub(r'^-\s*\[.\]\s*', '', task_text).strip()
        print(f"   🧠 [BG] Reasoning: {query[:60]}...")

        md_text = self._get_project_content(project_name) or ""
        prompt = (
            f"Project: {project_name}\n\n"
            f"Project file:\n{md_text[:1000]}\n\n"
            f"Task to complete:\n{query}\n\n"
            "Provide detailed, actionable output."
        )

        try:
            response = self._call_ollama(MODELS["chat"], prompt)
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
            return response
        except Exception as e:
            return f"Reasoning error: {e}"

    def _call_ollama(self, model: str, prompt: str) -> str:
        """Make a simple Ollama API call"""
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json()["response"]

    # ── File Helpers ───────────────────────────────────────────────────────────

    def _project_files_dir(self, project_name: str) -> Path:
        """Return (and create) the files directory for a project"""
        safe = re.sub(r'[^\w\-]', '_', project_name.strip())
        d = PROJECTS_DIR / f"{safe}_files"
        d.mkdir(exist_ok=True)
        return d

    def _save_code_file(self, project_name: str, task: str, code: str) -> str:
        """Save generated code to the project files directory"""
        slug = re.sub(r'[^\w]', '_', task[:40].strip().lower()).strip('_')
        ext = ".sh" if code.strip().startswith("#!") else \
              ".py" if ("def " in code or "import " in code) else \
              ".js" if ("function " in code or "const " in code) else ".txt"
        filename = f"{slug}{ext}"
        path = self._project_files_dir(project_name) / filename
        path.write_text(code, encoding="utf-8")
        return str(path)

    def _save_research_file(self, project_name: str, query: str, content: str):
        """Save full research result to the project files directory"""
        slug = re.sub(r'[^\w]', '_', query[:40].strip().lower()).strip('_')
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        path = self._project_files_dir(project_name) / f"research_{slug}_{ts}.md"
        path.write_text(f"# Research: {query}\n\n{content}", encoding="utf-8")

    # ── File Update ────────────────────────────────────────────────────────────

    def _mark_todo_done(self, project_name: str, todo_text: str, result: str):
        """Mark the todo checkbox as [x] and append a Work Log entry"""
        path = PROJECTS_DIR / f"{project_name}.md"
        if not path.exists():
            return

        text = path.read_text(encoding="utf-8")

        # Mark checkbox done
        escaped = re.escape(todo_text.strip())
        text = re.sub(
            rf'^({escaped})',
            lambda m: m.group(1).replace('- [ ]', '- [x]', 1),
            text,
            count=1,
            flags=re.MULTILINE,
        )

        # Append to Work Log
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        task_desc = re.sub(r'^-\s*\[.\]\s*', '', todo_text).strip()
        log_entry = f"\n### {ts} — {task_desc[:60]}\n{result.strip()}\n"

        work_log_header = "## Work Log"
        if work_log_header in text:
            text = text.rstrip() + "\n" + log_entry
        else:
            text = text.rstrip() + f"\n\n## Work Log\n{log_entry}"

        path.write_text(text, encoding="utf-8")
        print(f"   ✅ [BG] Task done and logged for '{project_name}'")

    def _generate_todo_list(self, project_name: str, md_text: str):
        """Call the Reasoning Model to create a detailed todo list and save it"""
        prompt = f"""You are a project planning assistant. Analyze this project and create a detailed, actionable todo list.

PROJECT FILE:
{md_text}

Generate a comprehensive todo list as GitHub-style markdown checkboxes.
Group items logically (e.g. Research, Design, Implementation, Testing).
Be specific — each item should be a concrete action, not vague.

Respond with ONLY the todo list items, like:
- [ ] Research X to determine Y
- [ ] Design the Z component
- [ ] Write script to handle W
(no preamble, no explanation — just the checkbox lines)"""

        try:
            result = self._call_ollama(MODELS["chat"], prompt)
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL | re.IGNORECASE).strip()

            if result:
                # Update project file with new todo list
                path = PROJECTS_DIR / f"{project_name}.md"
                content = path.read_text(encoding="utf-8")

                # Replace or add Todo List section
                todo_header = "## Todo List"
                if todo_header in content:
                    # Replace existing section
                    pattern = r'(## Todo List)(.*?)(?=\n## |\Z)'
                    content = re.sub(pattern, f"\\1\n{result}\n", content, flags=re.DOTALL)
                else:
                    # Append new section
                    content += f"\n\n## Todo List\n{result}\n"

                path.write_text(content, encoding="utf-8")
                print(f"   ✅ [BG] Todo list written for '{project_name}'")
        except Exception as e:
            print(f"   ⚠️  [BG] Todo generation failed: {e}")

    # ── Thread Main Loop ───────────────────────────────────────────────────────

    def run(self):
        """Main background worker loop"""
        while self._running:
            try:
                if self._is_idle() and self._cooldown_done():
                    self.autonomous_mode()
            except Exception as e:
                print(f"   ⚠️  [BG] Worker error: {e}")
            time.sleep(self.POLL_INTERVAL)

    def stop(self):
        """Gracefully stop the worker"""
        self._running = False
