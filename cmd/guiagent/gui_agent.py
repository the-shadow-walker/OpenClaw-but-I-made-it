"""
gui_agent.py — GUIAgent: vision-based desktop automation powered by qwen3.6:35b-Grindlewalt.

The agent takes screenshots, overlays an 8×8 grid, runs OCR for text positions,
sends the annotated image to the vision model (Ollama vision API), receives a JSON action,
executes it via xdotool, and loops.
"""

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

from ollama_agent_core import OllamaCommandAgent
from gui_screen import GUIScreen
from gui_input import GUIInput
from gui_tools import GUIToolRegistry, GUI_TOOLS_TEXT


# ── System prompts ─────────────────────────────────────────────────────────────

# Real KDE desktop (:0) — full Plasma environment
_KDE_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a KDE Plasma desktop on a Linux server.

═══════════════════════ ENVIRONMENT ═══════════════════════
OS: Arch Linux with KDE Plasma (X11 session, real desktop)
Display: :0 at 1920×1080 — the real user desktop with wallpaper, panels, and running apps
Window manager: kwin_x11
Browser: Brave Browser (command: brave-browser)
Terminal: Konsole (KDE's terminal — search for it in the app launcher)

WHAT YOU CAN DO:
• Open the application launcher: click the grid/menu icon in the taskbar (bottom-left area)
  OR press the Super key (Windows key) to open KRunner/launcher
• Open Brave Browser: right-click the desktop → Run Command → type "brave-browser", Enter
  OR click its icon in the taskbar, OR open a terminal and type "brave-browser &"
• Open a terminal: search for "Konsole" in the app launcher, or right-click desktop
• Control any open application: click on it, type, scroll, use keyboard shortcuts
• Use keyboard shortcuts: Ctrl+L (browser address bar), Ctrl+T (new tab), Super (launcher),
  Alt+F2 (KRunner run dialog), Ctrl+Alt+T (terminal shortcut if configured)

COORDINATE SYSTEM — 8×8 GRID:
The screen is divided into an 8×8 grid. top-left=(0,0), bottom-right=(8,8).
Decimals required for precision: (3.5, 1.2) not (3, 1).
Column 0=left edge → Column 8=right edge; Row 0=top → Row 8=bottom

Approximate KDE Plasma landmarks at 1920×1080:
  KDE taskbar (bottom panel)  ≈ y=7.7 to y=8.0
  App launcher button         ≈ (0.15, 7.85)
  Desktop body                ≈ (0.0, 0.0) to (8.0, 7.7)
  Browser address bar         ≈ (4.0, 0.5) [when browser is focused]
  Browser close button        ≈ (7.9, 0.15)

OCR TEXT POSITIONS (read every screenshot):
  "Firefox"  @ grid 0.40, 0.10   ← click x=0.40, y=0.10
  "http://…" @ grid 4.00, 0.50
ALWAYS use OCR coordinates for text elements — they are pixel-accurate.

HOW TO OPEN A BROWSER:
Option A: Alt+F2 → type "brave-browser" → Enter
Option B: Click app launcher → search "Brave" → click result
Option C: If a terminal is open: type "brave-browser &" → Enter

HOW TO NAVIGATE A BROWSER:
After opening: key {{"combo": "ctrl+l"}} → type URL → key {{"combo": "Return"}}
New tab: key {{"combo": "ctrl+t"}}
Scroll: scroll {{"direction": "down"}}

HOW TO OPEN A TERMINAL (Konsole):
Option A: key {{"combo": "ctrl+alt+t"}}
Option B: Alt+F2 → type "konsole" → Enter
Option C: Right-click desktop → open terminal

═══════════════════════ TASK & BUDGET ═══════════════════════
CURRENT TASK: {task}
ITERATION BUDGET: {max_iterations} total iterations
At {budget_warn} iterations remaining: call finish() immediately.

═══════════════════════ AVAILABLE TOOLS ═══════════════════════
{available_tools}

═══════════════════════ OUTPUT FORMAT ═══════════════════════
Every response MUST be one valid JSON object — NO prose, nothing else:
{{"thought": "step-by-step reasoning", "confidence": 85, "tool": "tool_name", "args": {{}}}}

═══════════════════════ RULES ═══════════════════════
1.  ALWAYS call screenshot first — see before acting
2.  After EVERY action (click, type, key, scroll), call screenshot to verify
3.  Use OCR coordinates for text — never invent positions
4.  To type in a field: click it first, verify focus in screenshot, then type
5.  If a click misses: screenshot → re-read OCR → recalculate → retry
6.  Waiting for app to load: wait {{"seconds": 3}} then screenshot
7.  confidence < 70: screenshot first, act second
8.  NEVER repeat the exact same tool+args 4 times in a row
9.  Task complete: finish {{"summary": "what was accomplished", "success": true}}
10. Irreversibly stuck: finish {{"summary": "what failed and why", "success": false}}

Begin by calling screenshot to see the current screen state.
"""

# Headless virtual display (:99) — Xvfb + xterm
_HEADLESS_SYSTEM_PROMPT = """\
You are a desktop automation agent controlling a headless Linux server desktop.

═══════════════════════ ENVIRONMENT ═══════════════════════
OS: Arch Linux server (no physical monitor)
Virtual display: Xvfb :99 — a full X11 display running at 1280×720 pixels
Window manager: kwin_x11 — already running
Baseline terminal: xterm — always visible on screen (black bg, white text)
Browser: Brave Browser (command: brave-browser)

WHAT YOU CAN DO:
• Open Brave Browser: click xterm → type "brave-browser &" → Enter.
  The browser WILL render — this is a full X11 display.
  (Brave is installed — do NOT use firefox or chromium.)
• Run any GUI app: type its name in xterm, press Enter.

COORDINATE SYSTEM — 8×8 GRID:
top-left=(0,0), bottom-right=(8,8). Decimals required: (3.5, 1.2).

Approximate landmarks at 1280×720:
  xterm window body         ≈ (0.1, 0.1) to (7.9, 7.5)
  xterm text/prompt area    ≈ (0.5, 4.0)
  browser address bar       ≈ (4.0, 0.5)  [after browser opens]
  browser close button      ≈ (7.85, 0.15)

OCR TEXT POSITIONS:
  "File"     @ grid 0.40, 0.10   ← click x=0.40, y=0.10
ALWAYS use OCR coordinates for text — pixel-accurate.

HOW TO USE XTERM:
1. Click inside xterm body to focus it
2. Type command, press Enter
3. Background apps: "brave-browser &" (not blocking)
4. Screenshot to verify result

HOW TO NAVIGATE A BROWSER:
Launch: type "brave-browser &" in xterm → Enter → wait 3s → screenshot
Navigate: key {{"combo": "ctrl+l"}} → type URL → key {{"combo": "Return"}}

═══════════════════════ TASK & BUDGET ═══════════════════════
CURRENT TASK: {task}
ITERATION BUDGET: {max_iterations} total iterations
At {budget_warn} iterations remaining: call finish() immediately.

═══════════════════════ AVAILABLE TOOLS ═══════════════════════
{available_tools}

═══════════════════════ OUTPUT FORMAT ═══════════════════════
Every response MUST be one valid JSON object — NO prose, nothing else:
{{"thought": "step-by-step reasoning", "confidence": 85, "tool": "tool_name", "args": {{}}}}

═══════════════════════ RULES ═══════════════════════
1.  ALWAYS call screenshot first
2.  After EVERY action, call screenshot to verify
3.  Use OCR coordinates for text
4.  To type: click field first, verify focus, then type
5.  If click misses: screenshot → re-read OCR → retry
6.  Waiting for load: wait {{"seconds": 3}} then screenshot
7.  confidence < 70: screenshot first
8.  NEVER repeat same tool+args 4× in a row
9.  Task complete: finish {{"summary": "...", "success": true}}
10. Stuck: finish {{"summary": "...", "success": false}}
11. Do NOT close xterm

Begin by calling screenshot.
"""


# ── Vision call ───────────────────────────────────────────────────────────────

def _call_vision(model: str, prompt: str, image_b64: str,
                 system: str = None, timeout: int = 180) -> str:
    """Send a base64 PNG image + text prompt to an Ollama vision model."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({
        "role": "user",
        "content": prompt,
        "images": [image_b64],
    })
    request_data = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/chat",
             "-d", json.dumps(request_data)],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = json.loads(result.stdout)
        content = raw["message"]["content"]
        # Strip <think>…</think> blocks produced by qwen3 reasoning models
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except subprocess.TimeoutExpired:
        return f"Vision call timed out after {timeout}s — try again"
    except (json.JSONDecodeError, KeyError) as e:
        stderr = result.stderr[:200] if result else ""
        return f"Vision call failed (parse error): {e}  stderr={stderr}"
    except Exception as e:
        return f"Vision call failed: {e}"


# ── GUIAgent ──────────────────────────────────────────────────────────────────

class GUIAgent:
    MODEL = "qwen3.6:35b-Grindlewalt"

    def __init__(self, display=":0", screen_w=1920, screen_h=1080,
                 event_cb=None, stop_event=None):
        self.display = display
        self.event_cb = event_cb
        self.stop_event = stop_event

        # Headless virtual display uses different resolution
        if display == ":99":
            screen_w, screen_h = 1280, 720
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.agent = OllamaCommandAgent(model=self.MODEL, fast_model=self.MODEL)
        self.screen = GUIScreen(display, screen_w, screen_h)
        self.input_ctrl = GUIInput(display, screen_w, screen_h)

        self.agent.tool_registry = GUIToolRegistry(
            screen=self.screen,
            input_ctrl=self.input_ctrl,
            event_cb=event_cb,
            stop_event=stop_event,
            safety_validator=self.agent.safety_validator,
            search_agent=self.agent.search_agent,
            memory=self.agent.memory,
        )

    def _refresh_xauth(self):
        """Find the current SDDM xauth cookie and merge it into ~/.Xauthority."""
        home = os.path.expanduser("~")
        xauth_dest = os.path.join(home, ".Xauthority")
        candidates = sorted(
            glob.glob("/tmp/xauth_*"),
            key=os.path.getmtime,
            reverse=True,
        )
        env_base = {**os.environ, "DISPLAY": self.display}
        for f in candidates:
            if not os.access(f, os.R_OK):
                continue
            # Test if this cookie gives access to the display
            r = subprocess.run(
                ["xdotool", "getdisplaygeometry"],
                env={**env_base, "XAUTHORITY": f},
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                # Merge into ~/.Xauthority
                extract = subprocess.run(
                    ["xauth", "extract", "-", self.display],
                    env={"XAUTHORITY": f},
                    capture_output=True,
                )
                subprocess.run(
                    ["xauth", "merge", "-"],
                    input=extract.stdout,
                    env={"XAUTHORITY": xauth_dest},
                    capture_output=True,
                )
                print(f"  Xauth refreshed from {f}")
                return True
        return False

    def setup_display(self):
        """Ensure the target display is accessible.

        For :0 (real KDE desktop): verify access, refresh xauth if needed.
        For :99 (headless): start Xvfb + kwin_x11 + xterm if not running.
        """
        env = {**os.environ, "DISPLAY": self.display}

        # Check if display is already accessible
        r = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            env=env, capture_output=True, text=True, timeout=5,
        )

        if r.returncode != 0:
            if self.display == ":0":
                # Real desktop: try refreshing xauth
                print("  Display :0 not accessible — refreshing xauth...")
                if self._refresh_xauth():
                    r = subprocess.run(
                        ["xdotool", "getdisplaygeometry"],
                        env=env, capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode == 0:
                        print(f"  Display :0 ready: {r.stdout.strip()}")
                        return
                raise RuntimeError(
                    "Cannot access display :0. Is KDE running? "
                    "Check: sudo systemctl status sddm"
                )
            else:
                # Headless: start Xvfb
                self._setup_headless()
                return

        geom = r.stdout.strip()
        if self.display == ":0":
            print(f"  KDE desktop :0 ready: {geom}")
        else:
            print(f"  Display {self.display} active: {geom}")
            # Headless display: ensure kwin + xterm are running
            self._ensure_headless_apps()

    def _setup_headless(self):
        """Start Xvfb :99 + kwin_x11 + xterm from scratch."""
        env = {**os.environ, "DISPLAY": self.display}
        print(f"  Starting Xvfb {self.display} ({self.screen_w}×{self.screen_h})...")
        subprocess.Popen(
            ["Xvfb", self.display, "-screen", "0",
             f"{self.screen_w}x{self.screen_h}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        r = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            env=env, capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"Xvfb failed to start on {self.display}: {r.stderr[:200]}"
            )
        print(f"  Xvfb ready: {r.stdout.strip()}")
        self._ensure_headless_apps()

    def _ensure_headless_apps(self):
        """Start kwin_x11 and xterm if not already running (headless display)."""
        env = {**os.environ, "DISPLAY": self.display}

        if subprocess.run(["pgrep", "-x", "kwin_x11"], capture_output=True).returncode != 0:
            print("  Starting kwin_x11...")
            subprocess.Popen(
                ["kwin_x11"], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(1.2)
            print("  kwin_x11 ready")

        if subprocess.run(["pgrep", "-x", "xterm"], capture_output=True).returncode != 0:
            print("  Starting xterm...")
            subprocess.Popen(
                ["xterm", "-geometry", "155x42+20+20",
                 "-bg", "black", "-fg", "white",
                 "-fa", "Monospace", "-fs", "11"],
                env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)
            print("  xterm ready")

    def run(self, task: str, max_iterations: int = 30) -> dict:
        """Run the GUI agent on a task. Returns run_react result dict."""
        self.setup_display()

        budget_warn = max(5, max_iterations // 5)
        template = _KDE_SYSTEM_PROMPT if self.display == ":0" else _HEADLESS_SYSTEM_PROMPT
        system_prompt = template.format(
            task=task,
            available_tools=GUI_TOOLS_TEXT,
            max_iterations=max_iterations,
            budget_warn=budget_warn,
        )

        self.agent.max_react_iterations = max_iterations
        self.agent.react_trace = []

        stop_watcher = threading.Event()
        if self.event_cb:
            watcher = threading.Thread(
                target=self._trace_watcher,
                args=(stop_watcher,),
                daemon=True,
            )
            watcher.start()

        try:
            result = self.agent.run_react(
                task,
                system_prompt_override=system_prompt,
                tool_whitelist=GUIToolRegistry.GUI_TOOL_NAMES,
            )
        finally:
            stop_watcher.set()

        if self.event_cb:
            try:
                self.event_cb("done", {
                    "success": result.get("success", False),
                    "summary": result.get("summary", ""),
                    "iterations": len(self.agent.react_trace),
                })
            except Exception:
                pass

        return result

    def _trace_watcher(self, stop_event: threading.Event):
        """Background thread: watches react_trace and emits thought events."""
        seen = 0
        while not stop_event.is_set():
            trace = self.agent.react_trace
            if len(trace) > seen:
                for entry in trace[seen:]:
                    thought = entry.get("thought", "")
                    if thought and self.event_cb:
                        try:
                            self.event_cb("thought", {
                                "iteration": entry.get("iteration", seen + 1),
                                "thought": thought,
                                "confidence": entry.get("confidence", 0),
                                "tool": entry.get("tool", ""),
                            })
                        except Exception:
                            pass
                seen = len(trace)
            stop_event.wait(0.1)


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "take a screenshot and describe the desktop"
    print(f"\nGUI Agent — model: {GUIAgent.MODEL}")
    print(f"Task: {task}\n")
    agent = GUIAgent()
    result = agent.run(task)
    success = result.get("success", False)
    summary = result.get("summary", "")
    print(f"\n{'OK' if success else 'DONE'}: {summary}")
