"""
gui_agent.py — GUIAgent: vision-based desktop automation powered by qwen3.6:35b-Grindlewalt.

The agent takes screenshots, overlays an 8×8 grid, runs OCR for text positions,
sends the annotated image to the vision model (Ollama vision API), receives a JSON action,
executes it via xdotool, and loops.
"""

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


# ── System prompt ─────────────────────────────────────────────────────────────

GUI_SYSTEM_PROMPT_TEMPLATE = """\
You are a desktop automation agent controlling a headless Linux server desktop.

═══════════════════════ ENVIRONMENT ═══════════════════════
OS: Arch Linux server (no physical monitor)
Virtual display: Xvfb :99 — a full X11 display running at 1280×720 pixels
Window manager: kwin_x11 — already running, handles window placement and focus
Baseline terminal: xterm — always visible on screen (black bg, white text, monospace font)
DISPLAY env is already set to :99 — any app you launch from xterm renders in this display

WHAT YOU CAN DO:
• Open a web browser: click xterm to focus it, type "firefox &" or "chromium &", press Enter.
  The browser WILL render and appear on screen — this is a fully functional X11 display.
• Run any GUI or terminal application: type its name in xterm, press Enter.
• Navigate websites, fill forms, click buttons — exactly as on a physical desktop.
• Use keyboard shortcuts: Ctrl+L (browser address bar), Ctrl+T (new tab), Ctrl+W (close tab),
  Ctrl+A (select all), Ctrl+C/V (copy/paste), Return (confirm), Escape (cancel), Tab (next field).

COORDINATE SYSTEM — 8×8 GRID:
The screen is divided into an 8×8 grid. top-left=(0,0), bottom-right=(8,8).
Decimals are required for precision: (3.5, 1.2) not (3, 1).
Column 0=left edge → Column 8=right edge
Row 0=top edge → Row 8=bottom edge

Approximate landmarks on a 1280×720 display:
  xterm window body         ≈ (0.1, 0.1) to (7.9, 7.5)
  xterm prompt/text area    ≈ (0.5, 4.0)  [center of terminal]
  browser title bar         ≈ (4.0, 0.15)
  browser address bar       ≈ (4.0, 0.5)  [after browser opens]
  browser close button (X)  ≈ (7.85, 0.15)
  browser page body         ≈ (4.0, 4.0)
  browser tabs bar          ≈ (2.0, 0.2)

OCR TEXT POSITIONS (read these carefully every screenshot):
After each screenshot you receive a list like:
  "File"     @ grid 0.40, 0.10   ← click x=0.40, y=0.10 to activate "File"
  "http://…" @ grid 4.00, 0.50   ← browser address bar is here
  "Submit"   @ grid 3.80, 6.20   ← button is here
ALWAYS prefer OCR coordinates for text elements — they are pixel-accurate.
If a text element is missing from OCR, use the image grid overlay to estimate.

HOW TO USE XTERM:
1. Click inside the xterm window body to give it keyboard focus
2. Type your command (you will see it appear on screen after screenshot)
3. Press Enter to run it
4. Call screenshot to see the result
5. Background apps: append " &" so xterm stays responsive (e.g. "firefox &")
6. The shell prompt looks like "$ " or "% " — you are ready when you see it

HOW TO USE A BROWSER:
Step 1 — Launch:    click xterm → type "firefox &" → key Return → wait 3s → screenshot
Step 2 — Navigate:  key {{"combo": "ctrl+l"}} → type {{"text": "https://example.com"}} → key {{"combo": "Return"}}
Step 3 — Interact:  screenshot to see page → use OCR coords to click links/buttons/fields
Step 4 — Type text: click the input field first → screenshot to confirm focus → type {{"text": "..."}}
Step 5 — Scroll:    scroll {{"direction": "down"}} when the browser page body is in focus

═══════════════════════ TASK & BUDGET ═══════════════════════
CURRENT TASK: {task}
ITERATION BUDGET: {max_iterations} total iterations
WARNING: At {budget_warn} iterations remaining, call finish() immediately with current progress.

═══════════════════════ AVAILABLE TOOLS ═══════════════════════
{available_tools}

═══════════════════════ OUTPUT FORMAT ═══════════════════════
Every single response MUST be one valid JSON object — NO prose, NO markdown, NOTHING else:
{{"thought": "step-by-step reasoning about what you see and what to do next", "confidence": 85, "tool": "tool_name", "args": {{}}}}

confidence 0–100: your certainty that this action is correct.
If confidence < 70: call screenshot to gather more evidence before acting.

═══════════════════════ RULES ═══════════════════════
1.  ALWAYS call screenshot first — never act blind
2.  After EVERY action (click, type, key, scroll), call screenshot to verify the result
3.  Use OCR coordinates for text — do not invent pixel positions
4.  To type in any field: click it first, verify focus in screenshot, then type
5.  If a click misses: screenshot → re-read OCR list → recalculate → retry
6.  Waiting for app to load: wait {{"seconds": 3}} then screenshot
7.  confidence < 70: screenshot first, act second
8.  NEVER repeat the exact same tool+args 4 times in a row — try a different approach
9.  Task complete: finish {{"summary": "what was accomplished", "success": true}}
10. Irreversibly stuck / impossible: finish {{"summary": "what failed and why", "success": false}}
11. Do NOT close xterm — it is your only way to launch new applications

Begin by calling screenshot to see the current screen state.
"""


# ── Vision call ───────────────────────────────────────────────────────────────

def _call_vision(model: str, prompt: str, image_b64: str,
                 system: str = None, timeout: int = 90) -> str:
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
        return "Vision call timed out after 90s — try again"
    except (json.JSONDecodeError, KeyError) as e:
        stderr = result.stderr[:200] if result else ""
        return f"Vision call failed (parse error): {e}  stderr={stderr}"
    except Exception as e:
        return f"Vision call failed: {e}"


# ── GUIAgent ──────────────────────────────────────────────────────────────────

class GUIAgent:
    MODEL = "qwen3.6:35b-Grindlewalt"

    def __init__(self, display=":99", screen_w=1280, screen_h=720, event_cb=None):
        self.display = display
        self.screen_w = screen_w
        self.screen_h = screen_h
        # Optional callback: event_cb(event_type: str, payload: dict)
        # Called for screenshot, action, result, vision, error events.
        self.event_cb = event_cb

        self.agent = OllamaCommandAgent(model=self.MODEL, fast_model=self.MODEL)
        self.screen = GUIScreen(display, screen_w, screen_h)
        self.input_ctrl = GUIInput(display, screen_w, screen_h)

        self.agent.tool_registry = GUIToolRegistry(
            screen=self.screen,
            input_ctrl=self.input_ctrl,
            call_vision_fn=lambda prompt, img: _call_vision(self.MODEL, prompt, img),
            event_cb=event_cb,
            safety_validator=self.agent.safety_validator,
            search_agent=self.agent.search_agent,
            memory=self.agent.memory,
        )

    def setup_display(self):
        """Ensure the virtual display is up with a window manager and terminal.

        Steps:
          1. Start Xvfb :99 if the display isn't alive yet.
          2. Start kwin_x11 if no window manager is running — needed so windows
             can be moved/resized and don't stack behind each other.
          3. Start xterm so there is always something visible on the screen
             (avoids the pure-black-screen problem on a headless server).
        """
        env = {**os.environ, "DISPLAY": self.display}

        # ── 1. Xvfb ─────────────────────────────────────────────────────────
        r = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            env=env, capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
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
        else:
            print(f"  Display {self.display} active: {r.stdout.strip()}")

        # ── 2. Window manager (kwin_x11, from installed KDE) ────────────────
        # pgrep searches by process name, not display — good enough since this
        # server only runs one virtual display.
        wm_running = subprocess.run(
            ["pgrep", "-x", "kwin_x11"],
            capture_output=True,
        ).returncode == 0

        if not wm_running:
            print("  Starting kwin_x11 window manager...")
            subprocess.Popen(
                ["kwin_x11"],
                env={**env, "DISPLAY": self.display},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(1.2)
            print("  kwin_x11 ready")
        else:
            print("  kwin_x11 already running")

        # ── 3. xterm baseline ────────────────────────────────────────────────
        # Always start a terminal so the display is never pitch-black.
        # Uses a geometry that fills most of the virtual screen.
        xterm_running = subprocess.run(
            ["pgrep", "-x", "xterm"],
            capture_output=True,
        ).returncode == 0

        if not xterm_running:
            print("  Starting xterm...")
            subprocess.Popen(
                ["xterm", "-geometry", "155x42+20+20",
                 "-bg", "black", "-fg", "white",
                 "-fa", "Monospace", "-fs", "11"],
                env={**env, "DISPLAY": self.display},
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)
            print("  xterm ready")
        else:
            print("  xterm already running")

    def run(self, task: str, max_iterations: int = 30) -> dict:
        """Run the GUI agent on a task. Returns run_react result dict."""
        self.setup_display()

        budget_warn = max(5, max_iterations // 5)
        system_prompt = GUI_SYSTEM_PROMPT_TEMPLATE.format(
            task=task,
            available_tools=GUI_TOOLS_TEXT,
            max_iterations=max_iterations,
            budget_warn=budget_warn,
        )

        self.agent.max_react_iterations = max_iterations
        self.agent.react_trace = []  # fresh trace for this run

        # Start trace watcher: polls react_trace and emits "thought" events
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

        # Emit done event
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
    task = " ".join(sys.argv[1:]) or "open xterm and type hello"
    print(f"\nGUI Agent — model: {GUIAgent.MODEL}")
    print(f"Task: {task}\n")
    agent = GUIAgent()
    result = agent.run(task)
    success = result.get("success", False)
    summary = result.get("summary", "")
    print(f"\n{'✅' if success else '⚠️ '} {'SUCCESS' if success else 'FINISHED'}")
    if summary:
        print(f"   {summary}")
