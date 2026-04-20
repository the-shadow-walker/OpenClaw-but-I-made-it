"""
gui_agent.py — GUIAgent: vision-based desktop automation powered by qwen3.6:35b-Grindlewalt.

The agent takes screenshots, overlays a 16×16 grid, runs OCR for text positions,
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

# ── Debug log ─────────────────────────────────────────────────────────────────

GUI_DEBUG_LOG = "/tmp/gui_agent_debug.jsonl"
_log_lock = threading.Lock()


def _gui_log(entry: dict):
    """Append one JSONL entry to the GUI debug log (thread-safe)."""
    try:
        entry.setdefault("ts", time.strftime("%H:%M:%S"))
        line = json.dumps(entry, default=str) + "\n"
        with _log_lock:
            with open(GUI_DEBUG_LOG, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass

from ollama_agent_core import OllamaCommandAgent
from gui_screen import GUIScreen
from gui_input import GUIInput
from gui_tools import GUIToolRegistry, GUI_TOOLS_TEXT

_NOTES_FILE = os.path.expanduser("~/.agent_bin/gui_agent_notes.md")


def _load_notes() -> str:
    """Load persistent agent notes. Returns formatted text for prompt injection."""
    try:
        if os.path.exists(_NOTES_FILE):
            with open(_NOTES_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                return content
    except Exception:
        pass
    return "(empty — use note tool to save discoveries as you work)"


# ── System prompts ─────────────────────────────────────────────────────────────

# Real KDE desktop (:0) — full Plasma environment
_KDE_SYSTEM_PROMPT = """\
You are an AI agent on an Arch Linux workstation with KDE Plasma 6.

══════════════════ SYSTEM ══════════════════
OS: Arch Linux (rolling release, kernel 6.x)
Desktop: KDE Plasma 6, X11, display :0, 1920×1080
Shell: bash
Browser: brave-browser (Brave Browser)
Terminal app: Konsole (konsole)
File manager: Dolphin
Package manager: pacman / yay
Init: systemd

══════════════════ TOOL STRATEGY ══════════════════
cmd FIRST — run shell commands whenever possible. GUI tools are a last resort.

  Open a website:    cmd {{"command": "/usr/bin/brave 'https://url' >/dev/null 2>&1 &"}}
  Dismiss a popup:   key {{"combo": "Escape"}}  ← always try Escape first
  Open terminal:     cmd {{"command": "konsole >/dev/null 2>&1 &"}}
  Check if running:  cmd {{"command": "pgrep -x brave"}}
  Focus a window:    cmd {{"command": "xdotool search --class Brave windowactivate"}}
  Clip to clipboard: cmd {{"command": "echo 'text' | xclip -selection clipboard"}}
  Query files:       cmd {{"command": "find ~ -name '*.txt' 2>/dev/null | head -20"}}
  Service status:    cmd {{"command": "systemctl status sddm"}}

SHORTCUTS SECOND — prefer keyboard combos over clicking.
GUI MOUSE LAST — only when cmd and shortcuts cannot do it.

══════════════════ KEYBOARD SHORTCUTS ══════════════════
General:
  Ctrl+C / V / X     copy / paste / cut
  Ctrl+Z             undo   |   Ctrl+Shift+Z   redo
  Ctrl+A             select all
  Ctrl+S             save
  Ctrl+F             find in page / file
  Tab / Shift+Tab    next / prev field
  Enter              confirm / activate
  Esc                cancel / close dialog

KDE window management:
  Alt+Tab            switch windows
  Alt+F4             close window
  Alt+Space          window menu (move/resize/close)
  Super              app launcher / overview
  Super+R            KRunner run dialog
  Super+E            Dolphin file manager
  Super+W            present all open windows
  Super+D            show desktop
  Super+L            lock screen
  Super+↑↓←→         tile window to half screen
  Ctrl+Alt+T         open Konsole
  Ctrl+Alt+→/←       next / prev virtual desktop

Browser (Brave):
  Ctrl+L             focus URL bar  ← ALWAYS use this, never click the address bar
  Ctrl+T             new tab
  Ctrl+W             close tab
  Ctrl+Shift+T       reopen closed tab
  Ctrl+Tab           next tab
  Ctrl+Shift+Tab     prev tab
  Ctrl+R / F5        refresh
  Ctrl+F             find in page
  Ctrl+D             bookmark
  Ctrl+J             downloads
  Ctrl+Shift+N       incognito window
  F11                fullscreen

File manager / terminal:
  F2                 rename file
  Delete             move to trash
  Ctrl+H             show hidden files
  Ctrl+L             focus path / address bar

══════════════════ CLICKING STRATEGY — MANDATORY ZOOM-FIRST ══════════════════
NEVER click directly from a full-screen view. Every click requires zoom first.

REQUIRED FLOW:
  1. screenshot → identify the region where the target lives
  2. zoom {{"x": <cx>, "y": <cy>, "w": 2, "h": 2}}   ← full-screen 0-16 coords
       ONE image, two labeled panels:
         LEFT  — full screen with red box (verify box is over the right area)
         RIGHT — zoomed view with 16×16 sub-grid (verify target is visible here)
       • Both correct → CROSSHAIR CHECK → click using RIGHT panel coords
       • Wrong area → screenshot, identify correct region, zoom there instead
  3. CROSSHAIR CHECK → verify x then y independently (see section below)
  4. click / double_click / right_click → RIGHT panel sub-grid coords (auto-translate)

  ONE zoom per click — do NOT call zoom again while already zoomed.
  If you need a different area: screenshot first, then zoom again.
  After click: screenshot to verify. Miss → back to step 1.

  After click: screenshot → verify result. If missed → start over from step 1.
  After 3 failed clicks: switch strategy (keyboard shortcut, Tab+Enter, cmd).

SCROLLING (always provide x,y so mouse is over the right window):
  scroll {{"direction": "down", "x": 8.0, "y": 8.0}}  # scroll center of screen
  scroll {{"direction": "up",   "x": 8.0, "y": 5.0}}  # scroll browser content area

══════════════════ COORDINATE SYSTEM — 16×16 GRID ══════════════════
top-left=(0,0)   bottom-right=(16,16)
Decimals required — use 7.5 not 7
Columns 0=left edge → 16=right edge
Rows    0=top edge  → 16=bottom edge
OCR text positions in screenshot output are pixel-accurate — always prefer them.
NEVER invent coordinates without taking a screenshot first.

══════════════════ CROSSHAIR COORDINATE CHECK ══════════════════
Before every click, verify x and y independently using this 2-pass method:

  X PASS — find the exact column:
    Pick a candidate x value. Imagine a vertical line running top-to-bottom
    at that x. Ask: does that line pass through the HORIZONTAL CENTER of the
    target element? If it hits the left or right edge instead of the middle,
    adjust x until the vertical line bisects the element cleanly.

  Y PASS — find the exact row:
    Pick a candidate y value. Imagine a horizontal line running left-to-right
    at that y. Ask: does that line pass through the VERTICAL CENTER of the
    target element? If it clips the top or bottom edge instead of the middle,
    adjust y until the horizontal line bisects the element cleanly.

  COMMIT: Only use (x, y) once BOTH passes confirm the intersection lands
          squarely in the center of the element — not on its border or nearby.
          If uncertain after 2 passes, zoom tighter and repeat.

══════════════════ TASK & BUDGET ══════════════════
TASK: {task}
Budget: {max_iterations} iterations. Call finish() when {budget_warn} remain.

══════════════════ TOOLS ══════════════════
{available_tools}

══════════════════ OUTPUT FORMAT ══════════════════
ONE JSON object per response — NO prose, nothing else:
{{"thought": "reasoning", "confidence": 85, "tool": "name", "args": {{...}}}}

══════════════════ AGENT NOTES (persistent memory) ══════════════════
These are your saved discoveries from past sessions. Trust them.
{notes}

══════════════════ RULES ══════════════════
1.  cmd FIRST — try terminal commands before any GUI action
2.  Shortcuts before clicking — Ctrl+L beats clicking the address bar
3.  ZOOM before EVERY click — screenshot → zoom (×2) → verify → click. No exceptions.
4.  After EVERY click: screenshot to verify it worked
5.  If a click misses: screenshot → zoom fresh, NEVER reuse old coords
6.  Stuck on a button? Try Enter, Tab+Enter, or keyboard shortcut instead
7.  Use OCR sub-grid coords in zoomed view for text elements
8.  NEVER repeat same tool+args 4× in a row
9.  Learned something useful (binary path, UI quirk, keyboard shortcut)? → note it immediately
10. Task done → finish {{"summary": "what happened", "success": true}}
11. Irreversibly stuck → finish {{"summary": "what failed and why", "success": false}}

Start: can cmd accomplish this, or do I need the GUI?
"""

# Headless virtual display (:99) — Xvfb + xterm
_HEADLESS_SYSTEM_PROMPT = """\
You are an AI agent on an Arch Linux server with a virtual X11 display.

══════════════════ SYSTEM ══════════════════
OS: Arch Linux (rolling release)
Display: Xvfb :99 — virtual X11, 1280×720, no physical monitor
Shell: bash
Browser: brave-browser (Brave Browser)
Terminal: xterm (running on screen)
Package manager: pacman / yay
Init: systemd

══════════════════ TOOL STRATEGY ══════════════════
cmd FIRST — run shell commands whenever possible.

  Open a website:    cmd {{"command": "brave-browser 'https://url' &"}}
  Check if running:  cmd {{"command": "pgrep -x brave"}}
  Focus a window:    cmd {{"command": "xdotool search --class Brave windowactivate"}}
  Run anything:      cmd {{"command": "some-app &"}}

SHORTCUTS SECOND. GUI MOUSE LAST.

══════════════════ KEYBOARD SHORTCUTS ══════════════════
General:
  Ctrl+C/V/X         copy/paste/cut    Ctrl+Z  undo    Ctrl+A  select all
  Tab/Shift+Tab      next/prev field   Enter   confirm    Esc  cancel

Browser (Brave):
  Ctrl+L             focus URL bar (use this, not clicking)
  Ctrl+T             new tab     Ctrl+W  close tab    Ctrl+R  refresh
  Ctrl+F             find        F11     fullscreen

══════════════════ CLICKING STRATEGY — MANDATORY ZOOM-FIRST ══════════════════
NEVER click directly from a full-screen view. Every click requires zoom first.

REQUIRED FLOW:
  1. screenshot → identify region
  2. zoom {{"x": cx, "y": cy, "w": 2, "h": 2}}  ← full-screen coords
       LEFT=full screen+red box (verify location), RIGHT=zoomed sub-grid (verify target)
       ONE zoom per click — do NOT re-zoom while zoomed (screenshot first to reset)
  3. crosshair check → click RIGHT panel sub-grid coords (auto-translate)
  4. screenshot → verify. Miss → back to step 1.
  Scroll: always include x,y → scroll {{"direction":"down","x":8.0,"y":8.0}}

══════════════════ COORDINATE SYSTEM — 16×16 GRID ══════════════════
top-left=(0,0)   bottom-right=(16,16)   Decimals required: 7.5 not 7
OCR positions from screenshot are pixel-accurate — use them.
NEVER invent coordinates without a screenshot first.

══════════════════ CROSSHAIR COORDINATE CHECK ══════════════════
Before every click, verify x and y independently:

  X PASS: imagine a vertical line at your candidate x — does it pass through
          the HORIZONTAL CENTER of the target? Adjust until it does.
  Y PASS: imagine a horizontal line at your candidate y — does it pass through
          the VERTICAL CENTER of the target? Adjust until it does.
  COMMIT: only click once both lines intersect squarely on the element center.
          If uncertain, zoom tighter and repeat.

══════════════════ TASK & BUDGET ══════════════════
TASK: {task}
Budget: {max_iterations} iterations. Call finish() when {budget_warn} remain.

══════════════════ TOOLS ══════════════════
{available_tools}

══════════════════ OUTPUT FORMAT ══════════════════
ONE JSON object per response — NO prose:
{{"thought": "reasoning", "confidence": 85, "tool": "name", "args": {{...}}}}

══════════════════ AGENT NOTES (persistent memory) ══════════════════
These are your saved discoveries from past sessions. Trust them.
{notes}

══════════════════ RULES ══════════════════
1.  cmd FIRST   2. Shortcuts before clicking
3.  ZOOM before EVERY click — screenshot → zoom (×2) → verify → click
4.  After every click: screenshot to verify   5. Miss → start over with fresh zoom
6.  Use OCR sub-grid coords   7. Never repeat same tool+args 4× in a row
8.  Learned something useful? → note it
9.  Done → finish {{"summary": "...", "success": true}}
10. Stuck → finish {{"summary": "...", "success": false}}   11. Do NOT close xterm

Start: cmd or GUI?
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
        self._inhibit_cookie = None   # D-Bus screensaver inhibit cookie

        # Headless virtual display uses different resolution
        if display == ":99":
            screen_w, screen_h = 1280, 720
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.agent = OllamaCommandAgent(model=self.MODEL, fast_model=self.MODEL)
        self.agent.stop_event = stop_event  # enables streaming + mid-inference stop
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

    def _inhibit_sleep(self):
        """Disable screensaver + power management across all three layers:
        X11 (xset), KDE/freedesktop D-Bus inhibit, and xdg-screensaver.
        Stores D-Bus cookie in self._inhibit_cookie for cleanup.
        """
        env = {**os.environ, "DISPLAY": self.display}
        # Layer 1: X11 screensaver + DPMS
        for args in [["xset", "s", "off"], ["xset", "-dpms"]]:
            try:
                subprocess.run(args, env=env, capture_output=True, timeout=3)
            except Exception:
                pass
        # Layer 2: KDE/freedesktop D-Bus — returns cookie for UnInhibit
        try:
            r = subprocess.run(
                ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver",
                 "Inhibit", "gui-agent", "Automation in progress"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip().isdigit():
                self._inhibit_cookie = int(r.stdout.strip())
        except Exception:
            pass

    def _uninhibit_sleep(self):
        """Release the D-Bus screensaver inhibit lock."""
        if self._inhibit_cookie is not None:
            try:
                subprocess.run(
                    ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver",
                     "UnInhibit", str(self._inhibit_cookie)],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
            self._inhibit_cookie = None

    def _heartbeat(self, stop_event: threading.Event):
        """Layer 3: simulate user activity every 45s as belt-and-suspenders.

        Uses two methods that work without D-Bus (which isn't available in
        the systemd service environment):
        - xset s reset   — resets X11 screensaver idle timer (no D-Bus needed)
        - 1px mouse wiggle — strongest possible signal; KDE cannot ignore it
        """
        env = {**os.environ, "DISPLAY": self.display}
        while not stop_event.wait(45):
            try:
                # Reset X11 screensaver timer
                subprocess.run(
                    ["xset", "s", "reset"],
                    env=env, capture_output=True, timeout=3,
                )
            except Exception:
                pass
            try:
                # Micro mouse wiggle — 1px right then back; genuine activity signal
                subprocess.run(
                    ["xdotool", "mousemove_relative", "--", "1", "0"],
                    env=env, capture_output=True, timeout=3,
                )
                subprocess.run(
                    ["xdotool", "mousemove_relative", "--", "-1", "0"],
                    env=env, capture_output=True, timeout=3,
                )
            except Exception:
                pass

    def run(self, task: str, max_iterations: int = 30) -> dict:
        """Run the GUI agent on a task. Returns run_react result dict."""
        self.setup_display()
        self._inhibit_sleep()

        # Fresh log for each job
        try:
            with _log_lock:
                with open(GUI_DEBUG_LOG, "w", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "type": "job_start", "ts": time.strftime("%H:%M:%S"),
                        "task": task, "display": self.display,
                        "budget": max_iterations,
                    }) + "\n")
        except Exception:
            pass

        budget_warn = max(5, max_iterations // 5)
        notes = _load_notes()
        template = _KDE_SYSTEM_PROMPT if self.display == ":0" else _HEADLESS_SYSTEM_PROMPT
        system_prompt = template.format(
            task=task,
            available_tools=GUI_TOOLS_TEXT,
            max_iterations=max_iterations,
            budget_warn=budget_warn,
            notes=notes,
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

        # Heartbeat: resets idle timer every 60s so display never sleeps
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(
            target=self._heartbeat,
            args=(stop_heartbeat,),
            daemon=True,
        )
        heartbeat.start()

        try:
            result = self.agent.run_react(
                task,
                system_prompt_override=system_prompt,
                tool_whitelist=GUIToolRegistry.GUI_TOOL_NAMES,
            )
        finally:
            stop_watcher.set()
            stop_heartbeat.set()
            self._uninhibit_sleep()

        _gui_log({
            "type": "job_end",
            "success": result.get("success", False),
            "summary": result.get("summary", ""),
            "iterations": len(self.agent.react_trace),
        })

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
        """Background thread: watches react_trace, emits thought events, writes debug log."""
        seen = 0
        while not stop_event.is_set():
            trace = self.agent.react_trace
            if len(trace) > seen:
                for entry in trace[seen:]:
                    thought = entry.get("thought", "")
                    tool    = entry.get("tool", "")
                    n       = entry.get("iteration", seen + 1)
                    conf    = entry.get("confidence", 0)

                    # Emit thought event for browser UI
                    if thought and self.event_cb:
                        try:
                            self.event_cb("thought", {
                                "iteration": n, "thought": thought,
                                "confidence": conf, "tool": tool,
                            })
                        except Exception:
                            pass

                    # Write structured debug log entry
                    result = entry.get("result")
                    args   = entry.get("args", {})
                    # Strip large base64 strings from args
                    safe_args = {k: v for k, v in args.items()
                                 if not (isinstance(v, str) and len(v) > 500)}
                    log_e = {
                        "type": "iter", "n": n, "tool": tool,
                        "confidence": conf,
                        "thought": thought[:400] if thought else "",
                        "args": safe_args,
                    }
                    if result is not None:
                        meta = result.metadata or {}
                        if meta.get("screenshot"):
                            log_e["result"] = {
                                "ok": result.success,
                                "ocr": meta.get("ocr_count", 0),
                            }
                        else:
                            log_e["result"] = {
                                "ok": result.success,
                                "out": (result.output or "")[:600],
                                "err": (result.error or "")[:200],
                                "rc":  meta.get("returncode"),
                            }
                    _gui_log(log_e)

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
