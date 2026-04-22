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
import shutil
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
from gui_profiles import ProfileStore
from gui_macros import MacroStore

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


# ── Math/science input reference (raw string — backslashes are LaTeX commands) ───
# Defined outside the format-string prompts so Python doesn't misinterpret \t \n etc.
_MATH_INPUT_GUIDE = r"""Math & Science Input (MathQuill — DeltaMath, Khan Academy, Desmos, etc.)
DO NOT type Unicode symbols (√ ∛ π θ ≤ etc.) — the widget will reject them.
Use LaTeX commands or button clicks instead. Type the command then Space to render.

── Navigation ──
  Left/Right         move cursor    Up/Down  enter/exit super/fraction slots
  Tab / Shift+Tab    next / prev template slot    Right  exit slot
  Ctrl+A  select all    Backspace  delete / exit empty slot

── Structures ──
  x^2  exponent (type base ^ exponent Right)     x_1  subscript (type base _ subscript Right)
  /    fraction (type numerator / denominator Right)     |x|  absolute value
  \sqrt{x}  square root (type \sqrt then content)
  nth root: click n-root button → type INDEX (3 for cube) → Tab → type RADICAND → Right

── Greek letters (type + Space) ──
  \alpha  \beta  \gamma  \delta  \epsilon  \zeta  \eta  \theta  \iota  \kappa
  \lambda  \mu  \nu  \xi  \pi  \rho  \sigma  \tau  \phi  \chi  \psi  \omega
  \Delta  \Theta  \Lambda  \Pi  \Sigma  \Omega  \Gamma  \Phi  \Psi

── Operators & relations ──
  \times  \div  \cdot  \pm  \mp  \le  \ge  \ne  \approx  \sim  \equiv  \cong
  \to  \gets  \Rightarrow  \leftrightarrow  \infty  \degree  \circ
  \angle  \perp  \parallel  \triangle  \overline{}  \vec{}

── Physics ──
  Scientific notation:  3.0\times10^8   (type \times then 10^8)
  Units:                kg\cdot m/s^2   N\cdot m   J/kg
  Vectors:              \vec{v}  or use vector button
  Delta:                \Delta x  (change in x)
  Common: \theta (angle)  \lambda (wavelength)  \mu (friction/micro-prefix)
          \omega (angular vel)  \alpha (angular accel)  \rho (density)
          \phi (flux/angle)  \eta (efficiency)  \tau (torque/period)  \sigma (stress)
          \Delta (change)  \Sigma (sum)  \nabla (gradient)

── Chemistry ──
  Subscripts:   H_2O   CO_2   C_6H_{12}O_6   SO_4^{2-}
  Charges:      Fe^{2+}   OH^-   Ca^{2+}   Cu^{2+}   NH_4^+
  Reaction arrow: \to  or click button
  Equilibrium:  click double-arrow button
  States:       (s)  (l)  (g)  (aq)  — type normally in parentheses

── Algebra / Geometry ──
  Fractions:    type numerator / denominator (auto-fraction in MathQuill)
  Absolute val: |expression|
  Log/ln:       \log   \ln   \log_{b}x  (subscript b)
  Trig:         \sin   \cos   \tan   \csc   \sec   \cot
  Inverse trig: \sin^{-1}   \cos^{-1}   \tan^{-1}  (or use button)
  Congruent:    \cong   Similar: \sim   Angle: \angle   Triangle: \triangle
  Line segment: \overline{AB}   Ray: \vec{AB}
  Limits:       \lim_{x\to 0}
  Sum/integral: \sum   \int  (use buttons when available)

── Recovery when input is rejected ──
  1. Ctrl+A then Backspace — clear field completely
  2. Screenshot to confirm blank
  3. Re-enter via LaTeX commands or button clicks — never paste Unicode symbols
  4. Screenshot after complex entry to verify it rendered correctly before submitting"""

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

  Open a website:    cmd {{"command": "/usr/bin/brave --start-maximized --no-restore-last-session --remote-debugging-port=9222 --remote-allow-origins=* 'https://url' >/dev/null 2>&1 &"}}
  ⚠️  NEVER combine pgrep with brave launch in one cmd — it causes a 60s pipe timeout.
      Check first: cmd {{"command": "pgrep -x brave"}}  THEN launch separately if needed.
  Dismiss OS popup:  key {{"combo": "Escape"}}  ← ONLY for OS-level popups, NEVER inside login/form modals
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
  Space              activate focused button / checkbox / toggle
  Delete             forward-delete character
  Backspace          backward-delete character
  Home / End         jump to line start / end
  Ctrl+Home/End      jump to document start / end
  Left/Right         move cursor one character
  Ctrl+Left/Right    move cursor one word
  Up/Down            move cursor one line / navigate menu / list
  PageUp/PageDown    scroll one screen

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

{math_guide}

══════════════════ CLICKING — YOU ARE AN ID-FIRST AGENT ══════════════════
Every screenshot has numbered markers drawn directly on each interactive element.
To click anything, use its ID number. The OS/browser tells us exactly where each
element is — there is no guessing, no coordinate math required.

  click {{"id": 14}}   ← ALWAYS do this when an ID is shown. 100% reliable.

COLOR LEGEND (trust level):
  [Blue]   DOM    — Perfect browser data. Pixel-accurate. Trust completely.
  [Green]  AT-SPI — Perfect desktop data. Pixel-accurate. Trust completely.
  [Orange] CV     — Visually inferred. Usually correct, but double-check if it misses.

THE HIERARCHY — follow this order every single time:
  Priority 1 — Numbered ID (exact, sourced from OS/browser metadata)
               click {{"id": N}}  ← no math, no estimation
  Priority 2 — OCR coordinate (element visible in OCR but not in ID list)
               click {{"x": f, "y": f}}  ← pixel-accurate from OCR
  Priority 3 — Grid crosshair (LAST RESORT — only for icons with no text and no ID)
               zoom to the area first, then use the X/Y pass method below

STALE IDs — if a click seems to miss or the UI changed:
  • Call rescan {{}} to refresh IDs from the current screen state (no new screenshot needed)
  • Call screenshot {{}} if the page/dialog may have changed (re-queries DOM + AT-SPI live)
  • A [Blue] or [Green] ID click should never miss — if it does, the element moved; rescan

ZOOM — for unlabeled graphical elements only:
  zoom {{"id": N}}  ← centers on element N's position automatically
  zoom {{"x": cx, "y": cy, "w": 2, "h": 2}}  ← manual center in grid coords
  One zoom per click. Call screenshot to reset before re-zooming.

  ⚠️  MODAL FORMS (login dialogs, popups): NEVER press Escape inside them.
      Escape closes the modal and loses all your progress.

SCROLLING (always provide x,y so mouse is over the right window):
  scroll {{"direction": "down", "x": 8.0, "y": 8.0}}  # scroll center of screen
  scroll {{"direction": "up",   "x": 8.0, "y": 5.0}}  # scroll browser content area

══════════════════ EFFICIENCY — BATCH SEQUENTIAL ACTIONS ══════════════════
screenshot is expensive — it costs an iteration AND model inference. Use it deliberately.

  ✓ CORRECT — batch a form sequence, one screenshot at the end:
      click username_field → type email → key Tab → type password → key Enter → screenshot

  ✗ WRONG — screenshot between every step:
      click field → screenshot → type email → screenshot → key Tab → screenshot ...

Take a screenshot WHEN:
  • You need current state (start of task, after navigation, after page load)
  • An action has uncertain outcome (form submit, dialog appears/disappears)
  • Something went wrong and you need to re-orient

Skip the screenshot WHEN:
  • Filling in a form field by field (type, Tab, type, Enter is one atomic sequence)
  • Pressing a modifier key (Ctrl+L, then just type the URL — no screenshot needed)
  • After wait — screenshot only if you actually need to verify something loaded

══════════════════ PLANNING ══════════════════
First iteration: call plan {{"steps": ["1. ...", "2. ...", ...]}} BEFORE any other action.
Keep it short (5–8 steps). Update it with another plan call if something changes.
The plan is shown on every screenshot so you always know where you are.

Example for login + assignment:
  plan {{"steps": [
    "1. Launch Brave to site",
    "2. Screenshot — verify loaded, get Login coord from DOM",
    "3. Click Login, screenshot — find form fields",
    "4. Click username, type, Tab, type password, Enter",
    "5. Screenshot — verify logged in",
    "6. Find and open first assignment",
    "7. Read problem, solve it, submit"
  ]}}

══════════════════ COORDINATE SYSTEM — 16×16 GRID ══════════════════
top-left=(0,0)   bottom-right=(16,16)
Decimals required — use 7.5 not 7
Columns 0=left edge → 16=right edge
Rows    0=top edge  → 16=bottom edge

HOW TO CLICK — follow this priority order every single time:

  STEP 1 — Element list (labeled "Elements (click by ID)"):
    If your target is listed → click {{"id": N}} DIRECTLY. No zoom, no coord estimation.
    The image shows numbered markers on each element. IDs are always current.

  STEP 2 — OCR list (labeled "OCR text fallback"):
    If the target's text appears in OCR but not the element list → use that coord.
    OCR coords are pixel-accurate. Do not estimate from the image.

  STEP 3 — Crosshair estimate (ONLY if target is absent from BOTH lists):
    Applies to unlabeled icons, purely graphical elements, or unknown UI areas.
    Zoom in first, then use the crosshair method below.

══════════════════ CROSSHAIR METHOD (step 3 only — image-based fallback) ══════════════════
Use ONLY when your target does NOT appear in the element list or OCR list.

  X PASS — find the exact column:
    Pick a candidate x. Imagine a vertical line top-to-bottom at that x.
    Does it pass through the HORIZONTAL CENTER of the target? Adjust until it does.

  Y PASS — find the exact row:
    Pick a candidate y. Imagine a horizontal line left-to-right at that y.
    Does it pass through the VERTICAL CENTER of the target? Adjust until it does.

  COMMIT: only click once both lines intersect squarely on the element center.
          If uncertain after 2 passes, zoom tighter and repeat.

  If an ID click misses → call rescan {{}} to refresh IDs and retry.
  If a coord misses → take a fresh screenshot (element list re-queries live), then retry.

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

══════════════════ TASK PROFILES (matched to this task) ══════════════════
Step-by-step playbooks from previous successful runs on similar tasks.
{profiles}

══════════════════ MACROS — instant single-turn replay ══════════════════
Cached action sequences from previous successful runs.
Run an entire flow in ONE iteration: sequence {{"macro": "MACRO_NAME"}}
If a macro covers your next step, USE IT — do not re-discover what is already known.
{macros}

══════════════════ RULES ══════════════════
1.  cmd FIRST — try terminal commands before any GUI action
2.  Shortcuts before clicking — Ctrl+L beats clicking the address bar
3.  ❌ Element in list → click {{"id": N}} DIRECTLY. No zoom, no coordinate estimation.
    Zoom ONLY for unmarked icons or pure graphical elements absent from all lists.
4.  Screenshot only when needed: after navigation/page-load/submit. During form fill: batch (click→type→Tab→type→click) ONE screenshot at the end.
5.  ID click miss → rescan {{}} to refresh IDs; coord miss → fresh screenshot
6.  ❌ NEVER press Escape inside a login or form modal — it will close/dismiss it
7.  Use ID from element list first; OCR coord second; crosshair estimate last resort
8.  NEVER repeat same tool+args 4× in a row
9.  Learned something useful? → note it immediately
10. Task done → save_profile to document the flow, then finish {{"summary": "...", "success": true}}
11. Irreversibly stuck → finish {{"summary": "what failed and why", "success": false}}
12. ❌ NEVER type Unicode math symbols (∛ √ ² π θ Σ ≤ ≥ ∞ etc.) — MathQuill REJECTS them silently
    Use: LaTeX commands (\\sqrt, \\theta, \\pi, \\le) + Space to render, or toolbar buttons
13. MathQuill nth root: click ⁿ√ button → type INDEX (3=cube, 4=fourth) → Tab → type RADICAND → Right to exit
14. MathQuill exponent: type BASE → ^ → type EXPONENT → Right    fraction: type NUM → / → DENOM → Right

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

  Open a website:    cmd {{"command": "/usr/bin/brave --start-maximized --no-restore-last-session --remote-debugging-port=9222 --remote-allow-origins=* 'https://url' >/dev/null 2>&1 &"}}
  ⚠️  NEVER combine pgrep with brave launch in one cmd — causes 60s pipe timeout.
      Check: cmd {{"command": "pgrep -x brave"}}  THEN launch separately if not running.
  Check if running:  cmd {{"command": "pgrep -x brave"}}
  Focus a window:    cmd {{"command": "xdotool search --class Brave windowactivate"}}
  Run anything:      cmd {{"command": "some-app >/dev/null 2>&1 &"}}

SHORTCUTS SECOND. GUI MOUSE LAST.

══════════════════ KEYBOARD SHORTCUTS ══════════════════
General:
  Ctrl+C/V/X         copy/paste/cut    Ctrl+Z  undo    Ctrl+A  select all
  Tab/Shift+Tab      next/prev field   Enter   confirm
  Space              activate button/checkbox    Delete  forward-delete
  Home/End           line start/end    Ctrl+Home/End  doc start/end
  Left/Right         move cursor    Ctrl+Left/Right   one word
  Up/Down            line / menu navigation
  PageUp/PageDown    scroll one screen
  Esc  cancel — ⚠️ NEVER inside a login/form modal (Escape will close it!)

Browser (Brave):
  Ctrl+L             focus URL bar (use this, not clicking)
  Ctrl+T             new tab     Ctrl+W  close tab    Ctrl+R  refresh
  Ctrl+F             find        F11     fullscreen

{math_guide}

══════════════════ CLICKING — YOU ARE AN ID-FIRST AGENT ══════════════════
Every screenshot has numbered markers drawn directly on each interactive element.
To click anything, use its ID number. No guessing, no coordinate math.

  click {{"id": 14}}   ← ALWAYS do this when an ID is shown. 100% reliable.

COLOR LEGEND (trust level):
  [Blue]   DOM    — Perfect browser data. Pixel-accurate. Trust completely.
  [Green]  AT-SPI — Perfect desktop data. Pixel-accurate. Trust completely.
  [Orange] CV     — Visually inferred. Usually correct, but double-check if it misses.

THE HIERARCHY:
  Priority 1 — ID → click {{"id": N}}
  Priority 2 — OCR coord → click {{"x": f, "y": f}}
  Priority 3 — Crosshair estimate (last resort, zoom first)

STALE IDs — if a click misses or the UI changed:
  rescan {{}}  → refresh IDs without new screenshot (fast)
  rescan {{"hint": "tiny X"}}  → relaxed CV thresholds to find smaller elements
  screenshot {{}}  → full refresh when page/dialog may have changed

ZOOM:
  zoom {{"id": N}}  ← auto-centers on element N
  zoom {{"x": cx, "y": cy, "w": 2, "h": 2}}  ← manual
  One zoom per click. Call screenshot to reset.

Scroll: always include x,y → scroll {{"direction":"down","x":8.0,"y":8.0}}

══════════════════ EFFICIENCY — BATCH SEQUENTIAL ACTIONS ══════════════════
screenshot is expensive — costs an iteration AND model inference. Use deliberately.

  ✓ CORRECT — batch a form sequence, one screenshot at the end:
      click username_field → type email → key Tab → type password → key Enter → screenshot

  ✗ WRONG — screenshot between every step:
      click field → screenshot → type email → screenshot → key Tab → screenshot ...

Take a screenshot WHEN:
  • Need current state (start of task, after navigation, after page load)
  • Uncertain outcome (form submit, dialog may appear)
  • Something went wrong and you need to re-orient

Skip the screenshot WHEN:
  • Filling a form field by field (click, type, Tab, type, Enter → ONE atomic sequence)
  • After a modifier key (Ctrl+L then type URL — no screenshot needed)
  • After wait — only screenshot if you need to verify something loaded

══════════════════ PLANNING ══════════════════
First iteration: call plan {{"steps": ["1. ...", "2. ...", ...]}} BEFORE any other action.
Keep it short (5–8 steps). Update it with another plan call if something changes.
The plan is shown on every screenshot so you always know where you are.

Example:
  plan {{"steps": [
    "1. Launch Brave to site",
    "2. Screenshot — verify loaded, get Login coord from DOM",
    "3. Click Login, screenshot — find form fields",
    "4. Click username, type, Tab, type password, Enter",
    "5. Screenshot — verify logged in",
    "6. Complete the task",
    "7. save_profile + finish"
  ]}}

══════════════════ COORDINATE SYSTEM — 16×16 GRID ══════════════════
top-left=(0,0)   bottom-right=(16,16)   Decimals required: 7.5 not 7

HOW TO CLICK — follow this priority order every single time:

  STEP 1 — Element list ("Elements (click by ID)"): target listed? click {{"id": N}} DIRECTLY.
  STEP 2 — OCR list ("OCR text fallback"): text in OCR? Copy that coord directly.
  STEP 3 — Crosshair (ONLY if absent from BOTH lists): icons, graphics, unlabeled elements.

NEVER estimate from the image when the element list has an entry for your target.

══════════════════ CROSSHAIR METHOD (step 3 only — image-based fallback) ══════════════════
  X PASS: vertical line at x — does it bisect the target's horizontal center?
  Y PASS: horizontal line at y — does it bisect the target's vertical center?
  COMMIT: click only once both lines intersect squarely on the element center.
          If uncertain, zoom tighter and repeat.
  If an ID click misses → call rescan {{}} to refresh IDs and retry.
  If a coord misses → take fresh screenshot (element list re-queries live page).

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

══════════════════ TASK PROFILES (matched to this task) ══════════════════
Step-by-step playbooks from previous successful runs on similar tasks.
{profiles}

══════════════════ MACROS — instant single-turn replay ══════════════════
Cached action sequences from previous successful runs.
Run an entire flow in ONE iteration: sequence {{"macro": "MACRO_NAME"}}
If a macro covers your next step, USE IT — do not re-discover what is already known.
{macros}

══════════════════ RULES ══════════════════
1.  cmd FIRST   2. Shortcuts before clicking
3.  ❌ Element in list → click {{"id": N}} DIRECTLY. No zoom, no coord estimation.
    Zoom ONLY for unmarked icons or pure graphical elements absent from all lists.
4.  Screenshot only after navigation/page-load/submit. During form fill: batch without screenshots.
5.  ID click miss → rescan {{}}; coord miss → fresh screenshot   6. ❌ NEVER press Escape inside login/form modal
7.  ID from element list first; OCR coord second; crosshair last resort
8.  Never repeat same tool+args 4× in a row   9. Learned something? → note it
10. Done → save_profile to document the flow, then finish {{"summary": "...", "success": true}}
11. Stuck → finish {{"summary": "...", "success": false}}   12. Do NOT close xterm
13. ❌ NEVER type Unicode math (∛ √ ² π θ etc.) — MathQuill REJECTS them silently
    Use LaTeX commands (\\sqrt, \\theta, \\pi) + Space, or toolbar buttons
14. MathQuill nth root: click ⁿ√ button → type INDEX (3=cube) → Tab → type RADICAND → Right

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

    def _ensure_atspi(self):
        """Check that the AT-SPI bus launcher is running; start it if not.

        QT_ACCESSIBILITY=1 must be set before Qt apps launch for them to publish
        their accessibility trees. We set it process-wide here so all subsequent
        subprocess.Popen / subprocess.run calls inherit it automatically.
        """
        # Set QT_ACCESSIBILITY globally so every Qt app we launch exposes its tree
        os.environ["QT_ACCESSIBILITY"] = "1"

        # Check if at-spi-bus-launcher is running
        r = subprocess.run(
            ["pgrep", "-x", "at-spi-bus-launcher"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"  AT-SPI bus launcher running (pid {r.stdout.strip()})")
            return

        # Try to start it
        print("  ⚠️  at-spi-bus-launcher not running — attempting to start...")
        try:
            env = {**os.environ, "DISPLAY": self.display}
            xauth = os.path.expanduser("~/.Xauthority")
            if os.path.exists(xauth):
                env["XAUTHORITY"] = xauth
            subprocess.Popen(
                ["at-spi-bus-launcher", "--launch-immediately"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)
            # Verify it started
            r2 = subprocess.run(["pgrep", "-x", "at-spi-bus-launcher"],
                                 capture_output=True, text=True)
            if r2.returncode == 0:
                print("  AT-SPI bus launcher started OK")
            else:
                print(
                    "  ⚠️  AT-SPI bus launcher failed to start — "
                    "AT-SPI elements will be unavailable. "
                    "Install: sudo pacman -S at-spi2-core"
                )
        except FileNotFoundError:
            print(
                "  ⚠️  at-spi-bus-launcher not found — "
                "install at-spi2-core: sudo pacman -S at-spi2-core"
            )
        except Exception as e:
            print(f"  ⚠️  AT-SPI launcher error: {e}")

    def setup_display(self):
        """Ensure the target display is accessible.

        For :0 (real KDE desktop): verify access, refresh xauth if needed.
        For :99 (headless): start Xvfb + kwin_x11 + xterm if not running.
        """
        # Ensure AT-SPI accessibility bus is live and QT_ACCESSIBILITY=1 is set
        self._ensure_atspi()

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

    def _get_session_env(self) -> dict:
        """Build env dict with DBUS_SESSION_BUS_ADDRESS from the running KDE session.

        The gui-agent systemd service doesn't inherit the user's D-Bus session bus,
        so qdbus calls fail silently. We find the address by reading /proc/<pid>/environ
        of a known KDE session process owned by the current user.
        """
        env = {**os.environ, "DISPLAY": self.display}
        xauth = os.path.expanduser("~/.Xauthority")
        if os.path.exists(xauth):
            env["XAUTHORITY"] = xauth

        if env.get("DBUS_SESSION_BUS_ADDRESS"):
            return env  # already set (interactive session)

        user = os.environ.get("USER", "Grindlewalt")
        for proc_name in ("plasmashell", "kwin_x11", "kded6", "kded5"):
            try:
                r = subprocess.run(
                    ["pgrep", "-x", "-u", user, proc_name],
                    capture_output=True, text=True, timeout=3,
                )
                if r.returncode != 0:
                    continue
                pid = r.stdout.strip().split()[0]
                with open(f"/proc/{pid}/environ", "rb") as f:
                    raw = f.read()
                for item in raw.split(b"\x00"):
                    if item.startswith(b"DBUS_SESSION_BUS_ADDRESS="):
                        env["DBUS_SESSION_BUS_ADDRESS"] = item[25:].decode()
                        print(f"  D-Bus session found via {proc_name} (pid {pid})")
                        return env
            except Exception:
                continue
        print("  ⚠️  D-Bus session bus not found — qdbus inhibit will be skipped")
        return env

    def _inhibit_sleep(self):
        """Disable screensaver + power management across four layers.

        Layer 1 — X11: set all DPMS timeouts to 0 and disable screensaver blank.
        Layer 2 — KDE config: write kscreenlockerrc Autolock=false for the session.
        Layer 3 — D-Bus: org.freedesktop.ScreenSaver.Inhibit (requires session bus).
        Layer 4 — Heartbeat: mouse wiggle every 45s (belt-and-suspenders, see _heartbeat).
        """
        env = self._get_session_env()

        # Layer 1: X11 — nuke all screensaver + DPMS timeouts
        for cmd in [
            ["xset", "s", "off"],           # disable X11 screensaver
            ["xset", "s", "0", "0"],        # blank/expose timeouts to 0
            ["xset", "-dpms"],              # disable DPMS
            ["xset", "dpms", "0", "0", "0"],# standby/suspend/off all to 0
        ]:
            try:
                subprocess.run(cmd, env=env, capture_output=True, timeout=3)
            except Exception:
                pass

        # Layer 2: KDE screen lock config — disable auto-lock for current session
        try:
            subprocess.run(
                ["kwriteconfig5", "--file", "kscreenlockerrc",
                 "--group", "Daemon", "--key", "Autolock", "false"],
                env=env, capture_output=True, timeout=5,
            )
        except Exception:
            pass

        # Layer 3: freedesktop ScreenSaver inhibit via D-Bus
        if env.get("DBUS_SESSION_BUS_ADDRESS"):
            try:
                r = subprocess.run(
                    ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver",
                     "Inhibit", "gui-agent", "Automation in progress"],
                    env=env, capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0 and r.stdout.strip().isdigit():
                    self._inhibit_cookie = int(r.stdout.strip())
                    print(f"  Screen inhibit cookie: {self._inhibit_cookie}")
            except Exception:
                pass

    def _uninhibit_sleep(self):
        """Release D-Bus screensaver inhibit and restore KDE auto-lock."""
        env = self._get_session_env()

        # Release D-Bus inhibit
        if self._inhibit_cookie is not None:
            try:
                subprocess.run(
                    ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver",
                     "UnInhibit", str(self._inhibit_cookie)],
                    env=env, capture_output=True, timeout=5,
                )
            except Exception:
                pass
            self._inhibit_cookie = None

        # Restore KDE auto-lock
        try:
            subprocess.run(
                ["kwriteconfig5", "--file", "kscreenlockerrc",
                 "--group", "Daemon", "--key", "Autolock", "true"],
                env=env, capture_output=True, timeout=5,
            )
        except Exception:
            pass

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

        # ── Create timestamped run archive directory ───────────────────────────
        slug = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_")
        ts   = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.expanduser(f"~/.agent_bin/runs/{ts}_{slug}")
        try:
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "task.json"), "w", encoding="utf-8") as f:
                json.dump({"task": task, "ts": ts, "display": self.display,
                           "budget": max_iterations}, f, indent=2)
        except Exception:
            run_dir = None  # archive disabled if dir creation fails

        # Point tool registry at the run dir so screenshots are saved there
        tr = self.agent.tool_registry
        tr._run_dir = run_dir
        tr._screenshot_counter = 0

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

        # Search for relevant task profiles — inject matched ones, skip the rest
        _ps = ProfileStore()
        matched = _ps.search(task, top_n=2)
        profiles = _ps.format_for_prompt(matched)

        # Load all macros — always inject full catalog (small, fast)
        _ms = MacroStore()
        macros = _ms.format_for_prompt()

        template = _KDE_SYSTEM_PROMPT if self.display == ":0" else _HEADLESS_SYSTEM_PROMPT
        system_prompt = template.format(
            task=task,
            available_tools=GUI_TOOLS_TEXT,
            max_iterations=max_iterations,
            budget_warn=budget_warn,
            notes=notes,
            profiles=profiles,
            macros=macros,
            math_guide=_MATH_INPUT_GUIDE,
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

        # ── Finalize run archive ───────────────────────────────────────────────
        if run_dir and os.path.isdir(run_dir):
            try:
                # summary.json
                with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "task":       task,
                        "success":    result.get("success", False),
                        "summary":    result.get("summary", ""),
                        "iterations": len(self.agent.react_trace),
                        "screenshots": self.agent.tool_registry._screenshot_counter,
                        "ts_end":     time.strftime("%Y%m%d_%H%M%S"),
                    }, f, indent=2)
                # trace.jsonl — agent thoughts + tool calls (strip large base64 blobs)
                with open(os.path.join(run_dir, "trace.jsonl"), "w", encoding="utf-8") as f:
                    for entry in self.agent.react_trace:
                        safe = {k: v for k, v in entry.items()
                                if not (isinstance(v, str) and len(v) > 800)}
                        f.write(json.dumps(safe, default=str) + "\n")
                # debug.jsonl — copy main debug log into the archive
                if os.path.exists(GUI_DEBUG_LOG):
                    shutil.copy2(GUI_DEBUG_LOG, os.path.join(run_dir, "debug.jsonl"))
                # latest symlink — points to most recent run dir
                latest_link = os.path.expanduser("~/.agent_bin/runs/latest")
                try:
                    os.unlink(latest_link)
                except FileNotFoundError:
                    pass
                os.symlink(run_dir, latest_link)
                print(f"  Run archive: {run_dir}")
            except Exception as _e:
                print(f"  ⚠️  Archive finalize failed: {_e}")

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
