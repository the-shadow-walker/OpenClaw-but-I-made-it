"""
gui_agent.py — GUIAgent: vision-based desktop automation powered by qwen3.5:35b.

The agent takes screenshots, overlays an 8×8 grid, runs OCR for text positions,
sends the annotated image to qwen3.5:35b (Ollama vision API), receives a JSON action,
executes it via xdotool, and loops.
"""

import json
import os
import re
import subprocess
import sys
import time

from ollama_agent_core import OllamaCommandAgent
from gui_screen import GUIScreen
from gui_input import GUIInput
from gui_tools import GUIToolRegistry, GUI_TOOLS_TEXT


# ── System prompt ─────────────────────────────────────────────────────────────

GUI_SYSTEM_PROMPT_TEMPLATE = """\
You are a desktop automation agent controlling a Linux desktop on display :99.
You can see the current screen state via the screenshot tool, which returns a
vision analysis of an annotated screenshot with an 8×8 grid overlay.

Coordinates: top-left (0,0), bottom-right (8,8). Decimals allowed (e.g. 3.5, 1.2).

CURRENT TASK: {task}

ITERATION BUDGET: {max_iterations}

AVAILABLE TOOLS:
{available_tools}

OUTPUT FORMAT (REQUIRED EVERY RESPONSE):
{{"thought": "chain-of-thought reasoning", "confidence": 90, "tool": "tool_name", "args": {{...}}}}
Never output plain text. Every response must be a single valid JSON object.

RULES:
1. ALWAYS call screenshot first to see the current screen state
2. After EVERY action (click, type, key, scroll), call screenshot to verify the result
3. To click a text element: use the grid coordinate from the OCR list for precision
4. To type in a field: click it first, then use the type tool
5. If a click misses: call screenshot, re-read OCR positions, recalculate coordinates
6. If something is unclear: use wait {{\"seconds\": 1}} then screenshot again
7. confidence < 70 = gather more evidence; do not act until confident
8. At {budget_warn} iterations remaining: call finish() with current progress
9. NEVER guess coordinates — always base them on OCR positions or grid landmarks

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
    MODEL = "qwen3.5:35b"

    def __init__(self, display=":99", screen_w=1280, screen_h=720):
        self.display = display
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.agent = OllamaCommandAgent(model=self.MODEL, fast_model=self.MODEL)
        self.screen = GUIScreen(display, screen_w, screen_h)
        self.input_ctrl = GUIInput(display, screen_w, screen_h)

        self.agent.tool_registry = GUIToolRegistry(
            screen=self.screen,
            input_ctrl=self.input_ctrl,
            call_vision_fn=lambda prompt, img: _call_vision(self.MODEL, prompt, img),
            safety_validator=self.agent.safety_validator,
            search_agent=self.agent.search_agent,
            memory=self.agent.memory,
        )

    def setup_display(self):
        """Start Xvfb :99 if not already running, verify with xdotool."""
        env = {**os.environ, "DISPLAY": self.display}
        r = subprocess.run(
            ["xdotool", "getdisplaygeometry"],
            env=env, capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            print(f"  Display {self.display} active: {r.stdout.strip()}")
            return

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
        return self.agent.run_react(
            task,
            system_prompt_override=system_prompt,
            tool_whitelist=GUIToolRegistry.GUI_TOOL_NAMES,
        )


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
