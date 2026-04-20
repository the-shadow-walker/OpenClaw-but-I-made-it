"""
gui_tools.py — GUIToolRegistry extending ToolRegistry with GUI automation tools.
"""

import json
import time

from react_tools import ToolRegistry, ToolResult
from gui_screen import GUIScreen
from gui_input import GUIInput


GUI_TOOL_SCHEMAS = {
    "screenshot":   '  screenshot    — {}',
    "click":        '  click         — {"x": float, "y": float}',
    "double_click": '  double_click  — {"x": float, "y": float}',
    "right_click":  '  right_click   — {"x": float, "y": float}',
    "type":         '  type          — {"text": str}',
    "key":          '  key           — {"combo": str}',
    "scroll":       '  scroll        — {"direction": "up"|"down"}',
    "drag":         '  drag          — {"x1": float, "y1": float, "x2": float, "y2": float}',
    "wait":         '  wait          — {"seconds": float}',
    "finish":       '  finish        — {"summary": str, "success": bool}',
}

GUI_TOOLS_TEXT = "\n".join(GUI_TOOL_SCHEMAS.values())


class GUIToolRegistry(ToolRegistry):
    """ToolRegistry for GUI automation — only GUI tools + finish.

    Overrides dispatch() entirely to avoid the hardcoded parent handler_map,
    which only knows about the base ReAct tools.
    """

    GUI_TOOL_NAMES = {
        "screenshot", "click", "double_click", "right_click",
        "type", "key", "scroll", "drag", "wait", "finish",
    }

    def __init__(self, screen: GUIScreen, input_ctrl: GUIInput,
                 event_cb=None, stop_event=None, **kwargs):
        super().__init__(**kwargs)
        self.TOOL_NAMES = self.GUI_TOOL_NAMES
        self.screen = screen
        self.input_ctrl = input_ctrl
        # Optional callback: event_cb(event_type: str, payload: dict)
        # Used by gui_server.py to stream live events to connected browsers.
        self.event_cb = event_cb
        # Optional threading.Event — if set, dispatch returns a finish(stop) result
        self.stop_event = stop_event
        # Set by _handle_screenshot; picked up by ollama_agent_core.run_react
        # to inline the image into the next ReAct model call (no separate vision call).
        self.pending_image = None

    # ── dispatch (full override) ─────────────────────────────────────────────

    def dispatch(self, tool, args, confidence, confirm_cb=None):
        """Route GUI tool calls — completely replaces parent handler_map."""

        # Stop-requested check — fires finish() so run_react exits cleanly
        if self.stop_event and self.stop_event.is_set():
            return self._handle_finish({"summary": "Stopped by user request.", "success": False})

        # Stuck-loop guard
        call_key = (tool, json.dumps(args, sort_keys=True))
        self._recent_calls.append(call_key)
        if (len(self._recent_calls) == 4 and len(set(self._recent_calls)) == 1):
            self._stuck_warn_count += 1
            self._recent_calls.clear()
            if self._stuck_warn_count >= 2:
                return ToolResult(
                    False, "",
                    "Stuck loop persists after warning. Terminating.",
                    {"stuck": True},
                )
            stuck_cmd = args.get("tool", str(args))[:100]
            return ToolResult(
                False, "",
                (
                    f"STUCK LOOP: called '{tool}' with identical args 4× in a row: {stuck_cmd}\n"
                    f"Take a different action — call screenshot to reassess the screen state."
                ),
                {"stuck": "warn"},
            )

        if tool not in self.TOOL_NAMES:
            available = sorted(self.TOOL_NAMES)
            return ToolResult(
                False, "",
                f"Unknown tool: '{tool}'. Available: {available}",
                {"available": available},
            )

        # Emit action event before execution (skip for screenshot — it emits its own)
        if self.event_cb and tool != "screenshot":
            try:
                self.event_cb("action", {"tool": tool, "args": args})
            except Exception:
                pass

        handlers = {
            "screenshot":   self._handle_screenshot,
            "click":        self._handle_click,
            "double_click": self._handle_double_click,
            "right_click":  self._handle_right_click,
            "type":         self._handle_type,
            "key":          self._handle_key,
            "scroll":       self._handle_scroll,
            "drag":         self._handle_drag,
            "wait":         self._handle_wait,
            "finish":       self._handle_finish,  # inherited from ToolRegistry
        }
        result = handlers[tool](args)

        # Emit result event (skip screenshot — vision response is already in result.output)
        if self.event_cb and tool != "screenshot":
            try:
                self.event_cb("result", {
                    "tool": tool,
                    "success": result.success,
                    "output": result.output[:500] if result.output else "",
                    "error": result.error[:300] if result.error else "",
                })
            except Exception:
                pass

        return result

    # ── GUI tool handlers ────────────────────────────────────────────────────

    def _handle_screenshot(self, args):
        try:
            img = self.screen.capture()
            grid_img = self.screen.overlay_grid(img)

            # Full-res base64 → browser UI only
            b64_full = self.screen.to_base64(grid_img)
            if self.event_cb:
                try:
                    self.event_cb("screenshot", {"image": b64_full})
                except Exception:
                    pass

            # Downscaled base64 → inlined into next ReAct model call
            # 960px wide instead of 1920 — ~6× smaller, much faster inference
            self.pending_image = self.screen.to_base64_model(grid_img, max_w=960)

            elements = self.screen.ocr_elements(img)
            text_map = self.screen.build_text_map(elements)
            w, h = img.size

            obs = (
                f"[SCREENSHOT {w}×{h} — 8×8 grid overlaid, image attached]\n"
                f"OCR text positions (use these coords for precise clicks):\n"
                f"{text_map}\n"
                f"Examine the image and decide your next action."
            )

            if self.event_cb:
                try:
                    self.event_cb("vision", {"text": f"{w}×{h} — {len(elements)} OCR elements"})
                except Exception:
                    pass

            return ToolResult(True, obs, "", {"screenshot": True, "ocr_count": len(elements)})
        except Exception as e:
            if self.event_cb:
                try:
                    self.event_cb("error", {"text": f"Screenshot failed: {e}"})
                except Exception:
                    pass
            return ToolResult(False, "", f"Screenshot failed: {e}", {})

    def _handle_click(self, args):
        try:
            x = float(args.get("x", 0))
            y = float(args.get("y", 0))
            if not (0 <= x <= 8 and 0 <= y <= 8):
                return ToolResult(False, "", f"Coords out of range: x={x}, y={y} (must be 0–8)", {})
            msg = self.input_ctrl.click(x, y)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Click failed: {e}", {})

    def _handle_double_click(self, args):
        try:
            x = float(args.get("x", 0))
            y = float(args.get("y", 0))
            if not (0 <= x <= 8 and 0 <= y <= 8):
                return ToolResult(False, "", f"Coords out of range: x={x}, y={y}", {})
            msg = self.input_ctrl.double_click(x, y)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Double-click failed: {e}", {})

    def _handle_right_click(self, args):
        try:
            x = float(args.get("x", 0))
            y = float(args.get("y", 0))
            if not (0 <= x <= 8 and 0 <= y <= 8):
                return ToolResult(False, "", f"Coords out of range: x={x}, y={y}", {})
            msg = self.input_ctrl.right_click(x, y)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Right-click failed: {e}", {})

    def _handle_type(self, args):
        try:
            text = str(args.get("text", ""))
            if not text:
                return ToolResult(False, "", "type requires 'text'", {})
            msg = self.input_ctrl.type_text(text)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Type failed: {e}", {})

    def _handle_key(self, args):
        try:
            combo = str(args.get("combo", ""))
            if not combo:
                return ToolResult(False, "", "key requires 'combo'", {})
            msg = self.input_ctrl.key(combo)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Key failed: {e}", {})

    def _handle_scroll(self, args):
        try:
            direction = str(args.get("direction", "down")).lower()
            if direction not in ("up", "down"):
                return ToolResult(False, "", "scroll direction must be 'up' or 'down'", {})
            msg = self.input_ctrl.scroll(direction)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Scroll failed: {e}", {})

    def _handle_drag(self, args):
        try:
            x1 = float(args.get("x1", 0))
            y1 = float(args.get("y1", 0))
            x2 = float(args.get("x2", 0))
            y2 = float(args.get("y2", 0))
            for coord, val in [("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)]:
                if not (0 <= val <= 8):
                    return ToolResult(False, "", f"{coord}={val} out of range (0–8)", {})
            msg = self.input_ctrl.drag(x1, y1, x2, y2)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Drag failed: {e}", {})

    def _handle_wait(self, args):
        try:
            seconds = float(args.get("seconds", 1))
            seconds = max(0.1, min(seconds, 30.0))  # clamp to 0.1–30s
            time.sleep(seconds)
            return ToolResult(True, f"Waited {seconds:.1f}s", "", {})
        except Exception as e:
            return ToolResult(False, "", f"Wait failed: {e}", {})
