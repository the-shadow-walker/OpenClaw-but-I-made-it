"""
gui_tools.py — GUIToolRegistry extending ToolRegistry with GUI automation tools.
"""

import json
import os
import subprocess
import time

from react_tools import ToolRegistry, ToolResult
from gui_screen import GUIScreen
from gui_input import GUIInput


GUI_TOOL_SCHEMAS = {
    "cmd":          '  cmd           — {"command": str}  # run shell command, returns output; append & for background',
    "screenshot":   '  screenshot    — {}  # full-screen; also exits zoom mode',
    "zoom":         '  zoom          — {"x": float, "y": float, "w": float, "h": float}  # w/h in grid cells (default 1×1); zoomed 16×16 sub-grid; clicks auto-translate',
    "click":        '  click         — {"x": float, "y": float}',
    "double_click": '  double_click  — {"x": float, "y": float}',
    "right_click":  '  right_click   — {"x": float, "y": float}',
    "type":         '  type          — {"text": str}',
    "key":          '  key           — {"combo": str}',
    "scroll":       '  scroll        — {"direction": "up"|"down", "x": float, "y": float}  # x,y: move mouse there first (required for browser scroll)',
    "drag":         '  drag          — {"x1": float, "y1": float, "x2": float, "y2": float}',
    "wait":         '  wait          — {"seconds": float}',
    "note":         '  note          — {"text": str}  # save a discovery to persistent notes (binary paths, UI quirks, tips)',
    "finish":       '  finish        — {"summary": str, "success": bool}',
}

GUI_TOOLS_TEXT = "\n".join(GUI_TOOL_SCHEMAS.values())


class GUIToolRegistry(ToolRegistry):
    """ToolRegistry for GUI automation — only GUI tools + finish.

    Overrides dispatch() entirely to avoid the hardcoded parent handler_map,
    which only knows about the base ReAct tools.
    """

    GUI_TOOL_NAMES = {
        "cmd", "screenshot", "zoom", "click", "double_click", "right_click",
        "type", "key", "scroll", "drag", "wait", "note", "finish",
    }

    NOTES_FILE = os.path.expanduser("~/.agent_bin/gui_agent_notes.md")

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
        # Set by _handle_zoom with [full_with_red_box, zoomed_grid] — two images so
        # the model sees both spatial context (full) and precision detail (zoomed).
        self.pending_images = None
        # Set by _handle_zoom: (x_min_px, y_min_px, x_max_px, y_max_px) of the zoomed region.
        # While set, click/scroll coords are auto-translated from zoom-space to full-screen.
        # Cleared by _handle_screenshot.
        self._zoom_region = None
        # Tracks the last tool dispatched — used to block back-to-back screenshots
        self._last_tool_called = None

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

        # Block consecutive screenshots — you must act between them (use wait + screenshot if loading)
        if tool == "screenshot" and self._last_tool_called == "screenshot":
            return ToolResult(
                False, "",
                (
                    "CONSECUTIVE SCREENSHOT BLOCKED: You already have a screenshot. "
                    "Take an action first (click, type, key, cmd, scroll, wait, zoom) "
                    "based on what you saw. If waiting for a page to load: "
                    'wait {"seconds": 2} then screenshot.'
                ),
                {"consecutive_screenshot": True},
            )

        # Emit action event before execution (skip for screenshot — it emits its own)
        if self.event_cb and tool != "screenshot":
            try:
                self.event_cb("action", {"tool": tool, "args": args})
            except Exception:
                pass

        handlers = {
            "cmd":          self._handle_cmd,
            "screenshot":   self._handle_screenshot,
            "zoom":         self._handle_zoom,
            "click":        self._handle_click,
            "double_click": self._handle_double_click,
            "right_click":  self._handle_right_click,
            "type":         self._handle_type,
            "key":          self._handle_key,
            "scroll":       self._handle_scroll,
            "drag":         self._handle_drag,
            "wait":         self._handle_wait,
            "note":         self._handle_note,
            "finish":       self._handle_finish,  # inherited from ToolRegistry
        }
        result = handlers[tool](args)
        self._last_tool_called = tool  # track for consecutive-screenshot guard

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

    def _handle_cmd(self, args):
        try:
            command = str(args.get("command", "")).strip()
            if not command:
                return ToolResult(False, "", "cmd requires 'command'", {})
            env = {**os.environ, "DISPLAY": self.input_ctrl.display}
            xauth = os.path.expanduser("~/.Xauthority")
            if os.path.exists(xauth):
                env["XAUTHORITY"] = xauth
            result = subprocess.run(
                command, shell=True, env=env,
                capture_output=True, text=True, timeout=60,
            )
            output = (result.stdout + result.stderr).strip()
            success = result.returncode == 0
            return ToolResult(
                success,
                output[:2000] if output else "(no output)",
                "" if success else f"exit code {result.returncode}",
                {"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "Command timed out after 60s (append & to run in background)", {})
        except Exception as e:
            return ToolResult(False, "", f"cmd failed: {e}", {})

    # ── Zoom helpers ─────────────────────────────────────────────────────────────

    def _apply_zoom(self, gx, gy):
        """Translate zoom-space 16×16 coords to full-screen 16×16 coords.
        Returns (gx, gy) unchanged if not in zoom mode."""
        if self._zoom_region is None:
            return gx, gy
        x_min, y_min, x_max, y_max = self._zoom_region
        # zoom coords [0,16] → pixel within cropped region
        px = x_min + (gx / 16.0) * (x_max - x_min)
        py = y_min + (gy / 16.0) * (y_max - y_min)
        # pixel → full-screen 16×16 grid coords
        sw = self.input_ctrl.screen_w
        sh = self.input_ctrl.screen_h
        return round(px / sw * 16, 3), round(py / sh * 16, 3)

    def _handle_zoom(self, args):
        try:
            cx = float(args.get("x", 8.0))  # center in full-screen 16×16
            cy = float(args.get("y", 8.0))
            w  = float(args.get("w", 1.0))  # width  in grid cells (default 1 cell)
            h  = float(args.get("h", 1.0))  # height in grid cells (default 1 cell)

            if not (0 <= cx <= 16 and 0 <= cy <= 16):
                return ToolResult(False, "", "zoom center out of range (0-16)", {})

            img = self.screen.capture()
            sw, sh = img.size
            cell_w = sw / 16.0
            cell_h = sh / 16.0

            # Compute crop boundaries
            cx_px = cx / 16.0 * sw
            cy_px = cy / 16.0 * sh
            x_min = max(0, int(cx_px - (w / 2.0) * cell_w))
            y_min = max(0, int(cy_px - (h / 2.0) * cell_h))
            x_max = min(sw, int(cx_px + (w / 2.0) * cell_w))
            y_max = min(sh, int(cy_px + (h / 2.0) * cell_h))

            if x_max - x_min < 4 or y_max - y_min < 4:
                return ToolResult(False, "", "zoom region too small — use larger w/h", {})

            zoom_w = x_max - x_min
            zoom_h = y_max - y_min

            # ── Image 1: full screen with 16×16 grid + red box marking zoom region ──
            cursor = self.input_ctrl._last_click_px
            full_grid = self.screen.overlay_grid(img, cursor=cursor)
            full_with_box = self.screen.draw_zoom_overlay(full_grid, x_min, y_min, x_max, y_max)

            # ── Image 2: zoomed region with its own 16×16 grid ──
            cropped = img.crop((x_min, y_min, x_max, y_max))
            zoom_cursor = None
            lcp = cursor
            if lcp and x_min <= lcp[0] <= x_max and y_min <= lcp[1] <= y_max:
                zoom_cursor = (lcp[0] - x_min, lcp[1] - y_min)
            zoomed_grid = self.screen.overlay_grid(cropped, cursor=zoom_cursor)

            # Send zoomed image to browser UI (for live display)
            b64_browser = self.screen.to_base64(zoomed_grid)
            if self.event_cb:
                try:
                    self.event_cb("screenshot", {"image": b64_browser})
                except Exception:
                    pass

            # Both images downscaled for model — [full_with_box, zoomed_grid]
            self.pending_images = [
                self.screen.to_base64_model(full_with_box, max_w=960),
                self.screen.to_base64_model(zoomed_grid, max_w=960),
            ]
            self.pending_image = None  # clear single-image slot

            # OCR on cropped region, coords remapped to zoom 16×16 space
            elements = self.screen.ocr_elements(cropped)
            for e in elements:
                e["grid_x"] = round(e["cx"] / zoom_w * 16, 2)
                e["grid_y"] = round(e["cy"] / zoom_h * 16, 2)
            text_map = self.screen.build_text_map(elements)

            self._zoom_region = (x_min, y_min, x_max, y_max)

            obs = (
                f"[ZOOM {zoom_w}×{zoom_h}px — centered at grid ({cx},{cy})]\n"
                f"You are receiving TWO images:\n"
                f"  IMAGE 1 — Full screen ({sw}×{sh}) with red box showing exactly where you zoomed.\n"
                f"  IMAGE 2 — Zoomed view with a 16×16 sub-grid. Use these coords to click.\n"
                f"Clicks in this sub-grid auto-translate to full-screen. Call screenshot to exit zoom.\n"
                f"OCR text in zoomed view (sub-grid coords):\n{text_map}\n"
                f"\n"
                f"══ VERIFY BEFORE CLICKING ══\n"
                f"Look at IMAGE 1: is the red box around the CORRECT region of the screen?\n"
                f"Look at IMAGE 2: is your target element CLEARLY VISIBLE?\n"
                f"  BOTH YES → click using sub-grid coords from IMAGE 2\n"
                f"             (zoom tighter if you need more precision: smaller w/h)\n"
                f"  NO (wrong area) → call screenshot, re-identify the target, zoom elsewhere\n"
                f"  NO (element not visible) → zoom a different area or zoom tighter"
            )
            if self.event_cb:
                try:
                    self.event_cb("vision", {"text": f"Zoom {zoom_w}×{zoom_h}px — {len(elements)} OCR — dual-image"})
                except Exception:
                    pass
            return ToolResult(True, obs, "", {"zoom": True, "ocr_count": len(elements)})
        except Exception as e:
            return ToolResult(False, "", f"Zoom failed: {e}", {})

    def _handle_screenshot(self, args):
        self._zoom_region = None   # always exit zoom mode on full screenshot
        self.pending_images = None  # clear dual-image from any prior zoom
        try:
            img = self.screen.capture()
            cursor = self.input_ctrl._last_click_px  # (px, py) or None
            grid_img = self.screen.overlay_grid(img, cursor=cursor)

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

            cursor_note = ""
            if cursor:
                gx = round(cursor[0] / w * 16, 2)
                gy = round(cursor[1] / h * 16, 2)
                cursor_note = f"Last click: pixel ({cursor[0]},{cursor[1]}) = grid ({gx},{gy}) — marked with red dot.\n"

            obs = (
                f"[SCREENSHOT {w}×{h} — 16×16 grid overlaid, image attached]\n"
                f"{cursor_note}"
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

    def _require_zoom(self, tool_name):
        """Return a ToolResult error if not currently in zoom mode, else None."""
        if self._zoom_region is None:
            return ToolResult(
                False, "",
                (
                    f"ZOOM REQUIRED before {tool_name}: you must zoom in to verify the target "
                    "is visible before clicking.\n"
                    "  1. zoom {\"x\": <cx>, \"y\": <cy>}  — first zoom to rough area\n"
                    "  2. zoom {\"x\": <cx>, \"y\": <cy>, \"w\": 0.5, \"h\": 0.5}  — tighter zoom\n"
                    f"  3. {tool_name} {{\"x\": ..., \"y\": ...}}  — only once target is confirmed visible"
                ),
                {"zoom_required": True},
            )
        return None

    def _handle_click(self, args):
        err = self._require_zoom("click")
        if err:
            return err
        try:
            x = float(args.get("x", 0))
            y = float(args.get("y", 0))
            if not (0 <= x <= 16 and 0 <= y <= 16):
                return ToolResult(False, "", f"Coords out of range: x={x}, y={y} (must be 0–16)", {})
            x, y = self._apply_zoom(x, y)
            msg = self.input_ctrl.click(x, y)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Click failed: {e}", {})

    def _handle_double_click(self, args):
        err = self._require_zoom("double_click")
        if err:
            return err
        try:
            x = float(args.get("x", 0))
            y = float(args.get("y", 0))
            if not (0 <= x <= 16 and 0 <= y <= 16):
                return ToolResult(False, "", f"Coords out of range: x={x}, y={y} (must be 0–16)", {})
            x, y = self._apply_zoom(x, y)
            msg = self.input_ctrl.double_click(x, y)
            return ToolResult(True, msg, "", {})
        except Exception as e:
            return ToolResult(False, "", f"Double-click failed: {e}", {})

    def _handle_right_click(self, args):
        err = self._require_zoom("right_click")
        if err:
            return err
        try:
            x = float(args.get("x", 0))
            y = float(args.get("y", 0))
            if not (0 <= x <= 16 and 0 <= y <= 16):
                return ToolResult(False, "", f"Coords out of range: x={x}, y={y} (must be 0–16)", {})
            x, y = self._apply_zoom(x, y)
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
            x = args.get("x")
            y = args.get("y")
            if x is not None and y is not None:
                x, y = float(x), float(y)
                x, y = self._apply_zoom(x, y)
            msg = self.input_ctrl.scroll(direction, x=x, y=y)
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
                if not (0 <= val <= 16):
                    return ToolResult(False, "", f"{coord}={val} out of range (0–16)", {})
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

    def _handle_note(self, args):
        try:
            text = str(args.get("text", "")).strip()
            if not text:
                return ToolResult(False, "", "note requires 'text'", {})
            notes_dir = os.path.dirname(self.NOTES_FILE)
            os.makedirs(notes_dir, exist_ok=True)
            timestamp = time.strftime("%Y-%m-%d %H:%M")
            entry = f"\n- [{timestamp}] {text}\n"
            with open(self.NOTES_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            return ToolResult(True, f"Note saved: {text[:120]}", "", {})
        except Exception as e:
            return ToolResult(False, "", f"Note failed: {e}", {})
