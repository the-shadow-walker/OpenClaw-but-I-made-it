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
from gui_dom import DOMExtractor
from gui_profiles import ProfileStore
from gui_macros import MacroStore
from gui_elements import ElementRegistry

try:
    from gui_atspi import ATSPIExtractor
    _ATSPI_OK = True
except (ImportError, Exception):
    _ATSPI_OK = False
    ATSPIExtractor = None

try:
    from gui_cv import CVExtractor
    _CV_OK = True
except (ImportError, Exception):
    _CV_OK = False
    CVExtractor = None


GUI_TOOL_SCHEMAS = {
    "plan":         '  plan          — {"steps": ["1. do x", "2. do y", ...]}  # set/update task plan; shown on every screenshot',
    "cmd":          '  cmd           — {"command": str}  # run shell command, returns output; append & for background',
    "screenshot":   '  screenshot    — {}  # full-screen; also exits zoom mode',
    "zoom":         '  zoom          — {"id": int} or {"x": float, "y": float, "w": float, "h": float}  # id centers zoom on that element; w/h in grid cells (default 2×2)',
    "click":        '  click         — {"id": int} or {"x": float, "y": float}  # prefer id when listed',
    "double_click": '  double_click  — {"id": int} or {"x": float, "y": float}  # prefer id when listed',
    "right_click":  '  right_click   — {"id": int} or {"x": float, "y": float}  # prefer id when listed',
    "rescan":       '  rescan        — {} or {"hint": "tiny X button"}  # re-discover elements; hint triggers relaxed CV (finds smaller elements)',
    "type":         '  type          — {"text": str}',
    "key":          '  key           — {"combo": str}',
    "scroll":       '  scroll        — {"direction": "up"|"down", "x": float, "y": float}  # x,y: move mouse there first (required for browser scroll)',
    "drag":         '  drag          — {"x1": float, "y1": float, "x2": float, "y2": float}',
    "wait":         '  wait          — {"seconds": float}',
    "note":         '  note          — {"text": str}  # save a general discovery (binary paths, UI quirks, tips)',
    "save_profile": '  save_profile  — {"name": str, "site": str, "task": str, "steps": str, "notes": str}  # document a repeatable task flow for future runs',
    "sequence":     '  sequence      — {"steps": [{"tool":"click","args":{"x":3.1,"y":5.2}}, ...]} or {"macro": "NAME"}  # execute multiple tools in ONE turn',
    "save_macro":   '  save_macro    — {"name": str, "description": str, "steps": [...]}  # cache an action sequence for instant replay next run',
    # ---- session continuity / cross-agent delegation ----
    "code_task":    '  code_task     — {"task": str, "max_iterations": int (default 25), "context_keys": [str] (optional)}\n'
                    '                  # Delegate a code/file/shell task to the CMD agent (subordinate). Snapshots GUI state,\n'
                    '                  # runs CMD ReAct in isolation, merges files-created back, pins ONE clean result.',
    "publish_context": '  publish_context — {"key": str, "value": str, "ttl_hours": int (default 24)}\n'
                       '                  # Write to the central shared_context board so cmd/swarm agents can read it.',
    "read_context": '  read_context  — {"key": str} OR {"prefix": str, "limit": int (default 10)}\n'
                    '                  # Read from the central shared_context board.',
    "finish":       '  finish        — {"summary": str, "success": bool}',
}

GUI_TOOLS_TEXT = "\n".join(GUI_TOOL_SCHEMAS.values())


class GUIToolRegistry(ToolRegistry):
    """ToolRegistry for GUI automation — only GUI tools + finish.

    Overrides dispatch() entirely to avoid the hardcoded parent handler_map,
    which only knows about the base ReAct tools.
    """

    GUI_TOOL_NAMES = {
        "plan", "cmd", "screenshot", "zoom", "click", "double_click", "right_click",
        "type", "key", "scroll", "drag", "wait", "note", "save_profile",
        "sequence", "save_macro", "rescan", "finish",
        # Cross-agent delegation + central context
        "code_task", "publish_context", "read_context",
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
        # Counts consecutive zoom calls since the last click/screenshot.
        # When this hits 2 the model is told to stop seeking precision and just click.
        self._zooms_since_action = 0
        # OCR elements from the last full screenshot, as raw pixel coords.
        # Reused by _handle_zoom so we never OCR tiny crops (unreliable at <240px).
        self._last_ocr_elements = []   # [{text, cx_px, cy_px}]
        # DOM elements from the last CDP query, as raw pixel coords.
        # Reused by _handle_zoom to filter into zoom-space.
        self._last_dom_elements = []   # [{tag, text, x_px, y_px, grid_x, grid_y}]
        self._dom = DOMExtractor(
            screen_w=self.input_ctrl.screen_w,
            screen_h=self.input_ctrl.screen_h,
        )
        # Current task plan — list of step strings set by plan tool.
        # Shown on every screenshot observation so the model never loses its roadmap.
        self._current_plan: list = []
        # Profile store for save_profile tool.
        self._profile_store = ProfileStore()
        # Macro store for sequence / save_macro tools.
        self._macro_store = MacroStore()
        # Set-of-Marks: element registry, optional AT-SPI + CV sources, last raw image.
        self._registry = ElementRegistry(
            screen_w=self.input_ctrl.screen_w,
            screen_h=self.input_ctrl.screen_h,
        )
        self._atspi = ATSPIExtractor() if _ATSPI_OK else None
        self._cv    = CVExtractor()    if _CV_OK    else None
        self._last_raw_img = None  # raw PIL image (no grid overlay); used by rescan
        # Run archive: set by gui_agent.run() before each task
        self._run_dir: str = None
        self._screenshot_counter: int = 0

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

        # Block excess zooming — 2 zooms in a row without clicking means stop fine-tuning and act
        if tool == "zoom" and self._zooms_since_action >= 2:
            return ToolResult(
                False, "",
                (
                    "TOO MANY ZOOMS: you have zoomed twice without clicking anything. "
                    "You are close enough — stop seeking perfection and click now. "
                    "Use the OCR coordinates from the last zoom observation, "
                    "do your X-pass then Y-pass mentally, and click. "
                    "If you are genuinely lost, call screenshot to reset."
                ),
                {"too_many_zooms": True},
            )

        # Emit action event before execution (skip for screenshot — it emits its own)
        if self.event_cb and tool != "screenshot":
            try:
                self.event_cb("action", {"tool": tool, "args": args})
            except Exception:
                pass

        handlers = {
            "plan":         self._handle_plan,
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
            "save_profile": self._handle_save_profile,
            "sequence":     self._handle_sequence,
            "save_macro":   self._handle_save_macro,
            "rescan":       self._handle_rescan,
            "finish":       self._handle_finish,  # inherited from ToolRegistry
            # Cross-agent delegation + central context
            "code_task":       self._handle_code_task,
            "publish_context": self._handle_gui_publish_context,
            "read_context":    self._handle_gui_read_context,
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
            # Block re-zooming while already in zoom mode — sub-zooms produce
            # tiny unreadable crops. One zoom level only: screenshot → zoom → click.
            if self._zoom_region is not None:
                return ToolResult(
                    False, "",
                    (
                        "ALREADY ZOOMED: you are in a zoom view. "
                        "Do not zoom again — either click your target now, "
                        "or call screenshot to return to full-screen and zoom elsewhere."
                    ),
                    {"already_zoomed": True},
                )

            # Support zoom {"id": N} — automatically center on that element
            eid = args.get("id")
            if eid is not None:
                el = self._registry.get_by_id(int(eid))
                if el is None:
                    return ToolResult(
                        False, "",
                        f"ID {eid} not found — call screenshot or rescan to refresh IDs",
                        {},
                    )
                cx = el.grid_x
                cy = el.grid_y
            else:
                cx = float(args.get("x", 8.0))
                cy = float(args.get("y", 8.0))

            w  = float(args.get("w", 2.0))   # default 2 cells wide
            h  = float(args.get("h", 2.0))   # default 2 cells tall

            if not (0 <= cx <= 16 and 0 <= cy <= 16):
                return ToolResult(False, "", "zoom center out of range (0-16)", {})

            # Enforce minimum zoom size so the crop is always readable
            w = max(w, 1.5)
            h = max(h, 1.5)

            cx_fs, cy_fs = cx, cy  # always full-screen coords (no sub-zoom)

            img = self.screen.capture()
            sw, sh = img.size
            cell_w = sw / 16.0
            cell_h = sh / 16.0

            cx_px  = cx_fs / 16.0 * sw
            cy_px  = cy_fs / 16.0 * sh

            # w/h are still in grid-cell units of the FULL SCREEN grid
            x_min = max(0, int(cx_px - (w / 2.0) * cell_w))
            y_min = max(0, int(cy_px - (h / 2.0) * cell_h))
            x_max = min(sw, int(cx_px + (w / 2.0) * cell_w))
            y_max = min(sh, int(cy_px + (h / 2.0) * cell_h))

            if x_max - x_min < 4 or y_max - y_min < 4:
                return ToolResult(False, "", "zoom region too small — use larger w/h", {})

            zoom_w = x_max - x_min
            zoom_h = y_max - y_min

            # ── Build full-screen context image with red box ───────────────────
            cursor        = self.input_ctrl._last_click_px
            full_grid     = self.screen.overlay_grid(img, cursor=cursor)
            full_with_box = self.screen.draw_zoom_overlay(full_grid, x_min, y_min, x_max, y_max)

            # ── Build zoomed detail image ──────────────────────────────────────
            cropped     = img.crop((x_min, y_min, x_max, y_max))
            zoom_cursor = None
            if cursor and x_min <= cursor[0] <= x_max and y_min <= cursor[1] <= y_max:
                zoom_cursor = (cursor[0] - x_min, cursor[1] - y_min)
            zoomed_grid = self.screen.overlay_grid(cropped, cursor=zoom_cursor)

            # ── Compose single side-by-side panel for the model ───────────────
            # One image = reliable; two separate images risk the model ignoring one.
            panel = self.screen.compose_zoom_panel(full_with_box, zoomed_grid, max_w=1280)

            # Send zoomed view to browser UI (live display), panel to model
            b64_browser = self.screen.to_base64(zoomed_grid)
            if self.event_cb:
                try:
                    self.event_cb("screenshot", {"image": b64_browser})
                except Exception:
                    pass

            self.pending_image  = self.screen.to_base64_model(panel, max_w=1280)
            self.pending_images = None   # not used for zoom anymore

            # ── Reuse stored OCR + DOM — filter both to zoom region ──────────
            # Never run tesseract on the tiny crop — chars are 4-6px at <300px.
            zoom_ocr = []
            for e in self._last_ocr_elements:
                cx, cy = e["cx_px"], e["cy_px"]
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    zoom_ocr.append({
                        "text":   e["text"],
                        "cx":     cx - x_min,
                        "cy":     cy - y_min,
                        "grid_x": round((cx - x_min) / zoom_w * 16, 2),
                        "grid_y": round((cy - y_min) / zoom_h * 16, 2),
                    })
            text_map = self.screen.build_text_map(zoom_ocr)

            zoom_dom = []
            for e in self._last_dom_elements:
                cx, cy = e["x_px"], e["y_px"]
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    zoom_dom.append({
                        "tag":    e["tag"],
                        "text":   e["text"],
                        "x_px":   cx - x_min,
                        "y_px":   cy - y_min,
                        "grid_x": round((cx - x_min) / zoom_w * 16, 2),
                        "grid_y": round((cy - y_min) / zoom_h * 16, 2),
                    })
            dom_zoom_map = self._dom.build_element_map(zoom_dom)

            dom_zoom_section = ""
            if dom_zoom_map:
                dom_zoom_section = (
                    f"Interactive elements in zoom (RIGHT panel coords, click these):\n"
                    f"{dom_zoom_map}\n"
                )

            # Update zoom region AFTER all calculations
            self._zoom_region = (x_min, y_min, x_max, y_max)
            self._zooms_since_action += 1

            obs = (
                f"[ZOOM {zoom_w}×{zoom_h}px — center ({cx_fs:.2f},{cy_fs:.2f})]\n"
                f"LEFT panel: full screen + red box. RIGHT panel: zoomed 16×16 sub-grid.\n"
                f"Sub-grid coords auto-translate to full-screen when you click.\n"
                f"{dom_zoom_section}"
                f"OCR text in zoomed view:\n{text_map}\n"
            )
            if self.event_cb:
                try:
                    self.event_cb("vision", {
                        "text": f"Zoom {zoom_w}×{zoom_h}px — {len(zoom_dom)} DOM + {len(zoom_ocr)} OCR"
                    })
                except Exception:
                    pass
            return ToolResult(True, obs, "", {
                "zoom": True,
                "ocr_count": len(zoom_ocr),
                "dom_count": len(zoom_dom),
            })
        except Exception as e:
            return ToolResult(False, "", f"Zoom failed: {e}", {})

    def _build_registry(self, raw_img, relaxed_cv: bool = False):
        """Merge AT-SPI + DOM + CV candidates into a fresh ElementRegistry.

        Order of operations:
          1. AT-SPI — desktop accessibility tree (highest priority)
          2. DOM    — Chrome DevTools Protocol (browser elements)
          3. CV     — OpenCV gap-filler, with AT-SPI+DOM bboxes masked out

        relaxed_cv: pass relaxed=True to CVExtractor (lower area/aspect thresholds)
                    used by rescan when the model hints at small/unusual elements.
        """
        sw, sh = raw_img.size
        reg = ElementRegistry(screen_w=sw, screen_h=sh)

        # Step 1: AT-SPI (highest priority)
        atspi_candidates = []
        if self._atspi is not None:
            try:
                for e in self._atspi.extract():
                    atspi_candidates.append({**e, "source": "atspi"})
            except Exception:
                atspi_candidates = []

        # Step 2: DOM elements
        dom_elements = self._dom.extract()
        self._last_dom_elements = dom_elements
        dom_candidates = []
        for e in dom_elements:
            dom_candidates.append({
                "tag":    e.get("tag", "element"),
                "text":   e.get("text", ""),
                "x_px":   float(e.get("x_px", 0)),
                "y_px":   float(e.get("y_px", 0)),
                "w_px":   float(e.get("w_px", e.get("width", 0))),
                "h_px":   float(e.get("h_px", e.get("height", 0))),
                "source": "dom",
            })

        # Step 3: CV gap-filler — pass combined AT-SPI+DOM list as the mask
        # so CV only looks at screen areas not already covered by known elements.
        cv_candidates = []
        if self._cv is not None:
            try:
                combined_known = atspi_candidates + dom_candidates
                for e in self._cv.extract(raw_img, existing_elements=combined_known,
                                          relaxed=relaxed_cv):
                    cv_candidates.append({**e, "source": "cv"})
            except Exception:
                cv_candidates = []

        # Debug log — per-source counts
        mode = " [relaxed]" if relaxed_cv else ""
        print(
            f"  [SoM{mode}] Found {len(atspi_candidates)} AT-SPI, "
            f"{len(dom_candidates)} DOM, {len(cv_candidates)} CV"
        )

        # Merge: cv first (lowest priority), dom, atspi last (highest wins on dup)
        reg.merge_elements(cv_candidates + dom_candidates + atspi_candidates)
        return reg, dom_elements

    def _handle_screenshot(self, args):
        self._zoom_region = None        # always exit zoom mode on full screenshot
        self.pending_images = None      # clear dual-image from any prior zoom
        self._zooms_since_action = 0    # reset zoom counter on full screenshot
        try:
            img = self.screen.capture()
            self._last_raw_img = img    # store raw image for rescan

            cursor = self.input_ctrl._last_click_px  # (px, py) or None
            w, h = img.size

            # ── Build element registry (AT-SPI + DOM + CV) ────────────────────
            self._registry, dom_elements = self._build_registry(img)

            # ── Render Set-of-Marks markers onto a copy of the image ──────────
            marked_img = self.screen.render_markers(img, self._registry)
            # Then draw grid + cursor on top of the markers
            grid_img = self.screen.overlay_grid(marked_img, cursor=cursor)

            # Full-res base64 → browser UI only
            b64_full = self.screen.to_base64(grid_img)
            if self.event_cb:
                try:
                    self.event_cb("screenshot", {"image": b64_full})
                except Exception:
                    pass

            # Downscaled base64 → inlined into next ReAct model call
            self.pending_image = self.screen.to_base64_model(grid_img, max_w=960)

            # OCR (for fallback text map — reuse for zoom)
            elements = self.screen.ocr_elements(img)
            self._last_ocr_elements = [
                {"text": e["text"], "cx_px": e["cx"], "cy_px": e["cy"]}
                for e in elements
            ]
            text_map = self.screen.build_text_map(elements)

            cursor_note = ""
            if cursor:
                gx = round(cursor[0] / w * 16, 2)
                gy = round(cursor[1] / h * 16, 2)
                cursor_note = f"Last click: pixel ({cursor[0]},{cursor[1]}) = grid ({gx},{gy}) — marked with red dot.\n"

            plan_section = ""
            if self._current_plan:
                plan_lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self._current_plan))
                plan_section = f"PLAN:\n{plan_lines}\n"

            # Element list — use registry (all sources merged, deduped, ID-assigned)
            element_list = self._registry.format_for_prompt()
            n_atspi = sum(1 for e in self._registry.elements if e.source == "atspi")
            n_dom   = sum(1 for e in self._registry.elements if e.source == "dom")
            n_cv    = sum(1 for e in self._registry.elements if e.source == "cv")

            obs = (
                f"[SCREENSHOT {w}×{h} — markers overlaid, image attached]\n"
                f"{cursor_note}"
                f"{plan_section}"
                f"Elements (click by ID — prefer ID over coordinates):\n"
                f"  Color legend: [Blue]=DOM  [Green]=AT-SPI  [Orange]=CV\n"
                f"{element_list}\n"
                f"OCR text (fallback — less precise):\n"
                f"{text_map}\n"
            )

            # ── Save screenshot + observation to run archive ──────────────────
            if self._run_dir:
                try:
                    self._screenshot_counter += 1
                    n = self._screenshot_counter
                    grid_img.save(os.path.join(self._run_dir, f"{n:04d}_screenshot.png"))
                    with open(os.path.join(self._run_dir, f"{n:04d}_observation.txt"),
                              "w", encoding="utf-8") as _f:
                        _f.write(obs)
                except Exception:
                    pass

            if self.event_cb:
                try:
                    self.event_cb("vision", {
                        "text": (
                            f"{w}×{h} — {n_dom} DOM + {n_atspi} AT-SPI + "
                            f"{n_cv} CV = {len(self._registry.elements)} total"
                        )
                    })
                except Exception:
                    pass

            return ToolResult(True, obs, "", {
                "screenshot": True,
                "ocr_count": len(elements),
                "dom_count": len(dom_elements),
                "element_count": len(self._registry.elements),
            })
        except Exception as e:
            if self.event_cb:
                try:
                    self.event_cb("error", {"text": f"Screenshot failed: {e}"})
                except Exception:
                    pass
            return ToolResult(False, "", f"Screenshot failed: {e}", {})

    def _resolve_click_coords(self, args, action_name="click"):
        """Resolve click coordinates from either {'id': N} or {'x': f, 'y': f}.

        Returns (x, y) as 16×16 grid floats, or raises ValueError with a message.
        """
        eid = args.get("id")
        if eid is not None:
            el = self._registry.get_by_id(int(eid))
            if el is None:
                raise ValueError(
                    f"ID {eid} not found in current element list — "
                    "call rescan {{}} or screenshot first to refresh IDs"
                )
            return el.grid_x, el.grid_y
        # Fall back to explicit x/y coords
        x = float(args.get("x", 0))
        y = float(args.get("y", 0))
        if not (0 <= x <= 16 and 0 <= y <= 16):
            raise ValueError(f"Coords out of range: x={x}, y={y} (must be 0–16)")
        return x, y

    def _handle_click(self, args):
        try:
            x, y = self._resolve_click_coords(args, "click")
            x, y = self._apply_zoom(x, y)
            self._zooms_since_action = 0
            msg = self.input_ctrl.click(x, y)
            return ToolResult(True, msg, "", {})
        except ValueError as e:
            return ToolResult(False, "", str(e), {})
        except Exception as e:
            return ToolResult(False, "", f"Click failed: {e}", {})

    def _handle_double_click(self, args):
        try:
            x, y = self._resolve_click_coords(args, "double_click")
            x, y = self._apply_zoom(x, y)
            self._zooms_since_action = 0
            msg = self.input_ctrl.double_click(x, y)
            return ToolResult(True, msg, "", {})
        except ValueError as e:
            return ToolResult(False, "", str(e), {})
        except Exception as e:
            return ToolResult(False, "", f"Double-click failed: {e}", {})

    def _handle_right_click(self, args):
        try:
            x, y = self._resolve_click_coords(args, "right_click")
            x, y = self._apply_zoom(x, y)
            self._zooms_since_action = 0
            msg = self.input_ctrl.right_click(x, y)
            return ToolResult(True, msg, "", {})
        except ValueError as e:
            return ToolResult(False, "", str(e), {})
        except Exception as e:
            return ToolResult(False, "", f"Right-click failed: {e}", {})

    def _handle_rescan(self, args):
        """Re-discover elements from the last raw screenshot without recapturing.

        Optional args:
          hint (str) — description of what to look for (e.g. "tiny X button")
                       triggers relaxed CV thresholds (smaller min area, wider aspect)
        """
        if self._last_raw_img is None:
            return ToolResult(
                False, "",
                "No screenshot yet — call screenshot {} first, then rescan",
                {},
            )
        try:
            hint = str(args.get("hint", "")).strip()
            relaxed = bool(hint)  # use relaxed CV thresholds when a hint is given

            self._registry, _ = self._build_registry(
                self._last_raw_img, relaxed_cv=relaxed
            )
            element_list = self._registry.format_for_prompt()
            n = len(self._registry.elements)

            hint_note = f" (relaxed CV thresholds — hint: '{hint}')" if hint else ""
            return ToolResult(
                True,
                (
                    f"Rescan complete{hint_note} — {n} elements found, IDs reassigned.\n"
                    f"Refer to the updated IDs below:\n"
                    f"{element_list}"
                ),
                "",
                {"rescan": True, "element_count": n, "relaxed": relaxed},
            )
        except Exception as e:
            return ToolResult(False, "", f"Rescan failed: {e}", {})

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

    def _handle_plan(self, args):
        """Store a task plan. Displayed on every subsequent screenshot observation."""
        try:
            steps = args.get("steps", [])
            if isinstance(steps, str):
                # Model sometimes passes a newline-separated string
                steps = [s.strip() for s in steps.strip().splitlines() if s.strip()]
            if not steps:
                return ToolResult(False, "", "plan requires 'steps' as a non-empty list or string", {})
            self._current_plan = [str(s) for s in steps]
            formatted = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self._current_plan))
            return ToolResult(True, f"Plan set ({len(self._current_plan)} steps):\n{formatted}", "", {})
        except Exception as e:
            return ToolResult(False, "", f"Plan failed: {e}", {})

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

    def _handle_save_profile(self, args):
        """Save a reusable task profile for future runs."""
        try:
            name  = str(args.get("name",  "")).strip()
            site  = str(args.get("site",  "")).strip()
            task  = str(args.get("task",  "")).strip()
            steps = str(args.get("steps", "")).strip()
            notes = str(args.get("notes", "")).strip()
            if not name or not steps:
                return ToolResult(False, "", "save_profile requires 'name' and 'steps'", {})
            saved = self._profile_store.save(name, site, task, steps, notes)
            return ToolResult(True, f"Profile saved: {saved}", "", {"profile": saved})
        except Exception as e:
            return ToolResult(False, "", f"save_profile failed: {e}", {})

    def _handle_sequence(self, args):
        """Execute a list of tool calls in order — DuckyScript style, one agent turn.

        Args:
          steps  — inline list: [{"tool": "click", "args": {"x": 3.1, "y": 5.2}}, ...]
          macro  — named macro: loads steps from MacroStore
        Screenshots within the sequence update pending_image normally.
        Stops on the first hard failure (non-screenshot, non-wait tool).
        """
        try:
            # Load steps from macro name or inline list
            macro_name = args.get("macro", "")
            if macro_name:
                macro = self._macro_store.load(macro_name)
                if not macro:
                    return ToolResult(False, "",
                        f"Macro '{macro_name}' not found. Available: "
                        f"{[m['name'] for m in self._macro_store.list_all()]}",
                        {})
                steps = macro.get("steps", [])
                header = f"[MACRO: {macro['name']} — {macro.get('description', '')}]\n"
            else:
                steps = args.get("steps", [])
                header = "[SEQUENCE]\n"

            if not steps:
                return ToolResult(False, "", "sequence requires 'steps' list or valid 'macro' name", {})

            # Build handler map (same as dispatch but restricted — no recursion)
            _handlers = {
                "plan":         self._handle_plan,
                "cmd":          self._handle_cmd,
                "screenshot":   self._handle_screenshot,
                "click":        self._handle_click,
                "double_click": self._handle_double_click,
                "right_click":  self._handle_right_click,
                "type":         self._handle_type,
                "key":          self._handle_key,
                "scroll":       self._handle_scroll,
                "drag":         self._handle_drag,
                "wait":         self._handle_wait,
                "note":         self._handle_note,
                "zoom":         self._handle_zoom,
                "rescan":       self._handle_rescan,
            }

            outputs = [header]
            for i, step in enumerate(steps):
                tool = step.get("tool", "")
                step_args = step.get("args", {})

                if tool in ("sequence", "save_macro", "save_profile", "finish"):
                    outputs.append(f"  step {i+1}: SKIP — '{tool}' not allowed inside sequence")
                    continue

                handler = _handlers.get(tool)
                if not handler:
                    outputs.append(f"  step {i+1}: ERROR — unknown tool '{tool}'")
                    continue

                result = handler(step_args)
                self._last_tool_called = tool

                status = "OK  " if result.success else "FAIL"
                detail = (result.output or result.error or "")[:300]
                outputs.append(f"  step {i+1} [{tool}] {status}: {detail}")

                # Stop on hard failure (not wait/screenshot which always "succeed")
                if not result.success and tool not in ("wait", "screenshot", "note"):
                    outputs.append(f"  [sequence stopped at step {i+1} due to failure]")
                    break

            return ToolResult(True, "\n".join(outputs), "", {
                "sequence": True,
                "steps_run": len(steps),
            })
        except Exception as e:
            return ToolResult(False, "", f"sequence failed: {e}", {})

    def _handle_save_macro(self, args):
        """Save an action sequence as a named macro for future single-turn replay."""
        try:
            name        = str(args.get("name",        "")).strip()
            description = str(args.get("description", "")).strip()
            steps       = args.get("steps", [])
            if not name:
                return ToolResult(False, "", "save_macro requires 'name'", {})
            if not steps or not isinstance(steps, list):
                return ToolResult(False, "", "save_macro requires 'steps' as a list of tool calls", {})
            saved = self._macro_store.save(name, description, steps)
            return ToolResult(True,
                f"Macro saved: {saved} ({len(steps)} steps)\n"
                f"Next run: sequence {{\"macro\": \"{saved}\"}}",
                "", {"macro": saved})
        except Exception as e:
            return ToolResult(False, "", f"save_macro failed: {e}", {})

    # ── Cross-agent delegation + central context (GUI side) ──────────────────

    def _handle_code_task(self, args):
        """Delegate a code/file/shell task to the CMD agent (GUI is parent here).

        Mirrors gui_task in CMD: snapshots GUI parent state, runs CMD ReAct in
        isolation, merges files-created back, returns ONE clean ToolResult.
        Sidechain trace is dumped to ~/.agent_bin/sidechains/ — never leaks
        into GUI conversation history.
        """
        try:
            task = str(args.get("task", "")).strip()
            if not task:
                return ToolResult(False, "", "code_task requires 'task'", {})
            max_iter = int(args.get("max_iterations", 25))
            context_keys = args.get("context_keys") or []

            # Lazy import to avoid circular dep at module load
            try:
                from subagent import SubAgentInvoker
            except ImportError as e:
                return ToolResult(False, "",
                    f"code_task unavailable — SubAgentInvoker missing: {e}", {})

            # GUIToolRegistry has no OllamaCommandAgent parent — build a minimal
            # one for snapshot/file-merge bookkeeping (degraded parent ok).
            class _ParentShim:
                _files_created: list = []
                _pinned_slot_keys: dict = {}
                pinned_messages: list = []
                _current_instruction: str = ""
                def _update_pinned(self, key, msg):
                    self._pinned_slot_keys[key] = msg
                def _refresh_file_manifest_pin(self):
                    pass
                def save_context(self, label):
                    return ""

            shim = _ParentShim()
            shim._current_instruction = f"GUI-delegated code_task: {task[:200]}"

            invoker = SubAgentInvoker(parent_agent=shim, memory=self.memory)
            sub_result = invoker.run(
                target="cmd",
                task=task,
                max_iterations=max_iter,
                context_keys=context_keys,
            )

            # Track files in GUI notes (so subsequent GUI screenshots can reference them)
            files = sub_result.files_created or []
            if files:
                try:
                    notes_dir = os.path.dirname(self.NOTES_FILE)
                    os.makedirs(notes_dir, exist_ok=True)
                    ts = time.strftime("%Y-%m-%d %H:%M")
                    with open(self.NOTES_FILE, "a", encoding="utf-8") as f:
                        f.write(f"\n- [{ts}] code_task delegated → CMD; files: {files}\n")
                except Exception:
                    pass

            summary = sub_result.parent_summary or "(no summary)"
            output = (
                f"[code_task → CMD] {summary}\n"
                f"Files created: {files if files else '(none)'}\n"
                f"Iterations: {sub_result.iterations_used}  Elapsed: {sub_result.elapsed_ms}ms"
            )
            return ToolResult(
                sub_result.success, output,
                "" if sub_result.success else (sub_result.error or "code_task failed"),
                {
                    "delegated_to": "cmd",
                    "files_created": files,
                    "sidechain_path": sub_result.sidechain_path,
                    "snapshot_path": sub_result.snapshot_path,
                },
            )
        except Exception as e:
            return ToolResult(False, "", f"code_task failed: {e}", {})

    def _handle_gui_publish_context(self, args):
        """Write a key/value to the central shared_context board (TTL'd)."""
        try:
            key = str(args.get("key", "")).strip()
            value = args.get("value", "")
            ttl_hours = int(args.get("ttl_hours", 24))
            if not key:
                return ToolResult(False, "", "publish_context requires 'key'", {})
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            else:
                value = str(value)
            ttl_seconds = ttl_hours * 3600
            self.memory.set_context(key, value, agent_id="gui", ttl=ttl_seconds)
            return ToolResult(
                True,
                f"Published context[{key}] (ttl={ttl_hours}h, agent=gui, len={len(value)})",
                "",
                {"key": key, "ttl_hours": ttl_hours},
            )
        except Exception as e:
            return ToolResult(False, "", f"publish_context failed: {e}", {})

    def _handle_gui_read_context(self, args):
        """Read from the central shared_context board (single key OR prefix listing)."""
        try:
            key = args.get("key")
            prefix = args.get("prefix")
            limit = int(args.get("limit", 10))
            if key:
                value = self.memory.get_context(str(key))
                if value is None:
                    return ToolResult(False, "", f"no context entry for key '{key}'", {})
                return ToolResult(
                    True, f"context[{key}] = {value}", "",
                    {"key": str(key), "value": value},
                )
            if prefix is not None:
                entries = self.memory.list_context(str(prefix))
                if not entries:
                    return ToolResult(
                        True, f"(no entries with prefix '{prefix}')", "",
                        {"prefix": str(prefix), "entries": []},
                    )
                lines = []
                for e in entries[:limit]:
                    val = e.get("value", "")
                    if len(val) > 200:
                        val = val[:200] + "…"
                    lines.append(f"  {e.get('key','?')} [{e.get('agent_id','?')}] = {val}")
                return ToolResult(
                    True,
                    f"context entries with prefix '{prefix}' (showing {len(lines)}/{len(entries)}):\n"
                    + "\n".join(lines),
                    "",
                    {"prefix": str(prefix), "count": len(entries)},
                )
            return ToolResult(False, "", "read_context requires 'key' or 'prefix'", {})
        except Exception as e:
            return ToolResult(False, "", f"read_context failed: {e}", {})
