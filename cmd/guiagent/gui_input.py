"""
gui_input.py — xdotool wrappers for mouse/keyboard control.
"""

import os
import subprocess
import time


class GUIInput:
    def __init__(self, display=":99", screen_w=1280, screen_h=720):
        self.display = display
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._last_click_px = None  # (px, py) of most recent click/double/right-click

    def _grid_to_px(self, gx, gy):
        """Convert 16×16 grid coordinates to pixel coordinates."""
        px = int((float(gx) / 16.0) * self.screen_w)
        py = int((float(gy) / 16.0) * self.screen_h)
        px = max(0, min(self.screen_w - 1, px))
        py = max(0, min(self.screen_h - 1, py))
        return px, py

    def _xdo(self, *args, timeout=10):
        """Run xdotool with DISPLAY set."""
        env = {**os.environ, "DISPLAY": self.display}
        return subprocess.run(
            ["xdotool"] + list(args),
            env=env, capture_output=True, text=True, timeout=timeout,
        )

    def _get_xterm_wid(self):
        """Return the first xterm window ID, or None if not found."""
        r = self._xdo("search", "--class", "XTerm")
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().split("\n")[0]
        return None

    def focus_xterm(self):
        """Find the xterm window and give it keyboard focus."""
        wid = self._get_xterm_wid()
        if wid:
            self._xdo("windowactivate", "--sync", wid)
            self._xdo("windowfocus", "--sync", wid)
            time.sleep(0.1)
            return f"Focused xterm window {wid}"
        return "xterm window not found"

    def _ensure_focused(self):
        """If no window has keyboard focus, focus xterm as fallback."""
        r = self._xdo("getactivewindow")
        if r.returncode != 0 or not r.stdout.strip():
            self.focus_xterm()

    def click(self, gx, gy):
        px, py = self._grid_to_px(gx, gy)
        self._xdo("mousemove", "--sync", str(px), str(py))
        self._xdo("click", "1")
        self._last_click_px = (px, py)
        return f"Clicked at grid ({gx}, {gy}) → pixel ({px}, {py})"

    def double_click(self, gx, gy):
        px, py = self._grid_to_px(gx, gy)
        self._xdo("mousemove", "--sync", str(px), str(py))
        self._xdo("click", "--repeat", "2", "--delay", "100", "1")
        self._last_click_px = (px, py)
        return f"Double-clicked at grid ({gx}, {gy}) → pixel ({px}, {py})"

    def right_click(self, gx, gy):
        px, py = self._grid_to_px(gx, gy)
        self._xdo("mousemove", "--sync", str(px), str(py))
        self._xdo("click", "3")
        self._last_click_px = (px, py)
        return f"Right-clicked at grid ({gx}, {gy}) → pixel ({px}, {py})"

    def type_text(self, text):
        # Ensure some window has focus so keystrokes land
        self._ensure_focused()
        self._xdo("type", "--clearmodifiers", "--", text, timeout=30)
        return f"Typed: {text[:60]}{'...' if len(text) > 60 else ''}"

    def key(self, combo):
        # Ensure some window has focus so the key combo lands
        self._ensure_focused()
        self._xdo("key", "--clearmodifiers", combo)
        return f"Key: {combo}"

    def scroll(self, direction, x=None, y=None, clicks=5):
        """Scroll up/down. If x,y given, move mouse there first (required for browser)."""
        if x is not None and y is not None:
            px, py = self._grid_to_px(x, y)
            self._xdo("mousemove", "--sync", str(px), str(py))
            time.sleep(0.05)
        btn = "4" if direction == "up" else "5"
        for _ in range(clicks):
            self._xdo("click", btn)
            time.sleep(0.04)
        pos = f" at ({x},{y})" if x is not None else ""
        return f"Scrolled {direction}{pos} ({clicks} clicks)"

    def drag(self, gx1, gy1, gx2, gy2):
        px1, py1 = self._grid_to_px(gx1, gy1)
        px2, py2 = self._grid_to_px(gx2, gy2)
        self._xdo("mousemove", "--sync", str(px1), str(py1))
        self._xdo("mousedown", "1")
        time.sleep(0.05)
        self._xdo("mousemove", "--sync", str(px2), str(py2))
        self._xdo("mouseup", "1")
        return f"Dragged ({gx1},{gy1})→({gx2},{gy2}) pixel ({px1},{py1})→({px2},{py2})"

    def get_screen_size(self):
        """Query actual display geometry via xdotool."""
        r = self._xdo("getdisplaygeometry")
        if r.returncode == 0:
            parts = r.stdout.strip().split()
            if len(parts) == 2:
                try:
                    return int(parts[0]), int(parts[1])
                except ValueError:
                    pass
        return self.screen_w, self.screen_h
