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

    def _grid_to_px(self, gx, gy):
        """Convert 8×8 grid coordinates to pixel coordinates."""
        px = int((float(gx) / 8.0) * self.screen_w)
        py = int((float(gy) / 8.0) * self.screen_h)
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

    def click(self, gx, gy):
        px, py = self._grid_to_px(gx, gy)
        self._xdo("mousemove", "--sync", str(px), str(py))
        self._xdo("click", "1")
        return f"Clicked at grid ({gx}, {gy}) → pixel ({px}, {py})"

    def double_click(self, gx, gy):
        px, py = self._grid_to_px(gx, gy)
        self._xdo("mousemove", "--sync", str(px), str(py))
        self._xdo("click", "--repeat", "2", "--delay", "100", "1")
        return f"Double-clicked at grid ({gx}, {gy}) → pixel ({px}, {py})"

    def right_click(self, gx, gy):
        px, py = self._grid_to_px(gx, gy)
        self._xdo("mousemove", "--sync", str(px), str(py))
        self._xdo("click", "3")
        return f"Right-clicked at grid ({gx}, {gy}) → pixel ({px}, {py})"

    def type_text(self, text):
        self._xdo("type", "--clearmodifiers", "--", text)
        return f"Typed: {text[:60]}{'...' if len(text) > 60 else ''}"

    def key(self, combo):
        self._xdo("key", "--clearmodifiers", combo)
        return f"Key: {combo}"

    def scroll(self, direction, clicks=5):
        btn = "4" if direction == "up" else "5"
        for _ in range(clicks):
            self._xdo("click", btn)
        return f"Scrolled {direction} ({clicks} clicks)"

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
