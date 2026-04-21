"""
gui_dom.py — DOM element extraction via Chrome DevTools Protocol (CDP).

Connects to a running Chromium/Brave instance on the debug port and
queries all interactive elements with exact bounding-rect positions.
Falls back silently to [] if no browser is running or CDP is unavailable.

Usage:
    dom = DOMExtractor(screen_w=1920, screen_h=1080)
    elements = dom.extract()   # [{tag, text, x_px, y_px, grid_x, grid_y}]
    print(dom.build_element_map(elements))
"""

import json
import urllib.request

try:
    import websocket
    _WS_OK = True
except ImportError:
    _WS_OK = False

CDP_PORT = 9222

# All interactive element types — covers buttons, links, inputs, ARIA roles
# Returns viewport-relative coords; Python adds the chrome offset (measured via CDP).
_DOM_SCRIPT = """
(function() {
    var sel = [
        'a[href]', 'button', 'input', 'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="tab"]',
        '[role="menuitem"]', '[role="option"]', '[role="checkbox"]',
        '[role="radio"]', '[role="switch"]', '[role="textbox"]',
        '[onclick]', 'label[for]', 'summary'
    ].join(', ');

    return Array.from(document.querySelectorAll(sel))
        .map(function(el) {
            var r = el.getBoundingClientRect();
            if (r.width < 2 || r.height < 2) return null;
            if (r.top < -r.height || r.left < -r.width) return null;
            var text = (
                el.innerText ||
                el.value ||
                el.getAttribute('aria-label') ||
                el.getAttribute('placeholder') ||
                el.getAttribute('title') ||
                el.getAttribute('alt') ||
                ''
            ).replace(/\\s+/g, ' ').trim().slice(0, 80);
            if (!text) return null;
            return {
                tag:  el.tagName.toLowerCase(),
                type: el.getAttribute('type') || '',
                text: text,
                x: Math.round(r.left + r.width  / 2),
                y: Math.round(r.top  + r.height / 2),
                w: Math.round(r.width),
                h: Math.round(r.height)
            };
        })
        .filter(function(e) { return e !== null; });
})()
"""


class DOMExtractor:
    def __init__(self, port: int = CDP_PORT, screen_w: int = 1920,
                 screen_h: int = 1080):
        self.port = port
        self.screen_w = screen_w
        self.screen_h = screen_h

    def _get_ws_url(self) -> str | None:
        """Return WebSocket debugger URL for the frontmost page tab."""
        try:
            url = f"http://localhost:{self.port}/json"
            req = urllib.request.urlopen(url, timeout=2)
            tabs = json.loads(req.read())
            for tab in tabs:
                if tab.get("type") == "page" and "webSocketDebuggerUrl" in tab:
                    return tab["webSocketDebuggerUrl"]
        except Exception:
            pass
        return None

    def _cdp(self, ws, id: int, method: str, params: dict = {}) -> dict:
        ws.send(json.dumps({"id": id, "method": method, "params": params}))
        return json.loads(ws.recv()).get("result", {})

    def extract(self, timeout: int = 5) -> list[dict]:
        """Extract all visible interactive elements from the current page.

        Returns a list of dicts:
            {tag, text, x_px, y_px, grid_x, grid_y}
        Coordinates are absolute screen pixels (chrome offset accounted for).
        Returns [] silently if no browser is reachable.
        """
        if not _WS_OK:
            return []
        ws_url = self._get_ws_url()
        if not ws_url:
            return []
        try:
            ws = websocket.create_connection(ws_url, timeout=timeout)

            # Measure chrome offset via CDP — window.outerHeight - window.innerHeight
            # is unreliable on Linux Brave; use actual layout metrics instead.
            try:
                win     = self._cdp(ws, 1, "Browser.getWindowForTarget")
                metrics = self._cdp(ws, 2, "Page.getLayoutMetrics")
                bounds   = win.get("bounds", {})
                viewport = metrics.get("visualViewport", {})
                win_x  = bounds.get("left", 0)
                win_y  = bounds.get("top",  0)
                win_h  = bounds.get("height", self.screen_h)
                win_w  = bounds.get("width",  self.screen_w)
                vp_h   = viewport.get("clientHeight", win_h)
                vp_w   = viewport.get("clientWidth",  win_w)
                off_y  = win_y + (win_h - vp_h)   # chrome height
                off_x  = win_x + max(0, (win_w - vp_w) // 2)
            except Exception:
                off_x, off_y = 0, 0

            # Query DOM elements (viewport-relative coords)
            result = self._cdp(ws, 3, "Runtime.evaluate", {
                "expression":    _DOM_SCRIPT,
                "returnByValue": True,
                "awaitPromise":  False,
            })
            ws.close()
        except Exception:
            return []

        try:
            raw_els = result.get("result", {}).get("value", [])
            if not isinstance(raw_els, list):
                return []
        except (KeyError, TypeError):
            return []

        out = []
        for e in raw_els:
            try:
                # Convert viewport-relative → absolute screen pixels
                sx = e["x"] + off_x
                sy = e["y"] + off_y
                out.append({
                    "tag":    e["tag"],
                    "text":   e["text"],
                    "x_px":   sx,
                    "y_px":   sy,
                    "grid_x": round(sx / self.screen_w * 16, 2),
                    "grid_y": round(sy / self.screen_h * 16, 2),
                })
            except (KeyError, TypeError, ZeroDivisionError):
                continue
        return out

    def build_element_map(self, elements: list[dict],
                          cap: int = 60) -> str:
        """Format DOM elements as a labelled text block for the model prompt."""
        if not elements:
            return ""
        lines = []
        for e in elements[:cap]:
            tag  = e["tag"]
            text = e["text"]
            gx   = e["grid_x"]
            gy   = e["grid_y"]
            lines.append(f'  {tag:8s} "{text:<45s}" @ grid ({gx:.2f}, {gy:.2f})')
        return "\n".join(lines)
