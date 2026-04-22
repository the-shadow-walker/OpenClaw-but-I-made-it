"""
gui_elements.py — UIElement dataclass + ElementRegistry (merge, ID assignment).

Part of the Set-of-Marks (SoM) upgrade: every detected element gets a numbered
marker drawn on the screenshot image; the model clicks {"id": N} instead of
estimating float coordinates.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UIElement:
    id: int = 0                # assigned by registry (1-based)
    tag: str = ""              # "button", "input", "a", "role:pushbutton", "cv:rect"
    text: str = ""             # label / aria-name / OCR text
    x_px: float = 0.0         # center x in full-screen pixels
    y_px: float = 0.0         # center y in full-screen pixels
    w_px: float = 0.0         # bounding box width
    h_px: float = 0.0         # bounding box height
    source: str = "dom"       # "atspi" | "dom" | "cv"
    grid_x: float = 0.0       # pre-computed 16×16 grid coord
    grid_y: float = 0.0


class ElementRegistry:
    SOURCE_PRIORITY = {"atspi": 0, "dom": 1, "cv": 2}

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.elements: List[UIElement] = []

    def merge_elements(self, candidates: List[dict]) -> None:
        """
        Merge candidate dicts into self.elements using center-point containment dedup.

        Each candidate: {tag, text, x_px, y_px, w_px, h_px, source}

        Dedup rule: if candidate B's center (x_px, y_px) falls inside an existing
        element A's bounding box (A.x_px ± A.w_px/2, A.y_px ± A.h_px/2), treat
        as the same element. Keep the higher-priority source (atspi > dom > cv).
        If same priority, keep whichever has more text.

        After all merges, sort top-to-bottom then left-to-right (reading order),
        assign IDs 1..N, compute grid_x/grid_y.
        """
        self.elements = []

        for cand in candidates:
            x_px  = float(cand.get("x_px", 0))
            y_px  = float(cand.get("y_px", 0))
            w_px  = float(cand.get("w_px", 0))
            h_px  = float(cand.get("h_px", 0))
            tag   = str(cand.get("tag", ""))
            text  = str(cand.get("text", ""))
            source = str(cand.get("source", "dom"))

            # Skip degenerate elements (no position)
            if x_px == 0 and y_px == 0 and w_px == 0 and h_px == 0:
                continue

            # Check if this candidate's center falls inside any existing element's bbox
            matched = None
            for existing in self.elements:
                half_w = max(existing.w_px / 2, 10)  # min 10px tolerance
                half_h = max(existing.h_px / 2, 10)
                if (abs(x_px - existing.x_px) <= half_w and
                        abs(y_px - existing.y_px) <= half_h):
                    matched = existing
                    break

            if matched is None:
                # New element — add it
                self.elements.append(UIElement(
                    id=0, tag=tag, text=text,
                    x_px=x_px, y_px=y_px, w_px=w_px, h_px=h_px,
                    source=source,
                ))
            else:
                # Duplicate — keep higher-priority source
                existing_prio = self.SOURCE_PRIORITY.get(matched.source, 99)
                cand_prio     = self.SOURCE_PRIORITY.get(source, 99)
                if cand_prio < existing_prio:
                    # Replace with higher-priority candidate
                    matched.tag    = tag
                    matched.text   = text
                    matched.x_px   = x_px
                    matched.y_px   = y_px
                    matched.w_px   = w_px
                    matched.h_px   = h_px
                    matched.source = source
                elif cand_prio == existing_prio and len(text) > len(matched.text):
                    # Same priority — prefer richer text label
                    matched.text = text

        self._assign_ids()

    def _assign_ids(self) -> None:
        """Sort by (y_px // row_band, x_px) then assign 1-based IDs and grid coords."""
        ROW_BAND = 40  # px — group elements in same horizontal band
        self.elements.sort(key=lambda e: (int(e.y_px) // ROW_BAND, e.x_px))
        for i, el in enumerate(self.elements, 1):
            el.id = i
            el.grid_x = round(el.x_px / self.screen_w * 16, 2)
            el.grid_y = round(el.y_px / self.screen_h * 16, 2)

    def get_by_id(self, eid: int) -> Optional[UIElement]:
        for el in self.elements:
            if el.id == eid:
                return el
        return None

    def format_for_prompt(self) -> str:
        """
        Format as text list for screenshot observation.

        Example lines:
          [5]  button "Login"           @ (7.68, 3.27)  [DOM]
          [6]  input  "Email"           @ (8.00, 7.10)  [DOM]
          [12] role:pushbutton "OK"     @ (9.10, 11.20) [AT-SPI]
          [23] cv:rect ""               @ (4.50, 2.10)  [CV]
        """
        if not self.elements:
            return "  (no elements detected)"
        source_label = {"atspi": "AT-SPI", "dom": "DOM", "cv": "CV"}
        lines = []
        for el in self.elements:
            src = source_label.get(el.source, el.source.upper())
            text_display = f'"{el.text[:40]}"' if el.text else '""'
            lines.append(
                f"  [{el.id}]  {el.tag:<20} {text_display:<44}"
                f" @ ({el.grid_x:.2f}, {el.grid_y:.2f})  [{src}]"
            )
        return "\n".join(lines)


# ── Minimal unit test (run directly to verify merge logic) ─────────────────

if __name__ == "__main__":
    reg = ElementRegistry(screen_w=1920, screen_h=1080)

    candidates = [
        # Two DOM elements
        {"tag": "button", "text": "Login",    "x_px": 960, "y_px": 400,
         "w_px": 120, "h_px": 40, "source": "dom"},
        {"tag": "input",  "text": "Username", "x_px": 960, "y_px": 300,
         "w_px": 200, "h_px": 36, "source": "dom"},
        # AT-SPI duplicate of the button (higher priority, should replace)
        {"tag": "role:pushbutton", "text": "Login Button", "x_px": 965, "y_px": 402,
         "w_px": 120, "h_px": 40, "source": "atspi"},
        # CV gap-filler (unique area)
        {"tag": "cv:rect", "text": "", "x_px": 200, "y_px": 100,
         "w_px": 80, "h_px": 30, "source": "cv"},
    ]

    reg.merge_elements(candidates)

    print(f"Elements after merge: {len(reg.elements)} (expected 3)")
    print()
    print(reg.format_for_prompt())
    print()

    el = reg.get_by_id(1)
    print(f"get_by_id(1): {el}")
    assert el is not None, "ID 1 should exist"

    # The button duplicate should have been merged into atspi source
    button_els = [e for e in reg.elements if "button" in e.tag.lower() or "Login" in e.text]
    assert any(e.source == "atspi" for e in button_els), \
        f"Expected atspi to win dedup, got {[e.source for e in button_els]}"
    print("All assertions passed.")
