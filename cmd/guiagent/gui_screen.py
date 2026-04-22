"""
gui_screen.py — screenshot capture, 16×16 grid overlay, OCR, base64 encoding.
"""

import base64
import io
import os
import subprocess

try:
    from PIL import Image, ImageDraw
    _PIL_OK = True
except ImportError:
    _PIL_OK = False
    Image = ImageDraw = None

try:
    import pytesseract
    _TESS_OK = True
except ImportError:
    _TESS_OK = False
    pytesseract = None


class GUIScreen:
    def __init__(self, display=":99", screen_w=1280, screen_h=720):
        self.display = display
        self.screen_w = screen_w
        self.screen_h = screen_h

    def capture(self):
        """Capture screenshot via ImageMagick import. Returns PIL Image."""
        if not _PIL_OK:
            raise RuntimeError("Pillow not installed — pip install Pillow")
        path = "/tmp/gui_shot.png"
        env = {**os.environ, "DISPLAY": self.display}
        r = subprocess.run(
            ["import", "-window", "root", path],
            env=env, capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Screenshot failed: {r.stderr.decode()[:200]}")
        return Image.open(path).convert("RGB")

    def overlay_grid(self, img, n=16, cursor=None):
        """Draw labeled n×n grid (and optional cursor dot) on a copy of img.

        cursor: (px, py) pixel coords of last click — draws a red dot + crosshair.
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        cell_w = w / n
        cell_h = h / n

        line_color = (220, 50, 50)

        # Grid lines
        for i in range(1, n):
            x = int(i * cell_w)
            y = int(i * cell_h)
            draw.line([(x, 0), (x, h)], fill=line_color, width=1)
            draw.line([(0, y), (w, y)], fill=line_color, width=1)

        # Column labels along top (0..n-1)
        for i in range(n):
            x = int((i + 0.5) * cell_w)
            draw.text((x - 4, 2), str(i), fill=line_color)

        # Row labels along left (0..n-1)
        for i in range(n):
            y = int((i + 0.5) * cell_h)
            draw.text((2, y - 6), str(i), fill=line_color)

        # Cursor: red filled circle + white outline + crosshair at last click
        if cursor:
            cx, cy = int(cursor[0]), int(cursor[1])
            r = max(8, int(min(w, h) / 80))   # ~12px at 1920×1080
            # White halo so dot is visible on any background
            draw.ellipse([cx-r-2, cy-r-2, cx+r+2, cy+r+2],
                         fill=(255, 255, 255))
            draw.ellipse([cx-r, cy-r, cx+r, cy+r],
                         fill=(255, 0, 0))
            # Crosshair lines
            arm = r * 3
            draw.line([(cx - arm, cy), (cx + arm, cy)],
                      fill=(255, 0, 0), width=2)
            draw.line([(cx, cy - arm), (cx, cy + arm)],
                      fill=(255, 0, 0), width=2)

        return img

    def ocr_elements(self, img):
        """Run tesseract OCR. Returns list of dicts: {text, cx, cy, grid_x, grid_y}."""
        if not _TESS_OK:
            return []
        try:
            data = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT
            )
        except Exception:
            return []

        w, h = img.size
        elements = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            try:
                conf = int(data["conf"][i])
            except (ValueError, TypeError):
                continue
            if conf < 15:
                continue
            # Skip single-character noise (toolbar icons, punctuation artifacts)
            if len(text) < 2:
                continue

            x = data["left"][i]
            y = data["top"][i]
            bw = data["width"][i]
            bh = data["height"][i]
            cx = x + bw / 2
            cy = y + bh / 2
            grid_x = round(cx / w * 16, 2)
            grid_y = round(cy / h * 16, 2)
            elements.append({
                "text": text,
                "cx": cx,
                "cy": cy,
                "grid_x": grid_x,
                "grid_y": grid_y,
            })
        return elements

    def build_text_map(self, elements):
        """Format OCR elements as a text block for the model prompt."""
        if not elements:
            return "  (no text detected by OCR)"
        lines = []
        for e in elements[:60]:  # cap at 60 elements to keep prompt concise
            lines.append(f'  "{e["text"]}"  @ grid {e["grid_x"]}, {e["grid_y"]}')
        return "\n".join(lines)

    def render_markers(self, img, registry, n_grid=16):
        """Draw colored bordered rectangles + outlined ID label boxes on a copy of img.

        Color by source:
          DOM    → Blue   (0, 120, 255)
          AT-SPI → Green  (0, 200, 80)
          CV     → Orange (255, 140, 0)

        For each element:
          1. Draw 1px colored border around its bounding box
          2. Draw "[ID]" label with: colored fill, white text, 1px black outline
             so it pops on both light and dark UI themes.

        Collision resolution — tries candidate positions in order until clear:
          1. Top-left of box (default)
          2. Nudge down up to 5× by label_h+2px
          3. Top-right corner of box
          4. Bottom-left corner of box
          Falls back to last tried position if all 8 attempts fail.

        The 16×16 grid overlay is applied on top by the caller (overlay_grid).
        """
        if not _PIL_OK:
            return img

        SOURCE_COLORS = {
            "dom":   (0, 120, 255),
            "atspi": (0, 200, 80),
            "cv":    (255, 140, 0),
        }
        DEFAULT_COLOR = (180, 180, 180)

        img = img.copy()
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        placed_labels = []  # list of (x1, y1, x2, y2) already drawn

        def rects_overlap(r1, r2):
            return not (r1[2] <= r2[0] or r1[0] >= r2[2] or
                        r1[3] <= r2[1] or r1[1] >= r2[3])

        def find_label_pos(lx0, ly0, label_w, label_h, bx1, by1, bx2, by2):
            """Return (lx, ly) for a non-colliding label position, or best fallback."""
            # Strategy 1: nudge down from initial position (up to 5 steps)
            lx, ly = lx0, ly0
            for _ in range(6):
                r = (lx, ly, lx + label_w, ly + label_h)
                if not any(rects_overlap(r, p) for p in placed_labels):
                    return lx, ly
                ly += label_h + 2

            # Strategy 2: top-right corner of bounding box
            lx, ly = bx2 - label_w, by1 - label_h - 1
            r = (lx, ly, lx + label_w, ly + label_h)
            if not any(rects_overlap(r, p) for p in placed_labels):
                return lx, ly

            # Strategy 3: bottom-left corner of bounding box
            lx, ly = bx1, by2 + 1
            r = (lx, ly, lx + label_w, ly + label_h)
            if not any(rects_overlap(r, p) for p in placed_labels):
                return lx, ly

            # Fallback: original top-left (accept overlap rather than no label)
            return lx0, ly0

        for el in registry.elements:
            color = SOURCE_COLORS.get(el.source, DEFAULT_COLOR)
            cx, cy = el.x_px, el.y_px
            hw = max(el.w_px / 2, 4)
            hh = max(el.h_px / 2, 4)

            bx1 = max(0,      int(cx - hw))
            by1 = max(0,      int(cy - hh))
            bx2 = min(iw - 1, int(cx + hw))
            by2 = min(ih - 1, int(cy + hh))

            # 1px colored border around bounding box
            draw.rectangle([bx1, by1, bx2, by2], outline=color)

            # Label geometry
            label   = f"[{el.id}]"
            label_w = len(label) * 7 + 4   # ~7px per char + padding
            label_h = 14

            # Initial candidate: top-left, just above the box
            init_lx = bx1
            init_ly = max(0, by1 - label_h - 1)

            lx, ly = find_label_pos(init_lx, init_ly, label_w, label_h,
                                     bx1, by1, bx2, by2)

            # Clamp to image
            lx = max(0, min(lx, iw - label_w - 1))
            ly = max(0, min(ly, ih - label_h - 1))

            # Draw label: 1px black outline + colored fill + white text
            draw.rectangle([lx - 1, ly - 1, lx + label_w + 1, ly + label_h + 1],
                           fill=(0, 0, 0))                   # black outline
            draw.rectangle([lx, ly, lx + label_w, ly + label_h],
                           fill=color)                        # source-colored fill
            draw.text((lx + 2, ly + 1), label, fill=(255, 255, 255))  # white text

            placed_labels.append((lx - 1, ly - 1, lx + label_w + 1, ly + label_h + 1))

        return img

    def draw_zoom_overlay(self, img, x_min, y_min, x_max, y_max):
        """Draw a thick red rectangle + label on img showing the zoomed region.

        Returns a copy of img with the overlay applied.
        Used to give the model spatial context when a zoom is active.
        """
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # Thick red rectangle border
        border = max(3, int(min(w, h) / 200))
        for i in range(border):
            draw.rectangle(
                [x_min - i, y_min - i, x_max + i, y_max + i],
                outline=(255, 0, 0),
            )

        # Semi-transparent red fill (draw a thin overlay by drawing many small lines)
        # Simple approach: just a bright corner marker and "ZOOMED" label
        arm = max(12, int(min(w, h) / 60))
        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_min, y_max), (x_max, y_max),
        ]
        for cx, cy in corners:
            # Horizontal arm
            x0 = cx - arm if cx == x_max else cx
            x1 = cx + arm if cx == x_min else cx
            draw.line([(x0, cy), (x1, cy)], fill=(255, 255, 0), width=border + 1)
            # Vertical arm
            y0 = cy - arm if cy == y_max else cy
            y1 = cy + arm if cy == y_min else cy
            draw.line([(cx, y0), (cx, y1)], fill=(255, 255, 0), width=border + 1)

        # "ZOOMED" label just above the box
        label = "ZOOMED VIEW"
        lx = max(2, x_min)
        ly = max(0, y_min - 14)
        draw.rectangle([lx - 2, ly - 2, lx + 90, ly + 12], fill=(200, 0, 0))
        draw.text((lx, ly), label, fill=(255, 255, 255))

        return img

    def compose_zoom_panel(self, full_with_box, zoomed_grid, max_w=1280):
        """Compose a single side-by-side image: [FULL SCREEN | ZOOMED VIEW].

        Returns one PIL Image with labeled panels so the model always sees
        both the spatial context and the precision detail in a single frame.
        """
        divider = 4
        label_h = 18
        left_w  = int(max_w * 0.62)
        right_w = max_w - left_w - divider

        fw, fh = full_with_box.size
        left  = full_with_box.resize((left_w,  int(fh * left_w  / fw)),  Image.LANCZOS)
        zw, zh = zoomed_grid.size
        right = zoomed_grid.resize((right_w, int(zh * right_w / zw)), Image.LANCZOS)

        total_h = label_h + max(left.height, right.height)
        canvas  = Image.new("RGB", (max_w, total_h), (20, 20, 20))
        draw    = ImageDraw.Draw(canvas)

        # Label bar — left
        draw.rectangle([0, 0, left_w - 1, label_h - 1], fill=(30, 30, 70))
        draw.text((4, 3), "IMAGE 1 — FULL SCREEN  (red box = zoomed region)", fill=(180, 180, 255))
        # Label bar — right
        draw.rectangle([left_w + divider, 0, max_w - 1, label_h - 1], fill=(30, 60, 30))
        draw.text((left_w + divider + 4, 3), "IMAGE 2 — ZOOMED VIEW  (use these coords to click)", fill=(180, 255, 180))

        canvas.paste(left,  (0,            label_h))
        canvas.paste(right, (left_w + divider, label_h))
        # Divider line
        draw.rectangle([left_w, 0, left_w + divider - 1, total_h], fill=(220, 50, 50))

        return canvas

    def to_base64(self, img):
        """Encode PIL Image as base64 PNG string (full resolution)."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def to_base64_model(self, img, max_w=960):
        """Encode PIL Image as base64 PNG, downscaled for model inference.

        Shrinks to max_w wide (preserving aspect ratio) before encoding.
        A 1920×1080 image goes from ~2MB base64 to ~300KB — 6× faster inference.
        The browser UI still gets the full-res image via to_base64().
        """
        w, h = img.size
        if w > max_w:
            scale = max_w / w
            img = img.resize((max_w, int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
