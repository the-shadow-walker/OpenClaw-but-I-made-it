"""
gui_screen.py — screenshot capture, 8×8 grid overlay, OCR, base64 encoding.
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

    def overlay_grid(self, img, n=8):
        """Draw labeled n×n grid on a copy of img. Returns new PIL Image."""
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
            if conf < 30:
                continue

            x = data["left"][i]
            y = data["top"][i]
            bw = data["width"][i]
            bh = data["height"][i]
            cx = x + bw / 2
            cy = y + bh / 2
            grid_x = round(cx / w * 8, 2)
            grid_y = round(cy / h * 8, 2)
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
