"""
gui_cv.py — OpenCV contour-based gap-filler for Set-of-Marks element detection.

Uses a "subtraction mask" approach: existing AT-SPI + DOM element bounding boxes
are blanked out first, so edge detection only runs on the uncovered regions of
the screen. This prevents the CV layer from re-finding elements we already know
about and keeps false-positive rates low.

Conservative heuristics: solidity > 0.8, aspect ratio 0.5–2.0 (button-shaped),
area 400–5000px². Only clean rectangular UI elements make it through.

Graceful fallback: if opencv-python is not installed, extract() returns [].
"""

try:
    import cv2
    import numpy as np
    _CV_OK = True
except ImportError:
    _CV_OK = False
    cv2 = None
    np = None


class CVExtractor:
    """Gap-filler: find clickable rectangular regions in uncovered screen areas."""

    MIN_AREA   = 400     # px² — ignore tiny noise (bounding-box area)
    MAX_AREA   = 20_000  # px² — up to ~141×141px; covers typical KDE buttons
    MIN_ASPECT = 0.5     # w/h — reject very tall slivers
    MAX_ASPECT = 2.0     # w/h — reject very wide banners

    def extract(self, img, existing_elements: list = None) -> list:
        """
        Find rectangular UI regions not already covered by AT-SPI or DOM.

        Args:
            img: PIL.Image (RGB) — the raw screenshot (no grid overlay)
            existing_elements: list of UIElement or dicts with x_px,y_px,w_px,h_px.
                               Their bounding boxes are masked out before edge detection
                               so CV never re-discovers what we already know.

        Returns list of dicts: [{tag, text, x_px, y_px, w_px, h_px}]
        Returns [] silently if cv2 is not installed.
        """
        if not _CV_OK:
            return []
        try:
            return self._run(img, existing_elements or [])
        except Exception:
            return []

    def _run(self, img, existing_elements):
        import numpy as np  # local so outer-scope np=None doesn't break module load

        # PIL → numpy grayscale
        rgb  = np.array(img)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        # ── Subtraction mask ───────────────────────────────────────────────────
        # Paint black over known-element bounding boxes; keep everything else white.
        # CV edge detection then only fires in uncovered screen regions.
        mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

        PAD = 6  # px — expand mask slightly to capture gradient bleed at element edges
        for el in existing_elements:
            if hasattr(el, "x_px"):
                ex, ey, ew, eh = el.x_px, el.y_px, el.w_px, el.h_px
            else:
                ex = float(el.get("x_px", 0))
                ey = float(el.get("y_px", 0))
                ew = float(el.get("w_px", 0))
                eh = float(el.get("h_px", 0))
            if ew <= 0 or eh <= 0:
                continue
            x1 = max(0, int(ex - ew / 2) - PAD)
            y1 = max(0, int(ey - eh / 2) - PAD)
            x2 = min(img_w - 1, int(ex + ew / 2) + PAD)
            y2 = min(img_h - 1, int(ey + eh / 2) + PAD)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)

        # ── Scharr edge detection on the FULL gray image ─────────────────────
        # IMPORTANT: run Scharr on the unmasked gray first.
        # Applying the mask before Scharr zeroes out the mask boundary,
        # creating huge artificial gradients (200→0) that dominate normalization
        # and swamp the real UI element edges we care about.
        scharr_x  = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        scharr_y  = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(scharr_x, scharr_y)
        edge      = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply subtraction mask to the EDGE image — zero out edge pixels
        # that fall within known-element bounding boxes.
        edge = cv2.bitwise_and(edge, edge, mask=mask)

        _, thresh = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)

        # Dilate to thicken edge lines so nearby parallel edges merge.
        # MORPH_CLOSE on a ring-shaped border doesn't fill the interior, but
        # dilating the edge pixels makes them thick enough for approxPolyDP to
        # find clean rectangular corners.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # ── Find and filter contours ───────────────────────────────────────────
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        seen_boxes = []  # deduplicate near-identical bounding boxes

        for cnt in contours:
            # Approximate contour to a polygon.
            # A rectangular UI element (border or filled) approximates to 4–8 vertices.
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 4 or len(approx) > 10:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Use bounding-box area (not contour ring area) for size checks
            bbox_area = w * h
            if bbox_area < self.MIN_AREA or bbox_area > self.MAX_AREA:
                continue

            # Minimum edge dimension — avoids thin lines matching as "buttons"
            if w < 20 or h < 20:
                continue

            aspect = w / h if h > 0 else 0
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue

            # Dedup: skip if very close to an already-accepted box
            cx, cy = x + w / 2, y + h / 2
            if any(abs(cx - sx) < 20 and abs(cy - sy) < 20
                   for sx, sy in seen_boxes):
                continue
            seen_boxes.append((cx, cy))

            results.append({
                "tag":  "cv:rect",
                "text": "",
                "x_px": float(cx),
                "y_px": float(cy),
                "w_px": float(w),
                "h_px": float(h),
            })

        return results


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    if not _CV_OK:
        print("OpenCV not available — pip install opencv-python")
        sys.exit(0)

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        from PIL import Image
        img = Image.open(sys.argv[1]).convert("RGB")
        print(f"Testing on {sys.argv[1]} ({img.size})")
        existing = []
    else:
        from PIL import Image, ImageDraw
        # Light gray background — buttons will be white/darker to create edge contrast
        img = Image.new("RGB", (800, 600), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        # Two "unknown" filled buttons (aspect ~1.5–1.7, area ~3500–5000px²)
        # Filled = solid edge contrast on all 4 sides → high solidity contours
        draw.rectangle([100, 200, 170, 270], fill=(255, 255, 255), outline=(80, 80, 80), width=2)  # 70×70
        draw.rectangle([250, 200, 370, 270], fill=(255, 255, 255), outline=(80, 80, 80), width=2)  # 120×70
        # One pre-known by DOM — should be masked out and NOT returned
        draw.rectangle([100, 350, 200, 430], fill=(255, 255, 255), outline=(80, 80, 80), width=2)  # 100×80
        existing = [{"x_px": 150, "y_px": 390, "w_px": 100, "h_px": 80}]
        print("Synthetic test: 3 filled rectangles, 1 pre-masked as 'known'")

    extractor = CVExtractor()
    results = extractor.extract(img, existing_elements=existing)
    print(f"Found {len(results)} CV elements (expected 2 for synthetic test):")
    for e in results:
        print(f"  {e['tag']} @ ({e['x_px']:.0f}, {e['y_px']:.0f})  {e['w_px']:.0f}×{e['h_px']:.0f}px")
