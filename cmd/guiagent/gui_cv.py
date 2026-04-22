"""
gui_cv.py — OpenCV contour-based gap-filler for Set-of-Marks element detection.

Finds rectangular clickable-looking regions (buttons, input fields, cards)
in the screenshot image when neither AT-SPI nor DOM can see them.

Conservative by design: high solidity requirement (0.85) means only clean
rectangular UI elements make it through, not random image noise.

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
    """Gap-filler: find clickable rectangular regions via edge detection."""

    # Contour filter parameters
    MIN_AREA   = 400      # px² — ignore tiny noise
    MAX_AREA   = 60_000   # px² — ignore whole-screen regions
    MIN_SOLIDITY    = 0.85   # contour area / convex-hull area (high = clean rectangle)
    MIN_ASPECT = 0.2      # w/h — reject very tall slivers
    MAX_ASPECT = 8.0      # w/h — reject very wide banners

    def extract(self, img, existing_elements: list = None) -> list:
        """
        Find rectangular UI regions not already covered by DOM/AT-SPI.

        Args:
            img: PIL.Image (RGB) — the raw screenshot (not grid-overlaid)
            existing_elements: list of UIElement or dicts with x_px,y_px,w_px,h_px
                               used to reject contours whose center is already covered.

        Returns list of dicts: [{tag, text, x_px, y_px, w_px, h_px}]
        Returns [] silently if cv2 not installed.
        """
        if not _CV_OK:
            return []

        try:
            return self._run(img, existing_elements or [])
        except Exception:
            return []

    def _run(self, img, existing_elements):
        # PIL → numpy BGR
        import numpy as np  # local import so outer-scope np=None doesn't break module load
        rgb = np.array(img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Scharr edge detection (better than Canny for low-contrast UI elements)
        scharr_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        magnitude = cv2.magnitude(scharr_x, scharr_y)
        edge = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold + morphological close to connect nearby edges
        _, thresh = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_h, img_w = gray.shape
        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_AREA or area > self.MAX_AREA:
                continue

            # Solidity check — hull area vs contour area
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < self.MIN_SOLIDITY:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if aspect < self.MIN_ASPECT or aspect > self.MAX_ASPECT:
                continue

            cx = x + w / 2
            cy = y + h / 2

            # Skip if center already covered by an existing element
            if self._is_covered(cx, cy, existing_elements):
                continue

            results.append({
                "tag":  "cv:rect",
                "text": "",
                "x_px": float(cx),
                "y_px": float(cy),
                "w_px": float(w),
                "h_px": float(h),
            })

        return results

    @staticmethod
    def _is_covered(cx, cy, existing_elements) -> bool:
        """Return True if (cx, cy) falls inside any existing element's bounding box."""
        for el in existing_elements:
            # Support both UIElement dataclass and plain dicts
            if hasattr(el, "x_px"):
                ex, ey, ew, eh = el.x_px, el.y_px, el.w_px, el.h_px
            else:
                ex = float(el.get("x_px", 0))
                ey = float(el.get("y_px", 0))
                ew = float(el.get("w_px", 0))
                eh = float(el.get("h_px", 0))
            half_w = max(ew / 2, 5)
            half_h = max(eh / 2, 5)
            if abs(cx - ex) <= half_w and abs(cy - ey) <= half_h:
                return True
        return False


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    if not _CV_OK:
        print("OpenCV not available.")
        print("Install with: pip install opencv-python")
        sys.exit(0)

    # Try to load a test image if provided, otherwise create a synthetic one
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        from PIL import Image
        img = Image.open(sys.argv[1]).convert("RGB")
        print(f"Testing on {sys.argv[1]} ({img.size})")
    else:
        # Synthetic test: white image with two gray rectangles
        try:
            from PIL import Image, ImageDraw
            img = Image.new("RGB", (800, 600), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            # Button 1
            draw.rectangle([100, 200, 300, 240], outline=(100, 100, 100), width=2)
            # Button 2
            draw.rectangle([100, 260, 500, 310], outline=(100, 100, 100), width=2)
            print("Testing on synthetic 800×600 image with 2 rectangles")
        except ImportError:
            print("Pillow not installed — cannot create test image")
            sys.exit(1)

    extractor = CVExtractor()
    elements = extractor.extract(img)
    print(f"Found {len(elements)} CV elements:")
    for e in elements:
        print(f"  {e['tag']} @ ({e['x_px']:.0f}, {e['y_px']:.0f})  {e['w_px']:.0f}×{e['h_px']:.0f}px")
