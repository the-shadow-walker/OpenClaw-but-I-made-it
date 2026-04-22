"""
gui_atspi.py — AT-SPI accessibility tree source for Set-of-Marks element detection.

Provides interactive element positions for desktop apps, KDE widgets, and
window chrome that Chrome DevTools Protocol cannot see.

Graceful fallback: if python-gobject is not installed or the AT-SPI2 daemon is
not running, extract() returns [] without raising.
"""

try:
    import gi
    gi.require_version("Atspi", "2.0")
    from gi.repository import Atspi
    _ATSPI_OK = True
except (ImportError, ValueError, Exception):
    _ATSPI_OK = False
    Atspi = None


class ATSPIExtractor:
    """Extract interactive elements from the AT-SPI2 accessibility tree."""

    # Roles we consider "interactive" (worth marking for the model)
    _INTERACTIVE_ROLES = None  # populated lazily after import check

    @classmethod
    def _get_interactive_roles(cls):
        if not _ATSPI_OK:
            return set()
        if cls._INTERACTIVE_ROLES is None:
            cls._INTERACTIVE_ROLES = {
                Atspi.Role.PUSH_BUTTON,
                Atspi.Role.TOGGLE_BUTTON,
                Atspi.Role.CHECK_BOX,
                Atspi.Role.RADIO_BUTTON,
                Atspi.Role.TEXT,
                Atspi.Role.PASSWORD_TEXT,
                Atspi.Role.COMBO_BOX,
                Atspi.Role.LIST_ITEM,
                Atspi.Role.MENU_ITEM,
                Atspi.Role.LINK,
                Atspi.Role.ENTRY,
                Atspi.Role.SPIN_BUTTON,
                Atspi.Role.SLIDER,
                Atspi.Role.TAB,
            }
        return cls._INTERACTIVE_ROLES

    def extract(self) -> list:
        """
        Traverse AT-SPI tree and return interactive elements.

        Returns list of dicts: [{tag, text, x_px, y_px, w_px, h_px}]
        Returns [] silently if AT-SPI is unavailable or the daemon isn't running.
        """
        if not _ATSPI_OK:
            return []

        results = []
        try:
            Atspi.init()
            desktop = Atspi.get_desktop(0)
            interactive_roles = self._get_interactive_roles()
            self._traverse(desktop, interactive_roles, results, depth=0, max_depth=12)
        except Exception:
            # Daemon not running, display not accessible, or other AT-SPI error
            return []

        return results

    def _traverse(self, node, interactive_roles, results, depth, max_depth):
        """Recursively walk the AT-SPI tree, collecting interactive leaf nodes."""
        if depth > max_depth:
            return
        try:
            role = node.get_role()
        except Exception:
            return

        # Collect this node if it's an interactive role
        if role in interactive_roles:
            element = self._extract_element(node, role)
            if element:
                results.append(element)

        # Recurse into children
        try:
            n_children = node.get_child_count()
        except Exception:
            return

        for i in range(n_children):
            try:
                child = node.get_child_at_index(i)
                if child is not None:
                    self._traverse(child, interactive_roles, results, depth + 1, max_depth)
            except Exception:
                continue

    def _extract_element(self, node, role) -> dict | None:
        """Extract position + label from a single AT-SPI node. Returns None if invalid."""
        try:
            component = node.get_component()
            if component is None:
                return None
            extents = component.get_extents(Atspi.CoordType.SCREEN)
            x, y, w, h = extents.x, extents.y, extents.width, extents.height

            # Skip off-screen or zero-size elements
            if w <= 0 or h <= 0 or x < -100 or y < -100:
                return None

            # Get element name / label
            try:
                name = node.get_name() or ""
            except Exception:
                name = ""

            # Map Atspi.Role to a human-readable tag
            role_name = self._role_to_tag(role)

            return {
                "tag":   role_name,
                "text":  name.strip(),
                "x_px":  float(x + w / 2),   # center
                "y_px":  float(y + h / 2),
                "w_px":  float(w),
                "h_px":  float(h),
            }
        except Exception:
            return None

    @staticmethod
    def _role_to_tag(role) -> str:
        """Convert Atspi.Role enum to a short human-readable string."""
        if not _ATSPI_OK:
            return "role:unknown"
        _map = {
            Atspi.Role.PUSH_BUTTON:   "role:pushbutton",
            Atspi.Role.TOGGLE_BUTTON: "role:togglebutton",
            Atspi.Role.CHECK_BOX:     "role:checkbox",
            Atspi.Role.RADIO_BUTTON:  "role:radiobutton",
            Atspi.Role.TEXT:          "role:text",
            Atspi.Role.PASSWORD_TEXT: "role:password",
            Atspi.Role.COMBO_BOX:     "role:combobox",
            Atspi.Role.LIST_ITEM:     "role:listitem",
            Atspi.Role.MENU_ITEM:     "role:menuitem",
            Atspi.Role.LINK:          "role:link",
            Atspi.Role.ENTRY:         "role:entry",
            Atspi.Role.SPIN_BUTTON:   "role:spinbutton",
            Atspi.Role.SLIDER:        "role:slider",
            Atspi.Role.TAB:           "role:tab",
        }
        return _map.get(role, "role:unknown")


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _ATSPI_OK:
        print("AT-SPI not available (gi.repository.Atspi not installed).")
        print("Install with: pip install python-gobject")
        print("System package: at-spi2-core (Arch: sudo pacman -S at-spi2-core)")
    else:
        print("AT-SPI available. Extracting elements...")
        extractor = ATSPIExtractor()
        elements = extractor.extract()
        print(f"Found {len(elements)} interactive elements:")
        for e in elements[:20]:
            print(f"  {e['tag']:<22} {repr(e['text']):<30} @ ({e['x_px']:.0f}, {e['y_px']:.0f})")
        if len(elements) > 20:
            print(f"  ... and {len(elements) - 20} more")
