"""Auto-generated: adds all swarm subdirs to sys.path so flat imports work."""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("", "base", "math", "engineer", "search", "agents", "server", "projects"):
    _d = os.path.join(_HERE, _sub) if _sub else _HERE
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
