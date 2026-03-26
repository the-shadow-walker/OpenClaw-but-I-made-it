"""Auto-generated: adds all swarm subdirs to sys.path so flat imports work."""
import sys, os
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in [_ROOT] + [os.path.join(_ROOT, d) for d in ('core', 'math', 'server', 'engineer')]:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
