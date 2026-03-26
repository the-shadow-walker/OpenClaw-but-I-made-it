import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
try:
    import _paths
except ImportError:
    for _d in [os.path.join(_ROOT, d) for d in ('core', 'compute', 'server', 'engineer')]:
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)
