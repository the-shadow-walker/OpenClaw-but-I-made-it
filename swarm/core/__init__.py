import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
try:
    import _paths
except ImportError:
    for _d in [os.path.join(_ROOT, d) for d in ('core', 'compute', 'server', 'engineer')]:
        if os.path.isdir(_d) and _d not in sys.path:
            sys.path.insert(0, _d)

# Re-export core.py contents so 'from core import Message' works
try:
    from core.core import *
    from core.core import (AgentType, MessageType, Task, Message, AgentMessage,
                           SharedMemory, message_bus, memory)
except Exception:
    pass
