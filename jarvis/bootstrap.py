"""Shared runtime startup helpers used by both the CLI and the daemon.

Both ``jarvis.cli`` and ``jarvis.run`` (and, in P5, the FastAPI server) need
the same two boot-time conveniences:

* ``_setup_logging()`` — idempotent stderr logging at INFO.
* ``_apply_workspace_override(cfg)`` — honor the ``JARVIS_WORKSPACE`` env var.

These live here, not in ``cli.py``, because they are runtime-startup concerns
shared by every entry point. Naming the module ``cli_helpers`` would mislead
the next reader into thinking these are CLI-specific.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

__all__ = ["setup_logging", "apply_workspace_override"]


def setup_logging() -> None:
    """Plain stderr logging at INFO. Idempotent (won't double-add handlers)."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def apply_workspace_override(cfg):  # type: ignore[no-untyped-def]
    """Honor JARVIS_WORKSPACE env var post-config-load."""
    override = os.environ.get("JARVIS_WORKSPACE")
    if override is None:
        return cfg
    cfg = cfg.model_copy(deep=True)
    object.__setattr__(
        cfg.paths,
        "workspace",
        Path(os.path.expandvars(os.path.expanduser(override))).resolve(),
    )
    return cfg
