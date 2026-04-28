"""RoleArbiter — P7 stub for master-mode tracking (full logic in P10).

Tracks which "master mode" (``code`` vs ``gui``) is in effect for each
conversation. P10 wires the setter on the first cmd:code / cmd:gui
delegation; P7 only exposes the dict + the ``is_master_mode`` predicate
that ``invoker.dispatch`` consults to decide whether to flag the
delegation as subordinate.

State is in-memory by design — fresh on daemon restart. Persisting it
would require a column on the ``conversations`` table or a side file,
neither of which is worth the complexity given that an ungraceful
restart legitimately resets the per-conversation routing context. The
first message after such a restart sees ``master_for(conv_id) is None``
even if a master had been set; that's a known, accepted degradation.

P7 never calls ``set_master`` — neither cmd:code nor cmd:gui targets
are routed yet. ``is_master_mode`` therefore returns ``False`` in
practice; the meta flag on the JSONL ``delegation_envelope`` event is
plumbed end-to-end so P10 can flip it on without further plumbing.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["RoleArbiter", "MasterMode"]

MasterMode = Literal["code", "gui"]


class RoleArbiter:
    """In-memory map ``conv_id -> master_mode``. First-write-wins."""

    def __init__(self) -> None:
        self._master_per_conv: dict[str, MasterMode] = {}

    def master_for(self, conv_id: str) -> MasterMode | None:
        return self._master_per_conv.get(conv_id)

    def set_master(self, conv_id: str, mode: MasterMode) -> None:
        """Record ``mode`` for ``conv_id`` if not already set (first-write-wins)."""
        self._master_per_conv.setdefault(conv_id, mode)

    def reset(self, conv_id: str) -> None:
        """Drop the master-mode entry for ``conv_id`` (e.g. on daily reset)."""
        self._master_per_conv.pop(conv_id, None)

    def is_master_mode(self, conv_id: str, target: str) -> bool:
        """True iff dispatch should run subordinate.

        That happens when a master is set on this conversation and the
        delegation target is cross-mode:

        * master ``code`` + target ``cmd:gui``  → True
        * master ``gui``  + target ``cmd:*`` (any non-gui cmd target) → True
        * otherwise → False
        """
        master = self.master_for(conv_id)
        if master is None:
            return False
        if master == "code" and target == "cmd:gui":
            return True
        return (
            master == "gui"
            and target.startswith("cmd:")
            and target != "cmd:gui"
        )
