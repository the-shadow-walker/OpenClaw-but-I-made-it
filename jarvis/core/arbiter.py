"""RoleArbiter — per-conversation master-mode tracking (P10).

Tracks which "master mode" (``code`` vs ``gui``) is in effect for each
conversation. The first ``cmd:code`` or ``cmd:gui`` dispatch in a
conversation claims the master role; subsequent cross-mode dispatches
(``code`` master + ``cmd:gui`` target, or ``gui`` master + ``cmd:code``
target) run subordinate.

State is in-memory by design — fresh on daemon restart. Persisting it
would require a column on the ``conversations`` table or a side file,
neither of which is worth the complexity given that an ungraceful
restart legitimately resets the per-conversation routing context. The
first message after such a restart sees ``master_for(conv_id) is None``
even if a master had been set; that's a known, accepted degradation.

Lifecycle hooks:

* ``invoker.dispatch`` calls :meth:`claim` on the FIRST cmd:code or
  cmd:gui in a conversation. ``cmd:react`` / ``cmd:quick`` / ``swarm:*``
  never claim or consult the arbiter.
* :meth:`Conversation.close <jarvis.core.conversation.Conversation.close>`
  calls :meth:`reset` so a closed conversation's master entry doesn't
  linger.
* :meth:`Conversation.open <jarvis.core.conversation.Conversation.open>`'s
  stale-row reset path also calls :meth:`reset` — it closes a stale row
  without instantiating a new ``Conversation`` object, so the close
  hook above doesn't fire for that path.
* The daily-close callback in ``run.py`` ALSO calls :meth:`reset`
  defensively for rows the scheduler closes directly.
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

    def claim(self, conv_id: str, mode: MasterMode) -> None:
        """Record ``mode`` for ``conv_id`` if not already set (first-write-wins)."""
        self._master_per_conv.setdefault(conv_id, mode)

    def reset(self, conv_id: str) -> None:
        """Drop the master-mode entry for ``conv_id`` (e.g. on conversation close)."""
        self._master_per_conv.pop(conv_id, None)

    def is_subordinate(self, conv_id: str, target: str) -> bool:
        """True iff this dispatch should run subordinate.

        Symmetric ``code ↔ gui`` cross-mode case only:

        * master ``code`` + target ``cmd:gui``  → True
        * master ``gui``  + target ``cmd:code`` → True
        * everything else → False

        Note this is intentionally narrower than the P7 stub: a ``gui``
        master no longer flags ``cmd:react`` / ``cmd:quick`` /
        ``cmd:chain`` as subordinate. Spec §14 only mentions the
        ``cmd:code ↔ cmd:gui`` cross-mode case, and broadcasting the flag
        on every cmd target the gui master happens to invoke would force
        CMD to interpret arbitration noise on tasks it has no opinion on.
        """
        master = self.master_for(conv_id)
        if master is None:
            return False
        return (
            (master == "code" and target == "cmd:gui")
            or (master == "gui" and target == "cmd:code")
        )
