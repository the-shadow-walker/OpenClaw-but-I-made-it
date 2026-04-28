"""RoleArbiter — P10 semantics (claim / is_subordinate / reset)."""

from __future__ import annotations

from jarvis.core.arbiter import RoleArbiter


def test_claim_then_get():
    a = RoleArbiter()
    a.claim("c1", "code")
    assert a.master_for("c1") == "code"


def test_first_write_wins():
    a = RoleArbiter()
    a.claim("c1", "code")
    a.claim("c1", "gui")  # ignored
    assert a.master_for("c1") == "code"


def test_reset_removes():
    a = RoleArbiter()
    a.claim("c1", "code")
    a.reset("c1")
    assert a.master_for("c1") is None
    # Reset of unknown id is a no-op.
    a.reset("never-set")


def test_master_for_unknown_returns_none():
    a = RoleArbiter()
    assert a.master_for("never-set") is None


def test_is_subordinate_pairs():
    a = RoleArbiter()

    # No master set → False for everything.
    assert a.is_subordinate("c1", "cmd:react") is False
    assert a.is_subordinate("c1", "cmd:gui") is False
    assert a.is_subordinate("c1", "cmd:code") is False

    # code master + cmd:gui target → True (cross-mode).
    a.claim("c1", "code")
    assert a.is_subordinate("c1", "cmd:gui") is True
    # code master + same-mode / non-arbitrated targets → False.
    assert a.is_subordinate("c1", "cmd:code") is False
    assert a.is_subordinate("c1", "cmd:react") is False
    assert a.is_subordinate("c1", "cmd:quick") is False
    assert a.is_subordinate("c1", "cmd:chain") is False

    # gui master + cmd:code target → True (symmetric cross-mode).
    a.claim("c2", "gui")
    assert a.is_subordinate("c2", "cmd:code") is True
    # gui master + same-mode → False.
    assert a.is_subordinate("c2", "cmd:gui") is False
    # P10 tightening: gui master no longer flags cmd:react / cmd:quick /
    # cmd:chain as subordinate. Only the symmetric cmd:code <-> cmd:gui
    # pair triggers subordinate semantics (spec §14).
    assert a.is_subordinate("c2", "cmd:react") is False
    assert a.is_subordinate("c2", "cmd:quick") is False
    assert a.is_subordinate("c2", "cmd:chain") is False

    # Swarm targets always False — arbitration is CMD-only.
    assert a.is_subordinate("c1", "swarm:engineer") is False
    assert a.is_subordinate("c2", "swarm:engineer") is False
