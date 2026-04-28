"""RoleArbiter — P7 stub semantics."""

from __future__ import annotations

from jarvis.core.arbiter import RoleArbiter


def test_set_then_get():
    a = RoleArbiter()
    a.set_master("c1", "code")
    assert a.master_for("c1") == "code"


def test_first_write_wins():
    a = RoleArbiter()
    a.set_master("c1", "code")
    a.set_master("c1", "gui")  # ignored
    assert a.master_for("c1") == "code"


def test_reset_removes():
    a = RoleArbiter()
    a.set_master("c1", "code")
    a.reset("c1")
    assert a.master_for("c1") is None
    # Reset of unknown id is a no-op.
    a.reset("never-set")


def test_master_for_unknown_returns_none():
    a = RoleArbiter()
    assert a.master_for("never-set") is None


def test_is_master_mode_pairs():
    a = RoleArbiter()

    # No master set → False for everything.
    assert a.is_master_mode("c1", "cmd:react") is False
    assert a.is_master_mode("c1", "cmd:gui") is False

    # code master + cmd:gui target → True (cross-mode).
    a.set_master("c1", "code")
    assert a.is_master_mode("c1", "cmd:gui") is True
    # code master + same-mode target → False.
    assert a.is_master_mode("c1", "cmd:react") is False
    assert a.is_master_mode("c1", "cmd:quick") is False

    # gui master + any non-gui cmd target → True.
    a.set_master("c2", "gui")
    assert a.is_master_mode("c2", "cmd:react") is True
    assert a.is_master_mode("c2", "cmd:quick") is True
    assert a.is_master_mode("c2", "cmd:chain") is True
    assert a.is_master_mode("c2", "cmd:gui") is False  # same-mode

    # Swarm targets always False (P7 + P8 both — arbitration is CMD-only).
    assert a.is_master_mode("c1", "swarm:engineer") is False
    assert a.is_master_mode("c2", "swarm:engineer") is False
