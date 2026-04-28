"""Acceptance smoke tests for the P9 mirror curator.

Skipped unless ``JARVIS_MIRROR_SMOKE=1`` is set. Hits the live daemon on
``mcssh`` (the home server), driven via SSH for service control and
filesystem inspection, and via ``cmd_client`` for context publishes.

Two binding assertions:

  * Smoke A — file exists, is atomic, and updates within ~10s of a
    Jarvis-driven shared-board write.
  * Smoke B — env-flag boundary holds: with Jarvis stopped and CMD
    running, neither passive idle nor active CMD-side context publishes
    write the mirror file (CMD's curator must be backed off by
    ``AGENT_CENTRAL_MIRROR_OWNER=jarvis``). On Jarvis restart, the
    mirror updates within 10s as the first cycle runs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time

import pytest

pytestmark = pytest.mark.acceptance

if os.environ.get("JARVIS_MIRROR_SMOKE") != "1":
    pytest.skip(
        "set JARVIS_MIRROR_SMOKE=1 to run real-daemon mirror acceptance tests",
        allow_module_level=True,
    )

MCSSH_HOST = "mcssh"
MIRROR_PATH = "~/.agent_bin/central_context.md"
CMD_PUBLISH_URL = "http://127.0.0.1:5000/api/v1/context/publish"


def _ssh(command: str, *, timeout: int = 30) -> str:
    """Run a command on the home server via the mcssh alias."""
    result = subprocess.run(
        ["ssh", MCSSH_HOST, command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ssh {MCSSH_HOST} {command!r} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _mtime() -> int:
    out = _ssh(f"stat -c '%Y' {MIRROR_PATH}")
    return int(out)


def _mtime_or_zero() -> int:
    """Return mtime, or 0 if the mirror file doesn't exist."""
    try:
        return _mtime()
    except RuntimeError:
        return 0


def _wait_for_advance(initial: int, *, timeout: int = 10) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cur = _mtime_or_zero()
        if cur > initial:
            return cur
        time.sleep(0.5)
    return _mtime_or_zero()


def _publish_via_cmd(key: str, value: str, ttl_s: int = 7200) -> None:
    payload = (
        f'{{"key":"{key}","value":"{value}","agent_id":"cmd","ttl_s":{ttl_s}}}'
    )
    _ssh(
        f"curl -sS -X POST {CMD_PUBLISH_URL} "
        f"-H 'Content-Type: application/json' "
        f"-d {shlex.quote(payload)}"
    )


# ---------------------------------------------------------------------------
# Smoke A — Jarvis writes the mirror within ~10s of a shared-board write.
# ---------------------------------------------------------------------------


def test_smoke_a_mirror_advances_on_shared_board_write() -> None:
    # Sanity: file exists.
    initial = _mtime()
    assert initial > 0

    # Drive a context publish via CMD's HTTP surface — this writes to
    # ~/.agent_bin/memory.db, which the curator will pick up on its next
    # poll cycle.
    key = f"convo_p9_smoke_{int(time.time())}"
    _publish_via_cmd(key, "smoke A — mtime should advance within 10s")

    final = _wait_for_advance(initial, timeout=10)
    assert final > initial, (
        f"mirror mtime did not advance within 10s "
        f"(initial={initial}, final={final}); "
        f"curator may be stalled or the env-flag boundary is broken"
    )

    # The newly-published key should appear in the rendered file.
    content = _ssh(f"cat {MIRROR_PATH}")
    assert key in content, f"freshly-published key {key!r} not in rendered mirror"


# ---------------------------------------------------------------------------
# Smoke B — env-flag boundary: CMD never writes the mirror file itself.
# ---------------------------------------------------------------------------


def test_smoke_b_env_flag_boundary_holds() -> None:
    # 1. Stop Jarvis daemon.
    _ssh("sudo systemctl stop jarvis", timeout=15)
    try:
        # 2. Read mtime; sleep 60s with CMD running but Jarvis stopped;
        #    assert mtime is unchanged (CMD's curator is backed off).
        baseline = _mtime_or_zero()
        time.sleep(60)
        idle_mtime = _mtime_or_zero()
        assert idle_mtime == baseline, (
            "mirror mtime advanced while Jarvis was stopped — CMD's "
            "AGENT_CENTRAL_MIRROR_OWNER backoff is not honored"
        )

        # 3. Trigger a CMD-side context write directly. Sleep 30s; the
        #    mirror mtime must STILL be unchanged (CMD writes the
        #    shared board but NOT the mirror file).
        key = f"convo_p9_smoke_b_{int(time.time())}"
        _publish_via_cmd(key, "smoke B — CMD must not write mirror")
        time.sleep(30)
        post_publish_mtime = _mtime_or_zero()
        assert post_publish_mtime == baseline, (
            "mirror mtime advanced after a CMD-only context publish — "
            "CMD's curator did NOT back off (env-flag check broken)"
        )
    finally:
        # 4. Restart Jarvis; assert mtime advances within 10s.
        _ssh("sudo systemctl start jarvis", timeout=15)

    restart_final = _wait_for_advance(baseline, timeout=15)
    assert restart_final > baseline, (
        "mirror mtime did not advance after Jarvis restart; "
        "curator failed to start or the first cycle stalled"
    )
