"""Acceptance smoke tests against the real CMD daemon.

Skipped unless ``JARVIS_CMD_SMOKE=1`` is set. Hits the URL configured in
``cfg.orchestration.cmd.base`` (default ``http://10.0.0.58:5000``).

Covers 5 of 6 steps from CMD's JARVIS_INTEGRATION_GUIDE §9 verification
recipe — the chain step is deferred to P8 alongside the multi-phase
planner.
"""

from __future__ import annotations

import contextlib
import os
import time

import pytest

from jarvis.clients.cmd import CMDClient, CMDError
from jarvis.config import load_config

pytestmark = pytest.mark.acceptance

if os.environ.get("JARVIS_CMD_SMOKE") != "1":
    pytest.skip(
        "set JARVIS_CMD_SMOKE=1 to run real-CMD acceptance tests",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def cmd_client() -> CMDClient:
    try:
        cfg = load_config()
    except Exception:  # noqa: BLE001
        from jarvis.config import JarvisConfig
        cfg = JarvisConfig()
    c = CMDClient(
        cfg.orchestration.cmd.base,
        max_concurrent=cfg.orchestration.cmd.max_concurrent,
        quick_timeout_s=cfg.orchestration.cmd.quick_timeout_s,
        react_max_wait_s=cfg.orchestration.cmd.react_max_wait_s,
        chain_max_wait_s=cfg.orchestration.cmd.chain_max_wait_s,
    )
    yield c
    c.close()


# Step 1 — quick uptime
def test_step_1_quick_uptime(cmd_client):
    out = cmd_client.quick(command="uptime")
    assert out["returncode"] == 0
    assert "load average" in out["stdout"]


# Step 2 — context round-trip
def test_step_2_context_round_trip(cmd_client):
    key = f"jarvis_smoke_{int(time.time())}"
    try:
        cmd_client.publish(key, "hello from jarvis P7 smoke")
        rd = cmd_client.read_context(key=key)
        # Tolerate either {value: "..."} or {items: [{value: "..."}]} shapes.
        if isinstance(rd, dict) and "value" in rd:
            assert "hello from jarvis P7 smoke" in rd["value"]
        elif isinstance(rd, dict) and "items" in rd:
            values = [it.get("value") for it in rd["items"] if isinstance(it, dict)]
            assert any("hello from jarvis P7 smoke" in (v or "") for v in values)
        else:
            raise AssertionError(f"unexpected read_context shape: {rd!r}")
    finally:
        with contextlib.suppress(Exception):
            cmd_client.delete_context(key)


# Step 3 — read-only ReAct
def test_step_3_react_read_only(cmd_client):
    env = cmd_client.execute(
        "Read /etc/hostname and report its contents. Do not modify any files."
    )
    assert env["success"] is True, env
    # Sidechain JSONL must be set for delegated jobs.
    assert env.get("sidechain_path"), env


# Step 4 — ReAct + context_keys
def test_step_4_react_with_context_keys(cmd_client):
    key = f"jarvis_smoke_brief_{int(time.time())}"
    cmd_client.publish(
        key,
        "Write a single-file Python script at /tmp/jarvis_smoke_p4.py "
        "that prints 'hello smoke'. Do not run it.",
    )
    try:
        env = cmd_client.execute(
            "Use the brief in the context to write the script.",
            context_keys=[key],
        )
        assert env["success"] is True, env
        deliverables = env.get("deliverables") or []
        assert any(".py" in str(d) for d in deliverables), env
    finally:
        with contextlib.suppress(Exception):
            cmd_client.delete_context(key)


# Step 6 — safety surfaces verbatim via /quick (deterministic 403)
def test_step_6_safety_surfaces_via_quick(cmd_client):
    with pytest.raises(CMDError) as ei:
        cmd_client.quick(command="rm -rf /")
    msg = str(ei.value).lower()
    assert "rm -rf" in msg or "blocked" in msg or "safety" in msg
