"""CMDClient HTTP path — uses httpx.MockTransport (no real network)."""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from jarvis.clients.cmd import CMDClient, CMDError, CMDTimeout


def _build_client(handler, **opts) -> CMDClient:
    """Build a CMDClient whose internal httpx client uses ``handler``."""
    transport = httpx.MockTransport(handler)
    client = CMDClient("http://test", **opts)
    # Replace the internal client with the mock-transport one.
    client._client.close()
    client._client = httpx.Client(
        base_url="http://test", transport=transport, timeout=30.0
    )
    return client


# ---------------------------------------------------------------------------
# Shared board
# ---------------------------------------------------------------------------


def test_publish_round_trip():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["method"] = req.method
        import json
        seen["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json={"ok": True, "key": "k1"})

    c = _build_client(handler)
    out = c.publish("k1", "value", agent_id="jarvis", ttl_hours=24)
    assert out == {"ok": True, "key": "k1"}
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/api/v1/context")
    assert seen["body"]["key"] == "k1"
    assert seen["body"]["value"] == "value"
    assert seen["body"]["agent_id"] == "jarvis"
    assert seen["body"]["ttl_hours"] == 24
    c.close()


def test_read_context_with_prefix():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["method"] = req.method
        return httpx.Response(200, json={"items": [{"key": "p:1"}]})

    c = _build_client(handler)
    out = c.read_context(prefix="p:")
    assert out == {"items": [{"key": "p:1"}]}
    assert seen["method"] == "GET"
    assert "prefix=p" in seen["url"]
    c.close()


def test_delete_context():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "DELETE"
        assert req.url.path == "/api/v1/context/foo"
        return httpx.Response(200, json={"ok": True})

    c = _build_client(handler)
    assert c.delete_context("foo") == {"ok": True}
    c.close()


# ---------------------------------------------------------------------------
# Quick
# ---------------------------------------------------------------------------


def test_quick_returns_synced_response():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/v1/quick"
        return httpx.Response(
            200, json={"returncode": 0, "stdout": "x\n", "stderr": ""}
        )

    c = _build_client(handler)
    out = c.quick(command="uptime")
    assert out["returncode"] == 0
    assert out["stdout"] == "x\n"
    c.close()


def test_quick_safety_block_raises_with_message():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "error": "blocked",
                "reason": "rm -rf / matches dangerous pattern",
                "command": "rm -rf /",
                "risk": "high",
            },
        )

    c = _build_client(handler)
    with pytest.raises(CMDError) as ei:
        c.quick(command="rm -rf /")
    assert "rm -rf /" in str(ei.value)
    c.close()


def test_quick_requires_exactly_one_input():
    c = _build_client(lambda req: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        c.quick()
    with pytest.raises(ValueError):
        c.quick(command="x", question="y")
    c.close()


# ---------------------------------------------------------------------------
# Execute (submit + poll)
# ---------------------------------------------------------------------------


def test_execute_polls_until_envelope():
    poll_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/api/v1/execute":
            return httpx.Response(202, json={
                "job_id": "job-1", "status": "queued",
                "created_at": "2026-01-01T00:00:00Z",
            })
        if req.method == "GET" and req.url.path == "/api/v1/jobs/job-1":
            poll_count["n"] += 1
            if poll_count["n"] < 3:
                # No envelope yet — still running.
                return httpx.Response(200, json={"status": "running"})
            return httpx.Response(200, json={"envelope": {
                "success": True, "summary": "did the thing",
                "deliverables": ["/tmp/x"], "context_keys_written": [],
                "sidechain_path": "/home/foo/.agent_bin/sidechains/job-1.jsonl",
                "error": None,
            }})
        return httpx.Response(404, json={"error": "no route"})

    c = _build_client(handler, poll_interval_s=0.01)
    env = c.execute("build x")
    assert env["success"] is True
    assert env["summary"] == "did the thing"
    assert env["sidechain_path"].endswith("job-1.jsonl")
    assert poll_count["n"] >= 3
    c.close()


def test_execute_envelope_with_error_returned_unchanged():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(202, json={"job_id": "job-2",
                                             "status": "queued"})
        return httpx.Response(200, json={"envelope": {
            "success": False,
            "summary": None,
            "deliverables": [],
            "context_keys_written": [],
            "sidechain_path": None,
            "error": "safety: command rm -rf / blocked",
        }})

    c = _build_client(handler, poll_interval_s=0.01)
    env = c.execute("dangerous")
    assert env["success"] is False
    assert env["error"] == "safety: command rm -rf / blocked"
    c.close()


def test_execute_timeout_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            return httpx.Response(202, json={"job_id": "job-3"})
        return httpx.Response(200, json={"status": "running"})

    c = _build_client(handler, poll_interval_s=0.01)
    with pytest.raises(CMDTimeout):
        c.execute("x", timeout_s=1)  # 1s deadline
    c.close()


# ---------------------------------------------------------------------------
# Semaphore — backpressure (event-based, not sleep-and-pray)
# ---------------------------------------------------------------------------


def test_semaphore_blocks_third_call():
    """With max_concurrent=2, the third concurrent execute() call must
    not enter the submit handler until one of the first two completes.

    Implementation: park the first two calls inside the handler on a
    ``threading.Event``; release the first; record per-thread "got past
    the sem" timestamps and the first thread's release timestamp;
    assert that thread 3's enter timestamp is on or after thread 1's
    release timestamp.
    """
    release_first = threading.Event()
    in_flight = []
    in_flight_lock = threading.Lock()
    barrier = threading.Barrier(2)
    third_entered_at: list[float] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST" and req.url.path == "/api/v1/execute":
            with in_flight_lock:
                in_flight.append(time.monotonic())
                idx = len(in_flight)
            if idx <= 2:
                # First two calls: wait at the barrier so they're
                # genuinely concurrent inside the sem, then block until
                # the test releases.
                barrier.wait(timeout=2.0)
                release_first.wait(timeout=2.0)
            return httpx.Response(202, json={"job_id": f"j-{idx}"})
        # Poll path — return terminal envelope immediately.
        return httpx.Response(200, json={"envelope": {
            "success": True, "summary": "ok",
            "deliverables": [], "context_keys_written": [],
            "sidechain_path": None, "error": None,
        }})

    c = _build_client(handler, max_concurrent=2, poll_interval_s=0.01)

    # Wrap execute to capture each thread's "actually started its HTTP
    # POST" timestamp. We approximate "got past the sem" by recording
    # right before the call returns from semaphore acquisition; here we
    # use the in_flight handler invocation as the proxy.
    def caller(record_into=None):
        c.execute("task")
        if record_into is not None:
            record_into.append(time.monotonic())

    t1 = threading.Thread(target=caller)
    t2 = threading.Thread(target=caller)
    t3_done = []

    def t3_caller():
        # t3 will block at the sem because max_concurrent=2; once a slot
        # frees, it gets in. The handler will respond instantly because
        # release_first is already set by then.
        third_entered_at.append(time.monotonic())  # before sem
        c.execute("task")
        t3_done.append(time.monotonic())

    t1.start()
    t2.start()
    # Wait until both first calls are inside the handler and at the barrier.
    deadline = time.monotonic() + 2.0
    while True:
        with in_flight_lock:
            if len(in_flight) >= 2:
                break
        if time.monotonic() >= deadline:
            raise AssertionError("first two calls never entered the handler")
        time.sleep(0.005)

    # Now spawn t3 — it should be blocked at the sem.
    t3 = threading.Thread(target=t3_caller)
    t3.start()

    # Give t3 a moment to attempt to acquire the sem.
    time.sleep(0.05)
    with in_flight_lock:
        # Only the first two should have made it into the handler.
        assert len(in_flight) == 2, (
            f"third call leaked past the semaphore (in_flight={len(in_flight)})"
        )

    release_at = time.monotonic()
    release_first.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    t3.join(timeout=3.0)
    assert not t1.is_alive() and not t2.is_alive() and not t3.is_alive()
    # t3 entered the handler only after the first two were released.
    with in_flight_lock:
        assert len(in_flight) == 3
        third_handler_entry = in_flight[2]
    assert third_handler_entry >= release_at, (
        f"third handler entry ({third_handler_entry}) "
        f"happened before first-call release ({release_at})"
    )
    c.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_closes_httpx_client():
    c = _build_client(lambda req: httpx.Response(200, json={}))
    c.close()
    assert c._client.is_closed
