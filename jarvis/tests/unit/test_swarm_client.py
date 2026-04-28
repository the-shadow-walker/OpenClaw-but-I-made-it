"""SwarmClient HTTP path — uses httpx.MockTransport (no real network).

Mirrors test_cmd_client.py's MockTransport rebind pattern.
"""

from __future__ import annotations

import threading
import time

import httpx
import pytest

from jarvis.clients.swarm import (
    SwarmBusy,
    SwarmClient,
    SwarmError,
    SwarmTimeout,
)


def _build_client(handler, **opts) -> SwarmClient:
    """Build a SwarmClient whose internal httpx client uses ``handler``."""
    transport = httpx.MockTransport(handler)
    client = SwarmClient("http://test", **opts)
    client._client.close()
    client._client = httpx.Client(
        base_url="http://test", transport=transport, timeout=30.0,
        headers=client._client.headers,
    )
    return client


# ---------------------------------------------------------------------------
# Dispatch — happy path
# ---------------------------------------------------------------------------


def _ok_envelope(extra: dict | None = None) -> dict:
    env = {
        "success": True, "summary": "ok",
        "deliverables": ["/tmp/out.md"], "context_keys_written": ["k1"],
        "sidechain_path": "/sc/x.jsonl", "error": None,
    }
    if extra:
        env.update(extra)
    return env


def test_dispatch_returns_envelope():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["method"] = req.method
        import json
        seen["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    env = c.dispatch("math", "derive equations of motion")
    assert env["success"] is True
    assert env["summary"] == "ok"
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/subagent/math")
    assert seen["body"]["task"] == "derive equations of motion"
    c.close()


def test_role_research_maps_to_deep_search():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    env = c.dispatch("research", "find background lit")
    assert env["success"] is True
    assert seen["url"].endswith("/subagent/deep_search")
    c.close()


def test_role_deep_search_also_routes_to_deep_search():
    """Tolerance: direct delegate(target='swarm:deep_search') must work."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    c.dispatch("deep_search", "x")
    assert seen["url"].endswith("/subagent/deep_search")
    c.close()


def test_engineer_role_routes_correctly():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    c.dispatch("engineer", "implement xyz")
    assert seen["url"].endswith("/subagent/engineer")
    c.close()


# ---------------------------------------------------------------------------
# Body shape
# ---------------------------------------------------------------------------


def test_context_keys_top_level_not_in_extra():
    """Per §12.0: context_keys is top-level, NOT nested in extra."""
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    c.dispatch("math", "x", context_keys=["a", "b"])
    body = seen["body"]
    assert body["context_keys"] == ["a", "b"]
    # NOT in extra:
    assert "context_keys" not in body.get("extra", {})
    assert "parent_context_keys" not in body
    c.close()


def test_extra_carries_timeout_and_max_iterations():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    c.dispatch("math", "x", max_iterations=12, timeout_s=600)
    extra = seen["body"]["extra"]
    assert extra["timeout_s"] == 600
    assert extra["max_iterations"] == 12
    c.close()


def test_timeout_clamped_at_server_hard_cap():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        seen["body"] = json.loads(req.content.decode("utf-8"))
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    c.dispatch("math", "x", timeout_s=10_000)  # > 3600 cap
    assert seen["body"]["extra"]["timeout_s"] == 3600
    c.close()


# ---------------------------------------------------------------------------
# Error paths — HTTP layer
# ---------------------------------------------------------------------------


def test_dispatch_400_raises_swarm_error():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    c = _build_client(handler)
    with pytest.raises(SwarmError):
        c.dispatch("math", "x")
    c.close()


def test_dispatch_429_raises_swarm_busy():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "all workers busy"})

    c = _build_client(handler)
    with pytest.raises(SwarmBusy):
        c.dispatch("math", "x")
    c.close()


def test_dispatch_500_returns_envelope_with_error():
    """§12.0: 500 still ships a contract envelope."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={
            "success": False, "summary": None, "deliverables": [],
            "context_keys_written": [], "sidechain_path": None,
            "error": "internal worker crash",
        })

    c = _build_client(handler)
    env = c.dispatch("math", "x")
    assert env["success"] is False
    assert "internal worker crash" in env["error"]
    c.close()


def test_dispatch_504_returns_envelope():
    """§12.0: 504 is the server-side timeout return; envelope shape."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(504, json={
            "success": False, "summary": None, "deliverables": [],
            "context_keys_written": [], "sidechain_path": None,
            "error": "specialist exceeded deadline",
        })

    c = _build_client(handler)
    env = c.dispatch("math", "x")
    assert env["success"] is False
    assert "deadline" in env["error"]
    c.close()


def test_dispatch_500_non_json_body_synthesizes_err_envelope():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"<html>oops</html>",
                              headers={"content-type": "text/html"})

    c = _build_client(handler)
    env = c.dispatch("math", "x")
    assert env["success"] is False
    assert "non-envelope body" in env["error"]
    c.close()


def test_dispatch_unexpected_status_raises():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(418, json={"error": "teapot"})

    c = _build_client(handler)
    with pytest.raises(SwarmError):
        c.dispatch("math", "x")
    c.close()


def test_dispatch_read_timeout_raises_swarm_timeout():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=req)

    c = _build_client(handler)
    with pytest.raises(SwarmTimeout):
        c.dispatch("math", "x")
    c.close()


def test_dispatch_request_error_raises_swarm_error():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conn refused", request=req)

    c = _build_client(handler)
    with pytest.raises(SwarmError):
        c.dispatch("math", "x")
    c.close()


# ---------------------------------------------------------------------------
# No-retry guarantee (§19 #7)
# ---------------------------------------------------------------------------


def test_dispatch_no_retries_on_failure():
    """A single dispatch() call must POST exactly once, regardless of
    success or failure. §19 #7 — no auto-retry."""
    counts = {"500": 0, "200": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counts["500"] += 1
        return httpx.Response(500, json={
            "success": False, "summary": None, "deliverables": [],
            "context_keys_written": [], "sidechain_path": None,
            "error": "boom",
        })

    c = _build_client(handler)
    c.dispatch("math", "x")
    assert counts["500"] == 1, "dispatch retried after failure"
    c.close()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_role_raises_value_error():
    c = _build_client(lambda req: httpx.Response(200, json=_ok_envelope()))
    with pytest.raises(ValueError):
        c.dispatch("nonsense", "x")
    c.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_token_sets_bearer_header():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler, auth_token="secret-tok")
    c.dispatch("math", "x")
    assert seen["auth"] == "Bearer secret-tok"
    c.close()


def test_no_auth_token_omits_bearer_header():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler)
    c.dispatch("math", "x")
    assert seen["auth"] is None
    c.close()


# ---------------------------------------------------------------------------
# Semaphore — backpressure (Barrier+Event, no sleep-and-pray)
# ---------------------------------------------------------------------------


def test_semaphore_blocks_third_call():
    """With max_concurrent=2, the third dispatch() must not enter the
    handler until one of the first two completes."""
    release_first = threading.Event()
    in_flight: list[float] = []
    in_flight_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def handler(req: httpx.Request) -> httpx.Response:
        with in_flight_lock:
            in_flight.append(time.monotonic())
            idx = len(in_flight)
        if idx <= 2:
            barrier.wait(timeout=2.0)
            release_first.wait(timeout=2.0)
        return httpx.Response(200, json=_ok_envelope())

    c = _build_client(handler, max_concurrent=2)

    def caller():
        c.dispatch("math", "x")

    t1 = threading.Thread(target=caller)
    t2 = threading.Thread(target=caller)
    t1.start()
    t2.start()
    deadline = time.monotonic() + 2.0
    while True:
        with in_flight_lock:
            if len(in_flight) >= 2:
                break
        if time.monotonic() >= deadline:
            raise AssertionError("first two never entered")
        time.sleep(0.005)

    t3 = threading.Thread(target=caller)
    t3.start()

    time.sleep(0.05)
    with in_flight_lock:
        assert len(in_flight) == 2, (
            f"third call leaked past sem (in_flight={len(in_flight)})"
        )

    release_at = time.monotonic()
    release_first.set()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)
    t3.join(timeout=3.0)
    assert not t1.is_alive() and not t2.is_alive() and not t3.is_alive()
    with in_flight_lock:
        assert len(in_flight) == 3
        assert in_flight[2] >= release_at, (
            "third handler entry preceded first-call release"
        )
    c.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_closes_httpx_client():
    c = _build_client(lambda req: httpx.Response(200, json=_ok_envelope()))
    c.close()
    assert c._client.is_closed


def test_context_manager_closes_on_exit():
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json=_ok_envelope())
    )
    with SwarmClient("http://test") as c:
        c._client.close()
        c._client = httpx.Client(
            base_url="http://test", transport=transport, timeout=30.0
        )
        c.dispatch("math", "x")
    assert c._client.is_closed
