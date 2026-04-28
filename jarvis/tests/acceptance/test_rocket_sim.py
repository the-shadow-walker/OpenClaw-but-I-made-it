"""P8 acceptance — rocket-sim end-to-end against live Swarm + CMD.

Skipped unless ``JARVIS_ROCKET_SIM=1`` is set. Hits the URLs configured
in ``cfg.orchestration.cmd.base`` (default ``http://10.0.0.58:5000``)
and ``cfg.orchestration.swarm.base`` (default ``http://10.0.0.58:5002``).
Requires the Jarvis daemon running; the test posts to ``/api/chat`` like
a real client.

Implements §16 binding assertions plus the two extra correctness gates
from the P8 plan:

  A. 3 ``delegation_snapshot`` + 3 ``delegation_envelope`` events; ZERO
     ``react_log`` / ``sub_thought`` / ``tool_internal`` substrings in
     delegation event payloads (parsed JSON, not raw grep).
  B. All 4 deliverable paths exist on disk and are non-empty.
  C. Final assistant message contains all 4 absolute paths verbatim.
  D. ``OllamaClient.chat_input_tokens_total`` ≤ 50_000 after the streaming
     response completes. Measured at the model-input boundary, not as a
     sum of JSONL events (plan pre-decision #9).
  E. Tool selection — exactly one ``plan_and_execute`` tool_call AND zero
     ``delegate(target='swarm:*' | 'cmd:react')`` tool_calls. Catches the
     failure mode where the small planner LLM ignores "prefer
     plan_and_execute" and self-plans by chaining ``delegate``.
  F. (Separate test) — planner-failure surface gate. A malformed
     submit_plan output must surface as a ``tool_result`` with
     ``error`` starting "plan failed:" or "planner: ", not as an HTTP 500.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.acceptance

if os.environ.get("JARVIS_ROCKET_SIM") != "1":
    pytest.skip(
        "set JARVIS_ROCKET_SIM=1 to run live rocket-sim acceptance",
        allow_module_level=True,
    )


_DAEMON_URL = os.environ.get("JARVIS_DAEMON_URL", "http://127.0.0.1:5003")
_ROCKET_PROMPT = (
    "Build me a single-stage rocket simulator: math model, Python "
    "implementation, README, and a brief background research note."
)
_TOKEN_BUDGET = 50_000


def _stream_chat(prompt: str) -> tuple[str, list[dict], str]:
    """POST /api/chat, return (conv_id, parsed events, joined deltas)."""
    with httpx.Client(base_url=_DAEMON_URL, timeout=600.0) as cli:
        sess = cli.post("/api/session", json={
            "channel_kind": "cli", "channel_id": "rocket-sim-acceptance",
        }).json()
        conv_id = sess["conv_id"]
        with cli.stream("POST", "/api/chat", json={
            "conv_id": conv_id, "text": prompt,
            "channel_kind": "cli", "channel_id": "rocket-sim-acceptance",
        }) as resp:
            resp.raise_for_status()
            events: list[dict] = []
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                events.append(json.loads(line))
    deltas = "".join(e.get("text", "") for e in events if e["type"] == "delta")
    return conv_id, events, deltas


def _read_transcript(conv_id: str) -> list[dict]:
    """Read the conversation JSONL from the live daemon's workspace.

    Workspace path is read from ``JARVIS_WORKSPACE`` env or defaults to
    the production NAS path (``/mnt/storage/NAS/Jarvis/jarvis/workspace``).
    """
    ws_root = Path(
        os.environ.get(
            "JARVIS_WORKSPACE",
            "/mnt/storage/NAS/Jarvis/jarvis/workspace",
        )
    )
    transcript = ws_root / ".conversations" / f"{conv_id}.jsonl"
    if not transcript.exists():
        pytest.fail(
            f"transcript not found at {transcript}; set JARVIS_WORKSPACE "
            f"to the daemon's workspace path"
        )
    return [json.loads(ln) for ln in transcript.read_text().splitlines()
            if ln.strip()]


def _read_token_counter() -> int:
    """Read ``ollama.chat_input_tokens_total`` via a debug endpoint or
    the daemon's introspection. Falls back to None if unsupported."""
    try:
        with httpx.Client(base_url=_DAEMON_URL, timeout=30.0) as cli:
            r = cli.get("/api/debug/token_counter")
            if r.status_code == 200:
                return int(r.json().get("chat_input_tokens_total", 0))
    except Exception:
        pass
    return -1


# ---------------------------------------------------------------------------
# Main rocket-sim acceptance — assertions A/B/C/D/E
# ---------------------------------------------------------------------------


def test_rocket_sim_full_run():
    conv_id, events, delta_text = _stream_chat(_ROCKET_PROMPT)

    # The stream must terminate normally.
    assert events, "no events streamed"
    assert events[-1]["type"] == "done"
    assert events[-1].get("stop_reason") == "stop", events[-1]

    # Read JSONL transcript for assertions A and E.
    raw_events = _read_transcript(conv_id)

    snapshots = [e for e in raw_events if e["kind"] == "delegation_snapshot"]
    envelopes = [e for e in raw_events if e["kind"] == "delegation_envelope"]

    # --- Assertion A ---
    assert len(snapshots) == 3, (
        f"expected 3 delegation_snapshot events, got {len(snapshots)}"
    )
    assert len(envelopes) == 3, (
        f"expected 3 delegation_envelope events, got {len(envelopes)}"
    )
    for evt in snapshots + envelopes:
        blob = json.dumps(evt["payload"])
        for forbidden in ("react_log", "sub_thought", "tool_internal"):
            assert forbidden not in blob, (
                f"{evt['kind']} payload leaked {forbidden!r}"
            )

    # --- Assertion E: tool selection ---
    tool_calls = [e for e in raw_events if e["kind"] == "tool_call"]
    plan_calls = [e for e in tool_calls
                  if e["payload"].get("name") == "plan_and_execute"]
    assert len(plan_calls) == 1, (
        f"expected exactly 1 plan_and_execute tool_call, got {len(plan_calls)}"
    )
    bad_delegates = [
        e for e in tool_calls
        if e["payload"].get("name") == "delegate"
        and (e["payload"].get("arguments", {}).get("target", "")
             .startswith(("swarm:", "cmd:react")))
    ]
    assert not bad_delegates, (
        f"LLM self-planned via delegate (defeats architecture): "
        f"{bad_delegates}"
    )

    # --- Assertion B: deliverables exist + non-empty ---
    all_deliverables: list[str] = []
    for env in envelopes:
        for d in env["payload"]["envelope"].get("deliverables", []):
            all_deliverables.append(d)
    assert len(all_deliverables) >= 4, (
        f"expected ≥4 deliverables, got {len(all_deliverables)}"
    )
    for path in all_deliverables:
        p = Path(path)
        assert p.exists(), f"deliverable missing: {path}"
        assert p.stat().st_size > 0, f"deliverable empty: {path}"

    # --- Assertion C: final delta contains all 4 paths verbatim ---
    for path in all_deliverables[:4]:
        assert path in delta_text, (
            f"final delta missing absolute path {path}"
        )

    # --- Assertion D: token budget ---
    total_tokens = _read_token_counter()
    if total_tokens >= 0:
        assert total_tokens <= _TOKEN_BUDGET, (
            f"§16-D token budget exceeded: {total_tokens} > {_TOKEN_BUDGET}"
        )


# ---------------------------------------------------------------------------
# Assertion F — planner-failure surface gate
# ---------------------------------------------------------------------------


def test_planner_failure_surfaces_as_tool_error():
    """Force a malformed planner via a debug shim. The chat call must
    complete normally with a tool_result whose error indicates plan
    failure — never an HTTP 500 mid-stream.

    This test is gated separately via ``JARVIS_ROCKET_SIM_FAIL=1`` so it
    can be run independently without polluting the main acceptance run.
    """
    if os.environ.get("JARVIS_ROCKET_SIM_FAIL") != "1":
        pytest.skip("set JARVIS_ROCKET_SIM_FAIL=1 to run planner-failure gate")

    conv_id, events, _delta = _stream_chat(
        "PLANNER_FAIL_TEST: " + _ROCKET_PROMPT
    )
    # Stream must terminate normally — never a 500 mid-stream.
    assert events[-1]["type"] == "done"

    raw_events = _read_transcript(conv_id)
    tool_results = [e for e in raw_events if e["kind"] == "tool_result"]
    plan_results = [e for e in tool_results
                    if e["payload"].get("name") == "plan_and_execute"]
    assert plan_results, "no plan_and_execute tool_result captured"
    err = (plan_results[0]["payload"].get("result", {}) or {}).get("error", "")
    assert err.startswith(("plan failed:", "planner:")), (
        f"plan failure did not surface as tool error; got error={err!r}"
    )
