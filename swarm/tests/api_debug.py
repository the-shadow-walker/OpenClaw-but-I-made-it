#!/usr/bin/env python3
"""
Swarm 3.0 API Debug / Health-Check Script
==========================================
Runs every endpoint in order and reports pass/fail with full details.
Hand the output back to Claude Code if anything looks wrong.

Usage:
    python3 tests/api_debug.py
    python3 tests/api_debug.py --server http://10.0.0.58:5002
    python3 tests/api_debug.py --server http://10.0.0.58:5002 --key MY_SECRET
    python3 tests/api_debug.py --skip-slow    # skip all query tests (just check routing)
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime

# ──────────────────────────────────────────────
# CONFIG  (override via CLI flags below)
# ──────────────────────────────────────────────
DEFAULT_SERVER  = "http://10.0.0.58:5002"
DEFAULT_API_KEY = None
QUERY_QUESTION  = "What is the speed of light in a vacuum?"
# How long to poll an async job before giving up (seconds)
ASYNC_POLL_SECS = 180

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' library not found. Run: pip install requests")

PASS = "\033[32m PASS\033[0m"
FAIL = "\033[31m FAIL\033[0m"
SKIP = "\033[33m SKIP\033[0m"
WARN = "\033[33m WARN\033[0m"

results = []  # list of (test_name, status, detail)  status: True/False/None/'warn'


def h(api_key):
    hdrs = {"Content-Type": "application/json"}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    return hdrs


def record(name, passed, detail=""):
    if passed == "warn":
        tag = WARN
    elif passed:
        tag = PASS
    else:
        tag = FAIL
    results.append((name, passed, detail))
    print(f"[{tag} ] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def safe_get(server, path, api_key=None, timeout=10):
    url = server + path
    try:
        r = requests.get(url, headers=h(api_key), timeout=timeout)
        return r, None
    except Exception as e:
        return None, str(e)


def safe_post(server, path, payload, api_key=None, timeout=10, stream=False):
    url = server + path
    try:
        r = requests.post(
            url, headers=h(api_key), json=payload,
            timeout=timeout, stream=stream,
        )
        return r, None
    except Exception as e:
        return None, str(e)


def pprint(data, indent=2):
    return json.dumps(data, indent=indent, default=str)


def wait_for_server_idle(server, max_wait=10):
    """Return True once active jobs drop to 0 (or max_wait exceeded)."""
    for _ in range(max_wait):
        r, err = safe_get(server, "/status", timeout=5)
        if err or r.status_code != 200:
            return False
        jobs = r.json().get("jobs", {})
        active = jobs.get("processing", 0) + jobs.get("pending", 0)
        if active == 0:
            return True
        time.sleep(1)
    return False


# ──────────────────────────────────────────────
# TESTS
# ──────────────────────────────────────────────

def test_health(server, api_key):
    section("1. GET /health")
    r, err = safe_get(server, "/health")
    if err:
        record("/health reachable", False, f"Connection error: {err}")
        return False
    ok = r.status_code == 200
    record("/health HTTP 200", ok, f"status={r.status_code}")
    if not ok:
        return False

    data = r.json()
    record("/health JSON parseable", True, pprint(data))
    orch_ok = data.get("orchestrator_available", False)
    record("orchestrator_available == True", orch_ok,
           "(orchestrator not loaded — pipeline won't work)" if not orch_ok else "")
    return True


def test_status(server, api_key):
    section("2. GET /status")
    r, err = safe_get(server, "/status")
    if err:
        record("/status reachable", False, err)
        return
    record("/status HTTP 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code != 200:
        return
    data = r.json()
    record("/status JSON parseable", True, pprint(data))

    cfg = data.get("config", {})
    # SearXNG missing is a warning, not a hard failure
    has_searxng = bool(cfg.get("searxng"))
    record("searxng configured",
           True if has_searxng else "warn",
           "" if has_searxng else "SEARXNG_URL not set — web search falls back to DuckDuckGo")
    record("auth_enabled matches expectation",
           cfg.get("auth_enabled") == bool(api_key),
           f"server auth_enabled={cfg.get('auth_enabled')} | we have key={bool(api_key)}")


def test_list_jobs(server, api_key):
    section("3. GET /jobs")
    r, err = safe_get(server, "/jobs")
    if err:
        record("/jobs reachable", False, err)
        return
    record("/jobs HTTP 200", r.status_code == 200, f"status={r.status_code}")
    if r.status_code == 200:
        data = r.json()
        record("/jobs returns total count", "total" in data,
               f"total={data.get('total', '?')} jobs in memory")


def test_result_404(server, api_key):
    section("4. GET /result/<bad_id> → expect 404")
    r, err = safe_get(server, "/result/doesnotexist")
    if err:
        record("/result 404 test", False, err)
        return
    record("/result returns 404 for unknown id", r.status_code == 404,
           f"got {r.status_code}")


def test_auth(server, api_key):
    section("5. Auth behavior")
    if api_key:
        # With auth enabled: no key → 401, wrong key → 403
        r_no_key, err = safe_post(server, "/query", {"question": "ping"},
                                  api_key=None, timeout=5)
        if err:
            record("POST /query 401 without key", False, err)
        else:
            record("POST /query returns 401 when key omitted",
                   r_no_key.status_code == 401,
                   f"got {r_no_key.status_code} — {r_no_key.text[:120]}")

        r_bad, err2 = safe_post(server, "/query", {"question": "ping"},
                                api_key="wrong_key", timeout=5)
        if err2:
            record("POST /query 403 for wrong key", False, err2)
        elif r_bad:
            record("POST /query returns 403 for wrong key",
                   r_bad.status_code == 403,
                   f"got {r_bad.status_code}")
    else:
        # No auth: just verify the open endpoints don't demand a key
        r, err = safe_get(server, "/health", timeout=5)
        if err:
            record("Auth disabled — open endpoints accessible", False, err)
        else:
            record("Auth disabled — open endpoints accessible",
                   r.status_code == 200,
                   f"/health returned {r.status_code} (no key sent)")
        print("         NOTE: /query endpoints not auth-tested since SWARM_API_KEY is off")


def test_query_async_and_wait(server, api_key, skip):
    """
    Submit one async job and poll until completion (up to ASYNC_POLL_SECS).
    Returns the completed answer string, or None on failure/skip.
    """
    section("6. POST /query_async → poll to completion")
    if skip:
        print(f"[{SKIP} ] Skipped (--skip-slow)")
        results.append(("/query_async", None, "skipped"))
        return None

    r, err = safe_post(server, "/query_async",
                       {"question": QUERY_QUESTION},
                       api_key=api_key, timeout=10)
    if err:
        record("/query_async reachable", False, err)
        return None

    record("/query_async HTTP 202", r.status_code == 202,
           f"status={r.status_code} body={r.text[:200]}")
    if r.status_code != 202:
        return None

    data   = r.json()
    job_id = data.get("job_id")
    record("/query_async returns job_id", bool(job_id), f"job_id={job_id}")
    if not job_id:
        return None

    print(f"         Polling /result/{job_id} (up to {ASYNC_POLL_SECS}s) ...")
    t0 = time.time()
    last_progress = ""
    while time.time() - t0 < ASYNC_POLL_SECS:
        time.sleep(2)
        rp, perr = safe_get(server, f"/result/{job_id}", timeout=5)
        if perr:
            print(f"         poll error: {perr}")
            continue
        pdata   = rp.json()
        status  = pdata.get("status", "?")
        progress = pdata.get("progress", "")
        if progress != last_progress:
            print(f"         [{round(time.time()-t0)}s] {status}: {progress[:80]}")
            last_progress = progress

        if status == "completed":
            elapsed = pdata.get("elapsed", "?")
            ans = pdata.get("answer", "")
            record(f"Async job completed in {elapsed}s", True)
            record("Async job has non-empty answer", bool(ans and len(ans) > 10),
                   f"answer[:200]:\n{ans[:200]}")
            return ans

        if status == "failed":
            record("Async job completed", False,
                   f"status=failed  error={pdata.get('error')}")
            return None

    # Timed out
    record(f"Async job completed within {ASYNC_POLL_SECS}s", False,
           f"Still {last_progress!r} after {ASYNC_POLL_SECS}s — "
           "Ollama may be slow/overloaded or the orchestrator is stuck")
    return None


def test_query_stream(server, api_key, skip):
    section("7. POST /query_stream (SSE — waits for idle first)")
    if skip:
        print(f"[{SKIP} ] Skipped (--skip-slow)")
        results.append(("/query_stream SSE", None, "skipped"))
        return

    # Wait until any previous job clears before firing another
    print("         Waiting for server to be idle ...")
    idle = wait_for_server_idle(server, max_wait=30)
    if not idle:
        record("/query_stream server idle before start", False,
               "Server still has active jobs after 30s wait — skipping stream test")
        return

    print(f"         Sending: '{QUERY_QUESTION}' via SSE ...")
    r, err = safe_post(server, "/query_stream",
                       {"question": QUERY_QUESTION},
                       api_key=api_key, timeout=300, stream=True)
    if err:
        record("/query_stream reachable", False, err)
        return

    record("/query_stream HTTP 200", r.status_code == 200,
           f"status={r.status_code}" + (f" — {r.text[:200]}" if r.status_code != 200 else ""))
    if r.status_code != 200:
        return

    record("/query_stream content-type is SSE",
           "text/event-stream" in r.headers.get("Content-Type", ""),
           f"Content-Type: {r.headers.get('Content-Type')}")

    seen_start  = False
    seen_phase  = False
    seen_answer = False
    seen_done   = False
    event_count = 0
    t0 = time.time()

    try:
        for raw_line in r.iter_lines(decode_unicode=True):
            if time.time() - t0 > 300:
                print("         (SSE wall-clock timeout)")
                break
            if not raw_line or not raw_line.startswith("data:"):
                continue
            try:
                evt = json.loads(raw_line[5:].strip())
            except json.JSONDecodeError:
                continue

            etype = evt.get("type", "?")
            event_count += 1
            elapsed = evt.get("elapsed", "?")

            if etype == "start":
                seen_start = True
                print(f"         SSE start     | job_id={evt.get('job_id')}")
            elif etype == "phase":
                seen_phase = True
                print(f"         SSE phase     | [{elapsed}s] {evt.get('phase_id')} — {evt.get('phase_name','')}")
            elif etype == "toks":
                print(f"         SSE toks      | {evt.get('tokens')} tok in {evt.get('seconds')}s ({evt.get('toks_per_sec')} tok/s)")
            elif etype == "answer":
                seen_answer = True
                ans = evt.get("answer", "")
                print(f"         SSE answer    | [{elapsed}s] len={len(ans)}")
                print(f"                         {ans[:150]}")
            elif etype == "done":
                seen_done = True
                print(f"         SSE done      | [{elapsed}s] total_events={event_count}")
                break
            elif etype == "heartbeat":
                print(f"         SSE heartbeat | [{elapsed}s]")
            elif etype == "error":
                print(f"         SSE error     | {evt.get('error')}")
            elif etype == "error_line":
                print(f"         SSE error_line| {evt.get('line','')[:100]}")
    except Exception as e:
        print(f"         SSE read exception: {e}")
        traceback.print_exc()

    record("SSE: received 'start' event",  seen_start)
    record("SSE: received 'phase' events", seen_phase,
           "(no phase events — orchestrator may not be printing phase markers)")
    record("SSE: received 'answer' event", seen_answer)
    record("SSE: received 'done' event",   seen_done)


def test_project_endpoints(server, api_key, skip):
    section("8. Project session endpoints")

    # Unknown session → 404 (or 503 if module missing)
    r, err = safe_get(server, "/project/session/fake_session_id")
    if err:
        record("/project/session GET reachable", False, err)
        return
    record("/project/session/<bad_id> returns 404 or 503",
           r.status_code in (404, 503),
           f"got {r.status_code} — {r.text[:120]}")
    if r.status_code == 503:
        print("         NOTE: project_session.py not loaded — project mode unavailable")
        return

    if skip:
        print(f"[{SKIP} ] /project/start skipped (--skip-slow)")
        results.append(("/project/start", None, "skipped"))
        return

    # Wait for server idle before firing a project (it uses LLM internally)
    print("         Waiting for server idle ...")
    wait_for_server_idle(server, max_wait=30)

    r2, err2 = safe_post(server, "/project/start",
                         {"description": "Debug test — a simple LED blinker circuit"},
                         api_key=api_key, timeout=60)
    if err2:
        record("/project/start reachable", False, err2)
        return

    record("/project/start HTTP 201",
           r2.status_code == 201,
           f"status={r2.status_code} body={r2.text[:300]}")
    if r2.status_code != 201:
        return

    data2 = r2.json()
    sid = data2.get("session_id") or data2.get("id")
    record("/project/start returns session_id", bool(sid), pprint(data2))

    if sid:
        r3, _ = safe_get(server, f"/project/session/{sid}")
        if r3:
            record("/project/session/<real_id> HTTP 200",
                   r3.status_code == 200,
                   f"status={r3.status_code} body={r3.text[:200]}")


# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────

def print_summary():
    section("SUMMARY")
    passed  = [r for r in results if r[1] is True]
    warned  = [r for r in results if r[1] == "warn"]
    failed  = [r for r in results if r[1] is False]
    skipped = [r for r in results if r[1] is None]

    for name, ok, _ in results:
        if ok is True:
            tag = PASS
        elif ok is False:
            tag = FAIL
        elif ok == "warn":
            tag = WARN
        else:
            tag = SKIP
        print(f"  [{tag} ] {name}")

    print(f"\n  Total: {len(results)} | "
          f"\033[32m{len(passed)} passed\033[0m | "
          f"\033[31m{len(failed)} failed\033[0m | "
          f"\033[33m{len(warned)} warnings\033[0m | "
          f"\033[33m{len(skipped)} skipped\033[0m")

    if warnings := [(n, d) for n, s, d in results if s == "warn"]:
        print("\n  Warnings (non-fatal):")
        for name, detail in warnings:
            print(f"    ~ {name}")
            if detail:
                print(f"      {detail[:200]}")

    if failed:
        print("\n  Failed tests:")
        for name, _, detail in failed:
            print(f"    - {name}")
            if detail:
                print(f"      {detail[:200]}")
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Swarm 3.0 API debug script")
    parser.add_argument("--server",     default=DEFAULT_SERVER)
    parser.add_argument("--key",        default=DEFAULT_API_KEY,
                        help="Bearer API key")
    parser.add_argument("--skip-slow",  action="store_true",
                        help="Skip query/stream/project tests (just check routing)")
    args = parser.parse_args()

    server  = args.server.rstrip("/")
    api_key = args.key

    print("=" * 60)
    print("  Swarm 3.0 API Debug Script")
    print(f"  Server    : {server}")
    print(f"  Auth      : {'Bearer ****' if api_key else 'none'}")
    print(f"  Question  : {QUERY_QUESTION}")
    print(f"  Poll limit: {ASYNC_POLL_SECS}s")
    print(f"  Time      : {datetime.now().isoformat()}")
    print("=" * 60)

    reachable = test_health(server, api_key)
    if not reachable:
        print("\n  Server unreachable — aborting.")
        print_summary()
        sys.exit(1)

    test_status(server, api_key)
    test_list_jobs(server, api_key)
    test_result_404(server, api_key)
    test_auth(server, api_key)

    # Run one query at a time — wait for it to finish before the next
    test_query_async_and_wait(server, api_key, skip=args.skip_slow)
    test_query_stream(server, api_key, skip=args.skip_slow)
    test_project_endpoints(server, api_key, skip=args.skip_slow)

    print_summary()
    failed = [r for r in results if r[1] is False]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
