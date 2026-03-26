#!/usr/bin/env python3
"""
run_me.py — Jarvis Agent CLI
The single entry point for all interactions with the ollama-cmd service.

Usage:
  python run_me.py "your question or task"          # ask anything, stream output
  python run_me.py --chain "build a flask app"      # multi-phase chain
  python run_me.py --health                         # service health check
  python run_me.py --jobs                           # list recent jobs
  python run_me.py --job <id>                       # check a specific job
  python run_me.py --chains                         # list all chains
  python run_me.py --chain-status <id>              # chain detail
  python run_me.py --sentinel                       # SENTINEL status
  python run_me.py --scan                           # run security scan
  python run_me.py --report                         # daily security report
  python run_me.py --alerts                         # recent security alerts
"""

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install requests")

BASE_URL = os.environ.get("JARVIS_URL", "http://10.0.0.58:5000")
SESSION  = requests.Session()

STATUS_ICON = {
    "QUEUED": "⏳", "RUNNING": "🔄", "COMPLETED": "✅",
    "FAILED": "❌", "CANCELLED": "🚫",
    "running": "🔄", "completed": "✅", "failed": "❌",
    "cancelled": "🚫", "decomposing": "🧠", "pending": "⏳",
    "passed": "✅", "ac_failed": "⚠️ ", "skipped": "⏭️ ",
}
SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}


# ── helpers ────────────────────────────────────────────────────────────────────

def get(path, **kwargs):
    r = SESSION.get(f"{BASE_URL}{path}", **kwargs)
    r.raise_for_status()
    return r.json()

def post(path, **kwargs):
    r = SESSION.post(f"{BASE_URL}{path}", **kwargs)
    r.raise_for_status()
    return r.json()

def stream_job(job_id):
    """Stream SSE output for a job, printing as it arrives."""
    with SESSION.get(f"{BASE_URL}/api/v1/jobs/{job_id}/stream", stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "output":
                    print(event["content"], end="", flush=True)
                elif event.get("type") == "complete":
                    print(f"\n\n  Done — {event.get('status', '?').upper()}", flush=True)
                    return event.get("status", "completed")
    return "completed"


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_health():
    h = get("/health")
    print(f"  Status:   {h.get('status', '?').upper()}")
    print(f"  Version:  {h.get('version', '?')}")
    print(f"  Jobs:     active={h.get('active_jobs', 0)}  queued={h.get('queued_jobs', 0)}")
    print(f"  Minions:  {h.get('minions', '?')}")
    print(f"\n  Features:")
    for feat in h.get("features", []):
        print(f"    ▸ {feat}")


def cmd_ask(prompt, budget=200, chain_mode=False, no_stream=False):
    if chain_mode:
        print(f"  Submitting chain goal: {prompt[:80]}")
        d = post("/api/v1/chains", json={"goal": prompt, "total_budget": budget})
        cid = d.get("chain_id", "")
        print(f"  chain_id = {cid}")
        print(f"\n  Phases:")
        for st in d.get("subtasks", []):
            print(f"    {st['index']}. {st['instruction'][:80]}")
        print(f"\n  Live logs:   sudo journalctl -u ollama-cmd -f")
        print(f"  Status:      python run_me.py --chain-status {cid}")
        return

    print(f"  Submitting: {prompt[:80]}{'…' if len(prompt) > 80 else ''}", flush=True)
    job = post("/api/v1/execute", json={"instruction": prompt})
    job_id = job["job_id"]
    print(f"  job_id = {job_id}\n", flush=True)

    if no_stream:
        while True:
            j = get(f"/api/v1/jobs/{job_id}")
            if j["status"] not in ("queued", "running"):
                print(j.get("output", ""))
                print(f"\n  Status: {j['status']}")
                break
            time.sleep(2)
    else:
        try:
            stream_job(job_id)
        except KeyboardInterrupt:
            print(f"\n  Interrupted — job {job_id} may still be running.")
            print(f"  Cancel: python run_me.py --cancel {job_id}")


def cmd_jobs():
    data = get("/api/v1/jobs", params={"limit": 20})
    jobs = data.get("jobs", [])
    if not jobs:
        print("  No jobs found.")
        return
    for j in jobs:
        icon = STATUS_ICON.get(j.get("status", ""), "❓")
        print(f"  {icon} {j.get('job_id','')[:8]}  [{j.get('status','?'):10s}]  "
              f"{j.get('instruction','')[:70]}")


def cmd_job(job_id):
    j = get(f"/api/v1/jobs/{job_id}")
    icon = STATUS_ICON.get(j.get("status", ""), "❓")
    print(f"  {icon} {job_id[:8]}  [{j.get('status','?')}]")
    output = j.get("output", "") or ""
    if output.strip():
        print(f"\n{output[-3000:]}")  # last 3k chars
    err = j.get("error", "")
    if err:
        print(f"\n  Error: {err}")


def cmd_cancel(job_id):
    r = SESSION.delete(f"{BASE_URL}/api/v1/jobs/{job_id}")
    r.raise_for_status()
    print(f"  Cancelled {job_id[:8]}")


def cmd_chains():
    d = get("/api/v1/chains")
    chains = d.get("chains", [])
    if not chains:
        print("  No chains found.")
        return
    for c in chains:
        icon = STATUS_ICON.get(c.get("status", ""), "❓")
        phase  = c.get("current_subtask_index", 0)
        total  = c.get("subtask_count", "?")
        print(f"  {icon} {c.get('chain_id','')[:8]}  [{c.get('status','?'):10s}]  "
              f"phase {phase}/{total}  {c.get('goal','')[:60]}")


def cmd_chain_status(chain_id):
    d = get(f"/api/v1/chains/{chain_id}")
    icon = STATUS_ICON.get(d.get("status", ""), "❓")
    print(f"\n  {icon} Chain {d.get('chain_id','')[:8]}  [{d.get('status','?').upper()}]")
    print(f"     Goal:  {d.get('goal','')[:100]}")
    print(f"     Phase: {d.get('current_subtask_index', 0)} / {len(d.get('subtasks', []))}\n")
    for st in d.get("subtasks", []):
        si = STATUS_ICON.get(st.get("status", "pending"), "❓")
        art = (st.get("artifact") or {}).get("summary", "")[:60]
        print(f"    {si} [{st.get('index','?')}] {st.get('instruction','')[:70]}")
        if art:
            print(f"         → {art}")


def cmd_sentinel():
    d = get("/api/v1/blueteam/status")
    watching = d.get("watching", False)
    print(f"\n  SENTINEL  {'👁️  ACTIVE' if watching else '  IDLE'}")
    print(f"    Threat level:  {d.get('threat_level', '?')}")
    print(f"    Recent alerts: {d.get('recent_alert_count', 0)}")
    print(f"    Last scan:     {'✅' if d.get('last_scan_success') else '—'}")
    summary = d.get("last_scan_summary", "")
    if summary:
        print(f"\n  Last report:\n    {summary[:300]}")


def cmd_scan(focus=""):
    print("  Submitting SENTINEL scan...", flush=True)
    d = post("/api/v1/blueteam/scan", json={"focus": focus})
    job_id = d["job_id"]
    print(f"  job_id = {job_id}\n")
    try:
        stream_job(job_id)
    except KeyboardInterrupt:
        print(f"\n  Interrupted.")


def cmd_report(fmt="md"):
    r = SESSION.get(f"{BASE_URL}/api/v1/blueteam/report", params={"format": fmt})
    if r.status_code == 404:
        print("  No report yet — run: python run_me.py --scan")
        return
    r.raise_for_status()
    if fmt == "json":
        d = r.json()
        print(d.get("report", ""))
        archived = d.get("archived_reports", [])
        if archived:
            print(f"\n  Archived reports ({len(archived)}):")
            for a in archived:
                print(f"    {a}")
    else:
        print(r.text)


def cmd_alerts(n=20):
    d = get("/api/v1/blueteam/alerts", params={"n": n})
    alerts = d.get("alerts", [])
    if not alerts:
        print("  No alerts on record.")
        return
    for a in alerts:
        sev  = a.get("severity", "?")
        ts   = a.get("ts", "")[:19]
        ev   = a.get("evidence", "").strip()
        print(f"  {SEV_ICON.get(sev,'❓')} [{ts}] [{sev:8s}] {a.get('finding','')}")
        if ev:
            for line in ev.splitlines()[:3]:
                print(f"               {line}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        prog="run_me.py",
        description="Jarvis Agent — ask questions, run tasks, check security.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prompt", nargs="?", help="Question or task to run")
    p.add_argument("--url",   default=BASE_URL, help="Service URL (or $JARVIS_URL)")
    p.add_argument("--chain", metavar="GOAL",   help="Run as multi-phase chain")
    p.add_argument("--budget", type=int, default=200, help="Chain iteration budget")
    p.add_argument("--no-stream", action="store_true", help="Poll instead of streaming")

    # status
    p.add_argument("--health",  action="store_true", help="Service health check")
    p.add_argument("--jobs",    action="store_true", help="List recent jobs")
    p.add_argument("--job",     metavar="ID",        help="Show a specific job")
    p.add_argument("--cancel",  metavar="ID",        help="Cancel a job")
    p.add_argument("--chains",  action="store_true", help="List all chains")
    p.add_argument("--chain-status", metavar="ID",   help="Chain detail")

    # sentinel
    p.add_argument("--sentinel", action="store_true", help="SENTINEL watcher status")
    p.add_argument("--scan",     action="store_true", help="Run a full security scan")
    p.add_argument("--scan-focus", metavar="AREA",   help="Focus area for --scan")
    p.add_argument("--report",   action="store_true", help="Show daily security report")
    p.add_argument("--alerts",   action="store_true", help="Show recent security alerts")

    args = p.parse_args()

    global BASE_URL
    BASE_URL = args.url.rstrip("/")

    try:
        if   args.health:                    cmd_health()
        elif args.jobs:                      cmd_jobs()
        elif args.job:                       cmd_job(args.job)
        elif args.cancel:                    cmd_cancel(args.cancel)
        elif args.chains:                    cmd_chains()
        elif args.chain_status:              cmd_chain_status(args.chain_status)
        elif args.sentinel:                  cmd_sentinel()
        elif args.scan:                      cmd_scan(args.scan_focus or "")
        elif args.report:                    cmd_report()
        elif args.alerts:                    cmd_alerts()
        elif args.chain:                     cmd_ask(args.chain, args.budget, chain_mode=True)
        elif args.prompt:                    cmd_ask(args.prompt, no_stream=args.no_stream)
        else:                                p.print_help()
    except requests.exceptions.ConnectionError:
        sys.exit(f"  Cannot reach {BASE_URL} — is ollama-cmd running?")
    except requests.exceptions.HTTPError as e:
        sys.exit(f"  HTTP {e.response.status_code}: {e.response.text[:200]}")
    except KeyboardInterrupt:
        print("\n  Aborted.")


if __name__ == "__main__":
    main()
