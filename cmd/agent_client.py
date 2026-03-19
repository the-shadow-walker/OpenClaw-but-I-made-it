#!/mnt/storage/NAS/Jarvis/.venv/bin/python3
"""Client library for Ollama Command Agent Service"""

import requests
import json
import time
from typing import Dict, Any, Optional, Generator

class AgentClient:
    """Client for Ollama Command Agent Service"""
    
    def __init__(self, base_url: str = "http://localhost:5000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers['X-API-Key'] = api_key
    
    def health(self) -> Dict[str, Any]:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def execute(
        self,
        instruction: str,
        searxng_url: str = "http://10.0.0.58:8080",
        model: str = "qwen3-coder:30b",
        async_mode: bool = True,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Execute a command"""
        payload = {
            "instruction": instruction,
            "model": model,
            "async": async_mode,
            "timeout": timeout
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/execute",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status and results"""
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for job to complete"""
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} exceeded timeout")
            
            job = self.get_job(job_id)
            
            if job['status'] not in ['queued', 'running']:
                return job
            
            time.sleep(2)
    
    def stream_output(self, job_id: str) -> Generator[Dict[str, Any], None, None]:
        """Stream job output (SSE)"""
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs/{job_id}/stream",
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    yield data
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """List jobs"""
        params = {'limit': limit}
        if status:
            params['status'] = status
        
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job"""
        response = self.session.delete(f"{self.base_url}/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    import os
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Send a prompt to the Ollama Agent Service and stream output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 agent_client.py "list all open ports"
  python3 agent_client.py "build a flask hello world app" 2>&1 | tee run.log
  python3 agent_client.py --chain "build a todo app with sqlite" --budget 200
  python3 agent_client.py --health
        """
    )
    parser.add_argument("prompt", nargs="?", help="Instruction to send to the agent")
    parser.add_argument("--url", default=os.environ.get("AGENT_URL", "http://10.0.0.58:5000"),
                        help="Service URL (default: http://10.0.0.58:5000, or $AGENT_URL)")
    parser.add_argument("--key", default=os.environ.get("AGENT_API_KEY"),
                        help="API key (default: $AGENT_API_KEY)")
    parser.add_argument("--health", action="store_true", help="Check service health and exit")
    parser.add_argument("--chain", metavar="GOAL", help="Submit as a multi-phase chain instead of single job")
    parser.add_argument("--chain-status", metavar="CHAIN_ID", help="Show status of a running/completed chain")
    parser.add_argument("--list-chains", action="store_true", help="List all chains")
    parser.add_argument("--list-jobs", action="store_true", help="List recent jobs")
    parser.add_argument("--restart-chain", metavar="CHAIN_ID", help="Restart a failed chain from its first non-passed subtask")
    parser.add_argument("--skip-subtask", nargs=2, metavar=("CHAIN_ID", "INDEX"), help="Mark a subtask as manually passed and advance the chain")
    parser.add_argument("--budget", type=int, default=200, help="Iteration budget for chains (default: 200)")
    parser.add_argument("--no-stream", action="store_true", help="Poll instead of streaming (fallback)")
    # SENTINEL blue team
    parser.add_argument("--blueteam-scan", action="store_true", help="Run a full SENTINEL security scan")
    parser.add_argument("--blueteam-focus", metavar="AREA", default="", help="Focus area for --blueteam-scan (e.g. 'SSH')")
    parser.add_argument("--blueteam-investigate", metavar="FINDING", help="Investigate a specific security finding")
    parser.add_argument("--blueteam-evidence", metavar="EVIDENCE", default="", help="Initial evidence for --blueteam-investigate")
    parser.add_argument("--blueteam-watch", action="store_true", help="Start SENTINEL continuous anomaly watcher")
    parser.add_argument("--blueteam-interval", type=int, default=60, help="Watch poll interval in seconds (default: 60)")
    parser.add_argument("--blueteam-stop", action="store_true", help="Stop SENTINEL watcher")
    parser.add_argument("--blueteam-alerts", action="store_true", help="Show recent SENTINEL security alerts")
    parser.add_argument("--blueteam-status", action="store_true", help="Show SENTINEL watcher status")

    args = parser.parse_args()

    client = AgentClient(base_url=args.url, api_key=args.key)

    # ── health check ──────────────────────────────────────────────────────────
    if args.health:
        h = client.health()
        print(f"Status:   {h.get('status')}")
        print(f"Version:  {h.get('version')}")
        print(f"Minions:  {h.get('minions')}")
        print(f"Jobs:     active={h.get('active_jobs')}  queued={h.get('queued_jobs')}")
        print(f"\nFeatures:")
        for f in h.get('features', []):
            print(f"  ▸ {f}")
        sys.exit(0)

    # ── list chains ───────────────────────────────────────────────────────────
    if args.list_chains:
        resp = client.session.get(f"{client.base_url}/api/v1/chains")
        resp.raise_for_status()
        chains = resp.json().get("chains", [])
        STATUS_ICON = {"running": "🔄", "completed": "✅", "failed": "❌", "cancelled": "🚫"}
        if not chains:
            print("No chains found.")
        for c in chains:
            icon = STATUS_ICON.get(c.get("status", ""), "❓")
            cid = c.get("chain_id", "")[:8]
            status = c.get("status", "?").upper()
            phase = c.get("current_subtask_index", 0)
            total = c.get("subtask_count", "?")
            goal = c.get("goal", "")[:80]
            print(f"  {icon} {cid}  [{status}]  phase {phase}/{total}  {goal}")
        sys.exit(0)

    # ── list jobs ─────────────────────────────────────────────────────────────
    if args.list_jobs:
        data = client.list_jobs()
        jobs = data.get("jobs", [])
        STATUS_ICON = {"QUEUED": "⏳", "RUNNING": "🔄", "COMPLETED": "✅", "FAILED": "❌", "CANCELLED": "🚫"}
        if not jobs:
            print("No jobs found.")
        for j in jobs:
            icon = STATUS_ICON.get(j.get("status", ""), "❓")
            jid = j.get("job_id", "")[:8]
            status = j.get("status", "?")
            instr = j.get("instruction", "")[:80]
            print(f"  {icon} {jid}  [{status}]  {instr}")
        sys.exit(0)

    # ── restart chain ─────────────────────────────────────────────────────────
    if args.restart_chain:
        resp = client.session.post(f"{client.base_url}/api/v1/chains/{args.restart_chain}/restart")
        resp.raise_for_status()
        d = resp.json()
        print(d.get("message", "Restarted"))
        sys.exit(0)

    # ── skip subtask ──────────────────────────────────────────────────────────
    if args.skip_subtask:
        chain_id, index = args.skip_subtask
        resp = client.session.post(
            f"{client.base_url}/api/v1/chains/{chain_id}/skip/{index}",
            json={"note": "Manually passed via --skip-subtask"},
        )
        resp.raise_for_status()
        d = resp.json()
        print(d.get("message", "Done"))
        sys.exit(0)

    # ── chain status ──────────────────────────────────────────────────────────
    if args.chain_status:
        resp = client.session.get(f"{client.base_url}/api/v1/chains/{args.chain_status}")
        resp.raise_for_status()
        d = resp.json()
        STATUS_ICON = {
            "running": "🔄", "completed": "✅", "failed": "❌",
            "cancelled": "🚫", "decomposing": "🧠",
        }
        SUBTASK_ICON = {
            "pending": "⏳", "running": "🔄", "passed": "✅",
            "failed": "❌", "ac_failed": "⚠️ ", "skipped": "⏭️ ",
        }
        chain_status = d.get("status", "?")
        icon = STATUS_ICON.get(chain_status, "❓")
        print(f"\n{icon} Chain {d.get('chain_id','')[:8]}  [{chain_status.upper()}]")
        print(f"   Goal: {d.get('goal','')[:100]}")
        print(f"   Phase: {d.get('current_subtask_index', 0)} / {len(d.get('subtasks', []))}")
        print()
        for st in d.get("subtasks", []):
            si = SUBTASK_ICON.get(st.get("status", "pending"), "❓")
            idx = st.get("index", "?")
            instr = st.get("instruction", "")[:70]
            st_status = st.get("status", "pending")
            artifact = st.get("artifact") or {}
            art_summary = artifact.get("summary", "")[:60]
            print(f"  {si} [{idx}] {instr}")
            if art_summary:
                print(f"       → {art_summary}")
            elif st_status == "running":
                print(f"       → running now...")
            ac = st.get("acceptance_result")
            if ac:
                ac_icon = "✅" if ac.get("passed") else "❌"
                print(f"       {ac_icon} AC: {ac.get('command','')[:50]}")
        print()
        sys.exit(0)

    # ── SENTINEL: scan ────────────────────────────────────────────────────────
    if args.blueteam_scan:
        resp = client.session.post(
            f"{client.base_url}/api/v1/blueteam/scan",
            json={"focus": args.blueteam_focus},
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["job_id"]
        print(f"[sentinel] Scan job submitted: {job_id}")
        print(f"[sentinel] Streaming output...\n")
        try:
            for event in client.stream_output(job_id):
                if event["type"] == "output":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "complete":
                    print(f"\n\n[sentinel] Scan done — status: {event.get('status', '?')}")
                    break
        except KeyboardInterrupt:
            print(f"\n[sentinel] Interrupted. Job {job_id} may still be running.")
        sys.exit(0)

    # ── SENTINEL: investigate ─────────────────────────────────────────────────
    if args.blueteam_investigate:
        resp = client.session.post(
            f"{client.base_url}/api/v1/blueteam/investigate",
            json={"finding": args.blueteam_investigate, "evidence": args.blueteam_evidence},
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job["job_id"]
        print(f"[sentinel] Investigation job submitted: {job_id}")
        print(f"[sentinel] Finding: {args.blueteam_investigate}")
        print(f"[sentinel] Streaming output...\n")
        try:
            for event in client.stream_output(job_id):
                if event["type"] == "output":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "complete":
                    print(f"\n\n[sentinel] Investigation done — status: {event.get('status', '?')}")
                    break
        except KeyboardInterrupt:
            print(f"\n[sentinel] Interrupted. Job {job_id} may still be running.")
        sys.exit(0)

    # ── SENTINEL: watch start ─────────────────────────────────────────────────
    if args.blueteam_watch:
        resp = client.session.post(
            f"{client.base_url}/api/v1/blueteam/watch/start",
            json={"interval": args.blueteam_interval},
        )
        resp.raise_for_status()
        d = resp.json()
        print(f"[sentinel] {d.get('message', 'Watcher started')}")
        sys.exit(0)

    # ── SENTINEL: watch stop ──────────────────────────────────────────────────
    if args.blueteam_stop:
        resp = client.session.post(f"{client.base_url}/api/v1/blueteam/watch/stop")
        resp.raise_for_status()
        d = resp.json()
        print(f"[sentinel] {d.get('message', 'Watcher stopping')}")
        sys.exit(0)

    # ── SENTINEL: alerts ──────────────────────────────────────────────────────
    if args.blueteam_alerts:
        resp = client.session.get(f"{client.base_url}/api/v1/blueteam/alerts",
                                  params={"n": 50})
        resp.raise_for_status()
        data = resp.json()
        alerts = data.get("alerts", [])
        if not alerts:
            print("[sentinel] No alerts on record.")
        else:
            SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
            for a in alerts:
                ts = a.get("ts", "")[:19]
                sev = a.get("severity", "?")
                icon = SEV_ICON.get(sev, "❓")
                finding = a.get("finding", "")
                print(f"  {icon} [{ts}] [{sev:8s}] {finding}")
        sys.exit(0)

    # ── SENTINEL: status ──────────────────────────────────────────────────────
    if args.blueteam_status:
        resp = client.session.get(f"{client.base_url}/api/v1/blueteam/status")
        resp.raise_for_status()
        d = resp.json()
        watching = d.get("watching", False)
        icon = "👁️  ACTIVE" if watching else "  IDLE"
        print(f"\n[sentinel] Watcher: {icon}")
        print(f"  Quick poll:    every {d.get('quick_interval', '?')}s")
        print(f"  Deep scan:     every {d.get('deep_interval', '?')}s")
        print(f"  Next deep:     {d.get('next_deep_scan_in', '—')}")
        print(f"  Has baseline:  {d.get('has_baseline', False)}")
        print(f"  Last scan:     {'✅' if d.get('last_scan_success') else '—'}")
        print(f"  Alert count:   {d.get('recent_alert_count', 0)}")
        summary = d.get("last_scan_summary", "")
        if summary:
            print(f"\n  Last report:   {summary[:200]}")
        recent = d.get("recent_alerts", [])
        if recent:
            print(f"\n  Recent alerts:")
            SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
            for a in recent:
                sev = a.get("severity", "?")
                print(f"    {SEV_ICON.get(sev,'❓')} [{sev}] {a.get('finding','')}")
        print()
        sys.exit(0)

    if not args.prompt and not args.chain:
        parser.print_help()
        sys.exit(1)

    # ── chain mode ────────────────────────────────────────────────────────────
    if args.chain:
        print(f"[chain] Submitting goal: {args.chain}")
        resp = client.session.post(
            f"{client.base_url}/api/v1/chains",
            json={"goal": args.chain, "total_budget": args.budget}
        )
        resp.raise_for_status()
        data = resp.json()
        chain_id = data.get("chain_id")
        print(f"[chain] chain_id={chain_id}")
        print(f"[chain] Subtasks:")
        for st in data.get("subtasks", []):
            print(f"  {st['index']}. {st['instruction'][:80]}")
        print(f"\n[chain] Watch live:   sudo journalctl -u ollama-agent -f")
        print(f"[chain] Poll status:  ./agent_client.py --chain-status {chain_id}")
        sys.exit(0)

    # ── single job mode ───────────────────────────────────────────────────────
    print(f"[job] Submitting: {args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}", flush=True)
    job = client.execute(args.prompt)
    job_id = job["job_id"]
    print(f"[job] job_id={job_id}", flush=True)
    print(f"[job] Streaming output (pipe through tee to record):\n", flush=True)

    if args.no_stream:
        result = client.wait_for_completion(job_id)
        print(result.get("output", ""))
        print(f"\n[job] Status: {result.get('status')}")
    else:
        try:
            for event in client.stream_output(job_id):
                if event["type"] == "output":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "complete":
                    print(f"\n\n[job] Done — status: {event.get('status', '?')}", flush=True)
                    break
        except KeyboardInterrupt:
            print(f"\n[job] Interrupted. Job {job_id} may still be running on the server.")
            print(f"       Cancel: curl -X DELETE {args.url}/api/v1/jobs/{job_id} -H 'X-API-Key: {args.key}'")
