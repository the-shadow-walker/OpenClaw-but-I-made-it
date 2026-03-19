# Ollama Agent — Event Mesh Implementation Guide

Service lives at `http://10.0.0.58:5000`. All endpoints require `X-API-Key` header.

---

## 1. Debug Logs

Two files written to `/mnt/storage/NAS/Jarvis/logs/` on every agent event.
Auto-rotate at 10 MB (renamed to `.1`).

| File | Format | Use for |
|------|--------|---------|
| `agent_debug.jsonl` | One JSON object per line | Machine parsing, LLM ingestion |
| `agent_debug.txt` | Human-readable trace | Debugging, tailing live |

**Tail live:**
```bash
tail -f /mnt/storage/NAS/Jarvis/logs/agent_debug.txt
```

**Parse JSONL with jq:**
```bash
# All react iterations for a specific job
jq 'select(.event=="react_iter" and .job_id=="abc12345")' logs/agent_debug.jsonl

# All failures
jq 'select(.event=="react_iter" and .result.success==false)' logs/agent_debug.jsonl

# Chain timeline
jq 'select(.event | startswith("chain") or . == "subtask_event")' logs/agent_debug.jsonl
```

**Event types in JSONL:**

| `event` | When fired | Key fields |
|---------|-----------|------------|
| `job_start` | Job picked up by worker | `job_id`, `instruction`, `chain_id` |
| `job_end` | Job finished (any status) | `job_id`, `status`, `success`, `iterations_used`, `summary` |
| `react_iter` | Every ReAct loop iteration | `job_id`, `iteration`, `thought`, `tool`, `args`, `result`, `confidence` |
| `model_call` | Every LLM call | `model`, `purpose`, `duration_ms` |
| `chain_start` | Chain submitted | `chain_id`, `goal`, `subtask_count` |
| `chain_end` | Chain completed/failed | `chain_id`, `status`, `goal` |
| `subtask_event` | Subtask status change | `chain_id`, `subtask_index`, `sub_event`, `ac_passed` |
| `inbox_job` | File picked up from inbox | `filename`, `kind`, `instruction`/`goal` |
| `error` | Exception caught | `context`, `error` |

**Override log directory:**
```bash
AGENT_LOG_DIR=/custom/path python server.py
```

---

## 2. SSE Event Stream

Real-time push of every event to connected clients. Reconnects automatically (retry: 3000ms).

```bash
# Connect
curl -N -H 'X-API-Key: YOUR_KEY' http://10.0.0.58:5000/api/v1/events

# Parse with jq
curl -sN -H 'X-API-Key: YOUR_KEY' http://10.0.0.58:5000/api/v1/events \
  | grep '^data:' | sed 's/^data: //' | jq .
```

**Python client:**
```python
import requests, json

def watch_events(api_key: str, host: str = "http://10.0.0.58:5000"):
    with requests.get(
        f"{host}/api/v1/events",
        headers={"X-API-Key": api_key},
        stream=True,
        timeout=None,
    ) as resp:
        for line in resp.iter_lines():
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                yield event

for evt in watch_events("YOUR_KEY"):
    if evt["event"] == "chain_end":
        print(f"Chain {evt['chain_id'][:8]} → {evt['status']}")
```

**Heartbeats:** `: heartbeat` comment lines are sent every 25 s to keep the connection alive. Ignore them.

---

## 3. Outbound Webhooks

Set the env var before starting the service. Multiple URLs are comma-separated.

```bash
AGENT_WEBHOOK_URLS=http://my-llm/hook,http://dashboard/events python server.py
# or in systemd unit: Environment=AGENT_WEBHOOK_URLS=http://...
```

**POST payload shape (all events):**
```json
{
  "event":     "job_completed",
  "timestamp": "2026-03-18T21:30:00.123456",
  "service":   "ollama-agent",
  "job_id":    "e09732b7-...",
  "instruction": "check disk usage",
  "success":   true,
  "summary":   "Disk at 42% usage on /mnt/storage",
  "iterations_used": 3,
  "chain_id":  null
}
```

**Webhook events:**

| `event` | Trigger |
|---------|---------|
| `job_completed` | Job finished successfully |
| `job_failed` | Job finished with failure |
| `chain_completed` | All chain subtasks passed |
| `chain_failed` | Chain aborted after subtask failure |
| `chain_cancelled` | Chain manually cancelled |
| `chain_subtask_passed` | Subtask AC check passed |
| `chain_subtask_failed` | Subtask AC check failed |

**Simple webhook receiver (Flask):**
```python
from flask import Flask, request
app = Flask(__name__)

@app.post("/hook")
def hook():
    evt = request.json
    print(f"[{evt['event']}] {evt.get('job_id','')[:8]} — {evt.get('summary','')[:80]}")
    return "", 200
```

---

## 4. State Snapshot

Current queue depth, active jobs, and running chains. Written atomically to disk on every job start/end.

```bash
# Via API
curl -H 'X-API-Key: YOUR_KEY' http://10.0.0.58:5000/api/v1/state | jq .

# Direct file read (no auth needed if on server)
cat /mnt/storage/NAS/Jarvis/agent_state.json
```

**Shape:**
```json
{
  "updated_at": "2026-03-18T21:30:00",
  "service_version": "3.2.0-mesh",
  "active_jobs": [
    {
      "job_id": "e09732b7-...",
      "instruction": "build a Flask API",
      "chain_id": "23413a38-...",
      "subtask_index": 3,
      "started_at": "2026-03-18T21:29:55"
    }
  ],
  "queued_jobs": 0,
  "running_chains": [
    {
      "chain_id": "23413a38-...",
      "goal": "Build a Kanban app...",
      "status": "running",
      "current_subtask_index": 3,
      "subtask_count": 8
    }
  ]
}
```

---

## 5. Inbox Drop (File-Based Inbound)

Drop a JSON file into `/mnt/storage/NAS/Jarvis/agent_inbox/`. The watcher polls every 5 seconds, submits the job/chain, and moves the file to `agent_inbox/processed/`.

**Single job:**
```json
// agent_inbox/my_job.json
{
  "instruction": "check if nginx is running and restart if not",
  "max_iterations": 10
}
```

**Chain:**
```json
// agent_inbox/my_chain.json
{
  "goal": "Set up a PostgreSQL database with a users table and seed 10 rows",
  "total_budget": 100
}
```

**From another LLM tool (Python):**
```python
import json, uuid, os

def submit_to_agent(instruction: str, inbox: str = "/mnt/storage/NAS/Jarvis/agent_inbox"):
    payload = {"instruction": instruction, "max_iterations": 25}
    path = os.path.join(inbox, f"job_{uuid.uuid4().hex[:8]}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"Dropped: {path}")
```

**Supported fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instruction` | str | — | Single-job task (mutually exclusive with `goal`) |
| `goal` | str | — | Multi-phase chain goal |
| `total_budget` | int | 200 | Total iteration budget (chains only) |
| `max_iterations` | int | 25 | Per-job limit (single jobs only) |
| `model` | str | `qwen3-coder:30b` | Override model |

---

## 6. REST API Quick Reference

```bash
BASE=http://10.0.0.58:5000
KEY="your-api-key"
H="X-API-Key: $KEY"

# Submit a single job
curl -X POST "$BASE/api/v1/execute" -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"instruction": "list open ports"}'

# Submit a chain
curl -X POST "$BASE/api/v1/chains" -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"goal": "Build a todo app in Flask", "total_budget": 200}'

# Check chain status
curl "$BASE/api/v1/chains/CHAIN_ID" -H "$H" | jq .

# List all chains
curl "$BASE/api/v1/chains" -H "$H" | jq '.chains[] | {chain_id, status, goal}'

# Restart a failed chain (from first non-passed subtask)
curl -X POST "$BASE/api/v1/chains/CHAIN_ID/restart" -H "$H"

# Skip a stuck subtask (manually mark passed)
curl -X POST "$BASE/api/v1/chains/CHAIN_ID/skip/1" -H "$H" \
  -H 'Content-Type: application/json' \
  -d '{"note": "Ran manually: npx create-react-app"}'

# Cancel a chain
curl -X DELETE "$BASE/api/v1/chains/CHAIN_ID" -H "$H"

# Live SSE stream
curl -N "$BASE/api/v1/events" -H "$H"

# State snapshot
curl "$BASE/api/v1/state" -H "$H" | jq .

# Health check (no auth)
curl "$BASE/health"
```

---

## 7. Chain State Files

Chains persist to `~/.agent_bin/chains/<chain_id>.json`. Survives restarts — `_resume_running_chains()` re-queues any chain in `running` state on startup.

```bash
# List all chain files
ls ~/.agent_bin/chains/

# Inspect a chain directly
cat ~/.agent_bin/chains/23413a38-eff1-4b8c-a5fe-8071cecab5bb.json | jq .

# Find all failed chains
jq -r 'select(.status=="failed") | "\(.chain_id[:8])  \(.goal[:60])"' \
  ~/.agent_bin/chains/*.json
```

---

## 8. `debug_logger` Module (for custom integrations)

Import directly to log your own events or subscribe to the feed:

```python
import debug_logger

# Log a custom event
debug_logger.log("my_tool_event", {"tool": "my_llm", "action": "submitted_job"})

# Subscribe to all events (called in-process, synchronously)
def my_handler(event_type: str, event: dict):
    if event_type == "chain_end" and event["status"] == "completed":
        notify_slack(event["chain_id"], event["goal"])

debug_logger.subscribe(my_handler)
debug_logger.unsubscribe(my_handler)  # cleanup
```

---

*Server: `mcshell.atomos.network` · Code: `/mnt/storage/NAS/Jarvis/cmd/` · Logs: `/mnt/storage/NAS/Jarvis/logs/`*
