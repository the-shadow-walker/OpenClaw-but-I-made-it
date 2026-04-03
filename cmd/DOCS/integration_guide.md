# Jarvis Integration Guide

How to connect any program, script, or service to the `ollama-cmd` agent.

**Base URL:** `http://10.0.0.58:5000`
**Auth:** None (local network)
**Streaming:** Server-Sent Events (SSE)

---

## Choose the Right Tier

| Tier | Endpoint | When to use | Example | Time |
|------|----------|-------------|---------|------|
| **Quick** | `POST /api/v1/quick` | Single shell command can answer it | "what's the uptime" / "is nginx running" / "how much RAM is free" | 1–3 s |
| **Task** | `POST /api/v1/execute` | Agent needs to think, write a file, or run a few steps | "write a python script that lists the biggest files" / "fix this config error" | 30–120 s |
| **Build** | `POST /api/v1/chains` | A human engineer would break it into tickets | "build a website with auth and a database" / "set up a monitoring stack" | 5–30 min |
| **Health** | `GET /api/v1/blueteam/report` | Just want to know if everything is okay | Daily smart summary of everything that happened | generated at 3 AM |

**Rule of thumb:**
- Can a single shell command answer it? → **Quick**
- Does it need the agent to think, write a file, or run a few steps? → **Task**
- Would a human engineer break it into tickets? → **Build**
- Just want to know if everything is okay? → **Health report**

---

## 1. Submit a Question / Task

The most common operation — send a natural-language instruction and get a result.

### curl
```bash
# Fire-and-forget (get job_id back immediately)
curl -s -X POST http://10.0.0.58:5000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"instruction": "what is the current CPU usage?"}'

# Response:
# {"job_id": "abc123...", "status": "queued"}
```

### Python
```python
import requests, time

BASE = "http://10.0.0.58:5000"

# Submit
r = requests.post(f"{BASE}/api/v1/execute",
                  json={"instruction": "list all running docker containers"})
job_id = r.json()["job_id"]

# Poll until done
while True:
    job = requests.get(f"{BASE}/api/v1/jobs/{job_id}").json()
    if job["status"] not in ("queued", "running"):
        print(job["output"])
        break
    time.sleep(2)
```

### Node.js / JavaScript
```javascript
const BASE = "http://10.0.0.58:5000";

const { job_id } = await fetch(`${BASE}/api/v1/execute`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ instruction: "show disk usage" }),
}).then(r => r.json());

// Poll
let job;
do {
  await new Promise(r => setTimeout(r, 2000));
  job = await fetch(`${BASE}/api/v1/jobs/${job_id}`).then(r => r.json());
} while (["queued", "running"].includes(job.status));

console.log(job.output);
```

---

## 2. Quick Commands (Synchronous, No ReAct Loop)

Use `/api/v1/quick` when you just need a fast answer to a simple question.
No job queue, no planning loop — the fast model picks a command, runs it, returns the result in 1-3 seconds.

### curl — natural language question
```bash
curl -s -X POST http://10.0.0.58:5000/api/v1/quick \
  -H "Content-Type: application/json" \
  -d '{"question": "what is the uptime of this server?"}'

# Response (immediate):
# {
#   "command":    "uptime",
#   "stdout":     " 13:45:12 up 3 days,  2:14,  1 user,  load average: 0.08, 0.11, 0.09\n",
#   "stderr":     "",
#   "returncode": 0,
#   "success":    true,
#   "elapsed_ms": 412,
#   "risk":       "safe"
# }
```

### curl — raw command (skips LLM entirely)
```bash
curl -s -X POST http://10.0.0.58:5000/api/v1/quick \
  -H "Content-Type: application/json" \
  -d '{"command": "df -h"}'
```

### Python
```python
import requests

def quick(question_or_cmd: str, raw: bool = False) -> str:
    """Ask Jarvis a quick question and get the answer synchronously."""
    key = "command" if raw else "question"
    r = requests.post("http://10.0.0.58:5000/api/v1/quick",
                      json={key: question_or_cmd}, timeout=20)
    r.raise_for_status()
    d = r.json()
    return d.get("stdout", "") or d.get("error", "")

print(quick("how much free memory is there?"))
print(quick("free -h", raw=True))
```

### When to use `/api/v1/quick` vs `/api/v1/execute`

| | `/api/v1/quick` | `/api/v1/execute` |
|---|---|---|
| Response time | 1–3 s | 30–120 s |
| Multi-step tasks | No | Yes |
| File creation/editing | No | Yes |
| LLM model | qwen2.5-coder:14b (fast) | qwen3-coder:30b (heavy) |
| Returns job_id | No | Yes |
| Risk limit | safe/low by default | Full ReAct safety |

**Use `/api/v1/quick` for:** uptime, disk usage, memory, running processes, service status, network info, reading a log file, system info.
**Use `/api/v1/execute` for:** anything that involves multiple steps, writing files, installing packages, debugging a failing service.

### Options
```json
{
  "question":   "what processes are using the most CPU?",
  "timeout":    15,          // seconds before command is killed (default 15)
  "allow_risk": "low"        // "safe" | "low" | "medium" (default "low")
}
```

A blocked command returns HTTP 403:
```json
{
  "error":   "Command blocked — risk \"high\" exceeds allowed \"low\"",
  "reason":  "...",
  "command": "rm -rf /",
  "risk":    "high"
}
```

---

## 3. Streaming Responses (SSE)

For real-time output as the agent works, use the SSE stream endpoint.
This is the preferred method for interactive use.

### curl
```bash
JOB_ID=$(curl -s -X POST http://10.0.0.58:5000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"instruction": "build a hello world flask app"}' | jq -r .job_id)

curl -N "http://10.0.0.58:5000/api/v1/jobs/$JOB_ID/stream"
```

### Python (requests)
```python
import requests, json

BASE = "http://10.0.0.58:5000"

job_id = requests.post(f"{BASE}/api/v1/execute",
    json={"instruction": "your task here"}).json()["job_id"]

with requests.get(f"{BASE}/api/v1/jobs/{job_id}/stream", stream=True) as r:
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["type"] == "output":
                print(event["content"], end="", flush=True)
            elif event["type"] == "complete":
                print(f"\nDone: {event['status']}")
                break
```

### JavaScript (EventSource)
```javascript
const jobRes = await fetch("http://10.0.0.58:5000/api/v1/execute", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ instruction: "your task" }),
});
const { job_id } = await jobRes.json();

const es = new EventSource(`http://10.0.0.58:5000/api/v1/jobs/${job_id}/stream`);
es.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (event.type === "output")   process.stdout.write(event.content);
  if (event.type === "complete") { es.close(); }
};
```

---

## 4. Multi-Phase Chains

For complex goals that need to be broken into multiple steps automatically.

```python
import requests

BASE = "http://10.0.0.58:5000"

# Submit chain
r = requests.post(f"{BASE}/api/v1/chains", json={
    "goal": "Build a REST API with user auth and SQLite database",
    "total_budget": 200,   # max total iterations across all phases
})
data = r.json()
chain_id = data["chain_id"]
print(f"Chain {chain_id} — {len(data['subtasks'])} phases:")
for st in data["subtasks"]:
    print(f"  {st['index']}. {st['instruction']}")

# Check status later
status = requests.get(f"{BASE}/api/v1/chains/{chain_id}").json()
print(status["status"])  # decomposing | running | completed | failed
```

---

## 5. SENTINEL Security Integration

### How the daily report cycle works

```
00:00 ──────────────────────── 03:00 ──────────────► next day
│                                  │
│  Smart logger runs 24/7          │  Agent wakes, reads last 24h
│  → ~/.agent_bin/smart_events.jsonl│  of events, writes report,
│  Auth failures, crashes,          │  wipes log clean, cycle restarts
│  SSH brute force, disk, OOM, etc. │  (capped at 15 minutes)
└──────────────────────────────────┘
```

### Read the latest daily report

This is the primary output — updated every night at 3 AM.

```bash
# Markdown (default) — paste directly into a terminal or chat
curl http://10.0.0.58:5000/api/v1/blueteam/report

# JSON — includes report text + list of archived past reports
curl "http://10.0.0.58:5000/api/v1/blueteam/report?format=json"
```

```python
import requests

r = requests.get("http://10.0.0.58:5000/api/v1/blueteam/report")
print(r.text)   # raw markdown

# or as structured JSON
data = requests.get("http://10.0.0.58:5000/api/v1/blueteam/report",
                    params={"format": "json"}).json()
print(data["report"])           # today's report
print(data["archived_reports"]) # list of past report filenames
```

Report file on disk: `~/.agent_bin/sentinel_report.md`
Archives: `~/.agent_bin/sentinel_archive/sentinel_report_YYYY-MM-DD.md`

### Trigger the daily report manually (no need to wait for 3 AM)

```bash
curl -s -X POST http://10.0.0.58:5000/api/v1/blueteam/daily_report/run \
  -H "Content-Type: application/json"

# Response (after ≤15 min):
# {"success": true, "report_length": 2341, "events_analyzed": 47, "report_path": "..."}
```

```python
import requests

r = requests.post("http://10.0.0.58:5000/api/v1/blueteam/daily_report/run",
                  timeout=960)   # allow up to 16 min
r.raise_for_status()

# Fetch the written report immediately
report = requests.get("http://10.0.0.58:5000/api/v1/blueteam/report").text
print(report)
```

### Trigger a deep scan (full ReAct loop — slower, more thorough)

Use this when you want the agent to actively dig into suspicious findings,
not just summarise what the smart logger collected.

```python
import requests

r = requests.post("http://10.0.0.58:5000/api/v1/blueteam/scan",
                  json={"focus": ""}, headers={"Content-Type": "application/json"})
job_id = r.json()["job_id"]
# poll or stream like any other job
```

### Get recent alerts
```python
alerts = requests.get("http://10.0.0.58:5000/api/v1/blueteam/alerts",
                      params={"n": 50}).json()["alerts"]
for a in alerts:
    print(f"[{a['severity']}] {a['finding']}")
```

---

## 6. Inbox Drop (File-Based Integration)

Drop a JSON file into `/mnt/storage/NAS/Jarvis/agent_inbox/` on the server.
The inbox watcher picks it up within ~5 seconds.

```json
// execute a task
{"instruction": "check if nginx is running and restart it if not"}

// blueteam scan
{"blueteam_scan": true}

// targeted investigation
{"blueteam_investigate": "unusual outbound traffic on port 4444"}
```

Useful for integrations that can write files but can't make HTTP requests
(e.g. shell scripts, cron jobs, other agents).

---

## 7. SSE Event Stream (Global)

Subscribe to all agent activity in real time:

```bash
curl -N http://10.0.0.58:5000/api/v1/events
```

Events emitted:
- `job_started` — new job picked up by worker
- `job_output` — agent printed something
- `job_completed` / `job_failed`
- `chain_started` / `chain_phase_complete` / `chain_completed`

---

## 8. Per-Endpoint SSE Streaming

### `POST /api/v1/quick/stream` — Streaming Quick Command

Runs a single command (or NL→command translation) and streams output line-by-line.
No ReAct loop — instant, low-latency.

**Request**

```bash
# GET form
curl -N "http://10.0.0.58:5000/api/v1/quick/stream?q=show+disk+usage"

# POST form — natural-language question
curl -N -X POST http://10.0.0.58:5000/api/v1/quick/stream \
     -H 'Content-Type: application/json' \
     -d '{"question": "show disk usage"}'

# POST form — explicit command
curl -N -X POST http://10.0.0.58:5000/api/v1/quick/stream \
     -H 'Content-Type: application/json' \
     -d '{"command": "df -h"}'
```

**Optional fields**: `timeout` (int seconds, default 15), `allow_risk` (`low`|`medium`|`high`).

**Events emitted** (each line: `data: <json>\n\n`):

| Event `type` | Fields | Description |
|---|---|---|
| `start` | `command: str` | Command resolved and about to run |
| `output` | `data: str` | One line of stdout/stderr |
| `done` | `returncode: int`, `success: bool`, `elapsed_ms: int` | Command finished |
| `error` | `msg: str` | Safety rejection or translation failure |

**Python example**:

```python
import requests, json

with requests.post(
    "http://10.0.0.58:5000/api/v1/quick/stream",
    json={"question": "show disk usage"},
    stream=True,
) as r:
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        if ev["type"] == "output":
            print(ev["data"], end="", flush=True)
        elif ev["type"] == "done":
            print(f"\n[exit {ev['returncode']} in {ev['elapsed_ms']}ms]")
            break
        elif ev["type"] == "error":
            print(f"Error: {ev['msg']}")
            break
```

---

### `GET /api/v1/jobs/<job_id>/stream` — Live ReAct Job Stream

Subscribe to a running agent job and receive structured thought/action/output events
as the ReAct loop executes. Also works after the job completes (drains buffered output).

**Events emitted**:

| Event `type` | Fields | Description |
|---|---|---|
| `thinking` | `content: str` | Agent's 💭 Thought for this iteration |
| `action` | `tool: str`, `command: str` | Tool chosen + command being run |
| `output` | `data: str`, `content: str`* | Command output chunk |
| `result` | `content: str` | Final ✅ FINISH answer from agent |
| `done` | `status: str` | Job terminal status (`completed`\|`failed`) |
| `complete`* | `status: str`, `success: bool` | *Backward-compat alias for `done`* |

\* `content` in output events and the `complete` event are kept for backward compatibility
with existing clients that use the old schema.

**Python example** (structured):

```python
import requests, json

job_id = requests.post(
    "http://10.0.0.58:5000/api/v1/execute",
    json={"instruction": "check what services are failing"},
).json()["job_id"]

with requests.get(
    f"http://10.0.0.58:5000/api/v1/jobs/{job_id}/stream",
    stream=True,
) as r:
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        ev = json.loads(line[6:])
        t = ev["type"]
        if t == "thinking":
            print(f"  💭 {ev['content']}")
        elif t == "action":
            print(f"  🎯 [{ev['tool']}] $ {ev['command']}")
        elif t == "output":
            print(ev["data"], end="", flush=True)
        elif t == "result":
            print(f"\n✅ {ev['content']}")
        elif t in ("done", "complete"):
            break
```

---

## 9. Adding the "Ask a Question" Pattern

The minimal integration — add an "ask Jarvis" button or command to any app:

```python
# jarvis.py — drop this file into your project
import requests, json, time

_BASE = "http://10.0.0.58:5000"

def ask(question: str, stream: bool = True) -> str:
    """Send a question to Jarvis. Returns the full output string."""
    job_id = requests.post(
        f"{_BASE}/api/v1/execute",
        json={"instruction": question},
    ).json()["job_id"]

    if stream:
        output = []
        with requests.get(f"{_BASE}/api/v1/jobs/{job_id}/stream", stream=True) as r:
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode() if isinstance(raw, bytes) else raw
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if ev["type"] == "output":
                        output.append(ev["content"])
                        print(ev["content"], end="", flush=True)
                    elif ev["type"] == "complete":
                        break
        return "".join(output)
    else:
        while True:
            job = requests.get(f"{_BASE}/api/v1/jobs/{job_id}").json()
            if job["status"] not in ("queued", "running"):
                return job.get("output", "")
            time.sleep(2)
```

Usage:
```python
from jarvis import ask
result = ask("what processes are using the most memory?")
```

---

## Endpoints Summary

| Method | Path | Body / Params |
|--------|------|---------------|
| GET  | `/health` | — |
| POST | `/api/v1/quick` | `{"question": str}` or `{"command": str}` — synchronous, no loop |
| POST | `/api/v1/execute` | `{"instruction": str}` |
| GET  | `/api/v1/jobs/<id>` | — |
| GET  | `/api/v1/jobs/<id>/stream` | — (SSE) — structured ReAct events |
| POST | `/api/v1/quick/stream` | `{"question": str}` or `{"command": str}` — (SSE) |
| GET  | `/api/v1/quick/stream` | `?q=<question>` — (SSE) |
| GET  | `/api/v1/jobs` | `?limit=50&status=completed` |
| DELETE | `/api/v1/jobs/<id>` | — |
| POST | `/api/v1/chains` | `{"goal": str, "total_budget": int}` |
| GET  | `/api/v1/chains/<id>` | — |
| GET  | `/api/v1/events` | — (SSE global stream) |
| GET  | `/api/v1/blueteam/report` | `?format=md\|json` — **read the daily report** |
| POST | `/api/v1/blueteam/daily_report/run` | — trigger report now (sync, ≤15 min) |
| POST | `/api/v1/blueteam/scan` | `{"focus": str}` — deep ReAct scan |
| GET  | `/api/v1/blueteam/alerts` | `?n=50` |
| GET  | `/api/v1/blueteam/status` | — |
