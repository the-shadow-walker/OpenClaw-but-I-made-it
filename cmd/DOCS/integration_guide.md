# Jarvis Integration Guide

How to connect any program, script, or service to the `ollama-cmd` agent.

**Base URL:** `http://10.0.0.58:5000`
**Auth:** None (local network)
**Streaming:** Server-Sent Events (SSE)

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

## 2. Streaming Responses (SSE)

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

## 3. Multi-Phase Chains

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

## 4. SENTINEL Security Integration

### Get the daily security report
```bash
# Markdown
curl http://10.0.0.58:5000/api/v1/blueteam/report

# JSON (includes list of archived reports)
curl "http://10.0.0.58:5000/api/v1/blueteam/report?format=json"
```

### Trigger a scan programmatically
```python
import requests

r = requests.post("http://10.0.0.58:5000/api/v1/blueteam/scan",
                  json={"focus": ""})  # focus="" = full scan
job_id = r.json()["job_id"]
# then stream or poll like any other job
```

### Get recent alerts
```python
alerts = requests.get("http://10.0.0.58:5000/api/v1/blueteam/alerts",
                      params={"n": 50}).json()["alerts"]
for a in alerts:
    print(f"[{a['severity']}] {a['finding']}")
```

---

## 5. Inbox Drop (File-Based Integration)

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

## 6. SSE Event Stream (Global)

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

## 7. Adding the "Ask a Question" Pattern

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
| POST | `/api/v1/execute` | `{"instruction": str}` |
| GET  | `/api/v1/jobs/<id>` | — |
| GET  | `/api/v1/jobs/<id>/stream` | — (SSE) |
| GET  | `/api/v1/jobs` | `?limit=50&status=completed` |
| DELETE | `/api/v1/jobs/<id>` | — |
| POST | `/api/v1/chains` | `{"goal": str, "total_budget": int}` |
| GET  | `/api/v1/chains/<id>` | — |
| GET  | `/api/v1/events` | — (SSE global stream) |
| GET  | `/api/v1/blueteam/report` | `?format=md\|json` |
| POST | `/api/v1/blueteam/scan` | `{"focus": str}` |
| GET  | `/api/v1/blueteam/alerts` | `?n=50` |
| GET  | `/api/v1/blueteam/status` | — |
