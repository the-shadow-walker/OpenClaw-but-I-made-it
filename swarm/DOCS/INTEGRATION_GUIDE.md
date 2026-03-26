# Swarm 3.0 — Integration Guide

How to connect external systems, services, and question sources to the Swarm API.

---

## 1. Overview

The Swarm API is a standard JSON-over-HTTP REST service.  It accepts natural-language questions (and engineering design prompts) and returns structured answers with citations.

Every response includes:

| Field | Description |
|-------|-------------|
| `answer` | Synthesized natural-language answer |
| `sources` | List of URLs consulted |
| `verified` | `true` if answer was cross-checked |
| `elapsed_seconds` | Wall-clock solve time |
| `job_id` | Unique ID (async endpoints only) |

---

## 2. Quickstart — Synchronous

```bash
curl -X POST http://10.0.0.58:5002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the specific impulse of HTPB propellant?"}'
```

Response:
```json
{
  "answer": "HTPB/AP composite propellant typically achieves an Isp of 250–265 s at sea level …",
  "sources": ["https://…"],
  "verified": true,
  "elapsed_seconds": 18.4
}
```

---

## 3. Async Pattern — Submit, Poll, Retrieve

Use async for long-running questions (math, engineering design) to avoid HTTP timeouts.

### Submit
```bash
curl -X POST http://10.0.0.58:5002/query_async \
  -H "Content-Type: application/json" \
  -d '{"question": "Design a 500 N thrust solid rocket motor"}'
# → { "job_id": "abc123", "status": "queued" }
```

### Poll
```bash
curl http://10.0.0.58:5002/result/abc123
# → { "status": "running" }   (keep polling)
# → { "status": "completed", "answer": "…" }
```

### Python snippet
```python
import requests, time

SERVER = "http://10.0.0.58:5002"

r = requests.post(f"{SERVER}/query_async", json={"question": "…"})
job_id = r.json()["job_id"]

while True:
    r = requests.get(f"{SERVER}/result/{job_id}")
    data = r.json()
    if data["status"] in ("completed", "error"):
        print(data.get("answer"))
        break
    time.sleep(3)
```

---

## 4. Webhook / Callback

Avoid polling entirely by providing a callback URL. Swarm POSTs the full result JSON to your endpoint when the job finishes.

```bash
curl -X POST http://10.0.0.58:5002/query_async \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the drag coefficient of a blunt nose cone?",
    "webhook_url": "https://my-service.example.com/swarm-callback"
  }'
```

Your endpoint receives a POST with the same JSON body as `/result/<job_id>`.

---

## 5. Adding a New Question Source

Five-line integration pattern — works for chat bots, voice assistants, IoT triggers, CI pipelines, etc.:

```python
import requests

SWARM = "http://10.0.0.58:5002"

def ask_swarm(question: str, webhook: str | None = None) -> dict:
    payload = {"question": question}
    if webhook:
        payload["webhook_url"] = webhook
    r = requests.post(f"{SWARM}/query_async", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()   # {"job_id": "…", "status": "queued"}
```

That's it. Swarm handles search, math, validation, and synthesis internally.

---

## 6. Authentication

If the server has `SWARM_API_KEY` set, every request must include:

```
Authorization: Bearer <your-api-key>
```

```python
headers = {"Authorization": "Bearer sk-swarm-…"}
requests.post(f"{SWARM}/query", json={"question": "…"}, headers=headers)
```

Set the key on the server:
```bash
export SWARM_API_KEY=sk-swarm-mysecretkey
sudo systemctl restart ollama-swarm
```

---

## 7. Embedding in Another Service

### Python
```python
import requests

def swarm_query(question: str, server="http://10.0.0.58:5002") -> str:
    """Blocking call — returns the answer string."""
    r = requests.post(f"{server}/query", json={"question": question}, timeout=180)
    r.raise_for_status()
    return r.json().get("answer", "")
```

### Node.js / TypeScript
```typescript
async function swarmQuery(question: string, server = "http://10.0.0.58:5002"): Promise<string> {
  const res = await fetch(`${server}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`Swarm error ${res.status}`);
  const data = await res.json();
  return data.answer ?? "";
}
```

---

## 8. Error Codes & Retries

| HTTP Code | Meaning | Action |
|-----------|---------|--------|
| 200 | Success | — |
| 400 | Bad request (missing `question` field) | Fix payload |
| 401 | Unauthorized (API key required/wrong) | Set correct key |
| 429 | Too many concurrent jobs | Back off and retry |
| 500 | Internal server error | Check `journalctl -u ollama-swarm` |
| 503 | LLM / search backend unavailable | Wait and retry |

Recommended retry strategy: exponential backoff starting at 5 s, cap at 60 s, max 5 retries.

```python
import time, requests

def swarm_with_retry(question: str, retries=5) -> dict:
    delay = 5
    for attempt in range(retries):
        try:
            r = requests.post("http://10.0.0.58:5002/query",
                              json={"question": question}, timeout=180)
            if r.status_code == 429:
                time.sleep(delay); delay = min(delay * 2, 60); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1: raise
            time.sleep(delay); delay = min(delay * 2, 60)
```

---

## 9. Project / Design Mode

For engineering design problems (BOM, TDS, firmware):

```bash
curl -X POST http://10.0.0.58:5002/project/new \
  -H "Content-Type: application/json" \
  -d '{"description": "Build an FPV racing drone with 4S battery", "budget": 300}'
```

Follow-up questions use the `session_id` returned:
```bash
curl -X POST http://10.0.0.58:5002/project/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_abc", "message": "Make the frame lighter"}'
```
