# Swarm 3 API Reference

Base URL: `http://<host>:<port>` (default port 5002 / 5000)

## Authentication

When `SWARM_API_KEY` is set on the server, protected endpoints require:

```
Authorization: Bearer <SWARM_API_KEY>
```

Open endpoints (`/health`, `/status`, `/jobs`, `/result/<id>`, `/project/session/<id>`) do not require auth.

---

## Error Codes

| Code | Meaning |
|---|---|
| 400 | Bad request — missing or invalid field |
| 401 | Missing or malformed Authorization header |
| 403 | Invalid API key |
| 404 | Job or session not found |
| 429 | Server busy — `MAX_CONCURRENT` limit reached, retry after `retry_after` seconds |
| 500 | Internal server error |
| 503 | Feature not available (e.g. project_session.py not installed) |

---

## Open Endpoints

### `GET /health`

Health check. Always returns 200 if the server is up.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-23T12:00:00",
  "orchestrator_available": true,
  "auth_enabled": false
}
```

---

### `GET /status`

Server status and job counts.

**Response:**
```json
{
  "server": "healthy",
  "orchestrator": true,
  "jobs": {
    "pending": 0,
    "processing": 1,
    "completed": 5,
    "failed": 0,
    "running": 1
  },
  "config": {
    "max_concurrent": 3,
    "searxng": true,
    "auth_enabled": true
  }
}
```

---

### `GET /jobs`

List all jobs (active and historical).

**Response:**
```json
{
  "total": 2,
  "jobs": [
    {
      "question": "...",
      "status": "completed",
      "answer": "...",
      "progress": "",
      "created_at": "2026-03-23T12:00:00",
      "completed_at": "2026-03-23T12:01:30",
      "error": null,
      "callback_url": null
    }
  ]
}
```

---

### `GET /result/<job_id>`

Get the result of a specific async job.

**Response (pending/processing):**
```json
{
  "job_id": "a1b2c3d4",
  "question": "What is the Isp of Raptor 2?",
  "status": "processing",
  "answer": null,
  "progress": "Starting research...",
  "error": null,
  "created_at": "2026-03-23T12:00:00",
  "completed_at": null
}
```

**Response (completed):**
```json
{
  "job_id": "a1b2c3d4",
  "question": "What is the Isp of Raptor 2?",
  "status": "completed",
  "answer": "The Raptor 2 engine has a specific impulse of approximately 363 s sea-level and 380 s vacuum...",
  "progress": "",
  "error": null,
  "created_at": "2026-03-23T12:00:00",
  "completed_at": "2026-03-23T12:01:45"
}
```

---

### `GET /project/session/<session_id>`

Get full session state (useful for polling from a frontend).

**Response:**
```json
{
  "session_id": "abc123def456",
  "state": "qa",
  "specs": {"description": "GPS weather station", "purpose": "outdoor monitoring"},
  "history": [
    {"question": "What is the primary purpose?", "answer": "outdoor monitoring"}
  ],
  "pending_question": {
    "question": "Should the device be portable or fixed installation?",
    "type": "choice",
    "options": ["Portable", "Fixed installation"],
    "recommendation": "Fixed is simpler for solar charging",
    "key": "installation_type"
  },
  "qa_count": 1,
  "result_markdown": null,
  "requirements": null,
  "created_at": 1711190400.0,
  "updated_at": 1711190460.0
}
```

---

## Protected Endpoints

### `POST /query`

Synchronous query. Blocks until the answer is ready.

**Request:**
```json
{
  "question": "What is the specific impulse of the Raptor 2 engine?",
  "since": "month"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | ✅ | The research question |
| `since` | string | ❌ | Date filter: `"day"`, `"week"`, `"month"`, `"year"` |

**Response:**
```json
{
  "question": "What is the specific impulse of the Raptor 2 engine?",
  "answer": "The Raptor 2 engine achieves approximately 363 s (sea-level) and 380 s (vacuum)...",
  "timestamp": "2026-03-23T12:01:45"
}
```

**Error (busy):**
```json
{"error": "Server busy", "retry_after": 30}
```
HTTP 429

---

### `POST /query_async`

Asynchronous query. Returns a `job_id` immediately; poll `/result/<job_id>` for the answer.

**Request:**
```json
{
  "question": "Explain transformer attention mechanisms",
  "since": "week",
  "callback_url": "https://your-server/webhook"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | ✅ | The research question |
| `since` | string | ❌ | Date filter |
| `callback_url` | string | ❌ | URL to POST the result to when complete |

**Response (202 Accepted):**
```json
{
  "job_id": "a1b2c3d4",
  "question": "Explain transformer attention mechanisms",
  "status": "pending",
  "timestamp": "2026-03-23T12:00:00",
  "callback_url": "https://your-server/webhook"
}
```

**Webhook payload** (POST to `callback_url` on completion):
```json
{
  "job_id": "a1b2c3d4",
  "status": "completed",
  "answer": "...",
  "timestamp": "2026-03-23T12:01:45"
}
```

On failure:
```json
{
  "job_id": "a1b2c3d4",
  "status": "failed",
  "error": "...",
  "timestamp": "2026-03-23T12:00:30"
}
```

---

### `POST /project/start`

Start a new interactive project Q&A session.

**Request:**
```json
{
  "description": "a GPS weather station with solar charging"
}
```

**Response (201 Created):**
```json
{
  "session_id": "abc123def456",
  "state": "qa",
  "qa_count": 0,
  "question": "What is the primary purpose of this device?",
  "type": "text",
  "options": [],
  "recommendation": "",
  "key": "purpose"
}
```

`type` is one of `"text"` | `"choice"` | `"multi"`.
When `type` is `"choice"` or `"multi"`, `options` lists the available choices.

---

### `POST /project/respond`

Submit an answer to the current pending question.

**Request:**
```json
{
  "session_id": "abc123def456",
  "answer": "Remote field monitoring"
}
```

**Response (more questions):**
```json
{
  "session_id": "abc123def456",
  "state": "qa",
  "qa_count": 1,
  "question": "Should it be portable or fixed installation?",
  "type": "choice",
  "options": ["Portable", "Fixed installation"],
  "recommendation": "Fixed suits solar charging",
  "key": "installation_type"
}
```

**Response (done):**
```json
{
  "session_id": "abc123def456",
  "state": "done",
  "result_markdown": "# Project Brief\n\n**Description:** a GPS weather station...",
  "requirements": {
    "requirements": {
      "battery_capacity": "20000 mAh",
      "solar_panel": "10W 6V panel",
      "...": "..."
    },
    "component_categories": ["MCU", "GPS module", "Weather sensors", "Solar charger IC"],
    "engineering_decisions": {
      "MCU choice": "ESP32 — low power, integrated WiFi, I2C/SPI/UART"
    },
    "notes": "Apply IP65 rating for outdoor enclosure"
  }
}
```

---

## Job Lifecycle

```
POST /query_async
      │
      ▼
  status: "pending"
      │
      ▼  (background thread starts)
  status: "processing"
      │
      ├─ success ──► status: "completed"  +  answer populated
      │                                    +  webhook fired (if callback_url set)
      │
      └─ error ────► status: "failed"     +  error populated
                                           +  webhook fired (if callback_url set)
```

---

## Project Session Flow

```
POST /project/start   {"description": "..."}
         │
         ▼
     state: "qa"  ◄────────────────────────────────┐
         │                                           │
         │  (present question + options to user)     │
         │                                           │
POST /project/respond {"session_id":"...", "answer":"..."}
         │
         │  [6–10 Q&A rounds]
         │
         ▼
     state: "done"
     result_markdown + requirements returned
```

---

## Python Client Example

```python
import requests
import time

BASE = "http://localhost:5000"
HEADERS = {"Authorization": "Bearer my-secret-key", "Content-Type": "application/json"}

# Async query with polling
resp = requests.post(f"{BASE}/query_async",
                     json={"question": "What is Oberth effect?"},
                     headers=HEADERS)
job_id = resp.json()["job_id"]

while True:
    r = requests.get(f"{BASE}/result/{job_id}")
    data = r.json()
    if data["status"] == "completed":
        print(data["answer"])
        break
    if data["status"] == "failed":
        print("Error:", data["error"])
        break
    time.sleep(5)

# Project session
session = requests.post(f"{BASE}/project/start",
                        json={"description": "a laser engraver"},
                        headers=HEADERS).json()

while session["state"] == "qa":
    print("Q:", session["question"])
    answer = input("> ")
    session = requests.post(f"{BASE}/project/respond",
                            json={"session_id": session["session_id"], "answer": answer},
                            headers=HEADERS).json()

print(session["result_markdown"])
```

---

## curl Examples

```bash
# Health check
curl http://localhost:5000/health

# Sync query (no auth)
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the speed of light?"}'

# Sync query (with auth + date filter)
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SWARM_API_KEY" \
  -d '{"question": "Latest SpaceX Starship news", "since": "week"}'

# Async query
curl -X POST http://localhost:5000/query_async \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SWARM_API_KEY" \
  -d '{"question": "Explain quantum entanglement"}'

# Poll result
curl http://localhost:5000/result/a1b2c3d4

# Start project session
curl -X POST http://localhost:5000/project/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SWARM_API_KEY" \
  -d '{"description": "a home automation hub"}'

# Respond to project question
curl -X POST http://localhost:5000/project/respond \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SWARM_API_KEY" \
  -d '{"session_id": "abc123def456", "answer": "Z-Wave and Zigbee"}'
```
