# Swarm 3 — Multi-Agent AI Research & Computation System

Swarm 3 answers complex questions by decomposing them into research tasks, executing parallel web searches, performing deterministic mathematical computation, and synthesising validated results into a sourced answer.

---

## Quick Start

```bash
# 1. Install dependencies
bash setup.sh server

# 2. Start local services (if not already running)
#    - Ollama:   ollama serve
#    - SearXNG:  docker run -p 8080:8080 searxng/searxng

# 3. Run a question
python3 swarm2_main.py "What is the specific impulse of the Raptor 2 engine?"

# 4. Interactive mode
python3 swarm2_main.py --interactive

# 5. Engineering design mode
python3 swarm2_main.py --engineer

# 6. Project mode (guided Q&A + BOM generation)
python3 swarm2_main.py --project

# 7. Start REST API server
python3 swarm_api_server.py --port 5000
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama LLM server |
| `SEARXNG_URL` | `http://localhost:8080` | Self-hosted search engine |
| `SWARM_SERVER` | _(none)_ | Remote swarm server for client mode |
| `SWARM_API_KEY` | _(none)_ | Bearer token for API auth (unset = no auth) |
| `SWARM_API_PORT` | `5002` | Port for the API server |
| `MAX_CONCURRENT_JOBS` | `3` | Max parallel async jobs |
| `TAVILY_API_KEY` | _(none)_ | Optional deep-search API key |
| `RAPIDAPI_KEY` | _(none)_ | Amazon product search (project mode) |
| `FLASK_DEBUG` | `False` | Enable Flask debug mode |

---

## Architecture Overview

```
Question
   │
   ▼
Phase 0: Classification ──────────────────── question_classifier.py
   │  THEORETICAL / MATHEMATICAL / HYBRID / ENGINEERING_DESIGN
   ▼
Phase 1: Planning ─────────────────────────── planner_agent.py
   │  Sub-questions list
   ▼
Phase 2: Research ─────────────────────────── search_parallel.py
   │  Parallel web search (SearXNG → DuckDuckGo → Google)
   │  Deep iterative search for THEORETICAL/HYBRID questions
   ▼
Phase 3-4: Math (MATHEMATICAL/HYBRID only) ── equation_generator.py
   │  LLM writes complete self-contained Python script
   │  python_compute.py executes it (SymPy / NumPy / SciPy)
   │  Up to 2 self-correction retries on failure
   ▼
Phase 5: Validation ───────────────────────── consensus_agent.py, equation_validator.py
   │  Cross-source fact checking, contradiction detection
   ▼
Phase 6: Synthesis ────────────────────────── writer_agent.py
   │  Final answer with citations
   ▼
Answer
```

All agents communicate through a central `SharedMemory` instance. Facts carry full provenance (source URL, agent, timestamp).

**Key design rule:** Mathematical computation is always performed deterministically via SymPy/NumPy/SciPy. LLMs write the code; they never compute the numbers.

---

## Key Files

| File | Purpose |
|---|---|
| `swarm2_main.py` | CLI entry point |
| `orchestrator_v2_1.py` | Main pipeline coordinator |
| `question_classifier.py` | Classify question type |
| `planner_agent.py` | Decompose question into sub-queries |
| `search_parallel.py` | Parallel + deep web search |
| `equation_generator.py` | Generate executable Python from problem |
| `python_compute.py` | Run generated scripts (SymPy/NumPy/SciPy) |
| `consensus_agent.py` | Cross-source validation |
| `writer_agent.py` | Final answer synthesis |
| `engineer_mode.py` | Engineering design mode (6-phase pipeline) |
| `project_mode.py` | Interactive project assistant + BOM |
| `project_session.py` | Stateless HTTP session wrapper for project mode |
| `swarm_api_server.py` | REST API server (Flask) |
| `core.py` + `shared_memory.py` | Shared memory implementation |

---

## API Quick Reference

See [API_REFERENCE.md](API_REFERENCE.md) for full documentation.

```bash
# Sync query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SWARM_API_KEY" \
  -d '{"question": "What is the boiling point of nitrogen?", "since": "month"}'

# Async query with webhook
curl -X POST http://localhost:5000/query_async \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain transformer attention mechanisms", "callback_url": "https://your-server/webhook"}'

# Start project session
curl -X POST http://localhost:5000/project/start \
  -H "Content-Type: application/json" \
  -d '{"description": "a solar-powered GPS weather station"}'
```

---

## Running Tests

```bash
# Math pipeline stress tests (3 complex multi-variable problems)
python3 test_math_stress.py

# Single problem
python3 test_math_stress.py --problem 1   # two-stage rocket
python3 test_math_stress.py --problem 2   # thermal analysis
python3 test_math_stress.py --problem 3   # orbital transfer

# Verify python_compute imports correctly
python3 python_compute.py

# Validate OpenAPI spec
python3 -m openapi_spec_validator openapi.yaml
```

---

## Deployment Modes

**Local (all-in-one):**
- Ollama + SearXNG + `swarm2_main.py` on one machine

**Client/Server:**
- Server: `python3 swarm_api_server.py --port 5000`
- Client: set `SWARM_SERVER=http://<server-ip>:5000`

**With auth:**
```bash
export SWARM_API_KEY=your-secret-token
python3 swarm_api_server.py --port 5000
```
All write endpoints require `Authorization: Bearer your-secret-token`.
