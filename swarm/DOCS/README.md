# Swarm 3.0 — Architecture & Getting Started

Multi-agent AI research and computation system. Answers complex questions by decomposing them into research tasks, executing parallel web searches, performing deterministic mathematical computation, and synthesizing validated results.

---

## Quick Start

### Client (any machine on the LAN)
```bash
python3 run_me.py "What is the terminal velocity of a 1 kg steel sphere?"
python3 run_me.py -i                      # interactive REPL
python3 run_me.py ask "Design a 500N motor"  # async + poll
```

### Server (start/stop the API)
```bash
sudo systemctl start  ollama-swarm
sudo systemctl stop   ollama-swarm
sudo systemctl status ollama-swarm
journalctl -u ollama-swarm -f
```

### Health check
```bash
curl http://10.0.0.58:5002/health
```

---

## Directory Layout

```
/mnt/storage/NAS/Jarvis/swarm/
├── run_me.py               ← client-only entry point
├── _paths.py               ← sys.path helper (imported by entry points)
├── swarm2_main.py          ← legacy local entry point (full pipeline locally)
├── setup.sh                ← dependency installer
├── ollama-swarm.service    ← systemd unit (copy to /etc/systemd/system/)
├── .gitignore
│
├── core/                   ← agents, search, shared infrastructure
│   ├── base_agent.py           BaseAgent (async Ollama HTTP)
│   ├── core.py                 SharedMemory base class
│   ├── shared_memory.py        Extended shared memory
│   ├── messages.py             Fact/message types
│   ├── question_classifier.py  THEORETICAL/MATHEMATICAL/HYBRID/ENGINEERING_DESIGN
│   ├── planner_agent.py        Sub-question decomposition
│   ├── writer_agent.py         Final synthesis (qwen2.5:14b)
│   ├── consensus_agent.py      Cross-source fact checking
│   ├── flexible_search_agent.py SearXNG → DDG → Google fallback
│   ├── search_parallel.py      Parallel search executor
│   ├── checklist_system.py     Research checklist tracking
│   ├── value_extractor.py      Numeric value + unit extraction (115 patterns)
│   ├── status_display.py       Live terminal status board
│   └── verifier_agent.py       Claim verification
│
├── math/                   ← equation pipeline
│   ├── equation_builder_agent.py  Symbolic equation construction
│   ├── equation_generator.py      Complete-Python code generation (qwen2.5:14b)
│   ├── equation_validator.py      Pre-execution sanity checks
│   ├── python_compute.py          Deterministic SymPy/Pint execution
│   ├── variable_mapper_agent.py   Symbol ↔ description mapping
│   ├── formal_calculator.py       Unit parsing + dimensional consistency
│   ├── math_verifier.py           Independent re-solve + cross-check
│   ├── material_props.py          CoolProp + mendeleev property lookups
│   └── physics_supervisor.py     Physics equation plan (phi4:14b, runs before coder)
│
├── engineer/               ← design / electronics / firmware
│   ├── engineer_mode.py        6-phase EngineerModeOrchestrator
│   ├── engineering_defaults.py 50-domain defaults (562 entries, 250 aliases)
│   ├── electronics_engine.py   Power budget, voltage rail map, pin conflicts
│   └── firmware_generator.py  MCU firmware generation (qwen2.5:14b)
│
├── server/                 ← orchestrator + REST API + project mode
│   ├── orchestrator_v2_1.py   Main 7-phase pipeline orchestrator
│   ├── swarm_api_server.py    Flask REST API (auth, rate-limit, webhooks)
│   ├── project_mode.py        Guided design Q&A + BOM + TDS
│   ├── project_session.py     Stateless HTTP session management
│   ├── project_chat.py        Project follow-up conversation
│   ├── deep_search_api.py     Tavily deep-search integration
│   └── safety_db.json         Component hazard lookup table
│
├── DOCS/
│   ├── README.md              ← this file
│   ├── API_REFERENCE.md       REST endpoint reference
│   ├── INTEGRATION_GUIDE.md   How to connect external systems
│   └── openapi.yaml           OpenAPI 3.0.3 spec
│
├── tests/
│   └── test_math_stress.py    3 complex math problems (rocket, thermal, orbital)
│
└── swarm_results/             Job output directory
```

---

## Execution Pipeline

```
Question
  │
  ▼
[Phase 0A] Classification (phi4:14b)
  THEORETICAL / MATHEMATICAL / HYBRID / ENGINEERING_DESIGN
  │
  ├─ ENGINEERING_DESIGN ──► engineer_mode.py (6 phases E1–E6)
  │
  ▼
[Phase 0B] Planning — sub-question decomposition (phi4:14b)
  │
  ▼
[Phase 1-2] Research — parallel SearXNG/DDG/Google searches
  │
  ▼
[Phase 3-4] Math solve (MATHEMATICAL / HYBRID only)
  ├── Physics Supervisor (phi4:14b) — derives equations, coord frames
  ├── Value Extraction — 115-pattern regex + LLM extraction
  ├── Equation Generator (qwen2.5:14b) — complete self-contained Python
  ├── Execution (SymPy + Pint, 90s timeout, 2 self-correction retries)
  ├── Bounds Check (35-entry physics limits table)
  ├── Unit Parsing (formal_calculator.py)
  ├── Independent Verification (qwen2.5:14b second solve)
  └── Debug Reconcile (third solve when first two disagree)
  │
  ▼
[Phase 5] Synthesis (qwen2.5:14b for math results, WriterAgent otherwise)
  │
  ▼
Answer + Sources + Verification status
```

---

## Setup

### Server dependencies
```bash
bash setup.sh server
```
Installs: Flask, Flask-CORS, requests, sympy, pint, numpy

### Optional enhanced math
```bash
pip install coolprop mendeleev uncertainties
```

### Environment variables
```bash
OLLAMA_BASE_URL=http://localhost:11434   # Ollama server (default)
SEARXNG_URL=http://localhost:8080        # SearXNG search (preferred)
SWARM_SERVER=http://10.0.0.58:5002      # For client (run_me.py)
SWARM_API_KEY=sk-…                      # Optional auth token
TAVILY_API_KEY=tvly-…                   # Optional deep-search
RAPIDAPI_KEY=…                          # Amazon product search (project mode)
```

---

## Models Used

| Model | Used for |
|-------|----------|
| `phi4:14b` | Classification, planning, consensus, physics supervisor, cross-check |
| `qwen2.5:14b` | Code generation, math synthesis, firmware, writer |

---

## Key Design Constraints

- **Deterministic computation** — Math is always executed via SymPy/Pint, never by an LLM guessing the answer
- **Graceful degradation** — Search falls back SearXNG → DDG → Google; math falls back to simpler methods
- **Zero-touch imports** — All 44 modules use flat `from base_agent import BaseAgent` style; `_paths.py` and `__init__.py` handle sys.path so no existing imports need changing
- **Provenance** — Every fact in SharedMemory carries source URL, agent name, and timestamp

---

## REST API Summary

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Server health + model status |
| GET  | `/status` | Job queue stats |
| GET  | `/jobs`   | List recent jobs |
| POST | `/query`  | Synchronous question (blocks until done) |
| POST | `/query_async` | Async submit → returns `job_id` |
| GET  | `/result/<job_id>` | Poll async result |
| POST | `/project/new`  | Start a design project session |
| POST | `/project/chat` | Continue a project session |
| GET  | `/project/<id>` | Fetch project summary |

See [API_REFERENCE.md](API_REFERENCE.md) and [openapi.yaml](openapi.yaml) for full details.

---

## Systemd Service

Install once on the server:
```bash
sudo cp /mnt/storage/NAS/Jarvis/swarm/ollama-swarm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ollama-swarm
sudo systemctl start  ollama-swarm
```

Logs:
```bash
journalctl -u ollama-swarm -n 100 -f
```

---

## Git

Repository: `http://10.0.0.58:3000/Grindlewalt/Jarvis`

```bash
cd /mnt/storage/NAS/Jarvis
git log --oneline -10
git pull origin main
```
