# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Canonical Location

**Primary location**: `mcssh:/mnt/storage/NAS/Jarvis/swarm/`
**Git remote**: `http://10.0.0.58:3000/Grindlewalt/Jarvis`

- **NEVER SCP to `~/swarm3` again** — the old `~/swarm3/` is legacy; all changes go to the new path
- SCP target: `mcssh:/mnt/storage/NAS/Jarvis/swarm/<subdir>/`
- All files in `/mnt/storage/NAS/Jarvis/swarm/` are Grindlewalt-owned — direct SCP works (no sudo needed)
- The local `/Users/grant/swarm3/` directory is kept in sync manually for editing

## Project Overview

Swarm 3.0 is a multi-agent AI research and computation system. It answers complex questions by decomposing them into research tasks, executing parallel web searches, performing deterministic mathematical computation, and synthesizing validated results.

## Running the System

```bash
# Client (from any machine on LAN)
python3 run_me.py "Your question here"     # single question
python3 run_me.py -i                        # interactive REPL
python3 run_me.py health                    # server health check

# Server management
sudo systemctl start  ollama-swarm
sudo systemctl stop   ollama-swarm
sudo systemctl status ollama-swarm
journalctl -u ollama-swarm -f

# Legacy local mode (on server)
python3 swarm2_main.py "Your question here"
python3 swarm2_main.py --interactive
python3 swarm_api_server.py --port 5002
```

## Key Environment Variables

```bash
OLLAMA_BASE_URL=http://localhost:11434   # Local Ollama LLM server (default)
SEARXNG_URL=http://localhost:8080        # Self-hosted search engine (preferred)
SWARM_SERVER=http://10.0.0.58:5002      # Remote swarm server (for run_me.py)
SWARM_API_KEY=<key>                      # Optional bearer token auth
TAVILY_API_KEY=tvly-xxx                  # Optional deep search API
RAPIDAPI_KEY=<key>                       # Amazon product search (project mode)
```

## Directory Layout

```
/mnt/storage/NAS/Jarvis/swarm/
├── run_me.py               ← client entry point (connect to remote server)
├── swarm2_main.py          ← legacy local entry point
├── _paths.py               ← sys.path helper (imported by all entry points)
├── ollama-swarm.service    ← systemd unit (installed at /etc/systemd/system/)
├── setup.sh
├── .gitignore
│
├── core/                   ← agents, search, shared infrastructure
│   ├── base_agent.py           BaseAgent (async Ollama HTTP)
│   ├── core.py                 SharedMemory base + AgentType enums
│   ├── shared_memory.py        Extended shared memory
│   ├── messages.py             Fact/message types
│   ├── question_classifier.py  THEORETICAL/MATHEMATICAL/HYBRID/ENGINEERING_DESIGN
│   ├── planner_agent.py        Sub-question decomposition
│   ├── writer_agent.py         Final synthesis (qwen2.5:14b)
│   ├── consensus_agent.py      Cross-source fact checking
│   ├── flexible_search_agent.py  SearXNG → DDG → Google fallback
│   ├── search_parallel.py      Parallel search executor
│   ├── checklist_system.py     Research checklist tracking
│   ├── value_extractor.py      Numeric value + unit extraction (115 patterns)
│   ├── status_display.py       Live terminal status board
│   └── verifier_agent.py       Claim verification
│
├── compute/                ← equation pipeline (was math/ — renamed to avoid stdlib shadow)
│   ├── equation_builder_agent.py
│   ├── equation_generator.py
│   ├── equation_validator.py
│   ├── python_compute.py
│   ├── variable_mapper_agent.py
│   ├── formal_calculator.py
│   ├── math_verifier.py
│   ├── material_props.py
│   └── physics_supervisor.py
│
├── engineer/               ← design / electronics / firmware
│   ├── engineer_mode.py
│   ├── engineering_defaults.py
│   ├── electronics_engine.py
│   └── firmware_generator.py
│
├── server/                 ← orchestrator + REST API + project mode
│   ├── orchestrator_v2_1.py
│   ├── swarm_api_server.py
│   ├── project_mode.py
│   ├── project_session.py
│   ├── project_chat.py
│   ├── deep_search_api.py
│   └── safety_db.json
│
├── DOCS/
│   ├── README.md
│   ├── API_REFERENCE.md
│   ├── INTEGRATION_GUIDE.md
│   └── openapi.yaml
│
└── tests/
    └── test_math_stress.py
```

## Architecture

### Execution Pipeline

1. **Classification** (`core/question_classifier.py`) — THEORETICAL, MATHEMATICAL, HYBRID, ENGINEERING_DESIGN
2. **Planning** (`core/planner_agent.py`) — Breaks the question into sub-questions
3. **Research** (`core/flexible_search_agent.py`) — Parallel web searches: SearXNG → DuckDuckGo → Google
4. **Variable Extraction** (`core/value_extractor.py`, `compute/variable_mapper_agent.py`) — Numeric values + units
5. **Equation System** (`compute/equation_builder_agent.py`, `compute/equation_generator.py`) — Complete Python
6. **Computation** (`compute/python_compute.py`) — Deterministic SymPy + Pint; **never LLM-based**
7. **Validation** (`core/consensus_agent.py`, `compute/equation_validator.py`) — Cross-source fact checking
8. **Synthesis** (`core/writer_agent.py`) — Final answer with citations

The orchestrator is `server/orchestrator_v2_1.py`. The API server is `server/swarm_api_server.py`.

### Import Strategy

All 44 files use flat imports (`from base_agent import BaseAgent`). The `_paths.py` file and `__init__.py` files in each subpackage add the necessary directories to `sys.path` so existing imports work unchanged.

**KEY NOTE**: The equation/math subpackage is named `compute/` (NOT `math/`). The name `math/` would shadow Python's stdlib `math` module and break Flask/werkzeug.

### Shared Memory

All agents communicate through `SharedMemory` (defined in `core/core.py`, extended in `core/shared_memory.py`). Every fact includes source URL, agent name, and timestamp.

### Deployment

- **systemd**: `ollama-swarm.service` runs `server/swarm_api_server.py` on port 5002
- **Client**: `run_me.py` connects to `http://10.0.0.58:5002`
- **Local**: `swarm2_main.py` runs the full pipeline locally

### Key Design Constraints

- **Deterministic computation**: Math always via SymPy/Pint, never LLM
- **Graceful degradation**: Search falls back; computation falls back to simpler methods
- **No requirements.txt**: Dependencies in `setup.sh`, installed manually
- **Zero-touch imports**: No existing imports changed; `_paths.py` handles sys.path

---

## Feature Manifest (updated each commit — verify these files exist before committing)

### Orchestration (server/)
| Feature | File | What it does |
|---|---|---|
| OrchestratorV3 | `server/orchestrator_v3.py` | Top-level dispatcher: HYBRID/MATH→ReAct, THEORETICAL→V2_1, ENGINEERING→engineer_mode |
| OrchestratorV2_1 | `server/orchestrator_v2_1.py` | Legacy orchestrator for THEORETICAL questions; delegated to by V3 |
| API Server | `server/swarm_api_server.py` | Flask REST API on :5002; /query_async, /query_stream (SSE), /result, /jobs, /logs, /metrics, /dashboard |
| Dashboard SPA | `server/dashboard.html` | Command Station web UI; phases/SPs/GPU/jobs widgets; zero CDN deps |
| Project Mode | `server/project_mode.py` | Multi-phase hardware project planner (BOM, safety, firmware) |
| Project Session | `server/project_session.py` | Stateless HTTP session wrapper for project Q&A; 2h TTL |

### ReAct Solver Pipeline (compute/)
| Feature | File | What it does |
|---|---|---|
| ReactSolver | `compute/react_solver.py` | Per-SP ReAct loop (qwen2.5-coder:14b, MAX_TURNS=15); tool dispatch: run_code/search/rag |
| Context Anchor | `compute/react_solver.py` | Injects problem given-values box + FORBIDDEN constants at top of every SP system prompt |
| Locked Results Ledger | `compute/react_solver.py` | `_locked_results` dict: RESULT: lines captured from code, re-injected each turn, never dropped |
| Force-inject given values | `compute/react_solver.py` | SP.inputs prepended to every code block as `# === GIVEN VALUES ===` constants |
| Loop detection | `compute/react_solver.py` | Breaks early if 3 consecutive turns have identical-length responses (stuck loop) |
| [LLMTOK] streaming | `compute/react_solver.py` | Batches tokens into `[LLMTOK]escaped` print lines for live terminal streaming |
| RAG tool | `compute/rag_tool.py` | ChromaDB wrapper (BAAI/bge-large-en-v1.5); `rag_search(query, domain, n)` |

### Planning (core/)
| Feature | File | What it does |
|---|---|---|
| PlannerV2 | `core/planner_v2.py` | Decomposes question into SolvePlan with dependency-ordered SubProblems |
| Requirement Shredder (Lock A) | `core/planner_v2.py` | Extracts all distinct requirements before planning; enforces 1:1 SP mapping |
| Question Classifier | `core/question_classifier.py` | THEORETICAL/MATHEMATICAL/HYBRID/ENGINEERING_DESIGN; CRITICAL RULES prevent numerical→THEORETICAL misclassification |
| THEORETICAL→HYBRID override | `server/orchestrator_v3.py` | Regex safety net: if question has numeric assignments + compute verbs, upgrades classification |

### Safety Locks (Prometheus)
| Feature | File | What it does |
|---|---|---|
| Lock A: Requirement Shredder | `core/planner_v2.py` | 1:1 SP-to-requirement mapping; warns if planner drops requirements |
| Lock B: Code Enforcement | `compute/react_solver.py` | FINAL_ANSWER before first run_code → REJECTED; Rule 0 in system prompt |
| Lock C: Negative Constraint Filter | `server/orchestrator_v3.py` | `_enforce_negative_constraints()`: scans for violations (no_formulas etc), rewrites offending sections |
| Writer Gag | `server/orchestrator_v3.py` | Writer receives ONLY verified RESULT: lines; never sees question text with numbers |
| Phase 3C fallback | `server/orchestrator_v3.py` | If constraint rewrite fails/times out, returns original answer untouched |

### Client (run_me.py)
| Feature | File | What it does |
|---|---|---|
| Token streaming (-t / :tokens) | `run_me.py` | `cmd_stream_tokens()`: color-coded live LLM output; THOUGHT=dim, ACTION=purple, RESULT=green |
| Rich Live panel | `run_me.py` | `_cmd_stream_rich()`: phases/SPs/wave tracker using rich.live.Live |
| Watch command (:watch) | `run_me.py` | `cmd_watch()`: tail-f style reconnect to running job via disk log polling |
| Logs command (:logs) | `run_me.py` | Fetch full disk log for any job (tail=N, grep=pat supported) |
| Result command (:result) | `run_me.py` | Fetch completed job answer; :results alias works too |
| Disk persistence | `server/swarm_api_server.py` | Final answer written to `swarm_results/<job_id>.log` immediately on completion |
| /result fallback | `server/swarm_api_server.py` | `get_result()` reads from disk log when job not in memory (survives restarts) |

### Infrastructure
| Feature | File | What it does |
|---|---|---|
| base_agent 600s timeout | `base_agent.py` | requests.post timeout=600 so qwen2.5 has time to load after deepseek unloads |
| VRAM handoff prints | `server/orchestrator_v3.py` | Prints `🔄 VRAM handoff: phi4→deepseek` at model transitions |
| _ProgressRouter | `server/swarm_api_server.py` | Captures stdout per-job; routes to SSE queue + disk log |
| JobAwareExecutor | `server/swarm_api_server.py` | Thread pool that stamps job_id before async tasks (fixes routing for run_in_executor) |
| /metrics endpoint | `server/swarm_api_server.py` | Returns GPU (nvidia-smi) + CPU/RAM (psutil) JSON for dashboard widget |
