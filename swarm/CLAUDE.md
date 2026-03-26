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
