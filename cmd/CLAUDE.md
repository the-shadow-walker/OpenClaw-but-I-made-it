# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Service

```bash
# Activate the virtual environment first
source .venv/bin/activate

# Start the REST API server (port 5000)
python server.py

# Run the agent directly (interactive mode)
python ollama_agent_core.py

# Use the client library from a Python script
python agent_client.py
```

**Dependencies:** Flask, Flask-CORS, requests — all installed in `.venv`.

**External requirements:**
- Ollama running locally at `http://localhost:11434` with model `qwen3-coder:30b`
- SearXNG instance at `http://10.0.0.58:8080` (optional, for web search)
- Set `AGENT_API_KEY` env var or a random key is generated on startup

## Architecture

This is a Python-based AI agent service that accepts natural language instructions, uses a local LLM (via Ollama) to plan and execute shell commands, with safety validation at each step.

### Three-tier structure

```
agent_client.py       → HTTP client library to interact with the service
server.py             → Flask REST API; async job queue with 3 worker threads
ollama_agent_core.py  → Core agent: LLM-driven planning, command execution, safety validation
agent_service.py      → Older/alternative version of server.py (mostly superseded)
```

### Job lifecycle

Jobs move through states: `QUEUED → RUNNING → COMPLETED / FAILED / CANCELLED`. Jobs are stored in memory — they are lost on restart.

### Execution flow

1. Client POSTs instruction to `/api/v1/execute`, gets back a `job_id`
2. A worker thread picks up the job and calls `OllamaCommandAgent.run()`
3. Agent calls Ollama to analyze the task and create a step-by-step plan
4. Each step is validated by `CommandSafetyValidator` before execution
5. On failure, agent asks LLM to analyze and retry (up to 3x)
6. Client polls `/api/v1/jobs/<job_id>` or streams via SSE at `/api/v1/jobs/<job_id>/stream`

### Safety validation (`CommandSafetyValidator` in `ollama_agent_core.py`)

All commands are assigned a risk level before execution:
- **Blocked**: Rejected outright (fork bombs, `rm -rf /`, writing to `/dev/sd*`, `mkfs`, etc.)
- **High/Medium**: Requires user confirmation and LLM-generated explanation
- **Low/Safe**: Executed directly

Protected paths (writes blocked): `/bin`, `/boot`, `/dev`, `/etc`, `/lib`, `/proc`, `/root`, `/sbin`, `/sys`, `/usr`

### Key classes

- `OllamaCommandAgent` — orchestrates LLM calls, plan creation, step execution, and retry/recovery
- `CommandSafetyValidator` — validates commands against risk patterns before execution
- `FlexibleSearchAgent` — web search via SearXNG
- `JobRunner` (server.py) — thread pool managing up to 3 concurrent jobs
- `OutputCapture` (server.py) — captures stdout/stderr during job execution
- `AgentClient` (agent_client.py) — HTTP client wrapping all API endpoints

## Deployment to Remote Server (mcssh)

**After making any changes, always SCP the modified files to the remote server.**

The remote server is accessed via the `mcssh` alias (`mcshell.atomos.network`).
Project lives at `/mnt/storage/NAS/Jarvis/` with a symlink `~/cmd → /mnt/storage/NAS/Jarvis`.
Git remote: `http://10.0.0.58:3000/Grindlewalt/Jarvis.git`

Only `server.py` is root-owned — it must be staged via `/tmp/` then `sudo cp`'d. All other files can be SCP'd directly to `~/cmd/`.

```bash
SCP_OPTS="-i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname mcshell.atomos.network'"

# server.py only — stage via /tmp/
scp $SCP_OPTS server.py Grindlewalt@mcshell.atomos.network:/tmp/
mcssh "sudo cp /tmp/server.py ~/cmd/"

# All other files — copy directly
scp $SCP_OPTS ollama_agent_core.py react_tools.py task_chain.py Grindlewalt@mcshell.atomos.network:~/cmd/
```

## Git Workflow

```bash
# On the server (~/cmd is symlinked to /mnt/storage/NAS/Jarvis)
cd ~/cmd
git add ollama_agent_core.py react_tools.py   # or specific files
git commit -m "description of change"
git push
```

Note: `server.py` is root-owned so `git add server.py` must be run as root or the file needs to be chowned first.

## REST API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Service health check |
| POST | `/api/v1/execute` | Submit a job (`{"instruction": "..."}`) |
| GET | `/api/v1/jobs/<job_id>` | Get job status and output |
| GET | `/api/v1/jobs/<job_id>/stream` | Stream output via SSE |
| GET | `/api/v1/jobs` | List all jobs |
| DELETE | `/api/v1/jobs/<job_id>` | Cancel a job |
| GET | `/api/v1/config` | Get service configuration |
| POST | `/api/v1/chains` | Decompose + run a multi-phase goal (`{"goal": "...", "total_budget": 200}`) |
| GET | `/api/v1/chains/<chain_id>` | Full chain state + subtask statuses |
| GET | `/api/v1/chains` | List all chains |
| DELETE | `/api/v1/chains/<chain_id>` | Cancel a chain |
