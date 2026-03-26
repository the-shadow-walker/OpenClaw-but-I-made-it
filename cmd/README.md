# Jarvis — `cmd/`

The core AI agent service. Accepts natural-language instructions, uses a local LLM (Ollama) to plan and execute shell commands, with safety validation at every step.

**Service:** `ollama-cmd` (systemd) — runs on port `5000`
**Models:** `qwen2.5-coder:14b` (fast/ReAct loop) · `qwen3-coder:30b` (code generation)

---

## Quick Start

```bash
# Ask a question / run a task
python run_me.py "list all open ports on this machine"

# Check service health
python run_me.py --health

# Run a multi-phase chain (complex goals)
python run_me.py --chain "build a FastAPI app with a SQLite database"

# SENTINEL security scan
python run_me.py --scan

# Get today's security report
python run_me.py --report
```

---

## Directory Structure

```
cmd/
  core/                  Agent internals
    ollama_agent_core.py   OllamaCommandAgent — ReAct loop, safety validator
    react_tools.py         ToolRegistry — 8 tools, stuck-loop guard, patch counter
    react_memory.py        AgentMemory — SQLite, system survey cache, runbooks

  chain/                 Multi-phase task orchestration
    task_chain.py          TaskDecomposer, SubtaskOrchestrator, TaskChain state

  blueteam/              SENTINEL defensive security agent
    blueteam_agent.py      BlueteamAgent, BlueteamToolRegistry, watch loop

  infra/                 Service infrastructure
    debug_logger.py        Structured JSON + text debug logs
    webhook_dispatcher.py  Fire-and-forget outbound webhooks
    agent-watchdog         Bash script for systemd OnFailure= integration
    sysinfo.sh             System information helper

  DOCS/                  Documentation
    integration_guide.md   How to integrate with the agent API
    EVENT_MESH.md          Event mesh / SSE architecture notes

  run_me.py              ← The only client you need
  README.md              This file
```

---

## `run_me.py` — Full Command Reference

```
python run_me.py "question"              Ask anything, streams response
python run_me.py --chain "goal"          Multi-phase chain (complex builds)
python run_me.py --budget 300            Chain iteration budget (default 200)
python run_me.py --no-stream             Poll instead of stream

Status
  --health                               Service version, active jobs, features
  --jobs                                 List 20 most recent jobs
  --job <id>                             Full output of a specific job
  --cancel <id>                          Cancel a running job
  --chains                               List all chains
  --chain-status <id>                    Phase-by-phase chain breakdown

SENTINEL Security
  --sentinel                             Watcher status + last report summary
  --scan                                 Run a full security scan (streams)
  --scan-focus "SSH"                     Focused scan on a specific area
  --report                               Today's full .md security report
  --alerts                               Recent security alerts
```

Set `JARVIS_URL` env var to change target (default `http://10.0.0.58:5000`).

---

## REST API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/health` | Health check |
| POST | `/api/v1/execute` | Submit a job `{"instruction": "..."}` |
| GET  | `/api/v1/jobs/<id>` | Job status + output |
| GET  | `/api/v1/jobs/<id>/stream` | SSE output stream |
| GET  | `/api/v1/jobs` | List jobs |
| DELETE | `/api/v1/jobs/<id>` | Cancel job |
| POST | `/api/v1/chains` | Submit chain `{"goal": "...", "total_budget": 200}` |
| GET  | `/api/v1/chains/<id>` | Chain state + subtask statuses |
| GET  | `/api/v1/blueteam/report` | Current SENTINEL daily report (.md) |
| POST | `/api/v1/blueteam/scan` | Trigger security scan |
| GET  | `/api/v1/blueteam/alerts` | Recent security alerts |
| GET  | `/api/v1/blueteam/status` | SENTINEL watcher status |

No authentication required (local network deployment).

See `DOCS/integration_guide.md` for integration examples.

---

## Service Management

```bash
# Status
sudo systemctl status ollama-cmd

# Restart
sudo systemctl restart ollama-cmd

# Logs (live)
sudo journalctl -u ollama-cmd -f

# Security report location
~/.agent_bin/sentinel_report.md
~/.agent_bin/sentinel_archive/   ← daily archives
```
