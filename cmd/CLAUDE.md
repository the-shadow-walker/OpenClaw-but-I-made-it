# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Key Rules
- **Always edit locally on Mac** (`/Users/grant/cmd/`), then SCP to remote. Never edit on remote directly.
- **Never push to `~/swarm3`** — swarm lives at `/mnt/storage/NAS/Jarvis/swarm/`
- **Git root** is `/mnt/storage/NAS/Jarvis/` — commit from there
- **Service name** is `ollama-cmd` (systemd unit: `ollama-cmd.service`)

## Directory Structure

```
/mnt/storage/NAS/Jarvis/
  server.py          ← root-level entry point (root-owned, deploy via /tmp/)
  cmd/
    core/            ollama_agent_core.py, react_tools.py, react_memory.py
    chain/           task_chain.py
    blueteam/        blueteam_agent.py
    infra/           debug_logger.py, webhook_dispatcher.py, watchdog, sysinfo
    DOCS/            integration_guide.md, EVENT_MESH.md
    run_me.py        single client entry point
    README.md
  swarm/             AI agent swarm (moved from ~/swarm3)
  agent_inbox/       drop JSON files here for inbox watcher
  logs/
```

## Running the Service

```bash
# Service management
sudo systemctl status ollama-cmd
sudo systemctl restart ollama-cmd
sudo journalctl -u ollama-cmd -f

# Run directly (dev)
source .venv/bin/activate
python server.py
```

## SSH / Deployment

```bash
# SSH alias
mcssh   =   ssh -i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network' Grindlewalt@mcshell.atomos.network

SCP_OPTS="-i ~/.ssh/mcssh -o 'ProxyCommand=cloudflared access ssh --hostname ssh.atomos.network'"

# server.py — root-owned, must stage via /tmp/
scp $SCP_OPTS server.py Grindlewalt@mcshell.atomos.network:/tmp/
mcssh "sudo cp /tmp/server.py /mnt/storage/NAS/Jarvis/server.py && sudo cp /tmp/server.py /mnt/storage/NAS/Jarvis/cmd/server.py"

# All other cmd/ files — SCP directly to subpackage paths
scp $SCP_OPTS cmd/core/ollama_agent_core.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/core/
scp $SCP_OPTS cmd/core/react_tools.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/core/
scp $SCP_OPTS cmd/chain/task_chain.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/chain/
scp $SCP_OPTS cmd/blueteam/blueteam_agent.py Grindlewalt@mcshell.atomos.network:/mnt/storage/NAS/Jarvis/cmd/blueteam/
```

## Git Workflow

```bash
# Commit from the Jarvis root (git root is there)
mcssh "cd /mnt/storage/NAS/Jarvis && git add -A && git commit -m 'description' && git push"

# server.py is now user-owned after sudo cp — git add works without sudo
```

Git remote: `http://10.0.0.58:3000/Grindlewalt/Jarvis.git`

## Architecture Notes

- `server.py` adds `cmd/`, `cmd/core/`, `cmd/chain/`, `cmd/blueteam/`, `cmd/infra/` to sys.path — all flat imports inside modules continue working without changes
- Fast model: `qwen2.5-coder:14b` (ReAct loop, 60s timeout)
- Heavy model: `qwen3-coder:30b` (create_file/patch_file content, 300s timeout)
- Service port: `5000`, no auth (local network)
- SENTINEL auto-watch: disabled at boot, triggered by cron at 3 AM daily
- Daily report: `~/.agent_bin/sentinel_report.md`, archives in `~/.agent_bin/sentinel_archive/`
