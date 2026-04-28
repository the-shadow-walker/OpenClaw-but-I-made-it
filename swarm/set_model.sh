#!/usr/bin/env bash
# set_model.sh — single-command model swap across all swarm roles.
#
# Usage:
#   ./set_model.sh qwen3.6:35b-Chain          # switch all 4 roles
#   ./set_model.sh batiai/qwen3.6-27b:iq4     # roll back to 27B
#   ./set_model.sh                            # show current models
#
# Updates:
#   1. swarm_results/model_config.json  (4 roles: planner/classifier/writer/solver)
#   2. /etc/systemd/system/ollama-swarm.service  (SWARM_MODEL_DEFAULT/_SOLVER/_PLANNER)
#   3. systemd daemon-reload + ollama-swarm restart
#
# Diagnostician (compute/react_solver.py DIAGNOSTICIAN_MODEL) is intentionally
# left as a *different* model so it provides a fresh second opinion at turn 14.
# Set SWARM_DIAGNOSTICIAN_MODEL in systemd to override.

set -euo pipefail

SWARM_DIR="/mnt/storage/NAS/Jarvis/swarm"
CONFIG_FILE="$SWARM_DIR/swarm_results/model_config.json"
UNIT_FILE="/etc/systemd/system/ollama-swarm.service"

# ── Show current state if no arg ───────────────────────────────────────────
if [[ $# -eq 0 ]]; then
    echo "Current model_config.json:"
    cat "$CONFIG_FILE" 2>/dev/null || echo "  (file missing)"
    echo
    echo "systemd env vars:"
    grep "SWARM_MODEL" "$UNIT_FILE" 2>/dev/null || true
    echo
    echo "Live (from /config/models endpoint):"
    curl -s http://localhost:5002/config/models 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  (server unreachable)"
    exit 0
fi

NEW_MODEL="$1"

# ── Safety: refuse to swap while jobs are running ──────────────────────────
RUNNING=$(curl -s http://localhost:5002/status 2>/dev/null \
          | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['jobs']['running']+d['jobs']['processing'])" 2>/dev/null || echo 0)

if [[ "$RUNNING" -gt 0 ]]; then
    echo "🚨 $RUNNING job(s) running. Restart will kill them."
    read -rp "Force restart anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

# ── 1. Persisted config (overrides env at startup) ─────────────────────────
cat > "$CONFIG_FILE" <<EOF
{
  "planner": "$NEW_MODEL",
  "classifier": "$NEW_MODEL",
  "writer": "$NEW_MODEL",
  "solver": "$NEW_MODEL"
}
EOF
echo "✓ Wrote $CONFIG_FILE"

# ── 2. systemd unit env vars (fallback when JSON missing) ──────────────────
sudo sed -i \
    -e "s|^Environment=SWARM_MODEL_DEFAULT=.*|Environment=SWARM_MODEL_DEFAULT=$NEW_MODEL|" \
    -e "s|^Environment=SWARM_MODEL_SOLVER=.*|Environment=SWARM_MODEL_SOLVER=$NEW_MODEL|" \
    -e "s|^Environment=SWARM_MODEL_PLANNER=.*|Environment=SWARM_MODEL_PLANNER=$NEW_MODEL|" \
    "$UNIT_FILE"
echo "✓ Updated $UNIT_FILE"

# ── 3. Reload + restart ────────────────────────────────────────────────────
sudo systemctl daemon-reload
sudo systemctl restart ollama-swarm
echo "✓ ollama-swarm restarted"

# ── Verify ─────────────────────────────────────────────────────────────────
sleep 3
echo
echo "Live models after restart:"
curl -s http://localhost:5002/config/models 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  {r}: {v[\"model\"]}') for r,v in d['current'].items()]" \
    2>/dev/null || echo "  (server still starting)"
