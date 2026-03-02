#!/bin/bash
# sysinfo.sh — comprehensive system snapshot for the agent
# Usage: bash ~/cmd/sysinfo.sh

SEP="======================================================================"

section() { echo; echo "$SEP"; echo "  $1"; echo "$SEP"; }

section "IDENTITY"
echo "User:     $(whoami)"
echo "Home:     $HOME"
echo "Shell:    $SHELL"
echo "Hostname: $(hostname -f 2>/dev/null || hostname)"
echo "Date:     $(date)"

section "OS / KERNEL"
uname -a
grep -E "^(NAME|VERSION|ID)=" /etc/os-release 2>/dev/null || sw_vers 2>/dev/null

section "HARDWARE"
echo "--- CPU ---"
lscpu 2>/dev/null | grep -E "Model name|CPU\(s\)|Thread|Core|Socket" || sysctl -n machdep.cpu.brand_string 2>/dev/null
echo
echo "--- Memory ---"
free -h 2>/dev/null || vm_stat 2>/dev/null | head -10
echo
echo "--- Disk ---"
df -h

section "NETWORK"
echo "--- Interfaces ---"
ip addr show 2>/dev/null | grep -E "^[0-9]+:|inet " | head -30

echo
echo "--- Listening ports ---"
ss -tulnp 2>/dev/null | grep LISTEN | sort -t: -k2 -n

echo
echo "--- Firewall (nft) ---"
sudo nft list ruleset 2>/dev/null \
  | grep -E "chain|tcp dport|udp dport|ip saddr|drop|accept" \
  | head -40 \
  || echo "(no nft access or not installed)"

section "SERVICES"
echo "--- Running ---"
systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null \
  | awk '{print $1, $4}' | head -30

echo
echo "--- Failed ---"
systemctl list-units --state=failed --no-pager --no-legend 2>/dev/null || echo "(none)"

section "LANGUAGES & RUNTIMES"
for cmd in "python3 --version" "python --version" "node --version" "npm --version" \
           "rustc --version" "cargo --version" "go version" \
           "java -version" "javac -version" "ruby --version" \
           "php --version" "perl --version"; do
    bin=$(echo $cmd | awk '{print $1}')
    if command -v "$bin" &>/dev/null; then
        echo "$($cmd 2>&1 | head -1)"
    fi
done

section "PYTHON ENVIRONMENT"
echo "Active python: $(which python3)"
echo "Venv: ${VIRTUAL_ENV:-none}"
if command -v pip3 &>/dev/null; then
    echo "Installed packages: $(pip3 list 2>/dev/null | tail -n +3 | wc -l)"
    echo "--- Key packages ---"
    pip3 list 2>/dev/null | grep -iE "fastapi|flask|django|sqlalchemy|alembic|psycopg|asyncpg|uvicorn|pydantic|jose|passlib|bcrypt|requests|httpx" | head -20
fi

section "NODE ENVIRONMENT"
if command -v node &>/dev/null; then
    echo "Node: $(node --version)"
    echo "npm:  $(npm --version 2>/dev/null)"
    if [ -f package.json ]; then
        echo "--- package.json deps ---"
        python3 -c "import json,sys; d=json.load(open('package.json')); [print(f'  {k}: {v}') for k,v in {**d.get('dependencies',{}), **d.get('devDependencies',{})}.items()]" 2>/dev/null
    fi
fi

section "DATABASES"
for db in postgresql mysql mariadb mongodb redis; do
    if systemctl is-active --quiet $db 2>/dev/null; then
        echo "$db: running"
    elif command -v $db &>/dev/null || command -v ${db}d &>/dev/null; then
        echo "$db: installed but not running"
    fi
done
if command -v psql &>/dev/null; then
    echo "PostgreSQL client: $(psql --version)"
fi
if command -v redis-cli &>/dev/null; then
    echo "Redis: $(redis-cli ping 2>/dev/null || echo 'not running')"
fi

section "DOCKER / CONTAINERS"
if command -v docker &>/dev/null; then
    echo "Docker: $(docker --version)"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | head -20
else
    echo "Docker: not installed"
fi

section "NGINX / WEB SERVER"
if command -v nginx &>/dev/null; then
    echo "nginx: $(nginx -v 2>&1)"
    echo "Config test: $(sudo nginx -t 2>&1 | tail -1)"
    echo "--- Sites/configs ---"
    ls /etc/nginx/sites-enabled/ 2>/dev/null || ls /etc/nginx/conf.d/ 2>/dev/null || echo "(no sites dir found)"
fi

section "FILESYSTEM LAYOUT (home)"
du -sh "$HOME"/*(N) "$HOME"/.[^.]*/(N) 2>/dev/null | sort -h | tail -20 \
  || du -sh "$HOME"/* 2>/dev/null | sort -h | tail -20

section "RECENT PROCESSES (top 10 by CPU)"
ps aux --sort=-%cpu 2>/dev/null | head -11 || ps aux | head -11

echo
echo "$SEP"
echo "  END OF SYSINFO"
echo "$SEP"
