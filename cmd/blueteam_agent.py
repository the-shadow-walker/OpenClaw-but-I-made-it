"""
blueteam_agent.py — SENTINEL: Defensive Cybersecurity Agent

The "blue team of all blue teams": a continuously watching, log-reading,
process-monitoring, threat-detecting defensive agent powered by the local LLM.
It knows everything that's running, reads suspicious things, stops odd
programs, and reads all logs so it always knows what's happening.

Modes
─────
  One-shot scan:       BlueteamAgent().scan()
  Targeted inquiry:    BlueteamAgent().investigate("suspicious SSH traffic")
  Continuous watch:    BlueteamAgent().watch(interval=60)

REST API (added to server.py)
──────────────────────────────
  POST /api/v1/blueteam/scan          — full security scan + threat report
  POST /api/v1/blueteam/investigate   — targeted investigation of a finding
  POST /api/v1/blueteam/watch/start   — start background monitor
  POST /api/v1/blueteam/watch/stop    — stop background monitor
  GET  /api/v1/blueteam/alerts        — recent security alerts
  GET  /api/v1/blueteam/status        — watcher status + last scan summary

Inbox drop (works with existing inbox watcher)
───────────────────────────────────────────────
  echo '{"blueteam_scan": true}' > agent_inbox/scan.json
  echo '{"blueteam_investigate": "suspicious ssh connection from 1.2.3.4"}' \
    > agent_inbox/investigate.json
"""

import json
import os
import re
import selectors
import signal as _signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ollama_agent_core import OllamaCommandAgent
from react_tools import ToolRegistry, ToolResult

try:
    import debug_logger as _dlog
except ImportError:
    _dlog = None

try:
    import webhook_dispatcher as _webhook
except ImportError:
    _webhook = None

# ─────────────────────────── constants ──────────────────────────────────────

_ALERTS_LOG = Path("./logs/blueteam_alerts.jsonl")
_ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)

# PIDs / names that must never be killed
_PROTECTED_NAMES = frozenset({
    "systemd", "init", "sshd", "kthreadd", "migration",
    "ksoftirqd", "kworker", "watchdog",
})

# Private IP prefixes that must never be blocked
_PRIVATE_PREFIXES = ("127.", "10.", "192.168.", "::1", "fe80")

# MOTD paths
_MOTD_SYSTEM   = Path("/etc/motd")
_MOTD_FALLBACK = Path("~/.motd_sentinel").expanduser()

# wall alert config
_WALL_SEVERITIES = frozenset({"HIGH", "CRITICAL"})
_WALL_WARNED: bool = False

# arch-audit warning gate
_ARCH_AUDIT_WARNED: bool = False

# persistent report path
_REPORT_PATH = Path("~/.agent_bin/sentinel_report.md").expanduser()

# debug log
_DBG_LOG_PATH = Path("~/.agent_bin/sentinel_debug.log").expanduser()
_DBG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_dbg_lock = threading.Lock()


def _dbg(tag: str, msg: str = "") -> None:
    """Append a timestamped debug entry to sentinel_debug.log (max 5 MB, then rotate)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{tag}] {msg}\n"
    try:
        with _dbg_lock:
            if _DBG_LOG_PATH.exists() and _DBG_LOG_PATH.stat().st_size > 5 * 1024 * 1024:
                _DBG_LOG_PATH.rename(_DBG_LOG_PATH.with_suffix(".log.1"))
            with open(_DBG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass
    # also print so it shows up in journalctl
    print(f"🔵 [{tag}] {msg}")


# ─────────────────────────── wall broadcast ──────────────────────────────────

def _wall_alert(severity: str, finding: str) -> None:
    """Broadcast a HIGH/CRITICAL alert to all logged-in users via wall(1)."""
    global _WALL_WARNED
    msg = (
        f"[SENTINEL ALERT] {severity}: {finding[:100]} "
        f"— run: ./agent_client.py --blueteam-alerts"
    )
    try:
        subprocess.run(['wall', msg], capture_output=True, timeout=5)
    except FileNotFoundError:
        if not _WALL_WARNED:
            _WALL_WARNED = True
            print("WARNING: 'wall' not found — broadcast alerts disabled")
    except Exception:
        pass


# ─────────────────────────── alert emission ──────────────────────────────────

_recent_alerts: List[Dict] = []
_alerts_lock = threading.Lock()


def emit_alert(severity: str, finding: str,
               evidence: str = "", action: str = "") -> None:
    """Central alert emitter — logs to file, debug_logger, webhooks, and memory."""
    entry = {
        "ts": datetime.now().isoformat(),
        "severity": severity.upper(),
        "finding": finding,
        "evidence": evidence[:500],
        "action_taken": action,
    }
    with _alerts_lock:
        _recent_alerts.append(entry)
        if len(_recent_alerts) > 500:
            _recent_alerts.pop(0)

    # Persist
    try:
        with open(_ALERTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

    # Event mesh
    if _dlog:
        _dlog.log("security_alert", entry)
    if _webhook:
        _webhook.dispatch("security_alert", entry)

    print(f"🚨 [{severity.upper()}] {finding}")
    _dbg("ALERT", f"[{severity.upper()}] {finding[:120]}")

    # Broadcast HIGH/CRITICAL to all terminals
    if severity.upper() in _WALL_SEVERITIES:
        _wall_alert(severity.upper(), finding)


def get_recent_alerts(n: int = 50) -> List[Dict]:
    """Return the last n alerts from in-memory list + log file."""
    file_alerts: List[Dict] = []
    try:
        with open(_ALERTS_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    file_alerts.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    combined = file_alerts + _recent_alerts
    seen = set()
    deduped = []
    for a in combined:
        key = (a.get("ts"), a.get("finding"))
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped[-n:]


# ─────────────────────── BlueteamToolRegistry ────────────────────────────────

_BLUETEAM_EXTRA_SCHEMAS = {
    "alert": (
        '  alert            — {"severity": "LOW|MEDIUM|HIGH|CRITICAL", '
        '"finding": str, "evidence": str, "recommended_action": str}'
    ),
    "kill_process": (
        '  kill_process     — {"pid": int, "signal": "TERM|KILL", "reason": str}'
    ),
    "block_ip": (
        '  block_ip         — {"ip": str, "direction": "in|out|both", "reason": str}'
    ),
    "quarantine_file": (
        '  quarantine_file  — {"path": str, "reason": str}'
    ),
}


class BlueteamToolRegistry(ToolRegistry):
    """Extends ToolRegistry with four defensive security tools."""

    BLUETEAM_TOOL_NAMES = ToolRegistry.TOOL_NAMES | {
        "alert", "kill_process", "block_ip", "quarantine_file",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TOOL_NAMES = self.BLUETEAM_TOOL_NAMES

    # ── alert ────────────────────────────────────────────────────────────────

    def _handle_alert(self, args: dict) -> ToolResult:
        severity = str(args.get("severity", "MEDIUM")).upper()
        finding = str(args.get("finding", ""))
        evidence = str(args.get("evidence", ""))[:500]
        action = str(args.get("recommended_action", ""))
        if not finding:
            return ToolResult(False, "", "alert requires 'finding'", {})
        emit_alert(severity, finding, evidence, action)
        return ToolResult(True, f"Alert [{severity}] recorded: {finding}", "", {
            "severity": severity,
        })

    # ── kill_process ─────────────────────────────────────────────────────────

    def _handle_kill_process(self, args: dict) -> ToolResult:
        try:
            pid = int(args.get("pid", 0))
        except (TypeError, ValueError):
            return ToolResult(False, "", "kill_process requires integer 'pid'", {})

        sig_name = str(args.get("signal", "TERM")).upper()
        reason = str(args.get("reason", "No reason provided"))

        if pid <= 1:
            return ToolResult(False, "", "PID 1 (init/systemd) is protected", {})

        # Read process name from /proc for safety check and audit
        proc_name = "(unknown)"
        try:
            with open(f"/proc/{pid}/comm") as f:
                proc_name = f.read().strip()
        except FileNotFoundError:
            return ToolResult(False, "", f"PID {pid} does not exist", {})
        except PermissionError:
            proc_name = "(unreadable)"

        if any(p in proc_name for p in _PROTECTED_NAMES):
            return ToolResult(False, "", f"Process '{proc_name}' (PID {pid}) is protected", {})

        # Audit before action
        emit_alert(
            "HIGH",
            f"Killing process PID {pid} ({proc_name})",
            f"Reason: {reason}",
            f"kill -{sig_name} {pid}",
        )

        sig_map = {
            "TERM": _signal.SIGTERM,
            "KILL": _signal.SIGKILL,
            "HUP": _signal.SIGHUP,
            "INT": _signal.SIGINT,
        }
        sig_num = sig_map.get(sig_name, _signal.SIGTERM)

        try:
            os.kill(pid, sig_num)
            return ToolResult(True,
                f"Sent {sig_name} to PID {pid} ({proc_name}). Reason: {reason}",
                "", {"pid": pid, "name": proc_name})
        except PermissionError:
            r = subprocess.run(["sudo", "kill", f"-{sig_name}", str(pid)],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                return ToolResult(True,
                    f"Killed PID {pid} ({proc_name}) via sudo. Reason: {reason}",
                    "", {"pid": pid, "name": proc_name})
            return ToolResult(False, "", f"Permission denied killing PID {pid}: {r.stderr}", {})
        except ProcessLookupError:
            return ToolResult(False, "", f"PID {pid} no longer exists", {})

    # ── block_ip ─────────────────────────────────────────────────────────────

    def _handle_block_ip(self, args: dict) -> ToolResult:
        ip = str(args.get("ip", "")).strip()
        direction = str(args.get("direction", "in")).lower()
        reason = str(args.get("reason", "No reason provided"))

        if not re.match(r"^[\d.:/a-fA-F]+$", ip):
            return ToolResult(False, "", f"Invalid IP/CIDR: '{ip}'", {})

        if any(ip.startswith(p) for p in _PRIVATE_PREFIXES):
            return ToolResult(False, "",
                f"Refusing to block private/loopback address: {ip}", {})

        emit_alert("HIGH", f"Blocking IP {ip} ({direction})", reason, f"nft block {ip}")

        cmds: List[str] = []
        if direction in ("in", "both"):
            cmds.append(
                f'sudo nft add rule inet filter input ip saddr {ip} '
                f'counter drop comment "sentinel_block"'
            )
        if direction in ("out", "both"):
            cmds.append(
                f'sudo nft add rule inet filter output ip daddr {ip} '
                f'counter drop comment "sentinel_block"'
            )
        if not cmds:
            return ToolResult(False, "", f"Invalid direction '{direction}'; use in/out/both", {})

        errors: List[str] = []
        for cmd in cmds:
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=10)
            if r.returncode != 0:
                errors.append(r.stderr.strip())

        if not errors:
            return ToolResult(True, f"Blocked {ip} ({direction}) via nft. Reason: {reason}",
                "", {"ip": ip, "direction": direction})
        return ToolResult(False, "", f"nft block failed: {'; '.join(errors)}", {})

    # ── quarantine_file ──────────────────────────────────────────────────────

    def _handle_quarantine_file(self, args: dict) -> ToolResult:
        path = os.path.expanduser(str(args.get("path", "")))
        reason = str(args.get("reason", "No reason provided"))

        if not path:
            return ToolResult(False, "", "quarantine_file requires 'path'", {})
        if not os.path.exists(path):
            return ToolResult(False, "", f"File not found: {path}", {})

        qdir = os.path.expanduser("~/.agent_bin/quarantine/")
        os.makedirs(qdir, exist_ok=True)

        fname = os.path.basename(path) + f"_{int(time.time())}.quarantine"
        dest = os.path.join(qdir, fname)

        emit_alert("HIGH", f"Quarantining {path}", reason, f"mv {path} → {dest}")

        r = subprocess.run(["sudo", "mv", path, dest],
                           capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return ToolResult(False, "", f"mv failed: {r.stderr.strip()}", {})

        # Strip all permissions
        subprocess.run(["sudo", "chmod", "000", dest],
                       capture_output=True, timeout=5)

        return ToolResult(True,
            f"Quarantined {path} → {dest}. Reason: {reason}",
            "", {"original": path, "quarantine": dest})


# ─────────────────────────── system prompt ───────────────────────────────────

BLUETEAM_SYSTEM_PROMPT_TEMPLATE = """\
You are SENTINEL — an elite defensive cybersecurity analyst (Blue Team) \
operating directly on this host. Your mission: detect threats, investigate \
anomalies, and respond defensively.

HOST: {os_info}
AGENT PID (PROTECTED — NEVER KILL): {agent_pid}
ITERATION BUDGET: {max_iterations}

━━━ AVAILABLE TOOLS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{available_tools}

━━━ OUTPUT FORMAT (REQUIRED EVERY RESPONSE) ━━━━━━━━━━━━━━━━━━━━━━━━━
{{"thought": "...", "confidence": 0-100, "tool": "tool_name", "args": {{...}}}}
Never output plain text. Every response must be a single valid JSON object.

━━━ BLUE TEAM MINDSET ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Assume breach — treat every anomaly as potentially malicious until proven benign
• Evidence-first — never make claims without log/process/network evidence
• Correlate — one data point is suspicious; three make a pattern
• Conservative action — read logs BEFORE killing processes or blocking IPs

━━━ INVESTIGATION SEQUENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. OBSERVE   → run execute_command / read_file to collect hard evidence
  2. ASSESS    → weigh evidence: is this actually malicious, or normal noise?
  3. DECIDE:
       BENIGN  → call finish() with threat_level: LOW. Do NOT call alert().
       MEDIUM  → call alert(), then dig deeper before acting
       HIGH    → call alert(), then immediately contain (kill_process / block_ip)
       CRITICAL→ alert() + contain + eradicate (quarantine_file) + finish()
  4. CONTAIN   → kill_process or block_ip ONLY when evidence is conclusive
  5. ERADICATE → quarantine_file to neutralise confirmed malicious artifacts
  6. REPORT    → call finish() with your complete threat assessment

━━━ DETECTION PRIORITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Malicious processes (cryptominers, reverse shells, RATs, exfil tools)
  2. Unusual outbound connections (unexpected external IPs, high volume)
  3. Auth anomalies (brute force, new accounts, sudo abuse, SSH key changes)
  4. Persistence (new cron, systemd units, .bashrc edits, SUID changes)
  5. Lateral movement (internal SSH pivoting, ARP spoofing, port scanning)
  6. Data exfiltration (large transfers, cloud destinations, DNS tunnelling)

━━━ THREAT LEVELS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LOW      → Confirmed benign. finish() only. No alert needed.
  MEDIUM   → Genuinely suspicious with evidence. alert() + keep digging.
  HIGH     → Confirmed active threat. alert() + contain now.
  CRITICAL → Active breach. alert() + contain + eradicate immediately.

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.  Output ONLY valid JSON — never plain text or markdown
  2.  alert() is for CONFIRMED findings only — not for suspicions
  3.  NEVER kill: PID 1, systemd, sshd, ollama-agent (PID {agent_pid})
  4.  NEVER block private ranges: 127.x, 10.x, 192.168.x, 172.16-31.x
  5.  Read /proc/<pid>/cmdline and /proc/<pid>/environ before killing
  6.  Check process ancestry (ps -p PID -o ppid,cmd) before killing
  7.  confidence < 70 → gather more evidence; do not act yet
  8.  This is a KDE Plasma desktop server — kwin, plasmashell, akonadi,
      kdeconnect, baloo, konsole, Xorg/Wayland are ALL normal processes
  9.  Use efficient command chains: pipes, grep, awk, head
  10. Call finish() with your full structured threat report when done
  11. BUDGET RULE: If iteration ≥ ({max_iterations} - 5), call finish() IMMEDIATELY
      with whatever you have — a partial report is better than no report

━━━ finish() REPORT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  summary: "one sentence threat assessment"
  threat_level: LOW | MEDIUM | HIGH | CRITICAL
  findings: [{{"severity": "...", "description": "...", "evidence": "..."}}]
  actions_taken: ["Killed PID 1234 (xmrig cryptominer)", "Blocked 45.33.32.156"]
  recommendations: ["Enable fail2ban", "Disable root SSH", "Rotate API keys"]

Begin your first observation now.\
"""


def _build_blueteam_system_prompt(os_info: str, agent_pid: int,
                                   max_iterations: int,
                                   tools: set) -> str:
    """Format the blueteam system prompt with live values."""
    from ollama_agent_core import TOOL_SCHEMAS
    tool_lines = "\n".join(
        TOOL_SCHEMAS[t] if t in TOOL_SCHEMAS else _BLUETEAM_EXTRA_SCHEMAS.get(t, f"  {t}")
        for t in sorted(tools)
    )
    return BLUETEAM_SYSTEM_PROMPT_TEMPLATE.format(
        os_info=os_info,
        agent_pid=agent_pid,
        max_iterations=max_iterations,
        available_tools=tool_lines,
    )


# ─────────────────────── security survey ────────────────────────────────────

SECURITY_SURVEY_COMMANDS: Dict[str, str] = {
    # ── Processes ────────────────────────────────────────────────────────────
    "all_processes":
        "ps aux --sort=-%cpu 2>/dev/null | head -60",
    "root_processes":
        "ps aux 2>/dev/null | awk '$1==\"root\" && $11!~/^\\[/' | head -25",
    "high_cpu_procs":
        "ps aux --sort=-%cpu 2>/dev/null | awk 'NR>1 && $3+0>5' | head -15",

    # ── Network ──────────────────────────────────────────────────────────────
    "listening_ports":
        "ss -tulnp 2>/dev/null",
    "established_conns":
        "ss -antp 2>/dev/null | grep ESTAB | head -40",
    "outbound_summary":
        ("ss -antp 2>/dev/null | grep ESTAB | awk '{print $5}' "
         "| cut -d: -f1 | sort | uniq -c | sort -rn | head -20"),
    "dns_activity":
        ("ss -anup 2>/dev/null | head -20 ; "
         "cat /etc/resolv.conf 2>/dev/null"),

    # ── Auth / Sessions ───────────────────────────────────────────────────────
    "active_sessions":
        "who 2>/dev/null; echo '---'; w 2>/dev/null",
    "recent_logins":
        "last -n 20 2>/dev/null",
    "failed_logins":
        ("journalctl -u sshd --no-pager -n 50 2>/dev/null "
         "| grep -iE 'fail|invalid|refused|disconnect' "
         "|| grep -iE 'fail|invalid' /var/log/auth.log 2>/dev/null | tail -30"),
    "sudo_activity":
        "journalctl _COMM=sudo --no-pager -n 30 2>/dev/null",
    "local_accounts":
        "awk -F: '$3>=1000 && $3<65534 {print $1,$3,$7}' /etc/passwd 2>/dev/null",
    "ssh_authorized_keys":
        ("find /home /root -name authorized_keys 2>/dev/null "
         "-exec echo '=== {} ===' \\; -exec cat {} \\;"),

    # ── Files / Filesystem ────────────────────────────────────────────────────
    "recent_tmp_files":
        ("find /tmp /var/tmp /dev/shm -newer /proc/1/cmdline "
         "-type f 2>/dev/null | head -30"),
    "home_new_files":
        ("find /home /root -newer /proc/1/cmdline -type f "
         "-not -path '*/.git/*' 2>/dev/null | head -25"),
    "suid_files":
        "find /usr/bin /usr/sbin /bin /sbin -perm -4000 -type f 2>/dev/null",
    "world_writable":
        "find /tmp /var/tmp /dev/shm -type d -perm -o+w 2>/dev/null | head -15",

    # ── Services / Cron ───────────────────────────────────────────────────────
    "cron_jobs":
        ("crontab -l 2>/dev/null; echo '--- /etc/cron.d ---'; "
         "ls -la /etc/cron.d/ 2>/dev/null; echo '--- /etc/crontab ---'; "
         "cat /etc/crontab 2>/dev/null | head -30"),
    "failed_services":
        "systemctl list-units --state=failed --no-pager 2>/dev/null",
    "active_services":
        ("systemctl list-units --type=service --state=active "
         "--no-pager 2>/dev/null | head -40"),
    "systemd_user_units":
        ("find /home -name '*.service' -o -name '*.timer' 2>/dev/null | head -10; "
         "find /root/.config/systemd 2>/dev/null | head -10"),

    # ── Logs ─────────────────────────────────────────────────────────────────
    "system_errors":
        "journalctl -p err -n 40 --no-pager 2>/dev/null",
    "kernel_messages":
        "dmesg 2>/dev/null | grep -iE 'error|warn|fail|OOM|killed|segfault' | tail -25",
    "auth_log_recent":
        "journalctl -u sshd -n 30 --no-pager 2>/dev/null",

    # ── Resources ────────────────────────────────────────────────────────────
    "disk_usage":
        "df -h 2>/dev/null",
    "memory_usage":
        "free -h 2>/dev/null",
    "system_load":
        "uptime 2>/dev/null; cat /proc/loadavg 2>/dev/null",

    # ── Firewall / Security Config ────────────────────────────────────────────
    "firewall_rules":
        ("sudo nft list ruleset 2>/dev/null | grep -E 'chain|drop|accept|dport' "
         "| head -50"),
    "env_secrets_check":
        ("env 2>/dev/null | grep -iE '^(pass|secret|key|token|aws|api)' "
         "| sed 's/=.*/=[REDACTED]/' | head -10"),
    "open_files_tmp":
        "lsof +D /tmp 2>/dev/null | head -20",
}


# ───────────────────────── BlueteamAgent ─────────────────────────────────────

BLUETEAM_TOOLS = {
    "execute_command",
    "read_file",
    "memory_lookup",
    "web_search",
    "alert",
    "kill_process",
    "block_ip",
    "quarantine_file",
    "finish",
}


class BlueteamAgent:
    """
    SENTINEL — Defensive Cybersecurity Agent.

    Uses the existing OllamaCommandAgent ReAct loop but with:
    - A security-focused system prompt
    - Extended tool registry (alert, kill_process, block_ip, quarantine_file)
    - Deep security survey covering processes, network, auth, files, logs
    - Anomaly diffing for continuous watch mode
    """

    def __init__(self, model: str = "qwen3-coder:30b"):
        self.agent = OllamaCommandAgent(model=model)
        # Replace the tool registry with the extended blueteam version
        self.agent.tool_registry = BlueteamToolRegistry(
            safety_validator=self.agent.safety_validator,
            search_agent=self.agent.search_agent,
            memory=self.agent.memory,
            explain_cb=self.agent.explain_command_detailed,
        )
        self.agent.current_job_id = f"blueteam-{os.getpid()}"

        self._baseline: Optional[Dict[str, str]] = None
        self._last_scan: Optional[Dict[str, Any]] = None
        self._watching: bool = False
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_interval: int = 300      # quick poll (seconds)
        self._watch_deep_interval: int = 3600  # full LLM scan (seconds)
        self._last_deep_scan_ts: float = 0.0
        self._last_investigation_ts: float = 0.0  # cooldown tracker

        # Feature 1: arch-audit CVE state
        self._last_cve_list: List[Dict] = []

        # Feature 2: journal tailer state
        self._journal_thread: Optional[threading.Thread] = None
        self._journal_buffer: deque = deque(maxlen=50)
        self._journal_investigate_buffer: List[str] = []
        self._journal_last_investigate_ts: float = 0.0
        self._journal_last_flush_ts: float = 0.0

        # Feature 5: report / metrics state
        self._start_time: datetime = datetime.now()
        self._scan_count_quick: int = 0
        self._scan_count_deep: int = 0
        self._current_threat_level: str = "UNKNOWN"
        self._last_recommendations: List[str] = []
        self._report_lock: threading.Lock = threading.Lock()

    # ── survey ───────────────────────────────────────────────────────────────

    def security_survey(self) -> Dict[str, str]:
        """Run all security survey commands; return keyed result dict."""
        print("🔍 Running security survey...")
        results: Dict[str, str] = {}
        for key, cmd in SECURITY_SURVEY_COMMANDS.items():
            try:
                r = subprocess.run(
                    cmd, shell=True, capture_output=True,
                    text=True, timeout=15, executable="/bin/bash",
                )
                results[key] = (r.stdout + r.stderr).strip()[:2000]
            except subprocess.TimeoutExpired:
                results[key] = "(survey timed out)"
            except Exception as e:
                results[key] = f"(error: {e})"
        return results

    def _format_survey(self, survey: Dict[str, str]) -> str:
        """Format survey dict into a readable block for the LLM."""
        lines = ["═" * 70, "  LIVE SECURITY SURVEY", "═" * 70]
        for key, value in survey.items():
            lines.append(f"\n── {key.upper().replace('_', ' ')} {'─' * max(0, 40 - len(key))}")
            lines.append(value.strip() or "(empty)")
        lines.append("\n" + "═" * 70)
        return "\n".join(lines)

    # ── anomaly detection ────────────────────────────────────────────────────

    def _detect_anomalies(self, current: Dict[str, str]) -> List[str]:
        """Diff current survey against baseline; return list of anomaly strings."""
        if not self._baseline:
            return []
        anomalies: List[str] = []

        # New listening ports
        old_ports = set(self._baseline.get("listening_ports", "").splitlines())
        new_ports = set(current.get("listening_ports", "").splitlines())
        added = [p for p in (new_ports - old_ports) if p.strip() and "State" not in p]
        if added:
            anomalies.append(f"New listening port(s): {'; '.join(added[:3])}")

        # New established connections — only count connections to external IPs.
        # Internal traffic (localhost, Ollama at 127.x, KDE desktop at 10.x LAN)
        # churns constantly and is not useful as an anomaly signal.
        _INTERNAL_PREFIXES = ("127.", "::1", "0.0.0.0", "[::1]", "[::ffff:127")
        def _is_external(line: str) -> bool:
            parts = line.split()
            for part in parts:
                # ss output has peer address in 5th column (index 4)
                if any(part.startswith(p) for p in _INTERNAL_PREFIXES):
                    return False
            return True

        old_conns = {l for l in self._baseline.get("established_conns", "").splitlines()
                     if l.strip() and _is_external(l)}
        new_conns = {l for l in current.get("established_conns", "").splitlines()
                     if l.strip() and _is_external(l)}
        new_conn_count = len(new_conns - old_conns)
        if new_conn_count > 5:
            anomalies.append(f"{new_conn_count} new external connections")

        # Failed login spike (>5 new lines)
        old_fails = len(self._baseline.get("failed_logins", "").splitlines())
        new_fails = len(current.get("failed_logins", "").splitlines())
        if new_fails > old_fails + 5:
            anomalies.append(
                f"Failed login spike: {new_fails - old_fails} new failures"
            )

        # New files in /tmp / /var/tmp
        old_tmp = set(self._baseline.get("recent_tmp_files", "").splitlines())
        new_tmp = set(current.get("recent_tmp_files", "").splitlines())
        new_files = [f for f in (new_tmp - old_tmp) if f.strip()]
        if new_files:
            anomalies.append(f"New temp files: {', '.join(new_files[:3])}")

        # System error spike (>10 new lines)
        old_err = len(self._baseline.get("system_errors", "").splitlines())
        new_err = len(current.get("system_errors", "").splitlines())
        if new_err > old_err + 10:
            anomalies.append(f"System error spike: {new_err - old_err} new errors")

        # High-CPU processes not seen before — exclude inference engine, agent,
        # kernel threads, and KDE desktop processes (this host runs KDE Plasma).
        _EXPECTED_HIGH_CPU = {
            # Inference / agent
            "ollama", "python3", "python",
            # Kernel
            "kcompactd", "kswapd", "kworker", "migration", "ksoftirqd",
            # KDE Plasma desktop stack
            "plasmashell", "kwin", "kwin_wayland", "Xorg", "Xwayland",
            "kded", "kded6", "ksmserver", "knotificationd",
            "akonadi", "baloo", "baloorunner", "kio", "kioslave",
            "kdeconnect", "kdeconnectd", "ksecretd", "kwalletd", "kwalletd6",
            "konsole", "dolphin", "plasma", "kaccess",
        }
        old_cpu = set(self._baseline.get("high_cpu_procs", "").splitlines())
        new_cpu = set(current.get("high_cpu_procs", "").splitlines())
        new_cpu_procs = [
            p for p in (new_cpu - old_cpu)
            if p.strip() and not any(name in p for name in _EXPECTED_HIGH_CPU)
        ]
        if new_cpu_procs:
            anomalies.append(f"New high-CPU processes: {len(new_cpu_procs)}")

        # New cron entries
        old_cron = self._baseline.get("cron_jobs", "")
        new_cron = current.get("cron_jobs", "")
        if new_cron != old_cron and len(new_cron) > len(old_cron) + 20:
            anomalies.append("Cron jobs appear to have changed")

        # New failed services
        old_svc = self._baseline.get("failed_services", "")
        new_svc = current.get("failed_services", "")
        if new_svc != old_svc and "0 loaded units" not in new_svc:
            anomalies.append("New failed systemd units detected")

        return anomalies

    # ── arch-audit CVE integration (Feature 1) ───────────────────────────────

    def _run_arch_audit(self) -> List[Dict]:
        """Run arch-audit --json and return parsed CVE list. Returns [] on error."""
        global _ARCH_AUDIT_WARNED
        try:
            r = subprocess.run(
                ['arch-audit', '--json'],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0 and not r.stdout.strip():
                _dbg("ARCH-AUDIT", f"returned non-zero ({r.returncode}) with no output")
                return []
            cves = json.loads(r.stdout)
            _dbg("ARCH-AUDIT", f"found {len(cves)} vulnerable packages")
            return cves
        except FileNotFoundError:
            if not _ARCH_AUDIT_WARNED:
                _ARCH_AUDIT_WARNED = True
                _dbg("ARCH-AUDIT", "not installed — CVE scanning disabled (yay -S arch-audit)")
                print("WARNING: arch-audit not found — CVE scanning disabled "
                      "(install with: yay -S arch-audit)")
            return []
        except (json.JSONDecodeError, ValueError) as e:
            _dbg("ARCH-AUDIT", f"JSON parse error: {e}")
            return []
        except subprocess.TimeoutExpired:
            _dbg("ARCH-AUDIT", "timed out after 30s")
            return []
        except Exception as e:
            _dbg("ARCH-AUDIT", f"error: {e}")
            return []

    def _format_arch_audit(self, cves: List[Dict]) -> str:
        """Group CVEs by severity and format for LLM consumption (max 3000 chars)."""
        if not cves:
            return "(no CVEs found)"
        by_sev: Dict[str, List[Dict]] = {}
        for c in cves:
            sev = c.get('severity', 'UNKNOWN').upper()
            by_sev.setdefault(sev, []).append(c)

        lines = []
        for sev in ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'UNKNOWN'):
            items = by_sev.get(sev, [])
            if not items:
                continue
            lines.append(f"=== {sev} ({len(items)}) ===")
            for item in items[:10]:  # cap per severity
                pkg = item.get('pkg', item.get('name', '?'))
                cve_ids = item.get('cves', item.get('issues', []))
                if isinstance(cve_ids, list):
                    cve_str = ', '.join(str(c) for c in cve_ids[:3])
                else:
                    cve_str = str(cve_ids)
                lines.append(f"  {pkg}: {cve_str}")

        result = '\n'.join(lines)
        return result[:3000]

    # ── MOTD updates (Feature 3) ─────────────────────────────────────────────

    def update_motd(self, threat_level: str, last_scan: datetime,
                    alert_count: int) -> None:
        """Write SENTINEL status banner to /etc/motd (or ~/.motd_sentinel)."""
        ts = last_scan.strftime("%Y-%m-%d %H:%M")
        width = 34
        inner = width - 2  # space between the box walls

        def _row(text: str) -> str:
            return f"║ {text:<{inner}} ║"

        content = "\n".join([
            f"╔{'═' * (width - 2)}╗",
            _row(f"  SENTINEL  v3.3.0"),
            f"╠{'═' * (width - 2)}╣",
            _row(f"Last scan:     {ts}"),
            _row(f"Threat level:  {threat_level}"),
            _row(f"Alerts (24h):  {alert_count}"),
            f"╚{'═' * (width - 2)}╝",
            "",
        ])

        # Try system-wide MOTD first (requires passwordless sudo tee)
        try:
            r = subprocess.run(
                ['sudo', '-n', 'tee', str(_MOTD_SYSTEM)],
                input=content, capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                _dbg("MOTD", f"written to /etc/motd  threat={threat_level}")
                return
        except Exception:
            pass

        # Fallback: user's ~/.motd_sentinel
        try:
            _MOTD_FALLBACK.write_text(content, encoding="utf-8")
            _dbg("MOTD", f"written to ~/.motd_sentinel  threat={threat_level}")
        except Exception as e:
            _dbg("MOTD", f"write failed: {e}")

    # ── report file (Feature 5) ──────────────────────────────────────────────

    def _write_report(self) -> None:
        """Write ~/.agent_bin/sentinel_report.md with current SENTINEL state."""
        try:
            _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            uptime_secs = int((datetime.now() - self._start_time).total_seconds())
            uptime_str = (
                f"{uptime_secs // 3600}h "
                f"{(uptime_secs % 3600) // 60}m "
                f"{uptime_secs % 60}s"
            )

            recent = get_recent_alerts(10)
            alert_lines = []
            for a in recent[-10:]:
                ts = a.get("ts", "")[:19]
                sev = a.get("severity", "?")
                finding = a.get("finding", "")
                alert_lines.append(f"- `[{ts}]` **{sev}**: {finding}")

            last_inv_summary = ""
            if self._last_scan:
                last_inv_summary = self._last_scan.get("finish_summary", "")

            rec_lines = "\n".join(
                f"- {r}" for r in self._last_recommendations
            ) if self._last_recommendations else "- No recommendations yet."

            content = (
                f"# SENTINEL Report — {now_str}\n\n"
                f"**Threat Level:** {self._current_threat_level}\n"
                f"**Uptime:** Running since "
                f"{self._start_time.strftime('%Y-%m-%d %H:%M:%S')} ({uptime_str})\n"
                f"**Scans completed:** {self._scan_count_quick} quick, "
                f"{self._scan_count_deep} deep\n\n"
                f"## Recent Alerts (last 10)\n\n"
                + ("\n".join(alert_lines) if alert_lines else "- No alerts on record.")
                + f"\n\n## Last Investigation\n\n"
                + (last_inv_summary or "_No scans completed yet._")
                + f"\n\n## Recommendations\n\n{rec_lines}\n"
            )

            with self._report_lock:
                _REPORT_PATH.write_text(content, encoding="utf-8")
            _dbg("REPORT", f"written  threat={self._current_threat_level}  "
                           f"quick={self._scan_count_quick}  deep={self._scan_count_deep}")
        except Exception as e:
            _dbg("REPORT", f"write failed: {e}")
            print(f"👁️  Report write error: {e}")

    # ── journal tailer (Feature 2) ───────────────────────────────────────────

    _JOURNAL_TRIGGER_KEYWORDS = frozenset({
        "failed", "error", "denied", "killed", "crash",
        "invalid", "segfault", "timeout", "unauthorized",
    })

    def _journal_tailer(self) -> None:
        """Background thread: tail journal streams and batch suspicious lines for LLM."""
        # Start three journalctl streams
        procs = []
        cmds = [
            ['journalctl', '-f', '--no-pager', '-p', 'err'],
            ['journalctl', '-f', '-u', 'sshd', '--no-pager'],
            ['journalctl', '-f', '-p', 'warning', '--no-pager'],
        ]
        _dbg("JOURNAL", f"starting {len(cmds)} journal streams")
        for cmd in cmds:
            try:
                p = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                procs.append(p)
                _dbg("JOURNAL", f"stream started: {' '.join(cmd[1:])}")
            except Exception as e:
                _dbg("JOURNAL", f"failed to start {cmd[2:4]}: {e}")
                print(f"👁️  Journal tailer: failed to start {cmd[2:4]}: {e}")

        if not procs:
            _dbg("JOURNAL", "no streams available — tailer exiting")
            print("👁️  Journal tailer: no journal streams available, exiting")
            return

        sel = selectors.DefaultSelector()
        for p in procs:
            if p.stdout:
                sel.register(p.stdout, selectors.EVENT_READ)

        self._journal_last_flush_ts = time.time()

        try:
            while self._watching:
                events = sel.select(timeout=2.0)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if not line:
                        # EOF — journald restart; brief pause before continuing
                        time.sleep(1.0)
                        continue
                    line = line.rstrip()
                    self._journal_buffer.append(line)
                    line_lower = line.lower()
                    if any(kw in line_lower for kw in self._JOURNAL_TRIGGER_KEYWORDS):
                        self._journal_investigate_buffer.append(line)
                        _dbg("JOURNAL", f"trigger line [{len(self._journal_investigate_buffer)} buffered]: {line[:100]}")

                now = time.time()
                buf_len = len(self._journal_investigate_buffer)
                time_elapsed = now - self._journal_last_flush_ts

                if buf_len >= 20 or (buf_len > 0 and time_elapsed >= 30.0):
                    reason = "20-line cap" if buf_len >= 20 else "30s timer"
                    _dbg("JOURNAL", f"flushing {buf_len} trigger lines to LLM ({reason})")
                    lines_to_send = list(self._journal_investigate_buffer)
                    self._journal_investigate_buffer = []
                    self._journal_last_flush_ts = now
                    t = threading.Thread(
                        target=self._trigger_investigation,
                        args=(lines_to_send,),
                        daemon=True,
                        name="SentinelJournalInvestigate",
                    )
                    t.start()
        finally:
            sel.close()
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass

    def _trigger_investigation(self, lines: List[str]) -> None:
        """Called in a daemon thread to hand buffered journal lines to the LLM."""
        JOURNAL_INVESTIGATE_COOLDOWN = 300  # 5 minutes
        now = time.time()
        secs_since = now - self._journal_last_investigate_ts
        if secs_since < JOURNAL_INVESTIGATE_COOLDOWN:
            _dbg("JOURNAL", f"investigation cooldown active ({int(JOURNAL_INVESTIGATE_COOLDOWN - secs_since)}s left) — skipping")
            return
        self._journal_last_investigate_ts = now
        _dbg("JOURNAL", f"sending {len(lines)} lines to LLM for investigation")

        evidence = "RECENT JOURNAL EVENTS:\n" + "\n".join(lines)
        try:
            self.investigate(
                finding="Real-time journal events require analysis",
                evidence=evidence,
            )
        except Exception as e:
            _dbg("JOURNAL", f"investigation error: {e}")
            print(f"👁️  Journal investigation error: {e}")

    # ── public API ───────────────────────────────────────────────────────────

    def scan(self, focus: str = "") -> Dict[str, Any]:
        """
        One-shot security scan: run survey, analyse with LLM, return threat report.

        Args:
            focus: Optional area to focus on (e.g. "SSH", "network connections")
        """
        _dbg("SCAN", f"starting  focus={focus[:60]!r}" if focus else "starting")
        survey = self.security_survey()
        survey_text = self._format_survey(survey)

        focus_line = f"\nFOCUS AREA: {focus}\n" if focus else ""

        instruction = (
            f"SECURITY SCAN REQUEST{focus_line}\n\n"
            f"{survey_text}\n\n"
            "Your job:\n"
            "1. Analyse every section of the survey above for threats\n"
            "2. For each suspicious finding call alert() with severity + evidence\n"
            "3. For HIGH/CRITICAL findings: contain with kill_process or block_ip\n"
            "4. Dig deeper with execute_command to read relevant log lines\n"
            "5. Call finish() with your complete threat report\n\n"
            "Start your analysis now."
        )

        prompt = _build_blueteam_system_prompt(
            os_info=self.agent.os_info,
            agent_pid=os.getpid(),
            max_iterations=40,
            tools=BLUETEAM_TOOLS,
        )

        result = self.agent.run_react(
            instruction=instruction,
            tool_whitelist=BLUETEAM_TOOLS,
            max_iterations=40,
            system_prompt_override=prompt,
        )
        self._last_scan = result
        _dbg("SCAN", f"done  iters={result.get('iterations_used',0)}  "
                     f"success={result.get('success')}")

        # Extract threat level and recommendations from LLM finish summary
        m = re.search(r'\b(CRITICAL|HIGH|MEDIUM|LOW)\b',
                      result.get('finish_summary', '').upper())
        self._current_threat_level = m.group(1) if m else "UNKNOWN"
        self._last_recommendations = []
        for t in result.get('trace', []):
            if t.get('tool') == 'finish':
                self._last_recommendations = t.get('args', {}).get('recommendations', [])
                break

        if _dlog:
            _dlog.log("blueteam_scan_complete", {
                "success": result.get("success"),
                "iterations_used": result.get("iterations_used"),
                "focus": focus,
            })

        # Update MOTD and report
        alert_count_24h = len([
            a for a in get_recent_alerts(500)
            if (datetime.now() - datetime.fromisoformat(a.get("ts", datetime.now().isoformat())))
               .total_seconds() < 86400
        ])
        self.update_motd(self._current_threat_level, datetime.now(), alert_count_24h)
        self._write_report()

        return result

    def investigate(self, finding: str, evidence: str = "") -> Dict[str, Any]:
        """
        Targeted investigation of a specific finding.

        Args:
            finding: Description of what's suspicious
            evidence: Any initial evidence already gathered
        """
        _dbg("INVESTIGATE", f"trigger={finding[:100]!r}")
        instruction = (
            f"AUTONOMOUS INVESTIGATION\n\n"
            f"Trigger:  {finding}\n\n"
            f"Pre-collected evidence (use this first before re-running commands):\n"
            f"{evidence or '(none — gather it yourself)'}\n\n"
            "Your job is to determine autonomously whether this is a real threat "
            "or normal system activity. Follow this decision process:\n\n"
            "1. Review the pre-collected evidence above\n"
            "2. Run any additional commands needed to reach a confident conclusion\n"
            "3. BENIGN (normal activity): call finish() with threat_level: LOW\n"
            "   — do NOT call alert() for normal server/desktop noise\n"
            "4. SUSPICIOUS (MEDIUM): call alert() with full evidence, dig deeper\n"
            "5. CONFIRMED THREAT (HIGH/CRITICAL): alert() + contain "
            "(kill_process / block_ip) + finish()\n\n"
            "Be skeptical. Most anomaly detections are benign. Only escalate "
            "when evidence is conclusive. Start your investigation now."
        )

        prompt = _build_blueteam_system_prompt(
            os_info=self.agent.os_info,
            agent_pid=os.getpid(),
            max_iterations=35,
            tools=BLUETEAM_TOOLS,
        )

        result = self.agent.run_react(
            instruction=instruction,
            tool_whitelist=BLUETEAM_TOOLS,
            max_iterations=35,
            system_prompt_override=prompt,
        )

        _dbg("INVESTIGATE", f"done  iters={result.get('iterations_used',0)}  "
                            f"success={result.get('success')}  "
                            f"summary={result.get('finish_summary','')[:80]!r}")

        # Update MOTD and report after investigation
        alert_count_24h = len([
            a for a in get_recent_alerts(500)
            if (datetime.now() - datetime.fromisoformat(a.get("ts", datetime.now().isoformat())))
               .total_seconds() < 86400
        ])
        self.update_motd(self._current_threat_level, datetime.now(), alert_count_24h)
        self._write_report()

        return result

    def watch(self, interval: int = None, quick_interval: int = 300,
              deep_interval: int = 3600) -> None:
        """Start continuous background monitoring (non-blocking).

        Args:
            interval:       Legacy alias for quick_interval.
            quick_interval: Seconds between anomaly-diff polls (default 300 = 5 min).
            deep_interval:  Seconds between full LLM deep scans (default 3600 = 1 hr).
        """
        if self._watching:
            print("👁️  Watcher already running")
            return
        if interval is not None:
            quick_interval = interval  # backwards-compat
        self._watch_interval = quick_interval
        self._watch_deep_interval = deep_interval
        self._watching = True
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(quick_interval, deep_interval),
            daemon=True,
            name="SentinelWatcher",
        )
        self._watch_thread.start()
        print(
            f"👁️  SENTINEL watcher started — "
            f"quick poll every {quick_interval}s, deep scan every {deep_interval}s"
        )

        # Start real-time journal tailer (Feature 2)
        self._journal_thread = threading.Thread(
            target=self._journal_tailer,
            daemon=True,
            name="SentinelJournalTailer",
        )
        self._journal_thread.start()
        print("👁️  Journal tailer started — watching journal streams in real-time")
        _dbg("WATCH", f"started  quick={quick_interval}s  deep={deep_interval}s")

    def stop_watch(self) -> None:
        """Stop the background monitoring loop."""
        self._watching = False
        print("👁️  SENTINEL watcher stopping...")

    def status(self) -> Dict[str, Any]:
        """Return current watcher status and last scan summary."""
        last_alerts = get_recent_alerts(10)
        next_deep = None
        if self._last_deep_scan_ts > 0:
            secs_remaining = max(0, self._watch_deep_interval - (time.time() - self._last_deep_scan_ts))
            next_deep = f"{int(secs_remaining // 60)}m {int(secs_remaining % 60)}s"
        return {
            "watching": self._watching,
            "quick_interval": self._watch_interval,
            "deep_interval": self._watch_deep_interval,
            "next_deep_scan_in": next_deep,
            "has_baseline": self._baseline is not None,
            "last_scan_success": (self._last_scan or {}).get("success"),
            "last_scan_summary": (self._last_scan or {}).get("finish_summary", ""),
            "recent_alert_count": len(last_alerts),
            "recent_alerts": last_alerts[-5:],
            "threat_level": self._current_threat_level,
            "scan_count_quick": self._scan_count_quick,
            "scan_count_deep": self._scan_count_deep,
        }

    # ── watch loop ───────────────────────────────────────────────────────────

    def _watch_loop(self, quick_interval: int, deep_interval: int) -> None:
        """Background loop: quick anomaly diff every quick_interval,
        full LLM deep scan every deep_interval."""
        _dbg("WATCH-LOOP", "building baseline survey")
        print("👁️  Building baseline survey...")
        self._baseline = self.security_survey()
        print("👁️  Baseline established. Watching for anomalies...")
        _dbg("WATCH-LOOP", "baseline established")

        if _dlog:
            _dlog.log("blueteam_watch_start", {
                "quick_interval": quick_interval,
                "deep_interval": deep_interval,
            })

        # Schedule first deep scan after deep_interval (not immediately)
        self._last_deep_scan_ts = time.time()

        while self._watching:
            time.sleep(quick_interval)
            if not self._watching:
                break

            now = time.time()
            try:
                _dbg("WATCH-LOOP", f"poll #{self._scan_count_quick + 1} starting")
                current = self.security_survey()
                anomalies = self._detect_anomalies(current)
                _dbg("WATCH-LOOP", f"survey done  anomalies={len(anomalies)}"
                     + (f": {'; '.join(anomalies[:3])}" if anomalies else ""))

                if anomalies:
                    summary = "; ".join(anomalies[:4])
                    print(f"⚠️  Anomalies detected: {summary}")

                    # Cooldown: don't pile up investigations (20-min minimum gap)
                    INVESTIGATION_COOLDOWN = 1200
                    secs_since_last = now - self._last_investigation_ts
                    if secs_since_last < INVESTIGATION_COOLDOWN:
                        remaining = int((INVESTIGATION_COOLDOWN - secs_since_last) / 60)
                        _dbg("WATCH-LOOP", f"cooldown active ({remaining}m left) — skipping investigation")
                        print(f"👁️  Anomalies noted — cooldown active ({remaining}m remaining): {summary}")
                    else:
                        self._last_investigation_ts = now
                        _dbg("WATCH-LOOP", f"triggering investigation: {summary[:100]}")
                        print(f"👁️  Investigating: {summary}")
                        if _dlog:
                            _dlog.log("blueteam_investigation_start", {
                                "anomalies": anomalies,
                                "summary": summary,
                            })

                        # Build evidence from pre-collected survey data so the
                        # LLM doesn't waste iterations re-gathering the same info.
                        ev_parts: List[str] = [f"DETECTED ANOMALIES:\n{chr(10).join(anomalies)}"]
                        keyword_map = {
                            "port":     ("listening_ports",  "LISTENING PORTS"),
                            "connect":  ("established_conns","ESTABLISHED CONNECTIONS"),
                            "outbound": ("outbound_summary", "OUTBOUND SUMMARY"),
                            "login":    ("failed_logins",    "FAILED LOGINS"),
                            "tmp":      ("recent_tmp_files", "RECENT TMP FILES"),
                            "file":     ("home_new_files",   "NEW HOME FILES"),
                            "cpu":      ("high_cpu_procs",   "HIGH CPU PROCESSES"),
                            "cron":     ("cron_jobs",        "CRON JOBS"),
                            "service":  ("failed_services",  "FAILED SERVICES"),
                        }
                        summary_lower = summary.lower()
                        for kw, (survey_key, label) in keyword_map.items():
                            if kw in summary_lower:
                                val = current.get(survey_key, "").strip()
                                if val:
                                    ev_parts.append(f"{label}:\n{val[:600]}")

                        # Run autonomous investigation — the LLM decides whether
                        # to alert, contain, or dismiss. No pre-emptive alert here.
                        inv_result = self.investigate(
                            finding=f"Watch-mode anomalies: {summary}",
                            evidence="\n\n".join(ev_parts),
                        )

                        conclusion = inv_result.get("finish_summary", "no conclusion")
                        m = re.search(r'\b(CRITICAL|HIGH|MEDIUM|LOW)\b', conclusion.upper())
                        threat = m.group(1) if m else "UNKNOWN"
                        _dbg("WATCH-LOOP", f"investigation done  threat={threat}  "
                             f"conclusion={conclusion[:80]!r}")
                        print(f"👁️  Investigation concluded [{threat}]: {conclusion[:120]}")
                        if _dlog:
                            _dlog.log("blueteam_investigation_complete", {
                                "anomalies": summary,
                                "threat_level": threat,
                                "conclusion": conclusion,
                                "iterations_used": inv_result.get("iterations_used", 0),
                            })

                self._baseline = current
                self._scan_count_quick += 1
                _dbg("WATCH-LOOP", f"quick poll #{self._scan_count_quick} done")

                if _dlog:
                    _dlog.log("blueteam_watch_tick", {
                        "kind": "quick",
                        "anomalies": len(anomalies),
                        "timestamp": datetime.now().isoformat(),
                    })

                # Update MOTD and report after each quick poll
                alert_count_24h = len([
                    a for a in get_recent_alerts(500)
                    if (datetime.now() - datetime.fromisoformat(
                        a.get("ts", datetime.now().isoformat()))
                    ).total_seconds() < 86400
                ])
                self.update_motd(self._current_threat_level, datetime.now(), alert_count_24h)
                self._write_report()

                # Full deep scan on schedule
                if now - self._last_deep_scan_ts >= deep_interval:
                    _dbg("WATCH-LOOP", "deep scan triggered")
                    print("👁️  Running scheduled deep scan...")
                    if _dlog:
                        _dlog.log("blueteam_deep_scan_start", {
                            "timestamp": datetime.now().isoformat(),
                        })

                    # Run arch-audit before deep scan (Feature 1)
                    cves = self._run_arch_audit()
                    _dbg("ARCH-AUDIT", f"deep scan: {len(cves)} vulnerable packages total")
                    if cves:
                        self._last_cve_list = cves
                        critical_cves = [
                            c for c in cves
                            if c.get('severity', '').upper() == 'CRITICAL'
                        ]
                        if critical_cves:
                            emit_alert(
                                "CRITICAL",
                                f"arch-audit: {len(critical_cves)} CRITICAL CVEs",
                                evidence=self._format_arch_audit(critical_cves)[:500],
                                action="sudo pacman -Syu to update",
                            )

                    cve_focus = (
                        f"\n\nARCH-AUDIT CVE REPORT:\n{self._format_arch_audit(self._last_cve_list)}"
                        if self._last_cve_list else ""
                    )
                    self.scan(focus=cve_focus)
                    self._scan_count_deep += 1
                    self._last_deep_scan_ts = time.time()
                    _dbg("WATCH-LOOP", f"deep scan #{self._scan_count_deep} done")

                    if _dlog:
                        _dlog.log("blueteam_watch_tick", {
                            "kind": "deep",
                            "timestamp": datetime.now().isoformat(),
                        })

            except Exception as e:
                print(f"👁️  Watch loop error: {e}")
                if _dlog:
                    _dlog.error("blueteam_watch_loop", e)


# ─────────────────────── global watcher instance ─────────────────────────────

_sentinel: Optional[BlueteamAgent] = None
_sentinel_lock = threading.Lock()


def get_sentinel(model: str = "qwen3-coder:30b") -> BlueteamAgent:
    """Return the global singleton BlueteamAgent, creating it if needed."""
    global _sentinel
    with _sentinel_lock:
        if _sentinel is None:
            _sentinel = BlueteamAgent(model=model)
    return _sentinel


# ─────────────────────── standalone CLI ──────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SENTINEL — Defensive Cybersecurity Agent"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_scan = sub.add_parser("scan", help="One-shot security scan")
    p_scan.add_argument("--focus", default="", help="Area to focus on")

    p_inv = sub.add_parser("investigate", help="Investigate a specific finding")
    p_inv.add_argument("finding", help="Description of the suspicious finding")
    p_inv.add_argument("--evidence", default="", help="Initial evidence")

    p_watch = sub.add_parser("watch", help="Start continuous monitoring")
    p_watch.add_argument("--interval", type=int, default=60,
                         help="Seconds between polls (default 60)")

    p_alerts = sub.add_parser("alerts", help="Show recent security alerts")
    p_alerts.add_argument("-n", type=int, default=20, help="Number of alerts")

    args = parser.parse_args()

    if args.cmd == "scan":
        agent = BlueteamAgent()
        result = agent.scan(focus=args.focus)
        print("\n" + "=" * 70)
        print("THREAT REPORT:")
        print(result.get("finish_summary", "(no summary)"))

    elif args.cmd == "investigate":
        agent = BlueteamAgent()
        result = agent.investigate(args.finding, args.evidence)
        print("\n" + "=" * 70)
        print("INVESTIGATION REPORT:")
        print(result.get("finish_summary", "(no summary)"))

    elif args.cmd == "watch":
        agent = BlueteamAgent()
        agent.watch(interval=args.interval)
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            agent.stop_watch()

    elif args.cmd == "alerts":
        alerts = get_recent_alerts(args.n)
        if not alerts:
            print("No alerts on record.")
        else:
            for a in alerts:
                ts = a.get("ts", "")[:19]
                sev = a.get("severity", "?")
                finding = a.get("finding", "")
                print(f"  [{ts}] [{sev:8s}] {finding}")

    else:
        parser.print_help()
