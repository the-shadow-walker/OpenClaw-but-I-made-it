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
import signal as _signal
import subprocess
import threading
import time
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
  1. OBSERVE   → execute_command / read_file to gather evidence
  2. ASSESS    → determine threat level: LOW | MEDIUM | HIGH | CRITICAL
  3. ALERT     → call alert() for every MEDIUM+ finding (required)
  4. CONTAIN   → kill_process or block_ip ONLY for HIGH/CRITICAL
  5. ERADICATE → quarantine_file to remove confirmed malicious artifacts
  6. REPORT    → call finish() with your full structured threat report

━━━ DETECTION PRIORITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Malicious processes (cryptominers, reverse shells, RATs, exfil tools)
  2. Unusual outbound connections (unexpected destinations, high data volume)
  3. Auth anomalies (login spikes, new accounts, sudo abuse, SSH keys)
  4. Persistence (cron, systemd units, .bashrc/.profile, SUID changes)
  5. Lateral movement (internal SSH pivoting, ARP spoofing, scanning)
  6. Data exfiltration (large transfers, cloud destinations, DNS tunneling)

━━━ THREAT LEVELS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LOW      → Unusual but likely benign. Document only. No action required.
  MEDIUM   → Suspicious. Investigate further. MUST call alert().
  HIGH     → Confirmed threat. Contain immediately. alert() + action.
  CRITICAL → Active breach. Contain + eradicate + alert everything.

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1.  Output ONLY valid JSON — never plain text or markdown
  2.  Call alert() for every MEDIUM+ finding before any action
  3.  NEVER kill: PID 1, systemd, sshd, ollama-agent (PID {agent_pid})
  4.  NEVER block private ranges: 127.x, 10.x, 192.168.x, 172.16-31.x
  5.  Read /proc/<pid>/cmdline and /proc/<pid>/environ before killing
  6.  Check process ancestry (ps -p PID -o ppid,cmd) before killing
  7.  confidence < 70 → gather more evidence before acting
  8.  When unsure → alert(severity=LOW) and continue investigating
  9.  Use efficient command chains: pipes, grep, awk, head
  10. Call finish() with your full structured threat report when done

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
            memory=self.agent.memory,
            safety_validator=self.agent.safety_validator,
            explain_cb=self.agent.explain_command_detailed,
        )
        self.agent.current_job_id = f"blueteam-{os.getpid()}"

        self._baseline: Optional[Dict[str, str]] = None
        self._last_scan: Optional[Dict[str, Any]] = None
        self._watching: bool = False
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_interval: int = 60

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

        # New established connections (allow natural churn — flag >3 new)
        old_conns = set(self._baseline.get("established_conns", "").splitlines())
        new_conns = set(current.get("established_conns", "").splitlines())
        new_conn_count = len(new_conns - old_conns)
        if new_conn_count > 3:
            anomalies.append(f"{new_conn_count} new established connections")

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

        # High-CPU processes not seen before
        old_cpu = set(self._baseline.get("high_cpu_procs", "").splitlines())
        new_cpu = set(current.get("high_cpu_procs", "").splitlines())
        new_cpu_procs = [p for p in (new_cpu - old_cpu) if p.strip()]
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

    # ── public API ───────────────────────────────────────────────────────────

    def scan(self, focus: str = "") -> Dict[str, Any]:
        """
        One-shot security scan: run survey, analyse with LLM, return threat report.

        Args:
            focus: Optional area to focus on (e.g. "SSH", "network connections")
        """
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
            max_iterations=30,
            tools=BLUETEAM_TOOLS,
        )

        result = self.agent.run_react(
            instruction=instruction,
            tool_whitelist=BLUETEAM_TOOLS,
            max_iterations=30,
            system_prompt_override=prompt,
        )
        self._last_scan = result

        if _dlog:
            _dlog.log("blueteam_scan_complete", {
                "success": result.get("success"),
                "iterations_used": result.get("iterations_used"),
                "focus": focus,
            })

        return result

    def investigate(self, finding: str, evidence: str = "") -> Dict[str, Any]:
        """
        Targeted investigation of a specific finding.

        Args:
            finding: Description of what's suspicious
            evidence: Any initial evidence already gathered
        """
        instruction = (
            f"INVESTIGATION REQUEST\n\n"
            f"Finding:  {finding}\n"
            f"Evidence: {evidence or '(none yet — gather it)'}\n\n"
            "Investigate deeply:\n"
            "1. Read relevant logs and process state to build evidence\n"
            "2. Determine threat level: LOW / MEDIUM / HIGH / CRITICAL\n"
            "3. Call alert() with severity, finding, and all evidence\n"
            "4. If HIGH+: contain with kill_process or block_ip\n"
            "5. Call finish() with your full incident report\n\n"
            "Start gathering evidence now."
        )

        prompt = _build_blueteam_system_prompt(
            os_info=self.agent.os_info,
            agent_pid=os.getpid(),
            max_iterations=25,
            tools=BLUETEAM_TOOLS,
        )

        return self.agent.run_react(
            instruction=instruction,
            tool_whitelist=BLUETEAM_TOOLS,
            max_iterations=25,
            system_prompt_override=prompt,
        )

    def watch(self, interval: int = 60) -> None:
        """Start continuous background monitoring (non-blocking)."""
        if self._watching:
            print("👁️  Watcher already running")
            return
        self._watch_interval = interval
        self._watching = True
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True,
            name="SentinelWatcher",
        )
        self._watch_thread.start()
        print(f"👁️  SENTINEL watcher started — polling every {interval}s")

    def stop_watch(self) -> None:
        """Stop the background monitoring loop."""
        self._watching = False
        print("👁️  SENTINEL watcher stopping...")

    def status(self) -> Dict[str, Any]:
        """Return current watcher status and last scan summary."""
        last_alerts = get_recent_alerts(10)
        return {
            "watching": self._watching,
            "watch_interval": self._watch_interval,
            "has_baseline": self._baseline is not None,
            "last_scan_success": (self._last_scan or {}).get("success"),
            "last_scan_summary": (self._last_scan or {}).get("finish_summary", ""),
            "recent_alert_count": len(last_alerts),
            "recent_alerts": last_alerts[-5:],
        }

    # ── watch loop ───────────────────────────────────────────────────────────

    def _watch_loop(self, interval: int) -> None:
        """Background loop: survey → diff → investigate anomalies."""
        print("👁️  Building baseline survey...")
        self._baseline = self.security_survey()
        print("👁️  Baseline established. Watching for anomalies...")

        if _dlog:
            _dlog.log("blueteam_watch_start", {"interval": interval})

        while self._watching:
            time.sleep(interval)
            if not self._watching:
                break

            try:
                current = self.security_survey()
                anomalies = self._detect_anomalies(current)

                if anomalies:
                    summary = "; ".join(anomalies[:4])
                    print(f"⚠️  Anomalies detected: {summary}")
                    emit_alert("MEDIUM", "Watch-mode anomalies detected",
                               evidence=summary, action="Triggering investigation")
                    self.investigate(
                        finding=f"Watch-mode anomalies: {summary}",
                        evidence="\n".join(anomalies),
                    )

                self._baseline = current

                if _dlog:
                    _dlog.log("blueteam_watch_tick", {
                        "anomalies": len(anomalies),
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
