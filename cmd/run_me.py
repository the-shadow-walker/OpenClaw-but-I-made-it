#!/usr/bin/env python3
"""
run_me.py — Jarvis CMD Agent CLI  v2.0
Connects to the ollama-cmd service (default: http://10.0.0.58:5000).

Usage:
  python run_me.py "your task"              # submit + live stream
  python run_me.py quick "df -h"            # instant command, no ReAct loop
  python run_me.py quick --ask "disk usage" # NL→command, stream result
  python run_me.py -i                       # interactive REPL
  python run_me.py health                   # service health check
  python run_me.py jobs                     # list recent jobs
  python run_me.py job <id>                 # show a specific job
  python run_me.py cancel <id>              # cancel a job
  python run_me.py chain "build flask app"  # multi-phase chain
  python run_me.py chains                   # list all chains
  python run_me.py chain-status <id>        # chain detail
  python run_me.py sentinel                 # SENTINEL status
  python run_me.py scan [focus]             # run security scan
  python run_me.py report                   # daily security report
  python run_me.py alerts                   # recent security alerts

Flags:
  -i / --interactive   REPL mode
  -q / --quiet         Answer only, no spinner or event labels
  -v / --verbose       Print all events including thinking
  -d / --debug         Print raw SSE JSON
  --no-stream          Poll instead of streaming
  -s / --server URL    Override server URL

Environment:
  JARVIS_URL   Base URL  (default: http://10.0.0.58:5000)
"""

import sys, os, json, time, argparse, threading
import urllib.request, urllib.error, urllib.parse

DEFAULT_SERVER = os.environ.get("JARVIS_URL", "http://10.0.0.58:5000")
POLL_INTERVAL  = 2
POLL_TIMEOUT   = 600

VERBOSITY  = 1      # 0=quiet, 1=normal, 2=verbose
DEBUG_MODE = False

# ── ANSI helpers ───────────────────────────────────────────────────────────────

_USE_COLOR = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else True

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BOLD      = lambda t: _c("1",  t)
DIM       = lambda t: _c("2",  t)
GREEN     = lambda t: _c("32", t)
YELLOW    = lambda t: _c("33", t)
CYAN      = lambda t: _c("36", t)
RED       = lambda t: _c("31", t)
BLUE      = lambda t: _c("34", t)
MAGENTA   = lambda t: _c("35", t)
B_GREEN   = lambda t: _c("92", t)
B_RED     = lambda t: _c("91", t)
B_CYAN    = lambda t: _c("96", t)
CLEAR     = "\033[2K\r" if _USE_COLOR else ""

STATUS_ICON = {
    "queued": "⏳", "running": "🔄", "completed": "✅",
    "failed": "❌", "cancelled": "🚫", "pending": "⏳",
    "decomposing": "🧠", "passed": "✅", "skipped": "⏭ ",
}
SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}


def _term_width():
    try:
        return os.get_terminal_size(fallback=(100, 40)).columns
    except Exception:
        return 100

def _progress_bar(pct, width=20, done=False):
    filled = int(round(pct / 100 * width))
    col    = B_GREEN if done else CYAN
    return col("[") + col("█" * filled) + DIM("░" * (width - filled)) + col("]")


# ── HTTP helpers (stdlib only) ─────────────────────────────────────────────────

def _headers():
    return {"Content-Type": "application/json", "Accept": "application/json"}

def _get(server, path, params=None):
    url = server.rstrip("/") + path
    if params:
        url += "?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(RED(f"[HTTP {e.code}] {e.read().decode()[:200]}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}  server={server}"), file=sys.stderr)
        sys.exit(1)

def _post(server, path, payload, timeout=60):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(server.rstrip("/") + path,
                                  data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(RED(f"[HTTP {e.code}] {e.read().decode()[:200]}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}"), file=sys.stderr)
        sys.exit(1)

def _delete(server, path):
    req = urllib.request.Request(server.rstrip("/") + path,
                                 method="DELETE", headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(RED(f"[HTTP {e.code}] {e.read().decode()[:200]}"), file=sys.stderr)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}"), file=sys.stderr)


# ── Spinner ────────────────────────────────────────────────────────────────────

class _Spinner:
    """
    Single-line background spinner.  Thread-safe.

    pause_print(fn) erases the spinner line, runs fn() to print content,
    then the spinner automatically redraws below it on the next tick —
    so callers never need to restart the thread.
    """
    _FRAMES = ["|", "/", "-", "\\"]

    def __init__(self):
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._tick   = 0
        self._t0     = time.time()
        self._text   = ""
        self._drawn  = False
        self._paused = False

    def start(self):
        self._t0    = time.time()
        self._drawn = False
        self._stop.clear()
        threading.Thread(target=self._worker, daemon=True).start()

    def update(self, text=""):
        with self._lock:
            self._text = text

    def _erase(self):
        """Erase spinner line. Caller must hold lock."""
        if self._drawn and _USE_COLOR:
            sys.stdout.write("\033[1A\033[2K")
            sys.stdout.flush()
        self._drawn = False

    def _draw(self):
        with self._lock:
            if self._paused or self._stop.is_set():
                return
            self._tick += 1
            el   = int(time.time() - self._t0)
            s    = self._FRAMES[self._tick % 4]
            txt  = f"  {DIM(self._text[:55])}" if self._text else ""
            line = f"  {s} {CYAN('working')}{txt}  {DIM(str(el) + 's')}"
            if self._drawn and _USE_COLOR:
                sys.stdout.write("\033[1A\033[2K")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            self._drawn = True

    def pause_print(self, fn):
        """Erase spinner, run fn() to print content, resume on next tick."""
        with self._lock:
            self._erase()
            self._paused = True
        try:
            fn()
        finally:
            with self._lock:
                self._paused = False
                self._drawn  = False  # spinner will redraw below printed content

    def stop(self):
        with self._lock:
            self._erase()
        self._stop.set()

    def _worker(self):
        while not self._stop.is_set():
            self._draw()
            self._stop.wait(0.15)


# ── SSE streaming ──────────────────────────────────────────────────────────────

def _sse_iter(url, timeout=600):
    """Generator: yields parsed SSE event dicts from url."""
    req = urllib.request.Request(
        url, headers={**_headers(), "Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        buf = ""
        while True:
            chunk = resp.read(256)
            if not chunk:
                return
            buf += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                for raw in block.splitlines():
                    if raw.startswith("data: "):
                        try:
                            yield json.loads(raw[6:])
                        except json.JSONDecodeError:
                            pass


def stream_job(server, job_id):
    """
    Subscribe to /api/v1/jobs/<id>/stream and render ReAct events.

      thinking → update spinner status text (shown in verbose mode)
      action   → print  🎯 [tool]  $ command
      output   → stream raw command output
      result   → print green result box
      done / complete → print completion line, stop
      error    → print error, stop
    """
    url   = server.rstrip("/") + f"/api/v1/jobs/{job_id}/stream"
    width = _term_width()
    t0    = time.time()
    sp    = _Spinner()
    done  = False

    if VERBOSITY >= 1:
        print()
        sp.start()

    try:
        for ev in _sse_iter(url):
            t = ev.get("type", "")

            if DEBUG_MODE:
                sp.pause_print(
                    lambda e=ev: print(DIM(f"  [dbg] {json.dumps(e)[:200]}"), flush=True))

            if t == "thinking":
                content = ev.get("content", "").strip()
                if content:
                    sp.update(text=content)
                    if VERBOSITY >= 2:
                        sp.pause_print(
                            lambda c=content: print(f"  {DIM('💭')} {DIM(c)}", flush=True))

            elif t == "action":
                tool    = ev.get("tool", "?")
                command = ev.get("command", "")
                sp.update(text=f"[{tool}] {command[:50]}")
                if VERBOSITY >= 1:
                    sp.pause_print(
                        lambda tl=tool, cmd=command:
                            print(f"  {B_CYAN('🎯')} {BOLD(tl)}  {DIM('$')} {CYAN(cmd)}",
                                  flush=True))

            elif t == "output":
                data = ev.get("data", "") or ev.get("content", "")
                if data:
                    sp.pause_print(lambda d=data: (
                        sys.stdout.write(d), sys.stdout.flush()))

            elif t == "result":
                content = ev.get("content", "").strip()
                if content:
                    sep = B_GREEN("━" * min(60, width))
                    def _show_result(c=content):
                        print(f"\n{sep}")
                        print(B_GREEN(BOLD("  ✅  RESULT")))
                        print(sep)
                        print(c)
                        print()
                    sp.pause_print(_show_result)

            elif t in ("done", "complete"):
                status = ev.get("status", "completed")
                el_s   = f"{int(time.time() - t0)}s"
                col    = B_GREEN if status == "completed" else B_RED
                icon   = "✅" if status == "completed" else "❌"
                sp.stop()
                print(f"  {col(icon + '  ' + status.upper())}  {DIM(el_s)}", flush=True)
                done = True
                break

            elif t == "error":
                msg = ev.get("msg", ev.get("error", "?"))
                sp.stop()
                print(f"\n  {B_RED('Error:')} {msg}", flush=True)
                done = True
                break

    except urllib.error.URLError as e:
        sp.stop()
        print(RED(f"\n[Connection error] {e.reason}"), file=sys.stderr)
    except KeyboardInterrupt:
        sp.stop()
        print(f"\n  {DIM('Interrupted — job may still be running.')}")
        print(f"  {DIM('Cancel:')}  python run_me.py cancel {job_id}")
    finally:
        if not done:
            sp.stop()


def stream_quick(server, question=None, command=None, timeout=15):
    """
    Stream via POST /api/v1/quick/stream.
    Events: start → output (lines) → done | error
    """
    payload = {"timeout": timeout}
    if question:
        payload["question"] = question
    elif command:
        payload["command"]  = command
    else:
        print(RED("  quick: provide --ask <question> or a command"), file=sys.stderr)
        return

    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        server.rstrip("/") + "/api/v1/quick/stream",
        data=data,
        headers={**_headers(), "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
            buf = ""
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    for raw in block.splitlines():
                        if not raw.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(raw[6:])
                        except json.JSONDecodeError:
                            continue
                        t = ev.get("type", "")
                        if t == "start":
                            print(f"  {DIM('$')} {CYAN(ev.get('command', ''))}", flush=True)
                            print()
                        elif t == "output":
                            sys.stdout.write(ev.get("data", ""))
                            sys.stdout.flush()
                        elif t == "done":
                            rc  = ev.get("returncode", 0)
                            ok  = ev.get("success", rc == 0)
                            el  = ev.get("elapsed_ms", int((time.time() - t0) * 1000))
                            col = B_GREEN if ok else B_RED
                            print(f"\n  {col('✅' if ok else '❌')}  exit {rc}  "
                                  f"{DIM(str(el) + 'ms')}", flush=True)
                            return
                        elif t == "error":
                            print(f"\n  {B_RED('Error:')} {ev.get('msg', '?')}", flush=True)
                            return
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}"), file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n  {DIM('Interrupted.')}")


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_health(server):
    h   = _get(server, "/health")
    st  = h.get("status", "?")
    col = B_GREEN if st == "ok" else B_RED
    print(f"\n  Status:   {col(st.upper())}")
    print(f"  Version:  {YELLOW(h.get('version', '?'))}")
    print(f"  Jobs:     active={h.get('active_jobs', 0)}  queued={h.get('queued_jobs', 0)}")
    model = h.get("model", "")
    if model:
        print(f"  Model:    {DIM(model)}")
    feats = h.get("features", [])
    if feats:
        print(f"\n  Features:")
        for f in feats:
            print(f"    {DIM('▸')} {f}")
    print()


def cmd_ask(server, prompt, chain_mode=False, budget=200, no_stream=False):
    if chain_mode:
        print(f"  {CYAN('Chain:')} {prompt[:80]}", flush=True)
        d   = _post(server, "/api/v1/chains", {"goal": prompt, "total_budget": budget})
        cid = d.get("chain_id", "")
        print(f"  {DIM('chain_id')} = {YELLOW(cid)}")
        print(f"\n  Phases:")
        for st in d.get("subtasks", []):
            print(f"    {DIM(str(st['index']) + '.')} {st['instruction'][:80]}")
        print(f"\n  {DIM('Status:')}  python run_me.py chain-status {cid}")
        return

    print(f"  {DIM('→')} {prompt[:80]}{'…' if len(prompt) > 80 else ''}", flush=True)
    resp   = _post(server, "/api/v1/execute", {"instruction": prompt})
    job_id = resp["job_id"]
    print(f"  {DIM('job_id')} = {YELLOW(job_id)}\n", flush=True)

    if no_stream:
        deadline = time.time() + POLL_TIMEOUT
        spin     = ["|", "/", "-", "\\"]
        tick     = 0
        while time.time() < deadline:
            j  = _get(server, f"/api/v1/jobs/{job_id}")
            st = j.get("status", "?")
            if st not in ("queued", "running"):
                out = (j.get("output") or "").strip()
                if out:
                    print(out)
                icon = STATUS_ICON.get(st, "❓")
                print(f"\n  {icon}  {st.upper()}")
                return
            tick += 1
            sys.stdout.write(f"{CLEAR}  {spin[tick % 4]}  [{YELLOW(st)}]")
            sys.stdout.flush()
            time.sleep(POLL_INTERVAL)
    else:
        stream_job(server, job_id)


def cmd_quick(server, question=None, command=None, timeout=15):
    if question:
        print(f"  {DIM('?')} {question[:80]}", flush=True)
    stream_quick(server, question=question, command=command, timeout=timeout)


def cmd_jobs(server, limit=20):
    data = _get(server, "/api/v1/jobs", params={"limit": limit})
    jobs = data.get("jobs", [])
    if not jobs:
        print(f"  {DIM('(no jobs)')}")
        return
    print()
    for j in jobs:
        st    = j.get("status", "?")
        jid   = j.get("job_id", "")
        instr = (j.get("instruction") or "")[:60]
        pct   = 100 if st == "completed" else (66 if st == "running" else
                (33 if st == "queued" else 0))
        bar   = _progress_bar(pct, width=16, done=(st == "completed"))
        col   = B_GREEN if st == "completed" else (B_RED if st == "failed" else YELLOW)
        print(f"  {YELLOW(jid[:8])}  {bar}  {col(st[:10]):12s}  {DIM(instr)}")
    print()


def cmd_job(server, job_id):
    j    = _get(server, f"/api/v1/jobs/{job_id}")
    st   = j.get("status", "?")
    icon = STATUS_ICON.get(st, "❓")
    col  = B_GREEN if st == "completed" else (B_RED if st == "failed" else YELLOW)
    print(f"\n  {icon}  {col(st.upper())}  {DIM(job_id[:8])}")
    instr = (j.get("instruction") or "").strip()
    if instr:
        print(f"  {DIM('Task:')} {instr[:100]}")
    out = (j.get("output") or "").strip()
    if out:
        print(f"\n{out[-4000:]}")
    err = j.get("error", "")
    if err:
        print(f"\n  {RED('Error:')} {err}")
    print()


def cmd_cancel(server, job_id):
    _delete(server, f"/api/v1/jobs/{job_id}")
    print(f"  {YELLOW('Cancelled')} {job_id[:8]}")


def cmd_chains(server):
    d      = _get(server, "/api/v1/chains")
    chains = d.get("chains", [])
    if not chains:
        print(f"  {DIM('(no chains)')}")
        return
    print()
    for c in chains:
        st    = c.get("status", "?")
        icon  = STATUS_ICON.get(st, "❓")
        phase = c.get("current_subtask_index", 0)
        total = c.get("subtask_count", "?")
        col   = B_GREEN if st == "completed" else YELLOW
        print(f"  {icon} {YELLOW(c.get('chain_id','')[:8])}  {col(st[:10]):12s}  "
              f"phase {phase}/{total}  {DIM(c.get('goal','')[:60])}")
    print()


def cmd_chain_status(server, chain_id):
    d    = _get(server, f"/api/v1/chains/{chain_id}")
    st   = d.get("status", "?")
    icon = STATUS_ICON.get(st, "❓")
    col  = B_GREEN if st == "completed" else YELLOW
    print(f"\n  {icon}  {col(st.upper())}  {DIM(d.get('chain_id','')[:8])}")
    print(f"     Goal:  {d.get('goal','')[:100]}")
    subs = d.get("subtasks", [])
    print(f"     Phase: {d.get('current_subtask_index', 0)} / {len(subs)}\n")
    for s in subs:
        si  = STATUS_ICON.get(s.get("status", "pending"), "❓")
        art = (s.get("artifact") or {}).get("summary", "")[:60]
        print(f"    {si} [{s.get('index','?')}] {s.get('instruction','')[:70]}")
        if art:
            print(f"         {DIM('→')} {art}")
    print()


def cmd_sentinel(server):
    d        = _get(server, "/api/v1/blueteam/status")
    watching = d.get("watching", False)
    col      = B_GREEN if watching else DIM
    print(f"\n  SENTINEL  {col('👁  ACTIVE' if watching else '  IDLE')}")
    print(f"    Threat level:  {d.get('threat_level', '?')}")
    print(f"    Recent alerts: {d.get('recent_alert_count', 0)}")
    print(f"    Last scan:     {'✅' if d.get('last_scan_success') else '—'}")
    summary = d.get("last_scan_summary", "")
    if summary:
        print(f"\n  Last report:\n    {summary[:300]}")
    print()


def cmd_scan(server, focus=""):
    label = f"  focus: {focus}" if focus else ""
    print(f"  {CYAN('SENTINEL scan')}{label}...", flush=True)
    d      = _post(server, "/api/v1/blueteam/scan", {"focus": focus})
    job_id = d["job_id"]
    print(f"  {DIM('job_id')} = {YELLOW(job_id)}\n")
    stream_job(server, job_id)


def cmd_report(server):
    req = urllib.request.Request(
        server.rstrip("/") + "/api/v1/blueteam/report", headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            try:
                d = json.loads(body)
                print(d.get("report", body))
            except json.JSONDecodeError:
                print(body)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  {DIM('No report yet')} — run: python run_me.py scan")
        else:
            print(RED(f"[HTTP {e.code}]"), file=sys.stderr)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}"), file=sys.stderr)


def cmd_alerts(server, n=20):
    d      = _get(server, "/api/v1/blueteam/alerts", params={"n": n})
    alerts = d.get("alerts", [])
    if not alerts:
        print(f"  {DIM('No alerts on record.')}")
        return
    print()
    for a in alerts:
        sev = a.get("severity", "?")
        ts  = a.get("ts", "")[:19]
        ev  = (a.get("evidence") or "").strip()
        print(f"  {SEV_ICON.get(sev,'❓')} [{DIM(ts)}] [{YELLOW(sev):8s}] {a.get('finding','')}")
        if ev:
            for line in ev.splitlines()[:3]:
                print(f"               {DIM(line)}")
    print()


# ── Interactive REPL ───────────────────────────────────────────────────────────

_REPL_HELP = """\
  Commands (prefix with :)
  ─────────────────────────────────────────────────────────────
  :health                  service health check
  :jobs                    list recent jobs
  :job <id>                show job output
  :cancel <id>             cancel a job
  :watch <id>              live-stream a running job's events
  :quick <cmd>             run a shell command directly (streaming)
  :ask <question>          NL→command via quick endpoint
  :chain <goal>            submit a multi-phase chain
  :chains                  list chains
  :chain-status <id>       chain detail
  :sentinel                SENTINEL watcher status
  :scan [focus]            run security scan
  :report                  daily security report
  :alerts                  recent security alerts
  :verbose [0|1|2]         cycle verbosity (0=quiet, 1=normal, 2=verbose)
  :debug                   toggle raw SSE debug output
  :help                    show this help
  :quit / q                exit

  Anything else is submitted as a task (full ReAct loop + live stream).
"""

def cmd_repl(server):
    global VERBOSITY, DEBUG_MODE
    width = _term_width()
    sep   = DIM("─" * min(60, width))
    print(f"\n{BOLD('Jarvis CMD')}  {DIM('v2.0')}  {DIM(server)}")
    print(f"  verbosity={VERBOSITY}  debug={DEBUG_MODE}")
    print(sep)
    print(f"  Type a task to run it, or {CYAN(':help')} for commands.")
    print(sep + "\n")

    while True:
        try:
            line = input(f"{CYAN('jarvis')}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue

        if line in (":quit", ":exit", "quit", "exit", "q"):
            print("Bye.")
            break
        elif line == ":help":
            print(_REPL_HELP)
        elif line == ":health":
            cmd_health(server)
        elif line == ":jobs":
            cmd_jobs(server)
        elif line.startswith(":job "):
            cmd_job(server, line.split(None, 1)[1].strip())
        elif line.startswith(":cancel ") or line.startswith(":kill "):
            cmd_cancel(server, line.split(None, 1)[1].strip())
        elif line.startswith(":watch ") or line == ":watch":
            parts = line.split()
            if len(parts) >= 2:
                try:
                    stream_job(server, parts[1])
                except KeyboardInterrupt:
                    print(f"\n  {DIM('Detached.')}")
            else:
                print(f"  Usage: :watch <job_id>")
        elif line.startswith(":quick "):
            cmd_quick(server, command=line[7:].strip())
        elif line.startswith(":ask "):
            cmd_quick(server, question=line[5:].strip())
        elif line.startswith(":chain "):
            cmd_ask(server, line[7:].strip(), chain_mode=True)
        elif line == ":chains":
            cmd_chains(server)
        elif line.startswith(":chain-status "):
            cmd_chain_status(server, line.split(None, 1)[1].strip())
        elif line == ":sentinel":
            cmd_sentinel(server)
        elif line == ":scan" or line.startswith(":scan "):
            focus = line[5:].strip()
            cmd_scan(server, focus=focus)
        elif line == ":report":
            cmd_report(server)
        elif line == ":alerts":
            cmd_alerts(server)
        elif line.startswith(":verbose"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                VERBOSITY = max(0, min(2, int(parts[1])))
            else:
                VERBOSITY = (VERBOSITY + 1) % 3
            labels = {0: "quiet (answer only)", 1: "normal (spinner+events)",
                      2: "verbose (all events)"}
            print(f"  Verbosity: {VERBOSITY} — {labels[VERBOSITY]}")
        elif line == ":debug":
            DEBUG_MODE = not DEBUG_MODE
            print(f"  Debug: {'ON' if DEBUG_MODE else 'OFF'}")
        elif line.startswith(":"):
            print(f"  {RED('Unknown:')} {line.split()[0]}  "
                  f"(type {CYAN(':help')} for commands)")
        else:
            try:
                cmd_ask(server, line)
            except KeyboardInterrupt:
                print(f"\n  {DIM('Interrupted.')}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    p = argparse.ArgumentParser(
        prog="run_me.py",
        description="Jarvis CMD Agent v2.0 — tasks, commands, security.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--server", "-s", default=DEFAULT_SERVER,
                   help=f"API base URL (default: {DEFAULT_SERVER})")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="Interactive REPL")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose — show all events including thinking")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Quiet — answer only, no spinner")
    p.add_argument("-d", "--debug", action="store_true",
                   help="Print raw SSE JSON for every event")
    p.add_argument("--no-stream", action="store_true",
                   help="Poll instead of streaming")
    p.add_argument("--budget", type=int, default=200,
                   help="Chain iteration budget (default: 200)")
    p.add_argument("args", nargs="*")
    opts = p.parse_args()

    global VERBOSITY, DEBUG_MODE
    server = opts.server.rstrip("/")
    if opts.quiet:
        VERBOSITY = 0
    elif opts.verbose:
        VERBOSITY = 2
    DEBUG_MODE = opts.debug

    if opts.interactive:
        cmd_repl(server)
        return

    args = opts.args
    if not args:
        p.print_help()
        return

    cmd  = args[0].lower()
    rest = " ".join(args[1:])

    try:
        if   cmd == "health":                    cmd_health(server)
        elif cmd == "jobs":                      cmd_jobs(server)
        elif cmd == "job":
            if not rest: p.error("Usage: run_me.py job <id>")
            cmd_job(server, rest.strip())
        elif cmd == "cancel":
            if not rest: p.error("Usage: run_me.py cancel <id>")
            cmd_cancel(server, rest.strip())
        elif cmd == "quick":
            if "--ask" in args:
                idx = args.index("--ask")
                cmd_quick(server, question=" ".join(args[idx + 1:]))
            else:
                cmd_quick(server, command=rest)
        elif cmd == "chain":
            if not rest: p.error("Usage: run_me.py chain \"goal\"")
            cmd_ask(server, rest, chain_mode=True, budget=opts.budget)
        elif cmd == "chains":                    cmd_chains(server)
        elif cmd in ("chain-status", "chain_status"):
            if not rest: p.error("Usage: run_me.py chain-status <id>")
            cmd_chain_status(server, rest.strip())
        elif cmd == "sentinel":                  cmd_sentinel(server)
        elif cmd == "scan":                      cmd_scan(server, focus=rest)
        elif cmd == "report":                    cmd_report(server)
        elif cmd == "alerts":                    cmd_alerts(server)
        else:
            # Treat as a task/instruction
            cmd_ask(server, " ".join(args), no_stream=opts.no_stream)

    except (urllib.error.URLError, ConnectionRefusedError):
        sys.exit(f"  Cannot reach {server} — is ollama-cmd running?")
    except KeyboardInterrupt:
        print("\n  Aborted.")


if __name__ == "__main__":
    main()
