#!/usr/bin/env python3
"""
Swarm 3.0 -- Client Entry Point
Connects to a remote Swarm API server (default: http://10.0.0.58:5002).

Usage:
  python3 run_me.py "question"              # single question (sync)
  python3 run_me.py stream "question"       # live SSE stream with progress bar
  python3 run_me.py -i                      # interactive REPL
  python3 run_me.py health                  # server health
  python3 run_me.py status                  # job queue status
  python3 run_me.py jobs                    # list recent jobs
  python3 run_me.py result <job_id>         # fetch async result
  python3 run_me.py ask "question"          # async submit + poll

Environment:
  SWARM_SERVER   Base URL  (default: http://10.0.0.58:5002)
  SWARM_API_KEY  Bearer token (optional)
"""

import sys, os, json, time, textwrap, argparse
import urllib.request, urllib.error, urllib.parse

DEFAULT_SERVER = os.environ.get("SWARM_SERVER", "http://10.0.0.58:5002")
API_KEY        = os.environ.get("SWARM_API_KEY", "")
POLL_INTERVAL  = 4
POLL_TIMEOUT   = 600

# Verbosity: 0=quiet (answer only), 1=normal (phase+answer), 2=verbose (all events)
VERBOSITY  = 1
DEBUG_MODE = False   # print raw SSE JSON when True

# -- ANSI helpers --------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else True

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BOLD    = lambda t: _c("1",  t)
DIM     = lambda t: _c("2",  t)
GREEN   = lambda t: _c("32", t)
YELLOW  = lambda t: _c("33", t)
CYAN    = lambda t: _c("36", t)
RED     = lambda t: _c("31", t)
BLUE    = lambda t: _c("34", t)
MAGENTA = lambda t: _c("35", t)
CLEAR   = "\033[2K\r" if _USE_COLOR else ""

# Phase colour map
_PHASE_COLORS = {
    "0A": CYAN, "0B": CYAN,
    "1":  BLUE,
    "2":  YELLOW,
    "3":  MAGENTA, "4": MAGENTA,
    "5":  GREEN,
}

def _phase_color(phase_id: str):
    for k, fn in _PHASE_COLORS.items():
        if phase_id.startswith(k):
            return fn
    return CYAN

# -- HTTP helpers --------------------------------------------------------------

def _headers():
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h

def _get(server, path):
    req = urllib.request.Request(server.rstrip("/") + path, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(RED(f"[HTTP {e.code}] {e.read().decode()}"), file=sys.stderr); sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}  server={server}"), file=sys.stderr); sys.exit(1)

def _post(server, path, payload, timeout=180):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(server.rstrip("/") + path,
                                  data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(RED(f"[HTTP {e.code}] {e.read().decode()}"), file=sys.stderr); sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}"), file=sys.stderr); sys.exit(1)

# -- Pretty printer ------------------------------------------------------------

def _print_answer(result: dict):
    answer  = result.get("answer") or result.get("result") or result.get("response", "")
    status  = result.get("status", "")
    job_id  = result.get("job_id", "")
    elapsed = result.get("elapsed") or result.get("elapsed_seconds")

    if job_id:
        print(f"\n{DIM('--- Job')} {BOLD(job_id)} {DIM('---')}")
    if status and status not in ("completed", "done"):
        print(f"Status : {YELLOW(status)}")
    if elapsed is not None:
        print(f"Time   : {GREEN(f'{float(elapsed):.1f}s')}")
    print()

    if answer:
        width = min(100, _term_width())
        for line in answer.splitlines():
            if len(line) > width:
                print(textwrap.fill(line, width=width))
            else:
                print(line)
    else:
        print(json.dumps(result, indent=2))
    print()

def _term_width():
    try:
        return os.get_terminal_size(fallback=(100, 40)).columns
    except Exception:
        return 100

# -- Commands ------------------------------------------------------------------

def cmd_health(server):
    data = _get(server, "/health")
    orch = GREEN("available") if data.get("orchestrator_available") else RED("unavailable")
    auth = YELLOW("on") if data.get("auth_enabled") else DIM("off")
    print(f"\n  Server  : {GREEN('healthy')}")
    print(f"  Orch    : {orch}")
    print(f"  Auth    : {auth}")
    print(f"  Time    : {data.get('timestamp','')}\n")

def cmd_status(server):
    data = _get(server, "/status")
    jobs = data.get("jobs", {})
    cfg  = data.get("config", {})
    print(f"\n  Pending    : {jobs.get('pending',0)}")
    print(f"  Processing : {YELLOW(str(jobs.get('processing',0)))}")
    print(f"  Completed  : {GREEN(str(jobs.get('completed',0)))}")
    print(f"  Failed     : {RED(str(jobs.get('failed',0)))}")
    print(f"  Max conc.  : {cfg.get('max_concurrent',3)}")
    print(f"  SearXNG    : {'yes' if cfg.get('searxng') else 'no'}\n")

def cmd_jobs(server):
    data = _get(server, "/jobs")
    jobs = data.get("jobs", [])
    if not jobs:
        print(DIM("  (no jobs)"))
        return
    for j in sorted(jobs, key=lambda x: x.get("created_at",""), reverse=True)[:20]:
        st = j.get("status","?")
        col = {"completed": GREEN, "failed": RED, "processing": YELLOW}.get(st, DIM)
        q  = (j.get("question","")[:55] + "...") if len(j.get("question","")) > 55 else j.get("question","")
        el = f"  {j.get('elapsed',''):.0f}s" if j.get("elapsed") else ""
        print(f"  {BOLD(j.get('job_id','?'))}  [{col(st):10s}]  {q}{DIM(el)}")
    print()

def cmd_result(server, job_id):
    data = _get(server, f"/result/{job_id}")
    # Show last few progress log lines
    log  = data.get("progress_log", [])
    if log:
        print(DIM("\n  Recent progress:"))
        for l in log[-5:]:
            print(DIM(f"    {l}"))
    _print_answer(data)

def cmd_ask(server, question):
    """Submit async and poll until done with a simple spinner."""
    print(f"Submitting: {BOLD(question[:70])}...")
    resp = _post(server, "/query_async", {"question": question})
    job_id = resp.get("job_id")
    if not job_id:
        print(RED("[Error] No job_id:"), resp, file=sys.stderr); sys.exit(1)

    print(f"Job ID: {BOLD(job_id)}  (polling every {POLL_INTERVAL}s)")
    deadline = time.time() + POLL_TIMEOUT
    spin     = ["|", "/", "-", "\\"]
    tick     = 0

    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        tick += 1
        data = _get(server, f"/result/{job_id}")
        st   = data.get("status","?")
        prog = data.get("progress","") or ""
        # trim progress to terminal width
        prog_short = prog[:_term_width()-30] if prog else ""
        print(f"{CLEAR}  {spin[tick%len(spin)]}  [{YELLOW(st)}]  {DIM(prog_short)}",
              end="", flush=True)
        if st in ("completed","done","failed","error"):
            print()
            _print_answer(data)
            return

    print(f"\n{RED('[Timeout]')} Job {job_id} not done after {POLL_TIMEOUT}s", file=sys.stderr)
    sys.exit(1)

def cmd_query(server, question):
    """Synchronous query."""
    print(f"Asking: {BOLD(question[:70])}...", flush=True)
    resp = _post(server, "/query", {"question": question}, timeout=600)
    _print_answer(resp)


# -- Streaming command ---------------------------------------------------------

def cmd_stream(server: str, question: str):
    """
    Connect to /query_stream (SSE) and display a live progress UI.

    Verbosity levels (set via --verbose / :verbose in REPL):
      0 = quiet   — answer only, no progress
      1 = normal  — live spinner with phase/tok/s (default)
      2 = verbose — every event printed as its own line
    Debug mode (--debug / :debug): also prints raw SSE JSON for each event.
    """
    global VERBOSITY, DEBUG_MODE

    url  = server.rstrip("/") + "/query_stream"
    data = json.dumps({"question": question}).encode()
    hdrs = {**_headers(), "Accept": "text/event-stream"}
    req  = urllib.request.Request(url, data=data, headers=hdrs, method="POST")

    width        = _term_width()
    phase_name   = "Initializing"
    phase_id     = ""
    agent_name   = ""
    model_name   = ""
    toks_ps      = 0.0
    total_tokens = 0
    elapsed      = 0.0
    t0           = time.time()
    last_toks_t  = 0.0
    spin         = ["|", "/", "-", "\\"]
    tick         = 0
    last_log     = ""

    if VERBOSITY >= 1:
        print(f"\n{BOLD('Swarm 3.0')}  {DIM(question[:80])}")
        print(DIM("-" * 60))

    # ── spinner/status line (VERBOSITY == 1) ─────────────────────────────────
    def _draw():
        nonlocal tick
        if VERBOSITY != 1:
            return
        tick += 1
        s = spin[tick % len(spin)]
        phase_col = _phase_color(phase_id) if phase_id else CYAN
        phase_str = phase_col(f"Phase {phase_id}: {phase_name}") if phase_id else CYAN(phase_name)
        toks_str  = (f"{GREEN(f'{toks_ps:.1f}')} tok/s"
                     if toks_ps > 0 and (time.time()-last_toks_t) < 10
                     else DIM("-- tok/s"))
        agent_str = (f"{DIM(agent_name)}{DIM('@')}{DIM(model_name)}"
                     if agent_name else DIM("waiting"))
        status_line = f"  {s} {phase_str}  {toks_str}  {DIM(f'{elapsed:.0f}s')}  {agent_str}"
        log_line    = f"  {DIM(last_log[:width-4])}" if last_log else ""
        if _USE_COLOR:
            print(f"\033[2A" if last_log else "", end="")
            print(f"{CLEAR}{status_line}", flush=True)
            if last_log:
                print(f"{CLEAR}{log_line}", flush=True)
        else:
            print(f"  [{elapsed:.0f}s] {phase_name} {toks_ps:.1f} tok/s  {last_log[:60]}",
                  flush=True)

    def _set_last_log(line: str):
        nonlocal last_log
        clean = line.strip()
        if set(clean) <= set("=- \t"):
            return
        if len(clean) > 4:
            last_log = clean

    # ── verbose line printer (VERBOSITY >= 2) ────────────────────────────────
    def _vprint(label: str, msg: str, color_fn=None):
        if VERBOSITY < 2:
            return
        tag = color_fn(label) if color_fn else DIM(label)
        print(f"  {tag}  {msg}", flush=True)

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            if VERBOSITY == 1:
                print()
                if _USE_COLOR:
                    print()

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
                            evt = json.loads(raw[6:])
                        except json.JSONDecodeError:
                            continue

                        if DEBUG_MODE:
                            print(DIM(f"  [dbg] {json.dumps(evt)[:200]}"), flush=True)

                        etype   = evt.get("type", "")
                        elapsed = evt.get("elapsed", time.time()-t0)

                        if etype == "start":
                            phase_name = "Starting..."
                            _vprint("start    ", f"job={evt.get('job_id','?')}", CYAN)
                            _draw()

                        elif etype == "phase":
                            phase_id   = evt.get("phase_id", "")
                            phase_name = evt.get("phase_name", "")
                            last_log   = ""
                            _vprint(f"phase {phase_id:<4}",
                                    f"[{elapsed:.0f}s] {phase_name}",
                                    _phase_color(phase_id))
                            _draw()

                        elif etype == "toks":
                            toks_ps      = evt.get("toks_per_sec", 0.0)
                            total_tokens += evt.get("tokens", 0)
                            last_toks_t  = time.time()
                            _vprint("toks     ",
                                    f"{evt.get('tokens')} tok  {toks_ps:.1f} tok/s  "
                                    f"{evt.get('seconds')}s",
                                    GREEN)
                            _draw()

                        elif etype == "agent":
                            agent_name = evt.get("agent", "")
                            model_name = evt.get("model", "")
                            _vprint("agent    ", f"{agent_name} @ {model_name}", MAGENTA)
                            _draw()

                        elif etype == "log":
                            line = evt.get("line", "")
                            _set_last_log(line)
                            _vprint("log      ", line[:width-20])
                            _draw()

                        elif etype == "error_line":
                            line = evt.get("line", "")
                            _set_last_log(RED("!") + " " + line)
                            _vprint("error    ", line[:width-20], RED)
                            _draw()

                        elif etype == "heartbeat":
                            _vprint("heartbeat", f"[{elapsed:.0f}s]", DIM)
                            _draw()

                        elif etype == "answer":
                            # Clear spinner line before printing answer
                            if VERBOSITY == 1 and _USE_COLOR:
                                print(f"\033[2A{CLEAR}\n{CLEAR}", end="")
                            print(DIM("-" * 60))
                            print(f"{GREEN('Done')}  "
                                  f"{DIM(f'elapsed={elapsed:.1f}s')}  "
                                  f"{DIM(f'tokens~{total_tokens}')}")
                            print(DIM("-" * 60))
                            print()
                            answer = evt.get("answer", "")
                            w = min(100, width)
                            for ln in answer.splitlines():
                                print(textwrap.fill(ln, width=w) if len(ln) > w else ln)
                            print()

                        elif etype == "error":
                            if VERBOSITY >= 1 and _USE_COLOR:
                                print(f"\033[2A{CLEAR}\n{CLEAR}", end="")
                            print(f"\n{RED('Error:')} {evt.get('error','unknown')}\n")

                        elif etype == "done":
                            return

    except urllib.error.HTTPError as e:
        print(RED(f"\n[HTTP {e.code}] {e.read().decode()}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"\n[Connection error] {e.reason}"), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW('Interrupted.')}")


# -- REPL ---------------------------------------------------------------------

def cmd_repl(server):
    global VERBOSITY, DEBUG_MODE
    print(f"{BOLD('Swarm 3.0 REPL')}  (server: {server})")
    print(f"Verbosity: {VERBOSITY}  Debug: {DEBUG_MODE}")
    print("Commands: :health :status :jobs :stream <q> :ask <q>")
    print("          :verbose [0|1|2]  :debug [on|off]  :quit\n")
    while True:
        try:
            line = input(f"{CYAN('swarm')}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not line: continue
        if line in (":quit",":exit","quit","exit","q"):
            print("Bye."); break
        elif line == ":health":   cmd_health(server)
        elif line == ":status":   cmd_status(server)
        elif line == ":jobs":     cmd_jobs(server)
        elif line.startswith(":stream "): cmd_stream(server, line[8:].strip())
        elif line.startswith(":ask "):    cmd_ask(server, line[5:].strip())
        elif line.startswith(":verbose"):
            parts = line.split()
            if len(parts) == 2 and parts[1].isdigit():
                VERBOSITY = max(0, min(2, int(parts[1])))
            else:
                VERBOSITY = (VERBOSITY + 1) % 3   # cycle through 0→1→2→0
            labels = {0: "quiet (answer only)", 1: "normal (spinner)", 2: "verbose (all events)"}
            print(f"  Verbosity set to {VERBOSITY} — {labels[VERBOSITY]}")
        elif line.startswith(":debug"):
            parts = line.split()
            if len(parts) == 2:
                DEBUG_MODE = parts[1].lower() in ("on", "1", "true", "yes")
            else:
                DEBUG_MODE = not DEBUG_MODE
            print(f"  Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")
        else:
            if _USE_COLOR:
                cmd_stream(server, line)
            else:
                cmd_query(server, line)


# -- CLI ----------------------------------------------------------------------

def main():
    # Ensure stdout is line-buffered even when piped
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(
        prog="run_me.py",
        description="Swarm 3.0 client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 run_me.py stream "What is the specific impulse of HTPB?"
              python3 run_me.py "What is 2+2?"
              python3 run_me.py -i
              python3 run_me.py health
              python3 run_me.py ask "Design a 500N thrust motor"
              python3 run_me.py result abc123
        """),
    )
    parser.add_argument("--server", "-s", default=DEFAULT_SERVER,
                        help=f"API base URL (default: {DEFAULT_SERVER})")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v = verbose, -vv = not used; "
                             "default is normal spinner mode). Use 0/1/2 in REPL with :verbose.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Verbosity 0: print answer only, no progress")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print raw SSE JSON for every event")
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()
    server = opts.server.rstrip("/")

    global VERBOSITY, DEBUG_MODE
    if opts.quiet:
        VERBOSITY = 0
    elif opts.verbose:
        VERBOSITY = min(2, 1 + opts.verbose)
    DEBUG_MODE = opts.debug

    if opts.interactive:
        cmd_repl(server); return

    args = opts.args
    if not args:
        parser.print_help(); return

    cmd = args[0].lower()

    if   cmd == "health":  cmd_health(server)
    elif cmd == "status":  cmd_status(server)
    elif cmd == "jobs":    cmd_jobs(server)
    elif cmd == "result":
        if len(args) < 2:
            print("Usage: run_me.py result <job_id>", file=sys.stderr); sys.exit(1)
        cmd_result(server, args[1])
    elif cmd == "ask":
        q = " ".join(args[1:])
        if not q: print("Usage: run_me.py ask \"question\"", file=sys.stderr); sys.exit(1)
        cmd_ask(server, q)
    elif cmd == "stream":
        q = " ".join(args[1:])
        if not q: print("Usage: run_me.py stream \"question\"", file=sys.stderr); sys.exit(1)
        cmd_stream(server, q)
    else:
        question = " ".join(args)
        if _USE_COLOR:
            cmd_stream(server, question)
        else:
            cmd_query(server, question)


if __name__ == "__main__":
    main()
