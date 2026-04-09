#!/usr/bin/env python3
"""
Swarm 3.7 -- Client Entry Point
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

import sys, os, json, time, textwrap, argparse, re, threading
import urllib.request, urllib.error, urllib.parse

DEFAULT_SERVER = os.environ.get("SWARM_SERVER", "http://10.0.0.58:5002")
API_KEY        = os.environ.get("SWARM_API_KEY", "")
POLL_INTERVAL  = 4
POLL_TIMEOUT   = 600

# Verbosity: 0=quiet (answer only), 1=normal (phase+answer), 2=verbose (all events)
VERBOSITY     = 1
DEBUG_MODE    = False   # print raw SSE JSON when True
STREAM_TOKENS = False   # -t/--tokens: stream raw LLM output token-by-token

# -- ANSI helpers --------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else True

def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BOLD       = lambda t: _c("1",   t)
DIM        = lambda t: _c("2",   t)
GREEN      = lambda t: _c("32",  t)
YELLOW     = lambda t: _c("33",  t)
CYAN       = lambda t: _c("36",  t)
RED        = lambda t: _c("31",  t)
BLUE       = lambda t: _c("34",  t)
MAGENTA    = lambda t: _c("35",  t)
PURPLE     = lambda t: _c("95",  t)   # bright magenta  — new SP turn
LIGHT_BLUE = lambda t: _c("94",  t)   # bright blue     — code blocks
B_GREEN    = lambda t: _c("92",  t)   # bright green    — SOLVED / RESULT
B_RED      = lambda t: _c("91",  t)   # bright red      — FAILED / errors
CLEAR   = "\033[2K\r" if _USE_COLOR else ""

def _colorize_llm_line(line: str, in_code: bool) -> tuple:
    """Return (colored_line, new_in_code_state).  Called once per complete line."""
    s = line.rstrip()
    # Code fence toggle
    if s.startswith("```"):
        new_in_code = not in_code
        return LIGHT_BLUE(s), new_in_code
    if in_code:
        return LIGHT_BLUE(s), True
    # Structural ReAct markers
    if s.startswith("THOUGHT:"):
        return DIM(s), False
    if s.startswith("ACTION:"):
        return PURPLE(BOLD("ACTION:")) + PURPLE(s[7:]), False
    if s.startswith("INPUT:"):
        return LIGHT_BLUE(s), False
    if s.startswith("END_INPUT") or s.startswith("END_ANSWER"):
        return DIM(s), False
    # Locked ledger / results
    if s.startswith("🔒 LOCKED RESULTS") or s.startswith("LOCKED RESULTS"):
        return B_GREEN(BOLD(s)), False
    if re.match(r'\s{2}\w+ = ', s) and not s.strip().startswith("#"):
        # indented ledger entries like "  r0 = 1.186 m"
        return B_GREEN(s), False
    # RESULT lines
    if s.startswith("RESULT:"):
        return B_GREEN(BOLD(s)), False
    # Final answer section
    if s.startswith("FINAL_ANSWER:"):
        return B_GREEN(BOLD("━"*50 + " FINAL ANSWER " + "━"*50)), False
    if s.startswith("STATUS: solved"):
        return B_GREEN(BOLD(s)), False
    if s.startswith("STATUS: failed"):
        return B_RED(BOLD(s)), False
    if s.startswith("STATUS:"):
        return YELLOW(s), False
    # Observation / verification
    if s.startswith("OBSERVATION:"):
        return CYAN(BOLD(s)), False
    if s.startswith("VERIFICATION:"):
        return YELLOW(s), False
    if s.startswith("CODE:"):
        return LIGHT_BLUE(s), False
    # Errors
    if s.startswith("EXECUTION ERROR") or s.startswith("ERROR:") or s.startswith("EXCEPTION"):
        return B_RED(s), False
    return s, False

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

# -- Rich Live panel (optional) -----------------------------------------------

try:
    from rich.live import Live as _RichLive
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.text import Text as _RichText
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


class LiveStatus:
    """Tracks Swarm job progress for the Rich Live panel display."""

    _STATUS_ICON = {'pending': '⏳', 'active': '🔄', 'done': '✅',
                    'solved': '✅', 'failed': '❌', 'timeout': '⏱'}

    def __init__(self, question: str):
        self.question = question[:80]
        self.t0 = time.time()
        self.phases: dict = {}   # id → {name, status, elapsed_s, notes}
        self.sps: dict = {}      # id → {status, turn, max_turn, tool, values}
        self.wave = ""
        self.q_type = ""
        self.n_solved = 0
        self.n_total = 0
        self._last_phase_id = ""

    def on_event(self, ev: dict):
        t = ev.get('type', '')

        if t == 'phase':
            pid  = ev.get('phase_id', '')
            name = ev.get('phase_name', '')
            el   = ev.get('elapsed_s') or ev.get('elapsed', 0)
            if self._last_phase_id and self._last_phase_id in self.phases:
                self.phases[self._last_phase_id]['status'] = 'done'
            self.phases[pid] = {'name': name, 'status': 'active', 'elapsed_s': el, 'notes': ''}
            self._last_phase_id = pid
            if pid.upper() in ('0A',) and name:
                self.q_type = name

        elif t == 'wave':
            w, tw = ev.get('wave', 1), ev.get('total_waves', 1)
            self.wave = f"Wave {w}/{tw}"
            for sp_id in ev.get('sps', []):
                if sp_id not in self.sps:
                    self.sps[sp_id] = {'status': 'pending', 'turn': 0,
                                       'max_turn': 15, 'tool': '—', 'values': '—'}
            self.n_total = max(self.n_total, len(self.sps))

        elif t == 'sp_turn':
            sp_id = ev.get('sp_id', '')
            if sp_id not in self.sps:
                self.sps[sp_id] = {'status': 'active', 'turn': 0,
                                   'max_turn': 15, 'tool': '—', 'values': '—'}
            self.sps[sp_id].update({
                'status': 'active',
                'turn': ev.get('turn', 0),
                'max_turn': ev.get('max_turns', 15),
                'tool': ev.get('tool', '—'),
            })

        elif t == 'sp_done':
            sp_id = ev.get('sp_id', '')
            st    = ev.get('status', 'failed')
            vals  = (ev.get('values') or '—')[:28]
            if sp_id not in self.sps:
                self.sps[sp_id] = {'status': st, 'turn': 0,
                                   'max_turn': 15, 'tool': '—', 'values': vals}
            self.sps[sp_id].update({'status': st, 'values': vals,
                                    'turn': ev.get('turns', 0)})
            if st == 'solved':
                self.n_solved += 1

        elif t == 'solve_done':
            self.n_solved = ev.get('solved', self.n_solved)
            self.n_total  = ev.get('total',  self.n_total)

    def render(self) -> "_RichPanel":
        elapsed = time.time() - self.t0
        mm, ss  = divmod(int(elapsed), 60)

        parts = ["[bold cyan]SWARM 3.2[/bold cyan]"]
        if self.q_type:
            parts.append(f"[dim]{self.q_type}[/dim]")
        if self.wave:
            parts.append(f"[yellow]{self.wave}[/yellow]")
        if self.n_total > 0:
            parts.append(f"[green]{self.n_solved}/{self.n_total}[/green]")
        parts.append(f"[dim]⏱ {mm:02d}:{ss:02d}[/dim]")
        title = "  ".join(parts)

        body = _RichText()
        body.append(f"Q: {self.question}\n", style="bold white")
        body.append("─" * 58 + "\n", style="dim")

        si = self._STATUS_ICON
        for pid, ph in self.phases.items():
            icon = si.get(ph['status'], '·')
            el   = f"  {ph['elapsed_s']:.1f}s" if ph.get('elapsed_s') else ''
            body.append(f"  {icon} ")
            body.append(f"{pid:<4}", style="cyan")
            body.append(f"  {ph['name']:<18}", style="white")
            body.append(f"{el}\n", style="dim green")

        if self.sps:
            body.append("─" * 58 + "\n", style="dim")
            for sp_id, sp in self.sps.items():
                st   = sp.get('status', 'pending')
                icon = si.get(st, '·')
                turn = sp.get('turn', 0)
                mx   = sp.get('max_turn', 15)
                tool = sp.get('tool', '—')[:10]
                vals = sp.get('values', '—')[:26]
                body.append(f"  {icon} ")
                body.append(f"{sp_id:<5}", style="cyan")
                if st == 'active':
                    body.append(f"  {turn:>2}/{mx}  ", style="yellow")
                    body.append(f"{tool:<12}", style="blue")
                elif st in ('solved', 'failed', 'timeout'):
                    turns_s = f"  {turn:>2} turns  " if turn else "           "
                    body.append(turns_s, style="dim")
                    style = "green" if st == 'solved' else "red"
                    body.append(vals, style=style)
                else:
                    body.append("  pending", style="dim")
                body.append("\n")

        return _RichPanel(body, title=title, border_style="cyan", padding=(0, 1))


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

def cmd_logs(server: str, job_id: str, tail: int = None, grep: str = None):
    """Fetch and print the full disk log for a completed or running job."""
    params = []
    if tail:
        params.append(f"tail={tail}")
    if grep:
        params.append(f"grep={urllib.parse.quote(grep)}")
    qs = ("?" + "&".join(params)) if params else ""
    req = urllib.request.Request(
        server.rstrip("/") + f"/logs/{job_id}{qs}",
        headers=_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(r.read().decode(errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 404:
            print(RED(f"No log for job {job_id} — is the job_id correct?"), file=sys.stderr)
        else:
            print(RED(f"[HTTP {e.code}] {body}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"[Connection error] {e.reason}"), file=sys.stderr)
        sys.exit(1)


# -- Watch (reconnect to running job) -----------------------------------------

def cmd_watch(server: str, job_id: str):
    """Tail the disk log of a running or completed job live (like tail -f).

    Polls /logs/<job_id> every 1.5s and prints new lines as they arrive.
    When STREAM_TOKENS is True, [LLMTOK] lines are rendered as raw token text.
    Exits when the job reaches completed/failed.
    """
    import time as _time

    print(f"\n{BOLD('◉')} Watching job {CYAN(job_id)}")
    print(DIM("─" * 60))

    seen = 0

    def _fetch_log():
        req = urllib.request.Request(
            server.rstrip("/") + f"/logs/{job_id}",
            headers=_headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read().decode(errors="replace").splitlines()
        except Exception:
            return None

    def _render_line(raw):
        if raw.startswith("[LLMTOK]"):
            if STREAM_TOKENS:
                content = raw[8:].replace("\\n", "\n").replace("\\\\", "\\")
                sys.stdout.write(content)
                sys.stdout.flush()
            # silently skip when tokens off
            return
        print(raw)

    try:
        while True:
            lines = _fetch_log()
            if lines is not None:
                for raw in lines[seen:]:
                    _render_line(raw)
                seen = len(lines)

            # Check completion
            status = ""
            try:
                result = _get(server, f"/result/{job_id}")
                status = result.get("status", "")
            except Exception:
                pass

            if status in ("completed", "failed"):
                # Drain any final lines before printing answer
                lines = _fetch_log()
                if lines:
                    for raw in lines[seen:]:
                        _render_line(raw)
                print()
                _print_answer(result)
                return

            _time.sleep(1.5)

    except KeyboardInterrupt:
        print(f"\n{DIM('[detached — job still running on server]')}")


# -- Token streaming command ---------------------------------------------------

def cmd_stream_tokens(server: str, question: str):
    """Token streaming mode — color-coded live output."""
    url  = server.rstrip("/") + "/query_stream"
    data = json.dumps({"question": question}).encode()
    hdrs = {**_headers(), "Accept": "text/event-stream"}
    req  = urllib.request.Request(url, data=data, headers=hdrs, method="POST")

    W = 72  # separator width

    def _sep(char="─"):
        return DIM(char * W)

    def _phase_banner(pid, name, el):
        el_s = f"  {DIM(str(el)+'s')}" if el else ""
        tag  = _phase_color(pid)(f" Phase {pid} ")
        return f"\n{'━'*4}{tag}{'━'*(W-8-len(pid)-len(name)-2)} {CYAN(name)}{el_s}"

    def _sp_new_banner(sp_id):
        """Red-outlined banner printed when a NEW sub-problem context starts."""
        bar = B_RED("┌" + "─"*(W-2) + "┐")
        mid = B_RED("│") + RED(BOLD(f"  ⚠  NEW CONTEXT — {sp_id}  (history cleared, fresh system prompt)".center(W-2))) + B_RED("│")
        bot = B_RED("└" + "─"*(W-2) + "┘")
        return f"\n{bar}\n{mid}\n{bot}"

    print(f"\n{BOLD('▶')} {question[:80]}")
    print(_sep("━"))

    cur_sp, cur_turn = "", -1
    tok_buf   = ""          # incomplete line buffer for llm_chunk
    in_code   = False       # are we inside a ``` block?

    def _flush_tok_buf():
        """Print whatever is left in tok_buf without a trailing newline."""
        nonlocal tok_buf, in_code
        if tok_buf:
            colored, in_code = _colorize_llm_line(tok_buf, in_code)
            sys.stdout.write(colored)
            sys.stdout.flush()
            tok_buf = ""

    def _handle_llm_chunk(content: str):
        nonlocal tok_buf, in_code
        tok_buf += content
        while "\n" in tok_buf:
            line, tok_buf = tok_buf.split("\n", 1)
            colored, in_code = _colorize_llm_line(line, in_code)
            print(colored)

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            buf = ""
            while True:
                raw_chunk = resp.read(256)
                if not raw_chunk:
                    break
                buf += raw_chunk.decode("utf-8", errors="replace")
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    for raw_line in block.splitlines():
                        if not raw_line.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(raw_line[6:])
                        except json.JSONDecodeError:
                            continue
                        etype = ev.get("type", "")

                        # ── Phase header ─────────────────────────────────────
                        if etype == "phase":
                            _flush_tok_buf()
                            pid  = ev.get("phase_id", "")
                            name = ev.get("phase_name", "")
                            el   = ev.get("elapsed_s", "")
                            print(_phase_banner(pid, name, el))
                            print(_sep())

                        # ── New SP turn ───────────────────────────────────────
                        elif etype == "sp_turn":
                            sp  = ev.get("sp_id", "")
                            trn = ev.get("turn", 0)
                            if sp != cur_sp:
                                # New sub-problem = fresh LLM context
                                _flush_tok_buf()
                                print(_sp_new_banner(sp))
                                cur_sp = sp
                            if trn != cur_turn:
                                _flush_tok_buf()
                                tool = ev.get("tool", "?")
                                mx   = ev.get("max_turns", 15)
                                print(f"\n{PURPLE(BOLD(f'  ┌─ [{sp}] Turn {trn}/{mx}'))}  {DIM('→')}  {PURPLE(tool)}")
                                cur_turn = trn

                        # ── Live LLM tokens ───────────────────────────────────
                        elif etype == "llm_chunk":
                            _handle_llm_chunk(ev.get("content", ""))

                        # ── SP completed ─────────────────────────────────────
                        elif etype == "sp_done":
                            _flush_tok_buf()
                            sp_id  = ev.get("sp_id", "")
                            st     = ev.get("status", "failed")
                            vals   = ev.get("values", "") or "—"
                            turns  = ev.get("turns", 0)
                            if st == "solved":
                                bar = B_GREEN("━" * W)
                                msg = B_GREEN(BOLD(f"  ✔  {sp_id} SOLVED  │  {vals}  │  {turns} turns"))
                            else:
                                bar = B_RED("━" * W)
                                msg = B_RED(BOLD(f"  ✘  {sp_id} FAILED  │  {turns} turns"))
                            print(f"\n{bar}\n{msg}\n{bar}")

                        # ── Wave / wave info ──────────────────────────────────
                        elif etype == "wave":
                            _flush_tok_buf()
                            wn   = ev.get("wave_num", "")
                            wt   = ev.get("wave_total", "")
                            sps  = ev.get("sp_ids", [])
                            print(f"\n{CYAN(f'  ⚡ Wave {wn}/{wt}:')}  {DIM(str(sps))}")

                        # ── Final answer ──────────────────────────────────────
                        elif etype == "answer":
                            _flush_tok_buf()
                            answer = ev.get("answer", "")
                            print(f"\n{B_GREEN('━'*W)}")
                            print(B_GREEN(BOLD("  ★  FINAL ANSWER")))
                            print(f"{B_GREEN('━'*W)}\n")
                            print(answer)
                            print(f"\n{_sep('━')}")
                            return

                        # ── Errors ────────────────────────────────────────────
                        elif etype in ("error", "error_line"):
                            _flush_tok_buf()
                            msg = ev.get("error", ev.get("line", "unknown"))
                            print(f"\n{B_RED('┌── ERROR ──')}\n{B_RED(msg)}\n{B_RED('└──────────')}")

    except KeyboardInterrupt:
        _flush_tok_buf()
        print(f"\n{DIM('[interrupted]')}")
    except Exception as e:
        print(f"\n{B_RED(f'Connection error: {e}')}")


# -- Streaming command (Rich panel) -------------------------------------------

def _cmd_stream_rich(server: str, question: str):
    """Rich Live panel variant of cmd_stream (used when rich is available)."""
    url  = server.rstrip("/") + "/query_stream"
    data = json.dumps({"question": question}).encode()
    hdrs = {**_headers(), "Accept": "text/event-stream"}
    req  = urllib.request.Request(url, data=data, headers=hdrs, method="POST")

    tracker = LiveStatus(question)
    console = _RichConsole()

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            with _RichLive(tracker.render(), console=console,
                           refresh_per_second=2, transient=False) as live:
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

                            if DEBUG_MODE:
                                console.print(DIM(f"  [dbg] {json.dumps(ev)[:200]}"))

                            tracker.on_event(ev)
                            live.update(tracker.render())

                            etype = ev.get("type", "")
                            if etype == "answer":
                                live.stop()
                                answer  = ev.get("answer", "")
                                elapsed = ev.get("elapsed", 0)
                                print(DIM("-" * 60))
                                print(f"{GREEN('Done')}  {DIM(f'elapsed={elapsed:.1f}s')}")
                                print(DIM("-" * 60))
                                print()
                                w = min(100, _term_width())
                                for ln in answer.splitlines():
                                    print(textwrap.fill(ln, width=w) if len(ln) > w else ln)
                                print()
                                return
                            elif etype == "done":
                                return
                            elif etype == "error":
                                live.stop()
                                print(f"\n{RED('Error:')} {ev.get('error', 'unknown')}\n")
                                return

    except urllib.error.HTTPError as e:
        print(RED(f"\n[HTTP {e.code}] {e.read().decode()}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(RED(f"\n[Connection error] {e.reason}"), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW('Interrupted.')}")


# -- Streaming command ---------------------------------------------------------

def cmd_stream(server: str, question: str):
    """
    Connect to /query_stream (SSE) and display a live progress UI.

    Verbosity levels (set via --verbose / :verbose in REPL):
      0 = quiet   — answer only, no progress
      1 = normal  — Rich Live panel (if rich installed) or spinner
      2 = verbose — every event printed as its own line
    Debug mode (--debug / :debug): also prints raw SSE JSON for each event.
    Token mode (--tokens / :tokens): raw LLM output streamed character-by-character.
    """
    global VERBOSITY, DEBUG_MODE, STREAM_TOKENS

    # Route to raw token streaming when -t/--tokens is active
    if STREAM_TOKENS:
        cmd_stream_tokens(server, question)
        return

    # Route to Rich Live panel when available and in normal verbosity
    if _HAS_RICH and VERBOSITY == 1 and _USE_COLOR:
        _cmd_stream_rich(server, question)
        return

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
    _last_lines  = 0               # how many lines spinner currently occupies
    _draw_lock   = threading.Lock()
    _stop_spin   = threading.Event()

    if VERBOSITY >= 1:
        print(f"\n{BOLD('Swarm 3.7')}  {DIM(question[:80])}")
        print(DIM("-" * 60))

    # ── spinner/status line — driven by background thread every 0.15s ────────
    def _draw():
        nonlocal tick, _last_lines
        if VERBOSITY != 1:
            return
        with _draw_lock:
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
                # Move up exactly as many lines as we drew last time, then redraw
                if _last_lines > 0:
                    print(f"\033[{_last_lines}A", end="", flush=True)
                print(f"{CLEAR}{status_line}")
                if last_log:
                    print(f"{CLEAR}{log_line}")
                    _last_lines = 2
                else:
                    _last_lines = 1
            else:
                print(f"  [{elapsed:.0f}s] {phase_name} {toks_ps:.1f} tok/s  {last_log[:60]}")

    def _spin_worker():
        """Background thread: tick the spinner every 0.15s regardless of SSE events."""
        while not _stop_spin.is_set():
            _draw()
            _stop_spin.wait(0.15)

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
            # Placeholder line so spinner has a row to overwrite on first draw
            if VERBOSITY == 1:
                print()

            # Start background thread: spinner ticks every 0.15s independent of SSE events
            if VERBOSITY == 1:
                _spin_thread = threading.Thread(target=_spin_worker, daemon=True)
                _spin_thread.start()

            def _stop_and_clear():
                """Stop spinner thread and erase its lines before printing output."""
                _stop_spin.set()
                if VERBOSITY == 1 and _USE_COLOR:
                    with _draw_lock:
                        if _last_lines > 0:
                            print(f"\033[{_last_lines}A", end="", flush=True)
                            for _ in range(_last_lines):
                                print(CLEAR, end="")
                            print(flush=True)
                elif VERBOSITY == 1:
                    print()  # end the no-color spinner line

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

                        elif etype == "phase":
                            phase_id   = evt.get("phase_id", "")
                            phase_name = evt.get("phase_name", "")
                            last_log   = ""
                            _vprint(f"phase {phase_id:<4}",
                                    f"[{elapsed:.0f}s] {phase_name}",
                                    _phase_color(phase_id))

                        elif etype == "toks":
                            toks_ps      = evt.get("toks_per_sec", 0.0)
                            total_tokens += evt.get("tokens", 0)
                            last_toks_t  = time.time()
                            _vprint("toks     ",
                                    f"{evt.get('tokens')} tok  {toks_ps:.1f} tok/s  "
                                    f"{evt.get('seconds')}s",
                                    GREEN)

                        elif etype == "agent":
                            agent_name = evt.get("agent", "")
                            model_name = evt.get("model", "")
                            _vprint("agent    ", f"{agent_name} @ {model_name}", MAGENTA)

                        elif etype == "log":
                            line = evt.get("line", "")
                            _set_last_log(line)
                            _vprint("log      ", line[:width-20])

                        elif etype == "error_line":
                            line = evt.get("line", "")
                            _set_last_log(RED("!") + " " + line)
                            _vprint("error    ", line[:width-20], RED)

                        elif etype == "heartbeat":
                            _vprint("heartbeat", f"[{elapsed:.0f}s]", DIM)

                        elif etype == "answer":
                            _stop_and_clear()
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
                            return

                        elif etype == "error":
                            _stop_and_clear()
                            print(f"\n{RED('Error:')} {evt.get('error','unknown')}\n")

                        elif etype == "done":
                            _stop_spin.set()
                            return

    except urllib.error.HTTPError as e:
        _stop_spin.set()
        print(RED(f"\n[HTTP {e.code}] {e.read().decode()}"), file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        _stop_spin.set()
        print(RED(f"\n[Connection error] {e.reason}"), file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        _stop_spin.set()
        print(f"\n{YELLOW('Interrupted.')}")


# -- REPL ---------------------------------------------------------------------

def cmd_repl(server):
    global VERBOSITY, DEBUG_MODE, STREAM_TOKENS
    print(f"{BOLD('Swarm 3.7 REPL')}  (server: {server})")
    print(f"Verbosity: {VERBOSITY}  Debug: {DEBUG_MODE}  Tokens: {STREAM_TOKENS}")
    print("Commands: :health :status :jobs :stream <q> :ask <q> :tokens")
    print("          :result <job_id>  -- show answer for a completed job")
    print("          :watch <job_id>   -- reconnect to running job (tail-f style)")
    print("          :logs <job_id> [tail=N] [grep=pat]")
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
        elif line.startswith(":result ") or line.startswith(":results "): cmd_result(server, line.split(" ", 1)[1].strip())
        elif line.startswith(":stream "): cmd_stream(server, line[8:].strip())
        elif line.startswith(":ask "):    cmd_ask(server, line[5:].strip())
        elif line == ":tokens":
            STREAM_TOKENS = not STREAM_TOKENS
            print(f"  Token streaming {'ON  (raw LLM output)' if STREAM_TOKENS else 'OFF (Rich panel)'}")
        elif line.startswith(":watch "):
            parts = line.split()
            if len(parts) >= 2:
                cmd_watch(server, parts[1])
            else:
                print("  Usage: :watch <job_id>")
        elif line.startswith(":logs "):
            parts = line.split()
            if len(parts) >= 2:
                job_id = parts[1]
                tail_n = grep_s = None
                for i, p in enumerate(parts):
                    if p.startswith("tail="):
                        try: tail_n = int(p[5:])
                        except ValueError: pass
                    if p.startswith("grep="):
                        grep_s = p[5:]
                cmd_logs(server, job_id, tail=tail_n, grep=grep_s)
            else:
                print("  Usage: :logs <job_id> [tail=N] [grep=pat]")
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
        elif line.startswith(":"):
            cmd = line.split()[0]
            print(f"  Unknown command: {cmd}")
            print("  Commands: :health :status :jobs :result :stream :ask :tokens :watch :logs :verbose :debug")
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
        description="Swarm 3.7 client",
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
    parser.add_argument("-t", "--tokens", action="store_true",
                        help="Stream raw LLM tokens to terminal (like ollama run)")
    parser.add_argument("args", nargs="*")
    opts = parser.parse_args()
    server = opts.server.rstrip("/")

    global VERBOSITY, DEBUG_MODE, STREAM_TOKENS
    if opts.quiet:
        VERBOSITY = 0
    elif opts.verbose:
        VERBOSITY = min(2, 1 + opts.verbose)
    DEBUG_MODE    = opts.debug
    STREAM_TOKENS = opts.tokens

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
    elif cmd == "logs":
        if len(args) < 2:
            print("Usage: run_me.py logs <job_id> [--tail N] [--grep pattern]",
                  file=sys.stderr); sys.exit(1)
        tail_n = grep_s = None
        for i, a in enumerate(args):
            if a == "--tail" and i + 1 < len(args):
                try: tail_n = int(args[i + 1])
                except ValueError: pass
            if a == "--grep" and i + 1 < len(args):
                grep_s = args[i + 1]
        cmd_logs(server, args[1], tail=tail_n, grep=grep_s)
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
