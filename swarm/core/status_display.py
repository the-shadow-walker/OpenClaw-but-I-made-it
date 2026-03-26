"""
Swarm 3.0 — Enhanced Real-Time Status Display

Uses Rich Live to render a live-updating panel showing:
  • Current phase + progress bar + elapsed time
  • Active operation (what's happening right now)
  • Key stats (sources, facts, searches, question type)
  • Scrolling activity log (last N captured print lines)

Stdout is intercepted during the Live session so all existing
print() calls from every module flow into the log automatically.
Stdout is fully restored when the context manager exits.

Usage:
    status = StatusDisplay(date_filter="week", save_markdown=True)
    with status:
        answer = await orchestrator.process_question(question, status=status)
    # stdout restored here — normal Rich console prints work fine again
"""

import sys
import re
from collections import deque
from datetime import datetime
from typing import Optional, Dict, Any

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.markup import escape
from rich.text import Text

# ── Phase metadata ────────────────────────────────────────────────────────────
TOTAL_PHASES = 5

PHASE_META = {
    0: ("Initializing",   "cyan"),
    1: ("Classification", "magenta"),
    2: ("Planning",       "blue"),
    3: ("Search",         "yellow"),
    4: ("Math",           "red"),
    5: ("Summary",        "green"),
    6: ("Markdown",       "cyan"),
}

# ── Stdout capture ────────────────────────────────────────────────────────────

class _StdoutCapture:
    """
    Redirect sys.stdout to a line callback without touching stderr.

    Lines are buffered until a newline is seen, then the stripped
    line is passed to the callback. Empty lines are dropped.
    """

    def __init__(self, callback):
        self._cb  = callback
        self._real = sys.stdout
        self._buf  = ""
        sys.stdout = self

    def write(self, text: str):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._cb(stripped)

    def flush(self):
        pass  # intentional no-op while capturing

    def restore(self):
        sys.stdout = self._real


# ── Main class ────────────────────────────────────────────────────────────────

class StatusDisplay:
    """
    Rich Live status panel for Swarm 3.0.

    Call set_phase() / set_stat() / set_activity() from the orchestrator
    for explicit state updates.  Everything printed to stdout is also
    captured automatically and streamed into the activity log.
    """

    def __init__(
        self,
        date_filter: Optional[str]  = None,
        save_markdown: bool          = False,
    ):
        self._log:        deque       = deque(maxlen=40)
        self._phase_num:  int         = 0
        self._phase_name: str         = "Initializing"
        self._phase_color: str        = "cyan"
        self._activity:   str         = "Starting Swarm 3.0..."
        self._stats: Dict[str, Any]   = {
            "Sources":  0,
            "Facts":    0,
            "Searches": 0,
            "Type":     "—",
        }
        self._config: Dict[str, str]  = {}
        if date_filter:
            self._config["Date filter"] = date_filter
        self._config["MD report"] = "ON" if save_markdown else "OFF"

        self._start_time: Optional[datetime] = None
        # Render on stderr so the captured stdout doesn't interfere
        self._stderr_console = Console(stderr=True)
        self._live:    Optional[Live]          = None
        self._capture: Optional[_StdoutCapture] = None

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "StatusDisplay":
        self._start_time = datetime.now()
        self._live = Live(
            self._render(),
            console=self._stderr_console,
            refresh_per_second=5,
            screen=False,
            transient=False,
        )
        self._live.__enter__()
        # Start capturing stdout AFTER Live starts so Rich's own init
        # messages aren't swallowed.
        self._capture = _StdoutCapture(self._on_print)
        return self

    def __exit__(self, *args):
        # Restore stdout first so Rich's own teardown can write normally
        if self._capture:
            self._capture.restore()
            self._capture = None
        if self._live:
            self._live.update(self._render())
            self._live.__exit__(*args)
            self._live = None

    # ── public API (called by orchestrator) ───────────────────────────────────

    def set_phase(self, num: int, name: Optional[str] = None):
        """Advance to a numbered phase."""
        self._phase_num   = num
        meta = PHASE_META.get(num, ("Running", "white"))
        self._phase_name  = name or meta[0]
        self._phase_color = meta[1]
        self._activity    = f"Starting {self._phase_name}…"
        self._refresh()

    def set_activity(self, msg: str):
        """Update the 'current operation' line."""
        self._activity = msg
        self._refresh()

    def set_stat(self, key: str, value: Any):
        """Update a named stat counter."""
        self._stats[key] = value
        self._refresh()

    def log(self, msg: str):
        """Add a line directly to the log (bypasses capture)."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[dim]{ts}[/dim]  {escape(msg)}")
        self._refresh()

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_print(self, raw: str):
        """Called for every line captured from stdout."""
        clean = _strip_ansi(raw)
        self._auto_extract(clean)
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[dim]{ts}[/dim]  {escape(clean)}")
        self._refresh()

    def _auto_extract(self, line: str):
        """
        Parse key metrics and activity hints from captured output.
        Patterns are matched against the existing print() messages
        across orchestrator, flexible_search_agent, and search_parallel.
        """

        # ── Activity hints ────────────────────────────────────────────────
        # "🔍 Searching: query" or "🔎 [task_id] Searching: query"
        m = re.search(r'Searching:\s*(.+)', line)
        if m:
            q = m.group(1).strip()
            self._activity = f"🔍 Searching: {q[:65]}"
            self._stats["Searches"] = self._stats.get("Searches", 0) + 1

        # "Fetching content from: title..."
        m = re.search(r'Fetching content from:\s*(.+)', line)
        if m:
            self._activity = f"📥 Fetching: {m.group(1).strip()[:65]}"

        # "Extracting from: title" or "Extracting: title"
        m = re.search(r'Extracting(?:\s+from)?:\s*(.+)', line)
        if m:
            self._activity = f"📑 Extracting: {m.group(1).strip()[:65]}"

        # Writer activity
        if "Writing research answer" in line or "Writing final answer" in line:
            self._activity = "✍️  Writing final answer…"
        if "Generating deep Markdown" in line:
            self._activity = "📄 Generating Markdown report…"

        # Think step in DeepSearch
        m = re.search(r'Think step.*?:\s*(\d+) facts so far', line)
        if m:
            self._activity = f"🤔 Reflecting on {m.group(1)} facts…"

        # Follow-up query generation
        m = re.search(r'Gaps:\s*(.+)', line)
        if m:
            self._activity = f"🔎 Gap identified: {m.group(1)[:60]}"

        # LLM writing (writer_agent)
        if "✅ Research answer written" in line or "✅ Answer written" in line:
            self._activity = "✅ Answer written"

        # Math phases
        if "Equation generated" in line:
            self._activity = "🔢 Equation generated, validating…"
        if "Execution successful" in line:
            self._activity = "✅ Computation complete"

        # ── Stat updates ──────────────────────────────────────────────────

        # "✅ Search complete - N sources fetched, M facts in memory"
        m = re.search(r'(\d+) sources fetched.*?(\d+) facts in memory', line)
        if m:
            self._stats["Sources"] = int(m.group(1))
            self._stats["Facts"]   = int(m.group(2))

        # "Deep search complete: N searches, M facts in memory"
        m = re.search(r'Deep search complete:\s*(\d+) searches,\s*(\d+) facts', line)
        if m:
            self._stats["Searches"] = int(m.group(1))
            self._stats["Facts"]    = int(m.group(2))

        # "M facts so far" (think step)
        m = re.search(r'(\d+) facts so far', line)
        if m:
            self._stats["Facts"] = max(self._stats.get("Facts", 0), int(m.group(1)))

        # "Found N results via backend" (flexible_search_agent)
        m = re.search(r'Found (\d+) results via', line)
        if m:
            self._stats["Sources"] = self._stats.get("Sources", 0) + int(m.group(1))

        # "Found N sources for main query" (orchestrator)
        m = re.search(r'Found (\d+) sources for main query', line)
        if m:
            self._stats["Sources"] = max(self._stats.get("Sources", 0), int(m.group(1)))

        # "📚 Found N sources" (search_parallel)
        m = re.search(r'📚.*?(\d+).*?source', line)
        if m:
            self._stats["Sources"] = self._stats.get("Sources", 0) + int(m.group(1))

        # "Extracted N facts" (search_parallel)
        m = re.search(r'Extracted (\d+) facts', line)
        if m:
            self._stats["Facts"] = self._stats.get("Facts", 0) + int(m.group(1))

        # "🎯 Type: THEORETICAL/MATHEMATICAL/..."  (orchestrator classifier)
        m = re.search(r'Type:\s*([A-Z]+)', line)
        if m and self._stats.get("Type") == "—":
            self._stats["Type"] = m.group(1).title()

    def _refresh(self):
        if self._live:
            self._live.update(self._render())

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _elapsed(self) -> str:
        if not self._start_time:
            return "00:00"
        s = (datetime.now() - self._start_time).total_seconds()
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    def _phase_bar(self) -> str:
        done = min(self._phase_num, TOTAL_PHASES)
        rest = TOTAL_PHASES - done
        bar  = f"[{self._phase_color}]{'█' * done}[/{self._phase_color}]"
        bar += f"[dim]{'░' * rest}[/dim]"
        return bar

    def _render(self) -> Layout:
        elapsed = self._elapsed()

        # ── Header row ────────────────────────────────────────────────────
        header_text = (
            f"[bold cyan]🤖 SWARM 2.1[/bold cyan]"
            f"  [dim]│[/dim]  "
            f"Phase [bold]{self._phase_num}[/bold]/[dim]{TOTAL_PHASES}[/dim]  "
            f"[bold {self._phase_color}]{self._phase_name.upper()}[/bold {self._phase_color}]"
            f"  [dim]│[/dim]  "
            f"⏱ [bold]{elapsed}[/bold]"
            f"  [dim]│[/dim]  "
            f"{self._phase_bar()}"
        )
        header = Panel(header_text, style="cyan", padding=(0, 1))

        # ── Stats table ───────────────────────────────────────────────────
        tbl = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        tbl.add_column(justify="right", style="dim", no_wrap=True)
        tbl.add_column(style="bold white", no_wrap=True)

        for k, v in self._stats.items():
            color = "white"
            if k == "Facts"   and isinstance(v, int) and v > 0:  color = "green"
            if k == "Sources" and isinstance(v, int) and v > 0:  color = "cyan"
            if k == "Searches"and isinstance(v, int) and v > 0:  color = "yellow"
            tbl.add_row(k, f"[{color}]{v}[/{color}]")

        tbl.add_row("", "")   # spacer
        for k, v in self._config.items():
            val_color = "green" if v == "ON" else ("yellow" if v not in ("—", "OFF") else "dim")
            tbl.add_row(f"[dim]{k}[/dim]", f"[{val_color}]{v}[/{val_color}]")

        stats_panel = Panel(tbl, title="[bold]Stats[/bold]", padding=(0, 1))

        # ── Activity panel ────────────────────────────────────────────────
        activity_panel = Panel(
            f"\n[bold yellow]{escape(self._activity)}[/bold yellow]\n",
            title="[bold]Current Operation[/bold]",
            style="yellow",
            padding=(0, 1),
        )

        # ── Log panel ─────────────────────────────────────────────────────
        recent   = list(self._log)[-18:]
        log_body = "\n".join(recent) if recent else "[dim]Waiting for activity…[/dim]"
        log_panel = Panel(log_body, title="[bold]Activity Log[/bold]", padding=(0, 1))

        # ── Assemble layout ───────────────────────────────────────────────
        root = Layout()
        root.split_column(
            Layout(header,    name="header", size=3),
            Layout(name="mid", size=7),
            Layout(log_panel, name="log"),
        )
        root["mid"].split_row(
            Layout(activity_panel, name="activity", ratio=2),
            Layout(stats_panel,    name="stats",    ratio=1),
        )
        return root


# ── Helpers ───────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mGKHFABCDJK]')

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)
