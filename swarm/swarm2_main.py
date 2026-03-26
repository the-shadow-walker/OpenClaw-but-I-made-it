#!/usr/bin/env python3
import _paths  # noqa: F401 — sets up sys.path
"""
Swarm 3.0 - Deterministic Research & Compute Architecture
Main Entry Point — with Project Mode

Usage:
    python3 swarm2_main.py "Your question here"
    python3 swarm2_main.py --interactive
    python3 swarm2_main.py --test
    python3 swarm2_main.py --project         ← jump straight to project mode

In interactive mode:
    • Type a question  → deep search + answer
    • Type 'project'   → enter project mode (full guided Q&A + Amazon sourcing)
    • Type 'help'      → show commands
    • Type 'quit'      → exit
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from orchestrator_v2_1 import OrchestratorV2_1

try:
    from status_display import StatusDisplay
    HAS_STATUS = True
except ImportError:
    HAS_STATUS = False

# Project mode — imported here so it's only loaded when needed
_project_mode_loaded = False

def _load_project_mode():
    global _project_mode_loaded, run_project_mode
    if not _project_mode_loaded:
        try:
            from project_mode import run_project_mode as _rpm
            run_project_mode = _rpm
            _project_mode_loaded = True
        except ImportError as e:
            run_project_mode = None
            print(f"⚠️  project_mode.py not found: {e}")
    return _project_mode_loaded


# Engineer mode — lazy-loaded on first use
_engineer_mode_loaded = False
run_engineer_mode_fn = None

def _load_engineer_mode():
    global _engineer_mode_loaded, run_engineer_mode_fn
    if not _engineer_mode_loaded:
        try:
            from engineer_mode import run_engineer_mode as _rem
            run_engineer_mode_fn = _rem
            _engineer_mode_loaded = True
        except ImportError as e:
            run_engineer_mode_fn = None
            print(f"⚠️  engineer_mode.py not found: {e}")
    return _engineer_mode_loaded

console = Console()

# ── PROJECT MODE TRIGGER DETECTION ──────────────────────────────────────────
# IMPORTANT: Only trigger on explicit commands, not on sentences that happen
# to contain the word "project".
#
# Triggers:
#   "project"         — bare word
#   "new project"
#   "start project"
#   "create project"
#   "begin project"
#   --project flag
#
# Does NOT trigger on:
#   "I'm working on a project about..."
#   "my project requires..."
#   "can you explain this project..."

import re as _re

_PROJECT_TRIGGERS = _re.compile(
    r"""^(?:
        project |
        new\s+project |
        start\s+project |
        start\s+a\s+project |
        create\s+project |
        create\s+a\s+project |
        begin\s+project |
        new\s+build |
        start\s+build
    )$""",
    _re.IGNORECASE | _re.VERBOSE,
)

def _is_project_trigger(text: str) -> bool:
    """Return True ONLY for explicit project mode commands."""
    return bool(_PROJECT_TRIGGERS.match(text.strip()))


# ── ENGINEER MODE TRIGGER DETECTION ──────────────────────────────────────────
# Triggers:
#   "engineer"            — bare word
#   "engineer mode"
#   "engineering design"
#   "design mode"
#   "new design"
#   --engineer / -e flag

_ENGINEER_TRIGGERS = _re.compile(
    r"""^(?:
        engineer |
        engineer\s+mode |
        engineering\s+design |
        design\s+mode |
        new\s+design
    )$""",
    _re.IGNORECASE | _re.VERBOSE,
)


def _is_engineer_trigger(text: str) -> bool:
    """Return True ONLY for explicit engineer mode commands."""
    return bool(_ENGINEER_TRIGGERS.match(text.strip()))


# ── SINGLE QUESTION ──────────────────────────────────────────────────────────

async def run_single_question(
    question: str,
    save_session: bool   = True,
    date_filter: str     = None,
    save_markdown: bool  = False,
    status_mode: bool    = False,
):
    console.print(Panel.fit(
        f"[bold cyan]Swarm 3.0 - Deterministic Research & Compute[/bold cyan]\n"
        f"Question: {question}",
        title="🚀 Starting Swarm 3.0",
        border_style="cyan"
    ))

    orchestrator = OrchestratorV2_1(
        max_search_concurrent=3,
        enable_verification=True,
        debug=False,
        searxng_url=os.getenv('SEARXNG_URL'),
        date_filter=date_filter,
        save_markdown=save_markdown,
    )

    if status_mode and HAS_STATUS:
        with StatusDisplay(date_filter=date_filter, save_markdown=save_markdown) as status:
            answer = await orchestrator.process_question(question, status=status)
        # stdout fully restored — safe to print normally now
    else:
        answer = await orchestrator.process_question(question)

    console.print()
    console.print(Panel(
        Markdown(answer),
        title="📝 Final Answer",
        border_style="green"
    ))

    if orchestrator.markdown_path:
        console.print(f"\n📄 Markdown report: [bold]{orchestrator.markdown_path}[/bold]")

    if save_session:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"/tmp/swarm2_session_{timestamp}.json"
        orchestrator.save_session(filepath)
        console.print(f"\n💾 Session saved to: {filepath}")


# ── INTERACTIVE MODE ──────────────────────────────────────────────────────────

async def run_interactive(
    initial_date_filter: str   = None,
    initial_save_markdown: bool = False,
    initial_status_mode: bool   = False,
):
    console.print(Panel.fit(
        "[bold cyan]Swarm 3.0 — Interactive Mode[/bold cyan]\n\n"
        "Commands:\n"
        "  [green]project[/green]         → enter Project Mode (design + source parts)\n"
        "  [green]engineer[/green]        → enter Engineer Mode (multi-variable design problems)\n"
        "  [green]debug on|off[/green]    → toggle verbose debug output\n"
        "  [green]verify on|off[/green]   → toggle answer verification\n"
        "  [green]since <period>[/green]  → date filter: day|week|month|year|YYYY-MM-DD|off\n"
        "  [green].md on|off[/green]      → save detailed Markdown report to /tmp/\n"
        "  [green]status on|off[/green]   → live status display (phases, stats, log)\n"
        "  [green]help[/green]            → show this help\n"
        "  [green]quit[/green]            → exit\n\n"
        "Anything else is treated as a research/compute question.",
        title="🤖 Swarm 3.0 Ready",
        border_style="cyan"
    ))

    enable_verification = True
    debug_mode          = False
    date_filter         = initial_date_filter   # e.g. "week", "month", "year", "2025-01-01"
    save_markdown       = initial_save_markdown
    status_mode         = initial_status_mode

    while True:
        console.print()
        try:
            question = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: console.input("[bold green]❓ YOU >[/bold green] ").strip()
            )
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        lower = question.lower().strip()

        # ── Exit ──────────────────────────────────────────────────────────
        if lower in ('quit', 'exit', 'q'):
            console.print("[yellow]👋 Shutting down Swarm 3.0...[/yellow]")
            break

        # ── Help ──────────────────────────────────────────────────────────
        if lower == 'help':
            console.print(
                "\n[bold]Commands:[/bold]\n"
                "  [green]project[/green]          → Project Mode (Q&A + Amazon sourcing + .md file)\n"
                "  [green]engineer[/green]         → Engineer Mode (multi-variable technical design + TDS)\n"
                "  [green]debug on|off[/green]      → Toggle verbose debug output\n"
                "  [green]verify on|off[/green]     → Toggle answer verification step\n"
                "  [green]since <period>[/green]    → Date filter: day|week|month|year|YYYY-MM-DD|off\n"
                "  [green].md on|off[/green]        → Save a detailed Markdown report after each answer\n"
                "  [green]status on|off[/green]     → Live status display (phases · stats · activity log)\n"
                "  [green]quit / exit[/green]       → Exit\n"
                "  [dim]Anything else[/dim]      → Deep research + compute answer"
            )
            continue

        # ── Debug toggle ─────────────────────────────────────────────────
        if lower.startswith('debug '):
            debug_mode = lower.split()[1] == 'on'
            console.print(f"[yellow]🔧 Debug: {'ON' if debug_mode else 'OFF'}[/yellow]")
            continue

        # ── Verify toggle ─────────────────────────────────────────────────
        if lower.startswith('verify '):
            enable_verification = lower.split()[1] == 'on'
            console.print(f"[yellow]✓ Verification: {'ON' if enable_verification else 'OFF'}[/yellow]")
            continue

        # ── Date filter ───────────────────────────────────────────────────
        if lower.startswith('since '):
            value = lower.split(None, 1)[1].strip()
            if value == 'off':
                date_filter = None
                console.print("[yellow]🗓️  Date filter: OFF (all time)[/yellow]")
            elif value in ('day', 'week', 'month', 'year') or _re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                date_filter = value
                console.print(f"[yellow]🗓️  Date filter: {date_filter}[/yellow]")
            else:
                console.print(
                    "[red]Invalid period. Use: day | week | month | year | YYYY-MM-DD | off[/red]"
                )
            continue

        # ── Markdown report toggle ────────────────────────────────────────
        if lower in ('.md on', '.md off', 'md on', 'md off'):
            save_markdown = lower.endswith('on')
            state = '[bold green]ON[/bold green]' if save_markdown else '[yellow]OFF[/yellow]'
            console.print(f"📄 Markdown report: {state}")
            if save_markdown:
                console.print("   [dim]A detailed .md file will be saved to /tmp/ after each answer.[/dim]")
            continue

        # ── Status display toggle ─────────────────────────────────────────
        if lower in ('status on', 'status off'):
            status_mode = lower.endswith('on')
            if status_mode and not HAS_STATUS:
                console.print("[red]❌ status_display.py not found.[/red]")
                status_mode = False
                continue
            state = '[bold green]ON[/bold green]' if status_mode else '[yellow]OFF[/yellow]'
            console.print(f"📊 Status display: {state}")
            if status_mode:
                console.print("   [dim]A live status panel will show during each query.[/dim]")
            continue

        # ── ENGINEER MODE TRIGGER ────────────────────────────────────────
        # Only fires on explicit "engineer" / "engineer mode" etc. commands.
        if _is_engineer_trigger(lower):
            if not _load_engineer_mode():
                console.print(
                    "[red]❌ engineer_mode.py not found. "
                    "Make sure it's in the same directory as swarm2_main.py[/red]"
                )
                continue

            # If user typed just the trigger word, prompt for the design problem
            if _ENGINEER_TRIGGERS.match(question.strip()):
                try:
                    question = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: console.input(
                            "[bold green]🔧 Design problem: [/bold green]"
                        ).strip()
                    )
                except (EOFError, KeyboardInterrupt):
                    continue
                if not question:
                    continue

            console.print(Panel.fit(
                "[bold green]Entering Engineer Mode[/bold green]\n"
                "I'll decompose your design problem, hunt for missing parameters,\n"
                "run an iterative simulation, and produce a Technical Data Sheet.",
                border_style="green"
            ))
            try:
                answer = await run_engineer_mode_fn(
                    problem=question,
                    searxng_url=os.getenv('SEARXNG_URL'),
                    debug=debug_mode,
                    save_markdown=save_markdown,
                )
                console.print(Panel(
                    Markdown(answer),
                    title="📐 Technical Data Sheet",
                    border_style="blue"
                ))
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Engineer mode interrupted.[/yellow]")
            continue

        # ── PROJECT MODE TRIGGER ─────────────────────────────────────────
        # Only fires on explicit "project" commands, not on casual mentions.
        if _is_project_trigger(lower):
            if not _load_project_mode():
                console.print(
                    "[red]❌ project_mode.py not found. "
                    "Make sure it's in the same directory as swarm2_main.py[/red]"
                )
                continue

            console.print(
                Panel.fit(
                    "[bold green]Entering Project Mode[/bold green]\n"
                    "I'll walk you through specs, requirements, and part sourcing.\n"
                    "Type 'cancel' at any prompt to return to normal mode.",
                    border_style="green"
                )
            )

            # Build a shared orchestrator so project mode can reuse deep search
            orchestrator = OrchestratorV2_1(
                max_search_concurrent=3,
                enable_verification=False,   # faster for project queries
                debug=debug_mode,
                searxng_url=os.getenv('SEARXNG_URL'),
                date_filter=date_filter,
                save_markdown=save_markdown,
            )

            try:
                await run_project_mode(orchestrator=orchestrator)
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Project mode interrupted.[/yellow]")

            console.print("[green]✅ Back to normal mode. Type 'project' to start another.[/green]")
            continue

        # ── NORMAL: Research / Compute question ──────────────────────────
        try:
            orchestrator = OrchestratorV2_1(
                max_search_concurrent=3,
                enable_verification=enable_verification,
                debug=debug_mode,
                searxng_url=os.getenv('SEARXNG_URL'),
                date_filter=date_filter,
                save_markdown=save_markdown,
            )

            if status_mode and HAS_STATUS:
                with StatusDisplay(
                    date_filter=date_filter,
                    save_markdown=save_markdown,
                ) as _status:
                    answer = await orchestrator.process_question(question, status=_status)
            else:
                answer = await orchestrator.process_question(question)

            console.print()
            console.print(Panel(
                Markdown(answer),
                title="📝 Answer",
                border_style="green"
            ))

            if orchestrator.markdown_path:
                console.print(
                    f"\n📄 Markdown report: [bold]{orchestrator.markdown_path}[/bold]"
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Interrupted[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            if debug_mode:
                import traceback
                traceback.print_exc()


# ── TEST SUITE ────────────────────────────────────────────────────────────────

async def run_test_suite():
    test_questions = [
        "How much thrust force is needed to lift a 5000 kg object on Earth?",
        "What is the weight of a 5000 lbm tungsten cube on Earth in lbf?",
        "If a crane can lift 10000 N and an object weighs 8000 N, what is the net force?",
    ]

    console.print(Panel.fit(
        f"[bold cyan]Swarm 3.0 — Test Suite[/bold cyan]\n"
        f"Running {len(test_questions)} test questions",
        title="🧪 Testing",
        border_style="cyan"
    ))

    results = []

    for i, question in enumerate(test_questions, 1):
        console.print(f"\n[bold]Test {i}/{len(test_questions)}:[/bold] {question}")
        console.print("="*70)

        try:
            orchestrator = OrchestratorV2_1(
                max_search_concurrent=2,
                enable_verification=True,
                debug=False,
                searxng_url=os.getenv('SEARXNG_URL'),
            )

            answer = await orchestrator.process_question(question)

            results.append({'question': question, 'success': True, 'answer': answer})
            console.print(Panel(
                answer[:300] + ("..." if len(answer) > 300 else ""),
                title=f"✅ Test {i} — PASSED",
                border_style="green"
            ))

        except Exception as e:
            results.append({'question': question, 'success': False, 'error': str(e)})
            console.print(Panel(str(e), title=f"❌ Test {i} — FAILED", border_style="red"))

    console.print("\n" + "="*70)
    passed = sum(1 for r in results if r['success'])
    console.print(
        f"[bold]Results:[/bold] {passed}/{len(results)} passed  "
        f"({passed/len(results)*100:.0f}%)"
    )


# ── ARCHITECTURE DIAGRAM ──────────────────────────────────────────────────────

def print_architecture():
    console.print("""
[cyan]Swarm 3.0 Architecture[/cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[bold]Normal Mode (any question):[/bold]
  Question → Classify → Plan → Search → Math? → Summarize → Answer

[bold]Project Mode (triggered by typing 'project'):[/bold]
  Guided Q&A → Compute torque/current → Amazon search → Validate
  → Select parts → Deep web search (optional) → [name].md

[bold]Core Principles:[/bold]
  ✓ No LLM does math
  ✓ All knowledge written to shared memory
  ✓ Amazon API for real, current pricing
  ✓ Parts validated against computed requirements
  ✓ Output is a clean Markdown file
""")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Swarm 3.0 — Deterministic Research & Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 swarm2_main.py "How much thrust for 5000 kg?"
  python3 swarm2_main.py --interactive
  python3 swarm2_main.py --project
  python3 swarm2_main.py --test
        """
    )
    parser.add_argument('question',  nargs='?',     help='Research/compute question')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('-p', '--project',     action='store_true', help='Jump straight to Project Mode')
    parser.add_argument('-e', '--engineer',    action='store_true', help='Engineer Mode (design problems)')
    parser.add_argument('-t', '--test',        action='store_true', help='Run test suite')
    parser.add_argument('--architecture',      action='store_true', help='Print architecture diagram')
    parser.add_argument('--searxng', type=str, default=None,        help='SearXNG URL')
    parser.add_argument('--no-save', action='store_true',           help='Do not save session')
    parser.add_argument(
        '--since', type=str, default=None, metavar='PERIOD',
        help='Restrict search results by date: day | week | month | year | YYYY-MM-DD'
    )
    parser.add_argument(
        '--md', action='store_true', default=False,
        help='Save a detailed Markdown research report to /tmp/ after answering'
    )
    parser.add_argument(
        '--status', action='store_true', default=False,
        help='Show live status display (phases, stats, activity log) while answering'
    )

    args = parser.parse_args()

    if args.searxng:
        os.environ['SEARXNG_URL'] = args.searxng

    if args.architecture:
        print_architecture()
        return

    if args.test:
        asyncio.run(run_test_suite())
        return

    # Direct project mode via flag
    if args.project:
        if not _load_project_mode():
            console.print("[red]❌ project_mode.py not found.[/red]")
            return
        asyncio.run(run_project_mode())
        return

    # Direct engineer mode via flag
    if args.engineer:
        if not args.question:
            console.print(
                "[red]❌ --engineer requires a design problem, e.g.:\n"
                '   python3 swarm2_main.py --engineer "Design a two-stage rocket to put 15,000 kg in LEO"[/red]'
            )
            return
        if not _load_engineer_mode():
            console.print("[red]❌ engineer_mode.py not found.[/red]")
            return
        asyncio.run(run_engineer_mode_fn(
            problem=args.question,
            searxng_url=os.getenv('SEARXNG_URL'),
            debug=False,
            save_markdown=args.md,
        ))
        return

    if args.interactive or not args.question:
        asyncio.run(run_interactive(
            initial_date_filter=args.since,
            initial_save_markdown=args.md,
            initial_status_mode=args.status,
        ))
        return

    asyncio.run(run_single_question(
        args.question,
        save_session=not args.no_save,
        date_filter=args.since,
        save_markdown=args.md,
        status_mode=args.status,
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Interrupted[/yellow]")
        sys.exit(0)
