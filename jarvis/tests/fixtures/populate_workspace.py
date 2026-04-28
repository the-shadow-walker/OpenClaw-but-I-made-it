"""Populate a Jarvis workspace with 50 planted facts across 9 files.

Used by:
  * the disposability test (test_index.py::test_disposability),
  * the CLI disposability test (test_cli.py::test_cli_search_output_stable_across_rebuild),
  * the manual exit-criterion eval (`python -m tests.fixtures.populate_workspace --root /tmp/p2-eval`).

Distribution (50 facts):
  MEMORY.md    — 10 evergreen lines (preferences, key dates, recurring people)
  USER.md      — 5 lines (name, locations, work)
  memory/<today>.md, <today-1>.md, ..., <today-4>.md  — 5 daily logs × 5 lines = 25
  projects/rocket-sim.md, jarvis-rebuild.md, garden.md  — 3 projects × ~3-4 lines = 10
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.files import write_markdown_atomic
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace

# Stable, deterministic content. Each line is one "fact"; the test queries hit
# distinctive tokens that should rank a particular file's chunk in the top-3.

_MEMORY_BODY = """\
# MEMORY.md

Curated long-term facts.

## Preferences
- Grant prefers concise replies with no fluff.
- Grant uses zsh on macOS for the daily driver shell.
- Default ollama fast model is qwen2.5:3b for low-latency replies.
- Coffee order: black drip, no sugar.

## Key dates
- Birthday: April 12.
- Started Jarvis-Mk3 rebuild in March 2026.

## People
- Tina is Grant's partner; she does landscape architecture.
- Roy is a longtime friend who runs the garden co-op.

## Work patterns
- Standup pattern: write yesterday/today/blockers in the daily log first thing.
- Friday rule: never deploy on Fridays unless it's a one-line config fix.
"""

_USER_BODY = """\
# USER.md

User-modeling facts shared across DM and group conversations.

- Name: Grant.
- Lives in Portland, Oregon.
- Works as a software engineer focused on AI infrastructure.
- Home server hostname is atomos; SSH alias is mcssh.
- Programming languages used daily: Python and TypeScript.
"""

# Five daily logs, each with five timestamped entries. Each carries
# distinctive tokens so search queries can target a specific day.
_DAILY_TEMPLATES: list[tuple[int, str]] = [
    (
        0,
        """\
# {iso}

[09:01] Daily standup: yesterday finished P1 prompt assembly; today starting P2 chunker.
[10:15] Reviewed FTS5 porter tokenizer behavior on apostrophes.
[12:30] Lunch with Tina at the new ramen place on Hawthorne.
[15:45] Pair-debugged a sqlite-vec extension load issue with Roy.
[17:50] Wrote up a rocket-sim fin-design note before logging off.
""",
    ),
    (
        1,
        """\
# {iso}

[08:50] Daily standup: yesterday merged P0 scaffolding; today P1 atomic file writes.
[11:00] Read python-frontmatter source — line offset is not exposed publicly.
[13:30] Garden co-op meeting; Roy needs help with the irrigation timer.
[16:10] Refactored TypeScript form validation in the side project.
[18:00] Sketched the jarvis-rebuild module boundary diagram.
""",
    ),
    (
        2,
        """\
# {iso}

[09:10] Daily standup: blocked on the Qwen tokenizer install on arch01.
[10:45] Decided to fall back to the 4-char approximation for now.
[12:00] Lunch alone; thought through MMR diversity for hybrid search.
[14:30] Garden harvest: pulled the first round of zucchini.
[19:00] Read about BM25 normalization vs cosine fusion.
""",
    ),
    (
        3,
        """\
# {iso}

[08:30] Daily standup: yesterday landed config schema; today integration tests.
[10:00] Ollama nomic-embed-text embedding dimension confirmed at 768.
[12:15] Tina's birthday planning — booked the dinner reservation.
[15:00] rocket-sim CFD run completed; results show fin sweep angle matters.
[18:30] Pushed a small fix to the daily-log rollover cron logic.
""",
    ),
    (
        4,
        """\
# {iso}

[09:00] Daily standup: deep work on the chunker's sliding window algorithm.
[11:30] Garden: planted the second-round basil starts near the south fence.
[14:00] Reviewed Anthropic's tool-use docs for the upcoming P5 work.
[16:45] Fast-model latency tuning on qwen2.5:3b — 320ms median.
[19:15] Wrote a TypeScript helper for the side project's timezone math.
""",
    ),
]

_PROJECT_ROCKET_SIM = """\
# rocket-sim

A small CFD-driven model rocket simulator.

## Goals
- Predict apogee within 5% for a given motor + airframe combo.
- Visualize the fin design's effect on stability margin.

## Fin design notes
- Trapezoidal fins with 30-degree sweep gave the best L/D in tests.
- Root chord 8cm, tip chord 4cm, span 6cm — current baseline geometry.
- Sweeping the angle past 35 degrees caused a sharp drag rise.

## Open questions
- Does adding a small tip cant reduce roll coupling?
"""

_PROJECT_JARVIS_REBUILD = """\
# jarvis-rebuild

Notes on the Jarvis-Mk3 rewrite.

## Architecture
- File-first memory under workspace/, indexed into SQLite + FTS5 + sqlite-vec.
- Three-gate Dreaming consolidates daily-log facts into MEMORY.md.

## Decisions
- Default tokenizer is qwen-native; approximation is fallback only.
- Jarvis CLI converges on jarvis.cli:main in P2 (reconcile, search, daemon).
"""

_PROJECT_GARDEN = """\
# garden

Roy's co-op garden plot tracking.

## Beds
- South bed: tomatoes (Sungold + Cherokee Purple) and basil.
- North bed: zucchini, cucumbers, summer squash.

## Recurring tasks
- Weekly: irrigation timer check; Roy usually handles the manifold.
- Monthly: top-dress with compost from the co-op pile.
"""


def populate(paths: WorkspacePaths) -> None:
    """Write all 50 planted facts. Idempotent (overwrites, but content is stable)."""
    bootstrap_workspace(paths)

    write_markdown_atomic(paths.memory_md, _MEMORY_BODY, tmp_dir=paths.tmp_dir)
    write_markdown_atomic(paths.user_md, _USER_BODY, tmp_dir=paths.tmp_dir)

    today = date.today()
    for offset, template in _DAILY_TEMPLATES:
        d = today - timedelta(days=offset)
        target = paths.daily_log(d)
        body = template.format(iso=d.isoformat())
        write_markdown_atomic(target, body, tmp_dir=paths.tmp_dir)

    write_markdown_atomic(
        paths.project("rocket-sim"), _PROJECT_ROCKET_SIM, tmp_dir=paths.tmp_dir
    )
    write_markdown_atomic(
        paths.project("jarvis-rebuild"), _PROJECT_JARVIS_REBUILD, tmp_dir=paths.tmp_dir
    )
    write_markdown_atomic(
        paths.project("garden"), _PROJECT_GARDEN, tmp_dir=paths.tmp_dir
    )


def _build_paths_for(root: Path) -> WorkspacePaths:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=root, shared_board=root.parent / "agent_bin")
    )
    return WorkspacePaths.from_config(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="populate_workspace")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="absolute path to the workspace to populate",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = _build_paths_for(root)
    populate(paths)
    print(f"populated workspace at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
