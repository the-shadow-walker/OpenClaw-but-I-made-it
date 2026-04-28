"""Rule-based router — one row per regex branch + boundary cases."""

from __future__ import annotations

import pytest

from jarvis.core.router import classify


@pytest.mark.parametrize(
    "text,expected",
    [
        # cmd_quick — leading status verb + short.
        ("is the disk full?", "cmd_quick"),
        ("are the workers running?", "cmd_quick"),
        ("how is the cpu load?", "cmd_quick"),
        ("uptime", "cmd_quick"),
        ("status", "cmd_quick"),

        # cmd_react — build/write/fix verbs.
        ("build a CLI for X", "cmd_react"),
        ("write a python script that does Y", "cmd_react"),
        ("fix the failing tests", "cmd_react"),
        ("debug the auth flow", "cmd_react"),

        # cmd_chain — explicit multi-phase phrasing.
        ("phase 1: scaffold; phase 2: tests", "cmd_chain"),
        ("first set up the project, then run the tests", "cmd_chain"),
        ("step 1: foo; step 2: bar", "cmd_chain"),
        ("multi-step: download then index", "cmd_chain"),

        # Mixed — chain dominates over react.
        ("build the project in phase 1 then verify in phase 2", "cmd_chain"),

        # Length boundary — long question is direct.
        ("is " + ("very long " * 30) + "?", "direct"),  # >200 chars

        # Plain content — direct.
        ("remember that I like Python", "direct"),
        ("hello there", "direct"),

        # Empty / whitespace.
        ("", "direct"),
        ("   ", "direct"),

        # P8 — swarm specialist hints.
        ("derive the equation for projectile motion", "swarm_math"),
        ("solve this ODE for harmonic motion", "swarm_math"),
        ("explain dynamics of a pendulum", "swarm_math"),
        ("generate a BOM for the controller", "swarm_engineer"),
        ("find a datasheet for the op-amp", "swarm_engineer"),
        ("design a PCB layout", "swarm_engineer"),
        ("research the literature on rocket nozzles", "swarm_research"),
        ("survey background sources for X", "swarm_research"),
        ("cite a paper on aerodynamics", "swarm_research"),

        # P8 — multi_phase via 'sim' noun.
        ("Build me a single-stage rocket simulator", "multi_phase"),
        ("design a flight simulation environment", "multi_phase"),
        # Permissive: the explanatory question still hints multi_phase
        # — router is hint-only; the LLM picks 'direct' downstream.
        ("explain how a flight sim works", "multi_phase"),

        # P8 — multi_phase via 'build/create/design ... and ... and'.
        ("build a model and write code and add docs", "multi_phase"),

        # Precedence: multi_phase dominates over math/engineer/research.
        ("derive the equations for the rocket simulator", "multi_phase"),
        ("research aerodynamics for the simulator", "multi_phase"),
    ],
)
def test_classify(text: str, expected: str) -> None:
    assert classify(text) == expected
