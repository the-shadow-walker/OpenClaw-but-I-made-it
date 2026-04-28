"""Rule-based router — classifies a user turn into a routing hint.

Pure function. The chat loop logs the result at DEBUG and otherwise
ignores it; the LLM remains the routing arbiter. P10 swaps this out for
an LLM-driven router that *does* short-circuit, but the wire-point and
log line stay; the regex implementation here is the placeholder hint.

P7 only ever produces ``cmd_quick / cmd_react / cmd_chain / direct``.
The ``swarm_*`` and ``multi_phase`` literals are reserved so the type
doesn't have to grow when P8/P10 ship.

Anti-pattern: never short-circuit the chat loop based on ``classify()``
— it logs only. The LLM is the arbiter. The DEBUG-level log keeps the
hint visible during P10 development without spamming journalctl on
every chat turn.
"""

from __future__ import annotations

import re
from typing import Literal

__all__ = ["RouterClass", "classify"]


RouterClass = Literal[
    "direct",
    "cmd_quick",
    "cmd_react",
    "cmd_chain",
    "swarm_math",
    "swarm_engineer",
    "swarm_research",
    "multi_phase",
]


# Quick-style intent: factual / status / shell-question phrasing.
_QUICK_RE = re.compile(
    r"^(is|are|how|what|whether|run|check|status|uptime|disk|memory|process)\b",
    re.IGNORECASE,
)

# ReAct-style intent: build / write / fix verbs that imply a multi-step coding task.
_BUILD_RE = re.compile(
    r"\b(create|build|write|implement|patch|fix|debug|refactor)\b",
    re.IGNORECASE,
)

# Chain-style intent: explicit multi-phase phrasing.
_CHAIN_RE = re.compile(
    r"\b(phase|chain|multi-step)\b|\bfirst\b.*\bthen\b|\bstep\s+\d+\b|\bphase\s+\d+\b",
    re.IGNORECASE,
)

# P8 specialist hints — intentionally permissive. The router is hint-only
# (DEBUG log; never short-circuits dispatch). The LLM is the final
# arbiter and will pick a direct answer over delegation when appropriate.
_MATH_RE = re.compile(
    r"\b(equation|derive|integrate|ode|solve|formula|kinematics|dynamics)\b",
    re.IGNORECASE,
)
_ENG_RE = re.compile(
    r"\b(bom|schematic|datasheet|cad|circuit|component|board|pcb)\b",
    re.IGNORECASE,
)
_RESEARCH_RE = re.compile(
    r"\b(research|background|literature|survey|sources?|cite)\b",
    re.IGNORECASE,
)

# Multi-phase: "sim"/"simulator"/"simulation" noun OR an explicit "build/
# create/design ... and ... and" enumeration. The "sim" branch is
# intentionally permissive — "explain how a flight sim works" will also
# match, but the router is hint-only and the LLM picks 'direct' for
# explanatory questions. Future readers: do NOT tighten this regex.
_MULTI_PHASE_RE = re.compile(
    r"\bsim(ulator|ulation)?\b"
    r"|\b(build|create|design)\b.*\band\b.*\band\b",
    re.IGNORECASE,
)


def classify(user_text: str) -> RouterClass:
    """Return the rule-based routing hint for ``user_text``.

    Precedence: ``multi_phase`` > ``cmd_chain`` > ``swarm_*`` >
    ``cmd_react`` > ``cmd_quick`` > ``direct``. ``cmd_quick`` requires
    text shorter than 200 chars (a 250-char "is X?" is more likely a
    discussion than a status query). Empty input is ``direct``.
    """
    text = (user_text or "").strip()
    if not text:
        return "direct"
    if _MULTI_PHASE_RE.search(text):
        return "multi_phase"
    if _CHAIN_RE.search(text):
        return "cmd_chain"
    if _MATH_RE.search(text):
        return "swarm_math"
    if _ENG_RE.search(text):
        return "swarm_engineer"
    if _RESEARCH_RE.search(text):
        return "swarm_research"
    if _BUILD_RE.search(text):
        return "cmd_react"
    if _QUICK_RE.search(text) and len(text) < 200:
        return "cmd_quick"
    return "direct"
