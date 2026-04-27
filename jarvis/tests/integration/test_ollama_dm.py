"""P1 exit-criterion live-call test.

Bootstraps a tmp workspace, plants a fact in MEMORY.md, builds a DM system
prompt, and asks qwen2.5:3b a grounded question. Skips cleanly if Ollama is
not reachable (Mac dev shell); MUST pass on arch01 where ollama serves
qwen2.5:3b at http://localhost:11434.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from jarvis.clients.ollama import OllamaClient
from jarvis.config import JarvisConfig, PathsConfig, load_config
from jarvis.core.prompt import assemble_system_prompt
from jarvis.memory.files import write_markdown_atomic
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace


def test_dm_grounded_answer(tmp_path: Path) -> None:
    # Use real config if present (so we hit the deployment's configured
    # ollama_host/chat_model), otherwise defaults.
    try:
        cfg = load_config()
    except Exception:
        cfg = JarvisConfig(
            paths=PathsConfig(
                workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin"
            )
        )

    # ALWAYS use a tmp workspace for this test — never pollute the real one.
    cfg = cfg.model_copy(deep=True)
    object.__setattr__(cfg.paths, "workspace", tmp_path / "ws")

    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)

    # Plant the fact.
    write_markdown_atomic(
        paths.memory_md,
        "# MEMORY.md\n\n- The user prefers TypeScript for new projects.\n",
        tmp_dir=paths.tmp_dir,
    )

    system = assemble_system_prompt(paths, "dm")
    client = OllamaClient(cfg.llm.ollama_host, timeout_s=120.0)
    try:
        try:
            answer = client.complete(
                cfg.llm.chat_model,
                [{"role": "user", "content": "what programming language does the user prefer?"}],
                system=system,
                temperature=0.0,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            pytest.skip(f"ollama unreachable at {cfg.llm.ollama_host}: {e}")
    finally:
        client.close()

    # qwen2.5:3b is small and may answer "TypeScript" or "TS"; token-boundary
    # regex prevents false positives on "tests" or "typescripts".
    assert re.search(r"\b(typescript|ts)\b", answer, re.IGNORECASE), (
        f"expected typescript/ts in answer, got: {answer!r}"
    )
