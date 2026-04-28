"""Light integration tests for jarvis.cli — subcommands, env override, daemon shim."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jarvis.config import JarvisConfig, PathsConfig
from jarvis.memory.chunker import configure_tokenizer
from jarvis.memory.workspace import WorkspacePaths, bootstrap_workspace
from tests.fixtures.populate_workspace import populate


@pytest.fixture(autouse=True)
def _approx_tokenizer():
    configure_tokenizer("approximation")


def _make_workspace(tmp_path: Path) -> Path:
    cfg = JarvisConfig(
        paths=PathsConfig(workspace=tmp_path / "ws", shared_board=tmp_path / "agent_bin")
    )
    paths = WorkspacePaths.from_config(cfg)
    bootstrap_workspace(paths)
    populate(paths)
    return paths.root


def _run_cli(argv: list[str], workspace: Path) -> subprocess.CompletedProcess:
    """Run `python -m jarvis.cli ...` with JARVIS_WORKSPACE set, no config file."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workspace.parent),
        "JARVIS_WORKSPACE": str(workspace),
    }
    # Make sure we don't pick up the developer's ~/.config/jarvis/config.yaml.
    env["JARVIS_CONFIG"] = str(workspace / "_no_such_config.yaml")
    # We want it to load defaults only. JARVIS_CONFIG at a missing path makes
    # load_config raise — instead just point HOME at a fresh dir so the default
    # ~/.config/jarvis/config.yaml is absent.
    env.pop("JARVIS_CONFIG", None)

    return subprocess.run(
        [sys.executable, "-m", "jarvis.cli", *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_reconcile_then_search(tmp_path: Path):
    workspace = _make_workspace(tmp_path)

    rec = _run_cli(["reconcile"], workspace)
    assert rec.returncode == 0, rec.stderr
    assert "reconciled" in rec.stdout

    res = _run_cli(["search", "trapezoidal", "-k", "3"], workspace)
    assert res.returncode == 0, res.stderr
    # At least one result line — the score-prefixed format.
    out_lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    assert any("rocket-sim" in ln for ln in out_lines), res.stdout


def test_cli_search_output_stable_across_rebuild(tmp_path: Path):
    workspace = _make_workspace(tmp_path)
    queries = [
        "rocket fin",
        "typescript",
        "daily log",
        "jarvis-rebuild",
        "garden",
        "standup",
        "preferences",
        "fast model",
        "grant",
        "reconcile",
    ]

    # First reconcile + capture stdout.
    rec1 = _run_cli(["reconcile"], workspace)
    assert rec1.returncode == 0, rec1.stderr
    before = []
    for q in queries:
        r = _run_cli(["search", q, "-k", "3"], workspace)
        assert r.returncode == 0, r.stderr
        before.append((q, r.stdout))

    # Wipe DB + rebuild.
    db = workspace / ".index" / "memory.sqlite"
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db) + suffix)
        if p.exists():
            p.unlink()
    rec2 = _run_cli(["reconcile"], workspace)
    assert rec2.returncode == 0, rec2.stderr

    after = []
    for q in queries:
        r = _run_cli(["search", q, "-k", "3"], workspace)
        assert r.returncode == 0, r.stderr
        after.append((q, r.stdout))

    assert before == after, "search output must be byte-equal across delete+rebuild"


def test_cli_daemon_starts_watcher_and_responds_to_sigterm(tmp_path: Path):
    """`jarvis daemon` brings up watcher + uvicorn, shuts down cleanly on SIGTERM.

    Catches four things at once: (a) daemon subcommand still routes into
    ``run.py``, (b) watcher actually starts, (c) FastAPI server hits its
    startup hook, (d) SIGTERM shuts it down cleanly with exit code 0.
    """
    import socket as _sock

    workspace = _make_workspace(tmp_path)

    # Pick a free port — port 5003 default conflicts in parallel test runs.
    s = _sock.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    # Write a config file with the test port so daemon binds something free.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"server:\n"
        f"  host: 127.0.0.1\n"
        f"  port: {port}\n"
        f"paths:\n"
        f"  workspace: {workspace}\n"
        f"  shared_board: {workspace.parent}/agent_bin\n",
        encoding="utf-8",
    )

    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workspace.parent),
        "JARVIS_WORKSPACE": str(workspace),
        "JARVIS_CONFIG": str(cfg_path),
    }
    # Inherit PYTHONPATH so the in-tree package import keeps working under
    # the subprocess.
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]

    proc = subprocess.Popen(
        [sys.executable, "-m", "jarvis.cli", "daemon"],
        env=env,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15.0
        seen = ""
        saw_watcher = False
        saw_server = False
        while time.monotonic() < deadline and not (saw_watcher and saw_server):
            line = proc.stderr.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            seen += line
            if "watcher running" in line:
                saw_watcher = True
            if "server listening" in line:
                saw_server = True
        assert saw_watcher, f"daemon never logged watcher start; saw:\n{seen}"
        assert saw_server, f"daemon never logged server start; saw:\n{seen}"
        proc.send_signal(signal.SIGTERM)
        # uvicorn 0.46+ catches the signal, runs graceful shutdown, then
        # re-raises so the parent can chain handlers (see Server.capture_signals).
        # That makes the documented graceful-exit code -SIGTERM, not 0.
        # systemd treats this as success (Type=simple + SIGTERM exit = clean).
        rc = proc.wait(timeout=10.0)
        assert rc in (0, -signal.SIGTERM), f"unexpected exit code: {rc}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_tokenizer_error_log_when_qwen_unavailable(tmp_path: Path, caplog):
    """In-process: configure_tokenizer('qwen-native') with transformers patched out
    must emit an ERROR via the chunker logger."""
    from jarvis.memory import chunker

    # Force the loader to fail so we hit the error path.
    real_loader = chunker._try_load_qwen_tokenizer
    chunker._try_load_qwen_tokenizer = lambda: None
    try:
        with caplog.at_level(logging.ERROR, logger="jarvis.memory.chunker"):
            resolved = chunker.configure_tokenizer("qwen-native")
        assert resolved == "approximation"
        assert any(
            r.levelno == logging.ERROR and "qwen-native" in r.message
            for r in caplog.records
        ), caplog.records
    finally:
        chunker._try_load_qwen_tokenizer = real_loader
        configure_tokenizer("approximation")
