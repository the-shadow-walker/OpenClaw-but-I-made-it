"""P0 smoke: verify package imports + config defaults validate.

The original ``test_run_stub_exits_zero`` was retired in P4: ``run.main``
no longer prints a banner and exits — it starts the watcher and blocks
on a signal. The subprocess-based equivalent lives in
``test_cli.py::test_cli_daemon_starts_watcher_and_responds_to_sigterm``.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import jarvis

    assert jarvis.__version__ == "0.1.0"


def test_config_defaults_validate(tmp_path, monkeypatch) -> None:
    """Defaults should construct cleanly and have the spec-mandated values."""
    # Point path-validating fields at a writable temp dir so the loader's mkdir succeeds
    # in CI / sandboxed runs that can't write to /mnt/storage/NAS/Jarvis.
    monkeypatch.delenv("JARVIS_CONFIG", raising=False)
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        "paths:\n"
        f"  workspace: {tmp_path / 'workspace'}\n"
        f"  shared_board: {tmp_path / 'agent_bin'}\n"
        "mirror:\n"
        f"  central_context_md: {tmp_path / 'agent_bin' / 'central_context.md'}\n",
        encoding="utf-8",
    )

    from jarvis.config import load_config

    cfg = load_config(cfg_yaml)
    assert cfg.server.port == 5003
    assert cfg.llm.chat_model == "qwen2.5:3b"
    assert cfg.conversation.compaction.keep_recent_turns == 6        # spec §2.5 superset
    assert cfg.dreaming.deep_sleep.gates.min_score == 0.80           # spec §2.4
    assert cfg.dreaming.enabled is False                             # spec §21.1
    assert cfg.heartbeat.enabled is False                            # spec §21.4
    assert cfg.mirror.enabled is False                               # spec §21.9


def test_run_main_importable() -> None:
    """``run.main`` is now blocking; just confirm the entry symbol exists."""
    from jarvis.run import main

    assert callable(main)
