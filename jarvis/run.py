"""Jarvis CLI entry point.

Phase P0: stub. Prints a banner, validates that config can be loaded if a config file
exists at the default path, exits 0. Future phases will dispatch to subcommands
(`jarvis chat`, `jarvis dreaming on`, etc.) and to the FastAPI server.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """P0 stub: print banner, optionally validate config, exit 0."""
    print("jarvis stub — phase P0")

    # If a config file exists at the default path, try loading it as a smoke check.
    # Absent config is fine at P0 — the loader has sensible defaults.
    from jarvis.config import DEFAULT_CONFIG_PATH, load_config

    cfg_path = Path(DEFAULT_CONFIG_PATH).expanduser()
    if cfg_path.exists():
        try:
            cfg = load_config(cfg_path)
            print(f"  config: loaded {cfg_path} — port {cfg.server.port}, workspace {cfg.paths.workspace}")
        except Exception as e:  # noqa: BLE001 — surface any config error
            print(f"  config: FAILED to load {cfg_path}: {e}", file=sys.stderr)
            return 1
    else:
        # Validate defaults still parse cleanly even with no file present.
        try:
            cfg = load_config(None)
            print(f"  config: defaults OK — port {cfg.server.port}")
        except Exception as e:  # noqa: BLE001
            print(f"  config: defaults FAILED to validate: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
