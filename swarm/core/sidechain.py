#!/usr/bin/env python3
"""
Sidechain — JSONL writer for ReAct traces.

When swarm runs as a subagent (SWARM_AS_SUBAGENT=1), each role's full ReAct
loop is dumped to ~/.agent_bin/sidechains/swarm_<role>_<job_id>_<ts>.jsonl
so the parent agent's transcript stays clean (only the SubAgentResult
summary flows back).

Direct API calls (where swarm is the top-level agent) skip the sidechain to
avoid disk bloat — make_sidechain returns None when the env flag is unset.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Optional

SIDECHAIN_DIR = os.path.expanduser("~/.agent_bin/sidechains")
ENV_FLAG = "SWARM_AS_SUBAGENT"


class SidechainWriter:
    """Line-buffered, thread-safe JSONL writer."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Touch the file so callers can rely on its existence
        with open(self.path, "a"):
            pass

    def write_event(self, event_type: str, **fields: Any) -> None:
        record = {"ts": datetime.utcnow().isoformat(), "event": event_type}
        record.update(fields)
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def close(self) -> None:
        # Files are opened per-write; nothing to close. Method here for symmetry.
        return


def make_sidechain(role: str, job_id: str) -> Optional[SidechainWriter]:
    """Return a SidechainWriter when invoked as a subagent, else None.

    Path: ~/.agent_bin/sidechains/swarm_<role>_<job_id>_<utcts>.jsonl
    """
    if os.getenv(ENV_FLAG, "0") != "1":
        return None
    safe_role = "".join(c if c.isalnum() else "_" for c in role)[:30] or "unknown"
    safe_job = "".join(c if c.isalnum() else "_" for c in job_id)[:30] or "nojob"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(SIDECHAIN_DIR, f"swarm_{safe_role}_{safe_job}_{ts}.jsonl")
    return SidechainWriter(path)


if __name__ == "__main__":
    os.environ[ENV_FLAG] = "1"
    sc = make_sidechain("test", "smoke")
    assert sc is not None
    sc.write_event("turn_header", turn=1, sp_id="SP_A")
    sc.write_event("run_code", n_chars=42, ok=True)
    print(f"OK — wrote 2 events to {sc.path}")
