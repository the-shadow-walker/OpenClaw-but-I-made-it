"""
debug_logger.py — Dual-format debug logger for the Ollama Agent Service.

Writes every agent event to two files simultaneously:
  ./logs/agent_debug.jsonl  — one JSON object per line (machine-readable)
  ./logs/agent_debug.txt    — human-readable formatted trace

Other modules subscribe to receive events for SSE / webhook dispatch:
    import debug_logger
    debug_logger.subscribe(my_fn)   # fn(event_type: str, event: dict)
    debug_logger.unsubscribe(my_fn)

Override log directory:
    AGENT_LOG_DIR=/path/to/logs
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_DIR = Path(os.environ.get("AGENT_LOG_DIR", "./logs"))
JSONL_PATH = LOG_DIR / "agent_debug.jsonl"
TXT_PATH = LOG_DIR / "agent_debug.txt"

MAX_BYTES = 10 * 1024 * 1024  # rotate at 10 MB

_lock = threading.Lock()
_subscribers: List = []


# ─────────────────────────────── internal ───────────────────────────────────

def _rotate_if_needed(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size > MAX_BYTES:
            path.rename(path.with_suffix(path.suffix + ".1"))
    except OSError:
        pass


def _fmt_txt(event_type: str, ts: str, data: Dict[str, Any]) -> str:
    lines = [f"[{ts}]  {event_type.upper()} {'─' * max(0, 60 - len(event_type))}"]

    if event_type == "react_iter":
        job_short = str(data.get("job_id", ""))[:8]
        lines.append(f"  Job  : {job_short}  iter {data.get('iteration')}/{data.get('max_iter')}")
        lines.append(f"  💭   : {data.get('thought', '')[:200]}")
        lines.append(f"  🎯   : {data.get('tool')}  ({data.get('confidence', '?')}%)")
        args = data.get("args", {})
        if "command" in args:
            lines.append(f"  cmd  : {str(args['command'])[:200]}")
        elif "path" in args:
            lines.append(f"  path : {args['path']}")
        result = data.get("result", {})
        ok = "✅" if result.get("success") else "❌"
        lines.append(f"  {ok}    : {str(result.get('output', ''))[:200]}")
        if not result.get("success") and result.get("error"):
            lines.append(f"  err  : {str(result.get('error', ''))[:200]}")
        if data.get("duration_ms"):
            lines.append(f"  ⏱️    : {data['duration_ms']} ms")

    elif event_type == "model_call":
        lines.append(f"  model  : {data.get('model')}")
        lines.append(f"  purpose: {data.get('purpose', 'react')}")
        lines.append(f"  ⏱️      : {data.get('duration_ms', '?')} ms")
        if data.get("error"):
            lines.append(f"  error  : {data['error'][:200]}")

    elif event_type == "job_start":
        lines.append(f"  job_id : {str(data.get('job_id', ''))[:8]}")
        lines.append(f"  instr  : {str(data.get('instruction', ''))[:150]}")
        if data.get("chain_id"):
            lines.append(f"  chain  : {str(data['chain_id'])[:8]}  subtask={data.get('subtask_index')}")

    elif event_type == "job_end":
        lines.append(f"  job_id : {str(data.get('job_id', ''))[:8]}")
        lines.append(f"  status : {data.get('status')}  success={data.get('success')}")
        lines.append(f"  iters  : {data.get('iterations_used', '?')}")
        if data.get("summary"):
            lines.append(f"  summary: {str(data['summary'])[:200]}")

    elif event_type == "chain_start":
        lines.append(f"  chain  : {str(data.get('chain_id', ''))[:8]}")
        lines.append(f"  goal   : {str(data.get('goal', ''))[:150]}")
        lines.append(f"  tasks  : {data.get('subtask_count')}")

    elif event_type == "chain_end":
        lines.append(f"  chain  : {str(data.get('chain_id', ''))[:8]}")
        lines.append(f"  status : {data.get('status')}")
        lines.append(f"  goal   : {str(data.get('goal', ''))[:150]}")

    elif event_type == "subtask_event":
        lines.append(f"  chain  : {str(data.get('chain_id', ''))[:8]}")
        lines.append(f"  idx    : {data.get('subtask_index')}")
        lines.append(f"  event  : {data.get('sub_event')}")
        lines.append(f"  instr  : {str(data.get('instruction', ''))[:150]}")
        if data.get("ac_command"):
            ok = "✅" if data.get("ac_passed") else "❌"
            lines.append(f"  AC {ok}  : {data['ac_command']}")

    elif event_type == "inbox_job":
        lines.append(f"  file   : {data.get('filename')}")
        lines.append(f"  kind   : {data.get('kind')}")
        lines.append(f"  instr  : {str(data.get('instruction', data.get('goal', '')))[:150]}")

    elif event_type == "error":
        lines.append(f"  context: {data.get('context', '')}")
        lines.append(f"  error  : {str(data.get('error', ''))[:300]}")

    else:
        for k, v in list(data.items())[:8]:
            lines.append(f"  {k}: {str(v)[:200]}")

    lines.append("")
    return "\n".join(lines) + "\n"


# ─────────────────────────────── public API ─────────────────────────────────

def subscribe(fn) -> None:
    """Register a callable invoked as fn(event_type, event_dict) on every log."""
    if fn not in _subscribers:
        _subscribers.append(fn)


def unsubscribe(fn) -> None:
    _subscribers[:] = [f for f in _subscribers if f is not fn]


def log(event_type: str, data: Dict[str, Any]) -> None:
    """Append event to both log files and notify all subscribers."""
    ts = datetime.now().isoformat()
    event = {"ts": ts, "event": event_type, **data}

    with _lock:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed(JSONL_PATH)
        _rotate_if_needed(TXT_PATH)

        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")

        with open(TXT_PATH, "a", encoding="utf-8") as f:
            f.write(_fmt_txt(event_type, ts, data))

    # Notify outside the lock to avoid deadlocks
    dead = []
    for fn in list(_subscribers):
        try:
            fn(event_type, event)
        except Exception:
            dead.append(fn)
    for fn in dead:
        unsubscribe(fn)


# ─────────────────────────── convenience wrappers ───────────────────────────

def job_start(job_id: str, instruction: str,
              chain_id: Optional[str] = None, subtask_index: Optional[int] = None):
    log("job_start", {
        "job_id": job_id,
        "instruction": instruction[:500],
        "chain_id": chain_id,
        "subtask_index": subtask_index,
    })


def job_end(job_id: str, instruction: str, status: str, success: bool,
            iterations_used: int = 0, summary: str = "",
            chain_id: Optional[str] = None):
    log("job_end", {
        "job_id": job_id,
        "instruction": instruction[:200],
        "status": status,
        "success": success,
        "iterations_used": iterations_used,
        "summary": summary[:300],
        "chain_id": chain_id,
    })


def react_iter(job_id: str, iteration: int, max_iter: int,
               thought: str, tool: str, args: dict, result: Any,
               confidence: int = 50, duration_ms: int = 0):
    result_dict = {
        "success": getattr(result, "success", False),
        "output": str(getattr(result, "output", ""))[:300],
        "error": str(getattr(result, "error", ""))[:200],
    }
    log("react_iter", {
        "job_id": job_id,
        "iteration": iteration,
        "max_iter": max_iter,
        "thought": thought[:300],
        "tool": tool,
        "args": {k: str(v)[:200] for k, v in args.items()},
        "confidence": confidence,
        "result": result_dict,
        "duration_ms": duration_ms,
    })


def model_call(model: str, purpose: str, duration_ms: int, error: str = ""):
    log("model_call", {
        "model": model,
        "purpose": purpose,
        "duration_ms": duration_ms,
        "error": error,
    })


def chain_start(chain_id: str, goal: str, subtask_count: int):
    log("chain_start", {
        "chain_id": chain_id,
        "goal": goal[:300],
        "subtask_count": subtask_count,
    })


def chain_end(chain_id: str, goal: str, status: str):
    log("chain_end", {
        "chain_id": chain_id,
        "goal": goal[:200],
        "status": status,
    })


def subtask_event(chain_id: str, sub_event: str, subtask_index: int,
                  instruction: str = "", ac_command: str = "",
                  ac_passed: Optional[bool] = None):
    log("subtask_event", {
        "chain_id": chain_id,
        "sub_event": sub_event,
        "subtask_index": subtask_index,
        "instruction": instruction[:200],
        "ac_command": ac_command,
        "ac_passed": ac_passed,
    })


def inbox_job(filename: str, kind: str, instruction: str = "", goal: str = ""):
    log("inbox_job", {
        "filename": filename,
        "kind": kind,
        "instruction": instruction[:200],
        "goal": goal[:200],
    })


def error(context: str, err: Exception):
    log("error", {"context": context, "error": str(err)[:500]})
