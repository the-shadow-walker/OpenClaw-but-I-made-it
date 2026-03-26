#!/usr/bin/env python3
"""
Ollama Command Agent Service
A REST API service that executes commands via the Ollama agent
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import threading
import queue
import time
import json
import uuid
import os
from datetime import datetime
from typing import Dict, Any, Optional
import secrets
import sys

# Modules live in cmd/ and its subpackages. Add all of them to sys.path so
# existing flat imports (from ollama_agent_core import X, etc.) keep working
# without any changes inside the module files themselves.
_CMD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmd")
for _sub in ("", "core", "chain", "blueteam", "infra"):
    _d = os.path.join(_CMD_DIR, _sub) if _sub else _CMD_DIR
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

# Import the agent (assumes ollama_agent.py is in same directory or cmd/)
from ollama_agent_core import OllamaCommandAgent
from task_chain import (HandoffExtractor, AcceptanceCriteriaRunner, SubtaskReplanner,
                        TaskDecomposer, TaskChain, cleanup_between_phases,
                        SubtaskOrchestrator, ImplementationArtifact)
import debug_logger
import webhook_dispatcher

try:
    from blueteam_agent import get_sentinel as _get_sentinel, get_recent_alerts as _bt_alerts
    _BLUETEAM_AVAILABLE = True
except ImportError:
    _BLUETEAM_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Thread-safe output capture: each worker thread routes print() to its own buffer
# without touching global builtins.print (which is not thread-safe to monkey-patch).
import builtins as _builtins
_print_local = threading.local()
_original_print = _builtins.print

def _router_print(*args, **kwargs):
    handler = getattr(_print_local, 'capture_fn', None)
    if handler is not None:
        handler(' '.join(map(str, args)))
    else:
        _original_print(*args, **kwargs)

_builtins.print = _router_print

# Configuration
API_KEY = os.environ.get('AGENT_API_KEY', secrets.token_urlsafe(32))
MAX_CONCURRENT_JOBS = 3
JOB_TIMEOUT = 3600  # 1 hour max per job

SERVICE_FEATURES = [
    "Studio Model Hierarchy (Director → Producer → Minions)",
    "SubtaskOrchestrator: 3-5 Minions per phase, clean context each",
    "Minion tool whitelists: CODER={read,create,patch,finish} COMMANDER={exec,server,read,finish}",
    "Patch no-op detection + per-file counter (block@5) + syntax check",
    "NUM_CTX 16384 (was 32768), history window 12 (was 20)",
    "Pinned messages: ARCH.md survives history trimming",
    "cleanup_between_phases(): kills zombie dev ports between subtasks",
    "TaskDecomposer: forced Phase-0 ARCH.md, modular layout, DB migration phase",
    "Dual debug logs: ./logs/agent_debug.jsonl + ./logs/agent_debug.txt",
    "Outbound webhooks: AGENT_WEBHOOK_URLS env var (fire-and-forget)",
    "Inbox watcher: drop job/chain JSON into ./agent_inbox/ for pickup",
    "SSE event stream: GET /api/v1/events | State snapshot: GET /api/v1/state",
    "SENTINEL blue team: /api/v1/blueteam/* (scan, investigate, watch, alerts, status)",
]

SERVICE_VERSION = "3.3.0-sentinel"

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}
job_queue = queue.Queue()
active_jobs = {}

# ── SSE / event mesh ─────────────────────────────────────────────────────────
# Per-connection queues; debug_logger fans events into all of them.
_sse_clients: set = set()
_sse_lock = threading.Lock()

def _sse_fan_out(event_type: str, event: dict) -> None:
    with _sse_lock:
        dead = set()
        for q in _sse_clients:
            try:
                q.put_nowait(event)
            except Exception:
                dead.add(q)
        _sse_clients.difference_update(dead)

debug_logger.subscribe(_sse_fan_out)

# ── state file ────────────────────────────────────────────────────────────────
_STATE_PATH = "./agent_state.json"

def _write_state() -> None:
    """Atomically refresh agent_state.json for external LLMs to poll."""
    try:
        from task_chain import TaskChain as _TC
        running_chains = [c for c in _TC.list_all() if c["status"] == "running"]
    except Exception:
        running_chains = []
    state = {
        "updated_at": datetime.now().isoformat(),
        "service_version": SERVICE_VERSION,
        "active_jobs": [
            {
                "job_id": jid,
                "instruction": j.get("instruction", "")[:200],
                "chain_id": j.get("chain_id"),
                "subtask_index": j.get("subtask_index"),
                "started_at": j.get("started_at"),
            }
            for jid, j in list(active_jobs.items())
        ],
        "queued_jobs": job_queue.qsize(),
        "running_chains": running_chains,
    }
    tmp = _STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, _STATE_PATH)

# ── inbox watcher ─────────────────────────────────────────────────────────────
_INBOX_DIR = "./agent_inbox"
_INBOX_PROCESSED = "./agent_inbox/processed"

def _inbox_watcher() -> None:
    """Poll ./agent_inbox/ every 5 s for job files dropped by other LLMs."""
    os.makedirs(_INBOX_DIR, exist_ok=True)
    os.makedirs(_INBOX_PROCESSED, exist_ok=True)
    while True:
        try:
            for fname in os.listdir(_INBOX_DIR):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(_INBOX_DIR, fname)
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                except Exception:
                    continue
                try:
                    if data.get("blueteam_scan") and _BLUETEAM_AVAILABLE:
                        # SENTINEL security scan
                        focus = data.get("focus", "")
                        job_id = str(uuid.uuid4())
                        job = {
                            'job_id': job_id, 'job_type': 'blueteam_scan',
                            'instruction': 'SENTINEL security scan (inbox)',
                            'focus': focus, 'model': 'qwen3-coder:30b',
                            'searxng_url': 'http://10.0.0.58:8080',
                            'status': 'queued',
                            'created_at': datetime.now().isoformat(),
                            'output': '', 'execution_log': [], 'success': None,
                        }
                        jobs[job_id] = job
                        job_runner.submit_job(job_id)
                        debug_logger.inbox_job(fname, "blueteam_scan", instruction="SENTINEL scan")
                        _original_print(f"[inbox] blueteam scan {job_id[:8]} submitted from {fname}")
                    elif data.get("blueteam_investigate") and _BLUETEAM_AVAILABLE:
                        # SENTINEL investigation
                        finding = str(data["blueteam_investigate"])
                        job_id = str(uuid.uuid4())
                        job = {
                            'job_id': job_id, 'job_type': 'blueteam_investigate',
                            'instruction': f'SENTINEL investigation: {finding[:100]}',
                            'finding': finding, 'evidence': data.get("evidence", ""),
                            'model': 'qwen3-coder:30b',
                            'searxng_url': 'http://10.0.0.58:8080',
                            'status': 'queued',
                            'created_at': datetime.now().isoformat(),
                            'output': '', 'execution_log': [], 'success': None,
                        }
                        jobs[job_id] = job
                        job_runner.submit_job(job_id)
                        debug_logger.inbox_job(fname, "blueteam_investigate", instruction=finding[:100])
                        _original_print(f"[inbox] blueteam investigate {job_id[:8]} submitted from {fname}")
                    elif data.get("goal"):
                        # Chain submission
                        goal = data["goal"]
                        budget = int(data.get("total_budget", 200))
                        decomposer = TaskDecomposer()
                        subtasks = decomposer.decompose(goal, budget)
                        chain = TaskChain.create(goal=goal, subtasks=subtasks)
                        _submit_subtask_job(chain.chain_id, chain.data, 0, 0, None)
                        debug_logger.inbox_job(fname, "chain", goal=goal)
                        _original_print(f"[inbox] chain {chain.chain_id[:8]} submitted from {fname}")
                    elif data.get("instruction"):
                        # Single job submission
                        instr = data["instruction"]
                        job_id = str(uuid.uuid4())
                        job = {
                            "job_id": job_id,
                            "instruction": instr,
                            "model": data.get("model", "qwen3-coder:30b"),
                            "searxng_url": "http://10.0.0.58:8080",
                            "max_iterations": int(data.get("max_iterations", 25)),
                            "status": "queued",
                            "created_at": datetime.now().isoformat(),
                            "output": "",
                            "execution_log": [],
                            "success": None,
                        }
                        jobs[job_id] = job
                        job_queue.put(job_id)
                        debug_logger.inbox_job(fname, "job", instruction=instr)
                        _original_print(f"[inbox] job {job_id[:8]} submitted from {fname}")
                    # Move to processed/
                    os.replace(fpath, os.path.join(_INBOX_PROCESSED, fname))
                except Exception as e:
                    _original_print(f"[inbox] error processing {fname}: {e}")
        except Exception:
            pass
        time.sleep(5)


class JobRunner:
    """Runs agent jobs in background threads"""
    
    def __init__(self):
        self.running = True
        self.workers = []
        
        # Start worker threads
        for i in range(MAX_CONCURRENT_JOBS):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"Worker-{i}")
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        """Worker thread that processes jobs from the queue"""
        while self.running:
            try:
                job_id = job_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if job_id not in jobs:
                continue
            
            job = jobs[job_id]
            
            try:
                # Update job status
                job['status'] = 'running'
                job['started_at'] = datetime.now().isoformat()
                active_jobs[job_id] = job
                debug_logger.job_start(job_id, job.get('instruction', ''),
                                       chain_id=job.get('chain_id'),
                                       subtask_index=job.get('subtask_index'))
                _write_state()

                # Create agent
                agent = OllamaCommandAgent(
                    model=job.get('model', 'qwen3-coder:30b'),
                    searxng_url=job.get('searxng_url', 'http://10.0.0.58:8080')
                )
                agent.current_job_id = job_id
                
                # Capture output — thread-local so concurrent workers don't conflict
                output_buffer = []

                def capture_print(text):
                    output_buffer.append(text)
                    job['output'] = '\n'.join(output_buffer)
                    # Mirror to stderr so journalctl shows live agent output
                    sys.stderr.write(text + '\n')
                    sys.stderr.flush()

                _print_local.capture_fn = capture_print

                try:
                    # Honor per-job max_iterations (watchdog uses 15 vs default 25)
                    if 'max_iterations' in job:
                        agent.max_react_iterations = int(job['max_iterations'])

                    # Chain subtasks use SubtaskOrchestrator (Producer tier):
                    # breaks each subtask into 3-5 micro-tasks → Minion agents.
                    # Single jobs bypass the orchestrator and run directly.
                    _original_print(f"[worker] job {job_id[:8]} starting — chain={bool(job.get('chain_id'))} subtask={job.get('subtask_index')}", flush=True)

                    if job.get('job_type') in ('blueteam_scan', 'blueteam_investigate'):
                        # Blue team security job — run via SENTINEL singleton
                        if not _BLUETEAM_AVAILABLE:
                            raise RuntimeError("blueteam_agent module not available")
                        sentinel = _get_sentinel()
                        sentinel.agent.current_job_id = job_id
                        if job['job_type'] == 'blueteam_scan':
                            result = sentinel.scan(focus=job.get('focus', ''))
                        else:
                            result = sentinel.investigate(
                                finding=job.get('finding', job['instruction']),
                                evidence=job.get('evidence', ''),
                            )
                        job['success'] = result.get('success', True)
                        # Point local agent's log attrs at the result so the shared
                        # store-results block below still works cleanly.
                        agent.execution_log = result.get('execution_log', [])
                        agent.react_trace = result.get('trace', [])

                    elif job.get('chain_id') and job.get('subtask_index') is not None:
                        _original_print(f"[worker] loading chain state for {job['chain_id'][:8]}", flush=True)
                        chain_state = TaskChain.load(job['chain_id'])
                        chain_data = chain_state.data
                        subtask_rec = chain_data['subtasks'][job['subtask_index']]

                        # Pin ARCH.md for phases after Phase 0 (Phase 0 creates it)
                        if job.get('subtask_index', 0) > 0:
                            # Search only shallow paths — never recurse into node_modules etc.
                            import glob as _glob
                            arch_candidates = (
                                _glob.glob(os.path.expanduser('~/*/DOCS/ARCH.md')) +
                                _glob.glob(os.path.expanduser('~/DOCS/ARCH.md'))
                            )
                            if arch_candidates:
                                try:
                                    with open(arch_candidates[0]) as _f:
                                        _arch = _f.read()
                                    agent.pinned_messages = [{
                                        'role': 'user',
                                        'content': f'[PINNED ARCH.md]\n{_arch[:3000]}'
                                    }]
                                    chain_data['arch_summary'] = _arch[:500]
                                except Exception:
                                    pass

                        _original_print(f"[worker] launching orchestrator for phase {job.get('subtask_index')}", flush=True)
                        orchestrator = SubtaskOrchestrator(agent)
                        artifact = orchestrator.orchestrate(subtask_rec, chain_data)
                        _original_print(f"[worker] orchestrator done: {artifact.status}", flush=True)

                        # Store artifact in chain state for future phases
                        chain_state.update_subtask(
                            job['subtask_index'],
                            {'artifact': artifact.to_dict()}
                        )

                        # Build react_result in the shape _advance_chain expects
                        result = {
                            'success': artifact.status in ('completed', 'partial'),
                            'finish_summary': artifact.summary,
                            'trace': [],
                            'iterations_used': sum(
                                r.get('iterations_used', 0)
                                for r in artifact.micro_task_reports
                            ),
                        }
                        job['success'] = result['success']
                    else:
                        # Single job — run the full ReAct loop directly
                        result = agent.run(
                            job['instruction'],
                            incoming_handoff=job.get('incoming_handoff'),
                        )
                        job['success'] = result.get('success', True) if result else True

                    # Store results
                    job['status'] = 'completed'
                    job['execution_log'] = agent.execution_log
                    job['_react_result'] = result
                    job['_react_trace'] = agent.react_trace

                except Exception as e:
                    job['status'] = 'failed'
                    job['error'] = str(e)
                    job['success'] = False

                finally:
                    _print_local.capture_fn = None
                
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = f"Job execution error: {str(e)}"
                job['success'] = False
            
            finally:
                job['completed_at'] = datetime.now().isoformat()
                if job_id in active_jobs:
                    del active_jobs[job_id]

                _react_result = job.get('_react_result') or {}
                debug_logger.job_end(
                    job_id,
                    instruction=job.get('instruction', ''),
                    status=job.get('status', 'unknown'),
                    success=bool(job.get('success')),
                    iterations_used=_react_result.get('iterations_used', 0),
                    summary=_react_result.get('finish_summary', job.get('error', '')),
                    chain_id=job.get('chain_id'),
                )
                webhook_dispatcher.job_completed(
                    job_id,
                    instruction=job.get('instruction', ''),
                    success=bool(job.get('success')),
                    summary=_react_result.get('finish_summary', job.get('error', '')),
                    iterations_used=_react_result.get('iterations_used', 0),
                    chain_id=job.get('chain_id'),
                )
                _write_state()

                # Advance chain if this is a chain sub-task job
                if job.get('chain_id'):
                    try:
                        _advance_chain(job['chain_id'], job.get('subtask_index', 0), job)
                    except Exception as chain_err:
                        print(f"❌ Chain advance error for {job['chain_id']}: {chain_err}")

                job_queue.task_done()
    
    def submit_job(self, job_id: str):
        """Submit a job to the queue"""
        job_queue.put(job_id)
    
    def stop(self):
        """Stop all workers"""
        self.running = False
        for worker in self.workers:
            worker.join(timeout=5)


# Initialize job runner
job_runner = JobRunner()


def _submit_subtask_job(chain_id: str, chain_data: dict, index: int, retry_count: int, incoming_handoff):
    """Create a job dict for a chain sub-task and enqueue it."""
    subtask = chain_data["subtasks"][index]
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "instruction": subtask["instruction"],
        "model": chain_data.get("model", "qwen3-coder:30b"),
        "searxng_url": "http://10.0.0.58:8080",
        "max_iterations": subtask.get("max_iterations", 25),
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "output": "",
        "execution_log": [],
        "success": None,
        # Chain-specific fields
        "chain_id": chain_id,
        "subtask_index": index,
        "retry_count": retry_count,
        "incoming_handoff": incoming_handoff,
    }
    jobs[job_id] = job

    # Update subtask record on disk
    chain = TaskChain.load(chain_id)
    chain.update_subtask(index, {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "retry_count": retry_count,
    })

    job_runner.submit_job(job_id)
    return job_id


def _advance_chain(chain_id: str, subtask_index: int, job: dict):
    """
    Called after a sub-task job completes (success or failure).
    Extracts handoff, runs acceptance criteria, then advances the chain
    to the next sub-task or marks it complete/failed.
    """
    import traceback as _tb

    chain = TaskChain.load(chain_id)
    chain_data = chain.data
    subtask = chain_data["subtasks"][subtask_index]
    retry_policy = chain_data.get("retry_policy", {})
    max_retries = retry_policy.get("max_retries_per_subtask", 1)

    # 1. Build a react_result dict for HandoffExtractor
    react_result = job.get("_react_result")
    if react_result is None:
        react_result = {
            "success": job.get("success", False),
            "finish_summary": job.get("error", "job failed with exception"),
            "trace": job.get("_react_trace", []),
            "iterations_used": 0,
        }

    # 2. Extract handoff using a temporary agent (LLM call)
    try:
        temp_agent = OllamaCommandAgent(model=chain_data.get("model", "qwen3-coder:30b"))
        extractor = HandoffExtractor(temp_agent)
        handoff = extractor.extract(subtask["instruction"], react_result)
    except Exception as e:
        print(f"⚠️  Handoff extraction failed: {e}; using minimal fallback")
        handoff = {
            "schema_version": 1,
            "facts": {},
            "finish_summary": react_result.get("finish_summary", ""),
            "success": react_result.get("success", False),
            "iterations_used": react_result.get("iterations_used", 0),
            "source_instruction": subtask["instruction"],
            "completed_at": datetime.now().isoformat(),
        }
        temp_agent = OllamaCommandAgent(model=chain_data.get("model", "qwen3-coder:30b"))

    # 3. Save handoff to subtask
    chain.update_subtask(subtask_index, {"handoff": handoff})

    # 4. Run acceptance criteria if job succeeded and criteria exist
    job_success = job.get("success", False)
    ac_criteria = subtask.get("acceptance_criteria")

    if job_success and ac_criteria:
        ac_runner = AcceptanceCriteriaRunner()
        ac_result = ac_runner.run(ac_criteria)
        passed = ac_result["passed"]
        chain.update_subtask(subtask_index, {"acceptance_result": ac_result})
    else:
        ac_result = None
        passed = job_success

    # 5. Mark subtask final status
    if passed:
        subtask_status = "passed"
    elif ac_result is not None and not ac_result["passed"]:
        subtask_status = "ac_failed"
    else:
        subtask_status = "failed"

    chain.update_subtask(subtask_index, {
        "status": subtask_status,
        "completed_at": datetime.now().isoformat(),
    })
    debug_logger.subtask_event(chain_id, subtask_status, subtask_index,
                               instruction=subtask.get("instruction", ""),
                               ac_command=(ac_result or {}).get("command", ""),
                               ac_passed=passed)
    webhook_dispatcher.subtask_result(chain_id, subtask_index,
                                      instruction=subtask.get("instruction", ""),
                                      ac_passed=passed,
                                      summary=job.get("_react_result", {}).get("finish_summary", "") if isinstance(job.get("_react_result"), dict) else "")

    # 6. Retry if not passed and retries remain
    retry_count = subtask.get("retry_count", 0)
    if not passed and retry_count < max_retries:
        chain = TaskChain.load(chain_id)
        print(f"🔁 Chain {chain_id[:8]}: retrying subtask {subtask_index} (attempt {retry_count + 1}/{max_retries})")
        _submit_subtask_job(chain_id, chain.data, subtask_index, retry_count + 1, handoff)
        return

    # 7. If not passed and retries exhausted: fail the chain
    if not passed:
        chain = TaskChain.load(chain_id)
        print(f"❌ Chain {chain_id[:8]}: subtask {subtask_index} failed, chain failed")
        chain.update_chain({"status": "failed", "completed_at": datetime.now().isoformat()})
        debug_logger.chain_end(chain_id, chain.data.get("goal", ""), "failed")
        webhook_dispatcher.chain_status_changed(chain_id, chain.data.get("goal", ""),
                                                "failed", len(chain.data.get("subtasks", [])))
        return

    # 8. Find next non-skipped subtask using replanner
    chain = TaskChain.load(chain_id)
    chain_data = chain.data
    subtasks = chain_data["subtasks"]
    replanner = SubtaskReplanner(temp_agent)

    next_index = subtask_index + 1
    while next_index < len(subtasks):
        candidate = subtasks[next_index]
        if candidate["status"] != "pending":
            next_index += 1
            continue

        # Ask replanner if this sub-task should proceed, adjust, or skip
        replan = replanner.replan(candidate, handoff, chain_data["goal"])

        if replan.get("skip"):
            print(f"⏭️  Chain {chain_id[:8]}: skipping subtask {next_index} — {replan.get('reason', '')}")
            chain.update_subtask(next_index, {
                "status": "skipped",
                "replan_applied": True,
                "replan_reason": replan.get("reason", ""),
                "completed_at": datetime.now().isoformat(),
            })
            next_index += 1
            continue

        # Apply adjusted instruction/criteria if replanner changed them
        new_instruction = replan.get("instruction", candidate["instruction"])
        new_ac = replan.get("acceptance_criteria", candidate.get("acceptance_criteria"))
        reason = replan.get("reason", "")
        changed = (new_instruction != candidate["instruction"] or new_ac != candidate.get("acceptance_criteria"))
        if changed:
            print(f"✏️  Chain {chain_id[:8]}: replanned subtask {next_index} — {reason}")
            chain.update_subtask(next_index, {
                "instruction": new_instruction,
                "acceptance_criteria": new_ac,
                "replan_applied": True,
                "replan_reason": reason,
            })
        break

    # 9. If no more subtasks remain: mark chain completed
    if next_index >= len(subtasks):
        chain = TaskChain.load(chain_id)
        print(f"✅ Chain {chain_id[:8]}: all subtasks done — chain completed")
        chain.update_chain({"status": "completed", "completed_at": datetime.now().isoformat()})
        debug_logger.chain_end(chain_id, chain.data.get("goal", ""), "completed")
        webhook_dispatcher.chain_status_changed(chain_id, chain.data.get("goal", ""),
                                                "completed", len(chain.data.get("subtasks", [])))
        return

    # 10. Clean up zombie ports, then submit next subtask
    print(f"🧹 Chain {chain_id[:8]}: cleaning up dev ports before next phase...")
    cleanup_between_phases()
    chain = TaskChain.load(chain_id)
    chain.update_chain({"current_subtask_index": next_index})
    print(f"▶️  Chain {chain_id[:8]}: advancing to subtask {next_index}")
    _submit_subtask_job(chain_id, chain.data, next_index, 0, handoff)


def _resume_running_chains():
    """Re-queue any chains that were in-flight when the server last died."""
    try:
        chain_summaries = TaskChain.list_all()
    except Exception:
        return
    for summary in chain_summaries:
        if summary['status'] != 'running':
            continue
        chain_id = summary['chain_id']
        try:
            chain = TaskChain.load(chain_id)
            data = chain.data
            subtasks = data.get('subtasks', [])
            # Find the first subtask that hasn't finished
            next_index = None
            for i, st in enumerate(subtasks):
                if st.get('status') not in ('passed', 'skipped', 'failed'):
                    next_index = i
                    break
            if next_index is None:
                # All subtasks done — chain just never got marked completed
                chain.update_chain({'status': 'completed', 'completed_at': datetime.now().isoformat()})
                print(f"✅ Chain {chain_id[:8]}: all subtasks done, marking completed on resume")
                continue
            # Reset any subtask stuck in "running" back to pending
            if subtasks[next_index].get('status') == 'running':
                chain.update_subtask(next_index, {'status': 'pending'})
            chain.update_chain({'current_subtask_index': next_index})
            print(f"🔄 Resuming chain {chain_id[:8]} from subtask {next_index}")
            _submit_subtask_job(chain_id, chain.data, next_index, 0, None)
        except Exception as e:
            print(f"⚠️  Could not resume chain {chain_id[:8]}: {e}")


def require_api_key(f):
    """Auth disabled — local-network deployment, no API key required."""
    return f


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ollama-command-agent',
        'version': SERVICE_VERSION,
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(active_jobs),
        'queued_jobs': job_queue.qsize(),
        'features': SERVICE_FEATURES,
        'minions': 'ready',
    })


@app.route('/api/v1/execute', methods=['POST'])
@require_api_key
def execute_command():
    """
    Execute a command via the Ollama agent
    
    Request body:
    {
        "instruction": "Run a vulnerability scan",
        "model": "qwen3-coder:30b",  // optional
        "searxng_url": "http://...",  // optional
        "async": true  // optional, default true
    }
    
    Returns:
    {
        "job_id": "uuid",
        "status": "queued",
        "created_at": "timestamp"
    }
    """
    data = request.get_json()
    
    if not data or 'instruction' not in data:
        return jsonify({'error': 'Missing instruction'}), 400
    
    # Create job
    job_id = str(uuid.uuid4())
    job = {
        'job_id': job_id,
        'instruction': data['instruction'],
        'model': data.get('model', 'qwen3-coder:30b'),
        'searxng_url': data.get('searxng_url', 'http://10.0.0.58:8080'),
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'output': '',
        'execution_log': [],
        'success': None
    }
    
    jobs[job_id] = job
    
    # Submit to queue
    job_runner.submit_job(job_id)
    
    # Return immediately (async by default)
    if data.get('async', True):
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'created_at': job['created_at']
        }), 202
    
    # Wait for completion (synchronous)
    else:
        timeout = data.get('timeout', 300)  # 5 min default
        start_time = time.time()
        
        while job['status'] in ['queued', 'running']:
            if time.time() - start_time > timeout:
                return jsonify({
                    'job_id': job_id,
                    'status': 'timeout',
                    'error': 'Job exceeded timeout'
                }), 408
            
            time.sleep(1)
        
        return jsonify(job)


@app.route('/api/v1/jobs/<job_id>', methods=['GET'])
@require_api_key
def get_job_status(job_id: str):
    """
    Get job status and results
    
    Returns:
    {
        "job_id": "uuid",
        "status": "completed|running|queued|failed",
        "instruction": "original instruction",
        "output": "captured output",
        "execution_log": [...],
        "success": true|false,
        "created_at": "timestamp",
        "started_at": "timestamp",
        "completed_at": "timestamp",
        "error": "error message if failed"
    }
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    # Exclude internal _-prefixed fields (e.g. _react_result, _react_trace)
    public_job = {k: v for k, v in job.items() if not k.startswith('_')}
    return jsonify(public_job)


@app.route('/api/v1/jobs/<job_id>/stream', methods=['GET'])
@require_api_key
def stream_job_output(job_id: str):
    """
    Stream job output in real-time using Server-Sent Events (SSE)
    """
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    def generate():
        job = jobs[job_id]
        last_output_len = 0
        
        while job['status'] in ['queued', 'running']:
            current_output = job.get('output', '')
            
            # Send new output
            if len(current_output) > last_output_len:
                new_output = current_output[last_output_len:]
                yield f"data: {json.dumps({'type': 'output', 'content': new_output})}\n\n"
                last_output_len = len(current_output)
            
            # Send status updates
            yield f"data: {json.dumps({'type': 'status', 'status': job['status']})}\n\n"
            
            time.sleep(0.5)
        
        # Send final status
        yield f"data: {json.dumps({'type': 'complete', 'status': job['status'], 'success': job.get('success')})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/v1/jobs', methods=['GET'])
@require_api_key
def list_jobs():
    """
    List all jobs
    
    Query params:
    - status: filter by status (queued|running|completed|failed)
    - limit: max jobs to return (default 50)
    """
    status_filter = request.args.get('status')
    limit = int(request.args.get('limit', 50))
    
    filtered_jobs = []
    
    for job in jobs.values():
        if status_filter and job['status'] != status_filter:
            continue
        
        filtered_jobs.append({
            'job_id': job['job_id'],
            'instruction': job['instruction'][:100],
            'status': job['status'],
            'created_at': job['created_at'],
            'success': job.get('success')
        })
    
    # Sort by creation time (newest first)
    filtered_jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({
        'jobs': filtered_jobs[:limit],
        'total': len(filtered_jobs)
    })


@app.route('/api/v1/jobs/<job_id>', methods=['DELETE'])
@require_api_key
def cancel_job(job_id: str):
    """Cancel a running or queued job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] in ['completed', 'failed']:
        return jsonify({'error': 'Cannot cancel completed job'}), 400
    
    job['status'] = 'cancelled'
    job['completed_at'] = datetime.now().isoformat()
    
    return jsonify({'message': 'Job cancelled', 'job_id': job_id})


@app.route('/api/v1/config', methods=['GET'])
@require_api_key
def get_config():
    """Get service configuration"""
    return jsonify({
        'max_concurrent_jobs': MAX_CONCURRENT_JOBS,
        'job_timeout': JOB_TIMEOUT,
        'default_model': 'qwen3-coder:30b',
        'searxng_url': 'http://10.0.0.58:8080'
    })


@app.route('/api/v1/watchdog', methods=['POST'])
@require_api_key
def watchdog():
    """
    Receive a systemd failure event and dispatch a recovery job.

    Request body:
    {
        "unit": "nginx.service",
        "event": "failed",
        "max_iterations": 15   // optional, default 15
    }

    Returns same job_id response as /api/v1/execute.
    """
    data = request.get_json()

    if not data or 'unit' not in data:
        return jsonify({'error': 'Missing unit name'}), 400

    unit = data['unit']
    event = data.get('event', 'failed')
    max_iterations = int(data.get('max_iterations', 15))

    instruction = (
        f"WATCHDOG: systemd unit '{unit}' has entered state '{event}'. "
        f"Investigate using journalctl, determine root cause, and attempt recovery."
    )

    job_id = str(uuid.uuid4())
    job = {
        'job_id': job_id,
        'instruction': instruction,
        'model': data.get('model', 'qwen3-coder:30b'),
        'searxng_url': data.get('searxng_url', 'http://10.0.0.58:8080'),
        'max_iterations': max_iterations,
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'output': '',
        'execution_log': [],
        'success': None,
        'watchdog': {'unit': unit, 'event': event},
    }

    jobs[job_id] = job
    job_runner.submit_job(job_id)

    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'created_at': job['created_at'],
        'unit': unit,
        'event': event,
    }), 202


@app.route('/api/v1/chains', methods=['POST'])
@require_api_key
def create_chain():
    """
    Decompose a goal and start a task chain.

    Request body:
    {
        "goal": "Build a FastAPI app with auth and a database",
        "total_budget": 100,     // optional, default 100
        "model": "qwen3-coder:30b",  // optional
        "retry_policy": {"max_retries_per_subtask": 1}  // optional
    }

    Returns:
    {
        "chain_id": "uuid",
        "status": "running",
        "subtask_count": 5,
        "created_at": "timestamp"
    }
    """
    data = request.get_json()
    if not data or 'goal' not in data:
        return jsonify({'error': 'Missing goal'}), 400

    goal = data['goal']
    total_budget = int(data.get('total_budget', 100))
    model = data.get('model', 'qwen3-coder:30b')
    retry_policy = data.get('retry_policy', {'max_retries_per_subtask': 1})

    # Decompose the goal using a temporary agent
    try:
        temp_agent = OllamaCommandAgent(model=model)
        decomposer = TaskDecomposer(temp_agent)
        subtasks = decomposer.decompose(goal, total_budget)
    except Exception as e:
        return jsonify({'error': f'Decomposition failed: {str(e)}'}), 500

    # Create chain on disk
    chain = TaskChain.create(
        goal=goal,
        subtasks=subtasks,
        total_budget=total_budget,
        model=model,
        retry_policy=retry_policy,
    )

    # Submit first sub-task
    debug_logger.chain_start(chain.chain_id, goal, len(subtasks))
    _submit_subtask_job(chain.chain_id, chain.data, 0, 0, None)

    return jsonify({
        'chain_id': chain.chain_id,
        'status': 'running',
        'subtask_count': len(subtasks),
        'created_at': chain.data['created_at'],
        'subtasks': [
            {
                'index': st['index'],
                'instruction': st['instruction'][:120],
                'max_iterations': st['max_iterations'],
            }
            for st in subtasks
        ],
    }), 202


@app.route('/api/v1/chains/<chain_id>', methods=['GET'])
@require_api_key
def get_chain(chain_id: str):
    """Get full chain state including sub-task summaries."""
    try:
        chain = TaskChain.load(chain_id)
    except FileNotFoundError:
        return jsonify({'error': 'Chain not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    d = chain.data

    # Attach brief job output summary to each subtask
    subtask_summaries = []
    for st in d.get('subtasks', []):
        summary = dict(st)
        job_id = st.get('job_id')
        if job_id and job_id in jobs:
            job = jobs[job_id]
            summary['job_status'] = job.get('status')
            summary['job_output_tail'] = (job.get('output', '') or '')[-500:]
        subtask_summaries.append(summary)

    return jsonify({**d, 'subtasks': subtask_summaries})


@app.route('/api/v1/chains', methods=['GET'])
@require_api_key
def list_chains():
    """List all chains (reads chain directory)."""
    try:
        chains = TaskChain.list_all()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    limit = int(request.args.get('limit', 50))
    status_filter = request.args.get('status')
    if status_filter:
        chains = [c for c in chains if c['status'] == status_filter]

    return jsonify({'chains': chains[:limit], 'total': len(chains)})


@app.route('/api/v1/chains/<chain_id>', methods=['DELETE'])
@require_api_key
def cancel_chain(chain_id: str):
    """Cancel a running chain."""
    try:
        chain = TaskChain.load(chain_id)
    except FileNotFoundError:
        return jsonify({'error': 'Chain not found'}), 404

    if chain.data['status'] in ('completed', 'failed', 'cancelled'):
        return jsonify({'error': 'Cannot cancel chain in terminal state'}), 400

    chain.update_chain({'status': 'cancelled', 'completed_at': datetime.now().isoformat()})
    return jsonify({'message': 'Chain cancelled', 'chain_id': chain_id})


@app.route('/api/v1/chains/<chain_id>/restart', methods=['POST'])
@require_api_key
def restart_chain(chain_id: str):
    """Restart a failed/stuck chain from its first non-passed subtask."""
    try:
        chain = TaskChain.load(chain_id)
    except FileNotFoundError:
        return jsonify({'error': 'Chain not found'}), 404

    data = chain.data
    subtasks = data.get('subtasks', [])

    # Find the first subtask that isn't done
    next_index = None
    for i, st in enumerate(subtasks):
        if st.get('status') not in ('passed', 'skipped'):
            next_index = i
            break

    if next_index is None:
        chain.update_chain({'status': 'completed', 'completed_at': datetime.now().isoformat()})
        return jsonify({'message': 'All subtasks already passed — chain marked completed', 'chain_id': chain_id})

    # Reset the target subtask and mark chain running
    chain.update_subtask(next_index, {'status': 'pending'})
    chain.update_chain({'status': 'running', 'current_subtask_index': next_index,
                        'completed_at': None})
    print(f"🔁 Restarting chain {chain_id[:8]} from subtask {next_index}")
    _submit_subtask_job(chain_id, chain.data, next_index, 0, None)
    return jsonify({'message': f'Restarting from subtask {next_index}', 'chain_id': chain_id,
                    'subtask_index': next_index})


@app.route('/api/v1/chains/<chain_id>/skip/<int:subtask_index>', methods=['POST'])
@require_api_key
def skip_subtask(chain_id: str, subtask_index: int):
    """Mark a subtask as manually passed and advance the chain to the next one."""
    try:
        chain = TaskChain.load(chain_id)
    except FileNotFoundError:
        return jsonify({'error': 'Chain not found'}), 404

    data = chain.data
    subtasks = data.get('subtasks', [])
    if subtask_index >= len(subtasks):
        return jsonify({'error': f'Subtask index {subtask_index} out of range'}), 400

    note = request.json.get('note', 'Manually marked passed') if request.json else 'Manually marked passed'
    chain.update_subtask(subtask_index, {'status': 'passed', 'manually_skipped': True, 'note': note})
    chain.update_chain({'status': 'running', 'current_subtask_index': subtask_index, 'completed_at': None})
    print(f"⏭️  Chain {chain_id[:8]}: subtask {subtask_index} manually passed — advancing")

    # Find the next pending subtask
    next_index = None
    for i in range(subtask_index + 1, len(subtasks)):
        if subtasks[i].get('status') not in ('passed', 'skipped'):
            next_index = i
            break

    if next_index is None:
        chain.update_chain({'status': 'completed', 'completed_at': datetime.now().isoformat()})
        return jsonify({'message': 'All subtasks done — chain completed', 'chain_id': chain_id})

    chain.update_chain({'current_subtask_index': next_index})
    _submit_subtask_job(chain_id, chain.data, next_index, 0, None)
    return jsonify({'message': f'Subtask {subtask_index} marked passed, advancing to {next_index}',
                    'chain_id': chain_id, 'next_subtask_index': next_index})


@app.route('/api/v1/events', methods=['GET'])
@require_api_key
def stream_events():
    """SSE stream of all agent events (job start/end, chain transitions, react iterations).

    Connect with:  curl -N -H 'X-API-Key: ...' http://host:5000/api/v1/events
    Each line:     data: <json>\\n\\n
    """
    import queue as _qmod

    client_q: _qmod.Queue = _qmod.Queue(maxsize=500)
    with _sse_lock:
        _sse_clients.add(client_q)

    def generate():
        yield "retry: 3000\n\n"  # tell client to reconnect after 3 s on drop
        try:
            while True:
                try:
                    event = client_q.get(timeout=25)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except _qmod.Empty:
                    yield ": heartbeat\n\n"  # keep connection alive
        finally:
            with _sse_lock:
                _sse_clients.discard(client_q)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/v1/state', methods=['GET'])
@require_api_key
def get_state():
    """Return current agent state as JSON — same data written to agent_state.json."""
    _write_state()
    try:
        with open(_STATE_PATH) as f:
            return Response(f.read(), mimetype='application/json')
    except FileNotFoundError:
        return jsonify({'error': 'State file not yet written'}), 503


# ── SENTINEL blue team endpoints ──────────────────────────────────────────────

@app.route('/api/v1/blueteam/scan', methods=['POST'])
@require_api_key
def blueteam_scan():
    """Start a full SENTINEL security scan (runs as a background job).

    Optional body: {"focus": "SSH"}
    Returns: {"job_id": "...", "status": "queued"}
    """
    if not _BLUETEAM_AVAILABLE:
        return jsonify({'error': 'blueteam_agent module not available'}), 503

    data = request.get_json() or {}
    focus = data.get('focus', '')

    job_id = str(uuid.uuid4())
    job = {
        'job_id': job_id,
        'job_type': 'blueteam_scan',
        'instruction': f'SENTINEL security scan{f" (focus: {focus})" if focus else ""}',
        'focus': focus,
        'model': 'qwen3-coder:30b',
        'searxng_url': 'http://10.0.0.58:8080',
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'output': '',
        'execution_log': [],
        'success': None,
    }
    jobs[job_id] = job
    job_runner.submit_job(job_id)
    return jsonify({'job_id': job_id, 'status': 'queued',
                    'created_at': job['created_at']}), 202


@app.route('/api/v1/blueteam/investigate', methods=['POST'])
@require_api_key
def blueteam_investigate():
    """Start a targeted SENTINEL investigation (runs as a background job).

    Body: {"finding": "suspicious SSH from 1.2.3.4", "evidence": "..."}
    Returns: {"job_id": "...", "status": "queued"}
    """
    if not _BLUETEAM_AVAILABLE:
        return jsonify({'error': 'blueteam_agent module not available'}), 503

    data = request.get_json() or {}
    finding = data.get('finding', '')
    if not finding:
        return jsonify({'error': 'Missing finding'}), 400

    job_id = str(uuid.uuid4())
    job = {
        'job_id': job_id,
        'job_type': 'blueteam_investigate',
        'instruction': f'SENTINEL investigation: {finding[:100]}',
        'finding': finding,
        'evidence': data.get('evidence', ''),
        'model': 'qwen3-coder:30b',
        'searxng_url': 'http://10.0.0.58:8080',
        'status': 'queued',
        'created_at': datetime.now().isoformat(),
        'output': '',
        'execution_log': [],
        'success': None,
    }
    jobs[job_id] = job
    job_runner.submit_job(job_id)
    return jsonify({'job_id': job_id, 'status': 'queued',
                    'created_at': job['created_at']}), 202


@app.route('/api/v1/blueteam/watch/start', methods=['POST'])
@require_api_key
def blueteam_watch_start():
    """Start the SENTINEL background anomaly watcher.

    Optional body: {"quick_interval": 300, "deep_interval": 3600}
    """
    if not _BLUETEAM_AVAILABLE:
        return jsonify({'error': 'blueteam_agent module not available'}), 503

    data = request.get_json() or {}
    quick = int(data.get('quick_interval', data.get('interval', 300)))
    deep = int(data.get('deep_interval', 3600))
    sentinel = _get_sentinel()
    sentinel.watch(quick_interval=quick, deep_interval=deep)
    return jsonify({
        'message': f'SENTINEL watcher started (quick={quick}s, deep={deep}s)',
        'watching': True, 'quick_interval': quick, 'deep_interval': deep,
    })


@app.route('/api/v1/blueteam/watch/stop', methods=['POST'])
@require_api_key
def blueteam_watch_stop():
    """Stop the SENTINEL background watcher."""
    if not _BLUETEAM_AVAILABLE:
        return jsonify({'error': 'blueteam_agent module not available'}), 503

    sentinel = _get_sentinel()
    sentinel.stop_watch()
    return jsonify({'message': 'SENTINEL watcher stopping', 'watching': False})


@app.route('/api/v1/blueteam/alerts', methods=['GET'])
@require_api_key
def blueteam_alerts():
    """Return recent SENTINEL security alerts.

    Query param: ?n=50
    """
    if not _BLUETEAM_AVAILABLE:
        return jsonify({'error': 'blueteam_agent module not available'}), 503

    n = int(request.args.get('n', 50))
    alerts = _bt_alerts(n)
    return jsonify({'alerts': alerts, 'count': len(alerts)})


@app.route('/api/v1/blueteam/status', methods=['GET'])
@require_api_key
def blueteam_status():
    """Return SENTINEL watcher status and last scan summary."""
    if not _BLUETEAM_AVAILABLE:
        return jsonify({'error': 'blueteam_agent module not available'}), 503

    sentinel = _get_sentinel()
    return jsonify(sentinel.status())


@app.route('/api/v1/blueteam/report', methods=['GET'])
def blueteam_report():
    """Return the current SENTINEL daily report as markdown (or JSON).

    Query params:
        format=md   (default) — returns raw markdown with Content-Type text/markdown
        format=json           — returns {"report": "...", "archived": [...]}
    """
    from pathlib import Path
    report_path = Path("~/.agent_bin/sentinel_report.md").expanduser()
    archive_dir = Path("~/.agent_bin/sentinel_archive").expanduser()

    if not report_path.exists():
        return jsonify({'error': 'No report available yet — run a blueteam scan first'}), 404

    content = report_path.read_text(encoding='utf-8')
    fmt = request.args.get('format', 'md')

    if fmt == 'json':
        archived = sorted(
            [p.name for p in archive_dir.glob("sentinel_report_*.md")]
        ) if archive_dir.exists() else []
        return jsonify({
            'report': content,
            'report_path': str(report_path),
            'archived_reports': archived,
        })

    return content, 200, {'Content-Type': 'text/markdown; charset=utf-8'}


if __name__ == '__main__':
    print("=" * 70)
    print(f"🚀 Ollama Command Agent Service  [{SERVICE_VERSION}]")
    print("=" * 70)
    print()
    print("  ✅ v3 STUDIO MODEL HIERARCHY IS RUNNING")
    print("  🤖 Minions are ready to go")
    print()
    for feat in SERVICE_FEATURES:
        print(f"  ▸ {feat}")
    print()
    print("=" * 70)
    print(f"Auth: DISABLED (local-network mode)")
    print(f"Port: 5000  |  Max Concurrent Jobs: {MAX_CONCURRENT_JOBS}")
    print(f"Endpoints:")
    print(f"  POST   /api/v1/execute           - Execute command")
    print(f"  GET    /api/v1/jobs/<id>         - Get job status")
    print(f"  GET    /api/v1/jobs/<id>/stream  - Stream output")
    print(f"  GET    /api/v1/jobs              - List jobs")
    print(f"  DELETE /api/v1/jobs/<id>         - Cancel job")
    print(f"  POST   /api/v1/chains            - Decompose goal & start chain")
    print(f"  GET    /api/v1/chains/<id>       - Get chain state")
    print(f"  GET    /api/v1/chains            - List all chains")
    print(f"  DELETE /api/v1/chains/<id>       - Cancel chain")
    print(f"  POST   /api/v1/chains/<id>/restart  - Restart failed chain")
    print(f"  POST   /api/v1/chains/<id>/skip/<n> - Mark subtask passed")
    print(f"  GET    /api/v1/events            - SSE event stream")
    print(f"  GET    /api/v1/state             - Agent state snapshot")
    print(f"  GET    /health                   - Health check")
    print(f"  POST   /api/v1/blueteam/scan     - SENTINEL security scan (job)")
    print(f"  POST   /api/v1/blueteam/investigate - SENTINEL investigation (job)")
    print(f"  POST   /api/v1/blueteam/watch/start - Start anomaly watcher")
    print(f"  POST   /api/v1/blueteam/watch/stop  - Stop anomaly watcher")
    print(f"  GET    /api/v1/blueteam/report   - Current SENTINEL daily report (.md)")
    print(f"  GET    /api/v1/blueteam/alerts   - Recent security alerts")
    print(f"  GET    /api/v1/blueteam/status   - Watcher status + last scan")
    print("=" * 70)
    print("\nChecking for interrupted chains...")
    _resume_running_chains()

    # Start inbox watcher (polls ./agent_inbox/ for files from other LLMs)
    threading.Thread(target=_inbox_watcher, daemon=True, name="InboxWatcher").start()
    print(f"📥 Inbox watcher started  →  {os.path.abspath(_INBOX_DIR)}")
    print(f"📝 Debug logs            →  {os.path.abspath('./logs/')}")
    print(f"📊 State file            →  {os.path.abspath(_STATE_PATH)}")

    # SENTINEL auto-watch disabled — triggered externally (cron at 3 AM)
    # To re-enable: POST /api/v1/blueteam/watch/start or uncomment below
    # if _BLUETEAM_AVAILABLE:
    #     _get_sentinel().watch(quick_interval=300, deep_interval=3600)
    #     print(f"👁️  SENTINEL auto-started  →  quick=5min  deep=1hr")

    print("\nStarting server...")

    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        job_runner.stop()
