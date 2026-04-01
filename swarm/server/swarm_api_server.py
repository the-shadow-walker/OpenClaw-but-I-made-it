import sys as _sys, os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
try:
    import _paths  # noqa: F401
except ImportError:
    for _d in [_os.path.join(_ROOT, d) for d in ('core', 'compute', 'server', 'engineer')]:
        if _os.path.isdir(_d) and _d not in _sys.path:
            _sys.path.insert(0, _d)
"""
Swarm 3.0 REST API Wrapper
==========================
Provides REST endpoints plus SSE streaming progress.

Usage:
    python3 swarm_api_server.py --port 5002

Auth:
    Set SWARM_API_KEY env var to enable Bearer token auth on write endpoints.
"""

import os
import sys
import json
import asyncio
import argparse
import uuid
import threading
import functools
import contextvars
import concurrent.futures
import queue as _Q
import re
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, Response, stream_with_context, send_file
from flask_cors import CORS
import logging

try:
    from orchestrator_v3 import OrchestratorV3
    OrchestratorV2_1 = OrchestratorV3   # keep legacy name for health checks
except ImportError:
    print("Warning: Could not import OrchestratorV3, trying V2_1 fallback")
    OrchestratorV3 = None
    try:
        from orchestrator_v2_1 import OrchestratorV2_1
    except ImportError:
        print("Warning: Could not import OrchestratorV2_1 either")
        OrchestratorV2_1 = None

try:
    from project_session import session_manager, ProjectSessionManager
    _HAS_PROJECT_SESSION = True
except ImportError:
    _HAS_PROJECT_SESSION = False
    session_manager = None


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    DEBUG      = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    HOST       = '0.0.0.0'
    PORT       = int(os.getenv('SWARM_API_PORT', 5002))
    SEARXNG_URL = os.getenv('SEARXNG_URL', None)
    RESULTS_DIR = Path('./swarm_results')
    RESULTS_DIR.mkdir(exist_ok=True)
    MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT_JOBS', 3))
    API_KEY    = os.getenv('SWARM_API_KEY', None)


# =============================================================================
# PROGRESS ROUTER  (thread-local stdout -> per-job queue)
# =============================================================================

_progress_queues: Dict[str, _Q.Queue] = {}
_pq_lock = threading.Lock()

# ContextVar so job_id propagates into run_in_executor threads automatically
# (Python 3.7+ copies context into every executor thread via run_in_executor)
_job_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar('swarm_job_id', default='')


class _TaggedExecutor(concurrent.futures.ThreadPoolExecutor):
    """
    Thread pool executor that stamps every spawned thread with _swarm_job_id
    BEFORE the submitted function runs.  This guarantees _ProgressRouter routes
    prints from run_in_executor threads (e.g. [LLMTOK] from react_solver) into
    the correct per-job queue and disk log — no ContextVar propagation needed.
    """
    def __init__(self, job_id: str, *args, **kwargs):
        self._job_id = job_id
        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        job_id = self._job_id
        @functools.wraps(fn)
        def _tagged(*a, **kw):
            threading.current_thread()._swarm_job_id = job_id
            return fn(*a, **kw)
        return super().submit(_tagged, *args, **kwargs)

def _register_pq(job_id: str, q: _Q.Queue):
    with _pq_lock:
        _progress_queues[job_id] = q

def _unregister_pq(job_id: str):
    with _pq_lock:
        _progress_queues.pop(job_id, None)


# Per-job disk log files (full transcript, uncapped)
import io as _io
_log_files: Dict[str, _io.TextIOWrapper] = {}
_lf_lock = threading.Lock()

def _open_log(job_id: str) -> str:
    """Open a per-job log file; returns the path."""
    path = Config.RESULTS_DIR / f"{job_id}.log"
    with _lf_lock:
        _log_files[job_id] = open(path, 'w', buffering=1)  # line-buffered
    return str(path)

def _close_log(job_id: str):
    with _lf_lock:
        f = _log_files.pop(job_id, None)
    if f:
        try:
            f.close()
        except Exception:
            pass


class _ProgressRouter:
    """
    Replaces sys.stdout once at startup.
    Worker threads set  threading.current_thread()._swarm_job_id = job_id
    to route their stdout lines into the corresponding job queue.
    """
    def __init__(self, real_out):
        self._real  = real_out
        self._local = threading.local()

    def _job(self):
        # Primary: thread attribute set by _worker on its own thread
        jid = getattr(threading.current_thread(), '_swarm_job_id', None)
        if jid:
            return jid
        # Fallback: ContextVar propagated into run_in_executor threads
        return _job_id_ctx.get() or None

    def write(self, text: str):
        self._real.write(text)
        self._real.flush()
        job_id = self._job()
        if not job_id:
            return
        buf = getattr(self._local, 'buf', '')
        buf += text
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.rstrip('\r')
            if line:
                with _pq_lock:
                    q = _progress_queues.get(job_id)
                if q is not None:
                    try:
                        q.put_nowait(line)
                    except _Q.Full:
                        pass
                # Tee every line to the per-job disk log (uncapped, includes [LLMTOK])
                with _lf_lock:
                    f = _log_files.get(job_id)
                if f:
                    try:
                        f.write(line + '\n')
                    except Exception:
                        pass
        self._local.buf = buf

    def flush(self):
        self._real.flush()

    def fileno(self):
        return self._real.fileno()

    def isatty(self):
        return False


# Install router once at import time
_real_stdout = sys.stdout
sys.stdout = _ProgressRouter(_real_stdout)


# =============================================================================
# AUTH
# =============================================================================

def require_api_key(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if Config.API_KEY is None:
            return f(*args, **kwargs)
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'Missing Authorization header'}), 401
        if auth[len('Bearer '):] != Config.API_KEY:
            return jsonify({'error': 'Invalid API key'}), 403
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# JOB MANAGER
# =============================================================================

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def create_job(self, question: str, callback_url: str = None) -> str:
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self.jobs[job_id] = {
                'question':    question,
                'status':      'pending',
                'answer':      None,
                'progress':    '',
                'progress_log': [],
                'log_path':    None,   # set by _worker when log file opens
                'created_at':  datetime.now().isoformat(),
                'completed_at': None,
                'error':       None,
                'callback_url': callback_url,
                'elapsed':     None,
            }
        return job_id

    def update_job(self, job_id: str, status: str, **kwargs):
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                self.jobs[job_id].update(kwargs)
                if status in ('completed', 'failed'):
                    self.jobs[job_id]['completed_at'] = datetime.now().isoformat()

    def append_log(self, job_id: str, line: str):
        with self._lock:
            if job_id in self.jobs:
                # Skip raw token lines from in-memory log — they go to disk only
                if not line.startswith('[LLMTOK]'):
                    log = self.jobs[job_id]['progress_log']
                    log.append(line)
                    if len(log) > 2000:
                        self.jobs[job_id]['progress_log'] = log[-1000:]
                # Update last progress line (strip emoji/debug noise for status field)
                clean = re.sub(r'[^\x20-\x7e]', '', line).strip()
                if clean and not clean.startswith('[LLMTOK]'):
                    self.jobs[job_id]['progress'] = clean[:120]

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            return self.jobs.get(job_id)

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        from datetime import datetime as dt
        with self._lock:
            now = dt.now()
            to_delete = []
            for job_id, job in self.jobs.items():
                if job['status'] in ('completed', 'failed') and job['completed_at']:
                    completed = dt.fromisoformat(job['completed_at'])
                    if (now - completed).total_seconds() > max_age_hours * 3600:
                        to_delete.append(job_id)
            for job_id in to_delete:
                del self.jobs[job_id]

    def count_active(self) -> int:
        with self._lock:
            return sum(1 for j in self.jobs.values()
                       if j['status'] in ('pending', 'processing'))


# =============================================================================
# ORCHESTRATOR FACTORY
# =============================================================================

def _make_orchestrator(date_filter: str = None):
    cls = OrchestratorV3 if OrchestratorV3 is not None else OrchestratorV2_1
    if cls is None:
        return None
    try:
        kwargs = dict(
            max_search_concurrent=3,
            enable_verification=True,
            debug=Config.DEBUG,
            searxng_url=Config.SEARXNG_URL,
        )
        if date_filter:
            kwargs['date_filter'] = date_filter
        return cls(**kwargs)
    except Exception as e:
        logging.error(f"Failed to create orchestrator: {e}")
        return None


async def _run_question(question: str, date_filter: str = None) -> str:
    orch = _make_orchestrator(date_filter)
    if orch is None:
        return "Error: Orchestrator not available"
    try:
        return await orch.process_question(question)
    except Exception as e:
        logging.error(f"Processing error: {e}")
        import traceback; traceback.print_exc()
        return f"Error processing question: {e}"


# =============================================================================
# WEBHOOK
# =============================================================================

def _fire_webhook(callback_url: str, payload: dict):
    try:
        import requests as req_lib
        req_lib.post(callback_url, json=payload, timeout=10)
        logging.info(f"Webhook delivered to {callback_url}")
    except Exception as e:
        logging.warning(f"Webhook delivery failed ({callback_url}): {e}")


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

job_manager    = JobManager()
running_jobs:  Dict[str, threading.Thread] = {}
_rj_lock       = threading.Lock()


# =============================================================================
# SHARED WORKER LOGIC
# =============================================================================

def _start_worker(job_id: str, question: str, since: str = None,
                  callback_url: str = None, progress_q: _Q.Queue = None):
    """Spin up a background thread for job_id.  Optionally routes stdout to progress_q."""

    def _worker():
        t0 = time.time()
        threading.current_thread()._swarm_job_id = job_id
        _job_id_ctx.set(job_id)   # belt-and-suspenders: ContextVar fallback
        log_path = _open_log(job_id)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Install tagged executor so every run_in_executor thread inherits job_id
        loop.set_default_executor(_TaggedExecutor(job_id, max_workers=8))
        try:
            job_manager.update_job(job_id, 'processing', progress='Initializing...',
                                   log_path=log_path)
            answer = loop.run_until_complete(_run_question(question, date_filter=since))
            elapsed = round(time.time() - t0, 1)
            job_manager.update_job(job_id, 'completed', answer=answer, elapsed=elapsed)
            # Persist final answer to disk immediately — survives service restarts
            with _lf_lock:
                lf = _log_files.get(job_id)
                if lf and answer:
                    lf.write(f"\n{'='*70}\nFINAL ANSWER\n{'='*70}\n{answer}\n{'='*70}\n")
                    lf.flush()
            logging.info(f"Job {job_id} completed in {elapsed}s")
            if callback_url:
                _fire_webhook(callback_url, {
                    'job_id': job_id, 'status': 'completed',
                    'answer': answer, 'elapsed': elapsed,
                    'timestamp': datetime.now().isoformat(),
                })
        except Exception as e:
            logging.error(f"Job {job_id} error: {e}")
            job_manager.update_job(job_id, 'failed', error=str(e))
            if callback_url:
                _fire_webhook(callback_url, {
                    'job_id': job_id, 'status': 'failed',
                    'error': str(e), 'timestamp': datetime.now().isoformat(),
                })
        finally:
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            loop.close()
            _close_log(job_id)
            if progress_q is not None:
                _unregister_pq(job_id)
                progress_q.put(None)  # sentinel
            with _rj_lock:
                running_jobs.pop(job_id, None)

    thread = threading.Thread(target=_worker, daemon=True, name=f'swarm-{job_id}')
    with _rj_lock:
        running_jobs[job_id] = thread
    thread.start()
    return thread


# =============================================================================
# ROUTES -- Health / Status
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':                'healthy',
        'timestamp':             datetime.now().isoformat(),
        'orchestrator_available': OrchestratorV2_1 is not None,
        'auth_enabled':          Config.API_KEY is not None,
    }), 200


@app.route('/status', methods=['GET'])
def status():
    try:
        jobs = job_manager.jobs
        pending    = sum(1 for j in jobs.values() if j['status'] == 'pending')
        processing = sum(1 for j in jobs.values() if j['status'] == 'processing')
        completed  = sum(1 for j in jobs.values() if j['status'] == 'completed')
        failed     = sum(1 for j in jobs.values() if j['status'] == 'failed')
        with _rj_lock:
            n_running = len(running_jobs)
        return jsonify({
            'server': 'healthy',
            'orchestrator': OrchestratorV2_1 is not None,
            'jobs': {'pending': pending, 'processing': processing,
                     'completed': completed, 'failed': failed, 'running': n_running},
            'config': {'max_concurrent': Config.MAX_CONCURRENT,
                       'searxng': bool(Config.SEARXNG_URL),
                       'auth_enabled': Config.API_KEY is not None},
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs', methods=['GET'])
def list_jobs():
    try:
        jobs = [{'job_id': jid, **jdata} for jid, jdata in job_manager.jobs.items()]
        return jsonify({'total': len(jobs), 'jobs': jobs}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id: str):
    try:
        job = job_manager.get_job(job_id)
        if job is None:
            # Fall back to log file — survives service restarts
            log_path = Config.RESULTS_DIR / f"{job_id}.log"
            if log_path.exists():
                text = log_path.read_text(errors='replace')
                # Extract FINAL ANSWER block if present
                answer = None
                if 'FINAL ANSWER' in text:
                    parts = text.split('=' * 70)
                    for i, p in enumerate(parts):
                        if 'FINAL ANSWER' in p and i + 1 < len(parts):
                            answer = parts[i + 1].strip()
                            break
                return jsonify({
                    'job_id':   job_id,
                    'status':   'completed' if answer else 'unknown',
                    'answer':   answer,
                    'source':   'log_file',
                    'log_path': str(log_path),
                }), 200
            return jsonify({'error': 'Job not found'}), 404
        return jsonify({
            'job_id':       job_id,
            'question':     job['question'],
            'status':       job['status'],
            'answer':       job['answer'],
            'progress':     job['progress'],
            'progress_log': job.get('progress_log', [])[-20:],
            'error':        job['error'],
            'created_at':   job['created_at'],
            'completed_at': job['completed_at'],
            'elapsed':      job.get('elapsed'),
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES -- Query
# =============================================================================

@app.route('/logs/<job_id>', methods=['GET'])
def get_logs(job_id: str):
    """
    Return the full disk log for a job as plain text.

    Query params:
      ?tail=N   — return only the last N lines
      ?grep=pat — filter lines matching pat (case-insensitive substring)
    """
    job = job_manager.get_job(job_id)
    if job is None:
        return jsonify({'error': 'Job not found'}), 404
    log_path = job.get('log_path')
    if not log_path or not Path(log_path).exists():
        # Fall back to in-memory log if file not ready yet
        lines = job.get('progress_log', [])
        text = '\n'.join(lines)
        if not text:
            return jsonify({'error': 'No log available yet'}), 404
        return Response(text, mimetype='text/plain')
    try:
        tail   = request.args.get('tail',  type=int)
        grep   = request.args.get('grep',  default='', type=str).lower()
        text   = Path(log_path).read_text(errors='replace')
        lines  = text.splitlines()
        if grep:
            lines = [l for l in lines if grep in l.lower()]
        if tail:
            lines = lines[-tail:]
        return Response('\n'.join(lines), mimetype='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/query', methods=['POST'])
@require_api_key
def query_sync():
    """Synchronous query -- blocks until answer ready."""
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        since = data.get('since', None)

        if job_manager.count_active() >= Config.MAX_CONCURRENT:
            return jsonify({'error': 'Server busy', 'retry_after': 30}), 429

        logging.info(f"Sync query: {question[:60]}...")
        t0 = time.time()

        job_id = job_manager.create_job(question)
        pq: _Q.Queue = _Q.Queue(maxsize=1000)
        _register_pq(job_id, pq)

        thread = _start_worker(job_id, question, since=since, progress_q=pq)
        thread.join()  # block until done

        job = job_manager.get_job(job_id)
        elapsed = round(time.time() - t0, 1)
        logging.info(f"Sync query completed in {elapsed}s")

        return jsonify({
            'question':  question,
            'answer':    job['answer'] if job else 'Error: job lost',
            'elapsed':   elapsed,
            'job_id':    job_id,
            'timestamp': datetime.now().isoformat(),
        }), 200

    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query_async', methods=['POST'])
@require_api_key
def query_async_endpoint():
    """Asynchronous query -- returns job_id immediately."""
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        since        = data.get('since', None)
        callback_url = data.get('callback_url', None)

        with _rj_lock:
            n_running = sum(1 for t in running_jobs.values() if t.is_alive())
        if n_running >= Config.MAX_CONCURRENT:
            return jsonify({'error': 'Server busy', 'retry_after': 30}), 429

        job_id = job_manager.create_job(question, callback_url=callback_url)
        logging.info(f"Async query (job {job_id}): {question[:60]}...")

        _start_worker(job_id, question, since=since, callback_url=callback_url)

        return jsonify({
            'job_id':       job_id,
            'question':     question,
            'status':       'pending',
            'timestamp':    datetime.now().isoformat(),
            'callback_url': callback_url,
        }), 202

    except Exception as e:
        logging.error(f"Async query error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES -- Streaming SSE
# =============================================================================

_LLMTOK_RE   = re.compile(r'^\[LLMTOK\](.*)')
_PHASE_RE_V3 = re.compile(r'Phase\s+([\dA-Z]+[ABC]?)\s+([\w]+)\s+[│|]\s*([\d.]+)s', re.IGNORECASE)
_PHASE_RE_V2 = re.compile(r'PHASE\s+([\dA-Z]+[AB]?)\s*:\s*(.+)', re.IGNORECASE)
_WAVE_RE     = re.compile(r'Wave\s+(\d+)/(\d+):\s*\[([^\]]+)\]')
_SP_TURN_RE  = re.compile(r'\[(SP\w+)\]\s+Turn\s+(\d+)/(\d+)\s+→\s+(\w+):\s*(.{0,60})')
_SP_DONE_RE  = re.compile(r'\[(SP\w+)\]\s+(SOLVED|FAILED|TIMEOUT)[^|│]*[|│]\s*(.+?)\s*[|│]\s*(\d+)\s+turns')
_DONE_RE     = re.compile(r'Done.*?(\d+)/(\d+)\s+SP.*?Total:\s*([\d.]+)s')
_TOK_RE      = re.compile(r'(\d+)\s+tokens?\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)')
_AGENT_RE    = re.compile(r'Initialized\s+(\S+)\s+\(.*?\)\s+using\s+(\S+)')
_ERROR_RE    = re.compile(r'Error:|ERROR:')


def _parse_event(line: str) -> dict:
    """Parse a stdout line into a structured SSE event dict (V3-aware)."""
    base = {'line': line}

    # LLM token chunk: "[LLMTOK]escaped content" — checked first for performance
    m = _LLMTOK_RE.match(line)
    if m:
        content = m.group(1).replace('\\n', '\n').replace('\\\\', '\\')
        return {**base, 'type': 'llm_chunk', 'content': content}

    # V3 phase: "Phase 0A Classification │ 0.2s"
    m = _PHASE_RE_V3.search(line)
    if m:
        return {**base, 'type': 'phase',
                'phase_id': m.group(1), 'phase_name': m.group(2).strip(),
                'elapsed_s': float(m.group(3))}

    # V2 phase: "PHASE 1: Research"
    m = _PHASE_RE_V2.search(line)
    if m:
        return {**base, 'type': 'phase',
                'phase_id': m.group(1), 'phase_name': m.group(2).strip()}

    # Wave: "Wave 1/2: [SP1,SP2,SP3]"
    m = _WAVE_RE.search(line)
    if m:
        return {**base, 'type': 'wave',
                'wave': int(m.group(1)), 'total_waves': int(m.group(2)),
                'sps': [s.strip() for s in m.group(3).split(',')]}

    # SP turn: "[SP1] Turn 3/15 → run_code: 'import numpy...'"
    m = _SP_TURN_RE.search(line)
    if m:
        return {**base, 'type': 'sp_turn',
                'sp_id': m.group(1), 'turn': int(m.group(2)), 'max_turns': int(m.group(3)),
                'tool': m.group(4), 'preview': m.group(5).strip()}

    # SP done: "[SP1] SOLVED | r_0=0.739m | 3 turns"
    m = _SP_DONE_RE.search(line)
    if m:
        return {**base, 'type': 'sp_done',
                'sp_id': m.group(1), 'status': m.group(2).lower(),
                'values': m.group(3).strip(), 'turns': int(m.group(4))}

    # Done summary
    m = _DONE_RE.search(line)
    if m:
        return {**base, 'type': 'solve_done',
                'solved': int(m.group(1)), 'total': int(m.group(2)),
                'total_s': float(m.group(3))}

    m = _TOK_RE.search(line)
    if m:
        return {**base, 'type': 'toks',
                'tokens': int(m.group(1)),
                'seconds': float(m.group(2)),
                'toks_per_sec': float(m.group(3))}

    m = _AGENT_RE.search(line)
    if m:
        return {**base, 'type': 'agent',
                'agent': m.group(1), 'model': m.group(2)}

    if _ERROR_RE.search(line):
        return {**base, 'type': 'error_line'}

    return {**base, 'type': 'log'}


# Backwards-compat alias used by query_stream
_classify_line = _parse_event


@app.route('/query_stream', methods=['POST'])
@require_api_key
def query_stream():
    """
    SSE streaming query.  Returns text/event-stream.

    Event types emitted:
      start      -- job begun          {job_id, question}
      phase      -- new pipeline phase {phase_id, phase_name}
      toks       -- LLM completion     {tokens, seconds, toks_per_sec}
      agent      -- agent started      {agent, model}
      log        -- generic line       {line}
      error_line -- error detected     {line}
      heartbeat  -- keep-alive every ~10s
      answer     -- final answer       {answer, elapsed}
      done       -- stream closed      {job_id, elapsed}
    """
    data = request.json or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    since = data.get('since', None)

    with _rj_lock:
        n_running = sum(1 for t in running_jobs.values() if t.is_alive())
    if n_running >= Config.MAX_CONCURRENT:
        return jsonify({'error': 'Server busy', 'retry_after': 30}), 429

    job_id = job_manager.create_job(question)
    pq: _Q.Queue = _Q.Queue(maxsize=1000)
    _register_pq(job_id, pq)

    _start_worker(job_id, question, since=since, progress_q=pq)

    def _sse(d: dict) -> str:
        return f"data: {json.dumps(d)}\n\n"

    @stream_with_context
    def generate():
        t0 = time.time()
        yield _sse({'type': 'start', 'job_id': job_id, 'question': question})

        while True:
            try:
                line = pq.get(timeout=10)
            except _Q.Empty:
                yield _sse({'type': 'heartbeat',
                             'elapsed': round(time.time() - t0, 1)})
                continue

            if line is None:          # sentinel -- worker finished
                elapsed = round(time.time() - t0, 1)
                job = job_manager.get_job(job_id)
                if job and job['status'] == 'completed':
                    yield _sse({'type': 'answer', 'answer': job['answer'],
                                 'elapsed': elapsed})
                elif job and job['status'] == 'failed':
                    yield _sse({'type': 'error',
                                 'error': job.get('error', 'unknown'),
                                 'elapsed': elapsed})
                yield _sse({'type': 'done', 'job_id': job_id,
                             'elapsed': elapsed})
                break

            elapsed = round(time.time() - t0, 1)
            evt = _classify_line(line)
            evt['elapsed'] = elapsed

            # Also update job progress log
            job_manager.append_log(job_id, line)

            yield _sse(evt)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'X-Accel-Buffering': 'no',
            'Cache-Control':     'no-cache',
            'Connection':        'keep-alive',
        }
    )


# =============================================================================
# ROUTES -- Project Session
# =============================================================================

@app.route('/project/start', methods=['POST'])
@require_api_key
def project_start():
    if not _HAS_PROJECT_SESSION:
        return jsonify({'error': 'project_session.py not available'}), 503
    data = request.json or {}
    description = data.get('description', '').strip()
    if not description:
        return jsonify({'error': 'No description provided'}), 400
    try:
        session = session_manager.create(description)
        return jsonify(ProjectSessionManager.to_response(session)), 201
    except Exception as e:
        logging.error(f"project/start error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/project/respond', methods=['POST'])
@require_api_key
def project_respond():
    if not _HAS_PROJECT_SESSION:
        return jsonify({'error': 'project_session.py not available'}), 503
    data     = request.json or {}
    session_id = data.get('session_id', '').strip()
    answer   = data.get('answer', '').strip()
    if not session_id:
        return jsonify({'error': 'No session_id provided'}), 400
    session = session_manager.get(session_id)
    if session is None:
        return jsonify({'error': 'Session not found or expired'}), 404
    if session.state == 'done':
        return jsonify(ProjectSessionManager.to_response(session)), 200
    try:
        session = session_manager.advance(session_id, answer)
        return jsonify(ProjectSessionManager.to_response(session)), 200
    except Exception as e:
        logging.error(f"project/respond error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/project/session/<session_id>', methods=['GET'])
def project_get_session(session_id: str):
    if not _HAS_PROJECT_SESSION:
        return jsonify({'error': 'project_session.py not available'}), 503
    session = session_manager.get(session_id)
    if session is None:
        return jsonify({'error': 'Session not found or expired'}), 404
    return jsonify(ProjectSessionManager.to_dict(session)), 200


# =============================================================================
# ROUTES -- SSE Fan-out (poll progress_log)
# =============================================================================

@app.route('/stream/<job_id>')
def stream_job(job_id: str):
    """
    SSE fan-out from a job's progress_log.
    Clients connect here after POST /query_async to get live structured events.

    Event types: phase, wave, sp_turn, sp_done, solve_done, toks, agent,
                 log, error_line, heartbeat, answer, done
    """
    def generate():
        job = job_manager.jobs.get(job_id)
        if not job:
            yield f"data: {json.dumps({'type': 'error', 'message': 'job not found'})}\n\n"
            return

        sent = 0
        last_hb = time.time()

        while True:
            log = job.get('progress_log', [])
            while sent < len(log):
                raw_line = log[sent].strip()
                sent += 1
                if raw_line:
                    evt = _parse_event(raw_line)
                    evt['elapsed'] = round(time.time() - (
                        datetime.fromisoformat(job['created_at']).timestamp()
                        if job.get('created_at') else time.time()
                    ), 1)
                    yield f"data: {json.dumps(evt)}\n\n"

            st = job.get('status')
            if st in ('completed', 'failed'):
                if job.get('answer'):
                    yield f"data: {json.dumps({'type': 'answer', 'answer': job['answer'], 'elapsed': job.get('elapsed', 0)})}\n\n"
                elif st == 'failed':
                    yield f"data: {json.dumps({'type': 'error', 'error': job.get('error', 'unknown')})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'job_id': job_id})}\n\n"
                return

            # Heartbeat every 25s
            if time.time() - last_hb > 25:
                yield f"data: {json.dumps({'type': 'heartbeat', 'elapsed': round(time.time(), 1)})}\n\n"
                last_hb = time.time()

            time.sleep(0.4)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                 'Connection': 'keep-alive'},
    )


# =============================================================================
# ROUTES -- Metrics (GPU + server stats)
# =============================================================================

@app.route('/metrics')
def metrics():
    """GPU + system stats polled by the dashboard every 3s."""
    gpu = {}
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits',
        ], timeout=3).decode().strip()
        parts = [p.strip() for p in out.split(',')]
        gpu = {
            'name': parts[0],
            'gpu_pct': int(parts[1]),
            'mem_pct': int(parts[2]),
            'mem_used_mb': int(parts[3]),
            'mem_total_mb': int(parts[4]),
            'temp_c': int(parts[5]),
        }
    except Exception as e:
        gpu = {'error': str(e)}

    mem_pct = cpu_pct = None
    if _HAS_PSUTIL:
        try:
            mem_pct = _psutil.virtual_memory().percent
            cpu_pct = _psutil.cpu_percent(interval=None)
        except Exception:
            pass

    active = sum(1 for j in job_manager.jobs.values() if j.get('status') == 'processing')
    return jsonify({
        'gpu': gpu,
        'swarm': {
            'status': 'online',
            'active_jobs': active,
            'port': Config.PORT,
            'model': 'phi4:14b',
        },
        'memory_pct': mem_pct,
        'cpu_pct': cpu_pct,
    })


# =============================================================================
# ROUTES -- Dashboard SPA
# =============================================================================

@app.route('/dashboard')
def dashboard():
    """Serve the Jarvis Command Station single-page app."""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard.html')
    if not os.path.exists(html_path):
        return '<h1>dashboard.html not found</h1><p>Deploy server/dashboard.html next to swarm_api_server.py</p>', 404
    return send_file(html_path, mimetype='text/html')


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Swarm 3.0 REST API Server")
    parser.add_argument('--port',    type=int, default=Config.PORT)
    parser.add_argument('--host',             default=Config.HOST)
    parser.add_argument('--debug',   action='store_true')
    parser.add_argument('--searxng', type=str)
    args = parser.parse_args()

    if args.searxng:
        Config.SEARXNG_URL = args.searxng
    if args.debug:
        Config.DEBUG = True

    auth_status = "enabled" if Config.API_KEY else "disabled (set SWARM_API_KEY to enable)"

    print("\n" + "=" * 62)
    print("Swarm 3.0 REST API Server")
    print("=" * 62)
    print(f"Host:         {args.host}:{args.port}")
    print(f"Debug:        {Config.DEBUG}")
    print(f"SearXNG:      {Config.SEARXNG_URL or 'Not configured'}")
    print(f"Max jobs:     {Config.MAX_CONCURRENT}")
    print(f"Auth:         {auth_status}")
    print(f"Project mode: {'available' if _HAS_PROJECT_SESSION else 'unavailable'}")
    print("\nEndpoints (open):")
    print("  GET  /health")
    print("  GET  /status")
    print("  GET  /jobs")
    print("  GET  /result/<id>")
    print("  GET  /logs/<id>             -- full job log (?tail=N&grep=pat)")
    print("  GET  /stream/<job_id>      -- SSE progress fan-out")
    print("  GET  /metrics              -- GPU + server stats")
    print("  GET  /dashboard            -- Command Station SPA")
    print("  GET  /project/session/<id>")
    print("\nEndpoints (auth-protected):")
    print("  POST /query                -- sync (blocking)")
    print("  POST /query_async          -- async, returns job_id")
    print("  POST /query_stream         -- SSE live progress stream")
    print("  POST /project/start")
    print("  POST /project/respond")
    print("=" * 62 + "\n")

    try:
        app.run(host=args.host, port=args.port, debug=Config.DEBUG, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
