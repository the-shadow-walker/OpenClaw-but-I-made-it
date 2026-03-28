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
import queue as _Q
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import logging

try:
    from orchestrator_v2_1 import OrchestratorV2_1
except ImportError:
    print("Warning: Could not import OrchestratorV2_1")
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

# ContextVar for job routing — propagates into asyncio.to_thread() executor threads
_current_job_id: contextvars.ContextVar[str] = contextvars.ContextVar("swarm_job", default="")

def _register_pq(job_id: str, q: _Q.Queue):
    with _pq_lock:
        _progress_queues[job_id] = q

def _unregister_pq(job_id: str):
    with _pq_lock:
        _progress_queues.pop(job_id, None)


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
        # ContextVar propagates through asyncio.to_thread; thread-local is fallback
        job_id = _current_job_id.get("")
        if not job_id:
            job_id = getattr(threading.current_thread(), "_swarm_job_id", "")
        return job_id or None

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
                'job_id':      job_id,
                'question':    question,
                'status':      'pending',
                'answer':      None,
                'progress':    '',
                'progress_log': [],
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
                log = self.jobs[job_id]['progress_log']
                log.append(line)
                if len(log) > 200:
                    self.jobs[job_id]['progress_log'] = log[-100:]
                # Update last progress line (strip emoji/debug noise for status field)
                clean = re.sub(r'[^\x20-\x7e]', '', line).strip()
                if clean:
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
    if OrchestratorV2_1 is None:
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
        return OrchestratorV2_1(**kwargs)
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
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            job_manager.update_job(job_id, 'processing', progress='Initializing...')
            async def _run_with_ctx():
                _current_job_id.set(job_id)
                return await _run_question(question, date_filter=since)
            answer = loop.run_until_complete(_run_with_ctx())
            elapsed = round(time.time() - t0, 1)
            job_manager.update_job(job_id, 'completed', answer=answer, elapsed=elapsed)
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
            loop.close()
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
        with job_manager._lock:
            jobs = [{'job_id': k, **v} for k, v in job_manager.jobs.items()]
        return jsonify({'total': len(jobs), 'jobs': jobs}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id: str):
    try:
        job = job_manager.get_job(job_id)
        if job is None:
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

_PHASE_RE   = re.compile(r'PHASE\s+([\dA-Z]+[AB]?)\s*:\s*(.+)', re.IGNORECASE)
_TOK_RE     = re.compile(r'(\d+)\s+tokens?\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+tok/s\)')
_AGENT_RE   = re.compile(r'Initialized\s+(\S+)\s+\(.*?\)\s+using\s+(\S+)')
_ERROR_RE   = re.compile(r'Error:|ERROR:')


def _classify_line(line: str) -> dict:
    """Parse a stdout line into a structured SSE event dict."""
    base = {'line': line}

    m = _PHASE_RE.search(line)
    if m:
        return {**base, 'type': 'phase',
                'phase_id': m.group(1), 'phase_name': m.group(2).strip()}

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
      heartbeat  -- keep-alive every ~30s
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
                line = pq.get(timeout=30)
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
