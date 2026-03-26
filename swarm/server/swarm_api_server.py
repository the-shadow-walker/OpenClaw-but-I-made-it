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

This runs on your server (same machine as Swarm orchestrator) and provides REST endpoints
that Jarvis can call to run deep searches asynchronously.

Usage:
    python3 swarm_api_server.py --port 5000

Then set SWARM_SERVER='http://your-server-ip:5000' when running Jarvis

Auth:
    Set SWARM_API_KEY env var to enable Bearer token auth on write endpoints.
    Leave unset to run without auth (dev/local mode).
"""

import os
import sys
import json
import asyncio
import argparse
import uuid
import threading
import functools
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Try to import orchestrator - adjust path as needed
try:
    from orchestrator_v2_1 import OrchestratorV2_1
except ImportError:
    print("⚠️ Could not import OrchestratorV2_1")
    print("   Make sure orchestrator_v2_1.py is in the parent directory")
    OrchestratorV2_1 = None

# Try to import project session manager
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
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    HOST = '0.0.0.0'
    PORT = int(os.getenv('SWARM_API_PORT', 5002))
    SEARXNG_URL = os.getenv('SEARXNG_URL', None)
    RESULTS_DIR = Path('./swarm_results')
    RESULTS_DIR.mkdir(exist_ok=True)
    MAX_CONCURRENT = int(os.getenv('MAX_CONCURRENT_JOBS', 3))
    API_KEY = os.getenv('SWARM_API_KEY', None)   # None = no auth required


# =============================================================================
# AUTH
# =============================================================================

def require_api_key(f):
    """
    Decorator that enforces Bearer token auth when SWARM_API_KEY is set.
    Requests to protected endpoints must include:
        Authorization: Bearer <SWARM_API_KEY>
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if Config.API_KEY is None:
            return f(*args, **kwargs)
        auth = request.headers.get('Authorization', '')
        if not auth.startswith('Bearer '):
            return jsonify({'error': 'Missing or malformed Authorization header'}), 401
        token = auth[len('Bearer '):]
        if token != Config.API_KEY:
            return jsonify({'error': 'Invalid API key'}), 403
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# JOB MANAGER
# =============================================================================

class JobManager:
    """Track async jobs"""

    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def create_job(self, question: str, callback_url: str = None) -> str:
        """Create a new job"""
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self.jobs[job_id] = {
                'question': question,
                'status': 'pending',
                'answer': None,
                'progress': '',
                'created_at': datetime.now().isoformat(),
                'completed_at': None,
                'error': None,
                'callback_url': callback_url,
            }
        return job_id

    def update_job(self, job_id: str, status: str, **kwargs):
        """Update job status"""
        with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                self.jobs[job_id].update(kwargs)
                if status in ('completed', 'failed'):
                    self.jobs[job_id]['completed_at'] = datetime.now().isoformat()

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job info"""
        with self._lock:
            return self.jobs.get(job_id)

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed jobs"""
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
        """Count jobs that are currently being processed."""
        with self._lock:
            return sum(
                1 for j in self.jobs.values()
                if j['status'] in ('pending', 'processing')
            )


# =============================================================================
# ORCHESTRATOR FACTORY
# =============================================================================

def _make_orchestrator(date_filter: str = None):
    """Create a fresh OrchestratorV2_1 instance per request to avoid state-bleed."""
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
        logging.error(f"❌ Failed to create orchestrator: {e}")
        return None


async def _run_question(question: str, date_filter: str = None) -> str:
    """Create a fresh orchestrator, run question, return answer string."""
    orch = _make_orchestrator(date_filter)
    if orch is None:
        return "Error: Orchestrator not available"
    try:
        return await orch.process_question(question)
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return f"Error processing question: {e}"


# =============================================================================
# WEBHOOK DELIVERY
# =============================================================================

def _fire_webhook(callback_url: str, payload: dict):
    """POST job result to callback_url. Errors are logged but not fatal."""
    try:
        import requests as req_lib
        req_lib.post(callback_url, json=payload, timeout=10)
        logging.info(f"🔔 Webhook delivered to {callback_url}")
    except Exception as e:
        logging.warning(f"⚠️  Webhook delivery failed ({callback_url}): {e}")


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if Config.DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize job manager
job_manager = JobManager()

# Track background threads (job_id → thread)
running_jobs: Dict[str, threading.Thread] = {}
_running_jobs_lock = threading.Lock()


# =============================================================================
# ROUTES — Health / Status (no auth)
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'orchestrator_available': OrchestratorV2_1 is not None,
        'auth_enabled': Config.API_KEY is not None,
    }), 200


@app.route('/status', methods=['GET'])
def status():
    """Get server status"""
    try:
        jobs = job_manager.jobs
        pending    = sum(1 for j in jobs.values() if j['status'] == 'pending')
        processing = sum(1 for j in jobs.values() if j['status'] == 'processing')
        completed  = sum(1 for j in jobs.values() if j['status'] == 'completed')
        failed     = sum(1 for j in jobs.values() if j['status'] == 'failed')
        with _running_jobs_lock:
            n_running = len(running_jobs)

        return jsonify({
            'server': 'healthy',
            'orchestrator': OrchestratorV2_1 is not None,
            'jobs': {
                'pending': pending,
                'processing': processing,
                'completed': completed,
                'failed': failed,
                'running': n_running,
            },
            'config': {
                'max_concurrent': Config.MAX_CONCURRENT,
                'searxng': bool(Config.SEARXNG_URL),
                'auth_enabled': Config.API_KEY is not None,
            }
        }), 200
    except Exception as e:
        logging.error(f"❌ Status error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    try:
        jobs = list(job_manager.jobs.values())
        return jsonify({'total': len(jobs), 'jobs': jobs}), 200
    except Exception as e:
        logging.error(f"❌ Jobs list error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/result/<job_id>', methods=['GET'])
def get_result(job_id: str):
    """Get result of async job"""
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
            'error':        job['error'],
            'created_at':   job['created_at'],
            'completed_at': job['completed_at'],
        }), 200
    except Exception as e:
        logging.error(f"❌ Result fetch error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES — Query (auth-protected)
# =============================================================================

@app.route('/query', methods=['POST'])
@require_api_key
def query_sync():
    """Synchronous query (blocking, returns answer immediately)"""
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        since = data.get('since', None)

        # Enforce concurrency limit
        active = job_manager.count_active()
        if active >= Config.MAX_CONCURRENT:
            return jsonify({'error': 'Server busy', 'retry_after': 30}), 429

        logging.info(f"📝 Sync query: {question[:60]}...")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        answer = loop.run_until_complete(_run_question(question, date_filter=since))
        loop.close()

        logging.info(f"✅ Completed")
        return jsonify({
            'question':  question,
            'answer':    answer,
            'timestamp': datetime.now().isoformat(),
        }), 200

    except Exception as e:
        logging.error(f"❌ Query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/query_async', methods=['POST'])
@require_api_key
def query_async_endpoint():
    """Asynchronous query (returns job ID immediately)"""
    try:
        data = request.json or {}
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        since        = data.get('since', None)
        callback_url = data.get('callback_url', None)

        # Enforce concurrency limit
        with _running_jobs_lock:
            n_running = sum(1 for t in running_jobs.values() if t.is_alive())
        if n_running >= Config.MAX_CONCURRENT:
            return jsonify({'error': 'Server busy', 'retry_after': 30}), 429

        job_id = job_manager.create_job(question, callback_url=callback_url)
        logging.info(f"📝 Async query (job {job_id}): {question[:60]}...")

        def process_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                job_manager.update_job(job_id, 'processing', progress='Starting research...')
                answer = loop.run_until_complete(_run_question(question, date_filter=since))
                job_manager.update_job(job_id, 'completed', answer=answer)
                logging.info(f"✅ Job {job_id} completed")

                # Fire webhook if requested
                if callback_url:
                    _fire_webhook(callback_url, {
                        'job_id':    job_id,
                        'status':    'completed',
                        'answer':    answer,
                        'timestamp': datetime.now().isoformat(),
                    })

            except Exception as e:
                logging.error(f"❌ Job {job_id} error: {e}")
                job_manager.update_job(job_id, 'failed', error=str(e))
                if callback_url:
                    _fire_webhook(callback_url, {
                        'job_id':    job_id,
                        'status':    'failed',
                        'error':     str(e),
                        'timestamp': datetime.now().isoformat(),
                    })
            finally:
                loop.close()
                with _running_jobs_lock:
                    running_jobs.pop(job_id, None)

        thread = threading.Thread(target=process_async, daemon=True)
        with _running_jobs_lock:
            running_jobs[job_id] = thread
        thread.start()

        return jsonify({
            'job_id':        job_id,
            'question':      question,
            'status':        'pending',
            'timestamp':     datetime.now().isoformat(),
            'callback_url':  callback_url,
        }), 202

    except Exception as e:
        logging.error(f"❌ Async query error: {e}")
        return jsonify({'error': str(e)}), 500


# =============================================================================
# ROUTES — Project Session (auth-protected)
# =============================================================================

@app.route('/project/start', methods=['POST'])
@require_api_key
def project_start():
    """
    Start a new project session.

    Request body:
        {"description": "a GPS weather station with solar charging"}

    Response:
        {"session_id": "...", "state": "qa", "question": "...", "type": "text", ...}
    """
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
        logging.error(f"❌ project/start error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/project/respond', methods=['POST'])
@require_api_key
def project_respond():
    """
    Submit an answer to the current pending question.

    Request body:
        {"session_id": "...", "answer": "..."}

    Response (while questions remain):
        {"session_id": "...", "state": "qa", "question": "...", "type": "text", ...}

    Response (when done):
        {"session_id": "...", "state": "done", "result_markdown": "...", "requirements": {...}}
    """
    if not _HAS_PROJECT_SESSION:
        return jsonify({'error': 'project_session.py not available'}), 503

    data = request.json or {}
    session_id = data.get('session_id', '').strip()
    answer     = data.get('answer', '').strip()

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
        logging.error(f"❌ project/respond error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/project/session/<session_id>', methods=['GET'])
def project_get_session(session_id: str):
    """Get full session state (no auth required — useful for polling)."""
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
    parser.add_argument('--port',    type=int,         default=Config.PORT,    help='Port to listen on')
    parser.add_argument('--host',                      default=Config.HOST,    help='Host to bind to')
    parser.add_argument('--debug',   action='store_true',                      help='Enable debug mode')
    parser.add_argument('--searxng', type=str,                                 help='SearXNG URL')

    args = parser.parse_args()

    if args.searxng:
        Config.SEARXNG_URL = args.searxng
    if args.debug:
        Config.DEBUG = True

    auth_status = f"enabled (SWARM_API_KEY set)" if Config.API_KEY else "disabled (set SWARM_API_KEY to enable)"

    print("\n" + "=" * 62)
    print("🚀 Swarm 3.0 REST API Server")
    print("=" * 62)
    print(f"Host:         {args.host}:{args.port}")
    print(f"Debug:        {Config.DEBUG}")
    print(f"SearXNG:      {Config.SEARXNG_URL or 'Not configured'}")
    print(f"Max jobs:     {Config.MAX_CONCURRENT}")
    print(f"Auth:         {auth_status}")
    print(f"Project mode: {'available' if _HAS_PROJECT_SESSION else 'unavailable'}")
    print("\nEndpoints (open):")
    print(f"  GET  /health                  - Health check")
    print(f"  GET  /status                  - Server status + job counts")
    print(f"  GET  /jobs                    - List all jobs")
    print(f"  GET  /result/<id>             - Get async job result")
    print(f"  GET  /project/session/<id>    - Get project session state")
    print("\nEndpoints (auth-protected):")
    print(f"  POST /query                   - Sync query (blocking)")
    print(f"  POST /query_async             - Async query (returns job_id)")
    print(f"  POST /project/start           - Start project Q&A session")
    print(f"  POST /project/respond         - Answer current project question")
    print("=" * 62 + "\n")

    try:
        app.run(host=args.host, port=args.port, debug=Config.DEBUG, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


if __name__ == '__main__':
    main()
