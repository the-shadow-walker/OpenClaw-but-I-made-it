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
import sys
import io
from datetime import datetime
from typing import Dict, Any, Optional
import secrets

# Import the agent
from ollama_agent_core import OllamaCommandAgent

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.environ.get('AGENT_API_KEY', secrets.token_urlsafe(32))
MAX_CONCURRENT_JOBS = 3
JOB_TIMEOUT = 3600

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}
job_queue = queue.Queue()
active_jobs = {}


class OutputCapture:
    """Captures print output for a job"""
    def __init__(self, job):
        self.job = job
        self.output = []
    
    def write(self, text):
        if text and text.strip():
            self.output.append(text)
            self.job['output'] = ''.join(self.output)
    
    def flush(self):
        pass


class JobRunner:
    """Runs agent jobs in background threads"""
    
    def __init__(self):
        self.running = True
        self.workers = []
        
        for i in range(MAX_CONCURRENT_JOBS):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"Worker-{i}")
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        """Worker thread that processes jobs"""
        while self.running:
            try:
                job_id = job_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            if job_id not in jobs:
                continue
            
            job = jobs[job_id]
            
            try:
                job['status'] = 'running'
                job['started_at'] = datetime.now().isoformat()
                active_jobs[job_id] = job
                
                # Create agent
                agent = OllamaCommandAgent(
                    model=job.get('model', 'qwen3-coder:30b'),
                    searxng_url=job.get('searxng_url', 'http://10.0.0.58:8080')
                )
                
                # Capture output
                output_capture = OutputCapture(job)
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = output_capture
                sys.stderr = output_capture
                
                try:
                    # Run the agent
                    agent.run(job['instruction'])
                    
                    job['status'] = 'completed'
                    job['execution_log'] = agent.execution_log
                    job['success'] = True
                    
                except Exception as e:
                    job['status'] = 'failed'
                    job['error'] = str(e)
                    job['success'] = False
                
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = f"Job execution error: {str(e)}"
                job['success'] = False
            
            finally:
                job['completed_at'] = datetime.now().isoformat()
                if job_id in active_jobs:
                    del active_jobs[job_id]
                
                job_queue.task_done()
    
    def submit_job(self, job_id: str):
        """Submit a job to the queue"""
        job_queue.put(job_id)
    
    def stop(self):
        """Stop all workers"""
        self.running = False


# Initialize job runner
job_runner = JobRunner()


def require_api_key(f):
    """Decorator to require API key"""
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key')
        
        if not provided_key:
            return jsonify({'error': 'API key required', 'message': 'Provide X-API-Key header'}), 401
        
        if provided_key != API_KEY:
            return jsonify({'error': 'Invalid API key'}), 403
        
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'ollama-command-agent',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(active_jobs),
        'queued_jobs': job_queue.qsize()
    })


@app.route('/api/v1/execute', methods=['POST'])
@require_api_key
def execute_command():
    """Execute a command"""
    data = request.get_json()
    
    if not data or 'instruction' not in data:
        return jsonify({'error': 'Missing instruction'}), 400
    
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
    job_runner.submit_job(job_id)
    
    if data.get('async', True):
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'created_at': job['created_at']
        }), 202
    else:
        timeout = data.get('timeout', 300)
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
    """Get job status"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(jobs[job_id])


@app.route('/api/v1/jobs/<job_id>/stream', methods=['GET'])
@require_api_key
def stream_job_output(job_id: str):
    """Stream job output (SSE)"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    def generate():
        job = jobs[job_id]
        last_output_len = 0
        
        while job['status'] in ['queued', 'running']:
            current_output = job.get('output', '')
            
            if len(current_output) > last_output_len:
                new_output = current_output[last_output_len:]
                yield f"data: {json.dumps({'type': 'output', 'content': new_output})}\n\n"
                last_output_len = len(current_output)
            
            yield f"data: {json.dumps({'type': 'status', 'status': job['status']})}\n\n"
            time.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'complete', 'status': job['status'], 'success': job.get('success')})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/v1/jobs', methods=['GET'])
@require_api_key
def list_jobs():
    """List all jobs"""
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
    
    filtered_jobs.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({
        'jobs': filtered_jobs[:limit],
        'total': len(filtered_jobs)
    })


@app.route('/api/v1/jobs/<job_id>', methods=['DELETE'])
@require_api_key
def cancel_job(job_id: str):
    """Cancel a job"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    if job['status'] in ['completed', 'failed']:
        return jsonify({'error': 'Cannot cancel completed job'}), 400
    
    job['status'] = 'cancelled'
    job['completed_at'] = datetime.now().isoformat()
    
    return jsonify({'message': 'Job cancelled', 'job_id': job_id})


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Ollama Command Agent Service")
    print("=" * 70)
    print(f"API Key: {API_KEY}")
    print(f"Port: 5000")
    print(f"Max Concurrent Jobs: {MAX_CONCURRENT_JOBS}")
    print(f"=" * 70)
    print("\nStarting server...")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        job_runner.stop()
