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

# Import the agent (assumes ollama_agent.py is in same directory)
from ollama_agent_core import OllamaCommandAgent
from task_chain import HandoffExtractor, AcceptanceCriteriaRunner, SubtaskReplanner, TaskDecomposer, TaskChain

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
API_KEY = os.environ.get('AGENT_API_KEY', secrets.token_urlsafe(32))
MAX_CONCURRENT_JOBS = 3
JOB_TIMEOUT = 3600  # 1 hour max per job

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}
job_queue = queue.Queue()
active_jobs = {}


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
                
                # Create agent
                agent = OllamaCommandAgent(
                    model=job.get('model', 'qwen3-coder:30b'),
                    searxng_url=job.get('searxng_url', 'http://10.0.0.58:8080')
                )
                
                # Capture output
                output_buffer = []
                
                def capture_print(text):
                    output_buffer.append(text)
                    job['output'] = '\n'.join(output_buffer)
                
                # Monkey patch print for this job (not ideal but works)
                original_print = print
                import builtins
                builtins.print = lambda *args, **kwargs: capture_print(' '.join(map(str, args)))
                
                try:
                    # Honor per-job max_iterations (watchdog uses 15 vs default 25)
                    if 'max_iterations' in job:
                        agent.max_react_iterations = int(job['max_iterations'])

                    # Run the agent
                    result = agent.run(
                        job['instruction'],
                        incoming_handoff=job.get('incoming_handoff'),
                    )

                    # Store results
                    job['status'] = 'completed'
                    job['execution_log'] = agent.execution_log
                    job['success'] = result.get('success', True) if result else True
                    job['_react_result'] = result
                    job['_react_trace'] = agent.react_trace

                except Exception as e:
                    job['status'] = 'failed'
                    job['error'] = str(e)
                    job['success'] = False

                finally:
                    builtins.print = original_print
                
            except Exception as e:
                job['status'] = 'failed'
                job['error'] = f"Job execution error: {str(e)}"
                job['success'] = False
            
            finally:
                job['completed_at'] = datetime.now().isoformat()
                if job_id in active_jobs:
                    del active_jobs[job_id]

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
        return

    # 10. Update current index and submit next subtask
    chain = TaskChain.load(chain_id)
    chain.update_chain({"current_subtask_index": next_index})
    print(f"▶️  Chain {chain_id[:8]}: advancing to subtask {next_index}")
    _submit_subtask_job(chain_id, chain.data, next_index, 0, handoff)


def require_api_key(f):
    """Decorator to require API key authentication"""
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
    """Health check endpoint"""
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


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Ollama Command Agent Service")
    print("=" * 70)
    print(f"API Key: {API_KEY}")
    print(f"Port: 5000")
    print(f"Max Concurrent Jobs: {MAX_CONCURRENT_JOBS}")
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
    print(f"  GET    /health                   - Health check")
    print("=" * 70)
    print("\nStarting server...")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        job_runner.stop()
