#!/usr/bin/env python3
"""
Deep Search API Server
======================
Flask API wrapper for the swarm2 deep research system

Endpoints:
- POST /query_async - Start research (returns job_id)
- GET /jobs/<job_id> - Check status
- GET /jobs/<job_id>/result - Get final answer

Usage:
    python deep_search_api.py --port 5002
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import threading
import uuid
import time
from datetime import datetime
from typing import Dict, Optional
import sys
import os

# Add swarm modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your swarm orchestrator
from orchestrator_v2_1 import OrchestratorV2_1
from shared_memory import SharedMemory


app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Job storage
jobs: Dict[str, Dict] = {}


def run_research_async(job_id: str, question: str):
    """Run research in background thread"""
    
    try:
        # Update job status
        jobs[job_id]['status'] = 'running'
        jobs[job_id]['started_at'] = datetime.now().isoformat()
        
        # Create orchestrator for this job (it creates its own memory)
        job_orchestrator = OrchestratorV2_1()
        
        # Run research synchronously in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Call the main processing method
            answer = loop.run_until_complete(
                job_orchestrator.process_question(question)
            )
            
            # Store result
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['answer'] = answer or 'No answer generated'
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
            
            # Store additional info from memory
            facts = job_orchestrator.memory.get_facts(validated_only=True)
            jobs[job_id]['sources'] = [str(f.source) for f in facts[:5]]
            jobs[job_id]['facts_found'] = len(facts)
            
        finally:
            loop.close()
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['completed_at'] = datetime.now().isoformat()
        
        import traceback
        print(f"❌ Research failed for {job_id}: {e}")
        traceback.print_exc()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'deep-search-api',
        'version': '1.0',
        'active_jobs': len([j for j in jobs.values() if j['status'] == 'running'])
    })


@app.route('/query_async', methods=['POST'])
def query_async():
    """
    Start a deep research query
    
    POST /query_async
    {
        "question": "What is the lightest bulletproof material?"
    }
    
    Returns:
    {
        "job_id": "abc123",
        "status": "queued",
        "message": "Research started"
    }
    """
    
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        # Create job
        job_id = str(uuid.uuid4())[:8]
        
        jobs[job_id] = {
            'job_id': job_id,
            'question': question,
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'started_at': None,
            'completed_at': None,
            'answer': None,
            'error': None,
            'progress': 'Initializing...'
        }
        
        # Start research in background thread
        thread = threading.Thread(
            target=run_research_async,
            args=(job_id, question),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': 'Research started'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id: str):
    """
    Get job status
    
    GET /jobs/<job_id>
    
    Returns:
    {
        "job_id": "abc123",
        "status": "running" | "completed" | "failed",
        "progress": "Searching...",
        "answer": "..." (if completed),
        "error": "..." (if failed)
    }
    """
    
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'question': job['question'],
        'created_at': job['created_at'],
        'progress': job.get('progress', '')
    }
    
    if job['status'] == 'completed':
        response['answer'] = job['answer']
        response['sources'] = job.get('sources', [])
        response['facts_found'] = job.get('facts_found', 0)
        response['completed_at'] = job['completed_at']
    
    elif job['status'] == 'failed':
        response['error'] = job.get('error', 'Unknown error')
        response['completed_at'] = job['completed_at']
    
    elif job['status'] == 'running':
        response['started_at'] = job['started_at']
    
    return jsonify(response)


@app.route('/jobs/<job_id>/result', methods=['GET'])
def get_job_result(job_id: str):
    """
    Get final result (waits for completion)
    
    GET /jobs/<job_id>/result
    
    Returns:
    {
        "answer": "...",
        "sources": [...],
        "completed_at": "..."
    }
    """
    
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    # Wait for completion (with timeout)
    timeout = 300  # 5 minutes
    start = time.time()
    
    while job['status'] in ['queued', 'running']:
        if time.time() - start > timeout:
            return jsonify({'error': 'Timeout waiting for result'}), 408
        
        time.sleep(1)
    
    if job['status'] == 'completed':
        return jsonify({
            'answer': job['answer'],
            'sources': job.get('sources', []),
            'facts_found': job.get('facts_found', 0),
            'completed_at': job['completed_at']
        })
    
    elif job['status'] == 'failed':
        return jsonify({
            'error': job.get('error', 'Unknown error')
        }), 500


@app.route('/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': [
            {
                'job_id': j['job_id'],
                'status': j['status'],
                'question': j['question'][:100],
                'created_at': j['created_at']
            }
            for j in jobs.values()
        ]
    })


@app.route('/cleanup', methods=['POST'])
def cleanup_old_jobs():
    """Remove completed jobs older than 1 hour"""
    
    from datetime import datetime, timedelta
    
    cutoff = datetime.now() - timedelta(hours=1)
    removed = 0
    
    for job_id in list(jobs.keys()):
        job = jobs[job_id]
        
        if job['status'] in ['completed', 'failed']:
            completed_at = datetime.fromisoformat(job['completed_at'])
            
            if completed_at < cutoff:
                del jobs[job_id]
                removed += 1
    
    return jsonify({
        'removed': removed,
        'remaining': len(jobs)
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Deep Search API Server')
    parser.add_argument('--port', type=int, default=5002, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🔍 Deep Search API Server")
    print("="*70)
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print("\nEndpoints:")
    print(f"  POST http://{args.host}:{args.port}/query_async")
    print(f"  GET  http://{args.host}:{args.port}/jobs/<job_id>")
    print(f"  GET  http://{args.host}:{args.port}/jobs/<job_id>/result")
    print("\n" + "="*70 + "\n")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True  # Important for background tasks
    )
