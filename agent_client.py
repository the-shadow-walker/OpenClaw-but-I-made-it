#!/usr/bin/env python3
"""Client library for Ollama Command Agent Service"""

import requests
import json
import time
from typing import Dict, Any, Optional, Generator

class AgentClient:
    """Client for Ollama Command Agent Service"""
    
    def __init__(self, base_url: str = "http://localhost:5000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers['X-API-Key'] = api_key
    
    def health(self) -> Dict[str, Any]:
        """Check service health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def execute(
        self,
        instruction: str,
        model: str = "qwen3-coder:30b",
        async_mode: bool = True,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """Execute a command"""
        payload = {
            "instruction": instruction,
            "model": model,
            "async": async_mode,
            "timeout": timeout
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/execute",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status and results"""
        response = self.session.get(f"{self.base_url}/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, job_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for job to complete"""
        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job {job_id} exceeded timeout")
            
            job = self.get_job(job_id)
            
            if job['status'] not in ['queued', 'running']:
                return job
            
            time.sleep(2)
    
    def stream_output(self, job_id: str) -> Generator[Dict[str, Any], None, None]:
        """Stream job output (SSE)"""
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs/{job_id}/stream",
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    yield data
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """List jobs"""
        params = {'limit': limit}
        if status:
            params['status'] = status
        
        response = self.session.get(
            f"{self.base_url}/api/v1/jobs",
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a job"""
        response = self.session.delete(f"{self.base_url}/api/v1/jobs/{job_id}")
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    import os
    
    # Test the client
    client = AgentClient(
        base_url="http://localhost:5000",
        api_key=os.environ.get('AGENT_API_KEY')
    )
    
    print("Health:", client.health())
    
    job = client.execute("List files in current directory")
    print(f"Job submitted: {job['job_id']}")
    
    print("\nStreaming output:")
    for event in client.stream_output(job['job_id']):
        if event['type'] == 'output':
            print(event['content'], end='')
        elif event['type'] == 'complete':
            print(f"\n\nCompleted: {event['status']}")
            break
