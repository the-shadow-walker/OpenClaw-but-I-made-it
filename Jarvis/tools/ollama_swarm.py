"""
tools/ollama_swarm.py
=====================
Client for the Ollama Swarm 3.0 research system (port 5002).
Wraps the deep-search integration with a proper typed client.

Service: http://10.0.0.58:5002
Auth: Authorization: Bearer <key> (optional)
"""

import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaSwarmClient:
    """Client for Swarm 3.0 at http://10.0.0.58:5002"""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self._headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def is_available(self) -> bool:
        """Check if the service is up"""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def query_sync(self, question: str, timeout: int = 180) -> Optional[dict]:
        """Blocking research query. Returns full result dict or None."""
        try:
            r = requests.post(
                f"{self.base_url}/query",
                headers=self._headers,
                json={"question": question},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"[Swarm] query_sync failed: {e}")
            return None

    def submit_async(self, question: str, webhook_url: str = None) -> Optional[str]:
        """Submit async research job. Returns job_id or None."""
        payload = {"question": question}
        if webhook_url:
            payload["webhook_url"] = webhook_url
        try:
            r = requests.post(
                f"{self.base_url}/query_async",
                headers=self._headers,
                json=payload,
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("job_id")
        except Exception as e:
            logger.error(f"[Swarm] submit_async failed: {e}")
            return None

    def poll_result(self, job_id: str, timeout: int = 300, interval: int = 5) -> Optional[dict]:
        """Poll async job until complete. Returns result or None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.base_url}/result/{job_id}",
                    headers=self._headers,
                    timeout=10,
                )
                r.raise_for_status()
                data = r.json()
                if data.get("status") in ("completed", "error"):
                    return data
                time.sleep(interval)
            except Exception as e:
                logger.error(f"[Swarm] poll {job_id} failed: {e}")
                return None
        return None

    def start_project(self, description: str, budget: float = 300) -> Optional[dict]:
        """Start an engineering design project session."""
        try:
            r = requests.post(
                f"{self.base_url}/project/new",
                headers=self._headers,
                json={"description": description, "budget": budget},
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"[Swarm] start_project failed: {e}")
            return None
