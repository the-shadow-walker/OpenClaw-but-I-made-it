"""
tools/ollama_cmd.py
===================
Client for the ollama-cmd autonomous agent service (port 5000).
Allows JARVIS to dispatch single-task jobs and multi-phase chains to the
ReAct-loop agent on arch01.

Service: http://10.0.0.58:5000
Auth: X-API-Key header (set OLLAMA_CMD_API_KEY in config)
"""

import json
import uuid
import time
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaCMDClient:
    """Client for the ollama-cmd service at http://10.0.0.58:5000"""

    def __init__(self, base_url: str, api_key: str = "", inbox_dir: Optional[Path] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.inbox_dir = inbox_dir
        self._headers = {"X-API-Key": api_key} if api_key else {}

    def is_available(self) -> bool:
        """Check if the service is up"""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def submit_job(self, instruction: str, max_iterations: int = 25) -> Optional[str]:
        """Submit a single-task job. Returns job_id or None on failure."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/execute",
                headers=self._headers,
                json={"instruction": instruction, "max_iterations": max_iterations},
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("job_id")
        except Exception as e:
            logger.error(f"[OllamaCMD] submit_job failed: {e}")
            return None

    def submit_chain(self, goal: str, total_budget: int = 200) -> Optional[str]:
        """Submit a multi-phase chain. Returns chain_id or None on failure."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/chains",
                headers=self._headers,
                json={"goal": goal, "total_budget": total_budget},
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("chain_id")
        except Exception as e:
            logger.error(f"[OllamaCMD] submit_chain failed: {e}")
            return None

    def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get current job status without blocking."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/jobs/{job_id}",
                headers=self._headers,
                timeout=5,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"[OllamaCMD] get_job_status {job_id} failed: {e}")
            return None

    def wait_for_job(self, job_id: str, timeout: int = 120) -> Optional[dict]:
        """Poll a job until completion. Returns result dict or None on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.get_job_status(job_id)
            if data and data.get("status") not in ("queued", "running"):
                return data
            time.sleep(3)
        logger.warning(f"[OllamaCMD] job {job_id} timed out after {timeout}s")
        return None

    def get_chain_status(self, chain_id: str) -> Optional[dict]:
        """Get chain status."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/chains/{chain_id}",
                headers=self._headers,
                timeout=5,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"[OllamaCMD] get_chain_status failed: {e}")
            return None

    def drop_to_inbox(self, instruction: str = None, goal: str = None) -> Optional[str]:
        """Write a job file to the inbox directory. Returns filename or None."""
        if not self.inbox_dir:
            logger.warning("[OllamaCMD] inbox_dir not configured")
            return None
        try:
            payload: dict = {}
            if goal:
                payload["goal"] = goal
            elif instruction:
                payload["instruction"] = instruction
                payload["max_iterations"] = 25
            else:
                return None
            filename = f"jarvis_{uuid.uuid4().hex[:8]}.json"
            path = self.inbox_dir / filename
            path.write_text(json.dumps(payload))
            logger.info(f"[OllamaCMD] Dropped to inbox: {path}")
            return filename
        except Exception as e:
            logger.error(f"[OllamaCMD] drop_to_inbox failed: {e}")
            return None

    def quick_query(self, question: str) -> Optional[dict]:
        """Synchronous single-command query (Tier 1 — 1-3s).
        Returns {command, stdout, stderr, returncode, success, elapsed_ms, risk} or None."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/quick",
                headers=self._headers,
                json={"question": question},
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"[OllamaCMD] quick_query failed: {e}")
            return None

    def get_state(self) -> Optional[dict]:
        """Get current queue/active jobs state snapshot."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/state",
                headers=self._headers,
                timeout=5,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error(f"[OllamaCMD] get_state failed: {e}")
            return None

    def get_security_report(self) -> Optional[str]:
        """Fetch the latest nightly SENTINEL security audit report (generated at 3 AM)."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/blueteam/report",
                headers=self._headers,
                timeout=10,
            )
            r.raise_for_status()
            # Returns markdown text by default
            return r.text
        except Exception as e:
            logger.error(f"[OllamaCMD] get_security_report failed: {e}")
            return None

    def get_security_alerts(self, n: int = 20) -> Optional[list]:
        """Fetch recent security alerts from SENTINEL."""
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/blueteam/alerts",
                headers=self._headers,
                params={"n": n},
                timeout=10,
            )
            r.raise_for_status()
            return r.json().get("alerts", [])
        except Exception as e:
            logger.error(f"[OllamaCMD] get_security_alerts failed: {e}")
            return None
