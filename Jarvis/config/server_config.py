"""
Server Configuration for JARVIS
All constants for the server deployment on arch01.
"""

import os
from pathlib import Path

# Server Settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5003

# Ollama Configuration (LLM inference - local to arch01)
OLLAMA_HOST = "http://localhost:11434"

# Tool Services
OLLAMA_CMD_URL = "http://10.0.0.58:5000"    # ollama-cmd autonomous agent
OLLAMA_CMD_API_KEY = ""                       # Set if auth required
OLLAMA_CMD_INBOX = Path("/mnt/storage/NAS/Jarvis/agent_inbox")

# Deep Search / Swarm Research
DEEP_SEARCH_URL = "http://10.0.0.58:5002"

# Model Selection
MODELS = {
    "chat":      "qwen2.5:3b",
    "reasoning": "qwen3:30b",
    "coding":    "qwen3-coder:30b"
}

# Memory Paths
# BASE_DIR = /mnt/storage/NAS/Jarvis/Jarvis  (parent of config/)
BASE_DIR = Path(__file__).parent.parent
MEMORY_DIR         = BASE_DIR / "memory"
FACTS_DB_PATH      = MEMORY_DIR / "facts.db"
WORKFLOWS_DB_PATH  = MEMORY_DIR / "workflows.db"
CHROMA_DIR         = MEMORY_DIR / "chroma"
JOURNAL_DIR        = MEMORY_DIR / "daily_logs"
PROJECTS_DIR       = MEMORY_DIR / "active_projects"
TASKS_DIR          = MEMORY_DIR / "background_tasks"
LEARNING_FILE      = MEMORY_DIR / "learning.json"
PERSONALITY_FILE   = MEMORY_DIR / "personality.json"
USER_PROFILE_FILE  = MEMORY_DIR / "user_profile.md"
RECENT_EMAILS_FILE = MEMORY_DIR / "recent-emails.md"
TOKEN_PATH         = BASE_DIR / "token.pickle"

# Session Configuration
SESSION_TIMEOUT_HOURS = 24
SESSION_DB_PATH = MEMORY_DIR / "sessions.db"

# Context Building
CONTEXT_BUDGET = 12000
MAX_HISTORY_MESSAGES = 10
MAX_VECTOR_RESULTS = 3
JOURNAL_SEARCH_DAYS = 30

# Background Worker
IDLE_THRESHOLD_SECONDS = 60
WORKER_CHECK_INTERVAL = 30

# Email Agent
EMAIL_CHECK_INTERVAL = 300
EMAIL_BATCH_SIZE = 10

# Safety Engine
RISK_THRESHOLD_REQUIRE_CONFIRMATION = 4

# TTS (Piper)
PIPER_MODEL_PATH = Path("/mnt/storage/NAS/Jarvis/piper_voices/en_GB-alan-medium.onnx")

# Logging
LOG_LEVEL = "INFO"
LOG_DIR = BASE_DIR / "logs"

# API Settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5002",  # Swarm dashboard via SSH tunnel
    "http://localhost:5003",
    "http://localhost:5000",
    "http://10.0.0.58:3000",
    "http://10.0.0.58:5003",
    "http://10.0.0.58:5002",  # Swarm dashboard
    "http://10.0.0.58:5000",  # CMD dashboard
]


def init_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [MEMORY_DIR, CHROMA_DIR, JOURNAL_DIR, PROJECTS_DIR, TASKS_DIR, LOG_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
