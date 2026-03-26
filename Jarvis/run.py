#!/usr/bin/env python3
"""
run.py — JARVIS entry point
Run from /mnt/storage/NAS/Jarvis/Jarvis/

Usage: python3 run.py
"""
import sys
from pathlib import Path

# Add Jarvis root to sys.path so all package imports resolve
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from config.server_config import SERVER_HOST, SERVER_PORT

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False,
    )
