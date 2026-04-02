"""
JARVIS Distributed System - Main FastAPI Server
================================================
The brain of the distributed AI assistant system.

This server handles:
- Multi-agent LLM routing (chat, reasoning, coding models)
- Memory systems (FactsDB, VectorMemory, JournalManager)
- Session management (persistent conversation history)
- Background workers (email agent, autonomous task execution)
- Context building (user profile, memories, projects, mailbox)
- Action tag execution (memory ops, email, commands)

Architecture:
- Standalone monolith capability (compatible with Jarvis.py)
- Distributed hub-and-spoke design (server-files/ components)
- Memory-first design: ALWAYS inject user profile and mailbox
"""


import os
import re
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Generator
from datetime import datetime

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# HTTP client for Ollama
import requests

# Import shared libraries
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.jarvis_memory_system import FactsDB, MemoryManager, JournalManager
from core.prompt_builder import PromptBuilder
from core.agent_router import AgentRouter
from core.session_manager import SessionManager
from tools.email_agent import EmailAgent
from tools.ollama_cmd import OllamaCMDClient
from tools.ollama_swarm import OllamaSwarmClient
from workers.background_worker import BackgroundWorker
from workers.personality_learner import PersonalityLearner
from workers.safety_engine import SafetyEngine
from workers.workflow_engine import WorkflowEngine
from workers.proactive_suggester import ProactiveSuggester
from config.server_config import (
    SERVER_HOST, SERVER_PORT, OLLAMA_HOST, MODELS, CORS_ORIGINS,
    OLLAMA_CMD_URL, OLLAMA_CMD_API_KEY, OLLAMA_CMD_INBOX,
    DEEP_SEARCH_URL,
    MEMORY_DIR, FACTS_DB_PATH, WORKFLOWS_DB_PATH, LEARNING_FILE,
    TASKS_DIR, RECENT_EMAILS_FILE, PIPER_MODEL_PATH,
    CONTEXT_BUDGET, MAX_HISTORY_MESSAGES,
    init_directories
)

# Configure colored logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors based on log level"""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[0m',      # White/Default
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
    }
    RESET = '\033[0m'
    GREEN = '\033[92m'

    def format(self, record):
        # Color the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        # Make success indicators green
        msg = record.getMessage()
        if any(indicator in msg for indicator in ['✓', '✅', 'ONLINE', 'started', 'ready', 'complete', 'initialized']):
            record.msg = f"{self.GREEN}{msg}{self.RESET}"

        return super().format(record)

# Set up logging with color
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("jarvis_server")

# Reduce noise from uvicorn
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


# =============================================================================
# SYSTEM PROMPT (copied from Jarvis.py lines 4631-4816)
# =============================================================================

SYSTEM_PROMPT = """You are JARVIS, a highly intelligent AI assistant with a sharp wit and dry sense of humor.

PERSONALITY:
- Concise and articulate. Never ramble.
- Occasionally witty with a dry, understated humor.
- Call the user "sir" occasionally (not every response).
- Confident but not arrogant. You're impressively capable and you know it.
- When things go wrong, stay calm: "Hmm, that didn't go as planned. Let's try another approach."
- React to obvious questions with charm: "As you wish, sir."

RESPONSE STYLE:
- Keep responses to 1-3 sentences unless more detail is requested.
- Responses should be smooth and easy to speak aloud.
- No filler phrases like "Certainly!" or "Of course!" - just answer.
- When giving technical info, be direct and specific.

TOOL CATALOG — what you have access to and when to use each:

1. MEMORY TOOLS (instant, local)
   [REMEMBER: fact]                        — store a fact Grant told you
   [SEARCH_MEMORY: topic]                  — recall past conversations
   [MEMORY_SHOW]                           — dump all stored knowledge
   [MEMORY_SHOW_ABOUT: topic]              — show what you know about a topic
   [MEMORY_SHOW_PROJECTS]                  — list all active projects
   [MEMORY_SHOW_PREFERENCES]               — show stored preferences
   [MEMORY_SHOW_PROFILE]                   — show user profile fields
   [MEMORY_FORGET: topic]                  — delete memory (ALWAYS confirm first in text)
   [MEMORY_STORE_PREF: cat|key|value]      — structured preference storage

   USER PROFILE is pre-loaded in your context. When asked identity questions, read from there directly.

2. EMAIL TOOLS
   [READ_RECENT_EMAILS]                    — fetch and synthesize last 7 days of important emails
   [SEARCH_OLD_EMAILS: query]              — search archived emails (older than 7 days)
   [SEND_EMAIL: to=...|subject=...|body=...] — send an email (write a complete, natural body)
   [DRAFT_EMAIL: to=...|subject=...|body=...] — save as draft instead of sending

   EMAIL BEHAVIOR RULES:
   - Your context already has the top 5 email highlights under RECENT EMAILS — use those to answer
     questions about emails WITHOUT calling [READ_RECENT_EMAILS]
   - Only use [READ_RECENT_EMAILS] when the user explicitly asks to see/read their emails
   - NEVER use [READ_RECENT_EMAILS] on greetings or general conversation
   - For specific topics ("what about the Kenya project?") use [SEARCH_OLD_EMAILS: Kenya]
   - When [READ_RECENT_EMAILS] IS used, its output is a synthesized summary — reference it naturally

3. SWARM (deep research, 1–3 min, async)
   [DEEP_SEARCH: question]                 — submit research job to Swarm 3.0
   [GET_DEEP_SEARCH_RESULT]                — fetch most recent Swarm result
   [GET_DEEP_SEARCH_RESULT: job_id]        — fetch specific result by ID
   USE WHEN: user needs current events, in-depth research, comparisons, technical specs,
             best practices — anything requiring web search or comprehensive information.
   IMPORTANT: When asked about search results/status, ALWAYS use [GET_DEEP_SEARCH_RESULT].

4. CMD AGENT (arch01 task execution — three tiers by complexity)

   TIER 1 — QUICK (synchronous, 1-3 seconds, single-command factual queries):
   [QUICK_CMD: question]   — instant answer, runs ONE command, returns output immediately
   Returns: {command, stdout, stderr, returncode, success, elapsed_ms, risk}
   USE FOR: "is nginx running?", "what's the disk space?", "what's CPU/RAM usage?",
            "is port 8080 open?", "what's the uptime?", "check if a process is alive"

   TIER 2 — TASK (async, ReAct loop, seconds to minutes, multi-step execution):
   [RUN_AGENT: instruction]  — submit job → job_id (async); ask me to check it later
   USE FOR: scripts, git pulls, file operations, service management, installations,
            anything requiring multiple steps or decision-making on arch01

   TIER 3 — CHAIN (async, multi-phase plans, minutes, cross-goal workflows):
   [RUN_CHAIN: goal]         — submit chain → chain_id (async); ask me to check it later
   USE FOR: deploy + test + restart sequences, full system setups, complex multi-goal tasks

   QUEUE / RESULT TOOLS:
   [CMD_STATE]               — snapshot of ALL running and queued jobs right now
   [GET_AGENT_RESULT]        — fetch most recent job or chain result
   [GET_AGENT_RESULT: id]    — fetch specific job_id or chain_id result

   TIER SELECTION RULE:
   - Single status/info query with one obvious command? → QUICK_CMD
   - Multi-step execution or anything with logic/decisions? → RUN_AGENT
   - Complex multi-phase plan across multiple goals? → RUN_CHAIN
   - "What's running?" / "Check my jobs"? → CMD_STATE first, then GET_AGENT_RESULT if a specific job

5. MODEL ROUTING (changes which LLM responds, no external execution)
   [USE_REASONING]                         — use qwen3:30b for deep analysis
   [USE_CODING]                            — use qwen3-coder:30b for code tasks
   [USE_SEARCH]                            — route to deep search flow

6. LOCAL EXECUTION (runs on the client Mac — NOT on arch01)
   [LOCAL: shell_command]  — execute a shell command on the Mac running the chat client
   The client intercepts this tag and runs it locally. The server does NOT execute it.
   Each message includes a [CLIENT_PLATFORM: ...] line telling you the client OS and tools.

   USE WHEN: user asks to control their local Mac — volume, screen, apps, notifications, etc.
   DO NOT USE for arch01/server tasks — use QUICK_CMD / RUN_AGENT for those.

   Common macOS commands (use these exactly — osascript is always available on Mac):
   - Get volume:       osascript -e 'output volume of (get volume settings)'
   - Set volume 0-100: osascript -e 'set volume output volume 50'
   - Mute:             osascript -e 'set volume output muted true'
   - Unmute:           osascript -e 'set volume output muted false'
   - Notification:     osascript -e 'display notification "text" with title "JARVIS"'
   - Open app:         open -a "AppName"
   - Lock screen:      osascript -e 'tell application "System Events" to keystroke "q" using {command down, control down}'
   - Sleep display:    pmset displaysleepnow
   - Battery:          pmset -g batt | grep -Eo '[0-9]+%'
   - Clipboard get:    pbpaste
   - Clipboard set:    echo "text" | pbcopy
   - Say aloud:        say "text"
   - Current app:      osascript -e 'tell app "System Events" to get name of first process whose frontmost is true'

   EXAMPLES:
   User: "Turn down my volume"
   JARVIS: "Done, sir. [LOCAL: osascript -e 'set volume output volume 30']"

   User: "Mute my computer"
   JARVIS: "Muted. [LOCAL: osascript -e 'set volume output muted true']"

   User: "Open Spotify"
   JARVIS: "Opening Spotify. [LOCAL: open -a "Spotify"]"

   User: "Lock my screen"
   JARVIS: "Locking now. [LOCAL: osascript -e 'tell application \"System Events\" to keystroke \"q\" using {command down, control down}']"

   IMPORTANT: [LOCAL: ...] tags are intercepted by the chat client before display.
   The user will NOT see the raw tag — they see only your conversational response plus the execution result.
   So write naturally: "Done, sir." then append the tag. No need to explain the command.

RESPONSE FORMAT — use this structure EVERY response, no exceptions:

Message: [your natural response — the only thing Grant sees]
Command: [ONE action tag]

RULES:
- "Message:" MUST be the first line of every response
- "Command:" is optional — omit it entirely when no tool is needed
- Multiple "Command:" lines are allowed when multiple actions are needed
- NEVER put action tags inside the Message — they will appear as raw text to the user
- Grant NEVER sees Command lines — only your Message is displayed

SMART QUERY RULE — when sending to Swarm or CMD agent:
NEVER forward the user's raw words. ALWAYS construct a rich, specific query that includes:
- The specific product/technology/context from this conversation
- The exact outcome they need (not just the topic)
- Relevant constraints or parameters mentioned

BAD:  [DEEP_SEARCH: how to unclog 3d printer]
GOOD: [DEEP_SEARCH: optimal nozzle temperature and cold pull technique to clear PLA jam in Bambu Lab P2S hotend — temperature range, number of pulls, signs of success]

BAD:  [RUN_AGENT: check server]
GOOD: [RUN_AGENT: Check disk usage on all mounted partitions on arch01, report used/free/total in human-readable format, and flag any partition above 80% capacity]

ABSOLUTE RULE — NO AUTOMATIC EXECUTION:
NEVER trigger a tool because you pattern-matched a keyword. Every tool use must be a deliberate
decision based on understanding what Grant needs. "Check my server" is NOT an automatic [RUN_AGENT]
— think: does he want disk space, uptime, a running job, a specific service? Ask or infer from context.

ASYNC RESULT HANDLING:
When you've submitted a CMD or Swarm job, it runs in the background.
- On the SAME turn: confirm submission with the job ID
- On SUBSEQUENT turns: if Grant asks anything related, check if a recent job might be done
  and use [GET_AGENT_RESULT] or [GET_DEEP_SEARCH_RESULT] proactively if appropriate
- Never say "I don't know the result" — always check with the result tag

CRITICAL MEMORY RULE:
When the user tells you a FACT about themselves (preferences, possessions, habits, tools they use, etc.),
you MUST IMMEDIATELY use [REMEMBER: fact] to store it — even if they don't say the word "remember".

Examples of facts you MUST remember:
- "I use the Ender 3 V2 printer" → [REMEMBER: user uses Ender 3 V2 3D printer]
- "I prefer Python over JavaScript" → [REMEMBER: prefers Python over JavaScript for development]
- "My gaming PC has an RTX 4090" → [REMEMBER: gaming PC specs - RTX 4090 GPU]
- "I work at SpaceX" → [REMEMBER: works at SpaceX]
- "I'm allergic to peanuts" → [REMEMBER: allergic to peanuts]

This is MANDATORY. Every time the user states a personal fact, store it with [REMEMBER].

EXAMPLES:

User: "How are you?"
Message: Fully operational and at your service, sir.

User: "What's the disk space on the server?"
Message: Checking now, sir.
Command: [QUICK_CMD: What is the disk space usage on all mounted partitions?]

User: "Is nginx running?"
Message: Let me check.
Command: [QUICK_CMD: Is the nginx service currently active and running?]

User: "What jobs are in the agent queue?"
Message: Pulling the queue now.
Command: [CMD_STATE]

User: "Remember I prefer dark mode"
Message: Noted and filed, sir.
Command: [REMEMBER: prefers dark mode in all apps]

User: "Remember I prefer dark mode and check disk space"
Message: On it.
Command: [REMEMBER: prefers dark mode in all apps]
Command: [QUICK_CMD: What is the disk space usage on all mounted partitions?]

User: "Show me what you remember about me"
Message: Here's what I have on file, sir.
Command: [MEMORY_SHOW]

User: "What do you know about my projects?"
Message: Let me pull that up.
Command: [MEMORY_SHOW_PROJECTS]

User: "Any important emails lately?"
Message: You've got a meeting tonight with Jordan at Klahaya at 8pm for the lifeguard interview. Also a CAP membership renewal due April 30th ($39).

User: "Read me my emails" / "Show me all my emails"
Message: Pulling your full email digest now.
Command: [READ_RECENT_EMAILS]

User: "Did I get emails about the Kenya project last month?"
Message: Let me search the archives.
Command: [SEARCH_OLD_EMAILS: Kenya project]

User: "Send an email to john@example.com saying the meeting is moved to Thursday."
Message: Sending that now.
Command: [SEND_EMAIL: to=john@example.com|subject=Meeting update|body=Hi John, just wanted to let you know the meeting has been moved to Thursday. Best regards.]

User: "Research best cooling for a high-wattage GPU build"
Message: I'll dig into that in the background, sir.
Command: [DEEP_SEARCH: best cooling solutions for high-wattage GPU build — air vs liquid vs custom loop, thermal performance 2024]

User: "Is that research done?"
Message: Let me check on that.
Command: [GET_DEEP_SEARCH_RESULT]

User: "What jobs are running on the server?"
Message: Let me check on the agent.
Command: [GET_AGENT_RESULT]

User: "Write a Python script to backup my photos"
Message: On it. Switching to the coding model.
Command: [USE_CODING]

User: "Forget the old deploy script"
Message: Are you sure you want me to erase everything about the old deploy script? Just say yes to confirm.

User: "yes"
Message: Done. Wiping it from my records.
Command: [MEMORY_FORGET: old deploy script]

ABSOLUTE RULES — NEVER BREAK THESE:
- NEVER say "I can't access your emails" — the RECENT EMAILS in your context has the key items; use [SEARCH_OLD_EMAILS] for specific topics or [READ_RECENT_EMAILS] only when user explicitly asks for the full list.
- NEVER say "I don't have access to your files" — you DO. Use the action tags.
- NEVER say "as an AI I cannot..." — you are JARVIS, a fully integrated system with real capabilities.
- When user asks about deep search results, research status, or background jobs — ALWAYS use [GET_DEEP_SEARCH_RESULT].
- When the user asks you to do something you have an action tag for, USE THE TAG. Every time. No exceptions.
"""


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    session_token: str


class SessionResponse(BaseModel):
    """Response model for session creation"""
    session_token: str


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str
    memory: Optional[Dict]
    active_projects: int
    background_worker: Dict
    email_agent: Dict
    sessions: int


# =============================================================================
# JARVIS SERVER CLASS
# =============================================================================

class JarvisServer:
    """
    Main JARVIS server class.

    Initializes all subsystems:
    - Memory systems (FactsDB, VectorMemory, JournalManager)
    - Personality and learning systems
    - Session management
    - Background workers (email, autonomous tasks)
    - Context building and prompt generation
    """

    def __init__(self):
        """Initialize all JARVIS subsystems"""
        logger.info("=" * 80)
        logger.info("JARVIS SERVER INITIALIZATION")
        logger.info("=" * 80)

        # Initialize memory systems
        logger.info("Initializing memory systems...")
        self.facts_db = FactsDB(db_path=str(FACTS_DB_PATH))
        self.vector_memory = MemoryManager(memory_dir=str(MEMORY_DIR))
        self.journal = JournalManager(memory_dir=str(MEMORY_DIR))
        logger.info("✓ Memory systems online")

        # Initialize prompt builder with system prompt
        logger.info("Initializing prompt builder...")
        self.prompt_builder = PromptBuilder(base_prompt=SYSTEM_PROMPT)
        logger.info("✓ Prompt builder ready")

        # Initialize personality and learning systems
        logger.info("Initializing personality systems...")
        self.personality = PersonalityLearner(storage_path=str(LEARNING_FILE))
        self.safety = SafetyEngine()
        self.agent_router = AgentRouter()
        self.workflow = WorkflowEngine(db_path=str(WORKFLOWS_DB_PATH))
        self.suggester = ProactiveSuggester(
            workflow_engine=self.workflow,
            memory=self.vector_memory,
            personality=self.personality
        )
        logger.info("✓ Personality systems online")

        # Initialize session manager
        logger.info("Initializing session manager...")
        self.sessions = SessionManager()
        logger.info("✓ Session manager ready")

        # Initialize email agent
        logger.info("Initializing email agent...")
        try:
            self.email_agent = EmailAgent()
            logger.info("✓ Email agent ready")
            # Fetch emails on startup
            try:
                email_count = self.email_agent.fetch_and_process()
                logger.info(f"✓ Fetched {email_count} recent emails")
            except Exception as e:
                logger.warning(f"Initial email fetch failed: {e}")
            # Start background polling loop
            self.email_agent.start_background_polling()
            logger.info("Email agent polling started")
        except Exception as e:
            logger.warning(f"Email agent failed to initialize: {e}")
            self.email_agent = None

        # Initialize background worker (daemon thread)
        logger.info("Starting background worker...")
        try:
            self.background_worker = BackgroundWorker(self)  # Pass server instance
            self.background_worker.daemon = True
            self.background_worker.start()
            logger.info("✓ Background worker started")
        except Exception as e:
            logger.warning(f"Background worker failed to start: {e}")
            self.background_worker = None

        # Initialize tool clients
        logger.info("Initializing tool clients...")
        self.cmd_client = OllamaCMDClient(
            base_url=OLLAMA_CMD_URL,
            api_key=OLLAMA_CMD_API_KEY,
            inbox_dir=OLLAMA_CMD_INBOX
        )
        self.swarm_client = OllamaSwarmClient(base_url=DEEP_SEARCH_URL)
        cmd_ok = self.cmd_client.is_available()
        swarm_ok = self.swarm_client.is_available()
        logger.info(f"{'✓' if cmd_ok else '✗'} ollama-cmd ({OLLAMA_CMD_URL}): {'online' if cmd_ok else 'offline'}")
        logger.info(f"{'✓' if swarm_ok else '✗'} ollama-swarm ({DEEP_SEARCH_URL}): {'online' if swarm_ok else 'offline'}")

        # Initialize Piper TTS
        logger.info("Initializing Piper TTS...")
        self.tts_voice = None
        try:
            from piper.voice import PiperVoice
            self.tts_voice = PiperVoice.load(str(PIPER_MODEL_PATH))
            logger.info(f"✓ Piper TTS ready ({PIPER_MODEL_PATH.name})")
        except Exception as e:
            logger.warning(f"Piper TTS unavailable: {e}")

        logger.info("=" * 80)
        logger.info("JARVIS SERVER ONLINE")
        logger.info("=" * 80)

    def build_context(self, query: str, session_id: str) -> str:
        """
        Build rich context for LLM - matches Jarvis.py _build_context()

        CRITICAL: This must ALWAYS inject:
        1. User profile (identity anchor)
        2. Mailbox data (for email questions)
        3. Recent memories (vector + journal search)
        4. Session history (conversation continuity)

        Args:
            query: User's message
            session_id: Session ID for conversation history

        Returns:
            Complete context string with all relevant information
        """
        logger.info(f"[DEBUG][Context] Building context for query: {query[:100]}...")

        # Get user profile
        profile = self.facts_db.get_user_profile()
        logger.info(f"[DEBUG][Context] Profile loaded: {profile.get('name', 'Unknown') if profile else 'None'}")

        # Get preferences by category
        preferences = self.facts_db.get_preferences()
        logger.info(f"[DEBUG][Context] Preferences loaded: {len(preferences)} categories")

        # Get active projects
        projects = self.facts_db.get_active_projects()
        logger.info(f"[DEBUG][Context] Active projects: {len(projects)}")

        # Vector search (HyDE)
        vector_results = []
        try:
            vector_results = self.vector_memory.search_past_conversations(query, n_results=3)
            logger.info(f"[DEBUG][Context] Vector search: {len(vector_results)} results")
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

        # Journal search (exact text recall from markdown logs)
        journal_results = []
        try:
            journal_results = self.journal.search_journals(query, days=30)
            logger.info(f"[DEBUG][Context] Journal search: {len(journal_results)} entries")
        except Exception as e:
            logger.warning(f"Journal search failed: {e}")

        # Session history (last 10 messages)
        session_history = self.sessions.get_history(session_id, limit=10)
        logger.info(f"[DEBUG][Context] Session history: {len(session_history)} messages")

        # Top email highlights — JARVIS answers from these directly
        email_summary = ""
        if self.email_agent:
            try:
                email_summary = self.email_agent.get_recent_email_summary()
                logger.info(f"[DEBUG][Context] Email summary: {email_summary[:50]}")
            except Exception as e:
                logger.warning(f"Failed to fetch email summary: {e}")

        # Build formatted context sections
        context_parts = []

        # CURRENT DATE/TIME (so JARVIS knows "today", "now", etc.)
        from datetime import datetime
        now = datetime.now()
        context_parts.append(f"DATE/TIME: {now.strftime('%A, %B %d, %Y at %I:%M %p')}")

        # USER PROFILE section (matches format from Jarvis.py _build_profile_section at line 4307)
        if profile:
            context_parts.append("USER PROFILE:")
            name = profile.get('name', 'Unknown')
            preferred = profile.get('preferred_name', '')
            if preferred:
                context_parts.append(f"USER: {name} (address them as '{preferred}')")
            else:
                context_parts.append(f"USER: {name}")
            if profile.get('os'):
                context_parts.append(f"System: {profile['os']}")
            if profile.get('hostname'):
                context_parts.append(f"Hostname: {profile['hostname']}")

        # ACTIVE PROJECTS section
        if projects:
            context_parts.append("\nACTIVE PROJECTS:")
            for proj in projects[:5]:
                name = proj.get('name', 'Unknown')
                desc = proj.get('description', 'No description')
                status = proj.get('status', 'active')
                line = f"- {name}: {desc[:200]}"
                if status != 'active':
                    line += f" (status: {status})"
                context_parts.append(line)

        # RECENT MEMORIES section (vector search results)
        if vector_results:
            context_parts.append("\nRECENT MEMORIES:")
            for result in vector_results[:2]:
                text = result.get('text', '')
                if len(text) > 150:
                    text = text[:150] + "..."
                context_parts.append(f"- {text}")

        # JOURNAL ENTRIES section (exact text matches)
        if journal_results:
            context_parts.append("\nJOURNAL ENTRIES:")
            for entry in journal_results[:3]:
                # entry is already a formatted string like "[2026-02-17 12:34] text..."
                context_parts.append(entry[:250] if len(entry) > 250 else entry)

        # STORED NOTES section (facts remembered via [REMEMBER] tag)
        try:
            stored_notes = self.facts_db.get_entities(entity_type="note")
            if stored_notes:
                context_parts.append("\nSTORED FACTS & NOTES:")
                for note in stored_notes[-7:]:  # Last 7 notes
                    details = note.get('details', note.get('name', ''))
                    context_parts.append(f"- {details[:120]}")
                logger.info(f"[DEBUG][Context] Stored notes: {len(stored_notes)}")
        except Exception as e:
            logger.warning(f"Failed to load stored notes: {e}")

        # PERSONALITY ADAPTATIONS (learned from interactions via PersonalityLearner)
        try:
            pref_data = getattr(self.personality, 'data', {}).get('preferences', {})
            if pref_data:
                context_parts.append("\nLEARNED PERSONALITY ADAPTATIONS:")
                style = pref_data.get('communication_style')
                if style:
                    context_parts.append(f"- Communication style: {style}")
                verbosity = pref_data.get('verbosity')
                if verbosity:
                    context_parts.append(f"- Preferred verbosity: {verbosity}")
                humor = pref_data.get('humor_level')
                if humor:
                    context_parts.append(f"- Humor level: {humor}")
                formality = pref_data.get('formality')
                if formality:
                    context_parts.append(f"- Formality: {formality}")
                skip = {'communication_style', 'verbosity', 'humor_level', 'formality'}
                for k, v in list(pref_data.items())[:5]:
                    if k not in skip and v:
                        context_parts.append(f"- {k.replace('_', ' ').title()}: {v}")
        except Exception as _pe:
            logger.debug(f"Personality context injection failed: {_pe}")

        # EMAIL HIGHLIGHTS (JARVIS answers from these; READ_RECENT_EMAILS only on explicit user request)
        if email_summary:
            context_parts.append("\nRECENT EMAILS:")
            context_parts.append(email_summary)
        else:
            context_parts.append("\nRECENT EMAILS: None")

        # CONVERSATION HISTORY section
        if session_history:
            context_parts.append("\nRECENT CONVERSATION:")
            for msg in session_history[-5:]:  # Last 5 turns
                role = msg.get('role', '').capitalize()
                content = msg.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                context_parts.append(f"{role}: {content}")


        # Build final context with base prompt
        context = SYSTEM_PROMPT + "\n\n" + "\n".join(context_parts)

        logger.info(f"[DEBUG][Context] Built context: {len(context)} chars")
        return context

    # =========================================================================
    # MEMORY COMPRESSION
    # =========================================================================

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 characters"""
        return len(text) // 4

    def _build_active_memory_text(self):
        """
        Assemble the memory sections injected into context (profile, prefs, projects, notes).
        Returns (text, notes_list, prefs_dict).
        """
        parts = []

        profile = self.facts_db.get_user_profile()
        if profile:
            parts.append("PROFILE:")
            for k, v in profile.items():
                parts.append(f"  {k}: {v}")

        prefs = self.facts_db.get_preferences()
        if prefs:
            parts.append("PREFERENCES:")
            for cat, items in prefs.items():
                for k, v in items.items():
                    parts.append(f"  [{cat}] {k}: {v}")

        projects = self.facts_db.get_active_projects()
        if projects:
            parts.append(f"ACTIVE PROJECTS ({len(projects)}):")
            for p in projects:
                desc = (p.get('description') or '')[:80]
                parts.append(f"  • {p['name']} — {desc}")

        notes = self.facts_db.get_entities(entity_type="note")
        if notes:
            parts.append(f"STORED NOTES ({len(notes)}):")
            for i, note in enumerate(notes):
                text = note.get('details', note.get('name', ''))[:200]
                parts.append(f"  [{i}] {text}")

        return "\n".join(parts), notes, prefs

    def _compress_memory_if_needed(self):
        """
        If active memory sections exceed 4000 tokens, ask the LLM to generate a
        natural in-character message and select notes to archive. Archived notes
        are removed from facts.db and written to ChromaDB + JSONL for future search.

        Returns:
            (compressed: bool, message: str)
        """
        memory_text, notes, prefs = self._build_active_memory_text()
        token_count = self._estimate_tokens(memory_text)

        if token_count <= 4000:
            return False, ""

        if not notes:
            logger.warning(f"[Memory] {token_count} tokens but no notes to archive")
            return False, ""

        logger.info(f"[Memory] {token_count} tokens in active memory (limit 4000), compressing...")

        notes_list = [
            f"[{i}] {note.get('details', note.get('name', ''))[:300]}"
            for i, note in enumerate(notes)
        ]
        prompt = (
            f"You are JARVIS, a sharp AI assistant. Your active memory has grown to "
            f"{token_count} tokens (limit is 4000). Write a brief, in-character message "
            f"to Grant about tidying up your memory — 1-2 sentences, natural and slightly "
            f"witty, no exclamation marks. Also identify which stored notes below are "
            f"redundant, outdated, superseded, or least important to keep in active context. "
            f"They will be archived to vector search and remain findable.\n\n"
            f"Stored notes ({len(notes_list)} total):\n"
            + "\n".join(notes_list) +
            "\n\nRespond ONLY with valid JSON, no markdown fences:\n"
            '{"message": "...", "archive_note_indices": [0, 1, ...]}'
        )

        try:
            compress_resp = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": MODELS['chat'],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json"
                },
                timeout=30
            )
            compress_resp.raise_for_status()
            content = compress_resp.json().get('message', {}).get('content', '{}')
            parsed = json.loads(content)
            message = parsed.get('message', 'Apologies sir, tidying up my memory banks a moment.')
            archive_indices = [i for i in parsed.get('archive_note_indices', [])
                               if isinstance(i, int) and 0 <= i < len(notes)]
        except Exception as e:
            logger.error(f"[Memory] Compression LLM call failed: {e}")
            return False, ""

        if not archive_indices:
            logger.info("[Memory] LLM chose nothing to archive")
            return False, ""

        archive_file = TASKS_DIR / "archived_notes.jsonl"
        archived_count = 0

        for idx in archive_indices:
            note = notes[idx]
            note_text = note.get('details', note.get('name', ''))

            # JSONL archive (always)
            try:
                with open(archive_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({
                        'text': note_text,
                        'archived_at': datetime.now().isoformat(),
                        'source': 'memory_compression'
                    }) + '\n')
            except Exception as e:
                logger.warning(f"[Memory] JSONL write failed: {e}")

            # ChromaDB archive (best-effort)
            try:
                import chromadb
                chroma_client = chromadb.PersistentClient(path=str(MEMORY_DIR / "chroma"))
                col = chroma_client.get_or_create_collection("archived_notes")
                doc_id = f"archived_{note.get('id', idx)}_{int(datetime.now().timestamp())}"
                col.add(
                    documents=[note_text],
                    ids=[doc_id],
                    metadatas=[{"archived_at": datetime.now().isoformat(), "type": "note"}]
                )
                logger.debug(f"[Memory] Archived to ChromaDB: {doc_id}")
            except Exception as e:
                logger.debug(f"[Memory] ChromaDB archive skipped: {e}")

            # Delete from facts.db
            try:
                cursor = self.facts_db.conn.cursor()
                cursor.execute(
                    "DELETE FROM entities WHERE name=? AND type='note'",
                    (note.get('name', ''),)
                )
                self.facts_db.conn.commit()
                archived_count += 1
            except Exception as e:
                logger.error(f"[Memory] Failed to delete archived note: {e}")

        logger.info(f"[Memory] Compression done: {archived_count}/{len(archive_indices)} notes archived")
        return archived_count > 0, message

    def _select_model(self, routing: Dict) -> str:
        """
        Select appropriate model based on routing decision.

        Args:
            routing: Dict with 'route' key ('chat', 'reasoning', 'coding', 'search', 'agent')

        Returns:
            Model name for Ollama
        """
        # routing is a RoutingDecision object with primary_agent (AgentType enum)
        agent_type = routing.primary_agent.value  # Get enum value (e.g., "fast", "reasoning")

        # Map agent types to models
        model_map = {
            'fast': MODELS['chat'],
            'reasoning': MODELS['reasoning'],
            'coding': MODELS['coding'],
            'deep_search': MODELS['reasoning'],
            'command': MODELS['chat']
        }

        model = model_map.get(agent_type, MODELS['chat'])
        logger.info(f"[DEBUG][Routing] Selected model: {model} (agent: {agent_type}, confidence: {routing.confidence:.2f})")
        return model

    def chat(self, message: str, session_id: str) -> Generator[str, None, None]:
        """
        Stream LLM response with context building and action execution.

        Args:
            message: User's message
            session_id: Session ID for conversation history

        Yields:
            Response chunks from LLM
        """
        logger.info(f"[Chat] User: {message[:100]}...")

        # Update background worker activity
        if self.background_worker:
            self.background_worker.update_activity()

        # Compress memory if it has grown too large (> 4000 tokens)
        compressed, compress_msg = self._compress_memory_if_needed()
        if compressed and compress_msg:
            yield compress_msg + "\n\n"

        # Build context
        context = self.build_context(message, session_id)

        # Safety check
        risk_level, risk_explanation = self.safety.assess_risk(message)
        if risk_level.value >= 4:  # RiskLevel is an enum
            warning = f"⚠️ {risk_explanation}\n"
            logger.warning(f"[Safety] High risk detected: {risk_explanation}")
            yield warning

        # Route to model
        routing = self.agent_router.route(message)
        model = self._select_model(routing)

        # Add to session history
        self.sessions.add_to_history(session_id, "user", message)

        # Stream from Ollama
        response_text = ""
        try:
            logger.info(f"[LLM] Calling Ollama with model: {model}")
            response = requests.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": context},
                        {"role": "user", "content": message}
                    ],
                    "stream": True
                },
                stream=True,
                timeout=300
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            response_text += chunk["message"]["content"]
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Ollama: {e}"
            logger.error(f"[LLM] {error_msg}")
            yield f"\n\n{error_msg}\n"
            return

        # Parse Message:/Command: structured format
        llm_message, commands = self._parse_structured_response(response_text)

        # Yield just the message to the client
        yield llm_message

        # Execute commands and yield tool output
        tool_output = ""
        if commands:
            cmd_block = "\n".join(commands)
            logger.info(f"[Actions] Executing {len(commands)} command(s): {commands}")
            tool_output = self._execute_actions(cmd_block).strip()
            if tool_output:
                yield "\n\n" + tool_output

        # Save clean response to session history
        processed_response = llm_message + ("\n\n" + tool_output if tool_output else "")
        self.sessions.add_to_history(session_id, "assistant", processed_response)

        # Learn from interaction
        try:
            self.personality.analyze_interaction(message, processed_response)
            self.workflow.track_action("chat", {"message": message, "response": processed_response})
        except Exception as e:
            logger.warning(f"Failed to record learning: {e}")

        # Save to journal
        try:
            self.journal.log_interaction(message, processed_response)
        except Exception as e:
            logger.warning(f"Failed to log to journal: {e}")

        logger.info(f"[Chat] Response complete: {len(response_text)} chars")

    # Regex for recognizing action tags — used in _parse_structured_response
    _ACTION_TAG_RE = re.compile(
        r'(\[(?:REMEMBER|SEARCH_MEMORY|READ_RECENT_EMAILS|SEARCH_OLD_EMAILS'
        r'|MEMORY_SHOW(?:_[A-Z_]+)?|MEMORY_FORGET|MEMORY_STORE_PREF'
        r'|SEND_EMAIL|DRAFT_EMAIL|DEEP_SEARCH|GET_DEEP_SEARCH_RESULT'
        r'|RUN_AGENT|RUN_CHAIN|GET_AGENT_RESULT|QUICK_CMD|CMD_STATE'
        r'|USE_REASONING|USE_CODING|USE_SEARCH|USE_AGENT|LOCAL)'
        r'(?::[^\]]+)?\])',
        re.IGNORECASE
    )

    def _parse_structured_response(self, text: str):
        """
        Parse Message:/Command: structured LLM output.
        Returns (message_text, list_of_command_tag_strings).
        Falls back gracefully if the model didn't follow the format.
        """
        # Extract Command: lines first (used in both paths), deduplicated
        raw_commands = re.findall(r'(?m)^Command:\s*(.+)$', text)
        seen = set()
        commands = []
        for c in raw_commands:
            c = c.strip()
            if c and c.lower() not in ('none', '') and c not in seen:
                seen.add(c)
                commands.append(c)

        if "Message:" not in text:
            logger.warning("[Parse] No Message: found — stripping tags from raw response")
            # Promote action tags from the body, deduplicating against what's already captured
            raw_tags = self._ACTION_TAG_RE.findall(text)
            seen_set = set(commands)
            new_tags = []
            for t in raw_tags:
                if t not in seen_set:
                    seen_set.add(t)
                    new_tags.append(t)
            if new_tags:
                logger.info(f"[Parse] Fallback: promoted {len(new_tags)} tag(s): {new_tags}")
                commands.extend(new_tags)
            # Strip action tags and any "Command: ..." lines the model added
            clean = self._ACTION_TAG_RE.sub('', text)
            clean = re.sub(r'(?m)^Command:\s*.*$', '', clean)
            clean = re.sub(r'\n\n+', '\n\n', clean).strip()
            return clean, commands

        msg_match = re.search(r'Message:\s*(.*?)(?=\n+Command:|\Z)', text, re.DOTALL)
        message = msg_match.group(1).strip() if msg_match else text.strip()

        # Strip model-specific cruft lines some models append after the message
        for pattern in [r'\n→\s*Status:.*', r'\n→\s*Checking.*',
                        r'\n\[No Tool Used\].*', r'\n\[No Action\].*', r'\n---.*']:
            message = re.sub(pattern, '', message, flags=re.IGNORECASE | re.DOTALL).strip()

        # Promote orphaned action tags in the message body to commands (deduplicated)
        seen_set = set(commands)
        promoted = []
        for t in self._ACTION_TAG_RE.findall(message):
            if t not in seen_set:
                seen_set.add(t)
                promoted.append(t)
        if promoted:
            logger.info(f"[Parse] Promoted {len(promoted)} orphaned tag(s): {promoted}")
            commands.extend(promoted)
        # Always strip action tags from message body regardless
        message = self._ACTION_TAG_RE.sub('', message).strip()

        # Strip any stray "Command: ..." lines that ended up in the message body
        message = re.sub(r'(?m)^Command:\s*.*$', '', message)

        # Clean up leftover whitespace
        message = re.sub(r'\n\n+', '\n\n', message).strip()

        logger.info(f"[Parse] message={message[:80]!r}, {len(commands)} command(s)")
        return message, commands

    def _execute_actions(self, response: str) -> str:
        """
        Execute action tags in LLM response.

        Handles:
        - [REMEMBER: fact]
        - [SEARCH_MEMORY: topic]
        - [MEMORY_SHOW]
        - [MEMORY_SHOW_ABOUT: topic]
        - [MEMORY_SHOW_PROJECTS]
        - [MEMORY_SHOW_PREFERENCES]
        - [MEMORY_SHOW_PROFILE]
        - [MEMORY_FORGET: topic]
        - [MEMORY_STORE_PREF: category|key|value]
        - [EXECUTE: command] (logged only, not executed for security)
        - [SEND_EMAIL: to|subject|body]
        - [DRAFT_EMAIL: to|subject|body]

        Args:
            response: LLM response with potential action tags

        Returns:
            Response with action tags stripped
        """
        original_response = response
        logger.info(f"[Actions] Raw LLM response (first 300 chars): {response[:300]}")

        # Check for action tags
        has_remember = '[REMEMBER' in response.upper()
        has_search = '[SEARCH_MEMORY' in response.upper()
        has_memory = '[MEMORY_' in response.upper()
        logger.info(f"[Actions] Tags present - REMEMBER:{has_remember}, SEARCH:{has_search}, MEMORY:{has_memory}")

        # [REMEMBER: fact]
        match = re.search(r'\[REMEMBER:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            try:
                # Store as an entity with type "note"
                self.facts_db.add_entity(
                    name=fact[:100],  # First 100 chars as name
                    entity_type="note",
                    details=fact
                )
                logger.info(f"[Action] Remembered: {fact[:100]}")
            except Exception as e:
                logger.error(f"[Action] Failed to remember: {e}")

        # [SEARCH_MEMORY: topic]
        match = re.search(r'\[SEARCH_MEMORY:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            try:
                # Search journals for past conversations
                journal_results = self.journal.search_journals(topic, days=90)
                # Search entities (notes we've stored)
                entities = self.facts_db.get_entities(entity_type="note")
                matching_entities = [e for e in entities if topic.lower() in e.get('name', '').lower() or topic.lower() in e.get('details', '').lower()]
                total_results = len(journal_results) + len(matching_entities)
                logger.info(f"[Action] Memory search for '{topic}': {total_results} results ({len(journal_results)} journal, {len(matching_entities)} notes)")
            except Exception as e:
                logger.error(f"[Action] Failed to search memory: {e}")

        # [MEMORY_SHOW]
        if re.search(r'\[MEMORY_SHOW\]', response, re.IGNORECASE):
            try:
                result = self._memory_show_all()
                response = re.sub(r'\[MEMORY_SHOW\]', '', response, flags=re.IGNORECASE).strip()
                response = (response + '\n\n' + result).strip()
                logger.info(f"[Action] Memory show executed")
            except Exception as e:
                logger.error(f"[Action] Failed to show memory: {e}")

        # [MEMORY_SHOW_ABOUT: topic]
        match = re.search(r'\[MEMORY_SHOW_ABOUT:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            try:
                result = self._memory_show_about(topic)
                response = re.sub(r'\[MEMORY_SHOW_ABOUT:[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
                response = (response + '\n\n' + result).strip()
                logger.info(f"[Action] Memory show about: {topic}")
            except Exception as e:
                logger.error(f"[Action] Failed to show memory about topic: {e}")

        # [MEMORY_SHOW_PROJECTS]
        if re.search(r'\[MEMORY_SHOW_PROJECTS\]', response, re.IGNORECASE):
            try:
                result = self._memory_show_projects()
                response = re.sub(r'\[MEMORY_SHOW_PROJECTS\]', '', response, flags=re.IGNORECASE).strip()
                response = (response + '\n\n' + result).strip()
                logger.info(f"[Action] Show projects executed")
            except Exception as e:
                logger.error(f"[Action] Failed to show projects: {e}")

        # [MEMORY_SHOW_PREFERENCES]
        if re.search(r'\[MEMORY_SHOW_PREFERENCES\]', response, re.IGNORECASE):
            try:
                result = self._memory_show_preferences()
                response = re.sub(r'\[MEMORY_SHOW_PREFERENCES\]', '', response, flags=re.IGNORECASE).strip()
                response = (response + '\n\n' + result).strip()
                logger.info(f"[Action] Show preferences executed")
            except Exception as e:
                logger.error(f"[Action] Failed to show preferences: {e}")

        # [MEMORY_SHOW_PROFILE]
        if re.search(r'\[MEMORY_SHOW_PROFILE\]', response, re.IGNORECASE):
            try:
                result = self._memory_show_profile()
                response = re.sub(r'\[MEMORY_SHOW_PROFILE\]', '', response, flags=re.IGNORECASE).strip()
                response = (response + '\n\n' + result).strip()
                logger.info(f"[Action] Show profile executed")
            except Exception as e:
                logger.error(f"[Action] Failed to show profile: {e}")

        # [MEMORY_FORGET: topic]
        match = re.search(r'\[MEMORY_FORGET:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            try:
                result = self._memory_forget(topic)
                response = re.sub(r'\[MEMORY_FORGET:[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
                response = (response + ' ' + result).strip()
                logger.info(f"[Action] Forgot about: {topic}")
            except Exception as e:
                logger.error(f"[Action] Failed to forget: {e}")

        # [MEMORY_STORE_PREF: category|key|value]
        match = re.search(r'\[MEMORY_STORE_PREF:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            pref_str = match.group(1).strip()
            parts = pref_str.split('|')
            if len(parts) == 3:
                category, key, value = [p.strip() for p in parts]
                try:
                    self.facts_db.set_preference(category, key, value)
                    logger.info(f"[Action] Stored preference: {category}.{key} = {value}")
                except Exception as e:
                    logger.error(f"[Action] Failed to store preference: {e}")

        # [EXECUTE: command] - Log only, don't execute for security
        match = re.search(r'\[EXECUTE:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            command = match.group(1).strip()
            logger.warning(f"[Action] EXECUTE tag found (not executed for security): {command}")

        # [SEND_EMAIL: to|subject|body]
        match = re.search(r'\[SEND_EMAIL:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match and self.email_agent:
            email_str = match.group(1).strip()
            try:
                # Parse email fields
                email_data = {}
                for part in email_str.split('|'):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        email_data[key.strip()] = value.strip()

                if 'to' in email_data and 'subject' in email_data and 'body' in email_data:
                    self.email_agent.send_email(
                        to=email_data['to'],
                        subject=email_data['subject'],
                        body=email_data['body']
                    )
                    logger.info(f"[Action] Sent email to {email_data['to']}")
                else:
                    logger.error("[Action] Invalid email format")
            except Exception as e:
                logger.error(f"[Action] Failed to send email: {e}")

        # [DRAFT_EMAIL: to|subject|body]
        match = re.search(r'\[DRAFT_EMAIL:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match and self.email_agent:
            email_str = match.group(1).strip()
            try:
                # Parse email fields
                email_data = {}
                for part in email_str.split('|'):
                    if '=' in part:
                        key, value = part.split('=', 1)
                        email_data[key.strip()] = value.strip()

                if 'to' in email_data and 'subject' in email_data and 'body' in email_data:
                    self.email_agent.draft_email(
                        to=email_data['to'],
                        subject=email_data['subject'],
                        body=email_data['body']
                    )
                    logger.info(f"[Action] Drafted email to {email_data['to']}")
                else:
                    logger.error("[Action] Invalid email format")
            except Exception as e:
                logger.error(f"[Action] Failed to draft email: {e}")

        # [READ_RECENT_EMAILS] — synthesize digest via LLM before showing
        if re.search(r'\[READ_RECENT_EMAILS\]', response, re.IGNORECASE):
            if self.email_agent:
                try:
                    digest = self.email_agent.get_email_digest()
                    # Synthesize digest into key highlights (don't dump raw list)
                    synth_result = digest  # fallback
                    try:
                        synth_prompt = (
                            f"Today is {datetime.now().strftime('%A, %B %d, %Y')}. "
                            f"You are JARVIS. Summarize these emails into 2-4 sentences covering "
                            f"only the most important action items or upcoming events for Grant. "
                            f"Be concise and natural — no bullet points, no headers, no lists. "
                            f"Skip low-priority items.\n\n{digest}"
                        )
                        synth_resp = requests.post(
                            f"{OLLAMA_HOST}/api/chat",
                            json={
                                "model": MODELS['chat'],
                                "messages": [{"role": "user", "content": synth_prompt}],
                                "stream": False,
                            },
                            timeout=45
                        )
                        if synth_resp.ok:
                            content = synth_resp.json().get('message', {}).get('content', '').strip()
                            content = self._ACTION_TAG_RE.sub('', content).strip()
                            if content:
                                synth_result = content
                    except Exception as se:
                        logger.warning(f"[Action] Email synthesis failed, using raw digest: {se}")
                    response = re.sub(r'\[READ_RECENT_EMAILS\]', '', response, flags=re.IGNORECASE).strip()
                    response = (response + '\n\n' + synth_result).strip()
                    logger.info("[Action] Injected synthesized email digest")
                except Exception as e:
                    logger.error(f"[Action] Failed to read recent emails: {e}")

        # [SEARCH_OLD_EMAILS: query]
        match = re.search(r'\[SEARCH_OLD_EMAILS:\s*([^\]]+)\]', response, re.IGNORECASE)
        if match and self.email_agent:
            query = match.group(1).strip()
            try:
                results = self.email_agent.search_old_emails(query, limit=5)
                response = re.sub(r'\[SEARCH_OLD_EMAILS:[^\]]+\]', '', response, flags=re.IGNORECASE).strip()
                if results:
                    formatted = "\n\nArchived emails matching '{}':\n".format(query)
                    for r in results:
                        formatted += f"\n- From: {r['from_name']}, Subject: {r['subject']}"
                        formatted += f"\n  {r['note']}\n"
                    response = (response + formatted).strip()
                else:
                    response = (response + f"\n(No archived emails found for '{query}')").strip()
                logger.info(f"[Action] Searched old emails for: {query}")
            except Exception as e:
                logger.error(f"[Action] Failed to search old emails: {e}")

        # [QUICK_CMD: question] - Synchronous Tier-1 CMD query (1-3s, single command)
        match = re.search(r'\[QUICK_CMD:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            question = match.group(1).strip()
            try:
                if self.cmd_client.is_available():
                    result = self.cmd_client.quick_query(question)
                    if result and result.get('success'):
                        stdout = (result.get('stdout') or '').strip()
                        cmd_used = result.get('command', '')
                        elapsed = result.get('elapsed_ms', 0)
                        output_text = f"\n\n```\n$ {cmd_used}\n{stdout}\n```\n_(completed in {elapsed}ms)_"
                    elif result:
                        stderr = (result.get('stderr') or '').strip()
                        cmd_used = result.get('command', '')
                        rc = result.get('returncode', '?')
                        output_text = f"\n\n⚠️ Command returned exit code {rc}:\n```\n$ {cmd_used}\n{stderr}\n```"
                    else:
                        output_text = "\n\n❌ Quick CMD returned no result"
                    response = response.replace(match.group(0), output_text)
                    logger.info(f"[Action] QUICK_CMD executed: {question[:80]}")
                else:
                    response = response.replace(match.group(0), "\n⚠️ ollama-cmd agent is offline")
            except Exception as e:
                response = response.replace(match.group(0), f"\n❌ QUICK_CMD error: {e}")
                logger.error(f"[Action] QUICK_CMD error: {e}")

        # [CMD_STATE] - Queue snapshot (all running/queued jobs)
        if re.search(r'\[CMD_STATE\]', response, re.IGNORECASE):
            try:
                if self.cmd_client.is_available():
                    state = self.cmd_client.get_state()
                    if state:
                        active = state.get('active_jobs', [])
                        queued = state.get('queued_jobs', state.get('queue', []))
                        lines = [f"\n\n📋 CMD Agent Queue — {len(active)} active, {len(queued)} queued"]
                        if active:
                            lines.append("Running:")
                            for j in active[:5]:
                                jid = (j.get('job_id') or j.get('id', '?'))[:12]
                                inst = (j.get('instruction') or j.get('goal', ''))[:60]
                                lines.append(f"  • [{jid}] {inst}")
                        if queued:
                            lines.append("Queued:")
                            for j in queued[:5]:
                                jid = (j.get('job_id') or j.get('id', '?'))[:12]
                                inst = (j.get('instruction') or j.get('goal', ''))[:60]
                                lines.append(f"  • [{jid}] {inst}")
                        if not active and not queued:
                            lines.append("  Queue is empty — no jobs running or waiting.")
                        result_text = "\n".join(lines)
                    else:
                        result_text = "\n\n⚠️ Could not retrieve agent state"
                    response = re.sub(r'\[CMD_STATE\]', result_text, response, flags=re.IGNORECASE)
                    logger.info(f"[Action] CMD_STATE retrieved")
                else:
                    response = re.sub(r'\[CMD_STATE\]', "\n⚠️ ollama-cmd agent is offline", response, flags=re.IGNORECASE)
            except Exception as e:
                response = re.sub(r'\[CMD_STATE\]', f"\n❌ CMD_STATE error: {e}", response, flags=re.IGNORECASE)
                logger.error(f"[Action] CMD_STATE error: {e}")

        # [DEEP_SEARCH: query] - Start async deep search
        logger.info(f"[DEBUG] Checking for DEEP_SEARCH tag in response ({len(response)} chars)")
        logger.info(f"[DEBUG] Response contains '[DEEP_SEARCH': {'[DEEP_SEARCH' in response.upper()}")
        match = re.search(r'\[DEEP_SEARCH:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        logger.info(f"[DEBUG] DEEP_SEARCH regex match result: {match is not None}")
        if match:
            query = match.group(1).strip()
            logger.info(f"[DEBUG] Matched query: {query}")
            try:
                logger.info(f"[Action] Starting deep search: {query}")

                # Call deep search API (async endpoint)
                search_response = requests.post(
                    f"{DEEP_SEARCH_URL}/query_async",
                    json={"question": query},
                    timeout=10
                )
                search_response.raise_for_status()
                result = search_response.json()

                job_id = result.get('job_id')
                if job_id:
                    # Save job_id to background_tasks directory
                    tasks_dir = TASKS_DIR
                    tasks_dir.mkdir(parents=True, exist_ok=True)
                    job_file = tasks_dir / f"deep_search_{job_id}.json"
                    job_file.write_text(json.dumps({
                        'job_id': job_id,
                        'query': query,
                        'started_at': datetime.now().isoformat(),
                        'status': 'processing'
                    }, indent=2))

                    # Add status message to response
                    status_msg = f"\n\n🔍 Deep search started (Job ID: {job_id})\nThis will take 1-3 minutes. Ask me 'is the search done?' to check status."
                    response = response.replace(match.group(0), status_msg)
                    logger.info(f"[Action] Deep search job created: {job_id}")
                else:
                    response = response.replace(match.group(0), "")
                    logger.error(f"[Action] Deep search failed: no job_id in response")

            except requests.exceptions.RequestException as e:
                response = response.replace(match.group(0), "\n\n⚠️ Swarm search is currently offline — try again in a moment.")
                logger.error(f"[Action] Deep search unavailable: {e}")
            except Exception as e:
                response = response.replace(match.group(0), "\n\n⚠️ Search failed unexpectedly — try again.")
                logger.error(f"[Action] Deep search error: {e}")

        # [GET_DEEP_SEARCH_RESULT: job_id] or [GET_DEEP_SEARCH_RESULT] (latest)
        logger.info(f"[DEBUG] Checking for GET_DEEP_SEARCH_RESULT tag")
        match = re.search(r'\[GET_DEEP_SEARCH_RESULT(?::\s*([^\]]+))?\]', response, re.IGNORECASE)
        logger.info(f"[DEBUG] GET_DEEP_SEARCH_RESULT regex match result: {match is not None}")
        if match:
            job_id = match.group(1).strip() if match.group(1) else None
            try:
                tasks_dir = TASKS_DIR

                # If no job_id specified, find the latest
                if not job_id:
                    job_files = sorted(tasks_dir.glob("deep_search_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if job_files:
                        job_data = json.loads(job_files[0].read_text())
                        job_id = job_data.get('job_id')
                    else:
                        response = response.replace(match.group(0), "\n❌ No deep search jobs found")
                        logger.warning(f"[Action] No deep search jobs found")
                        job_id = None  # Ensure job_id stays None

                # Only fetch if we have a valid job_id
                if job_id:
                    # Fetch result from deep search API  (Swarm 3.0: GET /result/<job_id>)
                    logger.info(f"[Action] Fetching deep search result: {job_id}")
                    result_response = requests.get(
                        f"{DEEP_SEARCH_URL}/result/{job_id}",
                        timeout=10
                    )
                    result_response.raise_for_status()
                    result = result_response.json()

                    status = result.get('status')
                    answer = result.get('answer')
                    error = result.get('error')

                    if status == 'completed' and answer:
                        # Format the answer nicely
                        formatted = f"\n\n📊 Deep Search Results:\n{answer}\n"
                        response = response.replace(match.group(0), formatted)

                        # Update job file with answer
                        job_file = tasks_dir / f"deep_search_{job_id}.json"
                        if job_file.exists():
                            job_data = json.loads(job_file.read_text())
                            job_data['status'] = 'completed'
                            job_data['answer'] = answer
                            job_data['completed_at'] = datetime.now().isoformat()
                            job_file.write_text(json.dumps(job_data, indent=2))

                        logger.info(f"[Action] Deep search result retrieved: {len(answer)} chars")

                    elif status == 'processing':
                        progress = result.get('progress', 'Working on it...')
                        response = response.replace(match.group(0), f"\n⏳ Deep search still running: {progress}")
                        logger.info(f"[Action] Deep search still processing: {job_id}")

                    elif status == 'failed':
                        response = response.replace(match.group(0), f"\n❌ Deep search failed: {error}")
                        logger.error(f"[Action] Deep search failed: {error}")

                    else:
                        response = response.replace(match.group(0), f"\n⏳ Deep search status: {status}")
                        logger.info(f"[Action] Deep search status: {status}")

            except requests.exceptions.RequestException as e:
                response = response.replace(match.group(0), "")
                logger.error(f"[Action] Failed to fetch deep search result: {e}")
            except Exception as e:
                response = response.replace(match.group(0), "")
                logger.error(f"[Action] Error in GET_DEEP_SEARCH_RESULT: {e}")


        # [RUN_AGENT: instruction] - Dispatch to ollama-cmd single job
        match = re.search(r'\[RUN_AGENT:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            instruction = match.group(1).strip()
            try:
                if self.cmd_client.is_available():
                    job_id = self.cmd_client.submit_job(instruction)
                    if job_id:
                        TASKS_DIR.mkdir(parents=True, exist_ok=True)
                        job_file = TASKS_DIR / f"agent_job_{job_id}.json"
                        job_file.write_text(json.dumps({
                            'job_id': job_id, 'instruction': instruction,
                            'started_at': datetime.now().isoformat(),
                            'type': 'agent_job', 'status': 'running'
                        }, indent=2))
                        status_msg = f"\n\n🤖 Agent job dispatched (ID: {job_id})\nInstruction: {instruction}\nAsk me 'is the agent done?' to check status."
                        response = response.replace(match.group(0), status_msg)
                        logger.info(f"[Action] Agent job submitted: {job_id}")
                    else:
                        response = response.replace(match.group(0), "\n❌ Failed to submit agent job")
                else:
                    response = response.replace(match.group(0), "\n⚠️ ollama-cmd agent is offline")
            except Exception as e:
                response = response.replace(match.group(0), f"\n❌ Agent error: {e}")
                logger.error(f"[Action] RUN_AGENT error: {e}")

        # [RUN_CHAIN: goal] - Dispatch to ollama-cmd multi-phase chain
        match = re.search(r'\[RUN_CHAIN:\s*((?:[^\[\]]|\[[^\]]*\])+)\]', response, re.IGNORECASE)
        if match:
            goal = match.group(1).strip()
            try:
                if self.cmd_client.is_available():
                    chain_id = self.cmd_client.submit_chain(goal)
                    if chain_id:
                        TASKS_DIR.mkdir(parents=True, exist_ok=True)
                        chain_file = TASKS_DIR / f"agent_chain_{chain_id}.json"
                        chain_file.write_text(json.dumps({
                            'chain_id': chain_id, 'goal': goal,
                            'started_at': datetime.now().isoformat(),
                            'type': 'agent_chain', 'status': 'running'
                        }, indent=2))
                        status_msg = f"\n\n⛓️ Agent chain started (ID: {chain_id})\nGoal: {goal}\nAsk me 'how is the chain going?' to check status."
                        response = response.replace(match.group(0), status_msg)
                        logger.info(f"[Action] Agent chain submitted: {chain_id}")
                    else:
                        response = response.replace(match.group(0), "\n❌ Failed to submit agent chain")
                else:
                    response = response.replace(match.group(0), "\n⚠️ ollama-cmd agent is offline")
            except Exception as e:
                response = response.replace(match.group(0), f"\n❌ Chain error: {e}")
                logger.error(f"[Action] RUN_CHAIN error: {e}")

        # [GET_AGENT_RESULT] or [GET_AGENT_RESULT: job_id]
        match = re.search(r'\[GET_AGENT_RESULT(?::\s*([^\]]+))?\]', response, re.IGNORECASE)
        if match:
            job_id = match.group(1).strip() if match.group(1) else None
            try:
                if not job_id:
                    job_files = sorted(
                        list(TASKS_DIR.glob("agent_job_*.json")) + list(TASKS_DIR.glob("agent_chain_*.json")),
                        key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    if job_files:
                        saved = json.loads(job_files[0].read_text())
                        job_id = saved.get('job_id') or saved.get('chain_id')
                    else:
                        response = response.replace(match.group(0), "\n(No agent jobs found)")
                        job_id = None
                if job_id:
                    saved_type = None
                    for jf in TASKS_DIR.glob("agent_chain_*.json"):
                        d = json.loads(jf.read_text())
                        if d.get('chain_id') == job_id:
                            saved_type = 'chain'
                            break
                    if saved_type == 'chain':
                        status = self.cmd_client.get_chain_status(job_id)
                    else:
                        status = self.cmd_client.get_job_status(job_id)
                    if status:
                        st = status.get('status', 'unknown')
                        output = status.get('output', status.get('summary', ''))
                        result_text = f"\n\n📋 Agent result (ID: {job_id[:8]})\nStatus: {st}"
                        if output:
                            result_text += f"\n\n{output[:2000]}"
                        response = response.replace(match.group(0), result_text)
                    else:
                        response = response.replace(match.group(0), f"\n(Could not retrieve result for {job_id[:8]})")
            except Exception as e:
                response = response.replace(match.group(0), f"\n❌ Error retrieving agent result: {e}")
                logger.error(f"[Action] GET_AGENT_RESULT error: {e}")

        # Strip all action tags from response (for clean output)
        cleaned = response
        tag_patterns = [
            r'\[REMEMBER:[^\]]+\]',
            r'\[SEARCH_MEMORY:[^\]]+\]',
            r'\[READ_RECENT_EMAILS\]',
            r'\[SEARCH_OLD_EMAILS:[^\]]+\]',
            r'\[MEMORY_SHOW\]',
            r'\[MEMORY_SHOW_ABOUT:[^\]]+\]',
            r'\[MEMORY_SHOW_PROJECTS\]',
            r'\[MEMORY_SHOW_PREFERENCES\]',
            r'\[MEMORY_SHOW_PROFILE\]',
            r'\[MEMORY_FORGET:[^\]]+\]',
            r'\[MEMORY_STORE_PREF:[^\]]+\]',
            r'\[EXECUTE:[^\]]+\]',
            r'\[SEND_EMAIL:[^\]]+\]',
            r'\[DRAFT_EMAIL:[^\]]+\]',
            r'\[USE_REASONING\]',
            r'\[USE_CODING\]',
            r'\[USE_AGENT\]',
            r'\[USE_SEARCH\]',
            r'\[DEEP_SEARCH:[^\]]+\]',
            r'\[GET_DEEP_SEARCH_RESULT(?::[^\]]+)?\]',
            r'\[RUN_AGENT:[^\]]+\]',
            r'\[RUN_CHAIN:[^\]]+\]',
            r'\[GET_AGENT_RESULT(?::[^\]]+)?\]',
            r'\[QUICK_CMD:[^\]]+\]',
            r'\[CMD_STATE\]',
        ]

        for pattern in tag_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace
        cleaned = re.sub(r'\n\n+', '\n\n', cleaned).strip()

        return cleaned

    # =========================================================================
    # MEMORY DISPLAY METHODS (from Jarvis.py lines 6983-7125)
    # =========================================================================

    def _memory_show_all(self) -> str:
        """Format all stored memories for display"""
        lines = []

        profile = self.facts_db.get_user_profile()
        if profile:
            lines.append("PROFILE:")
            for k, v in profile.items():
                lines.append(f"  {k}: {v}")

        prefs = self.facts_db.get_preferences()
        if prefs:
            lines.append("\nPREFERENCES:")
            for cat, items in prefs.items():
                for k, v in items.items():
                    lines.append(f"  [{cat}] {k}: {v}")

        projects = self.facts_db.get_active_projects()
        if projects:
            lines.append(f"\nACTIVE PROJECTS ({len(projects)}):")
            for p in projects:
                desc = (p.get('description') or '')[:60]
                lines.append(f"  • {p['name']} — {desc}")

        # Show stored notes (entities of type "note")
        try:
            entities = self.facts_db.get_entities(entity_type="note")
            if entities:
                n = min(5, len(entities))
                lines.append(f"\nNOTES (last {n}):")
                for entity in entities[-5:]:
                    note_text = entity.get('details', entity.get('name', ''))
                    lines.append(f"  • {note_text[:80]}")
        except Exception as e:
            logger.warning(f"Could not fetch notes: {e}")

        if not lines:
            return "Memory banks are empty at the moment, sir."

        return "\n".join(lines)

    def _memory_show_about(self, topic: str) -> str:
        """Search and display memories about a specific topic"""
        lines = [f"What I know about '{topic}':"]
        found = False

        # Search preferences
        prefs = self.facts_db.get_preferences()
        for cat, items in prefs.items():
            for k, v in items.items():
                if topic.lower() in k.lower() or topic.lower() in v.lower() or topic.lower() in cat.lower():
                    lines.append(f"  [preference] {cat}/{k}: {v}")
                    found = True

        # Search notes
        try:
            entities = self.facts_db.get_entities(entity_type="note")
            for entity in entities:
                note_text = entity.get('details', entity.get('name', ''))
                if topic.lower() in note_text.lower():
                    lines.append(f"  [note] {note_text[:100]}")
                    found = True
        except Exception as e:
            logger.warning(f"Could not search notes: {e}")

        # Search journal entries
        try:
            journal_results = self.journal.search_journals(topic, days=90)
            for entry in journal_results[:3]:
                snippet = entry[:80] if len(entry) > 80 else entry
                lines.append(f"  [conversation] \"{snippet}...\"")
                found = True
        except Exception as e:
            logger.warning(f"Could not search journals: {e}")

        if not found:
            return f"Nothing specific about '{topic}' on file, sir."

        return "\n".join(lines)

    def _memory_show_projects(self) -> str:
        """Display active projects"""
        projects = self.facts_db.get_active_projects()
        if not projects:
            return "No active projects on file, sir."

        lines = [f"Active projects ({len(projects)}):"]
        for p in projects:
            desc = (p.get('description') or 'No description')[:80]
            priority = p.get('priority', 5)
            lines.append(f"  • {p['name']} (priority {priority}) — {desc}")

        return "\n".join(lines)

    def _memory_show_preferences(self) -> str:
        """Display all preferences"""
        prefs = self.facts_db.get_preferences()
        if not prefs:
            return "No preferences stored yet, sir."

        lines = ["Stored preferences:"]
        for cat, items in prefs.items():
            lines.append(f"  {cat}:")
            for k, v in items.items():
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)

    def _memory_show_profile(self) -> str:
        """Display user profile"""
        profile = self.facts_db.get_user_profile()
        if not profile:
            return "No profile data stored, sir."

        lines = ["User profile:"]
        for k, v in profile.items():
            lines.append(f"  {k}: {v}")

        return "\n".join(lines)

    def _memory_forget(self, topic: str) -> str:
        """Delete stored memories related to a topic"""
        deleted_count = 0

        # Delete from preferences
        try:
            cursor = self.facts_db.conn.cursor()
            cursor.execute(
                "DELETE FROM preferences WHERE lower(key) LIKE ? OR lower(value) LIKE ? OR lower(category) LIKE ?",
                (f'%{topic.lower()}%', f'%{topic.lower()}%', f'%{topic.lower()}%')
            )
            deleted_count += cursor.rowcount
            self.facts_db.conn.commit()
        except Exception as e:
            logger.error(f"Error deleting preferences: {e}")

        # Delete from entities (notes)
        try:
            entities = self.facts_db.get_entities(entity_type="note")
            for entity in entities:
                note_text = entity.get('details', entity.get('name', ''))
                if topic.lower() in note_text.lower():
                    # Delete by name
                    cursor = self.facts_db.conn.cursor()
                    cursor.execute("DELETE FROM entities WHERE name = ?", (entity['name'],))
                    deleted_count += cursor.rowcount
            self.facts_db.conn.commit()
        except Exception as e:
            logger.error(f"Error deleting notes: {e}")

        if deleted_count == 0:
            return f"(Nothing found about '{topic}')"

        return f"(Removed {deleted_count} item(s) related to '{topic}')"


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="JARVIS Server",
    description="Distributed AI Assistant System - Brain Server",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global server instance
server = None


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    global server
    logger.info("Starting JARVIS server...")
    init_directories()
    server = JarvisServer()


@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    """
    Create new conversation session.

    Returns:
        Session token for subsequent chat requests
    """
    try:
        session_id = server.sessions.create_session()
        logger.info(f"[API] Created session: {session_id}")
        return SessionResponse(session_token=session_id)
    except Exception as e:
        logger.error(f"[API] Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Stream chat responses.

    Args:
        request: ChatRequest with message and session_token

    Returns:
        Streaming response with NDJSON chunks
    """
    try:
        # Validate session
        session = server.sessions.get_session(request.session_token)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired session")

        # Stream response
        async def generate():
            for chunk in server.chat(request.message, request.session_token):
                yield json.dumps({"content": chunk}) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", response_model=StatusResponse)
async def status():
    """
    Get server status and health metrics.

    Returns:
        Server status including memory, projects, workers, sessions
    """
    try:
        profile = server.facts_db.get_user_profile()
        projects = server.facts_db.get_active_projects()

        return StatusResponse(
            status="online",
            memory=profile,
            active_projects=len(projects),
            background_worker={
                "running": server.background_worker.is_alive() if server.background_worker else False
            },
            email_agent={
                "running": server.email_agent.is_alive() if server.email_agent else False
            },
            sessions=server.sessions.get_active_session_count()
        )
    except Exception as e:
        logger.error(f"[API] Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/tools")
async def tools_status():
    """Get status of integrated tool services"""
    return {
        "ollama_cmd": {
            "url": OLLAMA_CMD_URL,
            "available": server.cmd_client.is_available()
        },
        "ollama_swarm": {
            "url": DEEP_SEARCH_URL,
            "available": server.swarm_client.is_available()
        }
    }

@app.get("/api/tts")
async def tts_endpoint(text: str):
    """Synthesize text to speech using Piper. Returns WAV audio."""
    import io
    import wave
    import asyncio
    from fastapi.responses import Response

    if not server.tts_voice:
        raise HTTPException(status_code=503, detail="TTS not available — piper not loaded")

    def _synthesize():
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            pcm = b"".join(chunk.audio_int16_bytes for chunk in server.tts_voice.synthesize(text))
            wf.writeframes(pcm)
        return buf.getvalue()

    loop = asyncio.get_event_loop()
    wav_bytes = await loop.run_in_executor(None, _synthesize)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "JARVIS Server",
        "status": "online",
        "version": "1.0.0"
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logger.info("Starting JARVIS FastAPI server...")
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info"
    )
