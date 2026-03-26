#!/usr/bin/env python3
"""
patch_main.py — Patches main.py for the new package structure.
Run from /mnt/storage/NAS/Jarvis/Jarvis/

Usage: python3 patch_main.py
"""
import re
from pathlib import Path

SRC = Path("server/main.py")
print(f"Patching {SRC}...")
text = SRC.read_text()
original_len = len(text)

# ── 1. Fix sys.path insert ────────────────────────────────────────────────────
text = text.replace(
    "sys.path.insert(0, str(Path(__file__).parent))",
    "sys.path.insert(0, str(Path(__file__).parent.parent))"
)

# ── 2. Replace local import block ────────────────────────────────────────────
old_imports = """from jarvis_memory_system import FactsDB, MemoryManager, JournalManager
from prompt_builder import PromptBuilder
from agent_router import AgentRouter
from email_agent import EmailAgent
from personality_learner import PersonalityLearner
from safety_engine import SafetyEngine
from workflow_engine import WorkflowEngine
from proactive_suggester import ProactiveSuggester
from session_manager import SessionManager
from background_worker import BackgroundWorker
from server_config import (
    SERVER_HOST, SERVER_PORT, OLLAMA_HOST, MODELS, CORS_ORIGINS,
    DEEP_SEARCH_URL,
    MEMORY_DIR, FACTS_DB_PATH, WORKFLOWS_DB_PATH, LEARNING_FILE,
    CONTEXT_BUDGET, MAX_HISTORY_MESSAGES,
    init_directories
)"""

new_imports = """from core.jarvis_memory_system import FactsDB, MemoryManager, JournalManager
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
    TASKS_DIR, RECENT_EMAILS_FILE,
    CONTEXT_BUDGET, MAX_HISTORY_MESSAGES,
    init_directories
)"""

if old_imports in text:
    text = text.replace(old_imports, new_imports)
    print("  ✓ Import block updated")
else:
    print("  ✗ WARNING: Could not find import block to replace")

# ── 3. Add agent capabilities to SYSTEM_PROMPT ───────────────────────────────
new_section = """
9. AGENT TASKS (dispatch to the autonomous agent on arch01):
   [RUN_AGENT: instruction] - Run a single task/command on the autonomous agent
   [RUN_CHAIN: goal] - Run a complex multi-phase goal on the autonomous agent

   Use [RUN_AGENT] for: server commands, system checks, file operations, script execution
   Use [RUN_CHAIN] for: complex multi-step goals that need planning (deploy, setup service, etc.)
   Use [GET_AGENT_RESULT] or [GET_AGENT_RESULT: job_id] to check results

   Examples:
   User: "Check disk space on the server" → [RUN_AGENT: check disk space and report usage on all mounts]
   User: "Set up nginx on the server" → [RUN_CHAIN: Set up and configure nginx with a default site]

"""

marker = "ABSOLUTE RULES — NEVER BREAK THESE:"
if marker in text:
    text = text.replace(marker, new_section + marker)
    print("  ✓ SYSTEM_PROMPT updated with agent action tags")
else:
    print("  ✗ WARNING: Could not find ABSOLUTE RULES marker")

# ── 4. Add tool client initialization in __init__ ────────────────────────────
old_init_end = """        logger.info("=" * 80)
        logger.info("JARVIS SERVER ONLINE")"""

new_init_end = """        # Initialize tool clients
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

        logger.info("=" * 80)
        logger.info("JARVIS SERVER ONLINE")"""

if old_init_end in text:
    text = text.replace(old_init_end, new_init_end)
    print("  ✓ Tool clients added to __init__")
else:
    print("  ✗ WARNING: Could not find __init__ end marker")

# ── 5. Fix hardcoded recent_file path ────────────────────────────────────────
old_recent = 'recent_file = Path(__file__).parent.parent / "jarvis_memory" / "recent-emails.md"'
new_recent = 'recent_file = RECENT_EMAILS_FILE'
if old_recent in text:
    text = text.replace(old_recent, new_recent)
    print("  ✓ recent_file path fixed")
else:
    print("  ✗ WARNING: Could not find recent_file path")

# ── 6. Fix hardcoded tasks_dir paths ─────────────────────────────────────────
old_tasks = 'tasks_dir = Path(__file__).parent.parent / "jarvis_memory" / "background_tasks"'
new_tasks = 'tasks_dir = TASKS_DIR'
count = text.count(old_tasks)
if count > 0:
    text = text.replace(old_tasks, new_tasks)
    print(f"  ✓ tasks_dir path fixed ({count} occurrences)")
else:
    print("  ✗ WARNING: Could not find tasks_dir path")

# ── 7. Add new action tag handlers before cleanup section ────────────────────
new_handlers = '''
        # [RUN_AGENT: instruction] - Dispatch to ollama-cmd single job
        match = re.search(r\'\\[RUN_AGENT:\\s*([^\\]]+)\\]\', response, re.IGNORECASE)
        if match:
            instruction = match.group(1).strip()
            try:
                if self.cmd_client.is_available():
                    job_id = self.cmd_client.submit_job(instruction)
                    if job_id:
                        TASKS_DIR.mkdir(parents=True, exist_ok=True)
                        job_file = TASKS_DIR / f"agent_job_{job_id}.json"
                        job_file.write_text(json.dumps({
                            \'job_id\': job_id, \'instruction\': instruction,
                            \'started_at\': datetime.now().isoformat(),
                            \'type\': \'agent_job\', \'status\': \'running\'
                        }, indent=2))
                        status_msg = f"\\n\\n🤖 Agent job dispatched (ID: {job_id})\\nInstruction: {instruction}\\nAsk me \'is the agent done?\' to check status."
                        response = response.replace(match.group(0), status_msg)
                        logger.info(f"[Action] Agent job submitted: {job_id}")
                    else:
                        response = response.replace(match.group(0), "\\n❌ Failed to submit agent job")
                else:
                    response = response.replace(match.group(0), "\\n⚠️ ollama-cmd agent is offline")
            except Exception as e:
                response = response.replace(match.group(0), f"\\n❌ Agent error: {e}")
                logger.error(f"[Action] RUN_AGENT error: {e}")

        # [RUN_CHAIN: goal] - Dispatch to ollama-cmd multi-phase chain
        match = re.search(r\'\\[RUN_CHAIN:\\s*([^\\]]+)\\]\', response, re.IGNORECASE)
        if match:
            goal = match.group(1).strip()
            try:
                if self.cmd_client.is_available():
                    chain_id = self.cmd_client.submit_chain(goal)
                    if chain_id:
                        TASKS_DIR.mkdir(parents=True, exist_ok=True)
                        chain_file = TASKS_DIR / f"agent_chain_{chain_id}.json"
                        chain_file.write_text(json.dumps({
                            \'chain_id\': chain_id, \'goal\': goal,
                            \'started_at\': datetime.now().isoformat(),
                            \'type\': \'agent_chain\', \'status\': \'running\'
                        }, indent=2))
                        status_msg = f"\\n\\n⛓️ Agent chain started (ID: {chain_id})\\nGoal: {goal}\\nAsk me \'how is the chain going?\' to check status."
                        response = response.replace(match.group(0), status_msg)
                        logger.info(f"[Action] Agent chain submitted: {chain_id}")
                    else:
                        response = response.replace(match.group(0), "\\n❌ Failed to submit agent chain")
                else:
                    response = response.replace(match.group(0), "\\n⚠️ ollama-cmd agent is offline")
            except Exception as e:
                response = response.replace(match.group(0), f"\\n❌ Chain error: {e}")
                logger.error(f"[Action] RUN_CHAIN error: {e}")

        # [GET_AGENT_RESULT] or [GET_AGENT_RESULT: job_id]
        match = re.search(r\'\\[GET_AGENT_RESULT(?::\\s*([^\\]]+))?\\]\', response, re.IGNORECASE)
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
                        job_id = saved.get(\'job_id\') or saved.get(\'chain_id\')
                    else:
                        response = response.replace(match.group(0), "\\n(No agent jobs found)")
                        job_id = None
                if job_id:
                    saved_type = None
                    for jf in TASKS_DIR.glob("agent_chain_*.json"):
                        d = json.loads(jf.read_text())
                        if d.get(\'chain_id\') == job_id:
                            saved_type = \'chain\'
                            break
                    if saved_type == \'chain\':
                        status = self.cmd_client.get_chain_status(job_id)
                    else:
                        status = self.cmd_client.get_job_status(job_id)
                    if status:
                        st = status.get(\'status\', \'unknown\')
                        output = status.get(\'output\', status.get(\'summary\', \'\'))
                        result_text = f"\\n\\n📋 Agent result (ID: {job_id[:8]})\\nStatus: {st}"
                        if output:
                            result_text += f"\\n\\n{output[:2000]}"
                        response = response.replace(match.group(0), result_text)
                    else:
                        response = response.replace(match.group(0), f"\\n(Could not retrieve result for {job_id[:8]})")
            except Exception as e:
                response = response.replace(match.group(0), f"\\n❌ Error retrieving agent result: {e}")
                logger.error(f"[Action] GET_AGENT_RESULT error: {e}")

'''

cleanup_marker = "        # Strip all action tags from response (for clean output)"
if cleanup_marker in text:
    text = text.replace(cleanup_marker, new_handlers + cleanup_marker)
    print("  ✓ New action tag handlers inserted")
else:
    print("  ✗ WARNING: Could not find cleanup section marker")

# ── 8. Add new tags to cleanup list ─────────────────────────────────────────
old_cleanup_end = "            r'\\[GET_DEEP_SEARCH_RESULT(?::[^\\]]+)?\\]',"
new_cleanup_end = """            r'\\[GET_DEEP_SEARCH_RESULT(?::[^\\]]+)?\\]',
            r'\\[RUN_AGENT:[^\\]]+\\]',
            r'\\[RUN_CHAIN:[^\\]]+\\]',
            r'\\[GET_AGENT_RESULT(?::[^\\]]+)?\\]',"""
if old_cleanup_end in text:
    text = text.replace(old_cleanup_end, new_cleanup_end)
    print("  ✓ Cleanup tag list updated")
else:
    print("  ✗ WARNING: Could not find cleanup tag list end")

# ── 9. Add /api/tools endpoint after /api/status ────────────────────────────
tools_endpoint = '''

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

'''

root_marker = "@app.get(\"/\")"
if root_marker in text:
    text = text.replace(root_marker, tools_endpoint + root_marker)
    print("  ✓ /api/tools endpoint added")
else:
    print("  ✗ WARNING: Could not find root endpoint marker")

# ── Write result ─────────────────────────────────────────────────────────────
SRC.write_text(text)
print(f"\nDone. {original_len} → {len(text)} chars")
print("Patch complete.")
