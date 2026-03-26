"""
Multi-Agent Swarm Intelligence System
Real-time collaborative AI with specialized agents
"""

import asyncio
import json
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    SEARCH = "search"
    MATH = "math"
    CODING = "coding"

class MessageType(Enum):
    TASK_CREATED = "task_created"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    QUESTION = "question"
    ANSWER = "answer"
    INFO = "info"
    ERROR = "error"

@dataclass
class Message:
    """Message passed between agents"""
    id: str
    timestamp: str
    from_agent: str
    to_agent: str
    msg_type: MessageType
    content: str
    task_id: Optional[str] = None
    priority: int = 5
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

@dataclass
class Task:
    """A task to be completed by an agent"""
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None
    dependencies: List[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now().strftime("%H:%M:%S")
        if self.dependencies is None:
            self.dependencies = []


class SharedMemory:
    """Shared memory pool accessible by all agents"""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.tasks: Dict[str, Task] = {}
        self.facts: Dict[str, any] = {}
        self.research: Dict[str, List[str]] = {}
        self.agent_status: Dict[str, str] = {}
        
    def add_message(self, message: Message):
        """Add a message to the log"""
        self.messages.append(message)
        
    def add_task(self, task: Task):
        """Add a task to the pool"""
        self.tasks[task.id] = task
        
    def update_task(self, task_id: str, **kwargs):
        """Update task properties"""
        if task_id in self.tasks:
            for key, value in kwargs.items():
                setattr(self.tasks[task_id], key, value)
    
    def add_fact(self, key: str, value: any):
        """Store a fact"""
        self.facts[key] = value
        
    def get_fact(self, key: str) -> any:
        """Retrieve a fact"""
        return self.facts.get(key)
    
    def add_research(self, agent: str, findings: List[str]):
        """Store research findings"""
        if agent not in self.research:
            self.research[agent] = []
        self.research[agent].extend(findings)
    
    def set_agent_status(self, agent: str, status: str):
        """Update agent status"""
        self.agent_status[agent] = status
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks"""
        return [t for t in self.tasks.values() if t.status == "pending"]
    
    def get_recent_messages(self, n: int = 10) -> List[Message]:
        """Get recent messages"""
        return self.messages[-n:]


class MessageBus:
    """Central message bus for agent communication"""
    
    def __init__(self, memory: SharedMemory):
        self.memory = memory
        self.subscribers: Dict[str, asyncio.Queue] = {}
        
    def subscribe(self, agent_id: str) -> asyncio.Queue:
        """Subscribe an agent to receive messages"""
        queue = asyncio.Queue()
        self.subscribers[agent_id] = queue
        return queue
    
    async def publish(self, message: Message):
        """Publish a message to relevant subscribers"""
        self.memory.add_message(message)
        
        # Display message in real-time
        self._display_message(message)
        
        # Route to specific agent or broadcast
        if message.to_agent == "all":
            for queue in self.subscribers.values():
                await queue.put(message)
        elif message.to_agent in self.subscribers:
            await self.subscribers[message.to_agent].put(message)
    
    def _display_message(self, msg: Message):
        """Display message in real-time"""
        icons = {
            MessageType.TASK_CREATED: "📋",
            MessageType.TASK_ASSIGNED: "👷",
            MessageType.TASK_COMPLETED: "✅",
            MessageType.QUESTION: "❓",
            MessageType.ANSWER: "💡",
            MessageType.INFO: "ℹ️",
            MessageType.ERROR: "❌"
        }
        
        icon = icons.get(msg.msg_type, "📨")
        print(f"{icon} [{msg.timestamp}] {msg.from_agent} → {msg.to_agent}: {msg.content[:80]}")


# Export shared instances
memory = SharedMemory()
message_bus = MessageBus(memory)
