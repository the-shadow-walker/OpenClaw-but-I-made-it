"""
Agent Router - Intelligent Task Distribution
============================================
Routes user requests to the appropriate agent based on context analysis.

Agents:
- FAST_MODEL (phi4:14b): Quick responses, simple queries
- REASONING_MODEL (qwen3:30b): Complex analysis, thinking through problems
- CODING_MODEL (qwen3-coder:30b): Code generation, debugging, technical tasks
- COMMAND_AGENT: Shell commands, system operations
- DEEP_SEARCH_AGENT: Research, web search, multi-step information gathering

Features:
- Context-aware routing
- Multi-agent tasks (can use multiple agents)
- Async task handling
- Smart prompt generation for each agent
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class AgentType(Enum):
    """Available agent types"""
    FAST = "fast"                    # Quick responses
    REASONING = "reasoning"          # Deep thinking
    CODING = "coding"               # Code tasks
    COMMAND = "command"             # Shell/system commands
    DEEP_SEARCH = "deep_search"     # Research/web search


@dataclass
class RoutingDecision:
    """Result of routing analysis"""
    primary_agent: AgentType
    secondary_agents: List[AgentType]
    confidence: float
    reason: str
    async_execution: bool
    estimated_time: str
    prompt_modifications: Dict


class AgentRouter:
    """
    Intelligently routes tasks to appropriate agents.
    
    Analyzes user requests and determines:
    1. Which agent(s) to use
    2. Whether to execute asynchronously
    3. How to modify prompts for each agent
    """
    
    # Routing patterns - what agent for what type of query
    ROUTING_PATTERNS = {
        AgentType.COMMAND: {
            'keywords': [
                'run', 'execute', 'start', 'stop', 'restart', 'kill',
                'deploy', 'build', 'compile', 'install', 'update',
                'delete', 'remove', 'move', 'copy', 'create directory',
                'git ', 'docker ', 'kubectl ', 'npm ', 'pip ',
                'ssh', 'scp', 'rsync', 'curl', 'wget',
                'check status', 'show processes', 'list files',
                'kill process', 'terminate', 'service '
            ],
            'contexts': ['system', 'shell', 'terminal', 'server', 'process'],
            'priority': 10  # High priority - commands should go to command agent
        },
        
        AgentType.DEEP_SEARCH: {
            'keywords': [
                'deep search', 'research', 'look up', 'find information about',
                'what is the latest', 'current news', 'compare', 'best options',
                'search the web', 'find me', 'look into', 'investigate',
                'comprehensive', 'detailed research', 'market research',
                'lightest but', 'strongest but', 'most efficient',
                'bullet proof', 'what are the options', 'alternatives',
                'use the swarm', 'swarm it', 'swarm this', 'send to swarm',
                'use swarm', 'run through swarm', 'swarm:'
            ],
            'contexts': ['research', 'web', 'internet', 'search', 'comparison'],
            'priority': 8
        },
        
        AgentType.CODING: {
            'keywords': [
                'code', 'function', 'class', 'method', 'variable',
                'debug', 'error', 'exception', 'fix this code',
                'write a script', 'program', 'api', 'endpoint',
                'refactor', 'optimize', 'implement', 'algorithm',
                'python', 'javascript', 'typescript', 'rust', 'go',
                'sql', 'query', 'database', 'schema',
                'test', 'unit test', 'integration test'
            ],
            'contexts': ['code', 'programming', 'development', 'debugging'],
            'priority': 7
        },
        
        AgentType.REASONING: {
            'keywords': [
                'think about', 'analyze', 'consider', 'evaluate',
                'why', 'explain why', 'reasoning', 'logic',
                'pros and cons', 'trade-offs', 'implications',
                'this requires careful', 'let me think', 'complex',
                'strategy', 'plan', 'approach', 'decision',
                'compare and contrast', 'weigh the options',
                'what would happen if', 'consequences'
            ],
            'contexts': ['analysis', 'reasoning', 'thinking', 'planning'],
            'priority': 6
        },
        
        AgentType.FAST: {
            'keywords': [
                'quick', 'briefly', 'short', 'simple',
                'what is', 'who is', 'when did', 'where is',
                'remind me', 'set a timer', 'alarm',
                'play music', 'pause', 'skip', 'volume'
            ],
            'contexts': ['quick', 'simple', 'basic'],
            'priority': 1  # Default fallback
        }
    }
    
    # Time estimation patterns
    TIME_ESTIMATES = {
        AgentType.DEEP_SEARCH: "1-5 minutes",
        AgentType.REASONING: "30 seconds - 2 minutes",
        AgentType.CODING: "30 seconds - 3 minutes",
        AgentType.COMMAND: "varies based on command",
        AgentType.FAST: "instant"
    }
    
    # Async execution patterns
    ASYNC_INDICATORS = [
        'deep search', 'research', 'comprehensive',
        'will take', 'might take', 'long time',
        'in the background', 'while', 'meanwhile'
    ]
    
    def __init__(self):
        self.routing_history = []
        self.user_preferences = {}  # Learn preferred agents for certain tasks
    
    def route(self, user_message: str, context: Dict = None) -> RoutingDecision:
        """
        Analyze message and determine routing.
        
        Args:
            user_message: The user's request
            context: Additional context (recent actions, active project, etc.)
        
        Returns:
            RoutingDecision with agent selection and parameters
        """
        message_lower = user_message.lower()

        # Hard override: explicit swarm/deep-research request → always DEEP_SEARCH
        swarm_triggers = ['use the swarm', 'swarm it', 'swarm this', 'send to swarm',
                          'use swarm', 'run through swarm', 'swarm:']
        if any(t in message_lower for t in swarm_triggers):
            return RoutingDecision(
                primary_agent=AgentType.DEEP_SEARCH,
                secondary_agents=[AgentType.REASONING],
                confidence=1.0,
                reason="Explicit swarm request",
                async_execution=True,
                estimated_time="1-5 minutes",
                prompt_modifications=self._generate_prompt_modifications(
                    AgentType.DEEP_SEARCH, user_message, context
                )
            )

        # Score each agent
        scores = {}
        reasons = {}
        
        for agent_type, patterns in self.ROUTING_PATTERNS.items():
            score, reason = self._score_agent(message_lower, patterns, context)
            scores[agent_type] = score
            reasons[agent_type] = reason
        
        # Get highest scoring agent
        primary_agent = max(scores, key=scores.get)
        confidence = scores[primary_agent] / 10.0  # Normalize to 0-1
        
        # Get secondary agents (if score is close to primary)
        secondary_agents = []
        for agent, score in scores.items():
            if agent != primary_agent and score >= scores[primary_agent] * 0.6:
                secondary_agents.append(agent)
        
        # Determine if async execution
        async_exec = self._should_execute_async(message_lower, primary_agent)
        
        # Estimate time
        estimated_time = self.TIME_ESTIMATES.get(primary_agent, "varies")
        
        # Generate prompt modifications
        prompt_mods = self._generate_prompt_modifications(
            primary_agent, user_message, context
        )
        
        # Create decision
        decision = RoutingDecision(
            primary_agent=primary_agent,
            secondary_agents=secondary_agents,
            confidence=confidence,
            reason=reasons[primary_agent],
            async_execution=async_exec,
            estimated_time=estimated_time,
            prompt_modifications=prompt_mods
        )
        
        # Record for learning
        self.routing_history.append({
            'message': user_message,
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        })
        
        return decision
    
    def _score_agent(self, message: str, patterns: Dict, context: Dict) -> Tuple[float, str]:
        """Score how well an agent matches the request"""
        score = 0.0
        reason = ""
        
        # Check keywords
        for keyword in patterns['keywords']:
            if keyword in message:
                score += patterns['priority']
                if not reason:
                    reason = f"Detected '{keyword}'"
                else:
                    reason += f", '{keyword}'"
        
        # Check contexts
        if context:
            for ctx in patterns['contexts']:
                if context.get('type') == ctx or context.get('domain') == ctx:
                    score += patterns['priority'] * 0.5
        
        # Check for explicit agent request
        if 'use' in message and any(k in message for k in patterns['keywords'][:5]):
            score += patterns['priority'] * 1.5
            reason = "Explicit agent request"
        
        return min(score, 10.0), reason
    
    def _should_execute_async(self, message: str, agent: AgentType) -> bool:
        """Determine if task should run asynchronously"""
        # Deep search is always async
        if agent == AgentType.DEEP_SEARCH:
            return True
        
        # Check for async indicators
        for indicator in self.ASYNC_INDICATORS:
            if indicator in message:
                return True
        
        # Complex reasoning tasks
        if agent == AgentType.REASONING and 'complex' in message:
            return True
        
        return False
    
    def _generate_prompt_modifications(self, agent: AgentType, 
                                       message: str, context: Dict) -> Dict:
        """Generate modifications for the agent's prompt"""
        mods = {
            'system_addition': '',
            'context_injection': '',
            'output_format': ''
        }
        
        if agent == AgentType.DEEP_SEARCH:
            mods['system_addition'] = (
                "You are performing a deep search. Be thorough and comprehensive. "
                "Cite sources when possible. Structure your findings clearly."
            )
            mods['output_format'] = "structured_findings"
            
        elif agent == AgentType.REASONING:
            mods['system_addition'] = (
                "Think through this step by step. Consider multiple perspectives. "
                "Explain your reasoning clearly."
            )
            mods['output_format'] = "reasoned_analysis"
            
        elif agent == AgentType.CODING:
            mods['system_addition'] = (
                "You are a coding assistant. Provide clean, well-documented code. "
                "Explain key decisions in comments."
            )
            mods['output_format'] = "code_with_explanation"
            
        elif agent == AgentType.COMMAND:
            mods['system_addition'] = (
                "You are executing a system command. Be precise and safe. "
                "Show command output clearly."
            )
            mods['output_format'] = "command_result"
        
        return mods
    
    def build_agent_prompt(self, agent: AgentType, user_message: str,
                          base_context: str, modifications: Dict) -> str:
        """Build the full prompt for an agent"""
        parts = [base_context]
        
        if modifications.get('system_addition'):
            parts.append(f"\nINSTRUCTIONS: {modifications['system_addition']}")
        
        parts.append(f"\nUSER REQUEST: {user_message}")
        
        return "\n".join(parts)
    
    def get_response_prefix(self, agent: AgentType, async_mode: bool) -> str:
        """Get the appropriate response prefix based on agent and mode"""
        if async_mode:
            prefixes = {
                AgentType.DEEP_SEARCH: "🔄 Initiating deep search... This may take a minute, sir. In the meantime, is there anything else I can help with?",
                AgentType.REASONING: "🧠 Routing to reasoning model (qwen3:30b)... This requires careful consideration. I'll have an answer shortly.",
                AgentType.CODING: "💻 Analyzing code requirements... This may take a moment.",
                AgentType.COMMAND: "⚡ Executing command... I'll report back with results.",
                AgentType.FAST: "Processing..."
            }
        else:
            prefixes = {
                AgentType.DEEP_SEARCH: "🔍 Searching...",
                AgentType.REASONING: "🧠 Thinking...",
                AgentType.CODING: "💻 Working on code...",
                AgentType.COMMAND: "⚡ Executing...",
                AgentType.FAST: ""
            }
        
        return prefixes.get(agent, "")
    
    def learn_from_feedback(self, message: str, agent_used: AgentType, 
                           user_satisfied: bool):
        """Learn from user feedback to improve routing"""
        # Store feedback for future routing decisions
        key = self._extract_key_intent(message)
        
        if key:
            if key not in self.user_preferences:
                self.user_preferences[key] = {}
            
            if agent_used.value not in self.user_preferences[key]:
                self.user_preferences[key][agent_used.value] = {'positive': 0, 'negative': 0}
            
            if user_satisfied:
                self.user_preferences[key][agent_used.value]['positive'] += 1
            else:
                self.user_preferences[key][agent_used.value]['negative'] += 1
    
    def _extract_key_intent(self, message: str) -> Optional[str]:
        """Extract key intent from message for learning"""
        # Simple intent extraction
        verbs = re.findall(r'\b(search|find|run|execute|code|debug|explain|analyze)\b', 
                          message.lower())
        if verbs:
            return verbs[0]
        return None
    
    def get_statistics(self) -> Dict:
        """Get routing statistics"""
        if not self.routing_history:
            return {'total_routed': 0}
        
        agent_counts = {}
        for entry in self.routing_history:
            agent = entry['decision'].primary_agent.value
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        return {
            'total_routed': len(self.routing_history),
            'by_agent': agent_counts,
            'preferences_learned': len(self.user_preferences)
        }


# Multi-agent task coordination
class MultiAgentTask:
    """Coordinates tasks that require multiple agents"""
    
    def __init__(self, router: AgentRouter):
        self.router = router
        self.active_tasks = {}
    
    def create_task(self, user_message: str, context: Dict = None) -> Dict:
        """Create a multi-agent task"""
        decision = self.router.route(user_message, context)
        
        task = {
            'id': f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'message': user_message,
            'primary_agent': decision.primary_agent,
            'secondary_agents': decision.secondary_agents,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'results': {}
        }
        
        self.active_tasks[task['id']] = task
        return task
    
    def update_result(self, task_id: str, agent: AgentType, result: str):
        """Update task with agent result"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['results'][agent.value] = result
    
    def complete_task(self, task_id: str) -> Dict:
        """Mark task as complete and return results"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]['status'] = 'completed'
            return self.active_tasks.pop(task_id)
        return None


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("AGENT ROUTER EXAMPLE")
    print("="*70)
    
    router = AgentRouter()
    
    # Test routing
    test_messages = [
        "Can you run a deep search on the lightest, but still most bullet proof material?",
        "Think through the best approach for scaling our microservices",
        "Fix this Python code that's throwing an exception",
        "What's the weather today?",
        "Deploy the new version to production",
        "Execute a vulnerability scan on the server",
        "Write a function to parse JSON data",
        "What are the pros and cons of using Kubernetes vs Docker Swarm?",
    ]
    
    for msg in test_messages:
        print(f"\n🔍 Message: {msg}")
        decision = router.route(msg)
        print(f"   Agent: {decision.primary_agent.value}")
        print(f"   Confidence: {decision.confidence:.0%}")
        print(f"   Reason: {decision.reason}")
        print(f"   Async: {decision.async_execution}")
        print(f"   Est. Time: {decision.estimated_time}")
        print(f"   Response: {router.get_response_prefix(decision.primary_agent, decision.async_execution)}")
    
    print("\n✅ Agent Router ready for integration")
