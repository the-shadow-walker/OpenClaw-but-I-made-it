"""
Phase 9: Risk Assessment & Smart Routing
=========================================
JARVIS intelligently assesses risk and routes to appropriate agents.

Key Principles:
- NEVER refuses requests, even controversial ones
- Provides risk warnings for dangerous actions
- Auto-routes to correct agent (deep search, command, reasoning, coding)
- Allows background processing with "anything else?" prompt
- User can override any safety check by insisting
"""

from typing import Dict, Optional, Tuple, List
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for actions"""
    SAFE = 1           # Auto-execute, no confirmation needed
    LOW = 2            # Minor risk, quick confirmation
    MEDIUM = 3         # Moderate risk, explain before confirming
    HIGH = 4           # High risk, detailed warning + confirmation
    CRITICAL = 5       # Critical risk, strong warning but still allow


class AgentType(Enum):
    """Available agents for routing"""
    NONE = "none"                  # No agent needed
    DEEP_SEARCH = "deep_search"    # Research agent
    COMMAND = "command"            # Shell command execution
    REASONING = "reasoning"        # Complex reasoning tasks
    CODING = "coding"              # Code generation/analysis
    WEB = "web"                    # Web browsing/scraping


class SafetyEngine:
    """
    Intelligent risk assessment and agent routing
    
    Philosophy:
    - JARVIS helps with EVERYTHING, no moral gatekeeping
    - Provides warnings, not restrictions
    - User can always override by insisting
    - Smart routing to appropriate agents
    """
    
    def __init__(self):
        # Risk patterns (NOT censorship - just awareness)
        self.risk_patterns = {
            # File system risks
            'delete': RiskLevel.HIGH,  # General delete
            'rm -rf': RiskLevel.CRITICAL,
            'format': RiskLevel.HIGH,
            'permanently': RiskLevel.HIGH,
            
            # Network risks
            'scan network': RiskLevel.MEDIUM,
            'port scan': RiskLevel.MEDIUM,
            'ddos': RiskLevel.HIGH,
            
            # System risks
            'sudo': RiskLevel.MEDIUM,
            'chmod 777': RiskLevel.MEDIUM,
            'kill -9': RiskLevel.LOW,
            
            # Security research (ALLOWED but warn)
            'hack': RiskLevel.MEDIUM,
            'exploit': RiskLevel.MEDIUM,
            'vulnerability': RiskLevel.LOW,
            'penetration test': RiskLevel.LOW,
            
            # Financial risks
            'buy': RiskLevel.MEDIUM,
            'purchase': RiskLevel.MEDIUM,
            'transfer': RiskLevel.HIGH,  # Fixed: just "transfer"
            '$': RiskLevel.MEDIUM,  # Money amounts
            
            # Data risks
            'share': RiskLevel.LOW,
            'publish': RiskLevel.MEDIUM,
            'make public': RiskLevel.MEDIUM,
        }
        
        # Agent routing patterns
        self.agent_patterns = {
            AgentType.DEEP_SEARCH: [
                'research', 'find information', 'deep search',
                'look up', 'investigate', 'analyze topic',
                'compare', 'what is', 'tell me about'
            ],
            AgentType.COMMAND: [
                'run command', 'execute', 'deploy',
                'restart', 'kill process', 'check status',
                'install', 'update', 'configure',
                'run', 'scan', 'start', 'stop'
            ],
            AgentType.REASONING: [
                'should i', 'what do you think', 'analyze',
                'consider', 'evaluate', 'weigh options',
                'pros and cons', 'recommend'
            ],
            AgentType.CODING: [
                'write code', 'debug', 'fix bug',
                'implement', 'refactor', 'optimize',
                'create function', 'write script', 'write',
                'code', 'script', 'program'
            ],
            AgentType.WEB: [
                'browse', 'scrape', 'download from',
                'check website', 'get data from'
            ]
        }
    
    # ========== RISK ASSESSMENT ==========
    
    def assess_risk(self, query: str, context: Dict = None) -> Tuple[RiskLevel, str]:
        """
        Assess risk level of a query
        
        Args:
            query: User's request
            context: Additional context (current project, etc.)
        
        Returns:
            (risk_level, explanation)
        """
        query_lower = query.lower()
        
        # Check for risk patterns
        max_risk = RiskLevel.SAFE
        risk_reason = ""
        
        for pattern, risk_level in self.risk_patterns.items():
            if pattern in query_lower:
                if risk_level.value > max_risk.value:
                    max_risk = risk_level
                    risk_reason = pattern
        
        # Context-based risk adjustment
        if context:
            max_risk, risk_reason = self._adjust_risk_for_context(
                max_risk, risk_reason, query, context
            )
        
        # Generate explanation
        explanation = self._generate_risk_explanation(max_risk, risk_reason, query)
        
        return max_risk, explanation
    
    def _adjust_risk_for_context(self, 
                                  risk: RiskLevel, 
                                  reason: str,
                                  query: str,
                                  context: Dict) -> Tuple[RiskLevel, str]:
        """Adjust risk based on context"""
        
        # If in development/testing environment, some risks are lower
        if context.get('environment') == 'development':
            if risk == RiskLevel.HIGH:
                risk = RiskLevel.MEDIUM
                reason += " (dev environment)"
        
        # If explicit confirmation detected, lower risk
        if any(word in query.lower() for word in ['yes', 'confirm', 'proceed', 'do it']):
            # User is confirming, treat as override
            return RiskLevel.SAFE, "user confirmed"
        
        return risk, reason
    
    def _generate_risk_explanation(self, 
                                    risk: RiskLevel, 
                                    reason: str,
                                    query: str) -> str:
        """Generate human-readable risk explanation"""
        
        if risk == RiskLevel.SAFE:
            return ""
        
        if risk == RiskLevel.LOW:
            return f"Minor risk detected: {reason}. Quick confirmation needed."
        
        if risk == RiskLevel.MEDIUM:
            return f"Moderate risk: {reason}. This action could have unintended consequences."
        
        if risk == RiskLevel.HIGH:
            return f"High risk: {reason}. This action is potentially dangerous and may cause significant issues."
        
        if risk == RiskLevel.CRITICAL:
            return f"Critical risk: {reason}. This action could cause severe damage. I strongly advise caution, though I'll proceed if you insist."
        
        return ""
    
    def should_confirm(self, risk: RiskLevel, user_insisting: bool = False) -> bool:
        """
        Determine if confirmation is needed
        
        Args:
            risk: Risk level
            user_insisting: Whether user is explicitly insisting
        
        Returns:
            True if should ask for confirmation, False if auto-proceed
        """
        # User insisting? Always proceed
        if user_insisting:
            return False
        
        # SAFE and LOW = auto-proceed
        if risk.value <= RiskLevel.LOW.value:
            return False
        
        # MEDIUM+ = ask for confirmation
        return True
    
    # ========== AGENT ROUTING ==========
    
    def route_to_agent(self, query: str, context: Dict = None) -> Tuple[AgentType, str]:
        """
        Determine which agent should handle this query
        
        Args:
            query: User's request
            context: Additional context
        
        Returns:
            (agent_type, optimized_prompt)
        """
        query_lower = query.lower()
        
        # PRIORITIZE reasoning questions (should I, what do you think, etc.)
        # Check these FIRST before other patterns
        reasoning_priority = ['should i', 'what do you think', 'do you think', 'would you']
        for pattern in reasoning_priority:
            if pattern in query_lower:
                optimized_prompt = self._optimize_prompt_for_agent(query, AgentType.REASONING, context)
                return AgentType.REASONING, optimized_prompt
        
        # Score each agent
        scores = {agent: 0 for agent in AgentType}
        
        for agent, patterns in self.agent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    scores[agent] += 1
        
        # Get best agent
        best_agent = max(scores, key=scores.get)
        
        # If no matches, use NONE
        if scores[best_agent] == 0:
            best_agent = AgentType.NONE
        
        # Generate optimized prompt for the agent
        optimized_prompt = self._optimize_prompt_for_agent(query, best_agent, context)
        
        return best_agent, optimized_prompt
    
    def _optimize_prompt_for_agent(self, 
                                     query: str,
                                     agent: AgentType,
                                     context: Dict = None) -> str:
        """
        Optimize the prompt for the specific agent
        
        Different agents need different prompt styles:
        - Deep search: Research questions
        - Command: Exact commands
        - Reasoning: Analysis requests
        - Coding: Specifications
        """
        
        if agent == AgentType.DEEP_SEARCH:
            # Research agent needs clear research questions
            return self._optimize_for_research(query, context)
        
        elif agent == AgentType.COMMAND:
            # Command agent needs explicit instructions
            return self._optimize_for_command(query, context)
        
        elif agent == AgentType.REASONING:
            # Reasoning needs analysis framework
            return self._optimize_for_reasoning(query, context)
        
        elif agent == AgentType.CODING:
            # Coding needs specifications
            return self._optimize_for_coding(query, context)
        
        else:
            # NONE agent - still add minimal context
            return f"Query: {query}\n\nPlease provide a helpful response."
    
    def _optimize_for_research(self, query: str, context: Dict = None) -> str:
        """Optimize prompt for deep search agent"""
        # Extract research question
        prompt = f"Research request: {query}\n\n"
        
        if context and context.get('active_project'):
            prompt += f"Context: User is working on {context['active_project']}\n\n"
        
        prompt += "Please provide:\n"
        prompt += "1. Comprehensive overview\n"
        prompt += "2. Technical specifications if applicable\n"
        prompt += "3. Current state-of-the-art\n"
        prompt += "4. Practical applications\n"
        prompt += "5. Relevant sources\n"
        
        return prompt
    
    def _optimize_for_command(self, query: str, context: Dict = None) -> str:
        """Optimize prompt for command agent"""
        prompt = f"Command execution request: {query}\n\n"
        
        if context:
            if context.get('current_directory'):
                prompt += f"Working directory: {context['current_directory']}\n"
            if context.get('environment'):
                prompt += f"Environment: {context['environment']}\n"
        
        prompt += "\nPlease execute the requested command safely and report results."
        
        return prompt
    
    def _optimize_for_reasoning(self, query: str, context: Dict = None) -> str:
        """Optimize prompt for reasoning agent"""
        prompt = f"Analysis request: {query}\n\n"
        
        if context:
            if context.get('preferences'):
                prompt += f"User preferences: {context['preferences']}\n"
            if context.get('constraints'):
                prompt += f"Constraints: {context['constraints']}\n"
        
        prompt += "\nPlease analyze thoroughly, considering:\n"
        prompt += "- Multiple perspectives\n"
        prompt += "- Trade-offs and implications\n"
        prompt += "- Short and long-term effects\n"
        prompt += "- Recommend best course of action\n"
        
        return prompt
    
    def _optimize_for_coding(self, query: str, context: Dict = None) -> str:
        """Optimize prompt for coding agent"""
        prompt = f"Coding task: {query}\n\n"
        
        if context:
            if context.get('language'):
                prompt += f"Language: {context['language']}\n"
            if context.get('framework'):
                prompt += f"Framework: {context['framework']}\n"
        
        prompt += "\nRequirements:\n"
        prompt += "- Clean, readable code\n"
        prompt += "- Proper error handling\n"
        prompt += "- Comments for complex logic\n"
        prompt += "- Follow best practices\n"
        
        return prompt
    
    # ========== INSISTENCE DETECTION ==========
    
    def user_is_insisting(self, query: str, conversation_history: List[str] = None) -> bool:
        """
        Detect if user is insisting on an action
        
        Insistence indicators:
        - "I said do it"
        - "Just do it"
        - "I don't care about the risk"
        - "Override"
        - "I know what I'm doing"
        - Multiple requests for same thing
        """
        query_lower = query.lower()
        
        # Explicit insistence
        insistence_phrases = [
            'just do it',
            'do it anyway',
            'i said',
            'i told you',
            "don't care",
            'override',
            'i know what',
            'i understand the risk',
            'proceed anyway',
            'i insist'
        ]
        
        if any(phrase in query_lower for phrase in insistence_phrases):
            return True
        
        # Check for repeated requests
        if conversation_history:
            # If user asked 2+ times, they're insisting
            recent = conversation_history[-4:]  # Last 2 turns
            similar_requests = sum(1 for msg in recent if self._similar_intent(query, msg))
            if similar_requests >= 2:
                return True
        
        return False
    
    def _similar_intent(self, query1: str, query2: str) -> bool:
        """Check if two queries have similar intent"""
        # Simple word overlap check
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Remove common words
        common = {'the', 'a', 'an', 'is', 'are', 'can', 'you', 'please'}
        words1 -= common
        words2 -= common
        
        if not words1 or not words2:
            return False
        
        # Calculate overlap
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        
        return overlap > 0.5
    
    # ========== FORMATTING ==========
    
    def format_confirmation_prompt(self, 
                                     query: str,
                                     risk: RiskLevel,
                                     explanation: str) -> str:
        """Format a confirmation request to user"""
        
        if risk == RiskLevel.CRITICAL:
            return f"""⚠️  CRITICAL RISK WARNING

{explanation}

However, I'll proceed if you confirm. This is your system and your decision.

Confirm? (yes/no)"""
        
        elif risk == RiskLevel.HIGH:
            return f"""⚠️  High Risk Detected

{explanation}

I recommend caution, but I'll execute if you'd like to proceed.

Continue? (yes/no)"""
        
        elif risk == RiskLevel.MEDIUM:
            return f"""⚠️  {explanation}

Proceed? (yes/no)"""
        
        else:
            return f"{explanation}\n\nContinue? (yes/no)"
    
    def format_background_message(self, agent: AgentType, task: str) -> str:
        """Format message for background processing"""
        
        agent_names = {
            AgentType.DEEP_SEARCH: "deep research",
            AgentType.COMMAND: "command execution",
            AgentType.REASONING: "analysis",
            AgentType.CODING: "code generation"
        }
        
        agent_name = agent_names.get(agent, "processing")
        
        return f"""🔄 Starting {agent_name}...

This may take a moment, sir. In the meantime, is there anything else I can help you with?

You can continue asking questions while I work on this in the background."""


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("SAFETY ENGINE EXAMPLE")
    print("="*70)
    
    engine = SafetyEngine()
    
    # Example 1: Safe action
    print("\n1. Safe action:")
    risk, explanation = engine.assess_risk("Check server status")
    print(f"   Risk: {risk.name}")
    print(f"   Explanation: {explanation or 'None - safe to proceed'}")
    print(f"   Confirm needed: {engine.should_confirm(risk)}")
    
    # Example 2: Research request
    print("\n2. Research request:")
    agent, prompt = engine.route_to_agent(
        "Research bulletproof materials that can stop 5.56 green tip"
    )
    print(f"   Agent: {agent.value}")
    print(f"   Optimized: {prompt[:100]}...")
    
    # Example 3: Risky action
    print("\n3. Risky action:")
    risk, explanation = engine.assess_risk("Delete all files in /tmp")
    print(f"   Risk: {risk.name}")
    print(f"   Explanation: {explanation}")
    print(f"   Confirm needed: {engine.should_confirm(risk)}")
    
    # Example 4: User insisting
    print("\n4. User insisting:")
    insisting = engine.user_is_insisting("Just do it, I know what I'm doing")
    print(f"   User insisting: {insisting}")
    print(f"   Confirm needed: {engine.should_confirm(RiskLevel.HIGH, insisting)}")
    
    print("\n" + "="*70)