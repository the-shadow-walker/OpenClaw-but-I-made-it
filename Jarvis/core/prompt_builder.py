"""
Phase 4: Dynamic Prompt Builder
================================
Builds context-aware system prompts with user profile, memories, and projects.
"""

from typing import Dict, List, Optional
from datetime import datetime


class PromptBuilder:
    """Dynamically builds context-aware system prompts"""
    
    def __init__(self, base_prompt: str):
        """
        Initialize with base personality prompt
        
        Args:
            base_prompt: The base JARVIS personality/instructions
        """
        self.base_prompt = base_prompt
    
    def build_prompt(self,
                    user_profile: Dict[str, str] = None,
                    preferences: Dict[str, Dict] = None,
                    active_projects: List[Dict] = None,
                    recent_conversations: str = None,
                    past_memories: List[str] = None,
                    vector_search_results: List[Dict] = None,
                    server_status: str = None,
                    media_catalog: str = None,
                    journal_results: List[str] = None,
                    related_entities: List[str] = None,
                    context_budget: int = 12000,
                    architect_mode_project: str = None) -> str:
        """
        Build a complete system prompt with all available context.

        Uses a context budget to prioritise the most relevant information:
          1. User Profile   — always included (identity anchor)
          2. Journal Matches — exact text hits from daily logs (highest recall)
          3. Vector Matches  — HyDE semantic search results
          4. Related Entities — knowledge-graph neighbours
          5. Recent History  — working session context
          6. Projects / Preferences / Memories — trimmed to fit
          7. Server Status & Media Catalog — functional, appended last
        """
        # Base prompt is always included (not counted against budget)
        sections = [self.base_prompt]
        budget_used = 0

        def fits(text: str) -> bool:
            return (budget_used + len(text)) <= context_budget

        def add(text: str):
            nonlocal budget_used
            sections.append(text)
            budget_used += len(text)

        # ── Priority 1: User Profile (always) ────────────────────────────────
        if user_profile:
            s = self._build_profile_section(user_profile)
            if s:
                add(s)

        # ── Priority 2: Journal Matches (exact recall) ────────────────────────
        if journal_results:
            s = self._build_journal_section(journal_results)
            if s and fits(s):
                add(s)

        # ── Priority 3: Vector / HyDE Matches ────────────────────────────────
        if vector_search_results:
            s = self._build_search_results_section(vector_search_results)
            if s and fits(s):
                add(s)

        # ── Priority 4: Related Entities (knowledge graph) ───────────────────
        if related_entities:
            s = self._build_related_entities_section(related_entities)
            if s and fits(s):
                add(s)

        # ── Priority 5: Recent Conversation Context ───────────────────────────
        if recent_conversations and fits(recent_conversations):
            add(recent_conversations)

        # ── Priority 6: Projects ──────────────────────────────────────────────
        if active_projects:
            s = self._build_projects_section(active_projects)
            if s and fits(s):
                add(s)

        # ── Priority 7: Preferences ───────────────────────────────────────────
        if preferences:
            s = self._build_preferences_section(preferences)
            if s and fits(s):
                add(s)

        # ── Priority 8: Past Memories ─────────────────────────────────────────
        if past_memories:
            s = self._build_memories_section(past_memories)
            if s and fits(s):
                add(s)

        # ── Functional (always appended — not budgeted) ───────────────────────
        if server_status:
            sections.append(f"SERVER STATUS: {server_status}")
        if media_catalog:
            sections.append(media_catalog)

        # ── Architect Discovery Mode (always appended when active) ────────────
        if architect_mode_project:
            sections.append(
                f"CURRENT MODE: Project Discovery.\n"
                f"The active project '{architect_mode_project}' has undefined specs. "
                f"Your goal is to interview the user to fill out the Architecture and Tech Stack. "
                f"Do not offer generic advice; ask specific constraint questions."
            )

        return "\n\n".join(sections)
    
    def _build_profile_section(self, profile: Dict[str, str]) -> str:
        """Build user profile section"""
        if not profile:
            return ""
        
        parts = []
        
        # Name and preferred address
        name = profile.get('name', 'User')
        preferred = profile.get('preferred_name', '')
        
        if preferred:
            parts.append(f"USER: {name} (address them as '{preferred}')")
        else:
            parts.append(f"USER: {name}")
        
        # System info
        if profile.get('os'):
            parts.append(f"System: {profile['os']}")
        
        if profile.get('hostname'):
            parts.append(f"Hostname: {profile['hostname']}")
        
        return "USER PROFILE:\n" + "\n".join(parts)
    
    def _build_preferences_section(self, preferences: Dict[str, Dict]) -> str:
        """Build preferences section"""
        if not preferences:
            return ""
        
        lines = []
        
        for category, prefs in preferences.items():
            for key, value in prefs.items():
                lines.append(f"- {category}.{key}: {value}")
        
        if not lines:
            return ""
        
        return "USER PREFERENCES:\n" + "\n".join(lines)
    
    def _build_projects_section(self, projects: List[Dict]) -> str:
        """Build active projects section"""
        if not projects:
            return ""
        
        lines = []
        
        # Sort by priority (high to low), then by name for stability
        sorted_projects = sorted(projects, 
                                key=lambda p: (-p.get('priority', 5), p.get('name', '')))
        
        for proj in sorted_projects[:5]:  # Top 5 projects
            name = proj.get('name', 'Unknown')
            desc = proj.get('description', 'No description')
            status = proj.get('status', 'active')
            
            line = f"- {name}: {desc}"
            if status != 'active':
                line += f" (status: {status})"
            
            lines.append(line)
        
        return "ACTIVE PROJECTS:\n" + "\n".join(lines)
    
    def _build_memories_section(self, memories: List[str]) -> str:
        """Build important memories section"""
        if not memories:
            return ""
        
        # Show most recent memories
        recent = memories[-10:]  # Last 10
        lines = [f"- {m}" for m in recent]
        
        return "IMPORTANT MEMORIES:\n" + "\n".join(lines)
    
    def _build_journal_section(self, snippets: List[str]) -> str:
        """Build section for journal search matches (exact recall)."""
        if not snippets:
            return ""
        lines = ["JOURNAL MATCHES (exact recall from past sessions):"]
        for snippet in snippets[:5]:  # Cap at 5 snippets
            # Truncate very long snippets
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            lines.append(f"• {snippet}")
        return "\n".join(lines)

    def _build_related_entities_section(self, entities: List[str]) -> str:
        """Build section for knowledge-graph related entities."""
        if not entities:
            return ""
        lines = ["RELATED ENTITIES (knowledge graph):"]
        for entry in entities[:8]:
            lines.append(f"• {entry}")
        return "\n".join(lines)

    def _build_search_results_section(self, results: List[Dict]) -> str:
        """Build section for vector search results (query-specific context)"""
        if not results:
            return ""
        
        lines = ["RELEVANT PAST CONVERSATIONS:"]
        
        for i, result in enumerate(results[:3], 1):  # Top 3
            text = result.get('text', '')
            # Truncate if too long
            if len(text) > 200:
                text = text[:200] + "..."
            lines.append(f"{i}. {text}")
        
        return "\n".join(lines)
    
    def build_query_specific_prompt(self,
                                    query: str,
                                    base_context: str,
                                    vector_results: List[Dict] = None) -> str:
        """
        Build a prompt for a specific query with relevant past context
        
        Args:
            query: The user's current query
            base_context: Base context (profile, preferences, etc.)
            vector_results: Relevant past conversations for this query
        
        Returns:
            Query-specific prompt
        """
        sections = [base_context]
        
        if vector_results:
            search_section = self._build_search_results_section(vector_results)
            if search_section:
                sections.append(search_section)
        
        return "\n\n".join(sections)
    
    def should_inject_vector_search(self, query: str) -> bool:
        """
        Determine if vector search should be performed for this query
        
        Args:
            query: User's query
        
        Returns:
            True if query references past conversations
        """
        # Keywords that suggest user is asking about past
        past_keywords = [
            'remember', 'mentioned', 'told you', 'said', 'talked about',
            'discussed', 'last time', 'before', 'earlier', 'previous',
            'what did', 'when did', 'have we', 'did we', 'didn\'t i', 'didnt i'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in past_keywords)


# Test/Example usage
if __name__ == "__main__":
    # Example base prompt
    BASE = "You are JARVIS, an intelligent assistant."
    
    builder = PromptBuilder(BASE)
    
    # Example data
    profile = {
        'name': 'Grant',
        'preferred_name': 'sir',
        'os': 'macOS',
        'hostname': 'MacBook-Pro'
    }
    
    preferences = {
        'coding': {
            'language': 'Python',
            'style': 'pythonic'
        },
        'communication': {
            'tone': 'direct',
            'verbosity': 'concise'
        }
    }
    
    projects = [
        {'name': 'AtomosOps', 'description': 'DevOps platform', 'priority': 10},
        {'name': 'JARVIS', 'description': 'AI assistant', 'priority': 9}
    ]
    
    memories = [
        'traveling to Arizona Feb 12-17',
        'prefers detailed explanations for complex topics'
    ]
    
    # Build prompt
    prompt = builder.build_prompt(
        user_profile=profile,
        preferences=preferences,
        active_projects=projects,
        past_memories=memories
    )
    
    print("="*70)
    print("DYNAMIC PROMPT EXAMPLE")
    print("="*70)
    print(prompt)
    print("="*70)