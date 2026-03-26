"""
Phase 7: Proactive Suggestions
================================
Intelligent next-step suggestions that anticipate user needs.

Features:
- Multi-source suggestions (workflows, time, context, memory)
- Confidence-based filtering (only suggest when confident)
- Smart timing (don't interrupt, wait for natural breaks)
- Context awareness (current project, recent actions, goals)
- Reasoning transparency (explain WHY suggesting)
- Non-intrusive delivery (subtle, not annoying)
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import json


class ProactiveSuggester:
    """Intelligently suggests next actions based on patterns and context"""
    
    def __init__(self, 
                 workflow_engine=None,
                 memory=None,
                 personality=None):
        """
        Initialize with access to other systems
        
        Args:
            workflow_engine: WorkflowEngine for pattern-based suggestions
            memory: Memory system for context
            personality: Personality for proactivity level
        """
        self.workflow_engine = workflow_engine
        self.memory = memory
        self.personality = personality
        
        # Suggestion thresholds
        self.min_confidence = 0.6  # Only suggest if 60%+ confident
        self.cooldown_minutes = 15  # Don't suggest same thing within 15min
        
        # Track recent suggestions to avoid repetition
        self.recent_suggestions = []  # [(suggestion, timestamp), ...]
        self.max_recent = 10
        
        # Track user responses to suggestions
        self.suggestion_history = []  # For learning
    
    # ========== MAIN SUGGESTION ENGINE ==========
    
    def get_suggestions(self, 
                       current_context: Dict = None,
                       recent_actions: List[str] = None,
                       max_suggestions: int = 3) -> List[Dict]:
        """
        Get proactive suggestions from all sources
        
        Args:
            current_context: Current state (project, time, etc.)
            recent_actions: Recent actions taken
            max_suggestions: Max suggestions to return
        
        Returns:
            List of suggestions, each with:
            {
                'action': str,
                'reason': str,
                'confidence': float,
                'source': str (workflow/time/context/memory),
                'priority': int (1-5),
                'timing': str (now/soon/later)
            }
        """
        if not self._should_suggest():
            return []
        
        all_suggestions = []
        
        # 1. Workflow-based suggestions
        if self.workflow_engine and recent_actions:
            workflow_suggestions = self._get_workflow_suggestions(recent_actions)
            all_suggestions.extend(workflow_suggestions)
        
        # 2. Time-based suggestions
        if self.workflow_engine:
            time_suggestions = self._get_time_based_suggestions()
            all_suggestions.extend(time_suggestions)
        
        # 3. Context-based suggestions
        if current_context:
            context_suggestions = self._get_context_suggestions(current_context)
            all_suggestions.extend(context_suggestions)
        
        # 4. Memory-based suggestions
        if self.memory:
            memory_suggestions = self._get_memory_suggestions(current_context)
            all_suggestions.extend(memory_suggestions)
        
        # 5. Goal-based suggestions
            goal_suggestions = self._get_goal_suggestions()
            all_suggestions.extend(goal_suggestions)
        
        # Filter and rank
        filtered = self._filter_suggestions(all_suggestions)
        ranked = self._rank_suggestions(filtered)
        
        return ranked[:max_suggestions]
    
    def _should_suggest(self) -> bool:
        """Determine if it's appropriate to suggest anything"""
        if not self.personality:
            return True  # Default: allow suggestions
        
        # Check proactivity level
        proactivity = self.personality.get_proactivity()
        
        if proactivity <= 1:
            return False  # "Only when asked" - no proactive suggestions
        elif proactivity == 2:
            # Very selective - only high-confidence patterns
            self.min_confidence = 0.8
        elif proactivity >= 4:
            # More proactive
            self.min_confidence = 0.5
        
        return True
    
    # ========== WORKFLOW-BASED SUGGESTIONS ==========
    
    def _get_workflow_suggestions(self, recent_actions: List[str]) -> List[Dict]:
        """Get suggestions based on learned workflow patterns"""
        if not self.workflow_engine:
            return []
        
        suggestions = []
        
        # Get workflow predictions
        workflow_preds = self.workflow_engine.get_workflow_suggestions(recent_actions)
        
        for pred in workflow_preds:
            # Skip if recently suggested
            if self._recently_suggested(pred['next_action']):
                continue
            
            suggestions.append({
                'action': pred['next_action'],
                'reason': pred['reason'],
                'confidence': pred['confidence'],
                'source': 'workflow',
                'priority': self._calculate_priority(pred['confidence'], 'workflow'),
                'timing': 'now',
                'pattern_occurrences': pred.get('occurrences', 0)
            })
        
        return suggestions
    
    # ========== TIME-BASED SUGGESTIONS ==========
    
    def _get_time_based_suggestions(self) -> List[Dict]:
        """Get suggestions based on time patterns"""
        if not self.workflow_engine:
            return []
        
        suggestions = []
        time_preds = self.workflow_engine.get_time_based_suggestions()
        
        for pred in time_preds:
            if self._recently_suggested(pred['action']):
                continue
            
            suggestions.append({
                'action': pred['action'],
                'reason': pred['reason'],
                'confidence': pred['confidence'],
                'source': 'time_pattern',
                'priority': self._calculate_priority(pred['confidence'], 'time'),
                'timing': 'now'
            })
        
        return suggestions
    
    # ========== CONTEXT-BASED SUGGESTIONS ==========
    
    def _get_context_suggestions(self, context: Dict) -> List[Dict]:
        """Get suggestions based on current context"""
        suggestions = []
        
        # Check if there's an active project
        active_project = context.get('active_project')
        if active_project:
            project_suggestions = self._suggest_for_project(active_project)
            suggestions.extend(project_suggestions)
        
        # Check last action
        last_action = context.get('last_action')
        if last_action and self.workflow_engine:
            context_preds = self.workflow_engine.get_context_suggestions(last_action)
            
            for pred in context_preds:
                if self._recently_suggested(pred['action']):
                    continue
                
                suggestions.append({
                    'action': pred['action'],
                    'reason': pred['reason'],
                    'confidence': pred['confidence'],
                    'source': 'context',
                    'priority': self._calculate_priority(pred['confidence'], 'context'),
                    'timing': self._infer_timing(pred.get('typical_delay', 0))
                })
        
        return suggestions
    
    def _suggest_for_project(self, project: Dict) -> List[Dict]:
        """Suggest actions relevant to current project"""
        suggestions = []
        
        project_name = project.get('name', '').lower()
        status = project.get('status', 'active').lower()
        
        # Project-specific suggestions
        if 'deploy' in project_name or 'ops' in project_name:
            if status == 'active':
                suggestions.append({
                    'action': 'check_deployment_status',
                    'reason': f"Active deployment project: {project['name']}",
                    'confidence': 0.7,
                    'source': 'project_context',
                    'priority': 3,
                    'timing': 'soon'
                })
        
        if 'api' in project_name:
            suggestions.append({
                'action': 'test_endpoints',
                'reason': f"API project in progress: {project['name']}",
                'confidence': 0.6,
                'source': 'project_context',
                'priority': 2,
                'timing': 'later'
            })
        
        return suggestions
    
    # ========== MEMORY-BASED SUGGESTIONS ==========
    
    def _get_memory_suggestions(self, context: Dict) -> List[Dict]:
        """Get suggestions based on stored memories and goals"""
        if not self.memory:
            return []
        
        suggestions = []
        now = datetime.now()
        
        # Check for upcoming events/deadlines
        notes = self.memory.notes[-20:] if hasattr(self.memory, 'notes') else []
        
        for note_dict in notes:
            note = note_dict.get('note', '')
            note_lower = note.lower()
            
            # Detect deadlines
            if any(word in note_lower for word in ['deadline', 'due', 'by', 'before']):
                # Check if mentioning a date
                if self._mentions_near_date(note):
                    suggestions.append({
                        'action': 'review_deadline',
                        'reason': f"Upcoming deadline: {note[:50]}...",
                        'confidence': 0.75,
                        'source': 'memory',
                        'priority': 4,
                        'timing': 'now',
                        'details': note
                    })
            
            # Detect TODO items
            if 'todo' in note_lower or 'need to' in note_lower or 'should' in note_lower:
                suggestions.append({
                    'action': 'review_todo',
                    'reason': f"Pending task: {note[:50]}...",
                    'confidence': 0.65,
                    'source': 'memory',
                    'priority': 3,
                    'timing': 'soon',
                    'details': note
                })
        
        return suggestions
    
    def _mentions_near_date(self, text: str) -> bool:
        """Check if text mentions a date within next 7 days"""
        # Simple heuristic: check for day names, "tomorrow", "this week", etc.
        near_keywords = [
            'tomorrow', 'today', 'tonight',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'this week', 'next week', 'soon'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in near_keywords)
    
    # ========== GOAL-BASED SUGGESTIONS ==========
    
    def _get_goal_suggestions(self) -> List[Dict]:
        """Get suggestions based on user goals"""
        # Future: integrate with explicit goal tracking
        # For now, infer from projects and memories
        return []
    
    # ========== FILTERING & RANKING ==========
    
    def _filter_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Filter out low-confidence and duplicate suggestions"""
        filtered = []
        seen_actions = set()
        
        for sugg in suggestions:
            # Skip low confidence
            if sugg['confidence'] < self.min_confidence:
                continue
            
            # Skip duplicates (keep highest confidence)
            action = sugg['action']
            if action in seen_actions:
                # Check if this one is better
                existing = [s for s in filtered if s['action'] == action][0]
                if sugg['confidence'] > existing['confidence']:
                    filtered.remove(existing)
                    filtered.append(sugg)
                continue
            
            seen_actions.add(action)
            filtered.append(sugg)
        
        return filtered
    
    def _rank_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """Rank suggestions by priority and confidence"""
        def score(sugg):
            # Combine priority and confidence
            priority_weight = sugg['priority'] / 5.0  # Normalize to 0-1
            confidence_weight = sugg['confidence']
            
            # Priority matters more for high-priority items
            if sugg['priority'] >= 4:
                return (priority_weight * 0.6) + (confidence_weight * 0.4)
            else:
                return (priority_weight * 0.4) + (confidence_weight * 0.6)
        
        return sorted(suggestions, key=score, reverse=True)
    
    # ========== UTILITIES ==========
    
    def _calculate_priority(self, confidence: float, source: str) -> int:
        """Calculate priority (1-5) based on confidence and source"""
        # Base priority on confidence
        if confidence >= 0.9:
            base = 5
        elif confidence >= 0.75:
            base = 4
        elif confidence >= 0.6:
            base = 3
        elif confidence >= 0.45:
            base = 2
        else:
            base = 1
        
        # Adjust based on source
        if source == 'workflow':
            return min(base + 1, 5)  # Workflows are reliable
        elif source == 'memory':
            return min(base + 1, 5)  # User explicitly mentioned it
        elif source == 'time_pattern':
            return base  # Time patterns are helpful but not urgent
        else:
            return max(base - 1, 1)
    
    def _infer_timing(self, delay_seconds: int) -> str:
        """Infer when to suggest based on typical delay"""
        if delay_seconds < 60:
            return 'now'
        elif delay_seconds < 600:  # 10 minutes
            return 'soon'
        else:
            return 'later'
    
    def _recently_suggested(self, action: str) -> bool:
        """Check if action was recently suggested"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=self.cooldown_minutes)
        
        # Clean old suggestions
        self.recent_suggestions = [
            (a, t) for a, t in self.recent_suggestions
            if t > cutoff
        ]
        
        # Check if action was suggested recently
        return any(a == action for a, _ in self.recent_suggestions)
    
    def mark_suggested(self, action: str):
        """Mark an action as suggested (to avoid repetition)"""
        self.recent_suggestions.append((action, datetime.now()))
        
        # Keep only recent
        if len(self.recent_suggestions) > self.max_recent:
            self.recent_suggestions.pop(0)
    
    # ========== FEEDBACK LEARNING ==========
    
    def record_response(self, suggestion: Dict, accepted: bool, outcome: str = None):
        """
        Record user response to a suggestion
        
        Args:
            suggestion: The suggestion that was made
            accepted: Whether user accepted it
            outcome: Optional outcome description
        """
        self.suggestion_history.append({
            'suggestion': suggestion,
            'accepted': accepted,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        })
        
        # If user rejected, learn not to suggest in similar contexts
        if not accepted:
            # Lower confidence for this pattern
            if self.workflow_engine and suggestion.get('source') == 'workflow':
                pattern_id = suggestion.get('pattern_id')
                if pattern_id:
                    self.workflow_engine.record_feedback(
                        pattern_id, 
                        suggestion['action'], 
                        accepted=False
                    )
    
    # ========== FORMATTING ==========
    
    def format_suggestion(self, suggestion: Dict) -> str:
        """Format a suggestion for display"""
        action = suggestion['action']
        reason = suggestion['reason']
        confidence = suggestion['confidence']
        timing = suggestion.get('timing', 'now')
        
        # Different formats based on timing
        if timing == 'now':
            if confidence > 0.8:
                return f"Shall I {action}? {reason}"
            else:
                return f"Would you like me to {action}? {reason}"
        elif timing == 'soon':
            return f"You might want to {action} soon. {reason}"
        else:
            return f"Consider: {action}. {reason}"
    
    def format_all_suggestions(self, suggestions: List[Dict]) -> str:
        """Format multiple suggestions nicely"""
        if not suggestions:
            return ""
        
        if len(suggestions) == 1:
            return self.format_suggestion(suggestions[0])
        
        # Multiple suggestions
        lines = ["I have a few suggestions:"]
        for i, sugg in enumerate(suggestions, 1):
            conf_indicator = "✓✓" if sugg['confidence'] > 0.8 else "✓"
            lines.append(f"{i}. {conf_indicator} {sugg['action']}: {sugg['reason']}")
        
        return "\n".join(lines)
    
    # ========== STATISTICS ==========
    
    def get_statistics(self) -> Dict:
        """Get statistics about suggestions"""
        total = len(self.suggestion_history)
        if total == 0:
            return {
                'total_suggestions': 0,
                'accepted': 0,
                'rejected': 0,
                'acceptance_rate': 0.0
            }
        
        accepted = sum(1 for s in self.suggestion_history if s['accepted'])
        
        return {
            'total_suggestions': total,
            'accepted': accepted,
            'rejected': total - accepted,
            'acceptance_rate': accepted / total
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PROACTIVE SUGGESTER EXAMPLE")
    print("="*70)
    
    # Create suggester
    suggester = ProactiveSuggester()
    
    # Simulate context
    context = {
        'active_project': {
            'name': 'AtomosOps',
            'status': 'active'
        },
        'last_action': 'deploy'
    }
    
    recent_actions = ['deploy', 'scan']
    
    print("\nContext:")
    print(f"  Project: {context['active_project']['name']}")
    print(f"  Recent: {' → '.join(recent_actions)}")
    
    # Get suggestions (without actual workflow engine)
    # In real use, would be integrated
    
    print("\n" + "="*70)
    print("Example output when integrated:")
    print("  'Shall I restart the services? You usually do that after a scan.'")
    print("="*70)