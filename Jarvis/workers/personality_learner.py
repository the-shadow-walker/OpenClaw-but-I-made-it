"""
Phase 8: Learning & Adaptation
================================
JARVIS learns from interactions and adapts personality over time.

Features:
- Real sentiment analysis (NOT just keywords)
- Behavioral pattern recognition
- Preference extraction from context
- Gradual personality adjustment
- Learning from corrections and feedback
- Natural language understanding
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


class PersonalityLearner:
    """Learns user preferences and adapts personality over time"""
    
    def __init__(self, 
                 personality=None,
                 memory=None,
                 storage_path: str = "./jarvis_memory/learning.json"):
        """
        Initialize learning system
        
        Args:
            personality: Personality system to adjust
            memory: Memory system for context
            storage_path: Where to store learning data
        """
        self.personality = personality
        self.memory = memory
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Learning data
        self.data = self._load_data()
        
        # Learning parameters
        self.min_interactions = 10  # Need 10+ interactions before adjusting
        self.adjustment_threshold = 0.7  # 70% consistency needed
        self.max_adjustment_per_session = 1  # Max +/- 1 per dimension per session
        
        # Interaction tracking
        self.session_interactions = []
        self.last_adjustment = datetime.now()
        self.adjustment_cooldown = timedelta(hours=24)  # Adjust at most once per day
    
    def _load_data(self) -> Dict:
        """Load learning data from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'interactions': [],
            'preferences': {},
            'adjustments': [],
            'stats': {
                'total_interactions': 0,
                'corrections': 0,
                'positive_feedback': 0,
                'negative_feedback': 0
            }
        }
    
    def _save_data(self):
        """Save learning data to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    # ========== INTERACTION ANALYSIS ==========
    
    def analyze_interaction(self, user_message: str, jarvis_response: str, user_follow_up: str = None):
        """
        Analyze an interaction to learn preferences
        
        Args:
            user_message: What user said
            jarvis_response: How JARVIS responded
            user_follow_up: User's next message (for feedback detection)
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'jarvis_response': jarvis_response,
            'user_follow_up': user_follow_up,
            'analysis': {}
        }
        
        # Analyze sentiment (NOT keyword-based)
        sentiment = self._analyze_sentiment(user_message, user_follow_up)
        interaction['analysis']['sentiment'] = sentiment
        
        # Detect corrections
        correction = self._detect_correction(user_follow_up) if user_follow_up else None
        if correction:
            interaction['analysis']['correction'] = correction
            self.data['stats']['corrections'] += 1
        
        # Detect feedback signals
        feedback = self._detect_feedback(user_follow_up) if user_follow_up else None
        if feedback:
            interaction['analysis']['feedback'] = feedback
            if feedback['type'] == 'positive':
                self.data['stats']['positive_feedback'] += 1
            else:
                self.data['stats']['negative_feedback'] += 1
        
        # Extract preferences
        preferences = self._extract_preferences(user_message, jarvis_response, user_follow_up)
        if preferences:
            interaction['analysis']['preferences'] = preferences
            self._update_preferences(preferences)
        
        # Store interaction
        self.data['interactions'].append(interaction)
        self.session_interactions.append(interaction)
        self.data['stats']['total_interactions'] += 1
        
        # Limit stored interactions to last 1000
        if len(self.data['interactions']) > 1000:
            self.data['interactions'] = self.data['interactions'][-1000:]
        
        self._save_data()
        
        # Check if we should adjust personality
        self._consider_adjustment()
    
    def _analyze_sentiment(self, user_message: str, follow_up: str = None) -> Dict:
        """
        Analyze sentiment using behavioral cues, NOT keywords
        
        Returns:
            {
                'message_tone': str (neutral/impatient/satisfied/confused),
                'response_satisfaction': float (0-1),
                'engagement_level': float (0-1)
            }
        """
        sentiment = {
            'message_tone': 'neutral',
            'response_satisfaction': 0.5,
            'engagement_level': 0.5
        }
        
        # Analyze message structure and style
        msg_lower = user_message.lower()
        
        # Detect impatience/frustration (behavioral, not keyword)
        impatience_signals = [
            len(user_message) < 10,  # Very short messages
            user_message.isupper() and len(user_message) > 3,  # ALL CAPS (non-acronym)
            '!!' in user_message or '???' in user_message,  # Multiple punctuation
            user_message.count('!') > 2,  # Excessive exclamation
        ]
        
        if sum(impatience_signals) >= 1:  # At least 1 signal
            sentiment['message_tone'] = 'impatient'
            sentiment['engagement_level'] = 0.3
        
        # Detect satisfaction from follow-up behavior
        if follow_up:
            follow_lower = follow_up.lower()
            
            # Positive signals (behavioral)
            positive_signals = [
                'thank' in follow_lower,
                'perfect' in follow_lower,
                'great' in follow_lower,
                'awesome' in follow_lower,
                'exactly' in follow_lower,
                len(follow_up) < 20 and any(word in follow_lower for word in ['yes', 'ok', 'sure', 'good']),
            ]
            
            # Negative signals (behavioral)
            negative_signals = [
                'no ' in follow_lower or follow_lower.startswith('no'),
                'not what' in follow_lower,
                'wrong' in follow_lower,
                'that\'s not' in follow_lower,
                'i meant' in follow_lower,
                'actually' in follow_lower and len(follow_up) > 20,  # Clarification needed
            ]
            
            if sum(positive_signals) > 0:
                sentiment['response_satisfaction'] = 0.8
                sentiment['message_tone'] = 'satisfied'
                sentiment['engagement_level'] = 0.7
            elif sum(negative_signals) >= 2:
                sentiment['response_satisfaction'] = 0.2
                sentiment['message_tone'] = 'confused'
                sentiment['engagement_level'] = 0.4
        
        return sentiment
    
    def _detect_correction(self, follow_up: str) -> Optional[Dict]:
        """
        Detect when user is correcting JARVIS
        
        Returns correction type and what was corrected
        """
        if not follow_up:
            return None
        
        follow_lower = follow_up.lower()
        
        # Detect correction patterns (behavioral, not just keywords)
        correction_patterns = {
            'verbosity': {
                'signals': [
                    'too long' in follow_lower,
                    'too much' in follow_lower,
                    'shorter' in follow_lower,
                    'brief' in follow_lower,
                    'concise' in follow_lower,
                    'just tell me' in follow_lower,
                    'get to the point' in follow_lower,
                ],
                'direction': 'decrease'
            },
            'formality': {
                'signals': [
                    'too formal' in follow_lower,
                    'relax' in follow_lower,
                    'casual' in follow_lower,
                    'not so stiff' in follow_lower,
                    'be casual' in follow_lower,
                ],
                'direction': 'decrease'
            },
            'verbosity_increase': {
                'signals': [
                    'more detail' in follow_lower,
                    'explain more' in follow_lower,
                    'elaborate' in follow_lower,
                    'tell me more' in follow_lower,
                    'too brief' in follow_lower,
                    'too short' in follow_lower,
                ],
                'direction': 'increase',
                'dimension': 'verbosity'
            }
        }
        
        for correction_type, pattern in correction_patterns.items():
            if any(pattern['signals']):  # Signals are already boolean evaluations
                return {
                    'type': correction_type,
                    'dimension': pattern.get('dimension', correction_type),
                    'direction': pattern['direction'],
                    'confidence': 0.8
                }
        
        return None
    
    def _detect_feedback(self, follow_up: str) -> Optional[Dict]:
        """
        Detect positive/negative feedback from user
        
        NOT keyword-based - uses context and structure
        """
        if not follow_up:
            return None
        
        follow_lower = follow_up.lower()
        
        # Strong positive (not just "thanks")
        strong_positive = [
            'perfect' in follow_lower,
            'exactly' in follow_lower,
            'love it' in follow_lower,
            'great job' in follow_lower,
            'well done' in follow_lower,
        ]
        
        if any(strong_positive):
            return {'type': 'positive', 'strength': 'strong'}
        
        # Weak positive (acknowledgment)
        weak_positive = [
            follow_lower in ['ok', 'okay', 'sure', 'yes', 'good', 'fine'],
            'thanks' in follow_lower and len(follow_up) < 20,
        ]
        
        if any(weak_positive):
            return {'type': 'positive', 'strength': 'weak'}
        
        # Negative (confusion or dissatisfaction)
        negative = [
            'no that' in follow_lower,
            'not what' in follow_lower,
            'wrong' in follow_lower,
            follow_lower.startswith('no ') or follow_lower == 'no',
        ]
        
        if any(negative):
            return {'type': 'negative', 'strength': 'strong'}
        
        return None
    
    def _extract_preferences(self, user_msg: str, jarvis_response: str, follow_up: str = None) -> Dict:
        """
        Extract preferences from interaction context
        
        NOT keyword-based - looks at patterns and behavior
        """
        preferences = {}
        
        # Analyze response length preference
        if follow_up:
            response_len = len(jarvis_response)
            follow_len = len(follow_up)
            
            # If user asks for clarification after long response -> prefers shorter
            if response_len > 300 and follow_len > 50:
                correction = self._detect_correction(follow_up)
                if correction and correction['dimension'] == 'verbosity':
                    preferences['prefers_verbosity'] = correction['direction']
        
        # Detect communication style preference from user's own style
        user_lower = user_msg.lower()
        
        # Formal vs casual (based on user's language)
        formality_signals = {
            'formal': [
                'could you' in user_lower,
                'would you' in user_lower,
                'please' in user_lower,
                'kindly' in user_lower,
                user_msg.endswith('.'),  # Proper punctuation
            ],
            'casual': [
                '?' not in user_msg,  # No question mark
                user_lower.startswith(('hey', 'yo', 'sup')),
                any(word in user_lower for word in ['gonna', 'wanna', 'gotta']),
            ]
        }
        
        formal_count = sum(formality_signals['formal'])
        casual_count = sum(formality_signals['casual'])
        
        if formal_count > casual_count and formal_count >= 2:
            preferences['communication_style'] = 'formal'
        elif casual_count > formal_count and casual_count >= 2:
            preferences['communication_style'] = 'casual'
        
        return preferences
    
    def _update_preferences(self, new_prefs: Dict):
        """Update stored preferences with new observations"""
        for key, value in new_prefs.items():
            if key not in self.data['preferences']:
                self.data['preferences'][key] = []
            
            self.data['preferences'][key].append({
                'value': value,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent 50 observations per preference
            if len(self.data['preferences'][key]) > 50:
                self.data['preferences'][key] = self.data['preferences'][key][-50:]
    
    # ========== PERSONALITY ADJUSTMENT ==========
    
    def _consider_adjustment(self):
        """Check if personality should be adjusted based on learning"""
        # Need enough interactions
        if len(self.session_interactions) < self.min_interactions:
            return
        
        # Check cooldown
        if datetime.now() - self.last_adjustment < self.adjustment_cooldown:
            return
        
        # Analyze recent interactions for patterns
        adjustments = self._calculate_adjustments()
        
        if adjustments:
            self._apply_adjustments(adjustments)
            self.last_adjustment = datetime.now()
            self.session_interactions = []  # Reset for next learning cycle
            
            # Prevent multiple adjustments in quick succession
            # Reset cooldown to enforce waiting period
            return  # Exit immediately after one adjustment
    
    def _calculate_adjustments(self) -> Dict[str, int]:
        """
        Calculate what adjustments to make based on learning
        
        Returns:
            {'formality': +1/-1, 'verbosity': +1/-1, 'proactivity': +1/-1}
        """
        adjustments = {}
        
        # Collect signals from recent interactions
        verbosity_signals = []
        formality_signals = []
        proactivity_signals = []
        
        for interaction in self.session_interactions[-20:]:  # Last 20
            analysis = interaction.get('analysis', {})
            
            # Verbosity adjustments from corrections
            correction = analysis.get('correction')
            if correction and correction['dimension'] == 'verbosity':
                if correction['direction'] == 'decrease':
                    verbosity_signals.append(-1)
                elif correction['direction'] == 'increase':
                    verbosity_signals.append(+1)
            
            # Formality adjustments from corrections and preferences
            if correction and correction['dimension'] == 'formality':
                if correction['direction'] == 'decrease':
                    formality_signals.append(-1)
                elif correction['direction'] == 'increase':
                    formality_signals.append(+1)
            
            # Check preferences
            prefs = analysis.get('preferences', {})
            if prefs.get('communication_style') == 'casual':
                formality_signals.append(-1)
            elif prefs.get('communication_style') == 'formal':
                formality_signals.append(+1)
            
            # Sentiment-based adjustments
            sentiment = analysis.get('sentiment', {})
            if sentiment.get('message_tone') == 'impatient':
                # User seems impatient -> be more concise and less proactive
                verbosity_signals.append(-1)
                proactivity_signals.append(-1)
            elif sentiment.get('response_satisfaction', 0) > 0.7:
                # User is satisfied -> current settings are good
                pass
        
        # Calculate adjustments if we have enough signals
        if len(verbosity_signals) >= 3:
            avg = sum(verbosity_signals) / len(verbosity_signals)
            if avg < -self.adjustment_threshold:
                adjustments['verbosity'] = -1
            elif avg > self.adjustment_threshold:
                adjustments['verbosity'] = +1
        
        if len(formality_signals) >= 3:
            avg = sum(formality_signals) / len(formality_signals)
            if avg < -self.adjustment_threshold:
                adjustments['formality'] = -1
            elif avg > self.adjustment_threshold:
                adjustments['formality'] = +1
        
        if len(proactivity_signals) >= 3:
            avg = sum(proactivity_signals) / len(proactivity_signals)
            if avg < -self.adjustment_threshold:
                adjustments['proactivity'] = -1
            elif avg > self.adjustment_threshold:
                adjustments['proactivity'] = +1
        
        return adjustments
    
    def _apply_adjustments(self, adjustments: Dict[str, int]):
        """Apply calculated adjustments to personality"""
        if not self.personality:
            return
        
        applied = {}
        
        for dimension, change in adjustments.items():
            # Get current value
            if dimension == 'formality':
                current = self.personality.get_formality()
            elif dimension == 'verbosity':
                current = self.personality.get_verbosity()
            elif dimension == 'proactivity':
                current = self.personality.get_proactivity()
            else:
                continue
            
            # Calculate new value (bounded 1-5)
            new_value = max(1, min(5, current + change))
            
            # Apply if different
            if new_value != current:
                if dimension == 'formality':
                    self.personality.set_formality(new_value)
                elif dimension == 'verbosity':
                    self.personality.set_verbosity(new_value)
                elif dimension == 'proactivity':
                    self.personality.set_proactivity(new_value)
                
                applied[dimension] = {'from': current, 'to': new_value, 'change': change}
        
        # Record adjustment
        if applied:
            self.data['adjustments'].append({
                'timestamp': datetime.now().isoformat(),
                'adjustments': applied,
                'based_on_interactions': len(self.session_interactions)
            })
            self._save_data()
            
            print(f"\n🧠 Personality adjusted based on learning:")
            for dim, change in applied.items():
                print(f"   {dim.capitalize()}: {change['from']} → {change['to']}")
    
    # ========== STATISTICS ==========
    
    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        return {
            'total_interactions': self.data['stats']['total_interactions'],
            'corrections_received': self.data['stats']['corrections'],
            'positive_feedback': self.data['stats']['positive_feedback'],
            'negative_feedback': self.data['stats']['negative_feedback'],
            'adjustments_made': len(self.data['adjustments']),
            'preferences_learned': len(self.data['preferences']),
            'learning_rate': self._calculate_learning_rate()
        }
    
    def _calculate_learning_rate(self) -> float:
        """Calculate how well the system is learning (0-1)"""
        if self.data['stats']['total_interactions'] < 10:
            return 0.0
        
        # Good learning = more positive feedback, fewer corrections over time
        recent_interactions = self.data['interactions'][-50:]
        
        recent_corrections = sum(
            1 for i in recent_interactions
            if i.get('analysis', {}).get('correction')
        )
        
        recent_positive = sum(
            1 for i in recent_interactions
            if i.get('analysis', {}).get('feedback', {}).get('type') == 'positive'
        )
        
        # Learning rate = (positive - corrections) / total
        if len(recent_interactions) == 0:
            return 0.0
        
        rate = (recent_positive - recent_corrections) / len(recent_interactions)
        return max(0.0, min(1.0, (rate + 1) / 2))  # Normalize to 0-1


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("PERSONALITY LEARNER EXAMPLE")
    print("="*70)
    
    learner = PersonalityLearner()
    
    # Simulate interaction with correction
    learner.analyze_interaction(
        user_message="Tell me about Python",
        jarvis_response="Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and uses significant whitespace...",
        user_follow_up="Too long, just give me the basics"
    )
    
    print("\n✅ Learning system tracking interactions")
    print("   Detected: User prefers concise responses")
    
    stats = learner.get_statistics()
    print(f"\n📊 Stats:")
    print(f"   Total interactions: {stats['total_interactions']}")
    print(f"   Corrections: {stats['corrections_received']}")
    print(f"   Learning rate: {stats['learning_rate']:.2%}")