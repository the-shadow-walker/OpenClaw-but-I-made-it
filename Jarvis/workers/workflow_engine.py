"""
Phase 6: Pattern Recognition & Workflow Engine
===============================================
Advanced pattern recognition that learns user habits and workflows.

Features:
- Track action sequences with timestamps
- Detect repeated patterns using n-gram analysis
- Confidence scoring based on frequency
- Time-based patterns (e.g., "always does X at 9am")
- Conditional workflows (if X then suggest Y)
- Context-aware pattern matching
- Workflow suggestions with explanation
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re


class WorkflowEngine:
    """Advanced pattern recognition and workflow learning"""
    
    def __init__(self, db_path: str = "./jarvis_memory/workflows.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
        # Pattern detection settings
        self.min_pattern_occurrences = 3  # Need 3+ occurrences to suggest
        self.pattern_confidence_threshold = 0.6  # 60% confidence minimum
        self.max_time_gap = 3600  # 1 hour max between related actions
    
    def _init_db(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # Action history - every action user takes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                action_data TEXT,
                context TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Detected patterns - learned workflows
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE NOT NULL,
                sequence TEXT NOT NULL,
                occurrences INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.0,
                avg_time_gap INTEGER,
                context_tags TEXT,
                last_seen DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Time-based patterns (user does X at specific times)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                occurrences INTEGER DEFAULT 1,
                last_seen DATETIME,
                UNIQUE(action_type, hour_of_day, day_of_week)
            )
        """)
        
        # Context associations (when user does X, they often need Y)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_associations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger_action TEXT NOT NULL,
                suggested_action TEXT NOT NULL,
                co_occurrence_count INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.0,
                avg_delay_seconds INTEGER,
                last_seen DATETIME,
                UNIQUE(trigger_action, suggested_action)
            )
        """)
        
        # Workflow suggestions that were accepted/rejected
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suggestion_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER,
                suggestion TEXT NOT NULL,
                accepted BOOLEAN,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES patterns(id)
            )
        """)
        
        self.conn.commit()
    
    # ========== ACTION TRACKING ==========
    
    def track_action(self, action_type: str, action_data: Dict = None, context: str = None):
        """
        Record an action taken by the user
        
        Args:
            action_type: Type of action (e.g., 'deploy', 'scan', 'search')
            action_data: Additional data about the action
            context: Current context (e.g., 'working on project X')
        """
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO action_history (action_type, action_data, context)
            VALUES (?, ?, ?)
        """, (action_type, json.dumps(action_data or {}), context))
        
        self.conn.commit()
        
        # After tracking, check for patterns
        self._detect_patterns()
        self._track_time_patterns(action_type)
    
    def get_recent_actions(self, limit: int = 50) -> List[Dict]:
        """Get recent action history"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT action_type, action_data, context, timestamp
            FROM action_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== PATTERN DETECTION ==========
    
    def _detect_patterns(self):
        """Detect sequential patterns in recent actions"""
        # Get recent actions (last 100)
        recent = self.get_recent_actions(100)
        
        if len(recent) < 2:
            return
        
        # Look for 2-gram and 3-gram patterns
        for n in [2, 3, 4]:
            self._detect_ngram_patterns(recent, n)
    
    def _detect_ngram_patterns(self, actions: List[Dict], n: int):
        """Detect n-gram patterns (sequences of n actions)"""
        if len(actions) < n:
            return
        
        # Extract sequences
        sequences = []
        timestamps = []
        
        for i in range(len(actions) - n + 1):
            sequence = tuple(actions[i + j]['action_type'] for j in range(n))
            seq_timestamps = [actions[i + j]['timestamp'] for j in range(n)]
            
            # Check if actions are close in time
            time_gaps = self._calculate_time_gaps(seq_timestamps)
            
            if all(gap < self.max_time_gap for gap in time_gaps):
                sequences.append(sequence)
                timestamps.append(seq_timestamps)
        
        # Count occurrences
        sequence_counts = Counter(sequences)
        
        # Store patterns that occur multiple times
        for sequence, count in sequence_counts.items():
            if count >= self.min_pattern_occurrences:
                self._store_pattern(sequence, count)
    
    def _calculate_time_gaps(self, timestamps: List[str]) -> List[float]:
        """Calculate time gaps between timestamps in seconds"""
        gaps = []
        for i in range(len(timestamps) - 1):
            t1 = datetime.fromisoformat(timestamps[i])
            t2 = datetime.fromisoformat(timestamps[i + 1])
            gap = abs((t1 - t2).total_seconds())
            gaps.append(gap)
        return gaps
    
    def _store_pattern(self, sequence: Tuple[str, ...], occurrences: int):
        """Store or update a detected pattern"""
        cursor = self.conn.cursor()
        
        pattern_name = " → ".join(sequence)
        sequence_json = json.dumps(list(sequence))
        
        # Calculate confidence (based on frequency)
        total_actions = cursor.execute("SELECT COUNT(*) FROM action_history").fetchone()[0]
        confidence = min(occurrences / max(total_actions / 10, 1), 1.0)
        
        cursor.execute("""
            INSERT INTO patterns (pattern_name, sequence, occurrences, confidence, last_seen)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(pattern_name) DO UPDATE SET
                occurrences = occurrences + 1,
                confidence = ?,
                last_seen = CURRENT_TIMESTAMP
        """, (pattern_name, sequence_json, occurrences, confidence, confidence))
        
        self.conn.commit()
    
    # ========== TIME-BASED PATTERNS ==========
    
    def _track_time_patterns(self, action_type: str):
        """Track when user typically performs certain actions"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        cursor.execute("""
            INSERT INTO time_patterns (action_type, hour_of_day, day_of_week, last_seen)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(action_type, hour_of_day, day_of_week) DO UPDATE SET
                occurrences = occurrences + 1,
                last_seen = CURRENT_TIMESTAMP
        """, (action_type, now.hour, now.weekday()))
        
        self.conn.commit()
    
    def get_time_based_suggestions(self) -> List[Dict]:
        """Get suggestions based on current time patterns"""
        cursor = self.conn.cursor()
        now = datetime.now()
        
        # Find actions commonly done at this hour/day
        cursor.execute("""
            SELECT action_type, occurrences, hour_of_day, day_of_week
            FROM time_patterns
            WHERE hour_of_day = ? AND day_of_week = ? AND occurrences >= 3
            ORDER BY occurrences DESC
            LIMIT 3
        """, (now.hour, now.weekday()))
        
        suggestions = []
        for row in cursor.fetchall():
            suggestions.append({
                'type': 'time_based',
                'action': row['action_type'],
                'reason': f"You typically do this at {row['hour_of_day']:02d}:00 on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][row['day_of_week']]}",
                'confidence': min(row['occurrences'] / 10.0, 1.0)
            })
        
        return suggestions
    
    # ========== WORKFLOW SUGGESTIONS ==========
    
    def get_workflow_suggestions(self, recent_actions: List[str] = None) -> List[Dict]:
        """
        Get workflow suggestions based on recent actions
        
        Args:
            recent_actions: List of recent action types (optional)
        
        Returns:
            List of suggestions with confidence scores
        """
        if recent_actions is None:
            recent = self.get_recent_actions(5)
            recent_actions = [a['action_type'] for a in recent]
        
        if not recent_actions:
            return []
        
        cursor = self.conn.cursor()
        suggestions = []
        
        # Check for matching patterns
        for n in range(len(recent_actions), 0, -1):
            # Check if recent actions match start of any pattern
            recent_sequence = recent_actions[-n:]
            
            cursor.execute("""
                SELECT id, pattern_name, sequence, confidence, occurrences
                FROM patterns
                WHERE confidence >= ?
                ORDER BY occurrences DESC
            """, (self.pattern_confidence_threshold,))
            
            for row in cursor.fetchall():
                pattern_seq = json.loads(row['sequence'])
                
                # Check if recent actions match beginning of pattern
                if pattern_seq[:len(recent_sequence)] == recent_sequence:
                    # Suggest next action in pattern
                    if len(pattern_seq) > len(recent_sequence):
                        next_action = pattern_seq[len(recent_sequence)]
                        
                        suggestions.append({
                            'type': 'workflow_completion',
                            'pattern_id': row['id'],
                            'pattern_name': row['pattern_name'],
                            'next_action': next_action,
                            'full_pattern': pattern_seq,
                            'reason': f"You usually follow {' → '.join(recent_sequence)} with {next_action}",
                            'confidence': row['confidence'],
                            'occurrences': row['occurrences']
                        })
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = (s['type'], s.get('next_action', ''))
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)
        
        return sorted(unique_suggestions, key=lambda x: x['confidence'], reverse=True)[:3]
    
    # ========== CONTEXT ASSOCIATIONS ==========
    
    def learn_association(self, trigger: str, suggested: str, delay_seconds: int = 0):
        """Learn that action X often leads to action Y"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO context_associations (trigger_action, suggested_action, avg_delay_seconds, last_seen)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(trigger_action, suggested_action) DO UPDATE SET
                co_occurrence_count = co_occurrence_count + 1,
                avg_delay_seconds = (avg_delay_seconds + ?) / 2,
                last_seen = CURRENT_TIMESTAMP
        """, (trigger, suggested, delay_seconds, delay_seconds))
        
        self.conn.commit()
        
        # Update confidence
        self._update_association_confidence(trigger, suggested)
    
    def _update_association_confidence(self, trigger: str, suggested: str):
        """Calculate confidence for an association"""
        cursor = self.conn.cursor()
        
        # How often does trigger lead to suggested?
        cursor.execute("""
            SELECT co_occurrence_count FROM context_associations
            WHERE trigger_action = ? AND suggested_action = ?
        """, (trigger, suggested))
        
        row = cursor.fetchone()
        if not row:
            return
        
        co_count = row[0]
        
        # How often does trigger occur total?
        cursor.execute("""
            SELECT COUNT(*) FROM action_history WHERE action_type = ?
        """, (trigger,))
        
        total = cursor.fetchone()[0]
        
        confidence = min(co_count / max(total, 1), 1.0)
        
        cursor.execute("""
            UPDATE context_associations
            SET confidence = ?
            WHERE trigger_action = ? AND suggested_action = ?
        """, (confidence, trigger, suggested))
        
        self.conn.commit()
    
    def get_context_suggestions(self, trigger_action: str) -> List[Dict]:
        """Get suggestions based on what typically follows an action"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT suggested_action, confidence, avg_delay_seconds, co_occurrence_count
            FROM context_associations
            WHERE trigger_action = ? AND confidence >= 0.5
            ORDER BY confidence DESC
            LIMIT 3
        """, (trigger_action,))
        
        suggestions = []
        for row in cursor.fetchall():
            suggestions.append({
                'type': 'context_based',
                'action': row['suggested_action'],
                'reason': f"After {trigger_action}, you usually {row['suggested_action']}",
                'confidence': row['confidence'],
                'typical_delay': row['avg_delay_seconds']
            })
        
        return suggestions
    
    # ========== FEEDBACK LEARNING ==========
    
    def record_feedback(self, pattern_id: Optional[int], suggestion: str, accepted: bool):
        """Record whether user accepted or rejected a suggestion"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO suggestion_feedback (pattern_id, suggestion, accepted)
            VALUES (?, ?, ?)
        """, (pattern_id, suggestion, accepted))
        
        self.conn.commit()
        
        # If rejected, lower confidence
        if not accepted and pattern_id:
            cursor.execute("""
                UPDATE patterns
                SET confidence = confidence * 0.9
                WHERE id = ?
            """, (pattern_id,))
            self.conn.commit()
    
    # ========== ANALYTICS ==========
    
    def get_top_patterns(self, limit: int = 10) -> List[Dict]:
        """Get most common patterns"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, pattern_name, sequence, occurrences, confidence, last_seen
            FROM patterns
            ORDER BY occurrences DESC
            LIMIT ?
        """, (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get workflow learning statistics"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total actions tracked
        stats['total_actions'] = cursor.execute(
            "SELECT COUNT(*) FROM action_history"
        ).fetchone()[0]
        
        # Patterns learned
        stats['patterns_learned'] = cursor.execute(
            "SELECT COUNT(*) FROM patterns WHERE confidence >= ?"
            , (self.pattern_confidence_threshold,)
        ).fetchone()[0]
        
        # Time patterns
        stats['time_patterns'] = cursor.execute(
            "SELECT COUNT(*) FROM time_patterns WHERE occurrences >= 3"
        ).fetchone()[0]
        
        # Context associations
        stats['associations'] = cursor.execute(
            "SELECT COUNT(*) FROM context_associations WHERE confidence >= 0.5"
        ).fetchone()[0]
        
        # Suggestions accepted vs rejected
        accepted = cursor.execute(
            "SELECT COUNT(*) FROM suggestion_feedback WHERE accepted = 1"
        ).fetchone()[0]
        
        rejected = cursor.execute(
            "SELECT COUNT(*) FROM suggestion_feedback WHERE accepted = 0"
        ).fetchone()[0]
        
        stats['suggestions_accepted'] = accepted
        stats['suggestions_rejected'] = rejected
        stats['acceptance_rate'] = accepted / max(accepted + rejected, 1)
        
        return stats
    
    def clear_low_confidence_patterns(self):
        """Remove patterns with low confidence to keep DB clean"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            DELETE FROM patterns
            WHERE confidence < 0.3 AND occurrences < 2
        """)
        
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


# Example usage
if __name__ == "__main__":
    import tempfile
    
    temp_db = Path(tempfile.gettempdir()) / "test_workflows.db"
    engine = WorkflowEngine(str(temp_db))
    
    print("="*70)
    print("WORKFLOW ENGINE EXAMPLE")
    print("="*70)
    
    # Simulate a deployment workflow
    print("\n1. Simulating deployment workflow...")
    engine.track_action("deploy", {"target": "production"})
    engine.track_action("scan", {"type": "vulnerability"})
    engine.track_action("check_logs", {"service": "api"})
    
    # Do it again
    engine.track_action("deploy", {"target": "staging"})
    engine.track_action("scan", {"type": "vulnerability"})
    engine.track_action("check_logs", {"service": "api"})
    
    # And again
    engine.track_action("deploy", {"target": "production"})
    engine.track_action("scan", {"type": "vulnerability"})
    engine.track_action("check_logs", {"service": "api"})
    
    print("   Tracked 3 deployment workflows")
    
    # Check for suggestions
    print("\n2. Getting workflow suggestions after 'deploy' → 'scan'...")
    suggestions = engine.get_workflow_suggestions(['deploy', 'scan'])
    
    for s in suggestions:
        print(f"\n   Suggestion: {s['next_action']}")
        print(f"   Reason: {s['reason']}")
        print(f"   Confidence: {s['confidence']:.2%}")
    
    # Statistics
    print("\n3. Learning statistics:")
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    engine.close()
    temp_db.unlink()
    
    print("\n✅ Example complete")