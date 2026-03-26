"""
JARVIS Memory System - Phase 1: Foundation
==========================================
Storage layer for persistent memory and personality

Author: Built for Grant's JARVIS
Version: 1.0 - Foundation
"""

import re
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


# =============================================================================
# FACTS DATABASE - Structured Storage
# =============================================================================

class FactsDB:
    """SQLite database for structured facts about the user"""
    
    def __init__(self, db_path: str = "./jarvis_memory/facts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return dict-like rows
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=2000")
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        cursor = self.conn.cursor()
        
        # User profile table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (category, key)
            )
        """)
        
        # Projects table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'active',
                tech_stack TEXT,
                repo_url TEXT,
                priority INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_worked TIMESTAMP,
                notes TEXT
            )
        """)
        
        # Entities table (people, servers, services)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                relationship TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name, type)
            )
        """)
        
        # Habits/patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS habits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                context TEXT,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Phase 4: Entity knowledge graph
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_links (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source       TEXT NOT NULL,
                target       TEXT NOT NULL,
                relationship TEXT NOT NULL,
                strength     INTEGER DEFAULT 1,
                updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source, target, relationship)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prefs_cat ON preferences(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_habits_pattern ON habits(pattern)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_source ON entity_links(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_links_target ON entity_links(target)")
        self.conn.commit()
        print("✅ FactsDB schema initialized")
    
    # ========== USER PROFILE ==========
    
    def get_user_profile(self) -> Dict[str, str]:
        """Get complete user profile"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM user_profile")
        return {row['key']: row['value'] for row in cursor.fetchall()}
    
    def set_profile_field(self, key: str, value: str):
        """Set a single profile field"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO user_profile (key, value) 
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET 
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
        """, (key, value))
        self.conn.commit()
    
    def get_profile_field(self, key: str, default: str = None) -> Optional[str]:
        """Get a single profile field"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM user_profile WHERE key=?", (key,))
        row = cursor.fetchone()
        return row['value'] if row else default
    
    # ========== PREFERENCES ==========
    
    def get_preferences(self, category: str = None) -> Dict[str, Any]:
        """Get all preferences or preferences for a category"""
        cursor = self.conn.cursor()
        
        if category:
            cursor.execute("""
                SELECT key, value FROM preferences WHERE category=?
            """, (category,))
            return {row['key']: row['value'] for row in cursor.fetchall()}
        else:
            cursor.execute("SELECT category, key, value FROM preferences")
            prefs = {}
            for row in cursor.fetchall():
                if row['category'] not in prefs:
                    prefs[row['category']] = {}
                prefs[row['category']][row['key']] = row['value']
            return prefs
    
    def set_preference(self, category: str, key: str, value: str):
        """Set a preference"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO preferences (category, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT(category, key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
        """, (category, key, value))
        self.conn.commit()
    
    # ========== PROJECTS ==========
    
    def add_project(self, name: str, **kwargs):
        """Add a new project"""
        cursor = self.conn.cursor()
        
        fields = ['name']
        values = [name]
        placeholders = ['?']
        
        for key, value in kwargs.items():
            if key in ['description', 'status', 'tech_stack', 'repo_url', 'priority', 'notes']:
                fields.append(key)
                values.append(value)
                placeholders.append('?')
        
        query = f"""
            INSERT INTO projects ({', '.join(fields)})
            VALUES ({', '.join(placeholders)})
        """
        
        try:
            cursor.execute(query, values)
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Project already exists, update it
            self.update_project(name, **kwargs)
            return None
    
    def get_active_projects(self) -> List[Dict]:
        """Get all active projects"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM projects 
            WHERE status='active'
            ORDER BY priority DESC, last_worked DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    
    def update_project(self, name: str, **kwargs):
        """Update project fields"""
        cursor = self.conn.cursor()
        
        updates = []
        values = []
        
        for key, value in kwargs.items():
            if key in ['description', 'status', 'tech_stack', 'repo_url', 'priority', 'notes', 'last_worked']:
                updates.append(f"{key}=?")
                values.append(value)
        
        if not updates:
            return
        
        values.append(name)
        query = f"UPDATE projects SET {', '.join(updates)} WHERE name=?"
        cursor.execute(query, values)
        self.conn.commit()
    
    def touch_project(self, name: str):
        """Update last_worked timestamp for a project"""
        self.update_project(name, last_worked=datetime.now().isoformat())
    
    # ========== ENTITIES ==========
    
    def add_entity(self, name: str, entity_type: str, relationship: str = None, details: str = None):
        """Add a person, server, service, etc."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO entities (name, type, relationship, details)
                VALUES (?, ?, ?, ?)
            """, (name, entity_type, relationship, details))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Already exists, update
            cursor.execute("""
                UPDATE entities 
                SET relationship=?, details=?
                WHERE name=? AND type=?
            """, (relationship, details, name, entity_type))
            self.conn.commit()
    
    def get_entities(self, entity_type: str = None) -> List[Dict]:
        """Get entities, optionally filtered by type"""
        cursor = self.conn.cursor()
        
        if entity_type:
            cursor.execute("SELECT * FROM entities WHERE type=?", (entity_type,))
        else:
            cursor.execute("SELECT * FROM entities")
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== HABITS/PATTERNS ==========
    
    def track_habit(self, pattern: str, context: str = None):
        """Track a repeated action/pattern"""
        cursor = self.conn.cursor()
        
        # Check if pattern exists
        cursor.execute("SELECT id, frequency FROM habits WHERE pattern=?", (pattern,))
        row = cursor.fetchone()
        
        if row:
            # Increment frequency
            cursor.execute("""
                UPDATE habits 
                SET frequency=frequency+1, last_seen=CURRENT_TIMESTAMP, context=?
                WHERE id=?
            """, (context, row['id']))
        else:
            # New pattern
            cursor.execute("""
                INSERT INTO habits (pattern, context)
                VALUES (?, ?)
            """, (pattern, context))
        
        self.conn.commit()
    
    def get_common_habits(self, limit: int = 10) -> List[Dict]:
        """Get most common patterns"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM habits
            ORDER BY frequency DESC, last_seen DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    # ========== ENTITY LINKS (Phase 4: Knowledge Graph) ==========

    def link_entities(self, source: str, target: str, relationship: str):
        """
        Create or strengthen a directional link between two named entities.
        Strength increments each time the same link is confirmed.
        Example: link_entities("arch01", "AtomosOps", "hosts")
        """
        source = source.strip()
        target = target.strip()
        relationship = relationship.strip()
        if not source or not target or not relationship:
            return
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO entity_links (source, target, relationship, strength)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(source, target, relationship) DO UPDATE SET
                strength   = strength + 1,
                updated_at = CURRENT_TIMESTAMP
        """, (source, target, relationship))
        self.conn.commit()

    def get_related_entities(self, entity_name: str) -> List[Dict]:
        """
        Return everything linked to entity_name — both as source and target.
        UNION query covers both directions of the graph.
        Results sorted by strength descending (strongest links first).
        """
        entity_name = entity_name.strip()
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT target AS entity, relationship, strength FROM entity_links
            WHERE source = ?
            UNION
            SELECT source AS entity, relationship, strength FROM entity_links
            WHERE target = ?
            ORDER BY strength DESC
        """, (entity_name, entity_name))
        return [dict(row) for row in cursor.fetchall()]

    def _enforce_limits(self):
        """Cap habits and entities tables to prevent unbounded growth."""
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM habits WHERE id NOT IN "
            "(SELECT id FROM habits ORDER BY frequency DESC LIMIT 500)"
        )
        cursor.execute(
            "DELETE FROM entities WHERE id NOT IN "
            "(SELECT id FROM entities ORDER BY id DESC LIMIT 500)"
        )
        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()


# =============================================================================
# JOURNAL MANAGER - Layer 1 Raw Memory
# =============================================================================

class JournalManager:
    """
    Immutable, grep-able daily log of every conversation interaction.
    Saves to jarvis_memory/daily_logs/YYYY-MM-DD.md — one file per day.
    Never modifies past entries; only appends.
    """

    def __init__(self, memory_dir: str = "./jarvis_memory"):
        self.log_dir = Path(memory_dir) / "daily_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _today_file(self) -> Path:
        return self.log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

    def log_interaction(self, user_query: str, jarvis_response: str):
        """Append one Q/A pair to today's journal file."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = (
            f"\n## {ts}\n"
            f"**You:** {user_query.strip()}\n\n"
            f"**JARVIS:** {jarvis_response.strip()}\n"
        )
        try:
            with open(self._today_file(), "a", encoding="utf-8") as f:
                f.write(entry)
        except Exception:
            pass  # Journal is best-effort — never crash JARVIS

    def search_journals(self, query: str, days: int = 30) -> List[str]:
        """
        Text search over the last `days` days of markdown logs.
        Returns a list of matching snippet strings (timestamp + surrounding text).
        """
        query_lower = query.lower()
        snippets: List[str] = []

        # Collect the last N day files, most recent first
        log_files = sorted(self.log_dir.glob("*.md"), reverse=True)[:days]

        for log_file in log_files:
            try:
                lines = log_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue

            date_label = log_file.stem  # e.g. "2026-02-17"
            current_ts = ""

            for i, line in enumerate(lines):
                if line.startswith("## "):
                    current_ts = line[3:].strip()
                    continue

                if query_lower in line.lower():
                    # Grab a small window around the match
                    start = max(0, i - 1)
                    end = min(len(lines), i + 3)
                    context = " ".join(lines[start:end]).strip()
                    snippets.append(f"[{date_label} {current_ts}] {context}")

                    if len(snippets) >= 20:
                        return snippets

        return snippets


# =============================================================================
# MEMORY MANAGER - Main Orchestrator
# =============================================================================

class MemoryManager:
    """Central memory management system for JARVIS"""
    
    def __init__(self, memory_dir: str = "./jarvis_memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage components
        self.facts_db = FactsDB(str(self.memory_dir / "facts.db"))
        
        print("✅ MemoryManager initialized")
    
    # ========== USER PROFILE ==========
    
    def initialize_user_profile(self, **profile_data):
        """Set up initial user profile"""
        for key, value in profile_data.items():
            self.facts_db.set_profile_field(key, str(value))
        print(f"✅ User profile initialized with {len(profile_data)} fields")
    
    def get_user_profile(self) -> Dict[str, str]:
        """Get user profile"""
        return self.facts_db.get_user_profile()
    
    def update_profile(self, key: str, value: str):
        """Update a profile field"""
        self.facts_db.set_profile_field(key, value)
    
    # ========== PREFERENCES ==========
    
    def set_preference(self, category: str, key: str, value: str):
        """Set a user preference"""
        self.facts_db.set_preference(category, key, value)
        print(f"✅ Preference set: {category}.{key} = {value}")
    
    def get_preferences(self, category: str = None) -> Dict:
        """Get preferences"""
        return self.facts_db.get_preferences(category)
    
    # ========== PROJECTS ==========
    
    def add_project(self, name: str, **details):
        """Add or update a project"""
        project_id = self.facts_db.add_project(name, **details)
        if project_id:
            print(f"✅ Project added: {name}")
        else:
            print(f"✅ Project updated: {name}")
    
    def get_active_projects(self) -> List[Dict]:
        """Get all active projects"""
        return self.facts_db.get_active_projects()
    
    def touch_project(self, name: str):
        """Mark project as recently worked on"""
        self.facts_db.touch_project(name)
    
    # ========== ENTITIES ==========
    
    def remember_entity(self, name: str, entity_type: str, relationship: str = None, details: str = None):
        """Remember a person, server, service, etc."""
        self.facts_db.add_entity(name, entity_type, relationship, details)
        print(f"✅ Remembered {entity_type}: {name}")
    
    def get_entities(self, entity_type: str = None) -> List[Dict]:
        """Get known entities"""
        return self.facts_db.get_entities(entity_type)
    
    # ========== HABITS ==========
    
    def track_habit(self, pattern: str, context: str = None):
        """Track a repeated behavior"""
        self.facts_db.track_habit(pattern, context)
    
    def get_common_habits(self, limit: int = 10) -> List[Dict]:
        """Get most common patterns"""
        return self.facts_db.get_common_habits(limit)
    
    # ========== UTILITY ==========
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memory"""
        return {
            'profile_fields': len(self.get_user_profile()),
            'preferences': sum(len(v) for v in self.get_preferences().values()),
            'active_projects': len(self.get_active_projects()),
            'known_entities': len(self.get_entities()),
            'tracked_habits': len(self.get_common_habits(limit=1000))
        }
    
    def close(self):
        """Clean shutdown"""
        self.facts_db.close()


# =============================================================================
# PROJECT STATE MANAGER
# =============================================================================

class ProjectState:
    """
    Manages markdown project files in jarvis_memory/active_projects/.

    Each project gets its own .md file with structured sections:
        # Description
        # Current Status
        # Architecture
        # Todo List

    JARVIS reads and writes these files as it works — they are the
    externalized "brain" for each project.
    """

    SECTIONS = ["Description", "Current Status", "Architecture", "Todo List"]

    def __init__(self, memory_dir: str = "./jarvis_memory"):
        self.projects_dir = Path(memory_dir) / "active_projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def _project_path(self, name: str) -> Path:
        # Sanitize name for use as a filename
        safe = re.sub(r'[^\w\-]', '_', name.strip())
        return self.projects_dir / f"{safe}.md"

    def create_project(self, name: str, description: str) -> Path:
        """
        Create a new project markdown file with all standard sections.
        Does NOT overwrite an existing project file.
        Returns the path to the file.
        """
        path = self._project_path(name)
        if path.exists():
            return path  # Don't overwrite existing work

        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"# {name}\n\n"
        content += f"_Created: {ts}_\n\n"
        content += f"## Description\n{description.strip()}\n\n"
        content += "## Current Status\nJust started.\n\n"
        content += "## Architecture\n_Not yet defined._\n\n"
        content += "## Todo List\n- [ ] Define requirements\n- [ ] Break into subtasks\n"

        path.write_text(content, encoding="utf-8")
        return path

    def update_project_file(self, name: str, section: str, content: str,
                             replace: bool = False) -> bool:
        """
        Find a section header (e.g. '## Todo List') in the project file and
        either REPLACE its content or APPEND to it.

        Args:
            name:    Project name
            section: Section title (e.g. 'Todo List', 'Architecture')
            content: Text to insert under the section
            replace: If True, replace existing section content. If False, append.

        Returns:
            True on success, False if project file not found.
        """
        path = self._project_path(name)
        if not path.exists():
            return False

        text = path.read_text(encoding="utf-8")
        header = f"## {section}"

        # Match from the header to the next ## header (or end of file)
        pattern = re.compile(
            rf"(^{re.escape(header)}\s*\n)(.*?)(?=^## |\Z)",
            re.MULTILINE | re.DOTALL,
        )

        if replace:
            replacement = f"## {section}\n{content.strip()}\n\n"
            new_text, count = pattern.subn(replacement, text)
        else:
            # Append: insert before the next section boundary
            def _append(m):
                existing = m.group(2).rstrip()
                return f"## {section}\n{existing}\n{content.strip()}\n\n"
            new_text, count = pattern.subn(_append, text)

        if count == 0:
            # Section not found — add it at the end
            new_text = text.rstrip() + f"\n\n## {section}\n{content.strip()}\n"

        path.write_text(new_text, encoding="utf-8")
        return True

    def get_project(self, name: str) -> Optional[str]:
        """Return full markdown content of a project, or None if not found."""
        path = self._project_path(name)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def list_projects(self) -> List[str]:
        """Return names of all active project files (without extension)."""
        return [p.stem for p in sorted(self.projects_dir.glob("*.md"))]

    def delete_project(self, name: str) -> bool:
        """Move project to a 'archived' subfolder instead of hard-deleting."""
        path = self._project_path(name)
        if not path.exists():
            return False
        archive = self.projects_dir / "archived"
        archive.mkdir(exist_ok=True)
        path.rename(archive / path.name)
        return True


# =============================================================================
# USER PROFILE MD
# =============================================================================

class UserProfileMD:
    """
    Manages jarvis_memory/user_profile.md — a human-readable, growing
    portrait of the user built from emails, conversations, and observations.

    Sections: Identity, Interests, Work Style, Tech Stack, Relationships,
              Observations (catch-all for new insights).
    """

    DEFAULT_SECTIONS = [
        "Identity",
        "Interests & Hobbies",
        "Work Style",
        "Tech Stack",
        "Relationships",
        "Observations",
    ]

    def __init__(self, memory_dir: str = "./jarvis_memory"):
        self.path = Path(memory_dir) / "user_profile.md"
        if not self.path.exists():
            self._create_default()

    def _create_default(self):
        ts = datetime.now().strftime("%Y-%m-%d")
        lines = [f"# User Profile\n\n_Last updated: {ts}_\n"]
        for section in self.DEFAULT_SECTIONS:
            lines.append(f"\n## {section}\n_Nothing recorded yet._\n")
        self.path.write_text("\n".join(lines), encoding="utf-8")

    def append_insight(self, section: str, insight: str, source: str = "conversation"):
        """
        Append a bullet-point insight under the given section.
        Creates the section if it doesn't exist.

        Args:
            section: Which section to append to (e.g. 'Interests & Hobbies')
            insight: One-line fact or observation
            source:  Where this came from ('email', 'conversation', 'observation')
        """
        insight = insight.strip()
        if not insight:
            return

        text = self.path.read_text(encoding="utf-8")
        ts = datetime.now().strftime("%Y-%m-%d")
        bullet = f"- [{source} • {ts}] {insight}"

        header = f"## {section}"
        pattern = re.compile(
            rf"(^{re.escape(header)}\s*\n)(.*?)(?=^## |\Z)",
            re.MULTILINE | re.DOTALL,
        )

        def _append(m):
            body = m.group(2)
            # Remove the placeholder line if present
            body = re.sub(r"_Nothing recorded yet\._\s*", "", body)
            return f"## {section}\n{body.rstrip()}\n{bullet}\n\n"

        new_text, count = pattern.subn(_append, text)

        if count == 0:
            # Section doesn't exist — add it
            new_text = text.rstrip() + f"\n\n## {section}\n{bullet}\n"

        # Update the "last updated" timestamp
        new_text = re.sub(
            r"_Last updated:.*?_",
            f"_Last updated: {ts}_",
            new_text,
        )

        self.path.write_text(new_text, encoding="utf-8")

    def read(self) -> str:
        """Return the full profile markdown."""
        return self.path.read_text(encoding="utf-8")

    def get_section(self, section: str) -> str:
        """Return just the content of one section, or empty string."""
        text = self.path.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"^## {re.escape(section)}\s*\n(.*?)(?=^## |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        m = pattern.search(text)
        return m.group(1).strip() if m else ""


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n🧪 Testing JARVIS Memory System - Phase 1\n")
    
    # Initialize
    memory = MemoryManager()
    
    # Test 1: User Profile
    print("\n--- Test 1: User Profile ---")
    memory.initialize_user_profile(
        name="Grant",
        preferred_name="sir",
        os="macOS",
        terminal="zsh",
        editor="VS Code",
        timezone="America/Phoenix"
    )
    
    profile = memory.get_user_profile()
    print(f"Profile: {profile}")
    
    # Test 2: Preferences
    print("\n--- Test 2: Preferences ---")
    memory.set_preference("coding", "style", "pythonic")
    memory.set_preference("coding", "language_preference", "Python > JavaScript")
    memory.set_preference("communication", "tone", "direct")
    memory.set_preference("communication", "formality", "0.8")
    memory.set_preference("security", "level", "paranoid")
    
    prefs = memory.get_preferences()
    print(f"All preferences: {json.dumps(prefs, indent=2)}")
    
    # Test 3: Projects
    print("\n--- Test 3: Projects ---")
    memory.add_project(
        "AtomosOps",
        description="Infrastructure and automation platform",
        tech_stack="Python, Docker, Kubernetes",
        status="active",
        priority=10
    )
    
    memory.add_project(
        "JARVIS",
        description="AI assistant with personality",
        tech_stack="Python, Ollama, Whisper",
        status="active",
        priority=9
    )
    
    projects = memory.get_active_projects()
    print(f"Active projects: {len(projects)}")
    for p in projects:
        print(f"  - {p['name']}: {p['description']}")
    
    # Test 4: Entities
    print("\n--- Test 4: Entities ---")
    memory.remember_entity("arch01", "server", "primary production server", 
                          "10.0.0.58, runs AtomosOps services")
    memory.remember_entity("Alice", "person", "team member", 
                          "DevOps engineer, works on deployments")
    memory.remember_entity("Claude API", "service", "AI service", 
                          "Used for deep reasoning tasks")
    
    servers = memory.get_entities("server")
    print(f"Known servers: {[s['name'] for s in servers]}")
    
    # Test 5: Habits
    print("\n--- Test 5: Habits ---")
    memory.track_habit("deploy → vulnerability scan", "After production deployment")
    memory.track_habit("deploy → vulnerability scan", "After production deployment")
    memory.track_habit("deploy → vulnerability scan", "After production deployment")
    memory.track_habit("git commit → git push", "After code changes")
    memory.track_habit("git commit → git push", "After code changes")
    
    habits = memory.get_common_habits()
    print(f"Common patterns:")
    for h in habits:
        print(f"  - {h['pattern']} (x{h['frequency']})")
    
    # Test 6: Stats
    print("\n--- Test 6: Memory Stats ---")
    stats = memory.get_memory_stats()
    print(f"Memory statistics: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    memory.close()
    
    print("\n✅ Phase 1 tests complete!\n")