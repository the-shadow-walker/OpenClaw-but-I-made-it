#!/usr/bin/env python3
"""
patch_bugs.py — Fixes 5 bugs in the JARVIS server files.
Run from /mnt/storage/NAS/Jarvis/Jarvis/

Bugs fixed:
  #3: Action tag regex broken on nested brackets
  #4: Deep search errors embed in chat response stream
  #5: Personality learned but never applied to prompts
  #6: sessions.db grows forever (cleanup never called)
  #7: Background worker race condition on _last_work_at
"""
from pathlib import Path

MAIN = Path("server/main.py")
BG   = Path("workers/background_worker.py")

print("=" * 60)
print("JARVIS Bug Fix Patch")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Patch server/main.py
# ─────────────────────────────────────────────────────────────────────────────
text = MAIN.read_text()
original_len = len(text)
print(f"\nPatching {MAIN} ({original_len} chars)...")

# ── Bug #3: Fix action-tag regexes to allow one level of nested brackets ──────
# Pattern ([^\]]+) stops at first ], so [REMEMBER: Deploy to [arch01]] breaks.
# New: ((?:[^\[\]]|\[[^\]]*\])+) allows one inner [bracket] level.
NESTED = r'(?:[^\[\]]|\[[^\]]*\])+'

old_new_patterns = [
    (r"r'\[REMEMBER:\s*([^\]]+)\]'",          f"r'\\[REMEMBER:\\s*({NESTED})\\]'"),
    (r"r'\[SEARCH_MEMORY:\s*([^\]]+)\]'",     f"r'\\[SEARCH_MEMORY:\\s*({NESTED})\\]'"),
    (r"r'\[MEMORY_SHOW_ABOUT:\s*([^\]]+)\]'", f"r'\\[MEMORY_SHOW_ABOUT:\\s*({NESTED})\\]'"),
    (r"r'\[MEMORY_FORGET:\s*([^\]]+)\]'",     f"r'\\[MEMORY_FORGET:\\s*({NESTED})\\]'"),
    (r"r'\[MEMORY_STORE_PREF:\s*([^\]]+)\]'", f"r'\\[MEMORY_STORE_PREF:\\s*({NESTED})\\]'"),
    (r"r'\[EXECUTE:\s*([^\]]+)\]'",           f"r'\\[EXECUTE:\\s*({NESTED})\\]'"),
    (r"r'\[SEND_EMAIL:\s*([^\]]+)\]'",        f"r'\\[SEND_EMAIL:\\s*({NESTED})\\]'"),
    (r"r'\[DRAFT_EMAIL:\s*([^\]]+)\]'",       f"r'\\[DRAFT_EMAIL:\\s*({NESTED})\\]'"),
    (r"r'\[DEEP_SEARCH:\s*([^\]]+)\]'",       f"r'\\[DEEP_SEARCH:\\s*({NESTED})\\]'"),
    (r"r'\[RUN_AGENT:\s*([^\]]+)\]'",         f"r'\\[RUN_AGENT:\\s*({NESTED})\\]'"),
    (r"r'\[RUN_CHAIN:\s*([^\]]+)\]'",         f"r'\\[RUN_CHAIN:\\s*({NESTED})\\]'"),
]

fixed_count = 0
for old, new in old_new_patterns:
    if old in text:
        text = text.replace(old, new)
        fixed_count += 1

print(f"  {'✓' if fixed_count > 0 else '✗'} Bug #3: Fixed {fixed_count}/11 action-tag regex patterns for nested bracket support")

# ── Bug #4: Deep search connectivity errors should NOT embed in chat ──────────
# Fix DEEP_SEARCH submission errors (no job_id, connection error, generic)
old_ds_submit_err = '''\
                else:
                    response = response.replace(match.group(0), "\\n❌ Failed to start deep search (no job ID)")
                    logger.error(f"[Action] Deep search failed: no job_id in response")

            except requests.exceptions.RequestException as e:
                error_msg = f"\\n❌ Deep search unavailable: {e}"
                response = response.replace(match.group(0), error_msg)
                logger.error(f"[Action] Deep search connection error: {e}")
            except Exception as e:
                error_msg = f"\\n❌ Deep search error: {e}"
                response = response.replace(match.group(0), error_msg)
                logger.error(f"[Action] Deep search error: {e}")'''

new_ds_submit_err = '''\
                else:
                    response = response.replace(match.group(0), "")
                    logger.error(f"[Action] Deep search failed: no job_id in response")

            except requests.exceptions.RequestException as e:
                response = response.replace(match.group(0), "")
                logger.error(f"[Action] Deep search unavailable: {e}")
            except Exception as e:
                response = response.replace(match.group(0), "")
                logger.error(f"[Action] Deep search error: {e}")'''

if old_ds_submit_err in text:
    text = text.replace(old_ds_submit_err, new_ds_submit_err)
    print("  ✓ Bug #4a: DEEP_SEARCH submission errors silenced from chat")
else:
    print("  ✗ Bug #4a: Could not find DEEP_SEARCH submission error block")

# Fix GET_DEEP_SEARCH_RESULT connectivity errors
old_get_ds_err = '''\
            except requests.exceptions.RequestException as e:
                error_msg = f"\\n❌ Could not fetch search result: {e}"
                response = response.replace(match.group(0), error_msg)
                logger.error(f"[Action] Failed to fetch deep search result: {e}")
            except Exception as e:
                error_msg = f"\\n❌ Error fetching result: {e}"
                response = response.replace(match.group(0), error_msg)
                logger.error(f"[Action] Error in GET_DEEP_SEARCH_RESULT: {e}")'''

new_get_ds_err = '''\
            except requests.exceptions.RequestException as e:
                response = response.replace(match.group(0), "")
                logger.error(f"[Action] Failed to fetch deep search result: {e}")
            except Exception as e:
                response = response.replace(match.group(0), "")
                logger.error(f"[Action] Error in GET_DEEP_SEARCH_RESULT: {e}")'''

if old_get_ds_err in text:
    text = text.replace(old_get_ds_err, new_get_ds_err)
    print("  ✓ Bug #4b: GET_DEEP_SEARCH_RESULT connectivity errors silenced from chat")
else:
    print("  ✗ Bug #4b: Could not find GET_DEEP_SEARCH_RESULT error block")

# ── Bug #5: Inject personality traits into build_context() ───────────────────
old_personality_anchor = '''\
        # EMAIL SUMMARY (brief - use [READ_RECENT_EMAILS] for full details)
        if email_summary:'''

new_personality_anchor = '''\
        # PERSONALITY ADAPTATIONS (learned from interactions via PersonalityLearner)
        try:
            pref_data = getattr(self.personality, 'data', {}).get('preferences', {})
            if pref_data:
                context_parts.append("\\nLEARNED PERSONALITY ADAPTATIONS:")
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

        # EMAIL SUMMARY (brief - use [READ_RECENT_EMAILS] for full details)
        if email_summary:'''

if old_personality_anchor in text:
    text = text.replace(old_personality_anchor, new_personality_anchor)
    print("  ✓ Bug #5: Personality traits injected into build_context()")
else:
    print("  ✗ Bug #5: Could not find personality injection anchor in build_context()")

MAIN.write_text(text)
print(f"\n  server/main.py: {original_len} → {len(text)} chars")

# ─────────────────────────────────────────────────────────────────────────────
# Patch workers/background_worker.py  (bugs #6 and #7)
# ─────────────────────────────────────────────────────────────────────────────
bg_text = BG.read_text()
bg_original_len = len(bg_text)
print(f"\nPatching {BG} ({bg_original_len} chars)...")

# ── Bug #7: Add threading.Lock for _last_work_at ─────────────────────────────
old_init = '''\
        self._last_work_at = 0.0
        self._last_refine_time = 0.0
        self._last_user_interaction = time.time()
        self._running = True'''

new_init = '''\
        self._last_work_at = 0.0
        self._last_refine_time = 0.0
        self._last_user_interaction = time.time()
        self._running = True
        self._work_lock = threading.Lock()'''

if old_init in bg_text:
    bg_text = bg_text.replace(old_init, new_init)
    print("  ✓ Bug #7a: Added threading.Lock() for work state protection")
else:
    print("  ✗ Bug #7a: Could not find __init__ block for lock addition")

# Guard _cooldown_done with the lock
old_cooldown = '''\
    def _cooldown_done(self) -> bool:
        return (time.time() - self._last_work_at) >= self.WORK_COOLDOWN'''

new_cooldown = '''\
    def _cooldown_done(self) -> bool:
        with self._work_lock:
            return (time.time() - self._last_work_at) >= self.WORK_COOLDOWN'''

if old_cooldown in bg_text:
    bg_text = bg_text.replace(old_cooldown, new_cooldown)
    print("  ✓ Bug #7b: _cooldown_done() reads _last_work_at under lock")
else:
    print("  ✗ Bug #7b: Could not find _cooldown_done")

# Guard _last_work_at write with the lock (only first occurrence = in _do_background_work)
old_work_at = '        self._last_work_at = time.time()\n\n    # ── Email Self-Refinement'
new_work_at = '        with self._work_lock:\n            self._last_work_at = time.time()\n\n    # ── Email Self-Refinement'

if old_work_at in bg_text:
    bg_text = bg_text.replace(old_work_at, new_work_at)
    print("  ✓ Bug #7c: _last_work_at write protected by lock")
else:
    # Fallback: simpler pattern
    old_work_at2 = '        self._last_work_at = time.time()'
    new_work_at2 = '        with self._work_lock:\n            self._last_work_at = time.time()'
    if old_work_at2 in bg_text:
        bg_text = bg_text.replace(old_work_at2, new_work_at2, 1)
        print("  ✓ Bug #7c: _last_work_at write protected by lock (fallback match)")
    else:
        print("  ✗ Bug #7c: Could not find _last_work_at assignment")

# ── Bug #6: Call cleanup_expired_sessions() in run() loop ────────────────────
old_run = '''\
    def run(self):
        """Main background worker loop"""
        while self._running:
            try:
                if self._is_idle() and self._cooldown_done():
                    self.autonomous_mode()
            except Exception as e:
                print(f"   ⚠️  [BG] Worker error: {e}")
            time.sleep(self.POLL_INTERVAL)'''

new_run = '''\
    def run(self):
        """Main background worker loop"""
        _last_session_cleanup = 0.0
        while self._running:
            try:
                if self._is_idle() and self._cooldown_done():
                    self.autonomous_mode()
            except Exception as e:
                print(f"   ⚠️  [BG] Worker error: {e}")

            # Periodic session cleanup every 30 minutes
            if time.time() - _last_session_cleanup > 1800:
                try:
                    self.server.sessions.cleanup_expired_sessions()
                    _last_session_cleanup = time.time()
                except Exception as _ce:
                    pass

            time.sleep(self.POLL_INTERVAL)'''

if old_run in bg_text:
    bg_text = bg_text.replace(old_run, new_run)
    print("  ✓ Bug #6: Session cleanup call added to run() loop (every 30 min)")
else:
    print("  ✗ Bug #6: Could not find run() loop body to add session cleanup")

BG.write_text(bg_text)
print(f"\n  workers/background_worker.py: {bg_original_len} → {len(bg_text)} chars")

print("\n" + "=" * 60)
print("Bug patch complete.")
print("Restart jarvis: sudo systemctl restart jarvis")
print("=" * 60)
