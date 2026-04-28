"""Daily reset scheduler (BUILD_SPEC §17 conversation.reset).

Wraps APScheduler's ``BackgroundScheduler`` with two cron jobs:
  * ``dm_daily_at`` — sweeps DM / CLI / heartbeat conversations.
  * ``group_daily_at`` — sweeps group conversations.

Idle resets are passive (checked inside ``Conversation.open`` whenever a chat
request comes in) — no timer = no work when nobody is talking. The cron jobs
exist to sweep boundaries forward when the user is silent, so the next
morning's first message starts a fresh conversation.

The callback signature is ``on_reset(channel_kinds: list[str])`` — the
scheduler passes the targeted set of channel kinds, the caller decides what
to do (close stale rows, generate slug+summary, etc.). The callback runs on
APScheduler's worker thread.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from jarvis.core.conversation import ConversationConfig

logger = logging.getLogger(__name__)

__all__ = ["ResetScheduler"]


# Channel-kind groupings — kept in sync with ``ConversationConfig.daily_at_for``.
_DM_KINDS: list[str] = ["dm", "cli", "heartbeat"]
_GROUP_KINDS: list[str] = ["group"]


class ResetScheduler:
    """APScheduler wrapper for the two daily-reset cron jobs.

    ``on_reset(channel_kinds)`` is called at each fire time. Implementations
    typically iterate ``list_open_conversations(...)`` and close stale rows
    (with or without an LLM-generated slug+summary, depending on phase).
    """

    def __init__(
        self,
        cfg: ConversationConfig,
        on_reset: Callable[[list[str]], None],
    ) -> None:
        self._cfg = cfg
        self._on_reset = on_reset
        self._scheduler = BackgroundScheduler()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        dm_h, dm_m = (int(x) for x in self._cfg.dm_daily_at.split(":", 1))
        gp_h, gp_m = (int(x) for x in self._cfg.group_daily_at.split(":", 1))
        self._scheduler.add_job(
            self._fire_dm,
            CronTrigger(hour=dm_h, minute=dm_m),
            id="jarvis-reset-dm",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._fire_group,
            CronTrigger(hour=gp_h, minute=gp_m),
            id="jarvis-reset-group",
            replace_existing=True,
        )
        self._scheduler.start()
        self._started = True
        logger.info(
            "scheduler: daily resets dm=%s group=%s",
            self._cfg.dm_daily_at,
            self._cfg.group_daily_at,
        )

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            logger.exception("scheduler: shutdown raised")
        self._started = False

    # -- callbacks (kept simple so tests can fire them directly) ----------

    def _fire_dm(self) -> None:
        try:
            self._on_reset(list(_DM_KINDS))
        except Exception:
            logger.exception("scheduler: dm reset callback raised")

    def _fire_group(self) -> None:
        try:
            self._on_reset(list(_GROUP_KINDS))
        except Exception:
            logger.exception("scheduler: group reset callback raised")
