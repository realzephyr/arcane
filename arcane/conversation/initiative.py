"""Starting conversations.

:class:`InitiativePlanner` decides whether a personality should open a new
conversation in a channel. It is conservative by design: the channel must be
opted in, quiet for a while but not dead, the bot must not have spoken last,
and daily caps, minimum intervals, and a random chance all apply. Initiations
are logged in the database (:class:`InitiativeLog`) so caps survive restarts
and topics don't repeat.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from arcane.core.clock import from_timestamp, to_timestamp, utcnow
from arcane.database.database import Database
from arcane.memory.short_term import ChannelActivity
from arcane.personalities.base import InitiativeProfile

INITIATIVE_RETENTION = timedelta(days=14)


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    should_initiate: bool
    reason: str


class InitiativePlanner:
    """Decides when to start a conversation and what about."""

    def __init__(self, profile: InitiativeProfile, *, rng: random.Random | None = None) -> None:
        self._profile = profile
        self._rng = rng or random.Random()

    @property
    def check_interval(self) -> timedelta:
        return timedelta(minutes=self._profile.check_interval_minutes)

    def evaluate(
        self,
        *,
        now: datetime,
        activity: ChannelActivity,
        conversation_active: bool,
        initiations_today: int,
        last_initiation_at: datetime | None,
    ) -> InitiativeDecision:
        profile = self._profile
        if not profile.enabled:
            return InitiativeDecision(False, "disabled")
        if not self._within_active_hours(now):
            return InitiativeDecision(False, "outside active hours")
        if conversation_active:
            return InitiativeDecision(False, "conversation in progress")
        if activity.last_message_at is None:
            return InitiativeDecision(False, "no recent activity")
        if activity.last_message_is_self:
            return InitiativeDecision(False, "last message was ours")
        if now - activity.last_message_at < timedelta(minutes=profile.min_quiet_minutes):
            return InitiativeDecision(False, "channel not quiet long enough")
        if activity.last_human_message_at is None or now - activity.last_human_message_at > (
            timedelta(hours=profile.recent_activity_hours)
        ):
            return InitiativeDecision(False, "nobody has been around recently")
        if initiations_today >= profile.max_per_channel_per_day:
            return InitiativeDecision(False, "daily limit reached")
        if last_initiation_at is not None and now - last_initiation_at < timedelta(
            minutes=profile.min_interval_minutes
        ):
            return InitiativeDecision(False, "too soon since the last opener")
        if self._rng.random() >= profile.chance:
            return InitiativeDecision(False, "chance")
        return InitiativeDecision(True, "conditions met")

    def choose_topic(self, topics: Sequence[str], recent: Sequence[str]) -> str | None:
        """Pick a topic, avoiding recently used ones while alternatives remain."""
        if not topics:
            return None
        recent_set = set(recent)
        fresh = [topic for topic in topics if topic not in recent_set]
        return self._rng.choice(fresh or list(topics))

    def _within_active_hours(self, now: datetime) -> bool:
        window = self._profile.active_hours_utc
        if window is None:
            return True
        start, end = window
        hour = now.hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # window wraps past midnight


class InitiativeLog:
    """Persistent record of conversations a bot started."""

    def __init__(self, database: Database, bot_id: str) -> None:
        self._db = database
        self.bot_id = bot_id

    async def record(self, channel_id: int, topic: str, now: datetime | None = None) -> None:
        await self._db.execute(
            "INSERT INTO initiatives (bot_id, channel_id, topic, created_at) VALUES (?, ?, ?, ?)",
            (self.bot_id, channel_id, topic, to_timestamp(now or utcnow())),
        )

    async def count_since(self, channel_id: int, since: datetime) -> int:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) FROM initiatives WHERE bot_id = ? AND channel_id = ? "
            "AND created_at >= ?",
            (self.bot_id, channel_id, to_timestamp(since)),
        )
        return int(row[0]) if row is not None else 0

    async def last_at(self, channel_id: int) -> datetime | None:
        row = await self._db.fetch_one(
            "SELECT MAX(created_at) FROM initiatives WHERE bot_id = ? AND channel_id = ?",
            (self.bot_id, channel_id),
        )
        return from_timestamp(row[0]) if row is not None and row[0] is not None else None

    async def recent_topics(self, limit: int = 10) -> list[str]:
        rows = await self._db.fetch_all(
            "SELECT topic FROM initiatives WHERE bot_id = ? ORDER BY created_at DESC LIMIT ?",
            (self.bot_id, limit),
        )
        return [row["topic"] for row in rows]

    async def prune(self, now: datetime | None = None) -> int:
        cutoff = (now or utcnow()) - INITIATIVE_RETENTION
        removed = await self._db.execute(
            "DELETE FROM initiatives WHERE bot_id = ? AND created_at < ?",
            (self.bot_id, to_timestamp(cutoff)),
        )
        return max(removed, 0)
