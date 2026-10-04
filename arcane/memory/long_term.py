"""Long-term memory: user profiles and durable, important facts.

Messages are never stored here. Instead, small statements ("studies physics",
"thinks free will is an illusion") are written by a :class:`MemoryExtractor`
when a conversation ends, or directly through :meth:`LongTermMemory.remember`.

Growth is bounded: each user has a memory cap, and when it is exceeded the
lowest-value memories (by importance and recency) are evicted. Memories can
also expire. Duplicate or near-duplicate memories reinforce the existing entry
instead of creating a new one.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

import aiosqlite

from arcane.core.clock import from_timestamp, to_timestamp, utcnow
from arcane.database.database import Database
from arcane.database.models import MemoryKind, MemoryRecord, UserProfile
from arcane.memory.filters import contains_sensitive_data, normalize_memory_text, similarity

logger = logging.getLogger(__name__)

DUPLICATE_SIMILARITY = 0.75
RECENCY_HALF_LIFE_DAYS = 30.0
IMPORTANCE_WEIGHT = 0.75
MIN_MEMORY_CHARS = 4


def memory_score(record: MemoryRecord, now: datetime) -> float:
    """Value of a memory: mostly importance, partly how recently it was reinforced."""
    age_days = max((now - record.updated_at).total_seconds(), 0.0) / 86_400
    recency = math.exp(-math.log(2) * age_days / RECENCY_HALF_LIFE_DAYS)
    return IMPORTANCE_WEIGHT * record.importance + (1 - IMPORTANCE_WEIGHT) * recency


class LongTermMemory:
    """Durable memory for one bot."""

    def __init__(
        self,
        database: Database,
        bot_id: str,
        *,
        max_memories_per_user: int = 40,
        max_general_memories: int = 100,
    ) -> None:
        self._db = database
        self.bot_id = bot_id
        self.max_memories_per_user = max_memories_per_user
        self.max_general_memories = max_general_memories

    # ----------------------------------------------------------------- profiles

    async def touch_user(
        self,
        user_id: int,
        display_name: str,
        *,
        interaction: bool = False,
        now: datetime | None = None,
    ) -> None:
        """Create or refresh a user's profile. ``interaction`` counts an exchange."""
        timestamp = to_timestamp(now or utcnow())
        await self._db.execute(
            """
            INSERT INTO user_profiles (
                bot_id, user_id, display_name, first_seen_at, last_seen_at, interaction_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (bot_id, user_id) DO UPDATE SET
                display_name = excluded.display_name,
                last_seen_at = excluded.last_seen_at,
                interaction_count = interaction_count + excluded.interaction_count
            """,
            (self.bot_id, user_id, display_name, timestamp, timestamp, int(interaction)),
        )

    async def get_profile(self, user_id: int) -> UserProfile | None:
        row = await self._db.fetch_one(
            "SELECT * FROM user_profiles WHERE bot_id = ? AND user_id = ?",
            (self.bot_id, user_id),
        )
        if row is None:
            return None
        return UserProfile(
            bot_id=row["bot_id"],
            user_id=row["user_id"],
            display_name=row["display_name"],
            first_seen_at=from_timestamp(row["first_seen_at"]),
            last_seen_at=from_timestamp(row["last_seen_at"]),
            interaction_count=row["interaction_count"],
        )

    # ----------------------------------------------------------------- memories

    async def remember(
        self,
        content: str,
        *,
        user_id: int | None = None,
        kind: MemoryKind = MemoryKind.NOTE,
        importance: float = 0.5,
        source: str | None = None,
        ttl: timedelta | None = None,
        now: datetime | None = None,
    ) -> MemoryRecord | None:
        """Store a memory, reinforcing a near-duplicate if one exists.

        Returns the stored (or reinforced) record, or ``None`` if the content was
        rejected as empty or sensitive.
        """
        now = now or utcnow()
        text = normalize_memory_text(content)
        if len(text) < MIN_MEMORY_CHARS:
            return None
        if contains_sensitive_data(text):
            logger.info("[%s] rejected a memory that looked sensitive", self.bot_id)
            return None
        importance = min(max(importance, 0.0), 1.0)
        expires_at = to_timestamp(now + ttl) if ttl is not None else None

        existing = await self._memories_for(user_id)
        duplicate = max(
            existing,
            key=lambda record: similarity(record.content, text),
            default=None,
        )
        if duplicate is not None and similarity(duplicate.content, text) >= DUPLICATE_SIMILARITY:
            await self._db.execute(
                """
                UPDATE memories SET
                    content = ?, importance = ?, updated_at = ?, kind = ?,
                    expires_at = CASE WHEN ? IS NULL OR expires_at IS NULL THEN NULL
                                      ELSE MAX(expires_at, ?) END
                WHERE id = ?
                """,
                (
                    text,
                    max(duplicate.importance, importance),
                    to_timestamp(now),
                    kind.value,
                    expires_at,
                    expires_at,
                    duplicate.id,
                ),
            )
            return await self._get(duplicate.id)

        memory_id = await self._db.execute_returning_id(
            """
            INSERT INTO memories (
                bot_id, user_id, kind, content, importance, created_at, updated_at,
                expires_at, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.bot_id,
                user_id,
                kind.value,
                text,
                importance,
                to_timestamp(now),
                to_timestamp(now),
                expires_at,
                source,
            ),
        )
        await self._enforce_cap(user_id, now)
        return await self._get(memory_id)

    async def recall(
        self,
        user_id: int | None,
        *,
        limit: int = 8,
        now: datetime | None = None,
        touch: bool = True,
    ) -> list[MemoryRecord]:
        """The most valuable memories about ``user_id`` (``None`` = general)."""
        if limit <= 0:
            return []
        now = now or utcnow()
        records = [
            record
            for record in await self._memories_for(user_id)
            if record.expires_at is None or record.expires_at > now
        ]
        records.sort(key=lambda record: memory_score(record, now), reverse=True)
        selected = records[:limit]
        if touch and selected:
            placeholders = ",".join("?" for _ in selected)
            await self._db.execute(
                f"UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 "  # noqa: S608 - placeholders only
                f"WHERE id IN ({placeholders})",
                (to_timestamp(now), *(record.id for record in selected)),
            )
        return selected

    async def forget(self, memory_id: int) -> bool:
        changed = await self._db.execute(
            "DELETE FROM memories WHERE bot_id = ? AND id = ?", (self.bot_id, memory_id)
        )
        return changed > 0

    async def forget_user(self, user_id: int) -> int:
        """Delete every memory and the profile for ``user_id``. Returns rows removed."""
        async with self._db.transaction() as connection:
            memories = await connection.execute(
                "DELETE FROM memories WHERE bot_id = ? AND user_id = ?", (self.bot_id, user_id)
            )
            profiles = await connection.execute(
                "DELETE FROM user_profiles WHERE bot_id = ? AND user_id = ?",
                (self.bot_id, user_id),
            )
            return max(memories.rowcount, 0) + max(profiles.rowcount, 0)

    async def prune(self, now: datetime | None = None) -> int:
        """Delete expired memories. Returns the number removed."""
        removed = await self._db.execute(
            "DELETE FROM memories WHERE bot_id = ? AND expires_at IS NOT NULL AND expires_at <= ?",
            (self.bot_id, to_timestamp(now or utcnow())),
        )
        return max(removed, 0)

    # ---------------------------------------------------------------- internals

    async def _memories_for(self, user_id: int | None) -> list[MemoryRecord]:
        if user_id is None:
            rows = await self._db.fetch_all(
                "SELECT * FROM memories WHERE bot_id = ? AND user_id IS NULL", (self.bot_id,)
            )
        else:
            rows = await self._db.fetch_all(
                "SELECT * FROM memories WHERE bot_id = ? AND user_id = ?", (self.bot_id, user_id)
            )
        return [_memory_from_row(row) for row in rows]

    async def _get(self, memory_id: int) -> MemoryRecord | None:
        row = await self._db.fetch_one("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return _memory_from_row(row) if row is not None else None

    async def _enforce_cap(self, user_id: int | None, now: datetime) -> None:
        cap = self.max_general_memories if user_id is None else self.max_memories_per_user
        records = await self._memories_for(user_id)
        overflow = len(records) - cap
        if overflow <= 0:
            return
        records.sort(key=lambda record: memory_score(record, now))
        victims = [record.id for record in records[:overflow]]
        placeholders = ",".join("?" for _ in victims)
        await self._db.execute(
            f"DELETE FROM memories WHERE id IN ({placeholders})",  # noqa: S608 - placeholders only
            victims,
        )
        logger.debug("[%s] evicted %d low-value memories", self.bot_id, len(victims))


def _memory_from_row(row: aiosqlite.Row) -> MemoryRecord:
    return MemoryRecord(
        id=row["id"],
        bot_id=row["bot_id"],
        user_id=row["user_id"],
        kind=MemoryKind.parse(row["kind"]),
        content=row["content"],
        importance=row["importance"],
        created_at=from_timestamp(row["created_at"]),
        updated_at=from_timestamp(row["updated_at"]),
        last_accessed_at=(
            from_timestamp(row["last_accessed_at"]) if row["last_accessed_at"] is not None else None
        ),
        access_count=row["access_count"],
        expires_at=from_timestamp(row["expires_at"]) if row["expires_at"] is not None else None,
        source=row["source"],
    )
