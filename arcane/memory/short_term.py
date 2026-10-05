"""Short-term memory: recent messages and live conversation state.

Every message in a channel a bot is allowed to read is stored so prompts have
accurate, ordered context including messages from other people and reply
relationships. Storage is bounded two ways, both enforced by :meth:`prune`:

* **age**: messages older than the TTL are deleted;
* **volume**: each channel keeps at most ``max_messages_per_channel`` rows.

Conversation state (who the bot is talking to and since when) is persisted
too, so it survives restarts, and is deleted when the conversation expires.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiosqlite

from arcane.core.clock import from_timestamp, to_timestamp, utcnow
from arcane.core.models import HistoryMessage, IncomingMessage
from arcane.database.database import Database
from arcane.database.models import ConversationRecord

logger = logging.getLogger(__name__)

MAX_STORED_CONTENT_CHARS = 4000


@dataclass(frozen=True, slots=True)
class ChannelActivity:
    """Summary of recent activity in a channel."""

    last_message_at: datetime | None = None
    last_message_is_self: bool = False
    last_human_message_at: datetime | None = None
    last_self_message_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PruneStats:
    expired_messages: int = 0
    overflow_messages: int = 0

    @property
    def total(self) -> int:
        return self.expired_messages + self.overflow_messages


class ShortTermMemory:
    """Recent messages and conversation state for one bot."""

    def __init__(
        self,
        database: Database,
        bot_id: str,
        *,
        ttl: timedelta = timedelta(hours=24),
        max_messages_per_channel: int = 300,
    ) -> None:
        self._db = database
        self.bot_id = bot_id
        self.ttl = ttl
        self.max_messages_per_channel = max_messages_per_channel

    # ----------------------------------------------------------------- messages

    async def record_message(self, message: IncomingMessage) -> None:
        """Store (or update) a message observed in a channel."""
        reply = message.reply_to
        await self._db.execute(
            """
            INSERT INTO messages (
                bot_id, message_id, channel_id, guild_id, author_id, author_name,
                author_is_bot, is_self, content, reply_to_message_id,
                reply_to_author_name, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (bot_id, message_id) DO UPDATE SET content = excluded.content
            """,
            (
                self.bot_id,
                message.message_id,
                message.channel_id,
                message.channel.guild_id,
                message.author_id,
                message.author_name,
                int(message.author_is_bot),
                int(message.is_self),
                message.text_for_prompt()[:MAX_STORED_CONTENT_CHARS],
                reply.message_id if reply else None,
                reply.author_name if reply else None,
                to_timestamp(message.created_at),
            ),
        )

    async def update_message_content(self, message_id: int, content: str) -> bool:
        """Apply an edit. Returns False if the message is not stored."""
        changed = await self._db.execute(
            "UPDATE messages SET content = ? WHERE bot_id = ? AND message_id = ?",
            (content[:MAX_STORED_CONTENT_CHARS], self.bot_id, message_id),
        )
        return changed > 0

    async def delete_message(self, message_id: int) -> bool:
        """Forget a message (e.g. deleted on Discord)."""
        changed = await self._db.execute(
            "DELETE FROM messages WHERE bot_id = ? AND message_id = ?",
            (self.bot_id, message_id),
        )
        return changed > 0

    async def recent_messages(
        self,
        channel_id: int,
        limit: int,
        *,
        since: datetime | None = None,
    ) -> list[HistoryMessage]:
        """The newest ``limit`` messages in a channel, oldest first."""
        cutoff = since if since is not None else utcnow() - self.ttl
        rows = await self._db.fetch_all(
            """
            SELECT * FROM messages
            WHERE bot_id = ? AND channel_id = ? AND created_at >= ?
            ORDER BY created_at DESC, message_id DESC
            LIMIT ?
            """,
            (self.bot_id, channel_id, to_timestamp(cutoff), max(limit, 0)),
        )
        return [_history_from_row(row) for row in reversed(rows)]

    async def messages_between(
        self, channel_id: int, start: datetime, end: datetime, *, limit: int = 200
    ) -> list[HistoryMessage]:
        """Messages in ``[start, end]``, oldest first (used for memory extraction)."""
        rows = await self._db.fetch_all(
            """
            SELECT * FROM messages
            WHERE bot_id = ? AND channel_id = ? AND created_at BETWEEN ? AND ?
            ORDER BY created_at ASC, message_id ASC
            LIMIT ?
            """,
            (self.bot_id, channel_id, to_timestamp(start), to_timestamp(end), limit),
        )
        return [_history_from_row(row) for row in rows]

    async def channel_activity(self, channel_id: int) -> ChannelActivity:
        row = await self._db.fetch_one(
            """
            SELECT
                (SELECT MAX(created_at) FROM messages
                  WHERE bot_id = :bot AND channel_id = :channel) AS last_at,
                (SELECT is_self FROM messages
                  WHERE bot_id = :bot AND channel_id = :channel
                  ORDER BY created_at DESC, message_id DESC LIMIT 1) AS last_is_self,
                (SELECT MAX(created_at) FROM messages
                  WHERE bot_id = :bot AND channel_id = :channel
                    AND author_is_bot = 0) AS last_human_at,
                (SELECT MAX(created_at) FROM messages
                  WHERE bot_id = :bot AND channel_id = :channel AND is_self = 1) AS last_self_at
            """,
            {"bot": self.bot_id, "channel": channel_id},
        )
        if row is None:
            return ChannelActivity()
        return ChannelActivity(
            last_message_at=_optional_dt(row["last_at"]),
            last_message_is_self=bool(row["last_is_self"]),
            last_human_message_at=_optional_dt(row["last_human_at"]),
            last_self_message_at=_optional_dt(row["last_self_at"]),
        )

    # ------------------------------------------------------------- conversations

    async def save_conversation(self, record: ConversationRecord) -> None:
        await self._db.execute(
            """
            INSERT INTO conversations (
                bot_id, channel_id, guild_id, partner_id, partner_name, participants,
                started_at, last_activity_at, bot_turns, user_turns, awaiting_reply_since
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (bot_id, channel_id) DO UPDATE SET
                guild_id = excluded.guild_id,
                partner_id = excluded.partner_id,
                partner_name = excluded.partner_name,
                participants = excluded.participants,
                started_at = excluded.started_at,
                last_activity_at = excluded.last_activity_at,
                bot_turns = excluded.bot_turns,
                user_turns = excluded.user_turns,
                awaiting_reply_since = excluded.awaiting_reply_since
            """,
            (
                self.bot_id,
                record.channel_id,
                record.guild_id,
                record.partner_id,
                record.partner_name,
                json.dumps({str(k): v for k, v in record.participants.items()}),
                to_timestamp(record.started_at),
                to_timestamp(record.last_activity_at),
                record.bot_turns,
                record.user_turns,
                _optional_ts(record.awaiting_reply_since),
            ),
        )

    async def load_conversations(self) -> list[ConversationRecord]:
        rows = await self._db.fetch_all(
            "SELECT * FROM conversations WHERE bot_id = ?", (self.bot_id,)
        )
        return [_conversation_from_row(row) for row in rows]

    async def delete_conversation(self, channel_id: int) -> None:
        await self._db.execute(
            "DELETE FROM conversations WHERE bot_id = ? AND channel_id = ?",
            (self.bot_id, channel_id),
        )

    # -------------------------------------------------------------- maintenance

    async def prune(self, now: datetime | None = None) -> PruneStats:
        """Delete expired messages and enforce the per-channel cap."""
        now = now or utcnow()
        expired = await self._db.execute(
            "DELETE FROM messages WHERE bot_id = ? AND created_at < ?",
            (self.bot_id, to_timestamp(now - self.ttl)),
        )
        overflow = await self._db.execute(
            """
            DELETE FROM messages WHERE rowid IN (
                SELECT rowid FROM (
                    SELECT rowid, ROW_NUMBER() OVER (
                        PARTITION BY channel_id
                        ORDER BY created_at DESC, message_id DESC
                    ) AS position
                    FROM messages WHERE bot_id = ?
                ) WHERE position > ?
            )
            """,
            (self.bot_id, self.max_messages_per_channel),
        )
        stats = PruneStats(expired_messages=max(expired, 0), overflow_messages=max(overflow, 0))
        if stats.total:
            logger.debug(
                "[%s] pruned %d expired and %d overflow messages",
                self.bot_id,
                stats.expired_messages,
                stats.overflow_messages,
            )
        return stats


def _history_from_row(row: aiosqlite.Row) -> HistoryMessage:
    return HistoryMessage(
        message_id=row["message_id"],
        channel_id=row["channel_id"],
        author_id=row["author_id"],
        author_name=row["author_name"],
        content=row["content"],
        created_at=from_timestamp(row["created_at"]),
        is_self=bool(row["is_self"]),
        author_is_bot=bool(row["author_is_bot"]),
        reply_to_message_id=row["reply_to_message_id"],
        reply_to_author_name=row["reply_to_author_name"],
    )


def _conversation_from_row(row: aiosqlite.Row) -> ConversationRecord:
    participants_raw: Any = json.loads(row["participants"] or "{}")
    participants = (
        {int(k): str(v) for k, v in participants_raw.items()}
        if isinstance(participants_raw, dict)
        else {}
    )
    return ConversationRecord(
        bot_id=row["bot_id"],
        channel_id=row["channel_id"],
        guild_id=row["guild_id"],
        partner_id=row["partner_id"],
        partner_name=row["partner_name"],
        participants=participants,
        started_at=from_timestamp(row["started_at"]),
        last_activity_at=from_timestamp(row["last_activity_at"]),
        bot_turns=row["bot_turns"],
        user_turns=row["user_turns"],
        awaiting_reply_since=_optional_dt(row["awaiting_reply_since"]),
    )


def _optional_dt(value: float | None) -> datetime | None:
    return from_timestamp(value) if value is not None else None


def _optional_ts(value: datetime | None) -> float | None:
    return to_timestamp(value) if value is not None else None
