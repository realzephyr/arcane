"""Per-channel conversation state and conversational focus.

A conversation exists in a channel from the moment the bot engages with
someone until nobody has exchanged messages with it for the conversation
timeout. Within a conversation the bot has one **partner**: the person it is
mainly talking with. Other people who engage it directly are **participants**;
the bot answers them, but it only follows the partner's undirected messages.

The partner changes only when someone else engages the bot after the current
partner has gone quiet for longer than the focus timeout. That is what makes
the bot stick with one person instead of hopping between users.

The tracker is synchronous and free of I/O so it can be unit-tested directly;
the conversation handler persists snapshots via short-term memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from arcane.core.models import IncomingMessage
from arcane.database.models import ConversationRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversationState:
    """Mutable state of the conversation in one channel."""

    channel_id: int
    started_at: datetime
    last_activity_at: datetime
    guild_id: int | None = None
    partner_id: int | None = None
    partner_name: str | None = None
    partner_last_message_at: datetime | None = None
    participants: dict[int, str] = field(default_factory=dict)
    bot_turns: int = 0
    user_turns: int = 0
    last_bot_message_at: datetime | None = None
    awaiting_reply_since: datetime | None = None
    """Set when the bot posted an opener nobody has answered yet."""

    def is_active(self, now: datetime, timeout: timedelta) -> bool:
        return now - self.last_activity_at <= timeout

    def partner_is_focused(self, now: datetime, focus_timeout: timedelta) -> bool:
        """True while the partner keeps priority over other users."""
        return (
            self.partner_id is not None
            and self.partner_last_message_at is not None
            and now - self.partner_last_message_at <= focus_timeout
        )

    def awaiting_reply(self, now: datetime, window: timedelta) -> bool:
        return self.awaiting_reply_since is not None and now - self.awaiting_reply_since <= window

    def other_participant_names(self) -> list[str]:
        return [name for uid, name in self.participants.items() if uid != self.partner_id]

    def to_record(self, bot_id: str) -> ConversationRecord:
        return ConversationRecord(
            bot_id=bot_id,
            channel_id=self.channel_id,
            guild_id=self.guild_id,
            started_at=self.started_at,
            last_activity_at=self.last_activity_at,
            partner_id=self.partner_id,
            partner_name=self.partner_name,
            participants=dict(self.participants),
            bot_turns=self.bot_turns,
            user_turns=self.user_turns,
            awaiting_reply_since=self.awaiting_reply_since,
        )

    @classmethod
    def from_record(cls, record: ConversationRecord) -> ConversationState:
        return cls(
            channel_id=record.channel_id,
            guild_id=record.guild_id,
            started_at=record.started_at,
            last_activity_at=record.last_activity_at,
            partner_id=record.partner_id,
            partner_name=record.partner_name,
            partner_last_message_at=record.last_activity_at if record.partner_id else None,
            participants=dict(record.participants),
            bot_turns=record.bot_turns,
            user_turns=record.user_turns,
            awaiting_reply_since=record.awaiting_reply_since,
        )


class ConversationTracker:
    """Tracks conversations across all channels for one bot."""

    def __init__(
        self,
        bot_id: str,
        *,
        timeout: timedelta,
        focus_timeout: timedelta,
        opener_window: timedelta = timedelta(0),
    ) -> None:
        self.bot_id = bot_id
        self.timeout = timeout
        self.focus_timeout = focus_timeout
        self.opener_window = opener_window
        self._conversations: dict[int, ConversationState] = {}
        self._last_bot_message_at: dict[int, datetime] = {}

    # ------------------------------------------------------------------ queries

    def get(self, channel_id: int) -> ConversationState | None:
        return self._conversations.get(channel_id)

    def active(self, channel_id: int, now: datetime) -> ConversationState | None:
        state = self._conversations.get(channel_id)
        if state is not None and state.is_active(now, self.timeout):
            return state
        return None

    def is_active(self, channel_id: int, now: datetime) -> bool:
        return self.active(channel_id, now) is not None

    def last_bot_message_at(self, channel_id: int) -> datetime | None:
        return self._last_bot_message_at.get(channel_id)

    def __len__(self) -> int:
        return len(self._conversations)

    # ------------------------------------------------------------------ updates

    def note_engagement(self, message: IncomingMessage, now: datetime) -> ConversationState:
        """Record that the bot is answering ``message``; applies focus rules."""
        state = self.active(message.channel_id, now)
        if state is None:
            state = ConversationState(
                channel_id=message.channel_id,
                guild_id=message.channel.guild_id,
                started_at=now,
                last_activity_at=now,
            )
            previous = self._conversations.get(message.channel_id)
            if previous is not None and previous.awaiting_reply_since is not None:
                # Someone answered an opener after the conversation window lapsed.
                state.started_at = previous.awaiting_reply_since
            self._conversations[message.channel_id] = state

        author_id, author_name = message.author_id, message.author_name
        if state.partner_id is None or (
            state.partner_id != author_id and not state.partner_is_focused(now, self.focus_timeout)
        ):
            if state.partner_id is not None and state.partner_id != author_id:
                logger.debug(
                    "[%s] focus in channel %d moves from %s to %s",
                    self.bot_id,
                    message.channel_id,
                    state.partner_name,
                    author_name,
                )
            state.partner_id = author_id
            state.partner_name = author_name

        if state.partner_id == author_id:
            state.partner_last_message_at = now
        state.participants[author_id] = author_name
        state.user_turns += 1
        state.last_activity_at = now
        state.awaiting_reply_since = None
        return state

    def note_partner_activity(self, message: IncomingMessage, now: datetime) -> None:
        """The partner spoke (even if not to the bot): they're still around."""
        state = self.active(message.channel_id, now)
        if state is not None and state.partner_id == message.author_id:
            state.partner_last_message_at = now

    def note_bot_message(
        self,
        channel_id: int,
        now: datetime,
        *,
        guild_id: int | None = None,
        initiated: bool = False,
    ) -> ConversationState:
        """Record that the bot sent a message (a reply or an opener)."""
        self._last_bot_message_at[channel_id] = now
        state = self._conversations.get(channel_id)
        if state is None or (initiated and not state.is_active(now, self.timeout)):
            state = ConversationState(
                channel_id=channel_id,
                guild_id=guild_id,
                started_at=now,
                last_activity_at=now,
            )
            self._conversations[channel_id] = state
        state.bot_turns += 1
        state.last_bot_message_at = now
        state.last_activity_at = now
        if initiated:
            state.awaiting_reply_since = now
        return state

    def end(self, channel_id: int) -> ConversationState | None:
        return self._conversations.pop(channel_id, None)

    def expire(self, now: datetime) -> list[ConversationState]:
        """Remove and return conversations that have timed out.

        A conversation waiting for answers to an opener is kept until the
        opener window has lapsed too, so late answers are still recognised.
        """
        expired = [
            state
            for state in self._conversations.values()
            if not state.is_active(now, self.timeout)
            and not state.awaiting_reply(now, self.opener_window)
        ]
        for state in expired:
            del self._conversations[state.channel_id]
        stale_channels = [
            channel_id
            for channel_id, at in self._last_bot_message_at.items()
            if now - at > timedelta(days=1)
        ]
        for channel_id in stale_channels:
            del self._last_bot_message_at[channel_id]
        return expired

    def restore(self, records: list[ConversationRecord], now: datetime) -> int:
        """Load persisted conversations that are still active. Returns how many."""
        restored = 0
        for record in records:
            state = ConversationState.from_record(record)
            if state.is_active(now, self.timeout) or state.awaiting_reply(now, self.opener_window):
                self._conversations[state.channel_id] = state
                restored += 1
        return restored
