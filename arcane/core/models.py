"""Platform-agnostic domain models.

These dataclasses are the language spoken between layers. The Discord layer
converts ``discord.Message`` objects into :class:`IncomingMessage`; the
conversation core, memory, and prompt builder only ever see these types. That
keeps the core testable without Discord and reusable for other platforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Where a message was sent."""

    channel_id: int
    name: str | None = None
    guild_id: int | None = None
    guild_name: str | None = None
    topic: str | None = None
    is_dm: bool = False
    parent_id: int | None = None
    """For threads: the channel the thread belongs to."""

    @property
    def display_name(self) -> str:
        """A human-readable label for logs and prompts."""
        if self.is_dm:
            return "direct messages"
        if self.name:
            return f"#{self.name}"
        return f"channel {self.channel_id}"


@dataclass(frozen=True, slots=True)
class ReplyReference:
    """The message an incoming message replies to (Discord reply feature)."""

    message_id: int
    author_id: int | None = None
    author_name: str | None = None
    content: str | None = None
    is_self: bool = False
    """True when the referenced message was written by the receiving bot."""


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A message observed by a bot, normalised for the conversation core."""

    message_id: int
    channel: ChannelInfo
    author_id: int
    author_name: str
    content: str
    created_at: datetime
    author_is_bot: bool = False
    is_self: bool = False
    """True when the message was written by the receiving bot itself."""
    mentions_bot: bool = False
    """True when the receiving bot was @mentioned (or its role was)."""
    mentioned_user_ids: frozenset[int] = field(default_factory=frozenset)
    """Other users @mentioned anywhere in the message (the bot excluded)."""
    addressed_user_ids: frozenset[int] = field(default_factory=frozenset)
    """Other users the message is explicitly addressed to: @mentions at its very
    start, as in "@bob what do you think". A mention in passing ("i told @bob")
    does not count."""
    reply_to: ReplyReference | None = None
    attachments: tuple[str, ...] = ()
    """Short descriptions of attachments, e.g. ``"image: cat.png"``."""

    @property
    def channel_id(self) -> int:
        return self.channel.channel_id

    @property
    def is_dm(self) -> bool:
        return self.channel.is_dm

    @property
    def is_reply_to_self(self) -> bool:
        return self.reply_to is not None and self.reply_to.is_self

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip()) or bool(self.attachments)

    def text_for_prompt(self) -> str:
        """Message text including attachment hints, as the model should see it."""
        parts = [self.content.strip()] if self.content.strip() else []
        parts.extend(f"[{attachment}]" for attachment in self.attachments)
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """A message retrieved from short-term memory for prompt context."""

    message_id: int
    channel_id: int
    author_id: int
    author_name: str
    content: str
    created_at: datetime
    is_self: bool = False
    """True when this bot wrote the message."""
    author_is_bot: bool = False
    reply_to_message_id: int | None = None
    reply_to_author_name: str | None = None
