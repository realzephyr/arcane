"""Persistent entity models.

Plain immutable dataclasses mirroring stored rows. They carry no database
behaviour, so any layer can use them without depending on SQLite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class MemoryKind(StrEnum):
    """What a long-term memory describes."""

    FACT = "fact"
    """Background about a person: job, studies, where they're from (coarse)."""
    PREFERENCE = "preference"
    """Likes and dislikes."""
    OPINION = "opinion"
    """A stance they hold, e.g. on a philosophical question."""
    INTEREST = "interest"
    """Topics they care about."""
    TOPIC = "topic"
    """Something you discussed together."""
    NOTE = "note"
    """Anything else worth remembering."""

    @classmethod
    def parse(cls, value: object, default: MemoryKind | None = None) -> MemoryKind:
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return default if default is not None else cls.NOTE


@dataclass(frozen=True, slots=True)
class UserProfile:
    bot_id: str
    user_id: int
    display_name: str
    first_seen_at: datetime
    last_seen_at: datetime
    interaction_count: int = 0


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    bot_id: str
    user_id: int | None
    kind: MemoryKind
    content: str
    importance: float
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    access_count: int = 0
    expires_at: datetime | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    """Persisted snapshot of a channel's conversation state."""

    bot_id: str
    channel_id: int
    started_at: datetime
    last_activity_at: datetime
    guild_id: int | None = None
    partner_id: int | None = None
    partner_name: str | None = None
    participants: dict[int, str] = field(default_factory=dict)
    bot_turns: int = 0
    user_turns: int = 0
    awaiting_reply_since: datetime | None = None


@dataclass(frozen=True, slots=True)
class InitiativeRecord:
    id: int
    bot_id: str
    channel_id: int
    topic: str
    created_at: datetime
