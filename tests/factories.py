"""Builders for test data."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta

from arcane.core.clock import utcnow
from arcane.core.models import ChannelInfo, HistoryMessage, IncomingMessage, ReplyReference

BOT_USER_ID = 999
_ids = itertools.count(1000)

GENERAL = ChannelInfo(
    channel_id=10, name="general", guild_id=1, guild_name="Test Server", topic="anything goes"
)
DM = ChannelInfo(channel_id=20, is_dm=True)


def next_id() -> int:
    return next(_ids)


def incoming(
    content: str = "hello",
    *,
    author_id: int = 1,
    author_name: str = "alice",
    channel: ChannelInfo = GENERAL,
    created_at: datetime | None = None,
    mentions_bot: bool = False,
    reply_to: ReplyReference | None = None,
    author_is_bot: bool = False,
    is_self: bool = False,
    mentioned_user_ids: frozenset[int] = frozenset(),
    addressed_user_ids: frozenset[int] = frozenset(),
    message_id: int | None = None,
    attachments: tuple[str, ...] = (),
) -> IncomingMessage:
    return IncomingMessage(
        message_id=message_id if message_id is not None else next_id(),
        channel=channel,
        author_id=author_id,
        author_name=author_name,
        content=content,
        created_at=created_at or utcnow(),
        author_is_bot=author_is_bot or is_self,
        is_self=is_self,
        mentions_bot=mentions_bot,
        mentioned_user_ids=mentioned_user_ids,
        addressed_user_ids=addressed_user_ids,
        reply_to=reply_to,
        attachments=attachments,
    )


def from_bot(content: str, **kwargs: object) -> IncomingMessage:
    return incoming(
        content,
        author_id=BOT_USER_ID,
        author_name="mp3",
        is_self=True,
        **kwargs,  # type: ignore[arg-type]
    )


def reply_to_bot(message_id: int = 1, content: str = "earlier bot message") -> ReplyReference:
    return ReplyReference(
        message_id=message_id,
        author_id=BOT_USER_ID,
        author_name="mp3",
        content=content,
        is_self=True,
    )


def history(
    content: str,
    *,
    author_name: str = "alice",
    author_id: int = 1,
    is_self: bool = False,
    minutes_ago: float = 0,
    channel_id: int = GENERAL.channel_id,
    reply_to_author_name: str | None = None,
) -> HistoryMessage:
    return HistoryMessage(
        message_id=next_id(),
        channel_id=channel_id,
        author_id=BOT_USER_ID if is_self else author_id,
        author_name="mp3" if is_self else author_name,
        content=content,
        created_at=utcnow() - timedelta(minutes=minutes_ago),
        is_self=is_self,
        author_is_bot=is_self,
        reply_to_author_name=reply_to_author_name,
    )
