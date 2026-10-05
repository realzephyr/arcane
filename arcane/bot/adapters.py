"""Conversion from discord.py objects to Arcane's platform-agnostic models."""

from __future__ import annotations

from typing import Any

import discord

from arcane.core.models import ChannelInfo, IncomingMessage, ReplyReference

MAX_REPLY_SNIPPET_CHARS = 300

_CONVERSATIONAL_TYPES = frozenset({discord.MessageType.default, discord.MessageType.reply})


def is_conversational(message: discord.Message) -> bool:
    """True for regular messages and replies; False for joins, pins, boosts, ..."""
    return message.type in _CONVERSATIONAL_TYPES


def channel_info(channel: Any) -> ChannelInfo:
    """Describe any messageable Discord channel."""
    if isinstance(channel, (discord.DMChannel, discord.GroupChannel)):
        return ChannelInfo(
            channel_id=channel.id,
            name=getattr(channel, "name", None),
            is_dm=True,
        )

    guild = getattr(channel, "guild", None)
    topic = getattr(channel, "topic", None)
    parent_id: int | None = None
    if isinstance(channel, discord.Thread):
        parent_id = channel.parent_id
        if topic is None and channel.parent is not None:
            topic = getattr(channel.parent, "topic", None)
    return ChannelInfo(
        channel_id=channel.id,
        name=getattr(channel, "name", None),
        guild_id=guild.id if guild is not None else None,
        guild_name=guild.name if guild is not None else None,
        topic=topic,
        is_dm=False,
        parent_id=parent_id,
    )


def to_incoming(message: discord.Message, bot_user_id: int) -> IncomingMessage:
    """Normalise a ``discord.Message`` as seen by the bot with ``bot_user_id``.

    Mentions are read from the raw content, so a reply that merely pings its
    target counts as a reply, not as an extra mention.
    """
    reply_to = _reply_reference(message, bot_user_id)
    mentioned = set(message.raw_mentions)
    mentions_bot = bot_user_id in mentioned or _bot_role_mentioned(message)
    mentioned.discard(bot_user_id)

    return IncomingMessage(
        message_id=message.id,
        channel=channel_info(message.channel),
        author_id=message.author.id,
        author_name=message.author.display_name,
        content=message.clean_content,
        created_at=message.created_at,
        author_is_bot=message.author.bot,
        is_self=message.author.id == bot_user_id,
        mentions_bot=mentions_bot,
        mentioned_user_ids=frozenset(mentioned),
        reply_to=reply_to,
        attachments=_attachments(message),
    )


def _reply_reference(message: discord.Message, bot_user_id: int) -> ReplyReference | None:
    reference = message.reference
    if message.type is not discord.MessageType.reply or reference is None:
        return None
    if reference.message_id is None:
        return None
    resolved = reference.resolved
    if isinstance(resolved, discord.Message):
        return ReplyReference(
            message_id=resolved.id,
            author_id=resolved.author.id,
            author_name=resolved.author.display_name,
            content=resolved.clean_content[:MAX_REPLY_SNIPPET_CHARS],
            is_self=resolved.author.id == bot_user_id,
        )
    # Deleted or not delivered with the event: we only know which message it was.
    return ReplyReference(message_id=reference.message_id)


def _bot_role_mentioned(message: discord.Message) -> bool:
    """People often @mention the bot's managed role instead of the bot itself."""
    guild = message.guild
    if guild is None or not message.raw_role_mentions:
        return False
    self_role = guild.self_role
    return self_role is not None and self_role.id in message.raw_role_mentions


def _attachments(message: discord.Message) -> tuple[str, ...]:
    items: list[str] = []
    for attachment in message.attachments:
        content_type = attachment.content_type or ""
        if content_type.startswith("image/"):
            kind = "image"
        elif content_type.startswith("video/"):
            kind = "video"
        elif content_type.startswith("audio/"):
            kind = "audio"
        else:
            kind = "file"
        items.append(f"{kind}: {attachment.filename}")
    items.extend(f"sticker: {sticker.name}" for sticker in message.stickers)
    return tuple(items)
