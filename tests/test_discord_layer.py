"""Tests for the discord.py integration using spec'd mocks (no network)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from arcane.bot.adapters import channel_info, is_conversational, to_incoming
from arcane.bot.bot import ArcaneBot, default_intents
from arcane.bot.events import EventRouter
from arcane.bot.transport import SAFE_MENTIONS, DiscordTransport
from arcane.conversation.transport import TransportError
from arcane.personalities.registry import load_personality

BOT_ID = 999
NOW = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def _guild(self_role_id: int = 555) -> MagicMock:
    guild = MagicMock(spec=discord.Guild)
    guild.id = 1
    guild.name = "Test Server"
    guild.self_role = MagicMock(spec=discord.Role, id=self_role_id)
    return guild


def _text_channel(channel_id: int = 10) -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.id = channel_id
    channel.name = "general"
    channel.topic = "talk about anything"
    channel.guild = _guild()
    return channel


def _user(user_id: int = 1, name: str = "alice", *, bot: bool = False) -> MagicMock:
    user = MagicMock(spec=discord.Member)
    user.id = user_id
    user.display_name = name
    user.bot = bot
    return user


def _message(
    content: str = "hello",
    *,
    author: MagicMock | None = None,
    channel: Any = None,
    raw_mentions: list[int] | None = None,
    raw_role_mentions: list[int] | None = None,
    message_type: discord.MessageType = discord.MessageType.default,
    reference: Any = None,
    attachments: list[Any] | None = None,
    stickers: list[Any] | None = None,
) -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.id = 100
    message.content = content
    message.clean_content = content
    message.author = author or _user()
    message.channel = channel or _text_channel()
    message.guild = getattr(message.channel, "guild", None)
    message.created_at = NOW
    message.type = message_type
    message.raw_mentions = raw_mentions or []
    message.raw_role_mentions = raw_role_mentions or []
    message.reference = reference
    message.attachments = attachments or []
    message.stickers = stickers or []
    return message


# ------------------------------------------------------------------------ adapters


def test_plain_message_conversion() -> None:
    incoming = to_incoming(_message("hi all"), BOT_ID)
    assert incoming.content == "hi all"
    assert incoming.author_name == "alice"
    assert incoming.channel.name == "general"
    assert incoming.channel.guild_name == "Test Server"
    assert incoming.channel.topic == "talk about anything"
    assert not incoming.mentions_bot
    assert not incoming.is_self
    assert incoming.reply_to is None


def test_mentions_are_read_from_content() -> None:
    incoming = to_incoming(_message("@mp3 and @bob", raw_mentions=[BOT_ID, 2]), BOT_ID)
    assert incoming.mentions_bot
    assert incoming.mentioned_user_ids == frozenset({2})


@pytest.mark.parametrize(
    ("content", "mentions", "addressed"),
    [
        ("<@2> what do you think", [2], {2}),
        ("<@!2>, <@3>: thoughts?", [2, 3], {2, 3}),
        ("i told <@2> about it", [2], set()),
        (f"<@{BOT_ID}> <@2> settle this", [BOT_ID, 2], {2}),
        ("no mentions here", [], set()),
    ],
)
def test_addressed_users_are_leading_mentions_only(
    content: str, mentions: list[int], addressed: set[int]
) -> None:
    incoming = to_incoming(_message(content, raw_mentions=mentions), BOT_ID)
    assert incoming.addressed_user_ids == frozenset(addressed)
    assert BOT_ID not in incoming.mentioned_user_ids


def test_bot_role_mention_counts_as_mention() -> None:
    incoming = to_incoming(_message("@mp3", raw_role_mentions=[555]), BOT_ID)
    assert incoming.mentions_bot


def test_reply_to_bot_is_detected() -> None:
    original = _message("earlier", author=_user(BOT_ID, "mp3", bot=True))
    original.id = 50
    reference = MagicMock(spec=discord.MessageReference)
    reference.message_id = 50
    reference.resolved = original
    reply = _message("good point", message_type=discord.MessageType.reply, reference=reference)

    incoming = to_incoming(reply, BOT_ID)

    assert incoming.reply_to is not None
    assert incoming.reply_to.is_self
    assert incoming.reply_to.author_name == "mp3"
    assert incoming.is_reply_to_self
    assert not incoming.mentions_bot  # replying is not the same as an explicit mention


def test_reply_to_deleted_message_keeps_reference_only() -> None:
    reference = MagicMock(spec=discord.MessageReference)
    reference.message_id = 51
    reference.resolved = None
    incoming = to_incoming(
        _message("re", message_type=discord.MessageType.reply, reference=reference), BOT_ID
    )
    assert incoming.reply_to is not None
    assert incoming.reply_to.message_id == 51
    assert not incoming.reply_to.is_self


def test_attachments_and_stickers() -> None:
    image = MagicMock(spec=discord.Attachment, filename="cat.png", content_type="image/png")
    doc = MagicMock(spec=discord.Attachment, filename="notes.pdf", content_type=None)
    sticker = MagicMock(spec=discord.StickerItem)
    sticker.name = "wave"
    incoming = to_incoming(_message("", attachments=[image, doc], stickers=[sticker]), BOT_ID)
    assert incoming.attachments == ("image: cat.png", "file: notes.pdf", "sticker: wave")
    assert incoming.has_content


def test_dm_and_thread_channels() -> None:
    dm = MagicMock(spec=discord.DMChannel)
    dm.id = 20
    assert channel_info(dm).is_dm

    parent = _text_channel(10)
    thread = MagicMock(spec=discord.Thread)
    thread.id = 11
    thread.name = "a thread"
    thread.guild = parent.guild
    thread.topic = None
    thread.parent_id = 10
    thread.parent = parent
    info = channel_info(thread)
    assert info.parent_id == 10
    assert info.topic == "talk about anything"


def test_system_messages_are_not_conversational() -> None:
    assert is_conversational(_message())
    assert is_conversational(_message(message_type=discord.MessageType.reply))
    assert not is_conversational(_message(message_type=discord.MessageType.pins_add))


def test_own_message_is_flagged() -> None:
    incoming = to_incoming(_message(author=_user(BOT_ID, "mp3", bot=True)), BOT_ID)
    assert incoming.is_self and incoming.author_is_bot


# ----------------------------------------------------------------------- transport


def _client_with(channel: Any) -> MagicMock:
    client = MagicMock(spec=discord.Client)
    client.get_channel.return_value = channel
    client.fetch_channel = AsyncMock(side_effect=discord.NotFound(MagicMock(status=404), "gone"))
    return client


def _sendable_channel() -> MagicMock:
    channel = _text_channel()
    sent = MagicMock(spec=discord.Message)
    sent.id = 777
    sent.created_at = NOW
    sent.author = _user(BOT_ID, "mp3", bot=True)
    channel.send = AsyncMock(return_value=sent)
    return channel


async def test_send_uses_safe_mentions_and_reply_reference() -> None:
    channel = _sendable_channel()
    transport = DiscordTransport(_client_with(channel))

    result = await transport.send(10, "hello", reply_to_message_id=42)

    assert result.message_id == 777
    assert result.author_name == "mp3"
    _, kwargs = channel.send.call_args
    assert kwargs["allowed_mentions"] is SAFE_MENTIONS
    assert kwargs["mention_author"] is False
    assert kwargs["reference"].message_id == 42
    assert kwargs["reference"].fail_if_not_exists is False
    assert SAFE_MENTIONS.everyone is False and SAFE_MENTIONS.roles is False


async def test_send_errors_become_transport_errors() -> None:
    channel = _sendable_channel()
    channel.send.side_effect = discord.Forbidden(MagicMock(status=403, reason="Forbidden"), "no")
    transport = DiscordTransport(_client_with(channel))
    with pytest.raises(TransportError, match="permission"):
        await transport.send(10, "hello")

    missing = DiscordTransport(_client_with(None))
    with pytest.raises(TransportError, match="not accessible"):
        await missing.send(10, "hello")
    assert await missing.channel_info(10) is None


async def test_typing_failures_do_not_block() -> None:
    channel = _sendable_channel()

    class BrokenTyping:
        async def __aenter__(self) -> None:
            raise discord.HTTPException(MagicMock(status=500, reason="err"), "boom")

        async def __aexit__(self, *_exc: object) -> None:
            return None

    channel.typing = BrokenTyping
    transport = DiscordTransport(_client_with(channel))
    entered = False
    async with transport.typing(10):
        entered = True
    assert entered


async def test_typing_wraps_channel_typing() -> None:
    channel = _sendable_channel()
    events: list[str] = []

    @asynccontextmanager
    async def typing() -> AsyncIterator[None]:
        events.append("on")
        yield
        events.append("off")

    channel.typing = typing
    async with DiscordTransport(_client_with(channel)).typing(10):
        events.append("body")
    assert events == ["on", "body", "off"]


# -------------------------------------------------------------------------- router


def _router(**task_kwargs: Any) -> tuple[EventRouter, MagicMock, MagicMock]:
    client = MagicMock(spec=discord.Client)
    client.user = _user(BOT_ID, "mp3", bot=True)
    client.guilds = []
    handler = MagicMock()
    handler.bot_id = "mp3"
    for name in ("start", "close", "handle_message", "handle_edit", "handle_delete"):
        setattr(handler, name, AsyncMock())
    task = MagicMock()
    task.stop = AsyncMock()
    return EventRouter(client, handler, background_tasks=[task]), handler, task


async def test_router_starts_once_and_shuts_down_once() -> None:
    router, handler, task = _router()
    await router.on_ready()
    await router.on_ready()  # reconnect
    handler.start.assert_awaited_once()
    task.start.assert_called_once()

    await router.shutdown()
    await router.shutdown()
    handler.close.assert_awaited_once()
    task.stop.assert_awaited_once()


async def test_router_forwards_messages_and_survives_errors() -> None:
    router, handler, _ = _router()
    await router.on_message(_message("hey"))
    handler.handle_message.assert_awaited_once()

    handler.handle_message.side_effect = RuntimeError("boom")
    await router.on_message(_message("again"))  # must not raise

    await router.on_message(_message(message_type=discord.MessageType.pins_add))
    assert handler.handle_message.await_count == 2


async def test_router_forwards_edits_and_deletes() -> None:
    router, handler, _ = _router()
    edit = MagicMock(spec=discord.RawMessageUpdateEvent)
    edit.message_id = 5
    edit.message = None
    edit.data = {"content": "edited"}
    await router.on_raw_message_edit(edit)
    handler.handle_edit.assert_awaited_once_with(5, "edited")

    bulk = MagicMock(spec=discord.RawBulkMessageDeleteEvent)
    bulk.message_ids = {6, 7}
    await router.on_raw_bulk_message_delete(bulk)
    assert handler.handle_delete.await_count == 2


# ------------------------------------------------------------------------- client


def test_intents_are_minimal() -> None:
    intents = default_intents()
    assert intents.message_content and intents.guild_messages and intents.dm_messages
    assert not intents.members and not intents.presences


async def test_bot_client_configuration() -> None:
    bot = ArcaneBot(load_personality("mp3"))
    try:
        assert bot.allowed_mentions is SAFE_MENTIONS
        assert bot.personality.id == "mp3"
    finally:
        await bot.close()
