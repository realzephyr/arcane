"""Discord implementation of :class:`~arcane.conversation.transport.MessageTransport`."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import discord

from arcane.bot.adapters import channel_info
from arcane.conversation.transport import SentMessage, TransportError
from arcane.core.models import ChannelInfo

logger = logging.getLogger(__name__)

SAFE_MENTIONS = discord.AllowedMentions(
    everyone=False, users=False, roles=False, replied_user=False
)
"""Bots never ping anyone; a reply reference is shown without notifying the author."""


class DiscordTransport:
    """Typing, sending, and channel lookup through a ``discord.Client``."""

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    @contextlib.asynccontextmanager
    async def typing(self, channel_id: int) -> AsyncIterator[None]:
        channel = await self._messageable(channel_id)
        async with contextlib.AsyncExitStack() as stack:
            if channel is not None:
                try:
                    await stack.enter_async_context(channel.typing())
                except (discord.HTTPException, discord.ClientException) as exc:
                    # A missing typing indicator must never block the reply.
                    logger.debug("typing indicator failed in %d: %s", channel_id, exc)
            yield

    async def send(
        self,
        channel_id: int,
        content: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> SentMessage:
        channel = await self._messageable(channel_id)
        if channel is None:
            raise TransportError(f"channel {channel_id} is not accessible")

        reference = None
        if reply_to_message_id is not None:
            guild = getattr(channel, "guild", None)
            reference = discord.MessageReference(
                message_id=reply_to_message_id,
                channel_id=channel_id,
                guild_id=guild.id if guild is not None else None,
                fail_if_not_exists=False,
            )
        try:
            message = await channel.send(
                content,
                reference=reference,
                allowed_mentions=SAFE_MENTIONS,
                mention_author=False,
            )
        except discord.Forbidden as exc:
            raise TransportError(f"missing permission to send in {channel_id}") from exc
        except discord.HTTPException as exc:
            raise TransportError(f"Discord rejected the message: {exc}") from exc

        return SentMessage(
            message_id=message.id,
            created_at=message.created_at,
            author_id=message.author.id,
            author_name=message.author.display_name,
        )

    async def channel_info(self, channel_id: int) -> ChannelInfo | None:
        channel = await self._messageable(channel_id)
        return channel_info(channel) if channel is not None else None

    async def _messageable(self, channel_id: int) -> Any:
        channel: Any = self._client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(channel_id)
            except (discord.NotFound, discord.Forbidden):
                return None
            except discord.HTTPException as exc:
                logger.warning("could not fetch channel %d: %s", channel_id, exc)
                return None
        if not isinstance(channel, discord.abc.Messageable):
            return None
        return channel
