"""Discord implementation of :class:`~arcane.conversation.transport.MessageTransport`."""

from __future__ import annotations

import asyncio
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

TYPING_REFRESH_SECONDS = 8.0
TYPING_REQUEST_TIMEOUT_SECONDS = 5.0
MAX_TYPING_SECONDS = 30.0
"""Hard cap on one typing indicator, whatever happens while it is shown."""


class DiscordTransport:
    """Typing, sending, and channel lookup through a ``discord.Client``."""

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    @contextlib.asynccontextmanager
    async def typing(self, channel_id: int) -> AsyncIterator[None]:
        """Show "is typing..." while the context is open, for at most ``MAX_TYPING_SECONDS``.

        Discord shows the indicator for about 10 seconds per request, so it is
        refreshed every ``TYPING_REFRESH_SECONDS``. The refresh stops when the
        context closes or the cap is reached, whatever happens inside it, so the
        indicator can never run on forever.
        """
        channel = await self._messageable(channel_id)
        refresher: asyncio.Task[None] | None = None
        if channel is not None:
            refresher = asyncio.create_task(self._keep_typing(channel, channel_id))
        try:
            yield
        finally:
            if refresher is not None:
                refresher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresher

    @staticmethod
    async def _keep_typing(channel: Any, channel_id: int) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + MAX_TYPING_SECONDS
        while loop.time() < deadline:
            try:
                async with asyncio.timeout(TYPING_REQUEST_TIMEOUT_SECONDS):
                    await channel.typing()  # one request: shows the indicator ~10 s
            except Exception as exc:
                # A missing typing indicator must never block or break the reply.
                logger.debug("typing indicator failed in %d: %s", channel_id, exc)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)

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
