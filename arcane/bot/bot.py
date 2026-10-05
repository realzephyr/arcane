"""The Discord client for one personality."""

from __future__ import annotations

import logging
from typing import Any

import discord

from arcane.bot.events import EventRouter
from arcane.bot.transport import SAFE_MENTIONS
from arcane.personalities.base import Personality

logger = logging.getLogger(__name__)

MESSAGE_CACHE_SIZE = 500


def default_intents() -> discord.Intents:
    """The minimum gateway intents Arcane needs.

    ``message_content`` is privileged: enable "Message Content Intent" for the
    application in the Discord Developer Portal.
    """
    intents = discord.Intents.none()
    intents.guilds = True
    intents.guild_messages = True
    intents.dm_messages = True
    intents.message_content = True
    return intents


class ArcaneBot(discord.Client):
    """A Discord client bound to a personality; events go to an :class:`EventRouter`."""

    def __init__(self, personality: Personality, *, intents: discord.Intents | None = None) -> None:
        # Arcane never joins voice; skip discord.py's warnings about voice dependencies.
        discord.VoiceClient.warn_nacl = False
        discord.VoiceClient.warn_dave = False
        super().__init__(
            intents=intents or default_intents(),
            allowed_mentions=SAFE_MENTIONS,
            max_messages=MESSAGE_CACHE_SIZE,
            chunk_guilds_at_startup=False,
        )
        self.personality = personality
        self._router: EventRouter | None = None

    def attach(self, router: EventRouter) -> None:
        self._router = router

    async def on_ready(self) -> None:
        if self._router is not None:
            await self._router.on_ready()

    async def on_message(self, message: discord.Message) -> None:
        if self._router is not None:
            await self._router.on_message(message)

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        if self._router is not None:
            await self._router.on_raw_message_edit(payload)

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        if self._router is not None:
            await self._router.on_raw_message_delete(payload)

    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent) -> None:
        if self._router is not None:
            await self._router.on_raw_bulk_message_delete(payload)

    async def on_error(self, event_method: str, /, *args: Any, **kwargs: Any) -> None:
        logger.exception("[%s] unhandled error in %s", self.personality.id, event_method)

    async def close(self) -> None:
        if self._router is not None:
            await self._router.shutdown()
        await super().close()
