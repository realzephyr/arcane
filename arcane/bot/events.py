"""Discord event routing.

:class:`EventRouter` translates gateway events into calls on the
platform-agnostic :class:`~arcane.conversation.handler.ConversationHandler`
and owns the bot's background tasks. Every handler catches and logs its own
exceptions so one bad event can never take the bot down.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import discord

from arcane.bot.adapters import is_conversational, to_incoming
from arcane.conversation.handler import ConversationHandler
from arcane.core.tasks import PeriodicTask

logger = logging.getLogger(__name__)


class EventRouter:
    """Connects one Discord client to one conversation handler."""

    def __init__(
        self,
        client: discord.Client,
        handler: ConversationHandler,
        *,
        background_tasks: Sequence[PeriodicTask] = (),
    ) -> None:
        self._client = client
        self._handler = handler
        self._tasks = list(background_tasks)
        self._started = False
        self._stopped = False

    @property
    def bot_id(self) -> str:
        return self._handler.bot_id

    async def on_ready(self) -> None:
        user = self._client.user
        logger.info(
            "[%s] connected as %s (id %s) in %d server(s)",
            self.bot_id,
            user,
            user.id if user is not None else "?",
            len(self._client.guilds),
        )
        if self._started:
            return  # on_ready fires again after reconnects
        self._started = True
        try:
            await self._handler.start()
        except Exception:
            logger.exception("[%s] failed to restore conversation state", self.bot_id)
        for task in self._tasks:
            task.start()

    async def on_message(self, message: discord.Message) -> None:
        user = self._client.user
        if user is None or self._stopped or not is_conversational(message):
            return
        try:
            await self._handler.handle_message(to_incoming(message, user.id))
        except Exception:
            logger.exception("[%s] failed to handle message %d", self.bot_id, message.id)

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        message = getattr(payload, "message", None)
        if isinstance(message, discord.Message):
            content: object = message.clean_content
        else:
            content = payload.data.get("content")
        if isinstance(content, str):
            await self._handler.handle_edit(payload.message_id, content)

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        await self._handler.handle_delete(payload.message_id)

    async def on_raw_bulk_message_delete(self, payload: discord.RawBulkMessageDeleteEvent) -> None:
        for message_id in payload.message_ids:
            await self._handler.handle_delete(message_id)

    async def shutdown(self) -> None:
        """Stop background tasks and in-flight responses. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        for task in self._tasks:
            await task.stop()
        await self._handler.close()
