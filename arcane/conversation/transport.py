"""The interface the conversation core uses to act on a chat platform.

The conversation handler never touches discord.py directly. It shows typing
indicators, sends messages, and looks up channels through this protocol.
:class:`arcane.bot.transport.DiscordTransport` implements it for Discord; the
CLI's terminal chat and the test suite provide their own implementations.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from arcane.core.errors import ArcaneError
from arcane.core.models import ChannelInfo


class TransportError(ArcaneError):
    """The platform rejected an action (missing permissions, deleted channel, ...)."""


@dataclass(frozen=True, slots=True)
class SentMessage:
    """Identity of a message the bot just sent."""

    message_id: int
    created_at: datetime
    author_id: int
    author_name: str


class MessageTransport(Protocol):
    """Platform operations needed by the conversation core."""

    def typing(self, channel_id: int) -> AbstractAsyncContextManager[None]:
        """Show a typing indicator for as long as the context is open.

        Must never raise because the indicator could not be shown.
        """
        ...

    async def send(
        self,
        channel_id: int,
        content: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> SentMessage:
        """Send ``content``; optionally as a reply to another message.

        Raises:
            TransportError: if the message could not be sent.
        """
        ...

    async def channel_info(self, channel_id: int) -> ChannelInfo | None:
        """Describe a channel, or ``None`` if it is unknown or inaccessible."""
        ...
