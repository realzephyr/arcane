"""Test doubles for providers and transports."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from arcane.ai.providers.base import (
    ChatMessage,
    GenerationOptions,
    GenerationResult,
    LLMProvider,
    ProviderHealth,
)
from arcane.conversation.transport import SentMessage, TransportError
from arcane.core.clock import utcnow
from arcane.core.models import ChannelInfo
from tests.factories import BOT_USER_ID


class ScriptedProvider(LLMProvider):
    """Returns scripted replies (or raises scripted errors) in order.

    When the script runs out, the last entry is repeated.
    """

    name = "scripted"

    def __init__(self, *replies: str | Exception) -> None:
        if not replies:
            raise ValueError("at least one reply is required")
        self.replies = list(replies)
        self.calls: list[tuple[list[ChatMessage], GenerationOptions | None, str | None]] = []

    @property
    def default_model(self) -> str:
        return "test-model"

    @property
    def reply(self) -> str | Exception:
        return self.replies[0]

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        self.calls.append((list(messages), options, model))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return GenerationResult(content=reply, model=model or "test-model", provider=self.name)

    async def health_check(self, model: str | None = None) -> ProviderHealth:
        return ProviderHealth(available=True, detail="ok", model_available=True)


class GatedProvider(ScriptedProvider):
    """A scripted provider that waits for ``gate`` before answering."""

    def __init__(self, *replies: str | Exception) -> None:
        super().__init__(*replies)
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        self.started.set()
        await self.gate.wait()
        return await super().chat(messages, model=model, options=options)


class FakeTransport:
    """Records typing indicators and sent messages."""

    def __init__(self, channels: Sequence[ChannelInfo] = (), *, fail_sends: bool = False) -> None:
        self.channels = {channel.channel_id: channel for channel in channels}
        self.fail_sends = fail_sends
        self.sent: list[tuple[int, str, int | None]] = []
        self.events: list[str] = []
        self._ids = itertools.count(50_000)

    @asynccontextmanager
    async def typing(self, channel_id: int) -> AsyncIterator[None]:
        self.events.append("typing_on")
        try:
            yield
        finally:
            self.events.append("typing_off")

    async def send(
        self, channel_id: int, content: str, *, reply_to_message_id: int | None = None
    ) -> SentMessage:
        if self.fail_sends:
            raise TransportError("missing permissions")
        self.sent.append((channel_id, content, reply_to_message_id))
        self.events.append(f"send:{content}")
        return SentMessage(
            message_id=next(self._ids),
            created_at=utcnow(),
            author_id=BOT_USER_ID,
            author_name="mp3",
        )

    async def channel_info(self, channel_id: int) -> ChannelInfo | None:
        return self.channels.get(channel_id)

    @property
    def texts(self) -> list[str]:
        return [content for _, content, _ in self.sent]
