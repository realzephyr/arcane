"""Test doubles for providers and transports."""

from __future__ import annotations

from collections.abc import Sequence

from arcane.ai.providers.base import (
    ChatMessage,
    GenerationOptions,
    GenerationResult,
    LLMProvider,
    ProviderHealth,
)


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
