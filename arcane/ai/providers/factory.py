"""Provider registry: maps provider names to factories and owns instances.

Personalities name the provider they want (``ModelProfile.provider``). The
registry creates each provider once, shares it between bots, and closes them
all on shutdown.
"""

from __future__ import annotations

from collections.abc import Callable

from arcane.ai.providers.base import LLMProvider
from arcane.ai.providers.ollama import OllamaProvider
from arcane.config.settings import Settings
from arcane.core.errors import ConfigurationError

ProviderFactory = Callable[[Settings], LLMProvider]

_FACTORIES: dict[str, ProviderFactory] = {
    OllamaProvider.name: lambda settings: OllamaProvider(settings.ollama),
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register an additional provider factory (e.g. from a plugin)."""
    _FACTORIES[name.lower()] = factory


def available_providers() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


class ProviderRegistry:
    """Lazily creates and caches provider instances."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._providers: dict[str, LLMProvider] = {}

    def get(self, name: str | None = None) -> LLMProvider:
        """Return the provider called ``name`` (default: ``ARCANE_AI_PROVIDER``)."""
        key = (name or self._settings.ai_provider).lower()
        provider = self._providers.get(key)
        if provider is None:
            factory = _FACTORIES.get(key)
            if factory is None:
                raise ConfigurationError(
                    f"Unknown AI provider '{key}'. Available: {', '.join(available_providers())}"
                )
            provider = factory(self._settings)
            self._providers[key] = provider
        return provider

    async def close(self) -> None:
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()
