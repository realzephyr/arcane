"""The provider-agnostic interface every LLM backend implements.

The rest of Arcane depends only on these types. Supporting a new backend
(OpenAI-compatible APIs, llama.cpp, Anthropic, ...) means implementing
:class:`LLMProvider` and registering a factory in :mod:`arcane.ai.providers.factory`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

from arcane.core.errors import ArcaneError

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat-style prompt."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Sampling and output options. ``None`` means "use the provider default"."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repeat_penalty: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    json_mode: bool = False
    """Constrain the output to valid JSON (used for structured extraction)."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The outcome of a successful generation."""

    content: str
    model: str
    provider: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ProviderHealth:
    """Result of a provider health check."""

    available: bool
    detail: str
    model_available: bool | None = None
    models: tuple[str, ...] = ()


class ProviderError(ArcaneError):
    """Base class for provider failures.

    ``retryable`` tells callers whether trying again later might succeed.
    """

    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable


class ProviderUnavailableError(ProviderError):
    """The backend could not be reached."""

    retryable = True


class ProviderTimeoutError(ProviderError):
    """The backend did not answer in time."""

    retryable = True


class ModelNotFoundError(ProviderError):
    """The requested model is not installed on the backend."""

    retryable = False


class ProviderResponseError(ProviderError):
    """The backend answered with an error or an unexpected payload."""


class LLMProvider(ABC):
    """Asynchronous chat-completion backend."""

    name: ClassVar[str]

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Model used when a request does not specify one."""

    @abstractmethod
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        """Generate the next assistant message for ``messages``.

        Raises:
            ProviderError: on any backend failure (after internal retries).
        """

    @abstractmethod
    async def health_check(self, model: str | None = None) -> ProviderHealth:
        """Check connectivity and, if given, whether ``model`` is available.

        Must not raise for expected failures; report them in the result.
        """

    async def close(self) -> None:  # noqa: B027 - optional hook, intentionally empty
        """Release network resources. Safe to call more than once."""
