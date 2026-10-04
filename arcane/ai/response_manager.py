"""Generation orchestration: prompt → model → clean Discord messages.

:class:`ResponseManager` is the single entry point the conversation layer uses
to get words out of a model. It builds the prompt, applies the personality's
sampling settings, post-processes the output, and retries once when the
result is unusable (empty after cleaning, or a near-verbatim repeat of
something the bot said recently).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from difflib import SequenceMatcher

from arcane.ai.postprocess import ProcessedResponse, ResponsePostProcessor
from arcane.ai.prompts import PromptBuilder, PromptContext
from arcane.ai.providers.base import GenerationOptions, LLMProvider
from arcane.personalities.base import Personality

logger = logging.getLogger(__name__)

REPETITION_THRESHOLD = 0.9
REPETITION_MIN_CHARS = 20
"""Short replies like "lol fair" may legitimately repeat."""
RETRY_TEMPERATURE_BOOST = 0.15
_RECENT_SELF_MESSAGES = 6


@dataclass(frozen=True, slots=True)
class GeneratedReply:
    """A ready-to-send reply."""

    parts: tuple[str, ...]
    raw: str
    model: str
    duration_seconds: float
    attempts: int = 1
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)


class ResponseManager:
    """Generates replies for one personality."""

    def __init__(
        self,
        personality: Personality,
        provider: LLMProvider,
        *,
        model: str | None = None,
        prompt_builder: PromptBuilder | None = None,
        postprocessor: ResponsePostProcessor | None = None,
        max_attempts: int = 2,
    ) -> None:
        self._personality = personality
        self._provider = provider
        self._model = model or personality.model.model
        self._builder = prompt_builder or PromptBuilder()
        self._postprocessor = postprocessor or ResponsePostProcessor(personality)
        self._max_attempts = max(1, max_attempts)

    @property
    def model_name(self) -> str:
        return self._model or self._provider.default_model

    def base_options(self) -> GenerationOptions:
        profile = self._personality.model
        return GenerationOptions(
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            repeat_penalty=profile.repeat_penalty,
            max_tokens=profile.max_tokens,
            context_window=profile.context_window,
        )

    async def generate(self, context: PromptContext) -> GeneratedReply | None:
        """Generate a reply, or ``None`` if the model produced nothing usable.

        Raises:
            ProviderError: if the backend fails (after the provider's own retries).
        """
        messages = self._builder.build(context)
        speakers = {m.author_name for m in context.history if not m.is_self}
        recent_self = [m.content for m in context.history if m.is_self][-_RECENT_SELF_MESSAGES:]

        options = self.base_options()
        started = time.monotonic()
        for attempt in range(1, self._max_attempts + 1):
            result = await self._provider.chat(messages, model=self._model, options=options)
            processed = self._postprocessor.process(result.content, other_speakers=speakers)

            problem = self._problem_with(processed, recent_self)
            if problem is None:
                return GeneratedReply(
                    parts=processed.parts,
                    raw=result.content,
                    model=result.model,
                    duration_seconds=time.monotonic() - started,
                    attempts=attempt,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                )

            logger.info(
                "[%s] discarded generation attempt %d/%d: %s",
                self._personality.id,
                attempt,
                self._max_attempts,
                problem,
            )
            options = replace(
                options,
                temperature=min((options.temperature or 0.8) + RETRY_TEMPERATURE_BOOST, 1.5),
                seed=None,
            )
        return None

    @staticmethod
    def _problem_with(processed: ProcessedResponse, recent_self: list[str]) -> str | None:
        if processed.is_empty:
            return "empty after post-processing"
        candidate = _normalize(processed.text)
        if len(candidate) < REPETITION_MIN_CHARS:
            return None
        for previous in recent_self:
            if SequenceMatcher(None, candidate, _normalize(previous)).ratio() >= (
                REPETITION_THRESHOLD
            ):
                return "repeats a recent message"
        return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
