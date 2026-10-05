"""The personality schema.

A :class:`Personality` is pure, validated data. The engine never branches on a
personality's id: everything that differs between personalities (identity,
writing style, timing, behaviour, memory, model) is expressed through the
profiles below, each with sensible defaults so a new personality only needs to
override what makes it distinct.
"""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arcane.config.settings import PERSONALITY_ID_PATTERN

CHARS_PER_WORD = 5
"""Typing-test convention: one "word" is five characters, spaces included."""

_WORD_RE = re.compile(r"[a-z0-9']+")


class _Profile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StyleProfile(_Profile):
    """How the personality writes."""

    guidelines: tuple[str, ...] = ()
    """Concrete writing rules included in the system prompt."""
    max_response_chars: int = Field(default=900, ge=50, le=6000)
    """Hard cap on a whole reply (all messages combined)."""
    max_messages_per_reply: int = Field(default=3, ge=1, le=6)
    """How many separate Discord messages one reply may be split into."""
    banned_openers: tuple[str, ...] = ()
    """Extra phrases stripped from the start of replies (on top of the defaults)."""
    allow_exclamation_points: bool = True
    """When False, every exclamation point is removed from replies."""
    lowercase_starts: bool = False
    """When True, a capitalised first word of each message is lowercased ("Yeah" -> "yeah")."""
    blocked_patterns: tuple[str, ...] = ()
    """Regular expressions (case-insensitive) that must never appear in a reply. A
    reply that matches is discarded and regenerated; if every attempt matches, the
    bot stays silent. Use it for slurs or other terms you never want posted."""

    @field_validator("blocked_patterns")
    @classmethod
    def _validate_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid blocked pattern {pattern!r}: {exc}") from exc
        return value


class TimingProfile(_Profile):
    """Human-like pacing. Ranges are ``(min, max)`` seconds.

    The bot reads first (no typing indicator), then shows the typing indicator
    while the model generates, and sends once a human typing at
    ``typing_speed_wpm`` would have finished. Generation time counts towards the
    typing time, so a slow model never adds delay on top of the typing time.
    """

    typing_speed_wpm: float = Field(default=60.0, gt=0, le=250)
    """Typing speed in words per minute (one word = five characters)."""
    reading_speed_wpm: float = Field(default=300.0, gt=0)
    """Reading speed in words per minute for incoming messages."""
    reaction_seconds: tuple[float, float] = (0.3, 1.2)
    """Time to notice new messages before reading them."""
    pause_between_messages_seconds: tuple[float, float] = (0.4, 1.2)
    """Pause after sending one message of a split reply before typing the next."""
    min_typing_seconds: float = Field(default=0.8, ge=0)
    max_typing_seconds: float = Field(default=45.0, gt=0)
    """Upper bound for typing a single message (225 characters at 60 wpm)."""
    max_reading_seconds: float = Field(default=8.0, gt=0)
    """Upper bound for the whole reading phase, including follow-up messages."""
    variation: float = Field(default=0.15, ge=0, le=1)
    """Spread of the log-normal noise applied to reading and typing times."""

    @property
    def typing_cps(self) -> float:
        return self.typing_speed_wpm * CHARS_PER_WORD / 60

    @property
    def reading_cps(self) -> float:
        return self.reading_speed_wpm * CHARS_PER_WORD / 60

    @field_validator("reaction_seconds", "pause_between_messages_seconds")
    @classmethod
    def _validate_range(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if low < 0 or high < low:
            raise ValueError("ranges must satisfy 0 <= min <= max")
        return value


class InitiativeProfile(_Profile):
    """When the personality may start a conversation on its own."""

    enabled: bool = True
    check_interval_minutes: float = Field(default=15.0, gt=0)
    min_quiet_minutes: float = Field(default=90.0, ge=1)
    """The channel must have been silent at least this long."""
    recent_activity_hours: float = Field(default=24.0, gt=0)
    """Someone (human) must have spoken within this window; no talking to empty rooms."""
    min_interval_minutes: float = Field(default=240.0, ge=1)
    """Minimum time between two openers in the same channel."""
    max_per_channel_per_day: int = Field(default=2, ge=0)
    chance: float = Field(default=0.3, ge=0, le=1)
    """Probability of acting when every other condition is met."""
    active_hours_utc: tuple[int, int] | None = None
    """Optional ``(start_hour, end_hour)`` window in UTC; may wrap past midnight."""

    @field_validator("active_hours_utc")
    @classmethod
    def _validate_hours(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is not None and not all(0 <= hour <= 23 for hour in value):
            raise ValueError("hours must be between 0 and 23")
        return value


class BehaviorProfile(_Profile):
    """How the personality decides when to talk."""

    conversation_timeout_seconds: int = Field(default=600, ge=30)
    """Silence after which a conversation is over."""
    focus_timeout_seconds: int = Field(default=120, ge=5)
    """How long the current partner keeps priority after their last message."""
    name_mention_reply_chance: float = Field(default=0.85, ge=0, le=1)
    """Chance to answer when its name appears in text without an @mention."""
    spontaneous_reply_chance: float = Field(default=0.05, ge=0, le=1)
    """Chance to join an idle-channel message that matches its interests."""
    spontaneous_min_keyword_hits: int = Field(default=1, ge=1)
    spontaneous_min_words: int = Field(default=6, ge=1)
    spontaneous_cooldown_seconds: int = Field(default=900, ge=0)
    """Minimum time since its last message in the channel before joining uninvited."""
    opener_reply_window_seconds: int = Field(default=600, ge=0)
    """After posting an opener, treat the next message within this window as a reply."""
    respond_to_bots: bool = False
    max_replies_per_channel_per_minute: int = Field(default=15, ge=1)
    max_replies_per_user_per_minute: int = Field(default=12, ge=1)
    """Generous enough for a fast one-on-one conversation; replies to a burst of
    messages are batched, so spam still cannot push the bot past these."""
    stale_trigger_seconds: int = Field(default=180, ge=10)
    """Drop queued triggers older than this instead of answering late."""
    initiative: InitiativeProfile = InitiativeProfile()


class MemoryProfile(_Profile):
    """How much the personality remembers."""

    history_messages: int = Field(default=16, ge=2, le=200)
    """Recent channel messages included in each prompt. The window may grow by
    ``history_slack`` messages before it advances, so the start of the prompt
    stays stable for several turns and model servers can reuse their cache."""
    history_char_budget: int = Field(default=4000, ge=500)
    """Character budget for history; oldest messages are dropped first."""
    long_term_enabled: bool = True
    recall_limit: int = Field(default=6, ge=0, le=50)
    """Long-term memories about the user included in each prompt."""
    max_memories_per_user: int = Field(default=40, ge=1)
    min_messages_for_extraction: int = Field(default=3, ge=1)

    @property
    def history_slack(self) -> int:
        return max(self.history_messages // 2, 1)

    @property
    def max_history_messages(self) -> int:
        return self.history_messages + self.history_slack


class ModelProfile(_Profile):
    """Which model to use and how to sample it."""

    provider: str | None = None
    """Provider name; ``None`` uses ``ARCANE_AI_PROVIDER``."""
    model: str | None = None
    """Model name; ``None`` uses the provider's default (e.g. ``ARCANE_OLLAMA_MODEL``)."""
    temperature: float = Field(default=0.85, ge=0, le=2)
    top_p: float | None = Field(default=0.92, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repeat_penalty: float | None = Field(default=1.1, gt=0)
    max_tokens: int = Field(default=200, ge=16, le=4096)
    """Generation cap per reply. Short caps keep local models fast."""
    context_window: int | None = Field(default=4096, ge=512)
    """``num_ctx`` for Ollama. Every request (replies and memory extraction) uses the
    same value, because changing it between requests makes Ollama reload the model."""


class Personality(_Profile):
    """Everything that defines one AI participant."""

    id: str
    name: str
    """The name it goes by in conversation."""
    aliases: tuple[str, ...] = ()
    """Other names that count as being addressed (case-insensitive)."""
    description: str
    identity: str
    """Who it is, written in second person ("You are ...")."""
    traits: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()
    interest_keywords: frozenset[str] = frozenset()
    """Lowercase words or phrases that signal a topic it cares about."""
    conversation_topics: tuple[str, ...] = ()
    """Seeds for conversations it starts on its own."""
    example_messages: tuple[str, ...] = ()
    """Short samples of its voice, shown to the model as style reference."""

    style: StyleProfile = StyleProfile()
    timing: TimingProfile = TimingProfile()
    behavior: BehaviorProfile = BehaviorProfile()
    memory: MemoryProfile = MemoryProfile()
    model: ModelProfile = ModelProfile()

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not PERSONALITY_ID_PATTERN.match(value):
            raise ValueError("id must be lowercase letters, digits, underscores")
        return value

    @field_validator("interest_keywords", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: object) -> frozenset[str]:
        if not isinstance(value, (set, frozenset, list, tuple)):
            raise ValueError("interest_keywords must be a collection of strings")
        return frozenset(str(word).strip().lower() for word in value if str(word).strip())

    @model_validator(mode="after")
    def _validate_identity(self) -> Self:
        if not self.name.strip() or not self.identity.strip():
            raise ValueError("name and identity must not be empty")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        """All names it answers to, lowercased."""
        return tuple(dict.fromkeys(n.lower() for n in (self.name, *self.aliases) if n.strip()))

    def is_named_in(self, text: str) -> bool:
        """True when one of its names appears in ``text`` as a whole word."""
        lowered = text.lower()
        return any(
            re.search(rf"(?<![\w@]){re.escape(name)}(?![\w@])", lowered) is not None
            for name in self.names
        )

    def interest_hits(self, text: str) -> int:
        """How many of its interest keywords (or phrases) appear in ``text``."""
        lowered = text.lower()
        words = set(_WORD_RE.findall(lowered))
        hits = 0
        for keyword in self.interest_keywords:
            if " " in keyword:
                hits += keyword in lowered
            else:
                hits += keyword in words
        return hits
