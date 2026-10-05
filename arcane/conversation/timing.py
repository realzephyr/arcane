"""Human-like timing.

People don't answer instantly. They notice a message, read it, think, and
type at a finite speed, and they're never perfectly consistent.
:class:`HumanTiming` models each of those steps from the personality's
:class:`~arcane.personalities.base.TimingProfile`:

* **reading**: reaction time plus message length at reading speed;
* **thinking**: a base delay plus a bonus for complex messages;
* **typing**: reply length at typing speed, clamped to sensible bounds;
* **pauses** between consecutive messages of a split reply.

Every duration gets multiplicative log-normal noise, which is always positive
and skews long like real human timing. When humanisation is disabled every
delay is zero, which keeps development and tests fast.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from arcane.personalities.base import TimingProfile

MAX_READING_SECONDS = 12.0
MAX_THINKING_SECONDS = 7.0
COMPLEXITY_THINKING_BONUS = 3.0

_DEEP_WORDS = re.compile(
    r"\b(why|how|explain|think|believe|opinion|argue|argument|disagree|agree|"
    r"meaning|ethic\w*|moral\w*|conscious\w*|exist\w*|would you|what if)\b",
    re.IGNORECASE,
)


def estimate_complexity(text: str) -> float:
    """Rough complexity of a message in ``[0, 1]``: length, questions, depth."""
    if not text.strip():
        return 0.0
    length_score = min(len(text) / 400, 1.0)
    question_score = min(text.count("?") / 2, 1.0)
    depth_score = min(len(_DEEP_WORDS.findall(text)) / 3, 1.0)
    return min(0.5 * length_score + 0.2 * question_score + 0.3 * depth_score, 1.0)


@dataclass(slots=True)
class HumanTiming:
    """Computes human-like delays for one personality."""

    profile: TimingProfile
    enabled: bool = True
    rng: random.Random = field(default_factory=random.Random)

    # --------------------------------------------------------------- components

    def reading_delay(self, incoming_text: str) -> float:
        """Time between a message arriving and starting to think about it."""
        if not self.enabled:
            return 0.0
        reaction = self._uniform(self.profile.reaction_seconds)
        reading = len(incoming_text) / self.profile.reading_speed_cps
        return min(self._noisy(reaction + reading), MAX_READING_SECONDS)

    def thinking_delay(self, incoming_text: str) -> float:
        """Time spent thinking before typing starts."""
        if not self.enabled:
            return 0.0
        base = self._uniform(self.profile.thinking_seconds)
        bonus = estimate_complexity(incoming_text) * COMPLEXITY_THINKING_BONUS
        return min(self._noisy(base + bonus), MAX_THINKING_SECONDS)

    def typing_duration(self, outgoing_text: str) -> float:
        """How long typing ``outgoing_text`` takes."""
        if not self.enabled:
            return 0.0
        raw = self._noisy(len(outgoing_text) / self.profile.typing_speed_cps)
        return min(max(raw, self.profile.min_typing_seconds), self.profile.max_typing_seconds)

    def pause_between_messages(self) -> float:
        """Short pause after sending one message before typing the next."""
        if not self.enabled:
            return 0.0
        return self._uniform(self.profile.pause_between_messages_seconds)

    # --------------------------------------------------------------- composites

    def response_lead_time(self, incoming_text: str) -> float:
        """Reading plus thinking: everything before the first keystroke."""
        return self.reading_delay(incoming_text) + self.thinking_delay(incoming_text)

    def debounce_window(self) -> float:
        """How long to wait for follow-up messages before answering."""
        if not self.enabled:
            return 0.0
        return self.profile.debounce_seconds

    def max_debounce(self) -> float:
        if not self.enabled:
            return 0.0
        return self.profile.max_debounce_seconds

    # ---------------------------------------------------------------- internals

    def _uniform(self, bounds: tuple[float, float]) -> float:
        low, high = bounds
        return self.rng.uniform(low, high)

    def _noisy(self, value: float) -> float:
        sigma = self.profile.variation
        if sigma <= 0 or value <= 0:
            return max(value, 0.0)
        # Median-preserving log-normal noise.
        return value * math.exp(self.rng.gauss(0.0, sigma))
