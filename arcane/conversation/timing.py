"""Human-like pacing.

A person in a chat notices a message, reads it, then types a reply at a finite
speed. :class:`HumanTiming` models exactly those steps from the personality's
:class:`~arcane.personalities.base.TimingProfile`:

* **reaction**: a short moment to notice new messages;
* **reading**: message length at reading speed (no typing indicator yet);
* **typing**: reply length at typing speed (60 wpm by default). The typing
  indicator starts when the model starts generating, and generation time counts
  towards the typing time;
* **pauses** between consecutive messages of a split reply.

Reading and typing times get multiplicative log-normal noise, which is always
positive and skews slightly long like real human timing. When humanisation is
disabled every delay is zero, which keeps development and tests fast.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from arcane.personalities.base import TimingProfile


@dataclass(slots=True)
class HumanTiming:
    """Computes human-like delays for one personality."""

    profile: TimingProfile
    enabled: bool = True
    rng: random.Random = field(default_factory=random.Random)

    def reaction_delay(self) -> float:
        """Time to notice new messages before starting to read them."""
        if not self.enabled:
            return 0.0
        return self._uniform(self.profile.reaction_seconds)

    def reading_time(self, incoming_text: str) -> float:
        """Time to read ``incoming_text`` (without the reaction delay)."""
        if not self.enabled or not incoming_text:
            return 0.0
        return self._noisy(len(incoming_text) / self.profile.reading_cps)

    def max_reading_seconds(self) -> float:
        """Upper bound for a whole reading phase, follow-up messages included."""
        if not self.enabled:
            return 0.0
        return self.profile.max_reading_seconds

    def typing_duration(self, outgoing_text: str) -> float:
        """How long a human typing at the profile's speed needs for ``outgoing_text``."""
        if not self.enabled:
            return 0.0
        raw = self._noisy(len(outgoing_text) / self.profile.typing_cps)
        return min(max(raw, self.profile.min_typing_seconds), self.profile.max_typing_seconds)

    def pause_between_messages(self) -> float:
        """Short pause after sending one message before typing the next."""
        if not self.enabled:
            return 0.0
        return self._uniform(self.profile.pause_between_messages_seconds)

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
