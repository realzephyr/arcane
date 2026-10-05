"""Human-like pacing.

A person in a chat notices a message, reads it while working out what to say,
then types the reply. :class:`HumanTiming` models those steps from the
personality's :class:`~arcane.personalities.base.TimingProfile`:

* **reaction**: a short moment to notice new messages;
* **reading**: message length at reading speed, plus a short wait when the
  message is only an attention-getter ("yo", "mp3"). The model generates the
  reply during this phase, with no typing indicator;
* **typing**: once the reply is ready, a start-up moment plus the reply's
  length at typing speed, so short replies are quick and long ones take longer;
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

ATTENTION_GETTER_CHARS = 15
"""Messages up to this long without a question mark ("yo", "mp3", "hey u there")
usually have the actual message following right behind them."""


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

    def follow_up_wait(self, incoming_text: str) -> float:
        """Extra time to wait for the rest when ``incoming_text`` is just an opener."""
        if not self.enabled:
            return 0.0
        text = incoming_text.strip()
        if len(text) > ATTENTION_GETTER_CHARS or "?" in text:
            return 0.0
        return self.profile.follow_up_wait_seconds

    def max_reading_seconds(self) -> float:
        """Upper bound for a whole reading phase, follow-up messages included."""
        if not self.enabled:
            return 0.0
        return self.profile.max_reading_seconds

    def typing_duration(self, outgoing_text: str) -> float:
        """How long the typing indicator shows before ``outgoing_text`` is sent."""
        if not self.enabled:
            return 0.0
        profile = self.profile
        raw = self._noisy(profile.typing_start_seconds + len(outgoing_text) / profile.typing_cps)
        return min(max(raw, profile.min_typing_seconds), profile.max_typing_seconds)

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
