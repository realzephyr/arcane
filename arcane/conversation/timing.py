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
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from arcane.personalities.base import TimingProfile

_WORD_RE = re.compile(r"[a-z0-9']+")
_GREETINGS = frozenset(
    [
        "yo",
        "yoo",
        "yooo",
        "hey",
        "heyy",
        "hi",
        "hii",
        "hello",
        "sup",
        "wsg",
        "wassup",
        "ayo",
        "oi",
        "u",
        "you",
        "there",
        "here",
        "ok",
        "okay",
        "bro",
        "bruh",
        "dude",
        "man",
    ]
)
"""Words that make up a bare attention-getter like "yo mp3 u there"."""


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

    def follow_up_wait(self, incoming_text: str, names: Iterable[str] = ()) -> float:
        """Extra time to wait for the rest when ``incoming_text`` is only an opener.

        An opener is a message with nothing in it but greetings, the bot's
        names and mentions ("yo", "mp3", "hey mp3", "@mp3 u there"): the actual
        message usually follows a moment later. Real messages, however short
        ("lol", "fair point"), get no wait.
        """
        if not self.enabled or ("?" in incoming_text and len(incoming_text) > 20):
            return 0.0
        words = _WORD_RE.findall(incoming_text.lower())
        filler = _GREETINGS | {name.lower() for name in names}
        joined = " ".join(words)
        for name in names:
            joined = joined.replace(name.lower(), " ")
        leftover = [word for word in joined.split() if word not in filler]
        if not words or leftover:
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
