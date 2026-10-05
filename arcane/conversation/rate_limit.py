"""Anti-spam rate limiting.

Even direct mentions are rate limited: otherwise anyone could make the bot
flood a channel (and saturate the model) by spamming @mentions. Limits are
sliding windows per channel and per user.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from datetime import datetime, timedelta


class SlidingWindowCounter:
    """Counts events per key within a moving time window."""

    def __init__(self, window: timedelta) -> None:
        self._window = window
        self._events: dict[Hashable, deque[datetime]] = {}

    def count(self, key: Hashable, now: datetime) -> int:
        events = self._events.get(key)
        if not events:
            return 0
        self._evict(events, now)
        return len(events)

    def add(self, key: Hashable, now: datetime) -> None:
        events = self._events.setdefault(key, deque())
        self._evict(events, now)
        events.append(now)

    def cleanup(self, now: datetime) -> None:
        """Drop keys with no events in the window (bounds memory use)."""
        for key in list(self._events):
            events = self._events[key]
            self._evict(events, now)
            if not events:
                del self._events[key]

    def __len__(self) -> int:
        return len(self._events)

    def _evict(self, events: deque[datetime], now: datetime) -> None:
        cutoff = now - self._window
        while events and events[0] <= cutoff:
            events.popleft()


class ReplyRateLimiter:
    """Caps how often a bot replies per channel and per user."""

    def __init__(
        self,
        *,
        per_channel: int,
        per_user: int,
        window: timedelta = timedelta(minutes=1),
    ) -> None:
        self._per_channel = per_channel
        self._per_user = per_user
        self._channels = SlidingWindowCounter(window)
        self._users = SlidingWindowCounter(window)

    def allows(self, channel_id: int, user_id: int | None, now: datetime) -> bool:
        if self._channels.count(channel_id, now) >= self._per_channel:
            return False
        return user_id is None or self._users.count(user_id, now) < self._per_user

    def record(self, channel_id: int, user_id: int | None, now: datetime) -> None:
        self._channels.add(channel_id, now)
        if user_id is not None:
            self._users.add(user_id, now)

    def cleanup(self, now: datetime) -> None:
        self._channels.cleanup(now)
        self._users.cleanup(now)
