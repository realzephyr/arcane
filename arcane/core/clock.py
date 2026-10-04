"""Time helpers.

Arcane uses timezone-aware UTC datetimes everywhere in memory and unix
timestamps (float seconds) in the database. These helpers keep conversions in
one place.
"""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_timestamp(value: datetime) -> float:
    """Convert a datetime to a unix timestamp, treating naive values as UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


def from_timestamp(value: float) -> datetime:
    """Convert a unix timestamp to a timezone-aware UTC datetime."""
    return datetime.fromtimestamp(value, tz=UTC)
