"""Exception hierarchy shared across Arcane.

Every error raised deliberately by Arcane derives from :class:`ArcaneError`, so
callers can distinguish expected failures from programming errors.
"""

from __future__ import annotations


class ArcaneError(Exception):
    """Base class for all Arcane errors."""


class ConfigurationError(ArcaneError):
    """The application is misconfigured (missing token, invalid value, ...)."""


class PersonalityNotFoundError(ArcaneError):
    """A requested personality id does not exist or failed to load."""
