"""Logging configuration.

Console logging is always enabled; a size-rotated log file is optional. A
redaction filter scrubs anything that looks like a Discord token from every
record as defence in depth: tokens should never reach a log call, but if one
ever does it is masked before being written.
"""

from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

# Discord bot tokens: base64(user id) . timestamp . HMAC
_DISCORD_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{23,28}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,}")
_REDACTED = "[REDACTED]"

# Third-party loggers that are too chatty at DEBUG/INFO.
_NOISY_LOGGERS = {
    "discord": logging.INFO,
    "discord.gateway": logging.WARNING,
    "discord.http": logging.WARNING,
    "aiosqlite": logging.WARNING,
    "asyncio": logging.WARNING,
}


class SecretRedactingFilter(logging.Filter):
    """Mask Discord-token-shaped strings in log messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if _DISCORD_TOKEN_RE.search(message):
            record.msg = _DISCORD_TOKEN_RE.sub(_REDACTED, message)
            record.args = None
        return True


def redact(text: str) -> str:
    """Return ``text`` with token-shaped substrings masked."""
    return _DISCORD_TOKEN_RE.sub(_REDACTED, text)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root logger. Safe to call more than once."""
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(numeric_level)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    redactor = SecretRedactingFilter()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.addFilter(redactor)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)

    for name, minimum in _NOISY_LOGGERS.items():
        logging.getLogger(name).setLevel(max(minimum, numeric_level))
