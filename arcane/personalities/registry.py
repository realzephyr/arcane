"""Discovery and loading of personalities.

A personality lives in ``arcane/personalities/<id>/personality.py`` and exports
a module-level ``PERSONALITY`` instance of
:class:`~arcane.personalities.base.Personality`. The registry imports it by id,
so adding a personality never requires touching framework code.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from functools import cache
from pathlib import Path

from arcane.config.settings import PERSONALITY_ID_PATTERN
from arcane.core.errors import PersonalityNotFoundError
from arcane.personalities.base import Personality

logger = logging.getLogger(__name__)

_PACKAGE = "arcane.personalities"
_PACKAGE_DIR = Path(__file__).resolve().parent


def available_personalities() -> tuple[str, ...]:
    """Ids of every personality package that contains a ``personality.py``."""
    found = [
        module.name
        for module in pkgutil.iter_modules([str(_PACKAGE_DIR)])
        if module.ispkg and (_PACKAGE_DIR / module.name / "personality.py").is_file()
    ]
    return tuple(sorted(found))


@cache
def load_personality(personality_id: str) -> Personality:
    """Import and validate the personality called ``personality_id``.

    Raises:
        PersonalityNotFoundError: if it does not exist or is invalid.
    """
    if not PERSONALITY_ID_PATTERN.match(personality_id):
        raise PersonalityNotFoundError(f"Invalid personality id {personality_id!r}")

    module_name = f"{_PACKAGE}.{personality_id}.personality"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and module_name.startswith(exc.name):
            available = ", ".join(available_personalities()) or "none"
            raise PersonalityNotFoundError(
                f"Unknown personality '{personality_id}'. Available: {available}"
            ) from exc
        raise

    personality = getattr(module, "PERSONALITY", None)
    if not isinstance(personality, Personality):
        raise PersonalityNotFoundError(
            f"{module_name} must define PERSONALITY as a Personality instance"
        )
    if personality.id != personality_id:
        raise PersonalityNotFoundError(
            f"{module_name} declares id '{personality.id}', expected '{personality_id}'"
        )
    logger.debug("Loaded personality %s (%s)", personality.id, personality.description)
    return personality
