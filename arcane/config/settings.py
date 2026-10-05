"""Environment-driven application settings.

Configuration is read from environment variables, optionally pre-loaded from a
``.env`` file. Real environment variables always win over ``.env`` values.

Global settings use the ``ARCANE_`` prefix, Ollama settings use
``ARCANE_OLLAMA_``, and per-bot settings follow
``ARCANE_BOT_<PERSONALITY_ID>_<SETTING>`` (see :func:`load_bot_configs`).
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from arcane.core.errors import ConfigurationError

PERSONALITY_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
"""Personality ids double as Python package names and env-var fragments."""

_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

ChannelIdList = Annotated[tuple[int, ...], NoDecode]
"""Comma-separated channel ids in the environment, a tuple of ints in code."""


def parse_id_list(value: Any) -> tuple[int, ...]:
    """Parse ``"1, 2;3"`` (or an iterable of ids) into a tuple of positive ints."""
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = (part.strip() for part in re.split(r"[,;\s]+", value))
    elif isinstance(value, Iterable):
        items = value
    else:
        items = (value,)

    ids: list[int] = []
    for item in items:
        if item in ("", None):
            continue
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid id {item!r}: ids must be integers") from exc
        if parsed <= 0:
            raise ValueError(f"invalid id {item!r}: ids must be positive")
        ids.append(parsed)
    return tuple(dict.fromkeys(ids))  # de-duplicate, keep order


class OllamaSettings(BaseSettings):
    """Connection settings for the Ollama inference server."""

    model_config = SettingsConfigDict(env_prefix="ARCANE_OLLAMA_", extra="ignore")

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout_seconds: float = Field(default=120.0, gt=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    keep_alive: str = "24h"
    """How long Ollama keeps the model loaded after a request ("-1m" = forever)."""
    max_concurrent_requests: int = Field(default=1, ge=1, le=64)
    max_retries: int = Field(default=2, ge=0, le=10)
    think: bool | None = None
    """Ollama's ``think`` flag. ``None`` (the default) decides per model: thinking is
    turned off for models that have a thinking mode, because it delays every reply."""

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value

    @field_validator("think", mode="before")
    @classmethod
    def _empty_think_is_none(cls, value: Any) -> Any:
        return None if value == "" else value


class Settings(BaseSettings):
    """Global application settings."""

    model_config = SettingsConfigDict(env_prefix="ARCANE_", extra="ignore")

    env: Literal["development", "production"] = "production"
    log_level: str = "INFO"
    log_file: Path | None = None

    bots: Annotated[tuple[str, ...], NoDecode] = ("mp3",)
    ai_provider: str = "ollama"

    database_path: Path = Path("data/arcane.db")
    short_term_ttl_hours: float = Field(default=24.0, gt=0)
    max_messages_per_channel: int = Field(default=300, ge=20, le=10_000)
    maintenance_interval_seconds: int = Field(default=300, ge=10)
    long_term_memory_enabled: bool = True

    allowed_channel_ids: ChannelIdList = ()
    initiative_channel_ids: ChannelIdList = ()
    initiative_enabled: bool = True
    respond_in_dms: bool = True
    humanize: bool = True
    conversation_timeout_seconds: int | None = Field(default=None, ge=30)

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)

    @property
    def is_development(self) -> bool:
        return self.env == "development"

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: Any) -> str:
        level = str(value).strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"must be one of {', '.join(sorted(_LOG_LEVELS))}")
        return level

    @field_validator("log_file", "conversation_timeout_seconds", mode="before")
    @classmethod
    def _empty_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("bots", mode="before")
    @classmethod
    def _parse_bots(cls, value: Any) -> tuple[str, ...]:
        raw = value.split(",") if isinstance(value, str) else list(value)
        ids = [str(item).strip().lower() for item in raw if str(item).strip()]
        if not ids:
            raise ValueError("at least one personality id is required")
        for personality_id in ids:
            if not PERSONALITY_ID_PATTERN.match(personality_id):
                raise ValueError(
                    f"invalid personality id {personality_id!r}: use lowercase letters, "
                    "digits and underscores, starting with a letter"
                )
        return tuple(dict.fromkeys(ids))

    @field_validator("allowed_channel_ids", "initiative_channel_ids", mode="before")
    @classmethod
    def _parse_channel_ids(cls, value: Any) -> tuple[int, ...]:
        return parse_id_list(value)


class BotConfig(BaseModel):
    """Runtime configuration for one bot account bound to one personality."""

    model_config = ConfigDict(frozen=True)

    personality_id: str
    token: SecretStr
    model: str | None = None
    allowed_channel_ids: tuple[int, ...] = ()
    initiative_channel_ids: tuple[int, ...] = ()

    @staticmethod
    def env_prefix(personality_id: str) -> str:
        return f"ARCANE_BOT_{personality_id.upper()}_"


def load_settings(env_file: Path | str | None = ".env") -> Settings:
    """Load settings from the environment, after pre-loading ``env_file`` if present.

    Raises:
        ConfigurationError: if any value fails validation.
    """
    if env_file is not None and Path(env_file).is_file():
        load_dotenv(env_file, override=False)
    try:
        ollama = OllamaSettings()
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc, "ARCANE_OLLAMA_")) from exc
    try:
        return Settings(ollama=ollama)
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc, "ARCANE_")) from exc


def load_bot_configs(
    settings: Settings,
    environ: Mapping[str, str] | None = None,
) -> list[BotConfig]:
    """Build a :class:`BotConfig` for every personality listed in ``ARCANE_BOTS``.

    Per-bot variables (``ARCANE_BOT_<ID>_...``):

    * ``TOKEN`` (required): Discord bot token.
    * ``MODEL``: model override for this bot.
    * ``ALLOWED_CHANNEL_IDS`` / ``INITIATIVE_CHANNEL_IDS``: override the global lists.

    Raises:
        ConfigurationError: if a token is missing or a value is invalid.
    """
    env = os.environ if environ is None else environ
    configs: list[BotConfig] = []
    seen_tokens: set[str] = set()

    for personality_id in settings.bots:
        prefix = BotConfig.env_prefix(personality_id)
        token = env.get(f"{prefix}TOKEN", "").strip()
        if not token:
            raise ConfigurationError(
                f"Missing Discord token for bot '{personality_id}'. Set {prefix}TOKEN."
            )
        if token in seen_tokens:
            raise ConfigurationError(
                f"Bot '{personality_id}' reuses another bot's token. "
                "Each personality needs its own Discord application."
            )
        seen_tokens.add(token)

        try:
            allowed = _optional_ids(env, f"{prefix}ALLOWED_CHANNEL_IDS")
            initiative = _optional_ids(env, f"{prefix}INITIATIVE_CHANNEL_IDS")
        except ValueError as exc:
            raise ConfigurationError(
                f"Invalid channel list for bot '{personality_id}': {exc}"
            ) from exc

        model = env.get(f"{prefix}MODEL", "").strip() or None
        configs.append(
            BotConfig(
                personality_id=personality_id,
                token=SecretStr(token),
                model=model,
                allowed_channel_ids=(settings.allowed_channel_ids if allowed is None else allowed),
                initiative_channel_ids=(
                    settings.initiative_channel_ids if initiative is None else initiative
                ),
            )
        )
    return configs


def _optional_ids(env: Mapping[str, str], key: str) -> tuple[int, ...] | None:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    return parse_id_list(raw)


def _format_validation_error(exc: ValidationError, env_prefix: str) -> str:
    lines = ["Invalid configuration:"]
    for error in exc.errors():
        location = "_".join(str(part) for part in error["loc"]).upper()
        lines.append(f"  - {env_prefix}{location}: {error['msg']}")
    return "\n".join(lines)
