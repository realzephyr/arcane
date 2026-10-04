from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import SecretStr

from arcane.config.logging import SecretRedactingFilter, redact
from arcane.config.settings import (
    BotConfig,
    Settings,
    load_bot_configs,
    load_settings,
    parse_id_list,
)
from arcane.core.errors import ConfigurationError

FAKE_TOKEN = "MTIzNDU2Nzg5MDEyMzQ1Njc4OQ.GabcDE.abcdefghijklmnopqrstuvwxyz0123456789"


def test_defaults_are_safe() -> None:
    settings = Settings()
    assert settings.bots == ("mp3",)
    assert settings.env == "production"
    assert settings.allowed_channel_ids == ()
    assert settings.initiative_channel_ids == ()
    assert settings.humanize is True
    assert settings.ollama.base_url == "http://localhost:11434"


def test_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARCANE_BOTS", "mp3, debate_bot")
    monkeypatch.setenv("ARCANE_ALLOWED_CHANNEL_IDS", "111, 222;333")
    monkeypatch.setenv("ARCANE_LOG_LEVEL", "debug")
    monkeypatch.setenv("ARCANE_LOG_FILE", "")
    monkeypatch.setenv("ARCANE_OLLAMA_BASE_URL", "http://gpu-box:11434/")
    monkeypatch.setenv("ARCANE_OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("ARCANE_OLLAMA_THINK", "false")

    settings = load_settings(env_file=None)

    assert settings.bots == ("mp3", "debate_bot")
    assert settings.allowed_channel_ids == (111, 222, 333)
    assert settings.log_level == "DEBUG"
    assert settings.log_file is None
    assert settings.ollama.base_url == "http://gpu-box:11434"
    assert settings.ollama.model == "qwen3:8b"
    assert settings.ollama.think is False


def test_env_file_is_loaded_without_overriding_real_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ARCANE_HUMANIZE=false\nARCANE_LOG_LEVEL=ERROR\n")
    monkeypatch.setenv("ARCANE_LOG_LEVEL", "WARNING")

    settings = load_settings(env_file=env_file)

    assert settings.humanize is False
    assert settings.log_level == "WARNING"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ARCANE_BOTS", "Bad-Name"),
        ("ARCANE_BOTS", " , "),
        ("ARCANE_LOG_LEVEL", "LOUD"),
        ("ARCANE_ALLOWED_CHANNEL_IDS", "abc"),
        ("ARCANE_ALLOWED_CHANNEL_IDS", "-5"),
        ("ARCANE_MAX_MESSAGES_PER_CHANNEL", "5"),
        ("ARCANE_OLLAMA_BASE_URL", "localhost:11434"),
    ],
)
def test_invalid_values_raise_configuration_error(
    monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigurationError) as excinfo:
        load_settings(env_file=None)
    assert key in str(excinfo.value)


def test_parse_id_list_deduplicates() -> None:
    assert parse_id_list("1,2,2,3") == (1, 2, 3)
    assert parse_id_list(None) == ()
    assert parse_id_list([5, "6"]) == (5, 6)


def test_bot_configs_use_per_bot_overrides() -> None:
    settings = Settings(bots=("mp3", "debate_bot"), allowed_channel_ids=(1,))
    env = {
        "ARCANE_BOT_MP3_TOKEN": "token-a",
        "ARCANE_BOT_MP3_MODEL": "gemma3:12b",
        "ARCANE_BOT_DEBATE_BOT_TOKEN": "token-b",
        "ARCANE_BOT_DEBATE_BOT_ALLOWED_CHANNEL_IDS": "7,8",
    }

    mp3, debate = load_bot_configs(settings, env)

    assert mp3.personality_id == "mp3"
    assert mp3.token.get_secret_value() == "token-a"
    assert mp3.model == "gemma3:12b"
    assert mp3.allowed_channel_ids == (1,)
    assert debate.allowed_channel_ids == (7, 8)
    assert debate.model is None


def test_missing_token_is_reported() -> None:
    with pytest.raises(ConfigurationError, match="ARCANE_BOT_MP3_TOKEN"):
        load_bot_configs(Settings(), {})


def test_shared_token_is_rejected() -> None:
    settings = Settings(bots=("mp3", "other"))
    env = {"ARCANE_BOT_MP3_TOKEN": "same", "ARCANE_BOT_OTHER_TOKEN": "same"}
    with pytest.raises(ConfigurationError, match="reuses"):
        load_bot_configs(settings, env)


def test_token_is_not_exposed_in_repr() -> None:
    config = BotConfig(personality_id="mp3", token=SecretStr(FAKE_TOKEN))
    assert FAKE_TOKEN not in repr(config)
    assert FAKE_TOKEN not in str(config.model_dump())


def test_log_filter_redacts_tokens() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "token is %s", (FAKE_TOKEN,), None)
    SecretRedactingFilter().filter(record)
    assert FAKE_TOKEN not in record.getMessage()
    assert "[REDACTED]" in record.getMessage()
    assert redact(f"x {FAKE_TOKEN} y") == "x [REDACTED] y"
