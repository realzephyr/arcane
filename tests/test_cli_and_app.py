"""Tests for the CLI commands and the application lifecycle (Discord is faked)."""

from __future__ import annotations

import asyncio
import builtins
import io
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import discord
import pytest

from arcane import __version__
from arcane.app import EXIT_FAILURE, EXIT_OK, Application
from arcane.bot.bot import ArcaneBot
from arcane.cli import TerminalTransport, _chat, _check, main
from arcane.config.settings import OllamaSettings, Settings, load_bot_configs
from arcane.personalities.registry import load_personality
from tests.fakes import FakeOllama, chat_body


def _settings(
    tmp_path: Path, base_url: str = "http://127.0.0.1:9", **overrides: object
) -> Settings:
    fields: dict[str, object] = {
        "database_path": tmp_path / "arcane.db",
        "humanize": False,
        "ollama": OllamaSettings(base_url=base_url, max_retries=0),
    }
    fields.update(overrides)
    return Settings.model_validate(fields)


# --------------------------------------------------------------------------- main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_invalid_configuration_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("ARCANE_LOG_LEVEL", "LOUD")
    assert main(["--env-file", str(tmp_path / "none.env"), "run"]) == EXIT_FAILURE
    assert "ARCANE_LOG_LEVEL" in capsys.readouterr().err


def test_run_without_token_fails_fast(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setenv("ARCANE_DATABASE_PATH", str(tmp_path / "x.db"))
    assert main(["--env-file", str(tmp_path / "none.env"), "run"]) == EXIT_FAILURE
    assert "ARCANE_BOT_MP3_TOKEN" in caplog.text


# -------------------------------------------------------------------------- check


async def test_check_passes_with_everything_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_ollama: tuple[FakeOllama, str]
) -> None:
    _, url = fake_ollama
    monkeypatch.setenv("ARCANE_BOT_MP3_TOKEN", "token")
    out = io.StringIO()
    assert await _check(_settings(tmp_path, url), out) == EXIT_OK
    report = out.getvalue()
    assert "All checks passed." in report
    assert "token for mp3" in report
    assert "model llama3.1:8b" in report


async def test_check_reports_missing_model_and_token(
    tmp_path: Path, fake_ollama: tuple[FakeOllama, str]
) -> None:
    fake, url = fake_ollama
    fake.models = ["something-else:latest"]
    out = io.StringIO()
    assert await _check(_settings(tmp_path, url), out) == EXIT_FAILURE
    report = out.getvalue()
    assert "ARCANE_BOT_MP3_TOKEN" in report
    assert "ollama pull llama3.1:8b" in report


async def test_check_reports_unreachable_backend(tmp_path: Path) -> None:
    out = io.StringIO()
    assert await _check(_settings(tmp_path), out) == EXIT_FAILURE
    assert "ollama backend" in out.getvalue()


# --------------------------------------------------------------------------- chat


@pytest.fixture
def scripted_input(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    lines: list[str] = []

    def fake_input(prompt: str = "") -> str:
        if not lines:
            raise EOFError
        return lines.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    yield lines


async def test_terminal_chat_round_trip(
    tmp_path: Path,
    fake_ollama: tuple[FakeOllama, str],
    scripted_input: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake, url = fake_ollama
    fake.chat_responses.append((200, chat_body("mp3: honestly? both")))
    scripted_input.extend(["is math invented or discovered?", "", "/quit"])

    code = await _chat(
        _settings(tmp_path, url),
        personality_id="mp3",
        user_name="tester",
        humanize=False,
        persist=False,
    )

    assert code == EXIT_OK
    assert "mp3: honestly? both" in capsys.readouterr().out
    assert len(fake.requests) == 1
    assert not (tmp_path / "arcane.db").exists()  # throwaway in-memory database


async def test_terminal_transport_prints_messages() -> None:
    out = io.StringIO()
    transport = TerminalTransport("mp3", out)
    async with transport.typing(1):
        sent = await transport.send(1, "hello")
    assert out.getvalue() == "mp3: hello\n"
    assert sent.author_name == "mp3"
    assert (await transport.channel_info(1)) is not None


# ---------------------------------------------------------------------- lifecycle


def _app(tmp_path: Path, url: str, **settings_overrides: object) -> Application:
    settings = _settings(tmp_path, url, **settings_overrides)
    configs = load_bot_configs(settings, {"ARCANE_BOT_MP3_TOKEN": "token"})
    return Application(settings, configs)


async def test_application_runs_until_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_ollama: tuple[FakeOllama, str]
) -> None:
    started = asyncio.Event()

    async def fake_start(self: ArcaneBot, token: str, *, reconnect: bool = True) -> None:
        assert token == "token"
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(ArcaneBot, "start", fake_start)
    app = _app(tmp_path, fake_ollama[1])
    run = asyncio.create_task(app.run())
    await asyncio.wait_for(started.wait(), 5)

    app.request_stop()
    assert await asyncio.wait_for(run, 5) == EXIT_OK
    assert (tmp_path / "arcane.db").exists()
    # The model was preloaded with the num_ctx replies use, before connecting.
    fake = fake_ollama[0]
    assert [load["options"].get("num_ctx") for load in fake.loads] == [
        load_personality("mp3").model.context_window
    ]


async def test_application_reports_login_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_ollama: tuple[FakeOllama, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def failing_start(self: ArcaneBot, token: str, *, reconnect: bool = True) -> None:
        raise discord.LoginFailure("bad token")

    monkeypatch.setattr(ArcaneBot, "start", failing_start)
    assert await asyncio.wait_for(_app(tmp_path, fake_ollama[1]).run(), 5) == EXIT_FAILURE
    assert "Discord rejected the token" in caplog.text


async def test_application_warns_when_backend_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def quick_exit(self: ArcaneBot, token: str, *, reconnect: bool = True) -> None:
        return None

    monkeypatch.setattr(ArcaneBot, "start", quick_exit)
    assert await _app(tmp_path, "http://127.0.0.1:9").run() == EXIT_OK
    assert "Bots will stay silent" in caplog.text


async def test_initiative_task_runs_unless_disabled(
    tmp_path: Path, fake_ollama: tuple[FakeOllama, str]
) -> None:
    quiet = _app(tmp_path, fake_ollama[1], initiative_enabled=False)
    eager = _app(tmp_path, fake_ollama[1])
    try:
        (quiet_runtime,) = [quiet._build_runtime(c) for c in quiet._bot_configs]
        (eager_runtime,) = [eager._build_runtime(c) for c in eager._bot_configs]
        assert [t.name for t in quiet_runtime.tasks] == ["mp3-maintenance"]
        assert [t.name for t in eager_runtime.tasks] == ["mp3-maintenance", "mp3-initiative"]
    finally:
        for app in (quiet, eager):
            for runtime in app._runtimes:
                await runtime.client.close()
            await app._providers.close()


def test_application_requires_bots(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Application(_settings(tmp_path), [])
    assert isinstance(MagicMock(), MagicMock)
