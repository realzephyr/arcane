"""Command-line interface.

python main.py [run]            start every bot in ARCANE_BOTS
python main.py check            validate configuration and dependencies
python main.py chat -p mp3      talk to a personality in the terminal
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import logging
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TextIO

from arcane import __version__
from arcane.ai.providers.factory import ProviderRegistry
from arcane.app import EXIT_FAILURE, EXIT_OK, Application
from arcane.config.logging import setup_logging
from arcane.config.settings import BotConfig, Settings, load_bot_configs, load_settings
from arcane.conversation.transport import SentMessage
from arcane.core.clock import utcnow
from arcane.core.errors import ArcaneError
from arcane.core.models import ChannelInfo, IncomingMessage
from arcane.database.database import MEMORY_DATABASE, Database
from arcane.personalities.registry import available_personalities, load_personality
from arcane.runtime import build_conversation_handler

logger = logging.getLogger("arcane")

TERMINAL_CHANNEL = ChannelInfo(channel_id=1, is_dm=True)
TERMINAL_USER_ID = 1
TERMINAL_BOT_ID = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arcane",
        description="Arcane: AI-powered conversational Discord bots.",
    )
    parser.add_argument("--version", action="version", version=f"arcane {__version__}")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file to load before reading the environment (default: .env)",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="start all configured bots (default)")
    commands.add_parser("check", help="validate configuration, database, and AI backend")

    chat = commands.add_parser("chat", help="chat with a personality in the terminal")
    chat.add_argument(
        "-p", "--personality", default=None, help="personality id (default: first bot)"
    )
    chat.add_argument("-n", "--name", default="you", help="your display name")
    chat.add_argument(
        "--humanize", action="store_true", help="simulate typing delays like on Discord"
    )
    chat.add_argument(
        "--persist",
        action="store_true",
        help="use the configured database instead of a throwaway in-memory one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "run"
    try:
        settings = load_settings(args.env_file)
    except ArcaneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE

    if command in {"chat", "check"}:
        # Keep the terminal readable: only warnings unless DEBUG was requested.
        level = "DEBUG" if settings.log_level == "DEBUG" else "WARNING"
        setup_logging(level, settings.log_file)
    else:
        setup_logging(settings.log_level, settings.log_file)

    try:
        if command == "run":
            return _run(settings)
        if command == "check":
            return asyncio.run(_check(settings))
        if command == "chat":
            return asyncio.run(
                _chat(
                    settings,
                    personality_id=args.personality or settings.bots[0],
                    user_name=args.name,
                    humanize=args.humanize,
                    persist=args.persist,
                )
            )
    except ArcaneError as exc:
        logger.error("%s", exc)
        return EXIT_FAILURE
    except KeyboardInterrupt:
        return EXIT_OK
    raise AssertionError(f"unknown command {command}")  # pragma: no cover


# --------------------------------------------------------------------------- run


def _run(settings: Settings) -> int:
    bot_configs = load_bot_configs(settings)
    for config in bot_configs:
        load_personality(config.personality_id)  # fail fast on unknown personalities
    logger.info(
        "Starting Arcane %s with %d bot(s): %s",
        __version__,
        len(bot_configs),
        ", ".join(c.personality_id for c in bot_configs),
    )
    return asyncio.run(Application(settings, bot_configs).run())


# ------------------------------------------------------------------------- check


class _Checklist:
    def __init__(self, out: TextIO) -> None:
        self._out = out
        self.failed = False

    def ok(self, label: str, detail: str = "") -> None:
        self._line("ok", label, detail)

    def warn(self, label: str, detail: str = "") -> None:
        self._line("warn", label, detail)

    def fail(self, label: str, detail: str = "") -> None:
        self.failed = True
        self._line("FAIL", label, detail)

    def _line(self, status: str, label: str, detail: str) -> None:
        suffix = f": {detail}" if detail else ""
        print(f"  [{status:>4}] {label}{suffix}", file=self._out)


async def _check(settings: Settings, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    checklist = _Checklist(out)
    print(f"Arcane {__version__} preflight\n", file=out)

    checklist.ok("settings", f"env={settings.env}, bots={', '.join(settings.bots)}")
    print(f"  available personalities: {', '.join(available_personalities())}", file=out)

    configs: list[BotConfig] = []
    try:
        configs = load_bot_configs(settings)
        for config in configs:
            checklist.ok(f"token for {config.personality_id}", "set")
    except ArcaneError as exc:
        checklist.fail("bot tokens", str(exc))

    for personality_id in settings.bots:
        try:
            personality = load_personality(personality_id)
            checklist.ok(f"personality {personality_id}", personality.description)
        except ArcaneError as exc:
            checklist.fail(f"personality {personality_id}", str(exc))

    try:
        async with Database(settings.database_path) as database:
            checklist.ok("database", f"{database.path} (schema v{await database.schema_version()})")
    except Exception as exc:
        checklist.fail("database", f"{settings.database_path}: {exc}")

    registry = ProviderRegistry(settings)
    try:
        overrides = {config.personality_id: config.model for config in configs}
        targets: dict[tuple[str, str], None] = {}
        for personality_id in settings.bots:
            try:
                profile = load_personality(personality_id).model
                provider = registry.get(profile.provider)
            except ArcaneError as exc:
                checklist.fail(f"AI provider for {personality_id}", str(exc))
                continue
            model = overrides.get(personality_id) or profile.model or provider.default_model
            targets[(provider.name, model)] = None

        for provider_name, model in targets:
            provider = registry.get(provider_name)
            health = await provider.health_check(model)
            if not health.available:
                checklist.fail(f"{provider_name} backend", health.detail)
            elif health.model_available:
                checklist.ok(f"{provider_name} model {model}", health.detail)
            else:
                checklist.fail(f"{provider_name} model {model}", health.detail)
    finally:
        await registry.close()

    print("\nAll checks passed." if not checklist.failed else "\nSome checks failed.", file=out)
    return EXIT_FAILURE if checklist.failed else EXIT_OK


# -------------------------------------------------------------------------- chat


class TerminalTransport:
    """Prints a personality's messages to the terminal."""

    def __init__(self, bot_name: str, out: TextIO | None = None) -> None:
        self._bot_name = bot_name
        self._out = out or sys.stdout
        self._ids = itertools.count(1_000_000)

    @asynccontextmanager
    async def typing(self, channel_id: int) -> AsyncIterator[None]:
        interactive = self._out.isatty()
        if interactive:
            self._out.write(f"\033[2m{self._bot_name} is typing...\033[0m")
            self._out.flush()
        try:
            yield
        finally:
            if interactive:
                self._out.write("\r\033[K")
                self._out.flush()

    async def send(
        self, channel_id: int, content: str, *, reply_to_message_id: int | None = None
    ) -> SentMessage:
        print(f"{self._bot_name}: {content}", file=self._out, flush=True)
        return SentMessage(next(self._ids), utcnow(), TERMINAL_BOT_ID, self._bot_name)

    async def channel_info(self, channel_id: int) -> ChannelInfo | None:
        return TERMINAL_CHANNEL


async def _chat(
    settings: Settings,
    *,
    personality_id: str,
    user_name: str,
    humanize: bool,
    persist: bool,
) -> int:
    personality = load_personality(personality_id)
    chat_settings = settings.model_copy(update={"humanize": humanize, "respond_in_dms": True})
    database = Database(settings.database_path if persist else MEMORY_DATABASE)
    registry = ProviderRegistry(settings)
    await database.connect()
    try:
        provider = registry.get(personality.model.provider)
        handler = build_conversation_handler(
            personality=personality,
            settings=chat_settings,
            database=database,
            provider=provider,
            transport=TerminalTransport(personality.name),
        )
        model = personality.model.model or provider.default_model
        health = await provider.health_check(model)
        if not health.available or health.model_available is False:
            print(f"warning: {health.detail}", file=sys.stderr)
        else:
            await provider.warm_up(model=model, options=handler.response_manager.base_options())

        print(f"Chatting with {personality.name}. Type /quit to leave.\n")
        message_ids = itertools.count(1)
        while True:
            try:
                line = await asyncio.to_thread(input, f"{user_name}: ")
            except EOFError:
                break
            text = line.strip()
            if not sys.stdin.isatty():
                print(text)  # echo piped input so transcripts read naturally
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break
            await handler.handle_message(
                IncomingMessage(
                    message_id=next(message_ids),
                    channel=TERMINAL_CHANNEL,
                    author_id=TERMINAL_USER_ID,
                    author_name=user_name,
                    content=text,
                    created_at=utcnow(),
                )
            )
            await handler.drain()
        await handler.close()
    finally:
        await registry.close()
        await database.close()
    return EXIT_OK
