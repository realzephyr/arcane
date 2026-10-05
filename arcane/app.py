"""Application composition root and lifecycle.

:class:`Application` owns everything shared (database, provider registry) and
one :class:`BotRuntime` per configured personality. It runs all bots
concurrently on one event loop, keeps the remaining bots running if one fails
to log in, and shuts everything down cleanly on SIGINT/SIGTERM.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import signal
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import discord

from arcane.ai.providers.factory import ProviderRegistry
from arcane.bot.bot import ArcaneBot
from arcane.bot.events import EventRouter
from arcane.bot.transport import DiscordTransport
from arcane.config.settings import BotConfig, Settings
from arcane.conversation.handler import ConversationHandler
from arcane.core.tasks import PeriodicTask
from arcane.database.database import Database
from arcane.personalities.base import Personality
from arcane.personalities.registry import load_personality
from arcane.runtime import build_conversation_handler

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURE = 1

_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@dataclass
class BotRuntime:
    """Everything that belongs to one running bot."""

    config: BotConfig
    personality: Personality
    client: ArcaneBot
    handler: ConversationHandler
    tasks: list[PeriodicTask] = field(default_factory=list)


class Application:
    """Runs every configured bot until stopped."""

    def __init__(self, settings: Settings, bot_configs: list[BotConfig]) -> None:
        if not bot_configs:
            raise ValueError("at least one bot must be configured")
        self._settings = settings
        self._bot_configs = bot_configs
        self._database = Database(settings.database_path)
        self._providers = ProviderRegistry(settings)
        self._runtimes: list[BotRuntime] = []
        self._stop = asyncio.Event()
        self._warm_ups: list[Callable[[], Coroutine[Any, Any, bool]]] = []

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> int:
        """Run until stopped. Returns a process exit code."""
        try:
            await self._database.connect()
            self._runtimes = [self._build_runtime(config) for config in self._bot_configs]
            self._install_signal_handlers()
            await self._preflight()
            return await self._run_bots()
        finally:
            await self._shutdown()

    # ------------------------------------------------------------------ wiring

    def _build_runtime(self, config: BotConfig) -> BotRuntime:
        personality = load_personality(config.personality_id)
        provider = self._providers.get(personality.model.provider)
        client = ArcaneBot(personality)
        handler = build_conversation_handler(
            personality=personality,
            settings=self._settings,
            database=self._database,
            provider=provider,
            transport=DiscordTransport(client),
            model=config.model,
            allowed_channel_ids=config.allowed_channel_ids,
            initiative_channel_ids=config.initiative_channel_ids,
        )
        tasks = [
            PeriodicTask(
                f"{personality.id}-maintenance",
                handler.run_maintenance,
                interval=self._settings.maintenance_interval_seconds,
                jitter=0.1,
                run_immediately=True,
            )
        ]
        if handler.initiative_active:
            tasks.append(
                PeriodicTask(
                    f"{personality.id}-initiative",
                    handler.maybe_initiate,
                    interval=personality.behavior.initiative.check_interval_seconds,
                    jitter=0.25,
                )
            )
        client.attach(EventRouter(client, handler, background_tasks=tasks))
        logger.info(
            "[%s] configured: model=%s, channels=%s, initiative=%s",
            personality.id,
            config.model or personality.model.model or provider.default_model,
            ", ".join(map(str, config.allowed_channel_ids)) or "all",
            (", ".join(map(str, config.initiative_channel_ids)) or "any")
            if handler.initiative_active
            else "off",
        )
        if handler.initiative_active and not config.initiative_channel_ids:
            logger.info(
                "[%s] may chime into any channel it can talk in when idle; set "
                "%sINITIATIVE_CHANNEL_IDS (or ARCANE_INITIATIVE_CHANNEL_IDS) to limit where, "
                "or ARCANE_INITIATIVE_ENABLED=false to turn it off",
                personality.id,
                BotConfig.env_prefix(personality.id),
            )
        return BotRuntime(config, personality, client, handler, tasks)

    async def _preflight(self) -> None:
        """Warn early if the AI backend or a model is unavailable (not fatal)."""
        checked: set[tuple[str, str]] = set()
        for runtime in self._runtimes:
            provider = self._providers.get(runtime.personality.model.provider)
            model = (
                runtime.config.model or runtime.personality.model.model or provider.default_model
            )
            if (provider.name, model) in checked:
                continue
            checked.add((provider.name, model))
            health = await provider.health_check(model)
            if not health.available:
                logger.warning(
                    "AI backend '%s' is unavailable (%s). Bots will stay silent until it is "
                    "reachable.",
                    provider.name,
                    health.detail,
                )
            elif health.model_available is False:
                logger.warning("%s: %s", provider.name, health.detail)
            else:
                logger.info("%s: %s", provider.name, health.detail)
                # Load the model with the same num_ctx replies will use, so the first
                # message isn't slowed down by a model load. It runs in the background
                # while the bots log in: a slow load must not keep them offline.
                self._warm_ups.append(
                    functools.partial(
                        provider.warm_up,
                        model=model,
                        options=runtime.handler.response_manager.base_options(),
                    )
                )

    # ----------------------------------------------------------------- running

    async def _run_bots(self) -> int:
        bot_tasks = {
            asyncio.create_task(self._run_bot(runtime), name=f"bot-{runtime.personality.id}")
            for runtime in self._runtimes
        }
        stop_task = asyncio.create_task(self._stop.wait(), name="stop-signal")
        warm_ups = [
            asyncio.create_task(warm_up(), name="model-warm-up") for warm_up in self._warm_ups
        ]
        failures = 0
        try:
            pending = set(bot_tasks)
            while pending:
                done, _ = await asyncio.wait(
                    pending | {stop_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if stop_task in done:
                    logger.info("Shutdown requested")
                    return EXIT_OK
                for task in done:
                    pending.discard(task)
                    if not task.cancelled() and task.exception() is not None:
                        failures += 1
                if pending:
                    logger.warning("%d bot(s) still running", len(pending))
            return EXIT_FAILURE if failures else EXIT_OK
        finally:
            stop_task.cancel()
            for running in (*bot_tasks, *warm_ups):
                running.cancel()
            await asyncio.gather(stop_task, *bot_tasks, *warm_ups, return_exceptions=True)

    async def _run_bot(self, runtime: BotRuntime) -> None:
        bot_id = runtime.personality.id
        try:
            await runtime.client.start(runtime.config.token.get_secret_value())
        except discord.LoginFailure:
            logger.error(
                "[%s] Discord rejected the token. Check %sTOKEN.",
                bot_id,
                BotConfig.env_prefix(bot_id),
            )
            raise
        except discord.PrivilegedIntentsRequired:
            logger.error(
                "[%s] Message Content Intent is not enabled. Enable it in the Discord "
                "Developer Portal under Bot -> Privileged Gateway Intents.",
                bot_id,
            )
            raise
        except asyncio.CancelledError:
            raise
        except (discord.HTTPException, aiohttp.ClientError, OSError) as exc:
            logger.error("[%s] could not connect to Discord: %s", bot_id, exc)
            raise
        except Exception:
            logger.exception("[%s] stopped unexpectedly", bot_id)
            raise

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, self.request_stop)

    def _remove_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in _STOP_SIGNALS:
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.remove_signal_handler(sig)

    async def _shutdown(self) -> None:
        self._remove_signal_handlers()
        for runtime in self._runtimes:
            try:
                await runtime.client.close()
            except Exception:
                logger.exception("[%s] error during shutdown", runtime.personality.id)
        await self._providers.close()
        await self._database.close()
        logger.info("Arcane stopped")
