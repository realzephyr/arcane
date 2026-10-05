"""Wiring for one bot's conversation stack.

:func:`build_conversation_handler` assembles everything a personality needs
(memory, decision engine, focus tracking, rate limiting, timing, generation,
initiative) from settings and returns a ready
:class:`~arcane.conversation.handler.ConversationHandler`. It is platform-agnostic,
so the Discord runtime, the terminal chat, and tests all share the same wiring.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterable
from datetime import timedelta

from arcane.ai.providers.base import LLMProvider
from arcane.ai.response_manager import ResponseManager
from arcane.config.settings import Settings
from arcane.conversation.decision import DecisionEngine
from arcane.conversation.handler import Clock, ConversationHandler, HandlerConfig, Sleep
from arcane.conversation.initiative import InitiativeLog, InitiativePlanner
from arcane.conversation.rate_limit import ReplyRateLimiter
from arcane.conversation.state import ConversationTracker
from arcane.conversation.timing import HumanTiming
from arcane.conversation.transport import MessageTransport
from arcane.core.clock import utcnow
from arcane.database.database import Database
from arcane.memory.extraction import (
    LLMMemoryExtractor,
    MemoryConsolidator,
    MemoryExtractor,
    NullMemoryExtractor,
)
from arcane.memory.long_term import LongTermMemory
from arcane.memory.short_term import ShortTermMemory
from arcane.personalities.base import Personality


def build_conversation_handler(
    *,
    personality: Personality,
    settings: Settings,
    database: Database,
    provider: LLMProvider,
    transport: MessageTransport,
    model: str | None = None,
    allowed_channel_ids: Iterable[int] = (),
    initiative_channel_ids: Iterable[int] = (),
    clock: Clock = utcnow,
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> ConversationHandler:
    """Assemble the conversation stack for ``personality``."""
    rng = rng or random.Random()
    bot_id = personality.id
    behavior = personality.behavior
    timeout = timedelta(
        seconds=settings.conversation_timeout_seconds or behavior.conversation_timeout_seconds
    )

    short_term = ShortTermMemory(
        database,
        bot_id,
        ttl=timedelta(hours=settings.short_term_ttl_hours),
        max_messages_per_channel=settings.max_messages_per_channel,
    )
    long_term = LongTermMemory(
        database, bot_id, max_memories_per_user=personality.memory.max_memories_per_user
    )

    model_name = model or personality.model.model
    long_term_on = settings.long_term_memory_enabled and personality.memory.long_term_enabled
    extractor: MemoryExtractor = (
        LLMMemoryExtractor(provider, bot_name=personality.name, model=model_name)
        if long_term_on
        else NullMemoryExtractor()
    )

    return ConversationHandler(
        personality=personality,
        transport=transport,
        short_term=short_term,
        long_term=long_term,
        tracker=ConversationTracker(
            bot_id,
            timeout=timeout,
            focus_timeout=timedelta(seconds=behavior.focus_timeout_seconds),
            opener_window=timedelta(seconds=behavior.opener_reply_window_seconds),
        ),
        decision_engine=DecisionEngine(
            personality,
            respond_in_dms=settings.respond_in_dms,
            conversation_timeout=timeout,
            rng=rng,
        ),
        rate_limiter=ReplyRateLimiter(
            per_channel=behavior.max_replies_per_channel_per_minute,
            per_user=behavior.max_replies_per_user_per_minute,
        ),
        timing=HumanTiming(personality.timing, enabled=settings.humanize, rng=rng),
        response_manager=ResponseManager(personality, provider, model=model_name),
        initiative_planner=InitiativePlanner(behavior.initiative, rng=rng),
        initiative_log=InitiativeLog(database, bot_id),
        consolidator=MemoryConsolidator(
            short_term,
            long_term,
            extractor,
            min_user_messages=personality.memory.min_messages_for_extraction,
        ),
        config=HandlerConfig(
            allowed_channel_ids=frozenset(allowed_channel_ids),
            initiative_channel_ids=tuple(initiative_channel_ids),
            initiative_enabled=settings.initiative_enabled,
            respond_in_dms=settings.respond_in_dms,
            long_term_memory_enabled=settings.long_term_memory_enabled,
        ),
        clock=clock,
        sleep=sleep,
        rng=rng,
    )
