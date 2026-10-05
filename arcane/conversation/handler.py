"""The conversation pipeline for one bot.

:class:`ConversationHandler` ties the core together:

1. **Observe** every message in channels the bot may use: store it in
   short-term memory and keep track of channel activity.
2. **Decide** with the :class:`DecisionEngine` whether to respond and update
   conversational focus. Rate limits count replies actually sent and are
   re-checked just before generating, so queued triggers can't exceed them.
3. **Queue** the message on its channel's session. Each channel has at most
   one worker task, so the bot never talks over itself.
4. **Read and think** like a person: notice the new messages, then read them
   while the model writes the reply in the background. Nothing is shown in the
   channel yet. Follow-ups from the same person that arrive meanwhile are read
   too: the draft is thrown away and the reply is regenerated for everything
   they said.
5. **Type and send**: only once the reply is ready does the typing indicator
   appear, for as long as typing that reply takes (short replies are quick,
   long ones take longer). Then the message is sent. Further parts of a split
   reply are typed the same way.

It also exposes :meth:`run_maintenance` (expiry, long-term memory
consolidation, pruning) and :meth:`maybe_initiate` (chiming into the most
recently active channel when idle), which the bot runtime schedules
periodically.

The handler is platform-agnostic: it acts through a :class:`MessageTransport`.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, TypeVar

from arcane.ai.prompts import PromptContext, history_budget, history_cost
from arcane.ai.providers.base import ProviderError
from arcane.ai.response_manager import GeneratedReply, ResponseManager
from arcane.conversation.decision import Decision, DecisionEngine
from arcane.conversation.initiative import (
    REPLY_TOPIC_PREFIX,
    SKIPPED_CHANNEL_RE,
    InitiativeLog,
    InitiativePlanner,
)
from arcane.conversation.rate_limit import ReplyRateLimiter
from arcane.conversation.state import NOBODY, ConversationState, ConversationTracker
from arcane.conversation.timing import HumanTiming
from arcane.conversation.transport import MessageTransport, SentMessage, TransportError
from arcane.core.clock import utcnow
from arcane.core.models import ChannelInfo, HistoryMessage, IncomingMessage
from arcane.memory.extraction import MemoryConsolidator
from arcane.memory.long_term import LongTermMemory
from arcane.memory.short_term import ShortTermMemory
from arcane.personalities.base import Personality

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]

MAX_REGENERATIONS = 3
"""How often a draft may be thrown away because the person kept typing."""
FOLLOW_UP_WINDOW_SECONDS = 20.0
"""Follow-ups from the same person are folded into the reply until it starts being
typed, for at most this long (and only when humanised timing is on)."""
MAX_STALE_CHIME_IN_MESSAGES = 2
"""Drop a chime-in if more people's messages than this arrived while writing it."""
CONSOLIDATION_PATIENCE = timedelta(minutes=30)
"""Memory extraction gives way to live replies, but never waits longer than this."""
CONTEXT_WARNING_RATIO = 0.95
BOT_SHARE_SAMPLE = 20
"""Recent messages checked for how much of a channel is bots talking."""
HISTORY_LOW_WATER = 0.65
"""When history exceeds its character budget, cut it down to this fraction of
the budget in one jump, so the prompt prefix stays stable for several turns."""


@dataclass(frozen=True, slots=True)
class HandlerConfig:
    """Deployment-level switches for one bot (from settings, not the personality)."""

    allowed_channel_ids: frozenset[int] = frozenset()
    """Channels the bot may use; empty means all."""
    initiative_channel_ids: tuple[int, ...] = ()
    """Channels it may chime into on its own; empty means any channel it may use."""
    initiative_enabled: bool = True
    respond_in_dms: bool = True
    long_term_memory_enabled: bool = True


@dataclass(frozen=True, slots=True)
class MaintenanceReport:
    expired_conversations: int = 0
    memories_stored: int = 0
    messages_pruned: int = 0
    memories_pruned: int = 0


@dataclass(slots=True)
class _Trigger:
    message: IncomingMessage
    decision: Decision
    sequence: int


@dataclass(slots=True)
class _ChannelSession:
    channel_id: int
    pending: list[_Trigger] = field(default_factory=list)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None

    @property
    def busy(self) -> bool:
        running = self.task is not None and not self.task.done()
        return running or bool(self.pending) or self.lock.locked()


class ConversationHandler:
    """Observes messages, decides, and responds for one personality."""

    def __init__(
        self,
        *,
        personality: Personality,
        transport: MessageTransport,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        tracker: ConversationTracker,
        decision_engine: DecisionEngine,
        rate_limiter: ReplyRateLimiter,
        timing: HumanTiming,
        response_manager: ResponseManager,
        initiative_planner: InitiativePlanner,
        initiative_log: InitiativeLog,
        consolidator: MemoryConsolidator | None = None,
        config: HandlerConfig | None = None,
        clock: Clock = utcnow,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._personality = personality
        self._transport = transport
        self._short_term = short_term
        self._long_term = long_term
        self._tracker = tracker
        self._engine = decision_engine
        self._rate_limiter = rate_limiter
        self._timing = timing
        self._responses = response_manager
        self._planner = initiative_planner
        self._initiatives = initiative_log
        self._consolidator = consolidator
        self._config = config or HandlerConfig()
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()

        self._sessions: dict[int, _ChannelSession] = {}
        self._last_message_id: dict[int, int] = {}
        self._history_anchor: dict[int, int] = {}
        self._sequence = itertools.count()
        self._stale_after = timedelta(seconds=personality.behavior.stale_trigger_seconds)
        self._closed = False
        self._interrupts: set[asyncio.Event] = set()
        """Background model work in progress; set when someone needs an answer."""
        self._people_messages: dict[int, int] = {}
        """Per channel: how many people's messages the bot has seen."""
        self._author_seen_at: dict[tuple[int, int], int] = {}
        """Per (channel, author): the channel's message count at their latest message."""
        self._unanswered: dict[int, int] = {}
        """Per channel: chime-ins in a row nobody answered."""
        self._to_consolidate: list[tuple[ConversationState, datetime]] = []

    @property
    def bot_id(self) -> str:
        return self._short_term.bot_id

    @property
    def response_manager(self) -> ResponseManager:
        return self._responses

    @property
    def long_term_active(self) -> bool:
        return self._config.long_term_memory_enabled and self._personality.memory.long_term_enabled

    @property
    def initiative_active(self) -> bool:
        return self._config.initiative_enabled and self._personality.behavior.initiative.enabled

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Restore conversations that were active before a restart."""
        records = await self._short_term.load_conversations()
        restored = self._tracker.restore(records, self._clock())
        if restored:
            logger.info("[%s] restored %d active conversation(s)", self.bot_id, restored)

    async def drain(self) -> None:
        """Wait until every queued response has been handled."""
        while True:
            tasks = [s.task for s in self._sessions.values() if s.task and not s.task.done()]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        """Cancel in-flight responses. Safe to call more than once."""
        self._closed = True
        tasks = [s.task for s in self._sessions.values() if s.task and not s.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ inbound

    async def handle_message(self, message: IncomingMessage) -> Decision | None:
        """Process one observed message. Returns the decision, if one was made."""
        if self._closed or not self._observes(message.channel):
            return None

        self._last_message_id[message.channel_id] = message.message_id
        try:
            await self._short_term.record_message(message)
        except Exception:
            logger.exception("[%s] failed to store message %d", self.bot_id, message.message_id)
        if message.is_self:
            return None
        self._note_person_message(message)

        now = self._clock()
        decision = self._engine.decide(
            message,
            self._tracker.get(message.channel_id),
            now=now,
            rate_limiter=self._rate_limiter,
            last_bot_message_at=self._tracker.last_bot_message_at(message.channel_id),
        )
        if not decision.should_respond:
            self._tracker.note_partner_activity(message, now)
            logger.debug(
                "[%s] %s: ignoring %s (%s)",
                self.bot_id,
                message.channel.display_name,
                message.author_name,
                decision.reason,
            )
            return decision

        self._unanswered.pop(message.channel_id, None)
        logger.info(
            "[%s] %s: responding to %s (%s)",
            self.bot_id,
            message.channel.display_name,
            message.author_name,
            decision.reason,
        )
        self._tracker.note_engagement(message, now)
        await self._persist_conversation(message.channel_id)
        await self._touch_user(message)
        self._enqueue(message, decision)
        return decision

    async def handle_edit(self, message_id: int, content: str) -> None:
        try:
            await self._short_term.update_message_content(message_id, content)
        except Exception:
            logger.exception("[%s] failed to apply edit to message %d", self.bot_id, message_id)

    async def handle_delete(self, message_id: int) -> None:
        for session in self._sessions.values():
            session.pending = [t for t in session.pending if t.message.message_id != message_id]
        try:
            await self._short_term.delete_message(message_id)
        except Exception:
            logger.exception("[%s] failed to forget message %d", self.bot_id, message_id)

    def _note_person_message(self, message: IncomingMessage) -> None:
        count = self._people_messages.get(message.channel_id, 0) + 1
        self._people_messages[message.channel_id] = count
        if len(self._author_seen_at) > 10_000:
            self._author_seen_at.clear()
        self._author_seen_at[(message.channel_id, message.author_id)] = count

    def _observes(self, channel: ChannelInfo) -> bool:
        if channel.is_dm:
            return self._config.respond_in_dms
        allowed = self._config.allowed_channel_ids
        return (
            not allowed
            or channel.channel_id in allowed
            or (channel.parent_id is not None and channel.parent_id in allowed)
        )

    # ----------------------------------------------------------------- sessions

    def _session(self, channel_id: int) -> _ChannelSession:
        session = self._sessions.get(channel_id)
        if session is None:
            session = self._sessions[channel_id] = _ChannelSession(channel_id)
        return session

    def _enqueue(self, message: IncomingMessage, decision: Decision) -> None:
        session = self._session(message.channel_id)
        session.pending.append(_Trigger(message, decision, next(self._sequence)))
        session.wake.set()
        for interrupt in self._interrupts:
            interrupt.set()
        if session.task is None or session.task.done():
            session.task = asyncio.create_task(
                self._run_session(session), name=f"arcane:{self.bot_id}:{session.channel_id}"
            )

    async def _run_session(self, session: _ChannelSession) -> None:
        try:
            while session.pending and not self._closed:
                batch = self._take_batch(session)
                if not batch:
                    continue
                async with session.lock:
                    await self._respond(session, batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] response pipeline failed", self.bot_id)
            session.pending.clear()

    def _take_batch(self, session: _ChannelSession) -> list[_Trigger]:
        """Pick whose messages to answer next and remove them from the queue.

        Highest priority wins; ties prefer the conversation partner, then the
        oldest message. All queued messages from that author form one batch.
        """
        fresh = self._drop_stale(session)
        if not fresh:
            return []
        state = self._tracker.get(session.channel_id)
        partner = state.partner_id if state is not None else None
        best = max(
            fresh,
            key=lambda t: (t.decision.priority, t.message.author_id == partner, -t.sequence),
        )
        author = best.message.author_id
        session.pending = [t for t in fresh if t.message.author_id != author]
        return [t for t in fresh if t.message.author_id == author]

    def _take_follow_ups(self, session: _ChannelSession, author_id: int) -> list[_Trigger]:
        """Remove and return queued messages from ``author_id``."""
        fresh = self._drop_stale(session)
        follow_ups = [t for t in fresh if t.message.author_id == author_id]
        session.pending = [t for t in fresh if t.message.author_id != author_id]
        return follow_ups

    def _drop_stale(self, session: _ChannelSession) -> list[_Trigger]:
        now = self._clock()
        fresh = [t for t in session.pending if now - t.message.created_at <= self._stale_after]
        if len(fresh) < len(session.pending):
            logger.debug(
                "[%s] dropped %d stale trigger(s)", self.bot_id, len(session.pending) - len(fresh)
            )
        session.pending = fresh
        return fresh

    # --------------------------------------------------------------- responding

    async def _respond(self, session: _ChannelSession, batch: list[_Trigger]) -> None:
        target = batch[-1].message
        channel = target.channel

        # Limits count replies, not triggers. Re-check now, and hold the slot
        # while working on the reply so other channels can't overshoot meanwhile.
        reserved_at = self._clock()
        if not self._rate_limiter.allows(target.channel_id, target.author_id, reserved_at):
            logger.info(
                "[%s] %s: skipping reply to %s (rate limited)",
                self.bot_id,
                channel.display_name,
                target.author_name,
            )
            return
        self._rate_limiter.record(target.channel_id, target.author_id, reserved_at)

        loop = asyncio.get_running_loop()
        started = loop.time()
        sent = 0
        try:
            outcome = await self._read_and_think(session, batch)
            if outcome is None:
                return
            reply, batch = outcome
            target = batch[-1].message
            thought = loop.time() - started
            reference = target.message_id if self._should_reference(target) else None
            sent = await self._deliver(channel, reply.parts, reply_to=reference)
        finally:
            if not sent:
                self._rate_limiter.release(target.channel_id, target.author_id, reserved_at)
        if not sent:
            return
        typed = loop.time() - started - thought
        self._warn_if_context_is_full(reply)

        now = self._clock()
        self._tracker.note_bot_message(
            target.channel_id, now, guild_id=channel.guild_id, replied_to=target.author_id
        )
        await self._persist_conversation(target.channel_id)
        logger.info(
            "[%s] %s: replied to %s with %d message(s): read and generated in %.1fs "
            "(model %s took %.1fs, prompt %s tokens, %s cached, attempt %d), typed for %.1fs",
            self.bot_id,
            channel.display_name,
            target.author_name,
            sent,
            thought,
            reply.model,
            reply.duration_seconds,
            reply.prompt_tokens if reply.prompt_tokens is not None else "?",
            reply.cached_prompt_tokens if reply.cached_prompt_tokens is not None else "?",
            reply.attempts,
            typed,
        )

    async def _read_and_think(
        self, session: _ChannelSession, batch: list[_Trigger]
    ) -> tuple[GeneratedReply, list[_Trigger]] | None:
        """Notice and read the batch while the model drafts a reply; nothing shows yet.

        Generation starts at once, so the model works during the reaction and
        reading time instead of after it. The phase ends when the draft is ready
        and the messages have had time to be noticed and read (plus a short wait
        after a bare attention-getter like "yo"). If the same person sends more
        before typing starts, the draft is discarded and a new one is generated
        for everything they said, so a burst gets one answer.
        """
        loop = asyncio.get_running_loop()
        started = loop.time()
        absorb_until = started + (FOLLOW_UP_WINDOW_SECONDS if self._timing.enabled else 0.0)
        author = batch[0].message.author_id
        ready_at = started + self._timing.reaction_delay() + self._reading_time(batch)
        regenerations = 0

        while True:
            target = batch[-1].message
            context = await self._reply_context(target, _incoming_text(batch))
            draft = asyncio.create_task(self._responses.generate(context))
            try:
                while True:
                    session.wake.clear()
                    now = loop.time()
                    if regenerations < MAX_REGENERATIONS and now < absorb_until:
                        follow_ups = self._take_follow_ups(session, author)
                        if follow_ups:
                            batch = [*batch, *follow_ups]
                            regenerations += 1
                            ready_at = max(ready_at, now + self._reading_time(follow_ups))
                            break
                    if draft.done() and now >= ready_at:
                        return self._finish_draft(draft, batch)
                    waiting = asyncio.ensure_future(session.wake.wait())
                    try:
                        await asyncio.wait(
                            {waiting} if draft.done() else {waiting, draft},
                            timeout=max(ready_at - now, 0) if draft.done() else None,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    finally:
                        waiting.cancel()
            finally:
                if not draft.done():
                    draft.cancel()
                    # Never raises; a cancellation of this task itself still propagates.
                    await asyncio.wait({draft})
            if self._closed:
                return None
            logger.debug(
                "[%s] %s kept typing; regenerating for %d message(s)",
                self.bot_id,
                target.author_name,
                len(batch),
            )

    def _finish_draft(
        self, draft: asyncio.Task[GeneratedReply | None], batch: list[_Trigger]
    ) -> tuple[GeneratedReply, list[_Trigger]] | None:
        try:
            reply = draft.result()
        except ProviderError as exc:
            logger.warning("[%s] could not generate a reply: %s", self.bot_id, exc)
            return None
        if reply is None:
            logger.info("[%s] the model produced no usable reply", self.bot_id)
            return None
        return reply, batch

    def _reading_time(self, triggers: list[_Trigger]) -> float:
        """Reading time for ``triggers``, capped, plus a wait after a bare "yo"."""
        text = _incoming_text(triggers)
        reading = min(self._timing.reading_time(text), self._timing.max_reading_seconds())
        return reading + self._timing.follow_up_wait(text)

    def _warn_if_context_is_full(self, reply: GeneratedReply) -> None:
        model = self._personality.model
        if reply.prompt_tokens is None or model.context_window is None:
            return
        if reply.prompt_tokens + model.max_tokens > CONTEXT_WARNING_RATIO * model.context_window:
            logger.warning(
                "[%s] the prompt (%d tokens) nearly fills the context window (%d). Older "
                "messages get cut off and the prompt cache stops working, which slows "
                "replies down. Raise context_window or lower history_char_budget.",
                self.bot_id,
                reply.prompt_tokens,
                model.context_window,
            )

    async def _yielding(self, work: Coroutine[Any, Any, _T]) -> asyncio.Task[_T] | None:
        """Run background model work, but give way to anyone who needs an answer.

        Returns the finished task, or ``None`` if a message that needs a reply
        arrived first (the work is cancelled so the model is free for it).
        """
        interrupted = asyncio.Event()
        self._interrupts.add(interrupted)
        task = asyncio.create_task(work)
        waiting = asyncio.ensure_future(interrupted.wait())
        try:
            await asyncio.wait({task, waiting}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiting.cancel()
            self._interrupts.discard(interrupted)
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
        return None if task.cancelled() else task

    async def _reply_context(self, target: IncomingMessage, incoming_text: str) -> PromptContext:
        memory = self._personality.memory
        history = self._anchored_history(
            target.channel_id,
            await self._short_term.recent_messages(target.channel_id, memory.max_history_messages),
            window=memory.history_messages,
            slack=memory.history_slack,
            char_budget=history_budget(self._personality, target.channel),
        )
        state = self._tracker.get(target.channel_id)
        profile = await self._long_term.get_profile(target.author_id)
        memories = (
            await self._long_term.recall(target.author_id, limit=memory.recall_limit)
            if self.long_term_active
            else []
        )
        return PromptContext(
            personality=self._personality,
            channel=target.channel,
            history=history,
            now=self._clock(),
            mode="reply",
            target_user_id=target.author_id,
            target_user_name=target.author_name,
            target_message=incoming_text,
            profile=profile,
            memories=memories,
            partner_name=state.partner_name if state is not None else None,
            other_participants=state.other_participant_names() if state is not None else (),
        )

    def _anchored_history(
        self,
        channel_id: int,
        history: list[HistoryMessage],
        *,
        window: int,
        slack: int,
        char_budget: int,
    ) -> list[HistoryMessage]:
        """Pick the prompt history so its first message changes as rarely as possible.

        Local model servers (Ollama) reuse their cache for an unchanged prompt
        prefix. A window that slides by one message per turn changes the prefix
        every time and forces the whole history to be re-read. Instead, the
        window keeps its first message while it grows up to ``window + slack``
        messages, then jumps forward to the newest ``window`` messages. The
        character budget works the same way: when exceeded, the window jumps
        forward until it is back under ``HISTORY_LOW_WATER`` of the budget.
        """
        if not history:
            self._history_anchor.pop(channel_id, None)
            return history
        ids = [message.message_id for message in history]
        anchor = self._history_anchor.get(channel_id)
        start = ids.index(anchor) if anchor in ids else None
        if start is None or len(history) - start > window + slack:
            start = max(len(history) - window, 0)
        if history_cost(history[start:]) > char_budget:
            low_water = int(char_budget * HISTORY_LOW_WATER)
            while start < len(history) - 1 and history_cost(history[start:]) > low_water:
                start += 1
        self._history_anchor[channel_id] = ids[start]
        return history[start:]

    async def _generate(self, context: PromptContext) -> GeneratedReply | None:
        """Generate silently (no typing indicator while the model works)."""
        try:
            reply = await self._responses.generate(context)
        except ProviderError as exc:
            logger.warning("[%s] could not generate a message: %s", self.bot_id, exc)
            return None
        if reply is None:
            logger.info("[%s] the model produced no usable message", self.bot_id)
        return reply

    async def _deliver(
        self,
        channel: ChannelInfo,
        parts: tuple[str, ...],
        *,
        reply_to: int | None = None,
    ) -> int:
        """Type and send each part. Returns how many messages were sent.

        The typing indicator shows for as long as typing that part takes, then
        the part is sent. ``reply_to`` makes the first part a Discord reply.
        """
        loop = asyncio.get_running_loop()
        sent = 0
        for index, part in enumerate(parts):
            if index:
                await self._sleep(self._timing.pause_between_messages())
            duration = self._timing.typing_duration(part)
            started = loop.time()
            async with self._transport.typing(channel.channel_id):
                # Showing the indicator is a network round trip; it counts as typing.
                remaining = duration - (loop.time() - started)
                if remaining > 0:
                    await self._sleep(remaining)
                if not await self._send(channel, part, reply_to=reply_to if index == 0 else None):
                    break
            sent += 1
        return sent

    def _should_reference(self, target: IncomingMessage) -> bool:
        """Use Discord's reply feature only when other messages came in between."""
        if target.is_dm:
            return False
        return self._last_message_id.get(target.channel_id) != target.message_id

    async def _send(
        self, channel: ChannelInfo, content: str, *, reply_to: int | None = None
    ) -> bool:
        try:
            sent = await self._transport.send(
                channel.channel_id, content, reply_to_message_id=reply_to
            )
        except TransportError as exc:
            logger.warning("[%s] could not send to %s: %s", self.bot_id, channel.display_name, exc)
            return False
        await self._record_own_message(channel, sent, content)
        return True

    async def _record_own_message(self, channel: ChannelInfo, sent: SentMessage, text: str) -> None:
        """Store our message immediately; the gateway echo will upsert the same row."""
        self._last_message_id[channel.channel_id] = sent.message_id
        own = IncomingMessage(
            message_id=sent.message_id,
            channel=channel,
            author_id=sent.author_id,
            author_name=sent.author_name,
            content=text,
            created_at=sent.created_at,
            author_is_bot=True,
            is_self=True,
        )
        try:
            await self._short_term.record_message(own)
        except Exception:
            logger.exception("[%s] failed to store own message", self.bot_id)

    # --------------------------------------------------------------- initiative

    def engaged(self) -> bool:
        """True while the bot is talking with someone or working on a reply."""
        if any(session.busy for session in self._sessions.values()):
            return True
        return self._tracker.engaged(self._clock(), self._planner.idle_after)

    async def maybe_initiate(self) -> bool:
        """Chime into the most recently active channel if the planner agrees.

        Returns True if a message was sent.
        """
        if not self.initiative_active or self._closed:
            return False
        now = self._clock()
        decision = self._planner.evaluate(
            now=now,
            engaged=self.engaged(),
            last_initiation_at=await self._initiatives.last_at(),
        )
        if not decision.should_initiate:
            logger.debug("[%s] not chiming in: %s", self.bot_id, decision.reason)
            return False

        for channel_id, _ in await self._short_term.active_channels(
            now - self._planner.active_window
        ):
            channel = await self._chime_in_channel(channel_id)
            if channel is None:
                continue
            recent = await self._short_term.recent_messages(channel_id, BOT_SHARE_SAMPLE)
            bots = sum(1 for m in recent if m.is_self or m.author_is_bot)
            verdict = self._planner.evaluate_channel(
                now=now,
                activity=await self._short_term.channel_activity(channel_id),
                initiations_today=await self._initiatives.count_since(
                    channel_id, now - timedelta(days=1)
                ),
                last_initiation_at=await self._initiatives.last_at(channel_id),
                unanswered=self._unanswered.get(channel_id, 0),
                bot_share=bots / len(recent) if recent else 0.0,
            )
            if not verdict.should_initiate:
                logger.debug(
                    "[%s] not chiming into %s: %s",
                    self.bot_id,
                    channel.display_name,
                    verdict.reason,
                )
                continue
            # The most recently active eligible channel is the only candidate.
            if not self._planner.roll():
                return False
            return await self.initiate(channel_id)
        return False

    async def _chime_in_channel(self, channel_id: int) -> ChannelInfo | None:
        """The channel, if the bot may chime in there right now."""
        restricted = self._config.initiative_channel_ids
        session = self._sessions.get(channel_id)
        if session is not None and session.busy:
            return None
        channel = await self._transport.channel_info(channel_id)
        if channel is None or channel.is_dm or not self._observes(channel):
            return None
        if restricted:
            if not (
                channel.channel_id in restricted
                or (channel.parent_id is not None and channel.parent_id in restricted)
            ):
                return None
        elif channel.name and SKIPPED_CHANNEL_RE.search(channel.name):
            return None
        return channel

    async def initiate(self, channel_id: int, *, topic: str | None = None) -> bool:
        """Chime into ``channel_id``: reply to a recent message or post one of its own.

        With ``topic``, always posts a message of its own about it. The attempt
        counts towards cooldowns even if it fails. It is dropped if someone
        needs a real answer meanwhile, or if the chat moved on while writing.
        Returns True if something was sent.
        """
        channel = await self._transport.channel_info(channel_id)
        if channel is None:
            logger.warning("[%s] channel %d is not accessible", self.bot_id, channel_id)
            return False
        now = self._clock()
        if not self._rate_limiter.allows(channel_id, None, now):
            return False

        session = self._session(channel_id)
        async with session.lock:
            memory = self._personality.memory
            history = self._anchored_history(
                channel_id,
                await self._short_term.recent_messages(channel_id, memory.max_history_messages),
                window=memory.history_messages,
                slack=memory.history_slack,
                char_budget=history_budget(self._personality, channel),
            )
            target = (
                self._planner.choose_reply_target(self._personality, history, now)
                if topic is None
                else None
            )
            if target is None and topic is None:
                topic = self._planner.choose_topic(
                    self._personality.conversation_topics,
                    await self._initiatives.recent_topics(),
                )
                if topic is None:
                    return False
            logged_topic = f"{REPLY_TOPIC_PREFIX}{target.message_id}" if target else str(topic)
            await self._initiatives.record(channel_id, logged_topic, now)

            last = history[-1].created_at if history else None
            context = PromptContext(
                personality=self._personality,
                channel=channel,
                history=history,
                now=now,
                mode="join" if target is not None else "initiate",
                target_user_id=target.author_id if target is not None else None,
                target_user_name=target.author_name if target is not None else None,
                target_message=target.content if target is not None else None,
                topic=topic,
                quiet_for=now - last if last is not None else None,
            )
            seen_before = self._people_messages.get(channel_id, 0)
            draft = await self._yielding(self._responses.generate(context))
            if draft is None:
                logger.info(
                    "[%s] %s: dropped a chime-in, someone needs an answer",
                    self.bot_id,
                    channel.display_name,
                )
                return False
            try:
                reply = draft.result()
            except ProviderError as exc:
                logger.warning("[%s] could not generate a message: %s", self.bot_id, exc)
                return False
            if reply is None:
                logger.info("[%s] the model produced no usable message", self.bot_id)
                return False
            if self._chat_moved_on(channel_id, seen_before, target):
                logger.info(
                    "[%s] %s: dropped a chime-in, the chat moved on",
                    self.bot_id,
                    channel.display_name,
                )
                return False

            sent = await self._deliver(
                channel, reply.parts, reply_to=target.message_id if target is not None else None
            )
            if not sent:
                return False
            now = self._clock()
            self._rate_limiter.record(channel_id, None, now)
            self._unanswered[channel_id] = self._unanswered.get(channel_id, 0) + 1
            self._tracker.note_bot_message(
                channel_id,
                now,
                guild_id=channel.guild_id,
                initiated=True,
                awaiting_from=self._expected_answerer(history, target, now),
            )
            await self._persist_conversation(channel_id)

        if target is not None:
            logger.info(
                "[%s] %s: chimed in on %s's message",
                self.bot_id,
                channel.display_name,
                target.author_name,
            )
        else:
            logger.info("[%s] %s: chimed in about %s", self.bot_id, channel.display_name, topic)
        return True

    def _chat_moved_on(
        self, channel_id: int, seen_before: int, target: HistoryMessage | None
    ) -> bool:
        seen_now = self._people_messages.get(channel_id, 0)
        if seen_now - seen_before > MAX_STALE_CHIME_IN_MESSAGES:
            return True
        if target is None:
            return False
        return self._author_seen_at.get((channel_id, target.author_id), 0) > seen_before

    def _expected_answerer(
        self, history: list[HistoryMessage], target: HistoryMessage | None, now: datetime
    ) -> int | None:
        """Who may answer a chime-in with a plain message (see ``awaiting_from``).

        The person it replied to; or, after a message of its own, the only
        person active in the channel. In a busy channel nobody's next plain
        message counts (``0``): only replies to the bot, mentions or its name do.
        """
        if target is not None:
            return target.author_id
        window = self._planner.active_window
        authors = {
            m.author_id
            for m in history
            if not m.is_self and not m.author_is_bot and now - m.created_at <= window
        }
        return authors.pop() if len(authors) == 1 else NOBODY

    # -------------------------------------------------------------- maintenance

    async def run_maintenance(self) -> MaintenanceReport:
        """Expire conversations, consolidate memories, and prune storage."""
        now = self._clock()
        expired = self._tracker.expire(now)
        for state in expired:
            if self._tracker.get(state.channel_id) is None:
                await self._short_term.delete_conversation(state.channel_id)
            if self._consolidator is not None and self.long_term_active and state.participants:
                self._to_consolidate.append((state, now))
        stored = await self._consolidate_memories(now)

        pruned = await self._short_term.prune(now)
        memories_pruned = await self._long_term.prune(now)
        await self._initiatives.prune(now)
        self._rate_limiter.cleanup(now)
        for channel_id in [cid for cid, s in self._sessions.items() if not s.busy]:
            del self._sessions[channel_id]

        report = MaintenanceReport(
            expired_conversations=len(expired),
            memories_stored=stored,
            messages_pruned=pruned.total,
            memories_pruned=memories_pruned,
        )
        if expired or stored or pruned.total or memories_pruned:
            logger.info(
                "[%s] maintenance: %d conversation(s) ended, %d memory(ies) stored, "
                "%d message(s) and %d memory(ies) pruned",
                self.bot_id,
                report.expired_conversations,
                report.memories_stored,
                report.messages_pruned,
                report.memories_pruned,
            )
        return report

    async def _consolidate_memories(self, now: datetime) -> int:
        """Extract long-term memories from finished conversations.

        Extraction uses the same model as replies, so it waits while the bot is
        talking with someone and is cancelled (and retried later) when a
        message needs an answer. Conversations waiting longer than
        ``CONSOLIDATION_PATIENCE`` are processed regardless.
        """
        stored = 0
        consolidator = self._consolidator
        while self._to_consolidate and consolidator is not None and not self._closed:
            state, queued_at = self._to_consolidate[0]
            overdue = now - queued_at >= CONSOLIDATION_PATIENCE
            record = state.to_record(self.bot_id)
            task: asyncio.Task[int] | None
            if overdue:
                task = asyncio.create_task(consolidator.consolidate(record))
                await asyncio.wait({task})
            elif self.engaged():
                break
            else:
                task = await self._yielding(consolidator.consolidate(record))
            if task is None:
                break  # someone needs an answer; try again at the next maintenance run
            self._to_consolidate.pop(0)
            try:
                stored += task.result()
            except Exception:
                logger.exception("[%s] memory consolidation failed", self.bot_id)
        return stored

    # ------------------------------------------------------------------ helpers

    async def _persist_conversation(self, channel_id: int) -> None:
        state = self._tracker.get(channel_id)
        if state is None:
            return
        try:
            await self._short_term.save_conversation(state.to_record(self.bot_id))
        except Exception:
            logger.exception("[%s] failed to persist conversation state", self.bot_id)

    async def _touch_user(self, message: IncomingMessage) -> None:
        try:
            await self._long_term.touch_user(
                message.author_id, message.author_name, interaction=True, now=self._clock()
            )
        except Exception:
            logger.exception("[%s] failed to update user profile", self.bot_id)


def _incoming_text(batch: list[_Trigger]) -> str:
    return "\n".join(trigger.message.text_for_prompt() for trigger in batch)
