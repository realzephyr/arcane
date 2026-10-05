"""The conversation pipeline for one bot.

:class:`ConversationHandler` ties the core together:

1. **Observe** every message in channels the bot may use: store it in
   short-term memory and keep track of channel activity.
2. **Decide** with the :class:`DecisionEngine` whether to respond and update
   conversational focus. Rate limits count replies actually sent and are
   re-checked just before generating, so queued triggers can't exceed them.
3. **Queue** the message on its channel's session. Each channel has at most
   one worker task, so the bot never talks over itself, and bursts of messages
   are debounced into a single reply.
4. **Respond** like a person: wait to "read" and "think", show the typing
   indicator while the model generates, keep typing for as long as a human
   would need, send, and type each further part of a split reply.

It also exposes :meth:`run_maintenance` (expiry, long-term memory
consolidation, pruning) and :meth:`maybe_initiate` (starting conversations),
which the bot runtime schedules periodically.

The handler is platform-agnostic: it acts through a :class:`MessageTransport`.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from arcane.ai.prompts import PromptContext
from arcane.ai.providers.base import ProviderError
from arcane.ai.response_manager import GeneratedReply, ResponseManager
from arcane.conversation.decision import Decision, DecisionEngine
from arcane.conversation.initiative import InitiativeLog, InitiativePlanner
from arcane.conversation.rate_limit import ReplyRateLimiter
from arcane.conversation.state import ConversationTracker
from arcane.conversation.timing import HumanTiming
from arcane.conversation.transport import MessageTransport, SentMessage, TransportError
from arcane.core.clock import utcnow
from arcane.core.models import ChannelInfo, IncomingMessage
from arcane.memory.extraction import MemoryConsolidator
from arcane.memory.long_term import LongTermMemory
from arcane.memory.short_term import ShortTermMemory
from arcane.personalities.base import Personality

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]

INITIATIVE_HISTORY_MESSAGES = 12


@dataclass(frozen=True, slots=True)
class HandlerConfig:
    """Deployment-level switches for one bot (from settings, not the personality)."""

    allowed_channel_ids: frozenset[int] = frozenset()
    """Channels the bot may use; empty means all."""
    initiative_channel_ids: tuple[int, ...] = ()
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
        self._sequence = itertools.count()
        self._stale_after = timedelta(seconds=personality.behavior.stale_trigger_seconds)
        self._closed = False

    @property
    def bot_id(self) -> str:
        return self._short_term.bot_id

    @property
    def long_term_active(self) -> bool:
        return self._config.long_term_memory_enabled and self._personality.memory.long_term_enabled

    @property
    def initiative_active(self) -> bool:
        return (
            self._config.initiative_enabled
            and self._personality.behavior.initiative.enabled
            and bool(self._config.initiative_channel_ids)
        )

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
        try:
            await self._short_term.delete_message(message_id)
        except Exception:
            logger.exception("[%s] failed to forget message %d", self.bot_id, message_id)

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
        if session.task is None or session.task.done():
            session.task = asyncio.create_task(
                self._run_session(session), name=f"arcane:{self.bot_id}:{session.channel_id}"
            )

    async def _run_session(self, session: _ChannelSession) -> None:
        try:
            while session.pending and not self._closed:
                waited = await self._debounce(session)
                batch = self._take_batch(session)
                if not batch:
                    continue
                async with session.lock:
                    await self._respond(batch, already_waited=waited)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] response pipeline failed", self.bot_id)
            session.pending.clear()

    async def _debounce(self, session: _ChannelSession) -> float:
        """Wait until the channel has been quiet briefly. Returns seconds waited."""
        window = self._timing.debounce_window()
        if window <= 0:
            return 0.0
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self._timing.max_debounce()
        while True:
            session.wake.clear()
            timeout = min(window, deadline - loop.time())
            if timeout <= 0:
                break
            try:
                await asyncio.wait_for(session.wake.wait(), timeout)
            except TimeoutError:
                break
        return loop.time() - started

    def _take_batch(self, session: _ChannelSession) -> list[_Trigger]:
        """Pick whose messages to answer next and remove them from the queue.

        Highest priority wins; ties prefer the conversation partner, then the
        oldest message. All queued messages from that author form one batch.
        """
        now = self._clock()
        fresh = [t for t in session.pending if now - t.message.created_at <= self._stale_after]
        if len(fresh) < len(session.pending):
            logger.debug(
                "[%s] dropped %d stale trigger(s)", self.bot_id, len(session.pending) - len(fresh)
            )
        if not fresh:
            session.pending = []
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

    # --------------------------------------------------------------- responding

    async def _respond(self, batch: list[_Trigger], *, already_waited: float) -> None:
        target = batch[-1].message
        incoming_text = "\n".join(t.message.text_for_prompt() for t in batch)

        # Limits count replies, not triggers. Re-check now: other replies may have
        # gone out since this message was accepted.
        if not self._rate_limiter.allows(target.channel_id, target.author_id, self._clock()):
            logger.info(
                "[%s] %s: skipping reply to %s (rate limited)",
                self.bot_id,
                target.channel.display_name,
                target.author_name,
            )
            return

        lead = self._timing.response_lead_time(incoming_text) - already_waited
        if lead > 0:
            await self._sleep(lead)

        context = await self._reply_context(target, incoming_text)
        reply = await self._generate_and_deliver(target.channel, context, reply_target=target)
        if reply is None:
            return

        now = self._clock()
        self._rate_limiter.record(target.channel_id, target.author_id, now)
        self._tracker.note_bot_message(target.channel_id, now, guild_id=target.channel.guild_id)
        await self._persist_conversation(target.channel_id)
        logger.info(
            "[%s] %s: replied to %s with %d message(s) (model %s, %.1fs, attempt %d)",
            self.bot_id,
            target.channel.display_name,
            target.author_name,
            len(reply.parts),
            reply.model,
            reply.duration_seconds,
            reply.attempts,
        )

    async def _reply_context(self, target: IncomingMessage, incoming_text: str) -> PromptContext:
        memory = self._personality.memory
        history = await self._short_term.recent_messages(target.channel_id, memory.history_messages)
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

    async def _generate_and_deliver(
        self,
        channel: ChannelInfo,
        context: PromptContext,
        *,
        reply_target: IncomingMessage | None = None,
    ) -> GeneratedReply | None:
        """Generate while "typing", then send each part with human pacing."""
        loop = asyncio.get_running_loop()
        async with self._transport.typing(channel.channel_id):
            started = loop.time()
            try:
                reply = await self._responses.generate(context)
            except ProviderError as exc:
                logger.warning("[%s] could not generate a reply: %s", self.bot_id, exc)
                return None
            if reply is None:
                logger.info("[%s] the model produced no usable reply", self.bot_id)
                return None

            first, *rest = reply.parts
            remaining = self._timing.typing_duration(first) - (loop.time() - started)
            if remaining > 0:
                await self._sleep(remaining)
            reference = (
                reply_target.message_id
                if reply_target is not None and self._should_reference(reply_target)
                else None
            )
            if not await self._send(channel, first, reply_to=reference):
                return None

        for part in rest:
            await self._sleep(self._timing.pause_between_messages())
            async with self._transport.typing(channel.channel_id):
                await self._sleep(self._timing.typing_duration(part))
                if not await self._send(channel, part):
                    break
        return reply

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

    async def maybe_initiate(self) -> bool:
        """Start a conversation in one opted-in channel if the planner agrees."""
        if not self.initiative_active or self._closed:
            return False
        channels = list(self._config.initiative_channel_ids)
        self._rng.shuffle(channels)
        now = self._clock()

        for channel_id in channels:
            session = self._sessions.get(channel_id)
            if session is not None and session.busy:
                continue
            activity = await self._short_term.channel_activity(channel_id)
            decision = self._planner.evaluate(
                now=now,
                activity=activity,
                conversation_active=self._tracker.is_active(channel_id, now),
                initiations_today=await self._initiatives.count_since(
                    channel_id, now - timedelta(days=1)
                ),
                last_initiation_at=await self._initiatives.last_at(channel_id),
            )
            if not decision.should_initiate:
                logger.debug(
                    "[%s] not starting a conversation in %d: %s",
                    self.bot_id,
                    channel_id,
                    decision.reason,
                )
                continue
            quiet_for = now - activity.last_message_at if activity.last_message_at else None
            return await self.initiate(channel_id, quiet_for=quiet_for)
        return False

    async def initiate(
        self,
        channel_id: int,
        *,
        topic: str | None = None,
        quiet_for: timedelta | None = None,
    ) -> bool:
        """Post a conversation opener in ``channel_id``. Returns True if sent."""
        channel = await self._transport.channel_info(channel_id)
        if channel is None:
            logger.warning("[%s] initiative channel %d is not accessible", self.bot_id, channel_id)
            return False
        if topic is None:
            topic = self._planner.choose_topic(
                self._personality.conversation_topics,
                await self._initiatives.recent_topics(),
            )
        if topic is None:
            return False

        session = self._session(channel_id)
        async with session.lock:
            history = await self._short_term.recent_messages(
                channel_id, INITIATIVE_HISTORY_MESSAGES
            )
            context = PromptContext(
                personality=self._personality,
                channel=channel,
                history=history,
                now=self._clock(),
                mode="initiate",
                topic=topic,
                quiet_for=quiet_for,
            )
            reply = await self._generate_and_deliver(channel, context)
            if reply is None:
                return False
            now = self._clock()
            self._rate_limiter.record(channel_id, None, now)
            await self._initiatives.record(channel_id, topic, now)
            self._tracker.note_bot_message(
                channel_id, now, guild_id=channel.guild_id, initiated=True
            )
            await self._persist_conversation(channel_id)

        logger.info(
            "[%s] started a conversation in %s about %s", self.bot_id, channel.display_name, topic
        )
        return True

    # -------------------------------------------------------------- maintenance

    async def run_maintenance(self) -> MaintenanceReport:
        """Expire conversations, consolidate memories, and prune storage."""
        now = self._clock()
        expired = self._tracker.expire(now)
        stored = 0
        for state in expired:
            await self._short_term.delete_conversation(state.channel_id)
            if self._consolidator is not None and self.long_term_active and state.participants:
                try:
                    stored += await self._consolidator.consolidate(state.to_record(self.bot_id))
                except Exception:
                    logger.exception("[%s] memory consolidation failed", self.bot_id)

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
