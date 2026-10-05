"""The conversation pipeline for one bot.

:class:`ConversationHandler` ties the core together:

1. **Observe** every message in channels the bot may use: store it in
   short-term memory and keep track of channel activity.
2. **Decide** with the :class:`DecisionEngine` whether to respond and update
   conversational focus. Rate limits count replies actually sent and are
   re-checked just before generating, so queued triggers can't exceed them.
3. **Queue** the message on its channel's session. Each channel has at most
   one worker task, so the bot never talks over itself.
4. **Read** like a person: notice the new messages, then take as long as a
   human needs to read them. Follow-ups that arrive meanwhile are read too and
   answered together. No typing indicator is shown while reading.
5. **Respond**: the typing indicator starts exactly when the model starts
   generating. The reply is sent once a human typing at the personality's speed
   (60 wpm by default) would have finished, or as soon as generation finishes if
   that takes longer. Further parts of a split reply are typed the same way.

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

from arcane.ai.prompts import PromptContext, history_cost
from arcane.ai.providers.base import ProviderError
from arcane.ai.response_manager import GeneratedReply, ResponseManager
from arcane.conversation.decision import Decision, DecisionEngine
from arcane.conversation.initiative import InitiativeLog, InitiativePlanner
from arcane.conversation.rate_limit import ReplyRateLimiter
from arcane.conversation.state import ConversationTracker
from arcane.conversation.timing import HumanTiming
from arcane.conversation.transport import MessageTransport, SentMessage, TransportError
from arcane.core.clock import utcnow
from arcane.core.models import ChannelInfo, HistoryMessage, IncomingMessage
from arcane.memory.extraction import MemoryConsolidator
from arcane.memory.long_term import LongTermMemory
from arcane.memory.short_term import ShortTermMemory
from arcane.personalities.base import Personality

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
Sleep = Callable[[float], Awaitable[None]]

INITIATIVE_HISTORY_MESSAGES = 12
HISTORY_LOW_WATER = 0.65
"""When history exceeds its character budget, cut it down to this fraction of
the budget in one jump, so the prompt prefix stays stable for several turns."""


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
    read: bool = False
    """Whether the bot has already spent "reading" time on this message."""


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
                await self._read(session)
                batch = self._take_batch(session)
                if not batch:
                    continue
                async with session.lock:
                    await self._respond(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[%s] response pipeline failed", self.bot_id)
            session.pending.clear()

    async def _read(self, session: _ChannelSession) -> None:
        """Spend human reading time on unread queued messages.

        Starts with a short reaction delay, then the reading time of every unread
        message. Messages that arrive while reading extend the phase by their own
        reading time, so a burst is read in full and answered once. The whole
        phase is capped by the profile's ``max_reading_seconds``.
        """
        unread = [t for t in session.pending if not t.read]
        if not unread:
            return
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self._timing.reaction_delay()
        cap = started + self._timing.max_reading_seconds()
        while True:
            for trigger in session.pending:
                if not trigger.read:
                    trigger.read = True
                    deadline += self._timing.reading_time(trigger.message.text_for_prompt())
            session.wake.clear()
            remaining = min(deadline, cap) - loop.time()
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(session.wake.wait(), remaining)
            except TimeoutError:
                return

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

    async def _respond(self, batch: list[_Trigger]) -> None:
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

        context = await self._reply_context(target, incoming_text)
        reply = await self._generate_and_deliver(target.channel, context, reply_target=target)
        if reply is None:
            return

        now = self._clock()
        self._rate_limiter.record(target.channel_id, target.author_id, now)
        self._tracker.note_bot_message(target.channel_id, now, guild_id=target.channel.guild_id)
        await self._persist_conversation(target.channel_id)
        logger.info(
            "[%s] %s: replied to %s with %d message(s) (model %s, generated in %.1fs, "
            "prompt %s tokens, %s cached, attempt %d)",
            self.bot_id,
            target.channel.display_name,
            target.author_name,
            len(reply.parts),
            reply.model,
            reply.duration_seconds,
            reply.prompt_tokens if reply.prompt_tokens is not None else "?",
            reply.cached_prompt_tokens if reply.cached_prompt_tokens is not None else "?",
            reply.attempts,
        )

    async def _reply_context(self, target: IncomingMessage, incoming_text: str) -> PromptContext:
        memory = self._personality.memory
        history = self._anchored_history(
            target.channel_id,
            await self._short_term.recent_messages(target.channel_id, memory.max_history_messages),
            window=memory.history_messages,
            slack=memory.history_slack,
            char_budget=memory.history_char_budget,
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
