"""The decision engine: should the bot answer this message?

:class:`DecisionEngine` is deterministic given its inputs (randomness comes
from an injected ``random.Random``), has no I/O, and returns a
:class:`Decision` with a machine-readable reason that is logged for every
message. Rules, first match wins:

1. Ignore its own messages, other bots (unless enabled), and empty messages.
2. Direct messages: respond (if DMs are enabled).
3. An @mention, or a reply to one of its messages: respond.
4. Its name in the text: respond with high probability (always, when it comes
   from the person it's talking with).
5. Active conversation with a partner:
   * the partner keeps talking, not obviously to someone else: respond;
   * a participant talks after the partner went quiet: respond;
   * anyone else: ignore, keeping focus on the partner.
6. Someone answers a message the bot chimed in with (when it replied to a
   person, only that person's answer counts): respond.
7. A substantive message matching its interests in a quiet moment: small chance.
8. Otherwise ignore.

Every "respond" is then checked against the rate limiter.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum

from arcane.conversation.rate_limit import ReplyRateLimiter
from arcane.conversation.state import ConversationState
from arcane.core.models import IncomingMessage
from arcane.personalities.base import Personality


class Action(StrEnum):
    RESPOND = "respond"
    IGNORE = "ignore"


class Reason(StrEnum):
    OWN_MESSAGE = "own_message"
    BOT_AUTHOR = "bot_author"
    EMPTY = "empty"
    DMS_DISABLED = "dms_disabled"
    DIRECT_MESSAGE = "direct_message"
    MENTION = "mention"
    REPLY_TO_BOT = "reply_to_bot"
    NAME_MENTIONED = "name_mentioned"
    NAME_MENTIONED_SKIPPED = "name_mentioned_skipped"
    CONTINUATION = "continuation"
    ADDRESSED_ELSEWHERE = "addressed_elsewhere"
    FOCUSED_ON_OTHER_USER = "focused_on_other_user"
    OPENER_REPLY = "opener_reply"
    INTEREST = "interest"
    NOT_ADDRESSED = "not_addressed"
    RATE_LIMITED = "rate_limited"


class Priority(IntEnum):
    """How strongly a message calls for an answer; used to order queued triggers."""

    NONE = 0
    SPONTANEOUS = 1
    CONTINUATION = 2
    NAMED = 3
    DIRECT = 4


@dataclass(frozen=True, slots=True)
class Decision:
    action: Action
    reason: Reason
    priority: Priority = Priority.NONE

    @property
    def should_respond(self) -> bool:
        return self.action is Action.RESPOND

    @classmethod
    def respond(cls, reason: Reason, priority: Priority) -> Decision:
        return cls(Action.RESPOND, reason, priority)

    @classmethod
    def ignore(cls, reason: Reason) -> Decision:
        return cls(Action.IGNORE, reason)


class DecisionEngine:
    """Decides whether a personality responds to an incoming message."""

    def __init__(
        self,
        personality: Personality,
        *,
        respond_in_dms: bool = True,
        conversation_timeout: timedelta | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._personality = personality
        behavior = personality.behavior
        self._behavior = behavior
        self._respond_in_dms = respond_in_dms
        self._timeout = conversation_timeout or timedelta(
            seconds=behavior.conversation_timeout_seconds
        )
        self._focus_timeout = timedelta(seconds=behavior.focus_timeout_seconds)
        self._opener_window = timedelta(seconds=behavior.opener_reply_window_seconds)
        self._spontaneous_cooldown = timedelta(seconds=behavior.spontaneous_cooldown_seconds)
        self._rng = rng or random.Random()

    def decide(
        self,
        message: IncomingMessage,
        conversation: ConversationState | None,
        *,
        now: datetime,
        rate_limiter: ReplyRateLimiter | None = None,
        last_bot_message_at: datetime | None = None,
    ) -> Decision:
        if message.is_self:
            return Decision.ignore(Reason.OWN_MESSAGE)
        if message.author_is_bot and not self._behavior.respond_to_bots:
            return Decision.ignore(Reason.BOT_AUTHOR)
        if not message.has_content:
            return Decision.ignore(Reason.EMPTY)

        decision = self._classify(message, conversation, now, last_bot_message_at)
        if (
            decision.should_respond
            and rate_limiter is not None
            and not rate_limiter.allows(message.channel_id, message.author_id, now)
        ):
            return Decision.ignore(Reason.RATE_LIMITED)
        return decision

    # ---------------------------------------------------------------- internals

    def _classify(
        self,
        message: IncomingMessage,
        conversation: ConversationState | None,
        now: datetime,
        last_bot_message_at: datetime | None,
    ) -> Decision:
        if message.is_dm:
            if self._respond_in_dms:
                return Decision.respond(Reason.DIRECT_MESSAGE, Priority.DIRECT)
            return Decision.ignore(Reason.DMS_DISABLED)
        if message.mentions_bot:
            return Decision.respond(Reason.MENTION, Priority.DIRECT)
        if message.is_reply_to_self:
            return Decision.respond(Reason.REPLY_TO_BOT, Priority.DIRECT)

        addressed_elsewhere = self._addressed_elsewhere(message)
        in_conversation = (
            conversation is not None
            and conversation.partner_id is not None
            and conversation.is_active(now, self._timeout)
        )
        if not addressed_elsewhere and self._personality.is_named_in(message.content):
            is_partner = (
                in_conversation
                and conversation is not None
                and message.author_id == conversation.partner_id
            )
            if is_partner or self._rng.random() < self._behavior.name_mention_reply_chance:
                return Decision.respond(Reason.NAME_MENTIONED, Priority.NAMED)
            return Decision.ignore(Reason.NAME_MENTIONED_SKIPPED)

        if in_conversation and conversation is not None:
            focus_decision = self._within_conversation(
                message, conversation, now, addressed_elsewhere
            )
            if focus_decision is not None:
                return focus_decision

        if (
            conversation is not None
            and conversation.awaits_reply_from(message.author_id, now, self._opener_window)
            and not addressed_elsewhere
        ):
            return Decision.respond(Reason.OPENER_REPLY, Priority.CONTINUATION)

        if not addressed_elsewhere and self._wants_to_join(message, now, last_bot_message_at):
            return Decision.respond(Reason.INTEREST, Priority.SPONTANEOUS)
        return Decision.ignore(Reason.NOT_ADDRESSED)

    def _within_conversation(
        self,
        message: IncomingMessage,
        conversation: ConversationState,
        now: datetime,
        addressed_elsewhere: bool,
    ) -> Decision | None:
        if message.author_id == conversation.partner_id:
            if addressed_elsewhere:
                return Decision.ignore(Reason.ADDRESSED_ELSEWHERE)
            return Decision.respond(Reason.CONTINUATION, Priority.CONTINUATION)

        partner_focused = conversation.partner_is_focused(now, self._focus_timeout)
        if (
            not partner_focused
            and message.author_id in conversation.participants
            and not addressed_elsewhere
        ):
            return Decision.respond(Reason.CONTINUATION, Priority.CONTINUATION)
        if partner_focused:
            return Decision.ignore(Reason.FOCUSED_ON_OTHER_USER)
        return None  # partner went quiet; treat like an idle channel

    @staticmethod
    def _addressed_elsewhere(message: IncomingMessage) -> bool:
        """The message is clearly meant for someone other than the bot.

        That is a Discord reply to another person's message, or a message that
        opens with an @mention of someone else. Replying to one's own message,
        or mentioning someone in passing ("i told @bob"), does not count, so a
        conversation partner who keeps typing normally is still followed.
        """
        if message.mentions_bot:
            return False
        reply = message.reply_to
        if reply is not None and not reply.is_self and reply.author_id != message.author_id:
            return True
        return bool(message.addressed_user_ids)

    def _wants_to_join(
        self,
        message: IncomingMessage,
        now: datetime,
        last_bot_message_at: datetime | None,
    ) -> bool:
        behavior = self._behavior
        if behavior.spontaneous_reply_chance <= 0:
            return False
        if len(message.content.split()) < behavior.spontaneous_min_words:
            return False
        if (
            last_bot_message_at is not None
            and now - last_bot_message_at < self._spontaneous_cooldown
        ):
            return False
        if self._personality.interest_hits(message.content) < behavior.spontaneous_min_keyword_hits:
            return False
        return self._rng.random() < behavior.spontaneous_reply_chance
