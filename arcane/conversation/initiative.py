"""Chiming into chat on its own.

When a personality isn't talking with anyone, it does what a regular in a
server does: it looks at the channel where people were talking most recently
and joins in, either by replying to one of their recent messages that's worth
discussing or by dropping a message of its own about one of its conversation
topics.

:class:`InitiativePlanner` makes those decisions and is free of I/O:

* :meth:`~InitiativePlanner.evaluate` checks whether the bot may chime in at
  all right now (switch, active hours, idle, global interval);
* :meth:`~InitiativePlanner.evaluate_channel` checks one candidate channel
  (someone active recently, the bot didn't speak last, cooldown, daily cap);
* :meth:`~InitiativePlanner.choose_reply_target` picks the recent message most
  worth answering, if any;
* :meth:`~InitiativePlanner.choose_topic` picks a fresh topic for a message of
  its own.

Chime-ins are logged in the database (:class:`InitiativeLog`) so cooldowns and
caps survive restarts and topics don't repeat.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from arcane.ai.guards import detect_impossible_request
from arcane.core.clock import from_timestamp, to_timestamp, utcnow
from arcane.core.models import HistoryMessage
from arcane.database.database import Database
from arcane.memory.short_term import ChannelActivity
from arcane.personalities.base import InitiativeProfile, Personality

INITIATIVE_RETENTION = timedelta(days=14)
REPLY_TOPIC_PREFIX = "reply:"
"""Log entries for chime-ins that replied to a message are ``reply:<message id>``."""

MIN_REPLY_SCORE = 2.0
"""How discussable a message must be (see :func:`discussion_score`) to reply to it."""
MAX_BOT_SHARE = 0.25
"""Don't chime in where bots already wrote more than this share of recent messages."""
MAX_COOLDOWN_DOUBLINGS = 3
SKIPPED_CHANNEL_RE = re.compile(
    r"vent|support|serious|rule|announce|staff|mod|admin|log|ticket|report|welcome|verify",
    re.IGNORECASE,
)
"""Channel names where unprompted chatter is unwelcome; never chimed into."""

_STRONG_HOOK_RE = re.compile(
    r"\b(?:debate|argue|argument|moral(?:ly|s|ity)?|ethic(?:s|al)?|free\s+will|god|"
    r"conscious(?:ness)?|philosoph\w*|hot\s+take|unpopular\s+opinion|would\s+(?:you|u)\s+"
    r"rather|wyr|meaning\s+of\s+life|is\s+it\s+(?:ever\s+)?(?:ok|okay|wrong|right)|"
    r"right\s+or\s+wrong|overrated|underrated|goat)\b",
    re.IGNORECASE,
)
"""Signals a message is up for debate on its own."""
_WEAK_HOOK_RE = re.compile(
    r"\b(?:think|thoughts|opinion|believe|agree|disagree|should|shouldn'?t|better|worse|"
    r"best|worst|why|prove|evidence|fair|deserve|exist|purpose|soul|truth|smarter)\b",
    re.IGNORECASE,
)
"""Everyday words that make a topical message more discussable, never enough alone."""
_DISTRESS_RE = re.compile(
    r"\b(?:depress\w*|anxiety|kms|suicid\w*|self\s*harm|died|passed\s+away|funeral|"
    r"break\s*up|broke\s+up|my\s+life|crying|panic|hospital|abuse\w*|lonely)\b",
    re.IGNORECASE,
)
"""Someone going through something isn't an opening for a debate."""
_COMMAND_RE = re.compile(r"^\s*[!/.$?;>%]")
_URL_RE = re.compile(r"https?://\S+")
_ATTACHMENT_RE = re.compile(r"\[(?:image|video|audio|file|sticker)[^\]]*\]", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    should_initiate: bool
    reason: str


def discussion_score(personality: Personality, message: HistoryMessage, now: datetime) -> float:
    """How good ``message`` is to jump in on: topical, debatable, a question, fresh.

    Returns ``0`` for messages that aren't worth answering at all: commands,
    bare links or attachments, very short messages, messages aimed at someone
    (a leading @mention or a Discord reply to another person), someone venting,
    requests the bot can't fulfil, and anything without a real topical signal
    (one of its interests or a debate term).
    """
    text = _ATTACHMENT_RE.sub(" ", _URL_RE.sub(" ", message.content)).strip()
    words = text.split()
    if _COMMAND_RE.match(message.content) or text.startswith("@"):
        return 0.0
    if message.reply_to_author_name and message.reply_to_author_name.lower() not in (
        personality.names
    ):
        return 0.0
    if len(words) < 3 or _DISTRESS_RE.search(text):
        return 0.0
    if detect_impossible_request(text, names=personality.names) is not None:
        return 0.0
    interests = min(personality.interest_hits(text), 3)
    strong = min(len(_STRONG_HOOK_RE.findall(text)), 2)
    if not interests and not strong:
        return 0.0
    score = 1.5 * interests + 1.5 * strong
    score += 0.5 * min(len(_WEAK_HOOK_RE.findall(text)), 2)
    score += 1.0 if "?" in text else 0.0
    score += min(len(words) / 12, 1.0)
    age_minutes = max((now - message.created_at).total_seconds(), 0.0) / 60
    return max(score - 0.3 * age_minutes, 0.0)


class InitiativePlanner:
    """Decides when and where to chime in, and with what."""

    def __init__(self, profile: InitiativeProfile, *, rng: random.Random | None = None) -> None:
        self._profile = profile
        self._rng = rng or random.Random()

    @property
    def profile(self) -> InitiativeProfile:
        return self._profile

    @property
    def check_interval(self) -> timedelta:
        return timedelta(seconds=self._profile.check_interval_seconds)

    @property
    def idle_after(self) -> timedelta:
        return timedelta(seconds=self._profile.idle_seconds)

    @property
    def active_window(self) -> timedelta:
        return timedelta(seconds=self._profile.active_window_seconds)

    def evaluate(
        self,
        *,
        now: datetime,
        engaged: bool,
        last_initiation_at: datetime | None,
    ) -> InitiativeDecision:
        """May the bot chime in anywhere right now?"""
        profile = self._profile
        if not profile.enabled:
            return InitiativeDecision(False, "disabled")
        if not self._within_active_hours(now):
            return InitiativeDecision(False, "outside active hours")
        if engaged:
            return InitiativeDecision(False, "already talking with someone")
        if last_initiation_at is not None and now - last_initiation_at < timedelta(
            minutes=profile.min_interval_minutes
        ):
            return InitiativeDecision(False, "chimed in somewhere recently")
        return InitiativeDecision(True, "idle")

    def evaluate_channel(
        self,
        *,
        now: datetime,
        activity: ChannelActivity,
        initiations_today: int,
        last_initiation_at: datetime | None,
        unanswered: int = 0,
        bot_share: float = 0.0,
    ) -> InitiativeDecision:
        """Is this channel a good place to chime in?

        Args:
            unanswered: Chime-ins in a row nobody answered; each one doubles
                the channel cooldown (up to 8x).
            bot_share: Share of the channel's recent messages written by bots.
        """
        profile = self._profile
        if activity.last_human_message_at is None or (
            now - activity.last_human_message_at > self.active_window
        ):
            return InitiativeDecision(False, "nobody active")
        if activity.last_message_is_self:
            return InitiativeDecision(False, "last message was ours")
        if activity.last_message_at != activity.last_human_message_at:
            return InitiativeDecision(False, "a bot spoke last")
        if bot_share > MAX_BOT_SHARE:
            return InitiativeDecision(False, "bots already talk a lot here")
        if initiations_today >= profile.max_per_channel_per_day:
            return InitiativeDecision(False, "daily limit reached")
        cooldown = timedelta(minutes=profile.channel_cooldown_minutes) * (
            2 ** min(unanswered, MAX_COOLDOWN_DOUBLINGS)
        )
        if last_initiation_at is not None and now - last_initiation_at < cooldown:
            return InitiativeDecision(False, "chimed in here recently")
        return InitiativeDecision(True, "people are talking")

    def roll(self) -> bool:
        """The final random chance, so chime-ins don't run like clockwork."""
        return self._rng.random() < self._profile.chance

    def choose_reply_target(
        self,
        personality: Personality,
        messages: Sequence[HistoryMessage],
        now: datetime,
    ) -> HistoryMessage | None:
        """The recent message most worth answering, or ``None`` to post a message instead.

        Only people's messages younger than ``reply_max_age_seconds`` and newer
        than the bot's own last message count. Even with a good candidate, the
        bot replies only ``reply_chance`` of the time.
        """
        max_age = timedelta(seconds=self._profile.reply_max_age_seconds)
        last_own = max((m.created_at for m in messages if m.is_self), default=None)
        candidates = [
            message
            for message in messages
            if not message.is_self
            and not message.author_is_bot
            and now - message.created_at <= max_age
            and (last_own is None or message.created_at > last_own)
        ]
        scored = [(discussion_score(personality, m, now), m) for m in candidates]
        best = max(scored, key=lambda pair: pair[0], default=None)
        if best is None or best[0] < MIN_REPLY_SCORE:
            return None
        if self._rng.random() >= self._profile.reply_chance:
            return None
        return best[1]

    def choose_topic(self, topics: Sequence[str], recent: Sequence[str]) -> str | None:
        """Pick a topic, avoiding recently used ones while alternatives remain."""
        if not topics:
            return None
        recent_set = set(recent)
        fresh = [topic for topic in topics if topic not in recent_set]
        return self._rng.choice(fresh or list(topics))

    def _within_active_hours(self, now: datetime) -> bool:
        window = self._profile.active_hours_utc
        if window is None:
            return True
        start, end = window
        hour = now.hour
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end  # window wraps past midnight


class InitiativeLog:
    """Persistent record of the times a bot chimed in on its own."""

    def __init__(self, database: Database, bot_id: str) -> None:
        self._db = database
        self.bot_id = bot_id

    async def record(self, channel_id: int, topic: str, now: datetime | None = None) -> None:
        await self._db.execute(
            "INSERT INTO initiatives (bot_id, channel_id, topic, created_at) VALUES (?, ?, ?, ?)",
            (self.bot_id, channel_id, topic, to_timestamp(now or utcnow())),
        )

    async def count_since(self, channel_id: int, since: datetime) -> int:
        row = await self._db.fetch_one(
            "SELECT COUNT(*) FROM initiatives WHERE bot_id = ? AND channel_id = ? "
            "AND created_at >= ?",
            (self.bot_id, channel_id, to_timestamp(since)),
        )
        return int(row[0]) if row is not None else 0

    async def last_at(self, channel_id: int | None = None) -> datetime | None:
        """When it last chimed in, in ``channel_id`` or (``None``) anywhere."""
        if channel_id is None:
            row = await self._db.fetch_one(
                "SELECT MAX(created_at) FROM initiatives WHERE bot_id = ?", (self.bot_id,)
            )
        else:
            row = await self._db.fetch_one(
                "SELECT MAX(created_at) FROM initiatives WHERE bot_id = ? AND channel_id = ?",
                (self.bot_id, channel_id),
            )
        return from_timestamp(row[0]) if row is not None and row[0] is not None else None

    async def recent_topics(self, limit: int = 10) -> list[str]:
        """Topics of its latest messages of its own (replies to messages excluded)."""
        rows = await self._db.fetch_all(
            "SELECT topic FROM initiatives WHERE bot_id = ? AND topic NOT LIKE ? "
            "ORDER BY created_at DESC LIMIT ?",
            (self.bot_id, f"{REPLY_TOPIC_PREFIX}%", limit),
        )
        return [row["topic"] for row in rows]

    async def prune(self, now: datetime | None = None) -> int:
        cutoff = (now or utcnow()) - INITIATIVE_RETENTION
        removed = await self._db.execute(
            "DELETE FROM initiatives WHERE bot_id = ? AND created_at < ?",
            (self.bot_id, to_timestamp(cutoff)),
        )
        return max(removed, 0)
