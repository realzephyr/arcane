"""Tests for the conversation core: state, decisions, rate limits, timing, initiative."""

from __future__ import annotations

import random
import statistics
from datetime import timedelta

import pytest

from arcane.conversation.decision import DecisionEngine, Priority, Reason
from arcane.conversation.initiative import InitiativeLog, InitiativePlanner
from arcane.conversation.rate_limit import ReplyRateLimiter, SlidingWindowCounter
from arcane.conversation.state import ConversationState, ConversationTracker
from arcane.conversation.timing import HumanTiming
from arcane.core.clock import utcnow
from arcane.core.models import ReplyReference
from arcane.database.database import Database
from arcane.memory.short_term import ChannelActivity
from arcane.personalities.base import BehaviorProfile, InitiativeProfile, TimingProfile
from arcane.personalities.registry import load_personality
from tests.factories import DM, GENERAL, from_bot, incoming, reply_to_bot

MP3 = load_personality("mp3")
NOW = utcnow()
ALICE, BOB, CAROL = 1, 2, 3


class FixedRandom(random.Random):
    """``random()`` always returns the same value."""

    def __init__(self, value: float) -> None:
        super().__init__(0)
        self.value = value

    def random(self) -> float:
        return self.value


def engine(roll: float = 0.0, **kwargs: object) -> DecisionEngine:
    return DecisionEngine(MP3, rng=FixedRandom(roll), **kwargs)  # type: ignore[arg-type]


def tracker() -> ConversationTracker:
    return ConversationTracker(
        "mp3",
        timeout=timedelta(seconds=MP3.behavior.conversation_timeout_seconds),
        focus_timeout=timedelta(seconds=MP3.behavior.focus_timeout_seconds),
        opener_window=timedelta(seconds=MP3.behavior.opener_reply_window_seconds),
    )


def conversation_with(partner: int, name: str = "alice", *, seconds_ago: float = 10):
    at = NOW - timedelta(seconds=seconds_ago)
    return ConversationState(
        channel_id=GENERAL.channel_id,
        started_at=at - timedelta(minutes=2),
        last_activity_at=at,
        partner_id=partner,
        partner_name=name,
        partner_last_message_at=at,
        participants={partner: name},
        bot_turns=2,
        user_turns=2,
    )


# ---------------------------------------------------------------------- decisions


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (from_bot("my own words"), Reason.OWN_MESSAGE),
        (incoming("beep", author_is_bot=True, author_id=50), Reason.BOT_AUTHOR),
        (incoming("   "), Reason.EMPTY),
        (incoming("random chatter about games"), Reason.NOT_ADDRESSED),
    ],
)
def test_ignored_messages(message, reason) -> None:
    decision = engine().decide(message, None, now=NOW)
    assert not decision.should_respond
    assert decision.reason is reason


def test_attachment_only_message_is_not_empty() -> None:
    message = incoming("", mentions_bot=True, attachments=("image: x.png",))
    assert engine().decide(message, None, now=NOW).reason is Reason.MENTION


@pytest.mark.parametrize(
    ("message", "reason", "priority"),
    [
        (incoming("hi", channel=DM), Reason.DIRECT_MESSAGE, Priority.DIRECT),
        (incoming("hey you", mentions_bot=True), Reason.MENTION, Priority.DIRECT),
        (incoming("good point", reply_to=reply_to_bot()), Reason.REPLY_TO_BOT, Priority.DIRECT),
        (incoming("what does mp3 think"), Reason.NAME_MENTIONED, Priority.NAMED),
    ],
)
def test_direct_engagement_gets_a_response(message, reason, priority) -> None:
    decision = engine().decide(message, None, now=NOW)
    assert decision.should_respond
    assert decision.reason is reason
    assert decision.priority is priority


def test_dms_can_be_disabled() -> None:
    decision = engine(respond_in_dms=False).decide(incoming("hi", channel=DM), None, now=NOW)
    assert decision.reason is Reason.DMS_DISABLED


def test_name_mention_is_probabilistic() -> None:
    message = incoming("mp3 is around somewhere")
    assert engine(roll=0.99).decide(message, None, now=NOW).reason is (
        Reason.NAME_MENTIONED_SKIPPED
    )


def test_bots_can_be_enabled_per_personality() -> None:
    chatty = MP3.model_copy(
        update={"behavior": MP3.behavior.model_copy(update={"respond_to_bots": True})}
    )
    message = incoming("hey", author_is_bot=True, author_id=50, mentions_bot=True)
    decision = DecisionEngine(chatty, rng=FixedRandom(0)).decide(message, None, now=NOW)
    assert decision.reason is Reason.MENTION


def test_partner_continuation() -> None:
    conversation = conversation_with(ALICE)
    decision = engine().decide(incoming("and another thing"), conversation, now=NOW)
    assert decision.should_respond
    assert decision.reason is Reason.CONTINUATION


def test_partner_talking_to_someone_else_is_ignored() -> None:
    conversation = conversation_with(ALICE)
    to_bob = incoming("lol bob", reply_to=ReplyReference(message_id=5, author_name="bob"))
    pinging_bob = incoming(
        "@bob look", mentioned_user_ids=frozenset({BOB}), addressed_user_ids=frozenset({BOB})
    )
    for message in (to_bob, pinging_bob):
        decision = engine().decide(message, conversation, now=NOW)
        assert decision.reason is Reason.ADDRESSED_ELSEWHERE


def test_focus_stays_on_partner() -> None:
    conversation = conversation_with(ALICE)
    message = incoming("i have thoughts too", author_id=BOB, author_name="bob")
    decision = engine().decide(message, conversation, now=NOW)
    assert decision.reason is Reason.FOCUSED_ON_OTHER_USER


def test_others_still_get_answers_when_they_engage_directly() -> None:
    conversation = conversation_with(ALICE)
    message = incoming("mp3 what about me", author_id=BOB, author_name="bob", mentions_bot=True)
    assert engine().decide(message, conversation, now=NOW).reason is Reason.MENTION


def test_participant_picks_up_when_partner_goes_quiet() -> None:
    conversation = conversation_with(ALICE, seconds_ago=MP3.behavior.focus_timeout_seconds + 5)
    conversation.participants[BOB] = "bob"
    message = incoming("so anyway", author_id=BOB, author_name="bob")
    assert engine().decide(message, conversation, now=NOW).reason is Reason.CONTINUATION


def test_expired_conversation_is_not_continued() -> None:
    conversation = conversation_with(
        ALICE, seconds_ago=MP3.behavior.conversation_timeout_seconds + 5
    )
    decision = engine().decide(incoming("hello again"), conversation, now=NOW)
    assert decision.reason is Reason.NOT_ADDRESSED


def test_opener_reply() -> None:
    state = ConversationState(
        channel_id=GENERAL.channel_id,
        started_at=NOW - timedelta(minutes=5),
        last_activity_at=NOW - timedelta(minutes=5),
        awaiting_reply_since=NOW - timedelta(minutes=5),
        bot_turns=1,
    )
    decision = engine().decide(incoming("ooh good question"), state, now=NOW)
    assert decision.reason is Reason.OPENER_REPLY


def test_spontaneous_interest() -> None:
    message = incoming("honestly i think the bronze age collapse is the most interesting era")
    assert engine(roll=0.0).decide(message, None, now=NOW).reason is Reason.INTEREST
    assert engine(roll=0.5).decide(message, None, now=NOW).reason is Reason.NOT_ADDRESSED
    recently_spoke = NOW - timedelta(seconds=30)
    decision = engine(roll=0.0).decide(message, None, now=NOW, last_bot_message_at=recently_spoke)
    assert decision.reason is Reason.NOT_ADDRESSED
    short = incoming("history lol")
    assert engine(roll=0.0).decide(short, None, now=NOW).reason is Reason.NOT_ADDRESSED


def test_rate_limit_applies_to_mentions() -> None:
    limiter = ReplyRateLimiter(per_channel=10, per_user=2)
    limiter.record(GENERAL.channel_id, ALICE, NOW)
    limiter.record(GENERAL.channel_id, ALICE, NOW)
    decision = engine().decide(
        incoming("again", mentions_bot=True), None, now=NOW, rate_limiter=limiter
    )
    assert decision.reason is Reason.RATE_LIMITED
    other = incoming("hey", mentions_bot=True, author_id=BOB, author_name="bob")
    assert engine().decide(other, None, now=NOW, rate_limiter=limiter).should_respond


# --------------------------------------------------------------------- rate limits


def test_sliding_window_counter() -> None:
    counter = SlidingWindowCounter(timedelta(seconds=60))
    counter.add("k", NOW - timedelta(seconds=90))
    counter.add("k", NOW - timedelta(seconds=30))
    assert counter.count("k", NOW) == 1
    counter.cleanup(NOW + timedelta(seconds=120))
    assert len(counter) == 0


def test_channel_limit() -> None:
    limiter = ReplyRateLimiter(per_channel=2, per_user=10)
    for user in (ALICE, BOB):
        limiter.record(GENERAL.channel_id, user, NOW)
    assert not limiter.allows(GENERAL.channel_id, CAROL, NOW)
    assert limiter.allows(GENERAL.channel_id, CAROL, NOW + timedelta(seconds=61))


# ------------------------------------------------------------------------- tracker


def test_engagement_starts_conversation_with_partner() -> None:
    conversations = tracker()
    state = conversations.note_engagement(incoming("hey", mentions_bot=True), NOW)
    assert state.partner_id == ALICE
    assert conversations.is_active(GENERAL.channel_id, NOW)
    after = NOW + conversations.timeout + timedelta(seconds=1)
    assert not conversations.is_active(GENERAL.channel_id, after)


def test_focus_switches_only_after_partner_goes_quiet() -> None:
    conversations = tracker()
    conversations.note_engagement(incoming("hey", mentions_bot=True), NOW)
    bob = incoming("me too", author_id=BOB, author_name="bob", mentions_bot=True)

    state = conversations.note_engagement(bob, NOW + timedelta(seconds=20))
    assert state.partner_id == ALICE
    assert state.participants == {ALICE: "alice", BOB: "bob"}

    later = NOW + timedelta(seconds=MP3.behavior.focus_timeout_seconds + 30)
    state = conversations.note_engagement(bob, later)
    assert state.partner_id == BOB


def test_partner_activity_keeps_focus() -> None:
    conversations = tracker()
    conversations.note_engagement(incoming("hey", mentions_bot=True), NOW)
    later = NOW + timedelta(seconds=80)
    conversations.note_partner_activity(incoming("brb"), later)
    state = conversations.get(GENERAL.channel_id)
    assert state is not None
    assert state.partner_is_focused(later + timedelta(seconds=60), conversations.focus_timeout)


def test_opener_keeps_conversation_alive_for_reply_window() -> None:
    conversations = ConversationTracker(
        "mp3",
        timeout=timedelta(minutes=4),
        focus_timeout=timedelta(minutes=2),
        opener_window=timedelta(minutes=15),
    )
    state = conversations.note_bot_message(GENERAL.channel_id, NOW, initiated=True)
    assert state.awaiting_reply_since == NOW

    after_timeout = NOW + timedelta(minutes=5)
    assert conversations.expire(after_timeout) == []

    reply = conversations.note_engagement(incoming("good question"), after_timeout)
    assert reply.partner_id == ALICE
    assert reply.awaiting_reply_since is None
    assert reply.started_at == NOW


def test_expire_returns_finished_conversations_and_restore() -> None:
    conversations = tracker()
    conversations.note_engagement(incoming("hey", mentions_bot=True), NOW)
    conversations.note_bot_message(GENERAL.channel_id, NOW)
    record = conversations.get(GENERAL.channel_id).to_record("mp3")  # type: ignore[union-attr]

    expired = conversations.expire(NOW + timedelta(minutes=30))
    assert [state.partner_id for state in expired] == [ALICE]
    assert len(conversations) == 0

    restored = tracker()
    assert restored.restore([record], NOW + timedelta(seconds=5)) == 1
    assert restored.restore([record], NOW + timedelta(hours=1)) == 0


# -------------------------------------------------------------------------- timing


def test_timing_disabled_is_instant() -> None:
    timing = HumanTiming(TimingProfile(), enabled=False)
    assert timing.reaction_delay() == 0
    assert timing.reading_time("x" * 500) == 0
    assert timing.max_reading_seconds() == 0
    assert timing.typing_duration("x" * 500) == 0
    assert timing.pause_between_messages() == 0


def test_typing_is_60_wpm_by_default() -> None:
    profile = TimingProfile(variation=0.0)
    assert profile.typing_speed_wpm == 60
    assert profile.typing_cps == pytest.approx(5.0)  # 60 words x 5 chars per minute
    timing = HumanTiming(profile, rng=random.Random(1))
    # 300 characters is exactly one minute of typing at 60 wpm, clamped to the max.
    assert timing.typing_duration("x" * 100) == pytest.approx(20.0)
    assert timing.typing_duration("lol yeah") == pytest.approx(1.6)
    assert timing.typing_duration("x") == profile.min_typing_seconds
    assert timing.typing_duration("x" * 300) == profile.max_typing_seconds


def test_typing_speed_is_configurable_in_wpm() -> None:
    fast = HumanTiming(TimingProfile(typing_speed_wpm=120, variation=0.0))
    assert fast.typing_duration("x" * 100) == pytest.approx(10.0)


def test_reading_scales_with_length() -> None:
    timing = HumanTiming(TimingProfile(reading_speed_wpm=300, variation=0.0))
    assert timing.reading_time("x" * 250) == pytest.approx(10.0)
    assert timing.reading_time("") == 0
    reaction = HumanTiming(TimingProfile(reaction_seconds=(0.3, 1.2)), rng=random.Random(3))
    assert all(0.3 <= reaction.reaction_delay() <= 1.2 for _ in range(50))


def test_timing_has_human_variation() -> None:
    timing = HumanTiming(TimingProfile(), rng=random.Random(42))
    samples = [timing.typing_duration("a medium length message here") for _ in range(50)]
    assert statistics.pstdev(samples) > 0.1
    assert all(sample > 0 for sample in samples)
    # The noise is centred on the 60 wpm duration (28 chars -> 5.6 s).
    assert 4.5 < statistics.median(samples) < 6.7


# ---------------------------------------------------------------------- initiative


def _activity(quiet_minutes: float, *, self_last: bool = False) -> ChannelActivity:
    last = NOW - timedelta(minutes=quiet_minutes)
    return ChannelActivity(
        last_message_at=last,
        last_message_is_self=self_last,
        last_human_message_at=last,
    )


def _evaluate(planner: InitiativePlanner, **overrides: object):
    kwargs: dict[str, object] = {
        "now": NOW,
        "activity": _activity(120),
        "conversation_active": False,
        "initiations_today": 0,
        "last_initiation_at": None,
    }
    kwargs.update(overrides)
    return planner.evaluate(**kwargs)  # type: ignore[arg-type]


def test_initiative_when_conditions_met() -> None:
    planner = InitiativePlanner(InitiativeProfile(chance=1.0), rng=FixedRandom(0.0))
    assert _evaluate(planner).should_initiate


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"conversation_active": True}, "conversation in progress"),
        ({"activity": ChannelActivity()}, "no recent activity"),
        ({"activity": _activity(120, self_last=True)}, "last message was ours"),
        ({"activity": _activity(10)}, "channel not quiet long enough"),
        ({"activity": _activity(60 * 48)}, "nobody has been around recently"),
        ({"initiations_today": 2}, "daily limit reached"),
        ({"last_initiation_at": NOW - timedelta(hours=1)}, "too soon since the last opener"),
    ],
)
def test_initiative_blockers(overrides: dict[str, object], reason: str) -> None:
    planner = InitiativePlanner(InitiativeProfile(chance=1.0), rng=FixedRandom(0.0))
    decision = _evaluate(planner, **overrides)
    assert not decision.should_initiate
    assert decision.reason == reason


def test_initiative_chance_and_switches() -> None:
    assert _evaluate(
        InitiativePlanner(InitiativeProfile(chance=0.3), rng=FixedRandom(0.9))
    ).reason == ("chance")
    assert _evaluate(InitiativePlanner(InitiativeProfile(enabled=False))).reason == "disabled"
    night_only = InitiativeProfile(
        chance=1.0, active_hours_utc=((NOW.hour + 1) % 24, (NOW.hour + 2) % 24)
    )
    assert _evaluate(InitiativePlanner(night_only)).reason == "outside active hours"
    wrapping = InitiativeProfile(chance=1.0, active_hours_utc=((NOW.hour + 1) % 24, NOW.hour))
    assert not _evaluate(InitiativePlanner(wrapping, rng=FixedRandom(0))).should_initiate


def test_topic_choice_avoids_recent() -> None:
    planner = InitiativePlanner(InitiativeProfile(), rng=random.Random(3))
    topics = ["a", "b", "c"]
    for _ in range(20):
        assert planner.choose_topic(topics, recent=["a", "b"]) == "c"
    assert planner.choose_topic(topics, recent=topics) in topics
    assert planner.choose_topic([], recent=[]) is None


async def test_initiative_log() -> None:
    async with Database(":memory:") as db:
        log = InitiativeLog(db, "mp3")
        await log.record(GENERAL.channel_id, "stoicism", NOW - timedelta(days=20))
        await log.record(GENERAL.channel_id, "zero", NOW - timedelta(hours=2))
        assert await log.count_since(GENERAL.channel_id, NOW - timedelta(days=1)) == 1
        assert await log.last_at(GENERAL.channel_id) is not None
        assert await log.recent_topics() == ["zero", "stoicism"]
        assert await log.prune(NOW) == 1
        assert await log.last_at(999) is None


def test_behavior_defaults_are_consistent() -> None:
    behavior = BehaviorProfile()
    assert behavior.focus_timeout_seconds < behavior.conversation_timeout_seconds
