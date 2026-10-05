"""End-to-end tests of the conversation pipeline with fake transport and provider."""

from __future__ import annotations

import asyncio
import itertools
import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from arcane.ai.providers.base import LLMProvider, ProviderUnavailableError
from arcane.config.settings import Settings
from arcane.conversation.decision import Decision, Reason
from arcane.conversation.handler import ConversationHandler
from arcane.core.clock import utcnow
from arcane.core.models import ChannelInfo, ReplyReference
from arcane.database.database import Database
from arcane.personalities.base import Personality, TimingProfile
from arcane.personalities.registry import load_personality
from arcane.runtime import build_conversation_handler
from tests.factories import DM, GENERAL, incoming, reply_to_bot
from tests.fakes import FakeTransport, GatedProvider, RecordingProvider, ScriptedProvider

MP3 = load_personality("mp3")
OTHER_CHANNEL = ChannelInfo(channel_id=30, name="offtopic", guild_id=1, guild_name="Test Server")
ALICE, BOB = 1, 2


class Clock:
    def __init__(self) -> None:
        self.now = utcnow()

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@dataclass
class Harness:
    handler: ConversationHandler
    transport: FakeTransport
    provider: ScriptedProvider
    clock: Clock
    sleeps: list[float]
    db: Database

    async def say(self, *args: object, **kwargs: object) -> Decision | None:
        kwargs.setdefault("created_at", self.clock.now)
        return await self.handler.handle_message(incoming(*args, **kwargs))  # type: ignore[arg-type]

    async def settle(self) -> None:
        await self.handler.drain()


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database(":memory:") as database:
        yield database


def _personality(**updates: object) -> Personality:
    return MP3.model_copy(update=updates)


async def make_harness(
    db: Database,
    provider: ScriptedProvider | None = None,
    *,
    personality: Personality = MP3,
    settings: Settings | None = None,
    transport: FakeTransport | None = None,
    allowed: tuple[int, ...] = (),
    initiative: tuple[int, ...] = (),
) -> Harness:
    clock = Clock()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await asyncio.sleep(0)

    provider = provider or ScriptedProvider("sounds about right")
    transport = transport or FakeTransport([GENERAL, OTHER_CHANNEL])
    transport.clock = clock
    handler = build_conversation_handler(
        personality=personality,
        settings=settings or Settings(humanize=False),
        database=db,
        provider=provider,
        transport=transport,
        allowed_channel_ids=allowed,
        initiative_channel_ids=initiative,
        clock=clock,
        sleep=fake_sleep,
        rng=random.Random(7),
    )
    await handler.start()
    return Harness(handler, transport, provider, clock, sleeps, db)


async def test_mention_gets_a_reply_with_typing(db: Database) -> None:
    h = await make_harness(db)
    decision = await h.say("hey mp3, thoughts on stoicism?", mentions_bot=True)
    await h.settle()

    assert decision is not None and decision.reason is Reason.MENTION
    assert h.transport.texts == ["sounds about right"]
    assert h.transport.events == ["typing_on", "send:sounds about right", "typing_off"]
    _, _, reference = h.transport.sent[0]
    assert reference is None  # nothing happened in between, so a plain message

    stored = await h.handler._short_term.recent_messages(GENERAL.channel_id, 10)
    assert [m.content for m in stored] == ["hey mp3, thoughts on stoicism?", "sounds about right"]
    assert stored[-1].is_self
    profile = await h.handler._long_term.get_profile(ALICE)
    assert profile is not None and profile.interaction_count == 1


async def test_prompt_includes_history_and_target(db: Database) -> None:
    h = await make_harness(db)
    await h.say("i've been reading about the bronze age collapse", author_name="bob", author_id=BOB)
    await h.say("mp3 what caused it?", mentions_bot=True)
    await h.settle()

    messages, _, _ = h.provider.calls[0]
    assert "bob: i've been reading about the bronze age collapse" in messages[1].content
    assert 'replying to alice: "mp3 what caused it?"' in messages[-1].content


async def test_unaddressed_message_is_stored_but_ignored(db: Database) -> None:
    h = await make_harness(db)
    decision = await h.say("anyone up for games tonight")
    await h.settle()
    assert decision is not None and decision.reason is Reason.NOT_ADDRESSED
    assert h.transport.sent == []
    assert len(await h.handler._short_term.recent_messages(GENERAL.channel_id, 10)) == 1


async def test_channel_allow_list(db: Database) -> None:
    h = await make_harness(db, allowed=(GENERAL.channel_id,))
    assert await h.say("hello", channel=OTHER_CHANNEL, mentions_bot=True) is None
    thread = ChannelInfo(channel_id=99, name="thread", guild_id=1, parent_id=GENERAL.channel_id)
    assert await h.say("in a thread", channel=thread, mentions_bot=True) is not None
    await h.settle()
    assert await h.handler._short_term.recent_messages(OTHER_CHANNEL.channel_id, 10) == []


async def test_dms_respect_setting(db: Database) -> None:
    h = await make_harness(db, settings=Settings(humanize=False, respond_in_dms=False))
    assert await h.say("hi", channel=DM) is None


async def test_conversation_continues_and_stays_focused(db: Database) -> None:
    h = await make_harness(db)
    await h.say("mp3, is math invented?", mentions_bot=True)
    await h.settle()

    follow_up = await h.say("because it feels discovered to me")
    interloper = await h.say("unrelated: pizza tonight?", author_id=BOB, author_name="bob")
    await h.settle()

    assert follow_up is not None and follow_up.reason is Reason.CONTINUATION
    assert interloper is not None and interloper.reason is Reason.FOCUSED_ON_OTHER_USER
    assert len(h.transport.sent) == 2


FAST_READING = TimingProfile(
    reaction_seconds=(0.05, 0.05),
    reading_speed_wpm=100_000,
    pause_between_messages_seconds=(0, 0),
    min_typing_seconds=0,
    max_typing_seconds=0.01,
    max_reading_seconds=1.0,
    variation=0,
)


async def test_bursts_are_answered_once(db: Database) -> None:
    h = await make_harness(db, personality=_personality(timing=FAST_READING), settings=Settings())
    await h.say("mp3 quick question", mentions_bot=True)
    await h.say("actually two questions")
    await h.say("what's your favourite paradox")
    await h.settle()

    assert len(h.provider.calls) == 1
    context = "\n".join(m.content for m in h.provider.calls[0][0])
    assert "quick question actually two questions what's your favourite paradox" in context


async def test_reply_reference_when_others_spoke_in_between(db: Database) -> None:
    provider = GatedProvider("fair enough")
    h = await make_harness(db, provider)
    await h.say("mp3 you there?", mentions_bot=True)
    await provider.started.wait()
    await h.say("lol", author_id=BOB, author_name="bob")
    provider.gate.set()
    await h.settle()

    (_, _, reference) = h.transport.sent[0]
    assert reference is not None


async def test_split_reply_sends_multiple_messages(db: Database) -> None:
    h = await make_harness(db, ScriptedProvider("first thought\n\nsecond thought"))
    await h.say("mp3?", mentions_bot=True)
    await h.settle()

    assert h.transport.texts == ["first thought", "second thought"]
    assert h.transport.events == [
        "typing_on",
        "send:first thought",
        "typing_off",
        "typing_on",
        "send:second thought",
        "typing_off",
    ]
    assert all(reference is None for _, _, reference in h.transport.sent[1:])


async def test_provider_failure_is_silent(db: Database) -> None:
    h = await make_harness(db, ScriptedProvider(ProviderUnavailableError("ollama down")))
    await h.say("mp3?", mentions_bot=True)
    await h.settle()
    assert h.transport.sent == []
    state = h.handler._tracker.get(GENERAL.channel_id)
    assert state is not None and state.bot_turns == 0


async def test_transport_failure_is_handled(db: Database) -> None:
    transport = FakeTransport([GENERAL], fail_sends=True)
    h = await make_harness(db, ScriptedProvider("a\n\nb"), transport=transport)
    await h.say("mp3?", mentions_bot=True)
    await h.settle()
    assert transport.events == ["typing_on", "typing_off"]


async def test_rate_limit_counts_replies_not_triggers(db: Database) -> None:
    h = await make_harness(db)
    limit = MP3.behavior.max_replies_per_user_per_minute
    for i in range(limit):
        decision = await h.say(f"mp3 question {i}", mentions_bot=True)
        assert decision is not None and decision.reason is Reason.MENTION
        await h.settle()
    assert len(h.transport.sent) == limit

    blocked = await h.say("mp3 one more?", mentions_bot=True)
    assert blocked is not None and blocked.reason is Reason.RATE_LIMITED

    h.clock.advance(seconds=61)
    allowed = await h.say("mp3 ok now?", mentions_bot=True)
    assert allowed is not None and allowed.reason is Reason.MENTION
    await h.settle()


async def test_queued_spam_cannot_exceed_the_limit(db: Database) -> None:
    provider = GatedProvider("ok")
    h = await make_harness(db, provider)
    for i in range(10):
        await h.say(f"mp3 spam {i}", mentions_bot=True, author_id=100 + i, author_name=f"u{i}")
    provider.gate.set()
    await h.settle()
    assert len(h.transport.sent) <= MP3.behavior.max_replies_per_channel_per_minute


async def test_burst_uses_one_rate_limit_slot(db: Database) -> None:
    provider = GatedProvider("ok")
    h = await make_harness(db, provider)
    await h.say("mp3 first", mentions_bot=True)
    await provider.started.wait()
    for i in range(3):
        await h.say(f"follow-up {i}")
    provider.gate.set()
    await h.settle()
    # One reply for the first message, one batched reply for the follow-ups.
    assert len(h.transport.sent) == 2
    assert h.handler._rate_limiter._users.count(ALICE, h.clock.now) == 2


async def test_stale_triggers_are_dropped(db: Database) -> None:
    h = await make_harness(db)
    old = h.clock.now - timedelta(seconds=MP3.behavior.stale_trigger_seconds + 30)
    await h.say("mp3 hello?", mentions_bot=True, created_at=old)
    await h.settle()
    assert h.transport.sent == []


async def test_typing_starts_only_after_reading(db: Database) -> None:
    transport = FakeTransport([GENERAL, OTHER_CHANNEL])
    provider = RecordingProvider(transport.events, "sounds right")
    slow_reader = FAST_READING.model_copy(update={"reaction_seconds": (0.2, 0.2)})
    h = await make_harness(
        db,
        provider,
        personality=_personality(timing=slow_reader),
        settings=Settings(),
        transport=transport,
    )
    loop = asyncio.get_running_loop()
    said_at = loop.time()
    await h.say("mp3 you around", mentions_bot=True)
    await h.settle()

    # Reading happens first with no indicator; typing starts together with generation.
    assert transport.events == ["typing_on", "generate", "send:sounds right", "typing_off"]
    assert transport.event_times[0] - said_at >= 0.2


async def test_reply_takes_as_long_as_typing_at_60_wpm(db: Database) -> None:
    exact = FAST_READING.model_copy(
        update={"reaction_seconds": (0, 0), "min_typing_seconds": 0, "max_typing_seconds": 45}
    )
    reply = "x" * 50  # 50 characters at 5 per second (60 wpm) = 10 seconds
    h = await make_harness(
        db,
        ScriptedProvider(reply),
        personality=_personality(timing=exact),
        settings=Settings(),
    )
    await h.say("mp3?", mentions_bot=True)
    await h.settle()

    assert h.transport.texts == [reply]
    # Generation was instant, so the bot keeps typing for the rest of the 10 s.
    assert sum(h.sleeps) == pytest.approx(10.0, abs=0.1)


async def test_slow_generation_adds_no_extra_typing_time(db: Database) -> None:
    exact = FAST_READING.model_copy(
        update={"reaction_seconds": (0, 0), "min_typing_seconds": 0, "max_typing_seconds": 0.05}
    )
    provider = GatedProvider("ok")
    h = await make_harness(
        db, provider, personality=_personality(timing=exact), settings=Settings()
    )
    await h.say("mp3?", mentions_bot=True)
    await provider.started.wait()
    await asyncio.sleep(0.1)  # generation takes longer than typing "ok" would
    provider.gate.set()
    await h.settle()
    assert h.transport.texts == ["ok"]
    assert sum(h.sleeps) == 0


async def test_message_arriving_during_generation_gets_its_own_reply(db: Database) -> None:
    provider = GatedProvider("first reply", "second reply")
    h = await make_harness(db, provider)
    await h.say("mp3 what's up", mentions_bot=True)
    await provider.started.wait()
    follow_up = await h.say("also did you see the game")
    provider.gate.set()
    await h.settle()

    assert follow_up is not None and follow_up.reason is Reason.CONTINUATION
    assert h.transport.texts == ["first reply", "second reply"]


async def test_edits_and_deletes(db: Database) -> None:
    h = await make_harness(db)
    message = incoming("tpyo", created_at=h.clock.now)
    await h.handler.handle_message(message)
    await h.handler.handle_edit(message.message_id, "typo")
    stored = await h.handler._short_term.recent_messages(GENERAL.channel_id, 5)
    assert stored[0].content == "typo"
    await h.handler.handle_delete(message.message_id)
    assert await h.handler._short_term.recent_messages(GENERAL.channel_id, 5) == []


async def test_maintenance_expires_and_consolidates(db: Database) -> None:
    extraction = json.dumps(
        {"memories": [{"person": "alice", "content": "loves Hume", "importance": 0.9}]}
    )
    provider = ScriptedProvider("reply one", "reply two", "reply three", extraction)
    h = await make_harness(db, provider)
    for text in ("mp3, hume or kant?", "hume obviously", "the is-ought gap alone"):
        await h.say(text, mentions_bot=True)
        await h.settle()
        h.clock.advance(seconds=5)

    h.clock.advance(minutes=30)
    report = await h.handler.run_maintenance()

    assert report.expired_conversations == 1
    assert report.memories_stored == 1
    memories = await h.handler._long_term.recall(ALICE)
    assert [m.content for m in memories] == ["loves Hume"]
    assert await h.handler._short_term.load_conversations() == []


async def test_conversation_state_survives_restart(db: Database) -> None:
    h = await make_harness(db)
    await h.say("mp3?", mentions_bot=True)
    await h.settle()

    restarted = await make_harness(db)
    assert restarted.handler._tracker.is_active(GENERAL.channel_id, restarted.clock.now)
    decision = await restarted.say("so what do you think")
    assert decision is not None and decision.reason is Reason.CONTINUATION
    await restarted.settle()


async def test_initiative_posts_opener_once(db: Database) -> None:
    eager = MP3.behavior.model_copy(
        update={"initiative": MP3.behavior.initiative.model_copy(update={"chance": 1.0})}
    )
    h = await make_harness(
        db,
        ScriptedProvider("random thought: is math discovered or invented"),
        personality=_personality(behavior=eager),
        initiative=(GENERAL.channel_id,),
    )
    await h.say("gn all", created_at=h.clock.now - timedelta(hours=3))

    assert await h.handler.maybe_initiate()
    assert h.transport.texts == ["random thought: is math discovered or invented"]
    note = h.provider.calls[0][0][-1].content
    assert "Start a new conversation about" in note

    # The last message is now ours, so it won't talk into the void again.
    assert not await h.handler.maybe_initiate()

    answer = await h.say("ooh invented, definitely")
    assert answer is not None and answer.reason is Reason.OPENER_REPLY
    await h.settle()


async def test_initiative_requires_opt_in(db: Database) -> None:
    h = await make_harness(db)
    assert not h.handler.initiative_active
    assert not await h.handler.maybe_initiate()


async def test_reply_to_bot_message_is_answered(db: Database) -> None:
    h = await make_harness(db)
    decision = await h.say("nah", reply_to=reply_to_bot())
    await h.settle()
    assert decision is not None and decision.reason is Reason.REPLY_TO_BOT
    assert h.transport.texts == ["sounds about right"]


async def test_long_plain_conversation_is_followed(db: Database) -> None:
    h = await make_harness(db)
    first = await h.say("yo mp3", mentions_bot=True)
    assert first is not None and first.reason is Reason.MENTION
    await h.settle()

    # Ten plain messages in under a minute: no replies, no mentions, still a conversation.
    for i in range(10):
        h.clock.advance(seconds=5)
        decision = await h.say(f"plain message number {i}")
        assert decision is not None and decision.reason is Reason.CONTINUATION, i
        await h.settle()
    assert len(h.transport.sent) == 11


async def test_conversation_survives_other_people_talking(db: Database) -> None:
    h = await make_harness(db)
    await h.say("mp3 is cereal a soup", mentions_bot=True)
    await h.settle()

    other = await h.say("anyone got the homework", author_id=BOB, author_name="bob")
    assert other is not None and other.reason is Reason.FOCUSED_ON_OTHER_USER

    back = await h.say("like think about it, milk is the broth")
    assert back is not None and back.reason is Reason.CONTINUATION
    await h.settle()
    assert len(h.transport.sent) == 2
    system = h.provider.calls[-1][0]
    assert "replying to alice" in "\n".join(m.content for m in system)


async def test_partner_addressing_rules(db: Database) -> None:
    h = await make_harness(db)
    await h.say("mp3 you up", mentions_bot=True)
    await h.settle()

    passing = await h.say("i told bob about it", mentioned_user_ids=frozenset({BOB}))
    assert passing is not None and passing.reason is Reason.CONTINUATION
    own_reply = await h.say(
        "and another thing",
        reply_to=ReplyReference(message_id=1, author_id=ALICE, author_name="alice"),
    )
    assert own_reply is not None and own_reply.reason is Reason.CONTINUATION
    to_bob = await h.say(
        "@bob what do you think",
        mentioned_user_ids=frozenset({BOB}),
        addressed_user_ids=frozenset({BOB}),
    )
    assert to_bob is not None and to_bob.reason is Reason.ADDRESSED_ELSEWHERE
    await h.settle()


async def test_conversation_lasts_ten_minutes_of_silence(db: Database) -> None:
    h = await make_harness(db)
    await h.say("mp3 brb", mentions_bot=True)
    await h.settle()
    h.clock.advance(minutes=9)
    back = await h.say("ok back")
    assert back is not None and back.reason is Reason.CONTINUATION
    await h.settle()


async def test_history_window_start_stays_stable_for_cache_reuse(db: Database) -> None:
    window = MP3.memory.history_messages
    h = await make_harness(db, ScriptedProvider(*[f"reply {i}" for i in range(window * 3)]))
    await h.say("mp3 lets talk", mentions_bot=True)
    await h.settle()
    for i in range(window * 2):
        h.clock.advance(seconds=5)
        await h.say(f"message {i}")
        await h.settle()

    firsts = [call[0][1].content for call in h.provider.calls]
    changes = sum(1 for a, b in itertools.pairwise(firsts) if a != b)
    # Each turn adds two messages (theirs and ours). A one-message sliding window
    # would change the first history turn on nearly every reply once full; the
    # anchored window only jumps forward once per `slack` new messages.
    messages_added = 2 * len(firsts)
    assert 1 <= changes <= messages_added // MP3.memory.history_slack
    longest = max(len(call[0]) for call in h.provider.calls)
    assert longest <= MP3.memory.max_history_messages + 2


async def test_close_cancels_in_flight_work(db: Database) -> None:
    provider = GatedProvider("never sent")
    h = await make_harness(db, provider)
    await h.say("mp3?", mentions_bot=True)
    await provider.started.wait()
    await h.handler.close()
    assert h.transport.sent == []
    assert await h.say("mp3 again?", mentions_bot=True) is None


def test_harness_provider_type() -> None:
    assert issubclass(ScriptedProvider, LLMProvider)
