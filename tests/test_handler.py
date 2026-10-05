"""End-to-end tests of the conversation pipeline with fake transport and provider."""

from __future__ import annotations

import asyncio
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
from arcane.core.models import ChannelInfo
from arcane.database.database import Database
from arcane.personalities.base import Personality, TimingProfile
from arcane.personalities.registry import load_personality
from arcane.runtime import build_conversation_handler
from tests.factories import DM, GENERAL, incoming, reply_to_bot
from tests.fakes import FakeTransport, GatedProvider, ScriptedProvider

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
    assert 'replying to alice: "mp3 what caused it?"' in messages[0].content


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


async def test_bursts_are_answered_once(db: Database) -> None:
    fast = TimingProfile(
        reaction_seconds=(0, 0),
        thinking_seconds=(0, 0),
        pause_between_messages_seconds=(0, 0),
        min_typing_seconds=0,
        max_typing_seconds=0.01,
        variation=0,
        debounce_seconds=0.1,
        max_debounce_seconds=1.0,
    )
    h = await make_harness(db, personality=_personality(timing=fast), settings=Settings())
    await h.say("mp3 quick question", mentions_bot=True)
    await h.say("actually two questions")
    await h.say("what's your favourite paradox")
    await h.settle()

    assert len(h.provider.calls) == 1
    system = h.provider.calls[0][0][0].content
    assert "quick question actually two questions what's your favourite paradox" in system


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


async def test_humanized_delays_are_applied(db: Database) -> None:
    h = await make_harness(db, ScriptedProvider("x" * 70), settings=Settings())
    # No real waiting: sleeps are recorded by the fake. Debounce uses real time,
    # so shrink it for the test.
    h.handler._timing.profile = MP3.timing.model_copy(update={"debounce_seconds": 0.01})
    await h.say("mp3, why do you think we find symmetry beautiful?", mentions_bot=True)
    await h.settle()

    assert h.transport.texts == ["x" * 70]
    assert sum(h.sleeps) > 2  # reading + thinking + typing


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
    system = h.provider.calls[0][0][0].content
    assert "Start a new conversation about" in system

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
