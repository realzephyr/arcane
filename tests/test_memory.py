from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import timedelta

import pytest

from arcane.ai.providers.base import ProviderUnavailableError
from arcane.core.clock import utcnow
from arcane.core.models import ReplyReference
from arcane.database.database import Database
from arcane.database.models import ConversationRecord, MemoryKind
from arcane.memory.extraction import (
    LLMMemoryExtractor,
    MemoryConsolidator,
    NullMemoryExtractor,
    parse_candidates,
)
from arcane.memory.filters import contains_sensitive_data, normalize_memory_text, similarity
from arcane.memory.long_term import LongTermMemory
from arcane.memory.short_term import ShortTermMemory
from tests.factories import GENERAL, from_bot, history, incoming
from tests.fakes import ScriptedProvider


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    async with Database(":memory:") as database:
        yield database


@pytest.fixture
def short_term(db: Database) -> ShortTermMemory:
    return ShortTermMemory(db, "mp3", ttl=timedelta(hours=1), max_messages_per_channel=20)


@pytest.fixture
def long_term(db: Database) -> LongTermMemory:
    return LongTermMemory(db, "mp3", max_memories_per_user=3)


# ------------------------------------------------------------------ short-term


async def test_records_and_returns_messages_in_order(short_term: ShortTermMemory) -> None:
    now = utcnow()
    for offset, text in enumerate(["first", "second", "third"]):
        await short_term.record_message(
            incoming(text, created_at=now - timedelta(minutes=10 - offset))
        )
    await short_term.record_message(
        incoming(
            "a reply",
            created_at=now,
            reply_to=ReplyReference(message_id=1, author_name="bob"),
        )
    )

    recent = await short_term.recent_messages(GENERAL.channel_id, limit=3)

    assert [m.content for m in recent] == ["second", "third", "a reply"]
    assert recent[-1].reply_to_author_name == "bob"


async def test_messages_are_isolated_per_bot(db: Database) -> None:
    mp3 = ShortTermMemory(db, "mp3")
    other = ShortTermMemory(db, "other")
    await mp3.record_message(incoming("for mp3"))
    assert await other.recent_messages(GENERAL.channel_id, limit=10) == []


async def test_attachments_are_stored_as_text(short_term: ShortTermMemory) -> None:
    await short_term.record_message(incoming("look", attachments=("image: cat.png",)))
    (stored,) = await short_term.recent_messages(GENERAL.channel_id, limit=5)
    assert stored.content == "look [image: cat.png]"


async def test_edit_and_delete(short_term: ShortTermMemory) -> None:
    message = incoming("tpyo")
    await short_term.record_message(message)
    assert await short_term.update_message_content(message.message_id, "typo")
    (stored,) = await short_term.recent_messages(GENERAL.channel_id, limit=5)
    assert stored.content == "typo"
    assert await short_term.delete_message(message.message_id)
    assert await short_term.recent_messages(GENERAL.channel_id, limit=5) == []
    assert not await short_term.delete_message(message.message_id)


async def test_prune_enforces_ttl_and_channel_cap(short_term: ShortTermMemory) -> None:
    now = utcnow()
    await short_term.record_message(incoming("ancient", created_at=now - timedelta(hours=3)))
    for index in range(25):
        await short_term.record_message(
            incoming(f"msg {index}", created_at=now - timedelta(seconds=100 - index))
        )

    stats = await short_term.prune(now)

    assert stats.expired_messages == 1
    assert stats.overflow_messages == 5
    remaining = await short_term.recent_messages(GENERAL.channel_id, limit=100)
    assert len(remaining) == 20
    assert remaining[-1].content == "msg 24"


async def test_channel_activity(short_term: ShortTermMemory) -> None:
    empty = await short_term.channel_activity(GENERAL.channel_id)
    assert empty.last_message_at is None

    now = utcnow()
    await short_term.record_message(incoming("hi", created_at=now - timedelta(minutes=5)))
    await short_term.record_message(from_bot("hey", created_at=now))
    activity = await short_term.channel_activity(GENERAL.channel_id)

    assert activity.last_message_is_self is True
    assert activity.last_human_message_at is not None
    assert activity.last_message_at is not None
    assert activity.last_human_message_at < activity.last_message_at


async def test_conversation_round_trip(short_term: ShortTermMemory) -> None:
    now = utcnow()
    record = ConversationRecord(
        bot_id="mp3",
        channel_id=GENERAL.channel_id,
        started_at=now - timedelta(minutes=3),
        last_activity_at=now,
        partner_id=1,
        partner_name="alice",
        participants={1: "alice", 2: "bob"},
        bot_turns=2,
        user_turns=3,
    )
    await short_term.save_conversation(record)
    await short_term.save_conversation(replace(record, bot_turns=3))  # upsert

    (loaded,) = await short_term.load_conversations()
    assert loaded.participants == {1: "alice", 2: "bob"}
    assert loaded.bot_turns == 3
    assert abs((loaded.started_at - record.started_at).total_seconds()) < 0.01

    await short_term.delete_conversation(GENERAL.channel_id)
    assert await short_term.load_conversations() == []


# ------------------------------------------------------------------- long-term


async def test_profiles_track_interactions(long_term: LongTermMemory) -> None:
    await long_term.touch_user(1, "alice")
    await long_term.touch_user(1, "Alice!", interaction=True)
    await long_term.touch_user(1, "Alice!", interaction=True)
    profile = await long_term.get_profile(1)
    assert profile is not None
    assert profile.display_name == "Alice!"
    assert profile.interaction_count == 2
    assert await long_term.get_profile(2) is None


async def test_remember_and_recall(long_term: LongTermMemory) -> None:
    await long_term.remember("studies physics", user_id=1, importance=0.9, kind=MemoryKind.FACT)
    await long_term.remember("likes jazz", user_id=1, importance=0.3)
    await long_term.remember("plays chess", user_id=2, importance=0.5)

    memories = await long_term.recall(1, limit=5)

    assert [m.content for m in memories] == ["studies physics", "likes jazz"]
    assert memories[0].kind is MemoryKind.FACT
    again = await long_term.recall(1, limit=1)
    assert again[0].access_count == 1


async def test_near_duplicates_reinforce_instead_of_duplicating(
    long_term: LongTermMemory,
) -> None:
    first = await long_term.remember("studies physics at university", user_id=1, importance=0.4)
    second = await long_term.remember("Studies physics at university.", user_id=1, importance=0.8)
    assert first is not None and second is not None
    assert first.id == second.id
    assert second.importance == 0.8
    assert len(await long_term.recall(1, limit=10)) == 1


async def test_sensitive_and_empty_memories_are_rejected(long_term: LongTermMemory) -> None:
    assert await long_term.remember("email is alice@example.com", user_id=1) is None
    assert await long_term.remember("phone +1 555 123 4567", user_id=1) is None
    assert await long_term.remember("their password is hunter2", user_id=1) is None
    assert await long_term.remember("  ", user_id=1) is None
    assert await long_term.recall(1) == []


async def test_cap_evicts_lowest_value(long_term: LongTermMemory) -> None:
    await long_term.remember("core identity fact", user_id=1, importance=1.0)
    await long_term.remember("trivial detail", user_id=1, importance=0.05)
    await long_term.remember("favourite philosopher is Hume", user_id=1, importance=0.6)
    await long_term.remember("works as a nurse", user_id=1, importance=0.7)

    contents = {m.content for m in await long_term.recall(1, limit=10)}
    assert "trivial detail" not in contents
    assert len(contents) == 3


async def test_expiry_and_prune(long_term: LongTermMemory) -> None:
    now = utcnow()
    await long_term.remember("temporary plan", user_id=1, ttl=timedelta(hours=1), now=now)
    await long_term.remember("permanent fact", user_id=1, now=now)
    later = now + timedelta(hours=2)
    assert [m.content for m in await long_term.recall(1, now=later)] == ["permanent fact"]
    assert await long_term.prune(later) == 1


async def test_forget_user(long_term: LongTermMemory) -> None:
    await long_term.touch_user(1, "alice")
    await long_term.remember("likes Kant", user_id=1)
    await long_term.remember("likes Hume", user_id=2)
    assert await long_term.forget_user(1) == 2
    assert await long_term.recall(1) == []
    assert await long_term.get_profile(1) is None
    assert len(await long_term.recall(2)) == 1


# ---------------------------------------------------------------------- filters


def test_filters() -> None:
    assert contains_sensitive_data("reach me at bob@mail.org")
    assert contains_sensitive_data("api key sk-abcdef0123456789abcdef0123456789")
    assert not contains_sensitive_data("thinks Plato was overrated")
    assert normalize_memory_text("  - likes   jazz  ") == "likes jazz"
    assert len(normalize_memory_text("word " * 200)) <= 280
    assert similarity("studies physics", "Studies physics.") == 1.0
    assert similarity("likes jazz", "studies physics") == 0.0


# ------------------------------------------------------------------- extraction


def test_parse_candidates_is_lenient() -> None:
    raw = (
        "<think>hmm</think>```json\n"
        + json.dumps(
            {
                "memories": [
                    {"person": "Alice", "content": "studies physics", "importance": 0.9},
                    {"person": "alice", "content": "likes tea", "kind": "preference"},
                    {"person": "carol", "content": "not a participant"},
                    {"person": "bob", "content": "", "importance": 1},
                    {"person": "bob", "content": "trivial", "importance": 0.1},
                    "garbage",
                ]
            }
        )
        + "\n```"
    )
    candidates = parse_candidates(raw, {1: "alice", 2: "bob"}, min_importance=0.3)
    assert [(c.user_id, c.content) for c in candidates] == [
        (1, "studies physics"),
        (1, "likes tea"),
    ]
    assert candidates[1].kind is MemoryKind.PREFERENCE
    assert parse_candidates("not json at all", {1: "alice"}) == []


async def test_llm_extractor_uses_json_mode_and_survives_failures() -> None:
    provider = ScriptedProvider(
        json.dumps({"memories": [{"person": "alice", "content": "loves Spinoza"}]})
    )
    extractor = LLMMemoryExtractor(provider, bot_name="mp3")
    transcript = [history("spinoza is the best"), history("why?", is_self=True)]

    candidates = await extractor.extract(transcript, {1: "alice"})

    assert [c.content for c in candidates] == ["loves Spinoza"]
    messages, options, _ = provider.calls[0]
    assert options is not None and options.json_mode
    assert "alice: spinoza is the best" in messages[-1].content
    assert "mp3: why?" in messages[-1].content

    failing = LLMMemoryExtractor(ScriptedProvider(ProviderUnavailableError("down")), bot_name="mp3")
    assert await failing.extract(transcript, {1: "alice"}) == []
    assert await NullMemoryExtractor().extract(transcript, {1: "alice"}) == []


async def test_consolidator_stores_extracted_memories(
    short_term: ShortTermMemory, long_term: LongTermMemory
) -> None:
    now = utcnow()
    for offset, text in enumerate(["i study physics", "mostly quantum stuff", "it's wild"]):
        await short_term.record_message(
            incoming(text, created_at=now - timedelta(seconds=60 - offset))
        )
    provider = ScriptedProvider(
        json.dumps(
            {"memories": [{"person": "alice", "content": "studies physics", "importance": 0.8}]}
        )
    )
    consolidator = MemoryConsolidator(
        short_term, long_term, LLMMemoryExtractor(provider, bot_name="mp3")
    )
    conversation = ConversationRecord(
        bot_id="mp3",
        channel_id=GENERAL.channel_id,
        started_at=now - timedelta(minutes=1),
        last_activity_at=now,
        participants={1: "alice"},
    )

    assert await consolidator.consolidate(conversation) == 1
    assert [m.content for m in await long_term.recall(1)] == ["studies physics"]


async def test_consolidator_skips_short_conversations(
    short_term: ShortTermMemory, long_term: LongTermMemory
) -> None:
    now = utcnow()
    await short_term.record_message(incoming("hi", created_at=now))
    provider = ScriptedProvider('{"memories": []}')
    consolidator = MemoryConsolidator(
        short_term, long_term, LLMMemoryExtractor(provider, bot_name="mp3")
    )
    conversation = ConversationRecord(
        bot_id="mp3",
        channel_id=GENERAL.channel_id,
        started_at=now,
        last_activity_at=now,
        participants={1: "alice"},
    )
    assert await consolidator.consolidate(conversation) == 0
    assert provider.calls == []


async def test_extractor_uses_the_reply_context_window() -> None:
    """A different num_ctx makes Ollama reload the model, so they must match."""
    from arcane.ai.response_manager import ResponseManager
    from arcane.personalities.registry import load_personality

    mp3 = load_personality("mp3")
    provider = ScriptedProvider('{"memories": []}')
    extractor = LLMMemoryExtractor(
        provider, bot_name="mp3", context_window=mp3.model.context_window
    )
    await extractor.extract([history("i study physics")], {1: "alice"})
    _, extraction_options, _ = provider.calls[0]
    reply_options = ResponseManager(mp3, provider).base_options()
    assert extraction_options is not None
    assert extraction_options.context_window == reply_options.context_window


def _truncated_nine_memories() -> str:
    memories = [
        {"person": "alice" if i % 2 else "bob", "content": f"memory number {i}", "importance": 0.9}
        for i in range(9)
    ]
    complete = json.dumps({"memories": memories})
    # Cut off inside the seventh object, as the token limit would.
    return complete[: complete.index('"memory number 6"') + 5]


def test_truncated_output_keeps_the_complete_memories(caplog: pytest.LogCaptureFixture) -> None:
    raw = _truncated_nine_memories()
    with pytest.raises(ValueError):
        json.loads(raw)

    with caplog.at_level("INFO", logger="arcane.memory.extraction"):
        candidates = parse_candidates(raw, {1: "alice", 2: "bob"}, max_per_user=9)

    assert [c.content for c in candidates] == [f"memory number {i}" for i in range(6)]
    assert any("salvaged 6" in record.getMessage() for record in caplog.records)


def test_truncated_output_salvage_edge_cases() -> None:
    participants = {1: "alice"}
    item = '{"person": "alice", "content": "likes tea"}'
    (salvaged,) = parse_candidates('{"memories": [' + item + ", {", participants)
    assert salvaged.content == "likes tea"
    # A bare array in a code fence, cut off mid-string.
    assert len(parse_candidates("```json\n[" + item + ', {"person": "al', participants)) == 1
    assert parse_candidates('{"memories": [{"person": "ali', participants) == []
    assert parse_candidates('{"memories": [', participants) == []


async def test_llm_extractor_asks_for_few_memories_with_room_to_finish() -> None:
    provider = ScriptedProvider(_truncated_nine_memories())
    extractor = LLMMemoryExtractor(provider, bot_name="mp3", max_per_user=9)

    candidates = await extractor.extract([history("i study physics")], {1: "alice", 2: "bob"})

    assert len(candidates) == 6
    messages, options, _ = provider.calls[0]
    assert options is not None and options.max_tokens == 600
    assert "at most 3 memories per person and at most 8 in total" in messages[0].content
