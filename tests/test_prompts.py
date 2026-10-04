from __future__ import annotations

from datetime import UTC, datetime, timedelta

from arcane.ai.prompts import GROUND_RULES, PromptBuilder, PromptContext
from arcane.core.clock import utcnow
from arcane.database.models import MemoryKind, MemoryRecord, UserProfile
from arcane.personalities.registry import load_personality
from tests.factories import DM, GENERAL, history

MP3 = load_personality("mp3")
NOW = datetime(2026, 10, 4, 21, 30, tzinfo=UTC)


def _context(**overrides: object) -> PromptContext:
    fields: dict[str, object] = {
        "personality": MP3,
        "channel": GENERAL,
        "history": [
            history("anyone read meditations?", author_name="alice", minutes_ago=3),
            history("yeah, twice", is_self=True, minutes_ago=2),
            history("what did you think of it", author_name="alice", minutes_ago=1),
        ],
        "now": NOW,
        "target_user_id": 1,
        "target_user_name": "alice",
        "target_message": "what did you think of it",
    }
    fields.update(overrides)
    return PromptContext(**fields)  # type: ignore[arg-type]


def test_system_prompt_contains_personality_and_context() -> None:
    prompt = PromptBuilder().system_prompt(_context(partner_name="alice"))

    assert prompt.startswith(MP3.identity)
    assert MP3.traits[0] in prompt
    assert MP3.style.guidelines[0] in prompt
    assert GROUND_RULES[0] in prompt
    assert "#general on the server Test Server" in prompt
    assert "Channel topic: anything goes" in prompt
    assert "Sunday, 04 October 2026, 21:30 UTC" in prompt
    assert "mainly talking with alice" in prompt
    assert 'replying to alice: "what did you think of it"' in prompt


def test_honesty_rule_is_always_present() -> None:
    prompt = PromptBuilder().system_prompt(_context())
    assert "if someone sincerely asks whether you're a bot or an AI, be honest" in prompt


def test_user_memory_and_profile_are_included() -> None:
    now = utcnow()
    profile = UserProfile("mp3", 1, "alice", now, now, interaction_count=12)
    memory = MemoryRecord(1, "mp3", 1, MemoryKind.FACT, "studies classics", 0.8, now, now)
    prompt = PromptBuilder().system_prompt(_context(profile=profile, memories=[memory]))

    assert "talked with alice before (12 exchanges)" in prompt
    assert "- studies classics" in prompt


def test_history_roles_and_formatting() -> None:
    messages = PromptBuilder().build(_context())

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "alice: anyone read meditations?"
    assert messages[2].content == "yeah, twice"
    assert messages[-1].content == "alice: what did you think of it"


def test_consecutive_messages_are_merged_and_replies_annotated() -> None:
    context = _context(
        history=[
            history("hot take: plato was a poet", author_name="bob", minutes_ago=5),
            history("disagree", author_name="alice", minutes_ago=4, reply_to_author_name="bob"),
            history("first part", is_self=True, minutes_ago=3),
            history("second part", is_self=True, minutes_ago=3),
            history("wait what", author_name="alice", minutes_ago=1, reply_to_author_name="mp3"),
        ]
    )
    messages = PromptBuilder().build(context)

    assert messages[1].content == (
        "bob: hot take: plato was a poet\nalice (replying to bob): disagree"
    )
    assert messages[2].content == "first part\n\nsecond part"
    assert messages[3].content == "alice (replying to you): wait what"


def test_long_gaps_are_marked() -> None:
    context = _context(
        history=[
            history("old message", author_name="alice", minutes_ago=200),
            history("new message", author_name="alice", minutes_ago=1),
        ]
    )
    messages = PromptBuilder().build(context)
    assert "[3 hours later]\nalice: new message" in messages[1].content


def test_history_is_trimmed_to_budget() -> None:
    long_text = "x" * 2000
    many = [history(f"{i} {long_text}", minutes_ago=50 - i) for i in range(10)]
    messages = PromptBuilder().build(_context(history=many))
    included = "\n".join(m.content for m in messages[1:])
    assert "9 x" in included  # newest kept
    assert "0 x" not in included  # oldest dropped
    assert len(included) <= MP3.memory.history_char_budget + 2100


def test_initiate_mode_adds_topic_and_closing_turn() -> None:
    context = _context(
        mode="initiate",
        history=[history("gn everyone", author_name="bob", minutes_ago=180)],
        topic="whether math is discovered or invented",
        quiet_for=timedelta(hours=3),
        target_user_name=None,
        target_message=None,
    )
    messages = PromptBuilder().build(context)

    assert "Start a new conversation about whether math is discovered" in messages[0].content
    assert messages[-1].role == "user"
    assert messages[-1].content == "bob: gn everyone"

    empty = PromptBuilder().build(_context(mode="initiate", history=[], quiet_for=None))
    assert empty[-1].content == "[no new messages for a while]"


def test_dm_context_and_length_hints() -> None:
    prompt = PromptBuilder().system_prompt(_context(channel=DM, target_message="lol"))
    assert "private direct-message conversation with alice" in prompt
    assert "keep yours short" in prompt

    long_question = "so " + "what do you really think about free will " * 10 + "?"
    prompt = PromptBuilder().system_prompt(_context(target_message=long_question))
    assert "a few sentences is fine" in prompt


def test_build_always_ends_with_user_turn() -> None:
    context = _context(history=[history("my last word", is_self=True)])
    messages = PromptBuilder().build(context)
    assert messages[-1].role == "user"
