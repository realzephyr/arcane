from __future__ import annotations

from datetime import UTC, datetime, timedelta

from arcane.ai.prompts import GROUND_RULES, NOTE_HEADER, PromptBuilder, PromptContext
from arcane.core.clock import utcnow
from arcane.database.models import MemoryKind, MemoryRecord, UserProfile
from arcane.personalities.registry import load_personality
from tests.factories import DM, GENERAL, history

MP3 = load_personality("mp3")
NOW = datetime(2026, 10, 4, 21, 30, tzinfo=UTC)

BASE_HISTORY = [
    history("anyone read meditations?", author_name="alice", minutes_ago=3),
    history("yeah, twice", is_self=True, minutes_ago=2),
    history("what did you think of it", author_name="alice", minutes_ago=1),
]


def _context(**overrides: object) -> PromptContext:
    fields: dict[str, object] = {
        "personality": MP3,
        "channel": GENERAL,
        "history": BASE_HISTORY,
        "now": NOW,
        "target_user_id": 1,
        "target_user_name": "alice",
        "target_message": "what did you think of it",
    }
    fields.update(overrides)
    return PromptContext(**fields)  # type: ignore[arg-type]


def _note(context: PromptContext) -> str:
    return PromptBuilder().build(context)[-1].content


def test_system_prompt_contains_personality_rules_and_place() -> None:
    prompt = PromptBuilder().system_prompt(_context())

    assert prompt.startswith(MP3.identity)
    assert MP3.traits[0] in prompt
    assert MP3.style.guidelines[0] in prompt
    assert all(rule in prompt for rule in GROUND_RULES)
    assert "#general on the server Test Server" in prompt
    assert "Channel topic: anything goes" in prompt


def test_system_prompt_is_static_across_turns() -> None:
    """Nothing per-turn may leak into the system prompt, or the model cache breaks."""
    now = utcnow()
    profile = UserProfile("mp3", 1, "alice", now, now, interaction_count=12)
    memory = MemoryRecord(1, "mp3", 1, MemoryKind.FACT, "studies classics", 0.8, now, now)
    first = PromptBuilder().system_prompt(_context())
    later = PromptBuilder().system_prompt(
        _context(
            now=NOW + timedelta(hours=5),
            target_message="something else entirely",
            partner_name="alice",
            other_participants=["bob"],
            profile=profile,
            memories=[memory],
        )
    )
    assert first == later
    assert "21:30" not in first


def test_ground_rules_cover_honesty_limits_and_safety() -> None:
    rules = " ".join(GROUND_RULES)
    assert "if someone sincerely asks whether you're a bot or an AI, be honest" in rules
    assert "can't join voice channels or calls" in rules
    assert "Never agree" in rules
    assert "family friendly" in rules
    assert "no flirting" in rules


def test_turn_note_carries_everything_that_changes() -> None:
    now = utcnow()
    profile = UserProfile("mp3", 1, "alice", now, now, interaction_count=12)
    memory = MemoryRecord(1, "mp3", 1, MemoryKind.FACT, "studies classics", 0.8, now, now)
    note = _note(_context(partner_name="alice", profile=profile, memories=[memory]))

    assert note.startswith("what did you think of it") is False
    assert NOTE_HEADER in note
    assert "Sunday 21:30 UTC" in note
    assert "mainly talking with alice" in note
    assert "talked with alice before (12 exchanges)" in note
    assert "- studies classics" in note
    assert 'replying to alice: "what did you think of it"' in note


def test_note_is_appended_to_the_last_user_turn() -> None:
    messages = PromptBuilder().build(_context())

    assert [m.role for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "alice: anyone read meditations?"
    assert messages[2].content == "yeah, twice"
    last = messages[-1].content
    assert last.startswith("alice: what did you think of it\n\n" + NOTE_HEADER)


def test_consecutive_turns_share_a_long_prefix() -> None:
    """Turn N+1 must repeat turn N's messages verbatim, except the one with the note."""
    first = PromptBuilder().build(_context())
    grown = [
        *BASE_HISTORY,
        history("honestly kinda mid", is_self=True),
        history("bro its a classic", author_name="alice"),
    ]
    second = PromptBuilder().build(_context(history=grown, target_message="bro its a classic"))

    assert second[: len(first) - 1] == first[:-1]
    assert second[len(first) - 1].content == "alice: what did you think of it"


def test_capability_guard_adds_instruction() -> None:
    note = _note(_context(target_message="yo hop in vc rn"))
    assert "They're asking you to hop on a voice call" in note
    assert "Don't agree" in note
    plain = _note(_context(target_message="vc was so laggy yesterday"))
    assert "They're asking you to" not in plain
    # Addressing the bot by name in the middle of the lead-in still counts.
    named = _note(_context(target_message="yo mp3 wanna hop in vc"))
    assert "They're asking you to hop on a voice call" in named


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
    assert messages[3].content.startswith("alice (replying to you): wait what")


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


def test_initiate_mode_note() -> None:
    context = _context(
        mode="initiate",
        history=[history("gn everyone", author_name="bob", minutes_ago=180)],
        topic="whether math is discovered or invented",
        quiet_for=timedelta(hours=3),
        target_user_name=None,
        target_message=None,
    )
    messages = PromptBuilder().build(context)

    assert messages[-1].role == "user"
    assert messages[-1].content.startswith("bob: gn everyone\n\n" + NOTE_HEADER)
    assert "Nobody has said anything for 3 hours" in messages[-1].content
    assert "Start a new conversation about whether math is discovered" in messages[-1].content

    empty = PromptBuilder().build(_context(mode="initiate", history=[], quiet_for=None))
    assert empty[-1].role == "user"
    assert empty[-1].content.startswith(NOTE_HEADER)
    assert "for a while" in empty[-1].content


def test_dm_context_and_length_hints() -> None:
    prompt = PromptBuilder().system_prompt(_context(channel=DM, target_message="lol"))
    assert "private direct-message conversation with alice" in prompt
    assert "keep yours short" in _note(_context(channel=DM, target_message="lol"))

    long_question = "so " + "what do you really think about free will " * 10 + "?"
    assert "a few sentences is fine" in _note(_context(target_message=long_question))


def test_build_always_ends_with_user_turn() -> None:
    context = _context(history=[history("my last word", is_self=True)])
    messages = PromptBuilder().build(context)
    assert messages[-1].role == "user"
    assert messages[-1].content.startswith(NOTE_HEADER)
