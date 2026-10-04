from __future__ import annotations

import pytest

from arcane.ai.postprocess import (
    DISCORD_MESSAGE_LIMIT,
    ResponsePostProcessor,
    split_messages,
    truncate_text,
)
from arcane.ai.text import strip_reasoning
from arcane.personalities.registry import load_personality

PROCESSOR = ResponsePostProcessor(load_personality("mp3"))


def clean(raw: str, **kwargs: object) -> list[str]:
    return list(PROCESSOR.process(raw, **kwargs).parts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<think>let me think</think>yeah fair", ["yeah fair"]),
        ("<think>never closed", []),
        ("reasoning...</think>actual reply", ["actual reply"]),
        ("mp3: hey", ["hey"]),
        ("**mp3**: hey", ["hey"]),
        ("Assistant: hey", ["hey"]),
        ('"quoted reply"', ["quoted reply"]),
        ('he said "hi" and "bye"', ['he said "hi" and "bye"']),
        ("## Thoughts\n**free will** is tricky", ["Thoughts\nfree will is tricky"]),
        ("- one\n- two", ["one\ntwo"]),
        ("1. first\n2. second", ["first\nsecond"]),
        ("Great question! i think it's both", ["i think it's both"]),
        ("As an AI language model, I think so", ["I think so"]),
        ("Certainly.", ["Certainly."]),  # never strip a reply down to nothing
        ("*leans back*\nhonestly no idea", ["honestly no idea"]),
        ("[2 hours later]\nmorning", ["morning"]),
    ],
)
def test_cleanup(raw: str, expected: list[str]) -> None:
    assert clean(raw) == expected


def test_mass_mentions_are_neutralised() -> None:
    (part,) = clean("hey @everyone and @here look")
    assert "@everyone" not in part
    assert "@here" not in part
    assert "@​everyone" in part


def test_blank_lines_split_messages_up_to_limit() -> None:
    assert clean("first\n\nsecond") == ["first", "second"]
    assert clean("a\n\nb\n\nc\n\nd\n\ne") == ["a", "b", "c\nd\ne"]


def test_hallucinated_turns_are_cut() -> None:
    raw = "i think hume wins this one\nalice: no way\nmp3: yes way"
    assert clean(raw, other_speakers=["alice"]) == ["i think hume wins this one"]
    assert clean("alice: lol", other_speakers=["alice"]) == ["lol"]
    # The bot's own name is never treated as another speaker.
    assert clean("mp3: ok", other_speakers=["mp3"]) == ["ok"]


def test_length_is_capped_at_sentence_boundary() -> None:
    sentence = "this is a sentence about philosophy. "
    (part,) = clean(sentence * 60)
    assert len(part) <= load_personality("mp3").style.max_response_chars
    assert part.endswith(".")


def test_truncate_and_split_helpers() -> None:
    assert truncate_text("short", 10) == "short"
    assert truncate_text("one two three four five", 12).endswith("…")
    huge = ("word " * 1000).strip()
    chunks = split_messages(huge, max_parts=1, limit=DISCORD_MESSAGE_LIMIT)
    assert all(len(chunk) <= DISCORD_MESSAGE_LIMIT for chunk in chunks)
    assert " ".join(chunks).replace("…", "").split() == huge.split()
    assert split_messages("   ", 3) == []


def test_strip_reasoning_variants() -> None:
    assert strip_reasoning("<thinking>x</thinking>  y") == "y"
    assert strip_reasoning("plain") == "plain"
