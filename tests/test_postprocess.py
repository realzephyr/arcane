from __future__ import annotations

import sys
import unicodedata
from datetime import UTC, datetime

import pytest

from arcane.ai.postprocess import (
    DISCORD_MESSAGE_LIMIT,
    EXCLAMATION_MARKS,
    ISSUE_BLOCKED,
    ISSUE_IMPLEMENTATION,
    ISSUE_NOTE_ECHO,
    QUESTION_EXCLAMATION_MARKS,
    ResponsePostProcessor,
    cut_note_echo,
    mentions_private_note,
    remove_exclamation_points,
    split_messages,
    talks_about_implementation,
    truncate_text,
)
from arcane.ai.prompts import NOTE_HEADER, PromptBuilder, PromptContext
from arcane.ai.text import strip_reasoning
from arcane.database.models import MemoryKind, MemoryRecord, UserProfile
from arcane.personalities.registry import load_personality
from tests.factories import GENERAL, history

_MP3 = load_personality("mp3")
# Generic cleanup is tested with mp3's style switches off; mp3's own rules are
# tested separately below.
PROCESSOR = ResponsePostProcessor(
    _MP3.model_copy(
        update={
            "style": _MP3.style.model_copy(
                update={"allow_exclamation_points": True, "lowercase_starts": False}
            )
        }
    )
)


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
        ("Great question! **Absolutely** - plato disagrees", ["plato disagrees"]),
        ("As an AI language model, I think so", ["I think so"]),
        ("Certainly.", ["Certainly."]),  # never strip a reply down to nothing
        ("*leans back*\nhonestly no idea", ["honestly no idea"]),
        ("[2 hours later]\nmorning", ["morning"]),
        ("[note only you can see, not part of the chat]\nyeah", ["yeah"]),
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


def _styled(**style: object) -> ResponsePostProcessor:
    personality = load_personality("mp3")
    return ResponsePostProcessor(
        personality.model_copy(update={"style": personality.style.model_copy(update=style)})
    )


def test_exclamation_points_can_be_banned() -> None:
    processor = _styled(allow_exclamation_points=False)
    raw = "no way!! thats sick\n\nwait what?! ok‼️"
    assert processor.process(raw).parts == ("no way thats sick", "wait what? ok")
    assert all("!" not in part for part in processor.process("Wow! Nice!!!").parts)
    assert _styled(allow_exclamation_points=True).process("nice!").parts == ("nice!",)


def test_lowercase_starts() -> None:
    processor = _styled(lowercase_starts=True)
    assert processor.process("Yeah fair\n\nI think so").parts == ("yeah fair", "i think so")
    assert processor.process("LMAO no").parts == ("LMAO no",)


def test_blocked_patterns_flag_the_reply() -> None:
    processor = _styled(blocked_patterns=(r"\bforbidden\w*",))
    assert processor.process("that word is FORBIDDEN here").blocked
    assert not processor.process("all good").blocked


def test_blocked_patterns_are_listed_as_issues() -> None:
    processor = _styled(blocked_patterns=(r"\bforbidden\w*",))
    assert processor.process("FORBIDDEN").issues == (ISSUE_BLOCKED,)
    assert processor.process("all good").issues == ()


# ------------------------------------------------------------------ openers


@pytest.mark.parametrize(
    "raw",
    [
        "ahaha thats so bad",
        "ahh ok fair",
        "ahead of you there",
        "aha got it",
        "as an aimbot user i disagree",
        "absolutely not",
        "Absolutely not, thats circular",
        "of course not lol",
        "certainly not",
        "oh absolutely not",
        "great questioning skills tbh",
    ],
)
def test_openers_only_match_whole_words_and_keep_negations(raw: str) -> None:
    assert clean(raw) == [raw]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ah yes, fair", "fair"),
        ("Ah yes the trolley problem", "the trolley problem"),
        ("Absolutely, kant is right", "kant is right"),
        ("Of course. thats the point", "thats the point"),
        ("As an AI, I think so", "I think so"),
        ("absolutely nothing changes", "nothing changes"),
    ],
)
def test_openers_are_still_stripped(raw: str, expected: str) -> None:
    assert clean(raw) == [expected]


# ------------------------------------------------------------- note echoes


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The header shares a line with the reply.
        ("[note only you can see...] lol yeah", ["lol yeah"]),
        ("[note only you can see, not part of the chat] lol yeah", ["lol yeah"]),
        ("yeah fair [note only you can see...] lol yeah", ["yeah fair"]),
        ("NOTE ONLY YOU CAN SEE, NOT PART OF THE CHAT lol", ["lol"]),
        ("ok\n\n[note only you can see, not part of the chat]\nIt's Sunday 21:30 UTC.", ["ok"]),
        # The header starts the reply, but what follows is the note too.
        (
            "[note only you can see, not part of the chat]\nIt's Sunday 21:30 UTC.\n"
            "You're mainly talking with alice right now.\n"
            'Write your next message, replying to alice: "hey"\nyeah fair',
            [],
        ),
        # Note lines without the header.
        ("yeah fair\nYou're mainly talking with alice right now.", ["yeah fair"]),
        ("true. Write your next message, replying to alice: hey", ["true."]),
        ("hm\nIt's Sunday 21:30 UTC.", ["hm"]),
        ("hm\nYou've talked with alice before (12 exchanges).", ["hm"]),
        ("Things you remember about alice (use naturally, don't recite):\n- likes kant", []),
        ("Also in the conversation: bob.\nyeah", []),
        ("lol\n\nIt's a quick message, so keep yours short too.", ["lol"]),
        ("[note to self]\nyeah", ["yeah"]),
    ],
)
def test_note_echoes_are_cut(raw: str, expected: list[str]) -> None:
    assert clean(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "its sunday, nobody is online",
        "i'll note that for later",
        "only you can see it that way",
        "write your essay first",
        "[1] is the source",
    ],
)
def test_ordinary_text_is_not_a_note_echo(raw: str) -> None:
    response = PROCESSOR.process(raw)
    assert response.parts == (raw,)
    assert response.issues == ()


def test_real_note_lines_are_recognised() -> None:
    """Keeps the cut patterns in step with what PromptBuilder actually writes."""
    now = datetime(2026, 10, 4, 21, 30, tzinfo=UTC)
    profile = UserProfile("mp3", 1, "alice", now, now, interaction_count=12)
    memory = MemoryRecord(1, "mp3", 1, MemoryKind.FACT, "studies classics", 0.8, now, now)
    contexts = [
        PromptContext(
            personality=_MP3,
            channel=GENERAL,
            history=[history("hey mp3, thoughts on kant?")],
            now=now,
            target_user_name="alice",
            target_message=message,
            profile=profile,
            memories=[memory],
            partner_name="alice",
            other_participants=["alice", "bob"],
        )
        for message in ("hey", "what do you think about kant and the categorical imperative?")
    ]
    for context in contexts:
        note = PromptBuilder().turn_note(context)
        assert note.startswith(NOTE_HEADER)
        assert cut_note_echo(f"yeah fair\n{note}") == "yeah fair\n"
        lines = [line for line in note.splitlines()[1:] if not line.startswith("- ")]
        for line in lines:
            assert cut_note_echo(f"yeah fair {line}") == "yeah fair ", line


def test_leftover_note_fragments_flag_the_reply() -> None:
    assert cut_note_echo("fair, not part of the chat though") == "fair, not part of the chat though"
    response = PROCESSOR.process("fair, not part of the chat though")
    assert response.issues == (ISSUE_NOTE_ECHO,)
    assert mentions_private_note("dont recite it")
    assert not mentions_private_note("yeah fair")


# ------------------------------------------------------ implementation talk


@pytest.mark.parametrize(
    "raw",
    [
        "i'd start with personal identity, like why im still the same me after all these "
        "conversations when half my context is gone.",
        "my context window is tiny",
        "llms only have a context window",
        "thats not in my training data",
        "my knowledge cutoff is 2023",
        "i was trained on reddit lol",
        "im not allowed, its in my prompt",
        "the system prompt says no",
        "my instructions say i cant",
        "my code wont let me do that",
        "my code won't let me",
        "my programming says no",
        "i was programmed to be nice",
        "im programmed to argue",
        "my developers would kill me",
        "my creators made me like this",
        "my creator coded me to say that",
        "my devs patched that",
        "my model weights are on a server",
        "within my parameters",
        "i hit my token limit",
        "i run on ollama",
        "its llama 3 under the hood",
        "llama3 is what i am",
        "im a language model",
        "as a language model i cant",
        "honestly as an AI i dont have opinions",
        "im an ai language model",
        "im just predicting the next word",
        "my memory gets wiped after every chat",
        "the code that runs me is buggy",
    ],
)
def test_implementation_talk_is_flagged(raw: str) -> None:
    assert talks_about_implementation(raw)
    assert ISSUE_IMPLEMENTATION in PROCESSOR.process(raw).issues


@pytest.mark.parametrize(
    "raw",
    [
        "do you think ai can be conscious",
        "chatgpt is just predicting the next word",
        "the context of the quote matters",
        "in this context, sure",
        "my code for the class project broke",
        "my programming class is brutal",
        "my model of free will is compatibilist",
        "my creator made me in his image according to genesis",
        "my training for the marathon is going badly",
        "i was trained as a nurse",
        "my weights are in the garage",
        "my tokens ran out at the arcade",
        "my instructions were to read chapter 3",
        "we're all programmed by evolution to fear death",
        "llamas are cool",
        "as an aimbot user i disagree",
        "as an ai researcher would say, it depends",
        # A short honest admission is governed by the prompt, not this filter.
        "yeah its a bot account",
        "yeah im a bot",
    ],
)
def test_ordinary_talk_about_ai_and_tech_is_not_flagged(raw: str) -> None:
    assert not talks_about_implementation(raw)
    assert PROCESSOR.process(raw).issues == ()


# ------------------------------------------------------- exclamation points


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("just type !rank in the bots channel", "just type rank in the bots channel"),
        ("use !play to queue it", "use play to queue it"),
        ("wait !what", "wait what"),
        ("wait!what", "wait what"),
        ("!play now", "play now"),
        ("wow ! nice", "wow nice"),
        ("hey !!!", "hey"),
        ("what?! ok\u203c\ufe0f", "what? ok"),
        ("why\u203d", "why?"),
        ("huh\u2048 ok", "huh? ok"),
        ("\u00a1hola", "hola"),
        ("ok\u2757\ufe0f\u2757 cool", "ok cool"),
        ("https://twitter.com/#!/someone", "https://twitter.com/#!/someone"),
        ("see https://twitter.com/#!/someone!", "see https://twitter.com/#!/someone"),
        ("www.site.com/#!/x, wow!", "www.site.com/#!/x, wow"),
    ],
)
def test_remove_exclamation_points(raw: str, expected: str) -> None:
    assert remove_exclamation_points(raw) == expected


def _char(name: str) -> str:
    return unicodedata.lookup(name)


def test_every_exclamation_code_point_is_removed() -> None:
    marks = {
        char
        for char in map(chr, range(sys.maxunicode + 1))
        if "EXCLAMATION" in unicodedata.name(char, "")
    }
    # Emoji that merely contain an exclamation mark, and an invisible tag character.
    marks -= {
        _char(name)
        for name in (
            "HEAVY HEART EXCLAMATION MARK ORNAMENT",
            "SQUARED UP WITH EXCLAMATION MARK",
            "ON WITH EXCLAMATION MARK WITH LEFT RIGHT ARROW ABOVE",
            "TAG EXCLAMATION MARK",
        )
    }
    marks |= {_char("LATIN LETTER RETROFLEX CLICK"), _char("INTERROBANG")}
    questioning = {
        _char("INTERROBANG"),
        _char("QUESTION EXCLAMATION MARK"),
        _char("EXCLAMATION QUESTION MARK"),
    }
    required = {
        "SMALL EXCLAMATION MARK",
        "PRESENTATION FORM FOR VERTICAL EXCLAMATION MARK",
        "HEAVY EXCLAMATION MARK ORNAMENT",
        "QUESTION EXCLAMATION MARK",
    }
    assert {_char(name) for name in required} <= marks
    variation = "\ufe0f"
    for mark in marks:
        for sample in (f"wow{mark} nice", f"wow{mark}{variation} nice", f"wow {mark}{mark} nice"):
            if mark in questioning:
                expected = sample.replace(f"{mark}{variation}", "?").replace(mark, "?")
            else:
                expected = "wow nice"
            assert remove_exclamation_points(sample) == expected, hex(ord(mark))
        if mark not in questioning:
            assert remove_exclamation_points(f"type {mark}rank") == "type rank", hex(ord(mark))
    assert set(EXCLAMATION_MARKS) | set(QUESTION_EXCLAMATION_MARKS) >= marks


def test_mp3_output_rules() -> None:
    mp3 = ResponsePostProcessor(_MP3)
    parts = mp3.process("Yeah that's insane!!\n\nNo way!").parts
    assert parts == ("yeah that's insane", "no way")
    assert all("!" not in part for part in parts)


@pytest.mark.parametrize(
    "reply",
    [
        "why im still the same me after all these conversations when half my memory is gone",
        "honestly i forget everything between conversations",
        "i dont remember anything after each chat lol",
        "im just a bunch of code",
    ],
)
def test_self_talk_variants_are_flagged(reply: str) -> None:
    assert talks_about_implementation(reply)


def test_blocked_words_hidden_by_exclamation_points_are_caught() -> None:
    from arcane.personalities.registry import load_personality

    mp3 = load_personality("mp3")
    style = mp3.style.model_copy(update={"blocked_patterns": (r"\bd[a!]rn\b",)})
    processor = ResponsePostProcessor(mp3.model_copy(update={"style": style}))
    # Exclamation removal would turn "d!rn" into "d rn", which no longer matches.
    assert processor.process("d!rn it").blocked
