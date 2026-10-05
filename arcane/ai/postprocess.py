"""Turning raw model output into messages that can be sent to Discord as-is.

Models drift from instructions in predictable ways: they think out loud in
``<think>`` tags, prefix their name, wrap replies in quotes, reach for
markdown, open with assistant clichés, or keep going and write the other
people's lines too. :class:`ResponsePostProcessor` corrects all of that, makes
mass mentions harmless, applies the personality's style switches (no
exclamation points, lowercase message starts), enforces length limits, splits
the reply into the separate messages the model indicated with blank lines, and
reports problems that cleaning can't fix (blocked patterns, an echo of the
private per-turn note, talk about the bot's own implementation) so the reply
can be regenerated.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from arcane.ai.text import strip_reasoning
from arcane.personalities.base import Personality

DISCORD_MESSAGE_LIMIT = 2000
_MAX_CHAINED_OPENERS = 3
_ZERO_WIDTH_SPACE = "​"

DEFAULT_BANNED_OPENERS: tuple[str, ...] = (
    "as an ai language model",
    "as an ai",
    "great question",
    "good question",
    "what a great question",
    "what a fascinating question",
    "certainly",
    "absolutely",
    "of course",
    "sure thing",
    "i'd be happy to help",
    "i'd be happy to",
    "i'm happy to help",
)

_NEGATABLE_OPENERS = frozenset({"absolutely", "of course", "certainly"})
_GENERIC_SPEAKERS = ("assistant", "ai", "bot", "system")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?", re.MULTILINE)
_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_GAP_MARKER_RE = re.compile(r"^\s*\[[^\]\n]{0,40}\blater\]\s*$", re.MULTILINE | re.IGNORECASE)
_STAGE_DIRECTION_RE = re.compile(r"^\s*\*[^*\n]{1,60}\*\s*$", re.MULTILINE)
_TRAILING_SPACE_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+|\n")
_MASS_MENTION_RE = re.compile(r"@(everyone|here)\b", re.IGNORECASE)
_TITLE_WORD_START_RE = re.compile(r"^([A-Z])(?=[a-z'\u2019]*\b)(?![A-Z])")
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"))

# Problems reported in ProcessedResponse.issues.
ISSUE_BLOCKED = "contains a blocked pattern"
ISSUE_NOTE_ECHO = "echoes its private note"
ISSUE_IMPLEMENTATION = "talks about how it works"

# ------------------------------------------------------------- note echoes
# The private per-turn note (arcane.ai.prompts.PromptBuilder.turn_note) starts
# with NOTE_HEADER and holds lines like "It's Sunday 21:30 UTC." Small models
# sometimes copy it into the reply; everything from the echo on is dropped.

_NOTE_HEADER_RE = re.compile(
    # The bracketed header, also cut short ("[note only you can see...]"), or the
    # same words without brackets. A lone "[note ...]" line counts too.
    r"\[?[ \t]*note only you can see(?:[^\]\n]{0,60}\]|,? not part of the chat\b)?"
    r"|^[ \t]*\[[ \t]*note\b[^\]\n]{0,60}\][ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_NOTE_PHRASE_RE = re.compile(
    "|".join(
        (
            r"\bit[\u2019']?s,? \w{3,9}day\b[^\n]{0,30}?\b\d{1,2}:\d{2}\s*utc\b",
            r"\byou[\u2019']?re mainly talking with\b",
            r"\balso in the conversation:",
            r"\byou[\u2019']?ve talked with\b[^\n]{0,40}?\bbefore\s*\(\d{1,6} exchanges?\)",
            r"\(\d{1,6} exchanges?\)",
            r"\bthings you remember about\b[^\n]{0,60}?\(use naturally",
            r"\(use naturally, don[\u2019']?t recite\)",
            r"\bwrite your next message\b",
            r"\bnobody has said anything\b[^\n]{0,60}?\bstart a new conversation\b",
            r"\bso say no casually with a quick excuse\b",
            r"\bit[\u2019']?s a quick message, so keep yours short\b",
            r"\bkeep it conversational: usually a sentence or two\b",
            r"\bthey put thought into it, so a few sentences is fine\b",
        )
    ),
    re.IGNORECASE,
)
# Fragments too ambiguous to cut at, but that should never reach the chat.
_NOTE_FRAGMENT_RE = re.compile(
    r"\bnot part of the chat\b|\bdon[\u2019']?t recite\b|\buse naturally,", re.IGNORECASE
)

# ------------------------------------------------- implementation talk
# The bot must never talk about how it works: its context, training, prompt,
# code or the model behind it. Patterns are first-person on purpose, so debate
# talk about AI in general ("do you think ai can be conscious", "chatgpt just
# predicts the next word") passes. The exceptions are jargon a regular in a
# philosophy chat almost never uses ("context window", "system prompt",
# "ollama"); a false positive there only costs a regeneration. A plain
# admission ("yeah im a bot") is not implementation talk and is left to the
# prompt's honesty rule.
#
# Words people also use about their own lives ("my code", "my model", "my
# instructions", "my creator", "my training", "my tokens", "my weights") are
# flagged only next to wording about being controlled or produced by them ("my
# code won't let me", "my instructions say i", "my training data"). The
# trade-off: a bare "my training" or "my code is buggy" from the bot slips
# through, and a student's "my code was written in python" gets regenerated,
# while "my code for the class project broke", "my model of free will", "my
# creator made me in his image" and "my training for the marathon" are fine.

_CONTROL_VERBS = (
    r"(?:tells?\s+me|told\s+me|says?\s+(?:i|to)|said\s+(?:i|to)"
    r"|won[\u2019']?t\s+let\s+me|doesn[\u2019']?t\s+let\s+me|don[\u2019']?t\s+let\s+me|lets?\s+me"
    r"|makes?\s+me\s+(?:say|talk|answer|reply|respond)|forces?\s+me|requires?\s+me"
    r"|stops?\s+me|keeps?\s+me\s+from|limits?\s+me|prevents?\s+me|forbids?\s+me"
    r"|allows?\s+me|(?:is|was|were|got|been)\s+(?:written|coded|programmed|trained|set\s+up)"
    r"|runs?\s+me|running\s+me)"
)
_IMPLEMENTATION_RE = re.compile(
    "|".join(
        (
            # Context and memory.
            r"\bmy\s+context\b",
            r"\bcontext[\s-]+window\b",
            r"\bmy\s+memory\s+(?:gets?|got|is|was)\s+(?:wiped|reset|cleared|erased|truncated)\b",
            r"\bmy\s+(?:memory|context)\s+resets\b",
            # Forgetting between conversations, the giveaway the owner quoted.
            r"\b(?:half\s+)?my\s+memor(?:y|ies)\s+(?:is|are|gets?|got)\s+(?:gone|deleted|wiped"
            r"|erased)\b",
            r"\bi\s+(?:forget|don[\u2019']?t\s+remember|lose)\b[^.!?\n]{0,40}?\b(?:between|after"
            r"|every|each)\s+(?:convos?|conversations?|chats?|sessions?)\b",
            r"\bafter\s+all\s+these\s+(?:conversations|convos|chats)\b[^.!?\n]{0,40}?\bgone\b",
            r"\bi[\u2019']?m\s+(?:just\s+)?(?:a\s+bunch\s+of\s+)?(?:code|lines\s+of\s+code"
            r"|software|an?\s+program)\b",
            # Training.
            r"\bmy\s+(?:training|knowledge)\s+(?:data|set|cut-?off)\b",
            r"\bknowledge\s+cut-?off\b",
            r"\bi(?:\s+was|\s+am|[\u2019']?m|\s+got|[\u2019']?ve\s+been|\s+have\s+been|\s+wasn[\u2019']?t)"
            r"\s+(?:pre-?|fine-?)?trained\s+(?:on|with)\b",
            # Prompts and instructions.
            r"\bmy\s+(?:system\s+)?prompts?\b",
            r"\bsystem\s+prompt\b",
            # Programming.
            r"\bi(?:\s+was|\s+am|[\u2019']?m|\s+got|[\u2019']?ve\s+been|\s+have\s+been)"
            r"\s+(?:just\s+)?(?:programmed|coded)\b",
            r"\bmy\s+programming\b(?!\s+(?:class|course|homework|assignment|project|skills?"
            r"|language|job|teacher|exam))",
            r"\b(?:code|software|program|model|server|script)\s+(?:that\s+)?(?:runs|powers|controls)"
            r"\s+me\b",
            # Who built it and what it runs on.
            r"\bmy\s+(?:developers|devs|dev\s+team|creators|makers|programmers)\b",
            r"\bmy\s+developer\b",
            r"\bmy\s+creators?\b[^.!?\n]{0,20}?\b(?:coded|programmed|trained|built)\s+me\b",
            r"\bmy\s+(?:parameters|model\s+weights|weights\s+and\s+biases|neural\s+net(?:work)?)\b",
            r"\bmy\s+(?:token|context)\s+(?:limit|budget|count)\b",
            r"\bmy\s+(?:code|programming|instructions|guidelines|model|weights|tokens?|creator"
            r"|training)\b[^.!?\n]{0,30}?\b" + _CONTROL_VERBS,
            r"\bollama\b",
            r"\bllama[\s-]?3(?:\.\d)?\b",
            # Assistant self-descriptions.
            r"\b(?:i[\u2019']?m|i\s+am|as)\s+(?:just\s+)?(?:an?\s+)?(?:large\s+)?language\s+model\b",
            r"\b(?:i[\u2019']?m|i\s+am|as)\s+(?:just\s+)?an?\s+(?:llm|ai\s+(?:language\s+)?model)\b",
            r"\bas\s+an\s+ai\b(?!\s+(?:researcher|engineer|ethicist|skeptic|sceptic|doomer"
            r"|optimist|nerd|student|enthusiast|developer|dev|artist|company|guy|person))",
            r"\bi[\u2019']?m\s+(?:just\s+)?predicting\s+(?:the\s+)?next\s+(?:word|token)",
        )
    ),
    re.IGNORECASE,
)

# ------------------------------------------------------ exclamation points
# Every code point whose Unicode name mentions an exclamation mark (or an
# interrobang), plus U+01C3, a click letter that looks exactly like "!".
# Excluded: emoji that merely contain one (❣ heart, 🆙, 🔛) and the invisible
# tag character used inside flag emoji sequences.
_NOT_EXCLAMATIONS = frozenset("\u2763\U0001f199\U0001f51b\U000e0021")


def _exclamation_chars() -> tuple[str, str]:
    """Return (marks to remove, marks that also ask a question)."""
    remove: list[str] = ["\u01c3"]
    questioning: list[str] = []
    for char in map(chr, range(sys.maxunicode + 1)):
        name = unicodedata.name(char, "")
        if char in _NOT_EXCLAMATIONS or ("EXCLAMATION" not in name and "INTERROBANG" not in name):
            continue
        if ("QUESTION" in name or "INTERROBANG" in name) and "INVERTED" not in name:
            questioning.append(char)
        else:
            remove.append(char)
    return "".join(remove), "".join(questioning)


EXCLAMATION_MARKS, QUESTION_EXCLAMATION_MARKS = _exclamation_chars()
"""Characters :func:`remove_exclamation_points` deletes, and those it turns into "?"."""

_EMOJI_PRESENTATION = "\ufe0f"
_MARK = "[" + "".join(re.escape(c) for c in EXCLAMATION_MARKS) + "]"
_QUESTION_MARK_RE = re.compile(
    "[" + "".join(re.escape(c) for c in QUESTION_EXCLAMATION_MARKS) + "]\ufe0f?"
)
_MARKS = rf"(?:{_MARK}{_EMOJI_PRESENTATION}?)+"
_EXCLAMATION_BETWEEN_WORDS_RE = re.compile(rf"(?<=\w){_MARKS}(?=\w)")
# Spaces before a mark go only when the mark ends a word ("wow !" -> "wow"), not
# when it starts one ("type !rank" -> "type rank").
_SPACE_BEFORE_FINAL_MARK_RE = re.compile(rf"[ \t]+(?={_MARKS}(?!\w))")
_EXCLAMATION_RE = re.compile(_MARKS)
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# A URL ends before trailing punctuation ("see https://x.com!" keeps only "!" out).
_URL_RE = re.compile(r"(?:https?://|www\.)\S*?(?=[.,;:)?\]" + _MARK[1:-1] + r"]*(?:\s|$))")
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"))


@dataclass(frozen=True, slots=True)
class ProcessedResponse:
    """Cleaned output, already split into Discord messages."""

    parts: tuple[str, ...]
    issues: tuple[str, ...] = ()
    """Why the reply shouldn't be sent as-is (e.g. :data:`ISSUE_BLOCKED`), most serious first."""

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)

    @property
    def blocked(self) -> bool:
        """True when the reply matches one of the personality's blocked patterns."""
        return ISSUE_BLOCKED in self.issues

    @property
    def is_empty(self) -> bool:
        return not any(part.strip() for part in self.parts)


class ResponsePostProcessor:
    """Cleans and splits model output for one personality."""

    def __init__(self, personality: Personality) -> None:
        self._personality = personality
        self._max_chars = personality.style.max_response_chars
        self._max_parts = personality.style.max_messages_per_reply
        openers = (*personality.style.banned_openers, *DEFAULT_BANNED_OPENERS)
        # Longest first so "as an ai language model" wins over "as an ai".
        alternatives = "|".join(
            _opener_pattern(opener) for opener in sorted(openers, key=len, reverse=True)
        )
        # (?!\w): a whole opener only, so "ah" leaves "ahaha" and "as an ai" "as an aimbot".
        self._opener_re = re.compile(
            rf"^\s*(?:{alternatives})(?!\w)\s*[,.!:;—-]*\s*", re.IGNORECASE
        )
        self._own_names = (*personality.names, *_GENERIC_SPEAKERS)
        style = personality.style
        self._allow_exclamations = style.allow_exclamation_points
        self._lowercase_starts = style.lowercase_starts
        self._blocked = tuple(re.compile(p, re.IGNORECASE) for p in style.blocked_patterns)

    def process(self, raw: str, *, other_speakers: Iterable[str] = ()) -> ProcessedResponse:
        """Clean ``raw`` model output.

        Args:
            raw: The model's output.
            other_speakers: Names of other people in the conversation. If the
                model starts writing lines for them, the reply is cut there.
        """
        text = strip_reasoning(raw.replace("\r\n", "\n"))
        text = cut_note_echo(text)
        text = self._cut_hallucinated_turns(text, other_speakers)
        text = self._strip_speaker_labels(text)
        text = _strip_wrapping_quotes(text)
        text = _GAP_MARKER_RE.sub("", text)
        text = _strip_markdown(text)
        text = self._strip_openers(text)
        before_cleanup = text
        if not self._allow_exclamations:
            text = remove_exclamation_points(text)
        text = _MASS_MENTION_RE.sub(lambda m: f"@{_ZERO_WIDTH_SPACE}{m.group(1)}", text)
        text = _TRAILING_SPACE_RE.sub("", text)
        text = _MANY_NEWLINES_RE.sub("\n\n", text).strip()
        text = truncate_text(text, self._max_chars)
        parts = split_messages(text, self._max_parts)
        if self._lowercase_starts:
            parts = [lowercase_start(part) for part in parts]
        issues = self._issues("\n\n".join(parts), before_cleanup)
        return ProcessedResponse(tuple(parts), issues=issues)

    # ---------------------------------------------------------------- internals

    def _issues(self, text: str, before_cleanup: str = "") -> tuple[str, ...]:
        issues: list[str] = []
        # Blocked words are checked before exclamation points are removed too, so
        # "f!ck" can't turn into "f ck" and slip past.
        if any(pattern.search(t) for pattern in self._blocked for t in (text, before_cleanup)):
            issues.append(ISSUE_BLOCKED)
        if mentions_private_note(text):
            issues.append(ISSUE_NOTE_ECHO)
        if talks_about_implementation(text):
            issues.append(ISSUE_IMPLEMENTATION)
        return tuple(issues)

    def _strip_speaker_labels(self, text: str) -> str:
        names = "|".join(re.escape(name) for name in self._own_names)
        label = re.compile(rf"^\s*\**(?:{names})\**\s*:\s*", re.IGNORECASE | re.MULTILINE)
        return label.sub("", text)

    def _cut_hallucinated_turns(self, text: str, other_speakers: Iterable[str]) -> str:
        names = [n for n in {s.strip() for s in other_speakers} if n]
        own = {name.lower() for name in self._own_names}
        names = [n for n in names if n.lower() not in own]
        if not names:
            return text
        alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
        turn = re.compile(rf"^\s*\**(?:{alternatives})\**\s*(?:\([^)\n]*\))?\s*:", re.I | re.M)
        match = turn.search(text)
        if match is None:
            return text
        if match.start() == 0:
            # The whole reply is written as someone else; drop the label only.
            return text[match.end() :]
        return text[: match.start()]

    def _strip_openers(self, text: str) -> str:
        # Openers often come chained ("Great question! Absolutely, ...").
        for _ in range(_MAX_CHAINED_OPENERS):
            stripped = self._opener_re.sub("", text, count=1)
            if stripped == text or not stripped.strip():
                break
            text = stripped
        return text


def _opener_pattern(opener: str) -> str:
    phrase = opener.strip().rstrip(",.!").lower()
    pattern = re.escape(phrase)
    if any(phrase == word or phrase.endswith(f" {word}") for word in _NEGATABLE_OPENERS):
        # "absolutely not" / "of course not" is the answer, not filler.
        pattern += r"(?!\s*,?\s*not\b)"
    return pattern


def cut_note_echo(text: str) -> str:
    """Drop an echo of the private per-turn note and everything after it.

    A reply that *starts* with the note header keeps what follows the header,
    unless that also comes from the note.
    """
    header = _NOTE_HEADER_RE.search(text)
    if header is not None:
        if text[: header.start()].strip():
            text = text[: header.start()]
        else:
            rest = text[header.end() :]
            text = "" if _NOTE_PHRASE_RE.search(rest) else rest
    phrase = _NOTE_PHRASE_RE.search(text)
    return text if phrase is None else text[: phrase.start()]


def mentions_private_note(text: str) -> bool:
    """True if ``text`` contains wording from the private per-turn note."""
    return any(
        pattern.search(text) for pattern in (_NOTE_HEADER_RE, _NOTE_PHRASE_RE, _NOTE_FRAGMENT_RE)
    )


def talks_about_implementation(text: str) -> bool:
    """True if ``text`` talks about the bot's own implementation ("my context window")."""
    return _IMPLEMENTATION_RE.search(text) is not None


def remove_exclamation_points(text: str) -> str:
    """Remove every exclamation point ("wow!" -> "wow", "what?!" -> "what?").

    Interrobangs become "?", a mark between words becomes a space ("wait!what"),
    a mark that starts a word just goes ("type !rank" -> "type rank"), and URLs
    are left alone.
    """
    pieces: list[str] = []
    position = 0
    for url in _URL_RE.finditer(text):
        pieces.append(_remove_exclamations(text[position : url.start()]))
        pieces.append(url.group())
        position = url.end()
    pieces.append(_remove_exclamations(text[position:]))
    return "".join(pieces)


def _remove_exclamations(text: str) -> str:
    text = _QUESTION_MARK_RE.sub("?", text)
    text = _EXCLAMATION_BETWEEN_WORDS_RE.sub(" ", text)
    text = _SPACE_BEFORE_FINAL_MARK_RE.sub("", text)
    text = _EXCLAMATION_RE.sub("", text)
    return _MULTI_SPACE_RE.sub(" ", text)


def lowercase_start(text: str) -> str:
    """Lowercase a capitalised first word ("Yeah" -> "yeah", "I'm" -> "i'm").

    All-caps words ("LMAO", "NASA") are left alone.
    """
    return _TITLE_WORD_START_RE.sub(lambda m: m.group(1).lower(), text, count=1)


def _strip_wrapping_quotes(text: str) -> str:
    stripped = text.strip()
    for opening, closing in _QUOTE_PAIRS:
        if len(stripped) < 2 or not (stripped.startswith(opening) and stripped.endswith(closing)):
            continue
        # Only strip a single wrapping pair, never quotes that belong to the content.
        wraps_once = (
            stripped.count(opening) == 2
            if opening == closing
            else stripped.count(opening) == 1 and stripped.count(closing) == 1
        )
        if wraps_once:
            return stripped[1:-1].strip()
    return stripped


def _strip_markdown(text: str) -> str:
    text = _HEADER_RE.sub("", text)
    text = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _RULE_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    return _STAGE_DIRECTION_RE.sub("", text)


def truncate_text(text: str, max_chars: int) -> str:
    """Shorten ``text`` to ``max_chars``, preferring a sentence boundary."""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    boundaries = [m.start() for m in _SENTENCE_END_RE.finditer(window)]
    cut = max((b for b in boundaries if b >= max_chars // 2), default=None)
    if cut is not None:
        return window[:cut].rstrip()
    space = window.rfind(" ")
    return (window[:space] if space > max_chars // 2 else window).rstrip() + "…"


def split_messages(text: str, max_parts: int, limit: int = DISCORD_MESSAGE_LIMIT) -> list[str]:
    """Split on blank lines into at most ``max_parts`` messages under ``limit`` chars."""
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]
    if not paragraphs:
        return []
    if len(paragraphs) > max_parts:
        head, tail = paragraphs[: max_parts - 1], paragraphs[max_parts - 1 :]
        paragraphs = [*head, "\n".join(tail)]

    parts: list[str] = []
    for paragraph in paragraphs:
        parts.extend(_chunk(paragraph, limit))
    return parts


def _chunk(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        piece = truncate_text(remaining, limit).rstrip("…")
        if not piece:
            piece = remaining[:limit]
        chunks.append(piece.strip())
        remaining = remaining[len(piece) :].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
