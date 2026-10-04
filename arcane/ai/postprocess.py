"""Turning raw model output into messages that can be sent to Discord as-is.

Models drift from instructions in predictable ways: they think out loud in
``<think>`` tags, prefix their name, wrap replies in quotes, reach for
markdown, open with assistant clichés, or keep going and write the other
people's lines too. :class:`ResponsePostProcessor` corrects all of that, makes
mass mentions harmless, enforces length limits, and splits the reply into the
separate messages the model indicated with blank lines.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from arcane.ai.text import strip_reasoning
from arcane.personalities.base import Personality

DISCORD_MESSAGE_LIMIT = 2000
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
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"))


@dataclass(frozen=True, slots=True)
class ProcessedResponse:
    """Cleaned output, already split into Discord messages."""

    parts: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.parts)

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
            re.escape(opener.strip().rstrip(",.!").lower())
            for opener in sorted(openers, key=len, reverse=True)
        )
        self._opener_re = re.compile(rf"^\s*(?:{alternatives})\s*[,.!:;—-]*\s*", re.IGNORECASE)
        self._own_names = (*personality.names, *_GENERIC_SPEAKERS)

    def process(self, raw: str, *, other_speakers: Iterable[str] = ()) -> ProcessedResponse:
        """Clean ``raw`` model output.

        Args:
            raw: The model's output.
            other_speakers: Names of other people in the conversation. If the
                model starts writing lines for them, the reply is cut there.
        """
        text = strip_reasoning(raw.replace("\r\n", "\n"))
        text = self._cut_hallucinated_turns(text, other_speakers)
        text = self._strip_speaker_labels(text)
        text = _strip_wrapping_quotes(text)
        text = _GAP_MARKER_RE.sub("", text)
        text = _strip_markdown(text)
        text = self._strip_openers(text)
        text = _MASS_MENTION_RE.sub(lambda m: f"@{_ZERO_WIDTH_SPACE}{m.group(1)}", text)
        text = _TRAILING_SPACE_RE.sub("", text)
        text = _MANY_NEWLINES_RE.sub("\n\n", text).strip()
        text = truncate_text(text, self._max_chars)
        return ProcessedResponse(tuple(split_messages(text, self._max_parts)))

    # ---------------------------------------------------------------- internals

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
        stripped = self._opener_re.sub("", text, count=1)
        return stripped if stripped.strip() else text


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
