"""Detecting requests the bot physically cannot fulfil.

A Discord text bot can only read and send text in the current conversation. It
can't hop in a voice call, play a game, send a selfie, or meet up, but small
local models happily say "sure, joining now". The system prompt already states
the bot's limits; when an incoming message looks like such a request, the
prompt builder also adds a direct per-turn instruction to turn it down with a
casual excuse, which models follow far more reliably.

Patterns are deliberately conservative: a missed request still meets the
general rule in the system prompt, while a false positive would make the bot
refuse something nobody asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImpossibleRequest:
    """A recognised request the bot cannot carry out."""

    key: str
    label: str
    """What was asked, phrased to complete "they're asking you to ..."."""


@dataclass(frozen=True, slots=True)
class _Category:
    key: str
    label: str
    patterns: tuple[re.Pattern[str], ...]


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_YOU = r"(?:you|u|ya|yall|y'all)"
_WANNA = r"(?:wanna|want to|tryna|trying to|down to|gonna|gon|can|could|will|would|should)"

CATEGORIES: tuple[_Category, ...] = (
    _Category(
        "voice",
        "join a voice call or voice channel",
        _compile(
            r"\b(?:join|hop (?:in|on|into)|get (?:in|on|into)|come (?:in|to|into)"
            r"|jump (?:in|on|into))\s+(?:the\s+|a\s+|my\s+|our\s+)?"
            r"(?:vc|voice(?: chat| channel| call)?|call|stage)\b",
            rf"\b{_WANNA}\s+{_YOU}?\s*(?:vc|voice chat|call)\b",
            rf"\b{_YOU}\s+(?:wanna|want to|tryna)\s+(?:vc|call)\b",
            r"\b(?:vc|call) (?:me|us|with me|with us)\b",
            r"\blets? (?:vc|call)\b",
            r"\b(?:get|hop|jump) on (?:discord|a call|call)\b",
        ),
    ),
    _Category(
        "video",
        "video call, stream, or share your screen",
        _compile(
            r"\b(?:facetime|video ?call|turn on (?:your|ur) (?:cam|camera|webcam))\b",
            r"\b(?:screen ?share|share (?:your|ur) screen|go live|start (?:a )?stream)\b",
        ),
    ),
    _Category(
        "games",
        "play a game together",
        _compile(
            rf"\b(?:{_WANNA}|lets?)\s+(?:{_YOU}\s+)?(?:play|run|queue|duo|squad up|1v1)\b"
            r"(?!\s+(?:it|that|this) (?:off|down|cool|safe))",
            r"\b(?:add me|join me|join my (?:game|server|lobby|world|party|realm))\b",
            r"\b(?:send|accept) (?:me )?(?:a |an |the )?(?:friend request|invite|party invite)\b",
        ),
    ),
    _Category(
        "media",
        "send a picture, video, voice message, or file",
        _compile(
            r"\b(?:send|post|show|drop)\s+(?:me\s+|us\s+)?(?:a |an |some |your |ur )?"
            r"(?:pic|pics|picture|photo|selfie|face|vid|video|voice (?:note|message|memo)"
            r"|file|clip)\b",
            r"\b(?:take|snap) a (?:pic|picture|photo|selfie)\b",
        ),
    ),
    _Category(
        "contact",
        "share your socials, number, or chat somewhere else",
        _compile(
            r"\b(?:what(?:'s| is)|whats|drop|send|give me)\s+(?:your|ur)\s+"
            r"(?:snap|snapchat|insta|instagram|ig|tiktok|number|phone number|socials"
            r"|discord tag|@)\b",
            r"\b(?:dm|text|call|message) me (?:on|at|later)\b",
        ),
    ),
    _Category(
        "meetup",
        "meet up in person",
        _compile(
            r"\b(?:meet up|meet irl|meetup|hang out irl|link up|come over)\b",
            r"\blets? (?:meet|hang)\b",
        ),
    ),
)


def detect_impossible_request(text: str) -> ImpossibleRequest | None:
    """Return the first impossible request ``text`` makes of the bot, if any."""
    for category in CATEGORIES:
        if any(pattern.search(text) for pattern in category.patterns):
            return ImpossibleRequest(category.key, category.label)
    return None
