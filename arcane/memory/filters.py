"""Text hygiene for stored memories.

Long-term memories must be short, normalised, and free of obviously sensitive
data. These checks are deliberately conservative: rejecting a harmless memory
costs little, storing someone's phone number costs a lot.
"""

from __future__ import annotations

import re

MAX_MEMORY_CHARS = 280

_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Email addresses
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # Phone numbers: 7+ digits, optionally separated
    re.compile(r"(?:\+?\d[\s().-]?){7,}\d"),
    # Payment-card-like digit runs
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    # Discord tokens and long opaque secrets (API keys, hashes)
    re.compile(r"[A-Za-z0-9_-]{23,28}\.[A-Za-z0-9_-]{6,7}\.[A-Za-z0-9_-]{27,}"),
    re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{32,}\b"),
    # Credentials and precise location keywords
    re.compile(r"\b(password|passwd|passcode|api[ _-]?key|secret key|private key)\b", re.I),
    re.compile(r"\b(social security|ssn|home address|lives at|street address)\b", re.I),
)

_WORD_RE = re.compile(r"[a-z0-9']+")
# fmt: off
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or",
    "in", "on", "at", "for", "with", "about", "they", "them", "their", "he", "she", "his",
    "her", "it", "its", "this", "that", "likes", "like", "really", "very",
})
# fmt: on


def contains_sensitive_data(text: str) -> bool:
    """True when ``text`` looks like it contains personal or secret data."""
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


def normalize_memory_text(text: str) -> str:
    """Collapse whitespace, strip list markers, and cap length."""
    cleaned = " ".join(text.split()).strip(" -*•\"'")
    if len(cleaned) > MAX_MEMORY_CHARS:
        cleaned = cleaned[: MAX_MEMORY_CHARS - 1].rsplit(" ", 1)[0] + "…"
    return cleaned


def keywords(text: str) -> frozenset[str]:
    """Content words used for duplicate detection."""
    return frozenset(w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS)


def similarity(first: str, second: str) -> float:
    """Jaccard similarity of content words, in ``[0, 1]``."""
    a, b = keywords(first), keywords(second)
    if not a or not b:
        return 1.0 if first.strip().lower() == second.strip().lower() else 0.0
    return len(a & b) / len(a | b)
