"""Small text utilities shared by the AI layer."""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*\Z", re.IGNORECASE | re.DOTALL)
_ORPHAN_CLOSE_RE = re.compile(r"\A.*?</(think|thinking|reasoning)>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$")


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` style reasoning emitted by reasoning models.

    Handles complete blocks, a block cut off by the token limit, and output
    that starts mid-reasoning with only a closing tag.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = _ORPHAN_CLOSE_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


def strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence wrapping the whole text."""
    return _CODE_FENCE_RE.sub("", text.strip()).strip()
