"""Turning finished conversations into long-term memories.

:class:`MemoryExtractor` is a strategy interface. The default
:class:`LLMMemoryExtractor` asks the model for a few durable facts in JSON.
Future strategies (heuristics, embeddings, summarisation) can replace it
without touching the rest of the system.

:class:`MemoryConsolidator` runs the extractor when a conversation expires and
stores what it finds in :class:`~arcane.memory.long_term.LongTermMemory`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from arcane.ai.providers.base import ChatMessage, GenerationOptions, LLMProvider, ProviderError
from arcane.ai.text import strip_code_fence, strip_reasoning
from arcane.core.models import HistoryMessage
from arcane.database.models import ConversationRecord, MemoryKind
from arcane.memory.long_term import LongTermMemory
from arcane.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """A memory proposed by an extractor, not yet stored."""

    user_id: int
    content: str
    kind: MemoryKind
    importance: float


class MemoryExtractor(ABC):
    """Strategy for deciding what to remember from a conversation."""

    @abstractmethod
    async def extract(
        self,
        transcript: Sequence[HistoryMessage],
        participants: Mapping[int, str],
    ) -> list[MemoryCandidate]:
        """Propose memories about ``participants`` (user id -> display name)."""


class NullMemoryExtractor(MemoryExtractor):
    """Extractor that never remembers anything (long-term memory disabled)."""

    async def extract(
        self,
        transcript: Sequence[HistoryMessage],
        participants: Mapping[int, str],
    ) -> list[MemoryCandidate]:
        return []


EXTRACTION_SYSTEM_PROMPT = """\
You are the memory system of {bot_name}, a regular member of a Discord community.
Read the conversation and note what {bot_name} should remember about the people in it,
so future conversations with them feel continuous.

Remember only durable things: interests, opinions and positions they argued, what they
study or work on, books and ideas they mentioned, preferences, ongoing projects.
Skip small talk, jokes, passing moods, and anything about {bot_name} itself.
Never record sensitive personal data: contact details, precise locations, health,
finances, passwords, or anything they asked to keep private.

Write each memory as a short third-person statement without the person's name,
for example "studies physics at university" or "thinks free will is an illusion".
Rate importance from 0.0 (trivial) to 1.0 (central to who they are).
Write at most 3 memories per person and at most 8 in total, most important first.

Respond with JSON only, in exactly this shape:
{{"memories": [{{"person": "<name exactly as listed>", "content": "<statement>", \
"kind": "fact|preference|opinion|interest|topic|note", "importance": 0.5}}]}}
If nothing is worth remembering, respond with {{"memories": []}}."""


class LLMMemoryExtractor(MemoryExtractor):
    """Extract memories by asking the language model for structured output."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        bot_name: str,
        model: str | None = None,
        context_window: int | None = None,
        max_per_user: int = 4,
        min_importance: float = 0.35,
        max_transcript_chars: int = 6000,
    ) -> None:
        self._provider = provider
        self._bot_name = bot_name
        self._model = model
        # Must match the reply requests' num_ctx: Ollama reloads the model whenever
        # the context size changes between requests.
        self._context_window = context_window
        self._max_per_user = max_per_user
        self._min_importance = min_importance
        self._max_transcript_chars = max_transcript_chars

    async def extract(
        self,
        transcript: Sequence[HistoryMessage],
        participants: Mapping[int, str],
    ) -> list[MemoryCandidate]:
        if not transcript or not participants:
            return []

        lines = [
            f"{self._bot_name if message.is_self else message.author_name}: {message.content}"
            for message in transcript
            if message.content.strip()
        ]
        text = "\n".join(lines)
        if len(text) > self._max_transcript_chars:
            text = "…\n" + text[-self._max_transcript_chars :]

        names = ", ".join(sorted(set(participants.values())))
        messages = [
            ChatMessage("system", EXTRACTION_SYSTEM_PROMPT.format(bot_name=self._bot_name)),
            ChatMessage("user", f"People: {names}\n\nConversation:\n{text}"),
        ]
        try:
            result = await self._provider.chat(
                messages,
                model=self._model,
                options=GenerationOptions(
                    temperature=0.2,
                    # Room for 8 memories; a truncated reply is still salvaged.
                    max_tokens=600,
                    context_window=self._context_window,
                    json_mode=True,
                ),
            )
        except ProviderError as exc:
            logger.warning("Memory extraction failed: %s", exc)
            return []

        return parse_candidates(
            result.content,
            participants,
            max_per_user=self._max_per_user,
            min_importance=self._min_importance,
        )


def parse_candidates(
    raw: str,
    participants: Mapping[int, str],
    *,
    max_per_user: int = 4,
    min_importance: float = 0.0,
) -> list[MemoryCandidate]:
    """Parse extractor JSON leniently; invalid entries are skipped, never raised."""
    payload = _load_json(raw)
    items: Any = payload.get("memories", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return []

    by_name = {name.casefold(): user_id for user_id, name in participants.items()}
    per_user: dict[int, int] = {}
    candidates: list[MemoryCandidate] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        person = str(item.get("person") or item.get("user") or item.get("name") or "").strip()
        user_id = by_name.get(person.casefold().lstrip("@"))
        if user_id is None and person.isdigit() and int(person) in participants:
            user_id = int(person)
        content = str(item.get("content") or item.get("fact") or "").strip()
        if user_id is None or not content:
            continue
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        importance = min(max(importance, 0.0), 1.0)
        if importance < min_importance or per_user.get(user_id, 0) >= max_per_user:
            continue
        per_user[user_id] = per_user.get(user_id, 0) + 1
        candidates.append(
            MemoryCandidate(
                user_id=user_id,
                content=content,
                kind=MemoryKind.parse(item.get("kind"), MemoryKind.NOTE),
                importance=importance,
            )
        )
    return candidates


def _load_json(raw: str) -> Any:
    text = strip_code_fence(strip_reasoning(raw))
    try:
        return json.loads(text)
    except ValueError:
        pass
    # Skip text around the outermost object or array; an inner object alone is useless.
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    end = text.rfind("}" if text[start : start + 1] == "{" else "]")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except ValueError:
            pass
    salvaged = _salvage_items(text)
    if salvaged:
        logger.info(
            "Memory extraction output was cut off; salvaged %d complete memories", len(salvaged)
        )
    return salvaged


_DECODER = json.JSONDecoder()


def _salvage_items(text: str) -> list[Any]:
    """Decode the complete items of a JSON array that was cut off mid-way.

    Output that hits the token limit ends mid-object, so the whole document is
    invalid even though the first memories are fine.
    """
    key = text.find('"memories"')
    start = text.find("[", key if key != -1 else 0)
    if start == -1:
        return []
    items: list[Any] = []
    position = start + 1
    while True:
        while position < len(text) and text[position] in " \t\r\n,":
            position += 1
        if position >= len(text) or text[position] == "]":
            return items
        try:
            item, position = _DECODER.raw_decode(text, position)
        except ValueError:
            return items
        items.append(item)


class MemoryConsolidator:
    """Moves what matters from a finished conversation into long-term memory."""

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        extractor: MemoryExtractor,
        *,
        min_user_messages: int = 3,
    ) -> None:
        self._short_term = short_term
        self._long_term = long_term
        self._extractor = extractor
        self._min_user_messages = min_user_messages

    async def consolidate(self, conversation: ConversationRecord) -> int:
        """Extract and store memories. Returns the number of memories stored."""
        participants = dict(conversation.participants)
        if not participants:
            return 0
        transcript = await self._short_term.messages_between(
            conversation.channel_id,
            conversation.started_at - timedelta(minutes=2),
            conversation.last_activity_at,
        )
        user_messages = sum(1 for m in transcript if m.author_id in participants)
        if user_messages < self._min_user_messages:
            return 0

        candidates = await self._extractor.extract(transcript, participants)
        stored = 0
        for candidate in candidates:
            record = await self._long_term.remember(
                candidate.content,
                user_id=candidate.user_id,
                kind=candidate.kind,
                importance=candidate.importance,
                source=f"conversation:{conversation.channel_id}",
            )
            stored += record is not None
        if stored:
            logger.info(
                "[%s] stored %d long-term memories from a conversation in channel %d",
                conversation.bot_id,
                stored,
                conversation.channel_id,
            )
        return stored
