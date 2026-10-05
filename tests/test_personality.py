from __future__ import annotations

import pytest
from pydantic import ValidationError

from arcane.core.errors import PersonalityNotFoundError
from arcane.personalities.base import Personality, TimingProfile
from arcane.personalities.registry import available_personalities, load_personality


def _minimal(**overrides: object) -> Personality:
    fields: dict[str, object] = {
        "id": "tester",
        "name": "tester",
        "description": "test personality",
        "identity": "You are tester.",
    }
    fields.update(overrides)
    return Personality.model_validate(fields)


def test_mp3_is_discoverable_and_valid() -> None:
    assert "mp3" in available_personalities()
    mp3 = load_personality("mp3")
    assert mp3.id == "mp3"
    assert mp3.name == "mp3"
    assert mp3.conversation_topics
    assert mp3.style.guidelines
    assert load_personality("mp3") is mp3  # cached


def test_unknown_personality_is_reported() -> None:
    with pytest.raises(PersonalityNotFoundError, match=r"Available: .*mp3"):
        load_personality("does_not_exist")
    with pytest.raises(PersonalityNotFoundError, match="Invalid"):
        load_personality("../etc")


def test_personality_is_immutable() -> None:
    mp3 = load_personality("mp3")
    with pytest.raises(ValidationError):
        mp3.name = "other"  # type: ignore[misc]


def test_minimal_personality_gets_defaults() -> None:
    personality = _minimal()
    assert personality.behavior.conversation_timeout_seconds > 0
    assert personality.model.provider is None
    assert personality.style.max_messages_per_reply >= 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "Bad Id"},
        {"identity": "   "},
        {"unknown_field": True},
    ],
)
def test_invalid_personalities_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _minimal(**overrides)


def test_invalid_timing_range_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TimingProfile(reaction_seconds=(3.0, 1.0))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hey mp3 what do you think", True),
        ("MP3, thoughts?", True),
        ("ask mp 3 about it", True),
        ("convert it to an mp3 file", True),  # ambiguous, but a name match is a name match
        ("mp35 is not the name", False),
        ("email me at mp3@example.com", False),
        ("nothing to see", False),
    ],
)
def test_name_detection(text: str, expected: bool) -> None:
    assert load_personality("mp3").is_named_in(text) is expected


def test_interest_hits_handle_words_and_phrases() -> None:
    mp3 = load_personality("mp3")
    assert mp3.interest_hits("is free will even real or is it just simulation theory") == 2
    assert mp3.interest_hits("anyone up for some games tonight") == 0
    assert _minimal(interest_keywords=["Kant"]).interest_keywords == frozenset({"kant"})


def test_invalid_blocked_pattern_is_rejected() -> None:
    from arcane.personalities.base import StyleProfile

    with pytest.raises(ValidationError, match="invalid blocked pattern"):
        StyleProfile(blocked_patterns=("(unclosed",))


def test_mp3_persona_rules() -> None:
    """The owner's requirements for mp3, pinned so a persona edit can't drop them."""
    import json

    mp3 = load_personality("mp3")
    assert "!" not in json.dumps(mp3.model_dump(mode="json"))
    assert mp3.style.allow_exclamation_points is False
    assert mp3.style.lowercase_starts is True
    assert "18 year old" in mp3.identity
    assert "sincerely asks whether you're a bot or an ai you say yes" in mp3.identity
    guidelines = " ".join(mp3.style.guidelines).lower()
    assert "never type an exclamation point" in guidelines
    assert "join vc" in guidelines and "never say yes" in guidelines
    assert "no slurs" in guidelines and "no flirting" in guidelines
    assert mp3.timing.typing_speed_wpm == 60
