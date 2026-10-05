from __future__ import annotations

from datetime import UTC, datetime

import pytest

from arcane.ai.postprocess import ProcessedResponse
from arcane.ai.prompts import PromptContext
from arcane.ai.providers.base import ProviderUnavailableError
from arcane.ai.response_manager import ResponseManager
from arcane.core.clock import utcnow
from arcane.personalities.registry import load_personality
from tests.factories import GENERAL, history
from tests.fakes import ScriptedProvider

MP3 = load_personality("mp3")


def _context(*messages: object) -> PromptContext:
    return PromptContext(
        personality=MP3,
        channel=GENERAL,
        history=list(messages) or [history("hey mp3, thoughts on kant?")],  # type: ignore[arg-type]
        now=datetime(2026, 10, 4, tzinfo=UTC),
        target_user_name="alice",
        target_message="hey mp3, thoughts on kant?",
    )


async def test_generates_clean_parts_with_personality_options() -> None:
    provider = ScriptedProvider(
        "mp3: honestly kant is a lot\n\nbut the categorical imperative slaps"
    )
    manager = ResponseManager(MP3, provider)

    reply = await manager.generate(_context())

    assert reply is not None
    assert reply.parts == ("honestly kant is a lot", "but the categorical imperative slaps")
    assert reply.attempts == 1
    _, options, model = provider.calls[0]
    assert options is not None
    assert options.temperature == MP3.model.temperature
    assert options.max_tokens == MP3.model.max_tokens
    assert model is None  # provider default


async def test_model_override_is_used() -> None:
    provider = ScriptedProvider("ok")
    manager = ResponseManager(MP3, provider, model="gemma3:12b")
    await manager.generate(_context())
    assert provider.calls[0][2] == "gemma3:12b"
    assert manager.model_name == "gemma3:12b"


async def test_empty_output_is_retried_with_higher_temperature() -> None:
    provider = ScriptedProvider("<think>hmm</think>", "fine, kant wins")
    manager = ResponseManager(MP3, provider)

    reply = await manager.generate(_context())

    assert reply is not None and reply.parts == ("fine, kant wins",)
    assert reply.attempts == 2
    first, second = (call[1] for call in provider.calls)
    assert first is not None and second is not None
    assert second.temperature is not None and first.temperature is not None
    assert second.temperature > first.temperature


async def test_repetition_is_rejected() -> None:
    repeated = "i keep coming back to the idea that duty matters more than outcomes"
    provider = ScriptedProvider(repeated, "ok different angle: what about lying to a murderer")
    manager = ResponseManager(MP3, provider)
    context = _context(history("kant?"), history(repeated, is_self=True), history("and?"))

    reply = await manager.generate(context)

    assert reply is not None
    assert reply.parts == ("ok different angle: what about lying to a murderer",)


async def test_short_replies_may_repeat() -> None:
    provider = ScriptedProvider("lol fair")
    manager = ResponseManager(MP3, provider)
    context = _context(history("x"), history("lol fair", is_self=True), history("y"))
    reply = await manager.generate(context)
    assert reply is not None and reply.parts == ("lol fair",)


async def test_gives_up_after_max_attempts() -> None:
    provider = ScriptedProvider("")
    manager = ResponseManager(MP3, provider, max_attempts=2)
    assert await manager.generate(_context()) is None
    assert len(provider.calls) == 2


async def test_provider_errors_propagate() -> None:
    manager = ResponseManager(MP3, ScriptedProvider(ProviderUnavailableError("down")))
    with pytest.raises(ProviderUnavailableError):
        await manager.generate(_context())


async def test_blocked_replies_are_regenerated_or_dropped() -> None:
    strict = MP3.model_copy(
        update={"style": MP3.style.model_copy(update={"blocked_patterns": (r"\bbadword\b",)})}
    )
    provider = ScriptedProvider("that is a badword", "clean reply")
    reply = await ResponseManager(strict, provider).generate(_context())
    assert reply is not None and reply.parts == ("clean reply",)

    always_bad = ScriptedProvider("badword again")
    assert await ResponseManager(strict, always_bad).generate(_context()) is None


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        ("lol i forget stuff when half my context is gone", "talks about how it works"),
        ("fair, not part of the chat though", "echoes its private note"),
        ("ok\n\nWrite your next message, replying to alice", None),  # cut, not flagged
    ],
)
async def test_flagged_replies_are_regenerated_with_the_reason_logged(
    raw: str, problem: str | None, caplog: pytest.LogCaptureFixture
) -> None:
    provider = ScriptedProvider(raw, "kant would hate that")
    with caplog.at_level("INFO", logger="arcane.ai.response_manager"):
        reply = await ResponseManager(MP3, provider).generate(_context())

    assert reply is not None
    if problem is None:
        assert reply.attempts == 1 and reply.parts == ("ok",)
    else:
        assert reply.attempts == 2 and reply.parts == ("kant would hate that",)
        assert any(problem in record.getMessage() for record in caplog.records)


def test_problem_is_the_first_issue() -> None:
    processed = ProcessedResponse(
        ("x",), issues=("contains a blocked pattern", "talks about how it works")
    )
    assert processed.blocked
    assert ResponseManager._problem_with(processed, []) == "contains a blocked pattern"
    assert ResponseManager._problem_with(ProcessedResponse(("fine",)), []) is None
    assert not ProcessedResponse(("x",), issues=("talks about how it works",)).blocked


async def test_agreeing_to_an_impossible_request_is_regenerated() -> None:
    provider = ScriptedProvider("omw", "nah i dont do vc lol")
    manager = ResponseManager(MP3, provider)
    reply = await manager.generate(_vc_context("yo mp3 hop in vc"))
    assert reply is not None and reply.text == "nah i dont do vc lol"
    assert reply.attempts == 2
    retry_prompt = provider.calls[1][0][-1].content
    assert "turn it down casually" in retry_prompt


async def test_agreement_check_only_applies_to_impossible_requests() -> None:
    manager = ResponseManager(MP3, ScriptedProvider("bet"))
    reply = await manager.generate(_vc_context("mp3 debate me on free will"))
    assert reply is not None and reply.text == "bet"


def _vc_context(text: str) -> PromptContext:
    return PromptContext(
        personality=MP3,
        channel=GENERAL,
        history=[history(text, author_name="alice")],
        now=utcnow(),
        target_user_id=1,
        target_user_name="alice",
        target_message=text,
    )
