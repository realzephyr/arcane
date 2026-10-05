from __future__ import annotations

import asyncio
import socket
from typing import Any

import pytest
from aiohttp import web

from arcane.ai.providers import ollama as ollama_module
from arcane.ai.providers.base import (
    ChatMessage,
    GenerationOptions,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from arcane.ai.providers.factory import ProviderRegistry
from arcane.ai.providers.ollama import OllamaProvider, normalize_model_name
from arcane.config.settings import OllamaSettings, Settings
from arcane.core.errors import ConfigurationError
from tests.fakes import FakeOllama, chat_body


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_module, "_backoff_delay", lambda attempt: 0.0)


def _provider(base_url: str, **overrides: Any) -> OllamaProvider:
    return OllamaProvider(OllamaSettings(base_url=base_url, **overrides))


MESSAGES = [ChatMessage("system", "be nice"), ChatMessage("user", "hi")]


async def test_chat_sends_expected_payload(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.append((200, chat_body("hey there")))
    provider = _provider(url, keep_alive="10m")
    try:
        result = await provider.chat(
            MESSAGES,
            options=GenerationOptions(temperature=0.8, max_tokens=120, stop=("\nuser:",)),
        )
    finally:
        await provider.close()

    assert result.content == "hey there"
    assert result.provider == "ollama"
    assert result.prompt_tokens == 42
    assert result.completion_tokens == 7

    sent = fake.requests[0]
    assert sent["model"] == "llama3.1:8b"
    assert sent["stream"] is False
    assert sent["keep_alive"] == "10m"
    assert sent["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]
    assert sent["options"] == {"temperature": 0.8, "num_predict": 120, "stop": ["\nuser:"]}
    assert "format" not in sent
    assert "think" not in sent


async def test_chat_json_mode_and_think_flag(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    provider = _provider(url, think=False)
    try:
        await provider.chat(MESSAGES, model="qwen3:8b", options=GenerationOptions(json_mode=True))
    finally:
        await provider.close()

    sent = fake.requests[0]
    assert sent["model"] == "qwen3:8b"
    assert sent["format"] == "json"
    assert sent["think"] is False


async def test_missing_model_is_not_retried(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.append((404, {"error": 'model "nope" not found, try pulling it first'}))
    provider = _provider(url, max_retries=3)
    try:
        with pytest.raises(ModelNotFoundError):
            await provider.chat(MESSAGES, model="nope")
    finally:
        await provider.close()
    assert len(fake.requests) == 1


async def test_server_errors_are_retried(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.extend(
        [(500, {"error": "boom"}), (503, {"error": "busy"}), (200, chat_body("recovered"))]
    )
    provider = _provider(url, max_retries=2)
    try:
        result = await provider.chat(MESSAGES)
    finally:
        await provider.close()
    assert result.content == "recovered"
    assert len(fake.requests) == 3


async def test_retries_are_bounded(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.extend([(500, {"error": "boom"})] * 5)
    provider = _provider(url, max_retries=1)
    try:
        with pytest.raises(ProviderResponseError):
            await provider.chat(MESSAGES)
    finally:
        await provider.close()
    assert len(fake.requests) == 2


async def test_bad_request_is_not_retried(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.append((400, {"error": "invalid options"}))
    provider = _provider(url, max_retries=3)
    try:
        with pytest.raises(ProviderResponseError, match="invalid options"):
            await provider.chat(MESSAGES)
    finally:
        await provider.close()
    assert len(fake.requests) == 1


async def test_malformed_payload_raises(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.append((200, {"done": True}))
    provider = _provider(url)
    try:
        with pytest.raises(ProviderResponseError, match="missing message"):
            await provider.chat(MESSAGES)
    finally:
        await provider.close()


async def test_timeout_raises_timeout_error(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.delay = 0.5
    provider = _provider(url, timeout_seconds=0.1, max_retries=0)
    try:
        with pytest.raises(ProviderTimeoutError):
            await provider.chat(MESSAGES)
    finally:
        await provider.close()


async def test_unreachable_server_raises_unavailable() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    provider = _provider(f"http://127.0.0.1:{port}", max_retries=0)
    try:
        with pytest.raises(ProviderUnavailableError):
            await provider.chat(MESSAGES)
        health = await provider.health_check()
    finally:
        await provider.close()
    assert health.available is False


async def test_concurrency_is_bounded(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.delay = 0.05
    provider = _provider(url, max_concurrent_requests=1)
    try:
        await asyncio.gather(*(provider.chat(MESSAGES) for _ in range(4)))
    finally:
        await provider.close()
    assert fake.max_in_flight == 1
    assert len(fake.requests) == 4


async def test_health_check_reports_model_availability(
    fake_ollama: tuple[FakeOllama, str],
) -> None:
    _, url = fake_ollama
    provider = _provider(url)
    try:
        present = await provider.health_check("gemma3")
        missing = await provider.health_check("mistral-nemo")
        general = await provider.health_check()
    finally:
        await provider.close()

    assert present.available and present.model_available is True
    assert missing.available and missing.model_available is False
    assert "ollama pull mistral-nemo" in missing.detail
    assert general.models == ("gemma3:latest", "llama3.1:8b")


def test_normalize_model_name() -> None:
    assert normalize_model_name("llama3.1") == "llama3.1:latest"
    assert normalize_model_name("llama3.1:8b") == "llama3.1:8b"


async def test_registry_caches_and_rejects_unknown_providers() -> None:
    registry = ProviderRegistry(Settings())
    try:
        first = registry.get()
        assert first is registry.get("ollama")
        with pytest.raises(ConfigurationError, match="Unknown AI provider"):
            registry.get("nonexistent")
    finally:
        await registry.close()


async def test_warm_up_loads_with_the_reply_context_window(
    fake_ollama: tuple[FakeOllama, str],
) -> None:
    fake, url = fake_ollama
    provider = _provider(url, keep_alive="24h")
    try:
        assert await provider.warm_up(options=GenerationOptions(context_window=4096))
    finally:
        await provider.close()
    (load,) = fake.loads
    assert load["messages"] == []
    assert load["options"] == {"num_ctx": 4096}
    assert load["keep_alive"] == "24h"
    assert fake.requests == []


async def test_warm_up_failure_is_not_fatal() -> None:
    provider = _provider("http://127.0.0.1:9", max_retries=0)
    try:
        assert await provider.warm_up() is False
    finally:
        await provider.close()


async def test_cache_and_load_metrics_are_reported(
    fake_ollama: tuple[FakeOllama, str], caplog: pytest.LogCaptureFixture
) -> None:
    fake, url = fake_ollama
    body = {**chat_body("hi"), "prompt_eval_cached_count": 40, "load_duration": 5_000_000}
    fake.chat_responses.append((200, body))
    provider = _provider(url)
    try:
        result = await provider.chat(MESSAGES)
        assert result.cached_prompt_tokens == 40
        assert result.load_seconds == pytest.approx(0.005)

        # A multi-second load after the first request means the model was reloaded.
        fake.load_duration_ns = 3_000_000_000
        await provider.chat(MESSAGES)
    finally:
        await provider.close()
    assert "Ollama reloaded" in caplog.text


async def test_thinking_is_turned_off_for_thinking_models(
    fake_ollama: tuple[FakeOllama, str],
) -> None:
    fake, url = fake_ollama
    fake.capabilities = {
        "qwen3:8b": {"capabilities": ["completion", "thinking"]},
        "gpt-oss:20b": {
            "capabilities": ["completion", "thinking"],
            "thinking": {"values": ["low", "medium", "high"], "default": "medium"},
        },
    }
    provider = _provider(url)
    try:
        await provider.chat(MESSAGES, model="qwen3:8b")
        await provider.chat(MESSAGES, model="qwen3:8b")
        await provider.chat(MESSAGES, model="gpt-oss:20b")
        await provider.chat(MESSAGES, model="llama3.1:8b")
    finally:
        await provider.close()

    assert [r.get("think") for r in fake.requests] == [False, False, "low", None]
    assert fake.shows == ["qwen3:8b", "gpt-oss:20b", "llama3.1:8b"]  # asked once per model


async def test_configured_think_flag_wins(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.capabilities = {"qwen3:8b": {"capabilities": ["completion", "thinking"]}}
    provider = _provider(url, think=True)
    try:
        await provider.chat(MESSAGES, model="qwen3:8b")
    finally:
        await provider.close()
    assert fake.requests[0]["think"] is True
    assert fake.shows == []


async def test_health_check_reports_the_server_version(
    fake_ollama: tuple[FakeOllama, str],
) -> None:
    fake, url = fake_ollama
    provider = _provider(url)
    try:
        health = await provider.health_check("llama3.1:8b")
        fake.version = None
        without = await provider.health_check("llama3.1:8b")
    finally:
        await provider.close()
    assert health.detail.startswith("Ollama 0.35.1: ")
    assert without.detail == "model 'llama3.1:8b' is installed"


async def test_failed_capability_check_is_retried(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.capabilities = {"qwen3:8b": {"capabilities": ["completion", "thinking"]}}
    provider = _provider(url)
    real_show = fake.show
    calls = 0

    async def flaky_show(request: web.Request) -> web.StreamResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response({"error": "loading"}, status=503)
        return await real_show(request)

    fake.show = flaky_show  # type: ignore[method-assign]
    try:
        await provider.chat(MESSAGES, model="qwen3:8b")
        await provider.chat(MESSAGES, model="qwen3:8b")
    finally:
        await provider.close()
    assert [r.get("think") for r in fake.requests] == [None, False]
