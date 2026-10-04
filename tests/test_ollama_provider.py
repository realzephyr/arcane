from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

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

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@dataclass
class FakeOllama:
    """A scripted stand-in for the Ollama HTTP API."""

    chat_responses: list[tuple[int, Any]] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: ["llama3.1:8b", "gemma3:latest"])
    requests: list[dict[str, Any]] = field(default_factory=list)
    delay: float = 0.0
    in_flight: int = 0
    max_in_flight: int = 0

    async def chat(self, request: web.Request) -> web.StreamResponse:
        self.requests.append(await request.json())
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            status, body = (
                self.chat_responses.pop(0)
                if self.chat_responses
                else (200, _chat_body("default reply"))
            )
        finally:
            self.in_flight -= 1
        return web.json_response(body, status=status)

    async def tags(self, _request: web.Request) -> web.StreamResponse:
        return web.json_response({"models": [{"name": name} for name in self.models]})


def _chat_body(content: str) -> dict[str, Any]:
    return {
        "model": "llama3.1:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 42,
        "eval_count": 7,
    }


@pytest.fixture
async def fake_ollama() -> AsyncIterator[tuple[FakeOllama, str]]:
    fake = FakeOllama()
    app = web.Application()
    app.router.add_post("/api/chat", fake.chat)
    app.router.add_get("/api/tags", fake.tags)
    server = TestServer(app)
    await server.start_server()
    try:
        yield fake, str(server.make_url("")).rstrip("/")
    finally:
        await server.close()


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama_module, "_backoff_delay", lambda attempt: 0.0)


def _provider(base_url: str, **overrides: Any) -> OllamaProvider:
    return OllamaProvider(OllamaSettings(base_url=base_url, **overrides))


MESSAGES = [ChatMessage("system", "be nice"), ChatMessage("user", "hi")]


async def test_chat_sends_expected_payload(fake_ollama: tuple[FakeOllama, str]) -> None:
    fake, url = fake_ollama
    fake.chat_responses.append((200, _chat_body("hey there")))
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
        [(500, {"error": "boom"}), (503, {"error": "busy"}), (200, _chat_body("recovered"))]
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
