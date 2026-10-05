"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tests.fakes import FakeOllama


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove ARCANE_* variables so tests never depend on the developer's shell."""
    for key in list(os.environ):
        if key.startswith("ARCANE_"):
            monkeypatch.delenv(key, raising=False)
    yield


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
