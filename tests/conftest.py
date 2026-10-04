"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove ARCANE_* variables so tests never depend on the developer's shell."""
    for key in list(os.environ):
        if key.startswith("ARCANE_"):
            monkeypatch.delenv(key, raising=False)
    yield
