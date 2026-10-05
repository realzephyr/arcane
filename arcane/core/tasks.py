"""Background task helpers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class PeriodicTask:
    """Runs an async callable repeatedly until stopped.

    Exceptions from the callable are logged and never stop the loop, so one
    failed maintenance pass cannot silently disable maintenance forever.
    """

    def __init__(
        self,
        name: str,
        func: Callable[[], Awaitable[object]],
        *,
        interval: Callable[[], float] | float,
        jitter: float = 0.0,
        run_immediately: bool = False,
    ) -> None:
        self.name = name
        self._func = func
        self._interval = interval
        self._jitter = jitter
        self._run_immediately = run_immediately
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._run(), name=self.name)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    def _next_delay(self) -> float:
        base = self._interval() if callable(self._interval) else self._interval
        if self._jitter:
            base *= random.uniform(1 - self._jitter, 1 + self._jitter)
        return max(base, 0.0)

    async def _run(self) -> None:
        if not self._run_immediately:
            await asyncio.sleep(self._next_delay())
        while True:
            try:
                await self._func()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Background task %s failed", self.name)
            await asyncio.sleep(self._next_delay())
