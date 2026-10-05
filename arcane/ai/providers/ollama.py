"""Ollama backend using its native HTTP API (``/api/chat``, ``/api/tags``).

``aiohttp`` is used directly (it already ships with discord.py), which keeps
dependencies small and gives full control over timeouts, retries, and
concurrency:

* a semaphore bounds in-flight requests, because Ollama typically serves one
  request at a time and queued requests would otherwise time out;
* transient failures (connection errors, timeouts, 5xx) are retried with
  exponential backoff and jitter; client errors are not;
* models with a thinking mode (qwen3, deepseek-r1, ...) think before every
  answer unless told not to, which makes a chat bot slow. Unless ``think`` is
  configured, the provider asks Ollama once per model (``/api/show``) and turns
  thinking off, or down to "low" for models that can't switch it off.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Sequence
from typing import Any, ClassVar

import aiohttp

from arcane.ai.providers.base import (
    ChatMessage,
    GenerationOptions,
    GenerationResult,
    LLMProvider,
    ModelNotFoundError,
    ProviderError,
    ProviderHealth,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from arcane.config.settings import OllamaSettings

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 8.0
RELOAD_WARNING_SECONDS = 1.0


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: ~1s, ~2s, ~4s, ... capped."""
    base = min(2.0 ** (attempt - 1), _MAX_BACKOFF_SECONDS)
    return float(base * random.uniform(0.75, 1.25))


def normalize_model_name(name: str) -> str:
    """Ollama treats an untagged model name as ``:latest``."""
    return name if ":" in name else f"{name}:latest"


class OllamaProvider(LLMProvider):
    """:class:`LLMProvider` implementation for Ollama."""

    name: ClassVar[str] = "ollama"

    def __init__(
        self,
        settings: OllamaSettings,
        *,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._settings = settings
        self._session = session
        self._owns_session = session is None
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._served_requests = 0
        self._think: dict[str, bool | str | None] = {}

    @property
    def default_model(self) -> str:
        return self._settings.model

    # ------------------------------------------------------------------ public

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        if not messages:
            raise ValueError("messages must not be empty")
        options = options or GenerationOptions()
        model_name = model or self.default_model

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "keep_alive": self._settings.keep_alive,
        }
        ollama_options = self._build_options(options)
        if ollama_options:
            payload["options"] = ollama_options
        if options.json_mode:
            payload["format"] = "json"
        think = await self._think_value(model_name)
        if think is not None:
            payload["think"] = think

        started = time.monotonic()
        data = await self._request_with_retries("POST", "/api/chat", payload)
        elapsed = time.monotonic() - started

        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ProviderResponseError("Ollama response is missing message content")

        result = GenerationResult(
            content=message["content"],
            model=str(data.get("model", model_name)),
            provider=self.name,
            prompt_tokens=_optional_int(data.get("prompt_eval_count")),
            completion_tokens=_optional_int(data.get("eval_count")),
            duration_seconds=elapsed,
            cached_prompt_tokens=_optional_int(data.get("prompt_eval_cached_count")),
            load_seconds=_nanoseconds(data.get("load_duration")),
        )
        logger.debug(
            "Ollama generation: model=%s prompt_tokens=%s cached=%s completion_tokens=%s "
            "load=%.2fs prompt_eval=%.2fs total=%.2fs",
            result.model,
            result.prompt_tokens,
            result.cached_prompt_tokens,
            result.completion_tokens,
            result.load_seconds or 0.0,
            _nanoseconds(data.get("prompt_eval_duration")) or 0.0,
            elapsed,
        )
        if (
            self._served_requests > 0
            and result.load_seconds is not None
            and result.load_seconds > RELOAD_WARNING_SECONDS
        ):
            logger.warning(
                "Ollama reloaded %s (%.1fs). Keep num_ctx identical for every request to a "
                "model and keep_alive long, or each reload costs a full model load.",
                result.model,
                result.load_seconds,
            )
        self._served_requests += 1
        return result

    async def warm_up(
        self,
        *,
        model: str | None = None,
        options: GenerationOptions | None = None,
    ) -> bool:
        """Load the model by sending a chat request with no messages.

        The load-time options (``num_ctx``) are sent too, so the first real
        request finds the model loaded exactly as it needs it.
        """
        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": [],
            "keep_alive": self._settings.keep_alive,
        }
        ollama_options = self._build_options(options or GenerationOptions())
        if ollama_options:
            payload["options"] = ollama_options
        started = time.monotonic()
        try:
            await self._request_with_retries("POST", "/api/chat", payload)
        except ProviderError as exc:
            logger.warning("Could not preload %s: %s", payload["model"], exc)
            return False
        logger.info("Loaded %s in %.1fs", payload["model"], time.monotonic() - started)
        return True

    async def version(self) -> str | None:
        """The Ollama server version, or ``None`` if it can't be read."""
        try:
            data = await self._request("GET", "/api/version", None)
        except ProviderError:
            return None
        version = data.get("version")
        return str(version) if version else None

    async def list_models(self) -> tuple[str, ...]:
        """Names of the models installed on the Ollama server."""
        data = await self._request("GET", "/api/tags", None)
        models = data.get("models", [])
        if not isinstance(models, list):
            raise ProviderResponseError("Ollama /api/tags returned an unexpected payload")
        names = (
            str(entry.get("name") or entry.get("model"))
            for entry in models
            if isinstance(entry, dict) and (entry.get("name") or entry.get("model"))
        )
        return tuple(sorted(names))

    async def health_check(self, model: str | None = None) -> ProviderHealth:
        try:
            models = await self.list_models()
        except ProviderError as exc:
            return ProviderHealth(available=False, detail=str(exc))

        version = await self.version()
        server = f"Ollama {version}: " if version else ""
        if model is None:
            return ProviderHealth(
                available=True, detail=f"{server}{len(models)} model(s) installed", models=models
            )
        installed = {normalize_model_name(name) for name in models}
        model_available = normalize_model_name(model) in installed
        detail = (
            f"model '{model}' is installed"
            if model_available
            else f"model '{model}' is not installed; run: ollama pull {model}"
        )
        return ProviderHealth(
            available=True, detail=server + detail, model_available=model_available, models=models
        )

    async def close(self) -> None:
        if self._owns_session and self._session is not None:
            if not self._session.closed:
                await self._session.close()
            self._session = None

    # ---------------------------------------------------------------- internals

    async def _think_value(self, model: str) -> bool | str | None:
        """The ``think`` field for ``model``: configured, or chosen from its capabilities."""
        if self._settings.think is not None:
            return self._settings.think
        if model in self._think:
            return self._think[model]
        value: bool | str | None = None
        try:
            data = await self._request("POST", "/api/show", {"model": model})
        except ProviderError as exc:
            logger.debug("Could not read the capabilities of %s: %s", model, exc)
            data = {}
        capabilities = data.get("capabilities")
        if isinstance(capabilities, list) and "thinking" in capabilities:
            thinking = data.get("thinking")
            values = thinking.get("values") if isinstance(thinking, dict) else None
            if isinstance(values, list) and values and False not in values:
                value = "low" if "low" in values else None
            else:
                value = False
            logger.info(
                "%s has a thinking mode; %s for faster replies (set ARCANE_OLLAMA_THINK to "
                "override)",
                model,
                f"using think={value!r}" if value else "turning it off",
            )
        self._think[model] = value
        return value

    @staticmethod
    def _build_options(options: GenerationOptions) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "temperature": options.temperature,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "repeat_penalty": options.repeat_penalty,
            "num_predict": options.max_tokens,
            "num_ctx": options.context_window,
            "seed": options.seed,
            "stop": list(options.stop) if options.stop else None,
        }
        return {key: value for key, value in mapping.items() if value is not None}

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "arcane-bot"},
            )
            self._owns_session = True
        return self._session

    async def _request_with_retries(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        attempts = self._settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                async with self._semaphore:
                    return await self._request(method, path, payload)
            except ProviderError as exc:
                if not exc.retryable or attempt == attempts:
                    raise
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Ollama request failed (%s); retry %d/%d in %.1fs",
                    exc,
                    attempt,
                    attempts - 1,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        url = f"{self._settings.base_url}{path}"
        timeout = aiohttp.ClientTimeout(
            total=self._settings.timeout_seconds,
            connect=self._settings.connect_timeout_seconds,
        )
        session = self._get_session()
        try:
            async with session.request(method, url, json=payload, timeout=timeout) as response:
                body = await response.text()
                if response.status >= 400:
                    raise self._error_for_status(response.status, body)
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"Ollama did not respond within {self._settings.timeout_seconds:.0f}s"
            ) from exc
        except aiohttp.ClientConnectionError as exc:
            raise ProviderUnavailableError(
                f"cannot reach Ollama at {self._settings.base_url}: {exc}"
            ) from exc
        except aiohttp.ClientError as exc:
            raise ProviderError(f"Ollama request failed: {exc}", retryable=True) from exc

        try:
            data = json.loads(body)
        except ValueError as exc:
            raise ProviderResponseError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError("Ollama returned an unexpected payload")
        if data.get("error"):
            raise ProviderResponseError(f"Ollama error: {data['error']}")
        return data

    @staticmethod
    def _error_for_status(status: int, body: str) -> ProviderError:
        detail = body.strip()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("error"):
                detail = str(parsed["error"])
        except ValueError:
            pass
        detail = detail[:300] or f"HTTP {status}"

        if status == 404 and "not found" in detail.lower():
            return ModelNotFoundError(f"Ollama: {detail}")
        if status >= 500 or status == 429:
            return ProviderResponseError(f"Ollama HTTP {status}: {detail}", retryable=True)
        return ProviderResponseError(f"Ollama HTTP {status}: {detail}", retryable=False)


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _nanoseconds(value: Any) -> float | None:
    """Ollama reports durations in nanoseconds."""
    return value / 1e9 if isinstance(value, int | float) else None
