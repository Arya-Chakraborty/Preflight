"""Library mode: use Preflight in-process without running a server.

    import preflight
    client = preflight.wrap()
    resp = client.chat.completions.create(model="gemini/gemini-3.5-flash-lite", messages=[...])

A single background event loop serves all calls, so provider-SDK async hooks
(e.g. litellm's logging worker) run to completion instead of dying with a
"coroutine was never awaited" warning when a per-call loop closes.
"""

from __future__ import annotations

import asyncio
import threading

from preflight.config import Settings
from preflight.gateway import Gateway


class _Completions:
    def __init__(self, client: PreflightClient):
        self._client = client

    def create(self, *, model: str, messages: list[dict], session_id: str | None = None, **kwargs):
        if kwargs.pop("stream", False):
            raise ValueError("Library mode is non-streaming; use the proxy for streaming.")
        payload = {"model": model, "messages": messages, **kwargs}
        return self._client._run(self._client.gateway.handle(payload, session_id))


class _Chat:
    def __init__(self, client: PreflightClient):
        self.completions = _Completions(client)


class PreflightClient:
    def __init__(self, settings: Settings):
        self.gateway = Gateway(settings)
        self.chat = _Chat(self)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def stats(self) -> dict:
        return self.gateway.logger.summary()
