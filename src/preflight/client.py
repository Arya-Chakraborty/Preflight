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
import json
import queue
import threading
from collections.abc import Iterator

from preflight.config import Settings
from preflight.gateway import Gateway


class _Completions:
    def __init__(self, client: PreflightClient):
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        session_id: str | None = None,
        stream: bool = False,
        **kwargs,
    ):
        payload = {"model": model, "messages": messages, **kwargs}
        if stream:
            return self._client._iter_stream(payload, session_id)
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
        self.gateway._bg_loop = self._loop

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _iter_stream(self, payload: dict, session_id: str | None) -> Iterator[dict]:
        """Yield OpenAI-style chunk dicts from the gateway SSE stream."""
        q: queue.Queue[str | None] = queue.Queue()

        async def pump():
            try:
                async for line in self.gateway.handle_stream(payload, session_id):
                    q.put(line)
            finally:
                q.put(None)

        fut = asyncio.run_coroutine_threadsafe(pump(), self._loop)
        while True:
            line = q.get()
            if line is None:
                break
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue
        fut.result()

    def stats(self) -> dict:
        return self.gateway.logger.summary()
