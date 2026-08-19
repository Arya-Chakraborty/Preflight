"""Library mode: use Preflight in-process without running a server.

    import preflight
    client = preflight.wrap()
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[...])

Note: `create` is synchronous and runs its own event loop; call it from sync
code (scripts, notebooks). Inside an async application, use the proxy instead.
"""

from __future__ import annotations

import asyncio

from preflight.config import Settings
from preflight.gateway import Gateway


class _Completions:
    def __init__(self, gateway: Gateway):
        self._gateway = gateway

    def create(self, *, model: str, messages: list[dict], session_id: str | None = None, **kwargs):
        if kwargs.pop("stream", False):
            raise ValueError("Library mode is non-streaming; use the proxy for streaming.")
        payload = {"model": model, "messages": messages, **kwargs}
        return asyncio.run(self._gateway.handle(payload, session_id))


class _Chat:
    def __init__(self, gateway: Gateway):
        self.completions = _Completions(gateway)


class PreflightClient:
    def __init__(self, settings: Settings):
        self.gateway = Gateway(settings)
        self.chat = _Chat(self.gateway)

    def stats(self) -> dict:
        return self.gateway.logger.summary()
