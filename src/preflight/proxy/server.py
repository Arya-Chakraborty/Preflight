"""OpenAI-compatible local proxy. Point any client at http://host:port/v1."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from preflight.config import Settings
from preflight.gateway import Gateway

SESSION_HEADER = "x-preflight-session"


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="preflight", version="0.1.0")
    gateway = Gateway(settings)
    app.state.gateway = gateway

    @app.get("/health")
    async def health():
        return {"status": "ok", "memory_entries": gateway.memory.count()}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        session_id = request.headers.get(SESSION_HEADER)
        if payload.get("stream"):
            payload = {k: v for k, v in payload.items() if k != "stream"}
            return StreamingResponse(
                gateway.handle_stream(payload, session_id),
                media_type="text/event-stream",
            )
        response = await gateway.handle(payload, session_id)
        status = 502 if "error" in response and "choices" not in response else 200
        return JSONResponse(response, status_code=status)

    @app.get("/v1/models")
    async def models():
        # Pass-through convenience so SDK health checks succeed.
        return {"object": "list", "data": []}

    return app
