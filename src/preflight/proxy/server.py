"""OpenAI-compatible local proxy. Point any client at http://host:port/v1."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from preflight import __version__
from preflight.config import Settings
from preflight.costs.prices import listed_models
from preflight.gateway import Gateway

SESSION_HEADER = "x-preflight-session"
TASK_HEADER = "x-preflight-task"


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="preflight", version=__version__)
    gateway = Gateway(settings)
    app.state.gateway = gateway

    def _authorized(request: Request) -> bool:
        if not settings.api_key:
            return True
        auth = request.headers.get("authorization") or ""
        bearer = auth.removeprefix("Bearer ").strip() if auth.lower().startswith("bearer ") else ""
        header_key = request.headers.get("x-api-key") or ""
        return settings.api_key in (bearer, header_key)

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        if request.url.path in ("/health", "/"):
            return await call_next(request)
        if not _authorized(request):
            return JSONResponse(
                {"error": {"message": "unauthorized", "type": "preflight_auth"}},
                status_code=401,
            )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "memory_entries": gateway.memory.count(), "version": __version__}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        session_id = request.headers.get(SESSION_HEADER)
        task = request.headers.get(TASK_HEADER)
        if task and "task_type" not in payload:
            payload["task_type"] = task
        if payload.get("stream"):
            payload = {k: v for k, v in payload.items() if k != "stream"}
            return StreamingResponse(
                gateway.handle_stream(payload, session_id),
                media_type="text/event-stream",
            )
        response = await gateway.handle(payload, session_id)
        err = response.get("error") if isinstance(response, dict) else None
        if err and "choices" not in response:
            code = 429 if err.get("type") == "preflight_budget" else 502
            return JSONResponse(response, status_code=code)
        return JSONResponse(response, status_code=200)

    @app.get("/v1/models")
    async def models():
        data = [
            {"id": mid, "object": "model", "owned_by": "preflight"}
            for mid in listed_models()
        ]
        return {"object": "list", "data": data}

    @app.get("/v1/preflight/stats")
    async def stats():
        return gateway.logger.summary()

    @app.get("/preflight", response_class=HTMLResponse)
    async def dashboard():
        return _DASHBOARD_HTML

    return app


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>Preflight stats</title>
<style>
 body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; max-width: 48rem; }
 table { border-collapse: collapse; width: 100%; }
 td, th { text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid #ddd; }
 .muted { color: #666; }
</style>
</head><body>
<h1>Preflight</h1>
<p class="muted">Live outcome-log summary. Polls <code>/v1/preflight/stats</code>.</p>
<div id="summary">Loading…</div>
<script>
async function refresh() {
  const r = await fetch("/v1/preflight/stats");
  const s = await r.json();
  const saved = (s.baseline_usd || 0) - (s.realized_usd || 0);
  let html = `<p>Requests <b>${s.requests||0}</b> · realized
    <b>$${(s.realized_usd||0).toFixed(4)}</b> · baseline
    <b>$${(s.baseline_usd||0).toFixed(4)}</b> · saved
    <b>$${saved.toFixed(4)}</b></p><table><tr><th>Action</th><th>n</th><th>$</th></tr>`;
  for (const [a, row] of Object.entries(s.by_action || {})) {
    html += `<tr><td>${a}</td><td>${row.n}</td><td>$${(row.usd||0).toFixed(4)}</td></tr>`;
  }
  html += "</table>";
  document.getElementById("summary").innerHTML = html;
}
refresh();
setInterval(refresh, 4000);
</script>
</body></html>
"""
