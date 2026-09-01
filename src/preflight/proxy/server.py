"""OpenAI-compatible local proxy. Point any client at http://host:port/v1."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from preflight import __version__
from preflight.config import Settings, validate_bind
from preflight.costs.prices import listed_models
from preflight.gateway import Gateway
from preflight.logfmt import configure_logging

SESSION_HEADER = "x-preflight-session"
TASK_HEADER = "x-preflight-task"
REQUEST_ID_HEADER = "x-preflight-request-id"
_PROBE_PATHS = {"/health", "/ready", "/"}
# Query-string `key=` is for the dashboard (browsers cannot set headers on a
# page load). Never accept it on the chat API — keys would land in access logs.
_QUERY_KEY_PATHS = {"/preflight", "/ready"}
_MAX_BODY_BYTES = 10 * 1024 * 1024  # reject absurd chat payloads before parsing


def _token_match(given: str, expected: str) -> bool:
    """Constant-time compare that does not raise on length mismatch."""
    if not given:
        return False
    left = hashlib.sha256(given.encode("utf-8")).digest()
    right = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(left, right)


def _request_id_headers(response: dict | None) -> dict[str, str]:
    if not isinstance(response, dict):
        return {}
    rid = (response.get("preflight") or {}).get("request_id")
    return {REQUEST_ID_HEADER: rid} if rid else {}


def create_app(settings: Settings) -> FastAPI:
    configure_logging()
    validate_bind(settings)
    gateway = Gateway(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        gateway.close()

    app = FastAPI(title="preflight", version=__version__, lifespan=lifespan)
    app.state.gateway = gateway

    def _presented_secrets(request: Request) -> tuple[str, ...]:
        auth = request.headers.get("authorization") or ""
        bearer = auth.removeprefix("Bearer ").strip() if auth.lower().startswith("bearer ") else ""
        header_key = request.headers.get("x-api-key") or ""
        secrets = [bearer, header_key]
        if request.url.path in _QUERY_KEY_PATHS:
            secrets.append(request.query_params.get("key") or "")
        return tuple(secrets)

    def _authorized(request: Request) -> bool:
        if not settings.api_key:
            return True
        return any(_token_match(s, settings.api_key) for s in _presented_secrets(request))

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        if request.url.path in _PROBE_PATHS:
            return await call_next(request)
        if not _authorized(request):
            return JSONResponse(
                {"error": {"message": "unauthorized", "type": "preflight_auth"}},
                status_code=401,
            )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": __version__,
            "pid": os.getpid(),
        }

    @app.get("/ready")
    async def ready(request: Request):
        report = gateway.readiness()
        code = 200 if report["ok"] else 503
        # /ready is unauthenticated so orchestrators can probe it, but the full
        # report leaks internal paths/errors — only expose it to authorized callers.
        if settings.api_key and not _authorized(request):
            report = {"status": report["status"], "ok": report["ok"]}
        return JSONResponse(report, status_code=code)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        raw = await request.body()
        if len(raw) > _MAX_BODY_BYTES:
            return JSONResponse(
                {"error": {"message": "request body too large", "type": "preflight_request"}},
                status_code=413,
            )
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(
                {"error": {"message": "invalid JSON body", "type": "preflight_request"}},
                status_code=400,
            )
        if not isinstance(payload, dict):
            return JSONResponse(
                {"error": {"message": "body must be a JSON object", "type": "preflight_request"}},
                status_code=400,
            )
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
        headers = _request_id_headers(response)
        err = response.get("error") if isinstance(response, dict) else None
        if err and "choices" not in response:
            code = 429 if err.get("type") == "preflight_budget" else 502
            return JSONResponse(response, status_code=code, headers=headers)
        return JSONResponse(response, status_code=200, headers=headers)

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
    async def dashboard(request: Request):
        # Embed whichever secret authenticated this request so the page's fetch
        # to /v1/preflight/stats can send x-api-key. Open with /preflight?key=.
        key = ""
        if settings.api_key:
            key = next(
                (s for s in _presented_secrets(request) if _token_match(s, settings.api_key)),
                "",
            )
        return _dashboard_html(key)

    return app


def _dashboard_html(api_key: str) -> str:
    return _DASHBOARD_HTML.replace("__PREFLIGHT_KEY__", json.dumps(api_key))


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
const PREFLIGHT_KEY = __PREFLIGHT_KEY__;
async function refresh() {
  const headers = PREFLIGHT_KEY ? {"x-api-key": PREFLIGHT_KEY} : {};
  const r = await fetch("/v1/preflight/stats", {headers});
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
