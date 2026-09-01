import pytest
from fastapi.testclient import TestClient

from preflight.proxy.server import create_app
from tests.conftest import ANSWER_TEXT, make_payload


@pytest.fixture
def client(settings, provider_calls):
    app = create_app(settings)
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "pid" in body


def test_ready(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["lock_held"] is True


def test_chat_completion(client):
    resp = client.post("/v1/chat/completions", json=make_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["choices"][0]["message"]["content"] == ANSWER_TEXT
    assert body["preflight"]["action"] == "A5"


def test_chat_completion_stream(client):
    resp = client.post("/v1/chat/completions", json={**make_payload(), "stream": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert ANSWER_TEXT in text
    assert text.rstrip().endswith("data: [DONE]")


def test_session_header_scopes_cache(client, provider_calls):
    payload = make_payload("What is the tallest mountain?")
    client.post("/v1/chat/completions", json=payload, headers={"x-preflight-session": "s1"})
    client.post("/v1/chat/completions", json=payload, headers={"x-preflight-session": "s2"})
    # Exact cache is cross-session by design: second call is served from cache.
    assert len(provider_calls) == 1


def test_upstream_error_is_502(settings, failing_provider):
    app = create_app(settings)
    client = TestClient(app)
    resp = client.post("/v1/chat/completions", json=make_payload())
    assert resp.status_code == 502
    assert "error" in resp.json()


def test_models_and_stats(client):
    models = client.get("/v1/models")
    assert models.status_code == 200
    ids = {m["id"] for m in models.json()["data"]}
    assert "gpt-4o-mini" in ids
    stats = client.get("/v1/preflight/stats")
    assert stats.status_code == 200
    assert "requests" in stats.json()
    dash = client.get("/preflight")
    assert dash.status_code == 200
    assert "Preflight" in dash.text


def test_api_key_required(settings, provider_calls):
    settings.api_key = "secret"
    app = create_app(settings)
    client = TestClient(app)
    denied = client.post("/v1/chat/completions", json=make_payload())
    assert denied.status_code == 401
    ok = client.post(
        "/v1/chat/completions",
        json=make_payload(),
        headers={"x-api-key": "secret"},
    )
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200
