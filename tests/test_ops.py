import json
import logging
import sqlite3

import pytest

from preflight.analyzer.embeddings import HashingEmbedder
from preflight.config import BindError, Settings, validate_bind
from preflight.db import (
    SCHEMA_VERSION,
    SchemaTooNewError,
    atomic_write_text,
    connection,
    migrate,
    schema_version,
    set_schema_version,
)
from preflight.gateway import Gateway
from preflight.lock import DataDirLock, DataDirLocked
from preflight.outcomes.logger import _SCHEMA, OutcomeLogger
from preflight.proxy.server import create_app
from tests.conftest import make_payload


def test_stamps_v03_sqlite_as_v1(tmp_path):
    path = tmp_path / "outcomes.sqlite3"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.close()
    OutcomeLogger(tmp_path)
    with connection(path) as c:
        assert schema_version(c) == SCHEMA_VERSION == 1


def test_migration_step_runs_once(tmp_path):
    path = tmp_path / "x.sqlite3"
    ran: list[int] = []

    def step2(conn):
        ran.append(2)
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('migrated', 'yes')")

    with connection(path) as conn:
        migrate(conn, target=1)
        migrate(conn, target=2, steps={2: step2})
        migrate(conn, target=2, steps={2: step2})
        assert schema_version(conn) == 2
        assert ran == [2]


def test_second_gateway_same_dir_fails(settings):
    g1 = Gateway(settings)
    with pytest.raises(DataDirLocked, match="already in use"):
        Gateway(settings)
    g1.close()
    g1.close()  # idempotent
    g2 = Gateway(settings)
    g2.close()


def test_separate_dirs_both_ok(tmp_path, settings):
    a = settings.model_copy(update={"data_dir": tmp_path / "a"})
    b = settings.model_copy(update={"data_dir": tmp_path / "b"})
    a.ensure_dirs()
    b.ensure_dirs()
    g1 = Gateway(a)
    g2 = Gateway(b)
    g1.close()
    g2.close()


def test_loopback_bind_ok_without_key(settings):
    validate_bind(settings)


def test_public_bind_requires_key_and_cap(settings):
    settings.host = "0.0.0.0"
    with pytest.raises(BindError, match="api_key"):
        validate_bind(settings)
    settings.api_key = "secret"
    with pytest.raises(BindError, match="spend_cap"):
        validate_bind(settings)
    settings.spend_cap_usd = 5.0
    validate_bind(settings)
    create_app(settings)  # must not raise


def test_hashing_auto_fallback_warns(settings, caplog, monkeypatch):
    settings.embedder = "auto"

    def fake_build(*_a, **_k):
        return HashingEmbedder(32)

    monkeypatch.setattr("preflight.gateway.build_embedder", fake_build)
    with caplog.at_level(logging.WARNING, logger="preflight"):
        g = Gateway(settings)
        g.close()
    assert "hashing" in caplog.text


async def test_request_json_log_fields(settings, provider_calls, caplog):
    gateway = Gateway(settings)
    try:
        with caplog.at_level(logging.INFO, logger="preflight"):
            await gateway.handle(make_payload())
        records = [r for r in caplog.records if getattr(r, "event", None) == "request"]
        assert records
        rec = records[-1]
        assert rec.action == "A5"
        assert rec.request_id
        assert rec.latency_ms >= 0
    finally:
        gateway.close()


def test_migrate_refuses_newer_schema(tmp_path):
    path = tmp_path / "x.sqlite3"
    with connection(path) as conn:
        set_schema_version(conn, SCHEMA_VERSION + 5)
    with connection(path) as conn:
        with pytest.raises(SchemaTooNewError):
            migrate(conn)


def test_atomic_write_text_replaces(tmp_path):
    target = tmp_path / "model.json"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text() == "two"
    # No stray temp files left behind.
    assert list(tmp_path.iterdir()) == [target]


def test_lock_holder_pid_written(tmp_path):
    lock = DataDirLock(tmp_path)
    lock.acquire()
    try:
        assert (tmp_path / "preflight.lock").read_text().strip().isdigit()
    finally:
        lock.release()


def test_settings_reject_out_of_range():
    with pytest.raises(ValueError):
        Settings(theta_high=1.5)
    with pytest.raises(ValueError):
        Settings(epsilon=-0.1)
    with pytest.raises(ValueError):
        Settings(spend_cap_usd=-1)
    with pytest.raises(ValueError):
        Settings(theta_low=0.9, theta_high=0.5)


def test_import_preflight_does_not_hijack_logging():
    logger = logging.getLogger("preflight")
    # A NullHandler is attached, propagation is left to the application.
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


async def test_spend_cap_uses_cached_total(settings, provider_calls, monkeypatch):
    settings.spend_cap_usd = 100.0
    gateway = Gateway(settings)
    calls = {"n": 0}
    real_summary = gateway.logger.summary

    def counting_summary():
        calls["n"] += 1
        return real_summary()

    monkeypatch.setattr(gateway.logger, "summary", counting_summary)
    try:
        for _ in range(3):
            await gateway.handle(make_payload("q"))
        # The global cap must not re-scan the whole table on each request.
        assert calls["n"] == 0
    finally:
        gateway.close()


def test_ready_hides_internals_without_key(settings, provider_calls):
    from fastapi.testclient import TestClient

    settings.api_key = "secret"
    client = TestClient(create_app(settings))
    body = client.get("/ready").json()
    assert set(body) == {"status", "ok"}
    authed = client.get("/ready", headers={"x-api-key": "secret"}).json()
    assert "reasons" in authed and "lock_held" in authed


def test_chat_completion_rejects_bad_json(settings, provider_calls):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(settings))
    resp = client.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


def test_chat_completion_sets_request_id_header(settings, provider_calls):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(settings))
    resp = client.post("/v1/chat/completions", json=make_payload())
    assert resp.headers.get("x-preflight-request-id")


def test_upstream_error_sets_request_id_header(settings, failing_provider):
    from fastapi.testclient import TestClient

    client = TestClient(create_app(settings))
    resp = client.post("/v1/chat/completions", json=make_payload())
    assert resp.status_code == 502
    assert resp.headers.get("x-preflight-request-id")
    assert resp.json()["preflight"]["request_id"]


def test_short_api_key_is_401_not_500(settings, provider_calls):
    from fastapi.testclient import TestClient

    settings.api_key = "secret"
    client = TestClient(create_app(settings))
    resp = client.post(
        "/v1/chat/completions",
        json=make_payload(),
        headers={"x-api-key": "x"},
    )
    assert resp.status_code == 401


def test_chat_rejects_query_string_key(settings, provider_calls):
    from fastapi.testclient import TestClient

    settings.api_key = "secret"
    client = TestClient(create_app(settings))
    resp = client.post("/v1/chat/completions", json=make_payload(), params={"key": "secret"})
    assert resp.status_code == 401


def test_dashboard_embeds_header_key(settings, provider_calls):
    from fastapi.testclient import TestClient

    settings.api_key = "secret"
    client = TestClient(create_app(settings))
    resp = client.get("/preflight", headers={"x-api-key": "secret"})
    assert resp.status_code == 200
    assert json.dumps("secret") in resp.text


def test_ready_503(settings, provider_calls, monkeypatch):
    from fastapi.testclient import TestClient

    app = create_app(settings)
    monkeypatch.setattr(
        app.state.gateway,
        "readiness",
        lambda: {
            "status": "not_ready",
            "ok": False,
            "reasons": ["memory sqlite: boom"],
            "lock_held": False,
            "embedder": None,
        },
    )
    client = TestClient(app)
    resp = client.get("/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert "reasons" in body


def test_dashboard_embeds_key(settings, provider_calls):
    from fastapi.testclient import TestClient

    settings.api_key = "secret"
    client = TestClient(create_app(settings))
    resp = client.get("/preflight", params={"key": "secret"})
    assert resp.status_code == 200
    assert json.dumps("secret") in resp.text
