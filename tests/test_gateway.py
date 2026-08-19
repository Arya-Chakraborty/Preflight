import pytest

from preflight.gateway import Gateway
from tests.conftest import ANSWER_TEXT, make_payload


@pytest.fixture
def gateway(settings):
    return Gateway(settings)


async def test_passthrough(gateway, provider_calls):
    resp = await gateway.handle(make_payload())
    assert resp["choices"][0]["message"]["content"] == ANSWER_TEXT
    assert resp["preflight"]["action"] == "A5"
    assert len(provider_calls) == 1
    row = gateway.logger.rows()[0]
    assert row["action"] == "A5"
    assert row["cost_realized"] > 0
    assert row["tokens_out"] == 10


async def test_exact_cache_hit_skips_api(gateway, provider_calls):
    payload = make_payload("What is the capital of France?")
    await gateway.handle(payload)
    resp2 = await gateway.handle(payload)
    assert len(provider_calls) == 1  # second answer came from cache
    assert resp2["preflight"]["action"] == "A1"
    assert resp2["choices"][0]["message"]["content"] == ANSWER_TEXT
    actions = [r["action"] for r in gateway.logger.rows()]
    assert actions == ["A5", "A1"]


async def test_analyzer_failure_degrades_to_a5(gateway, provider_calls, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("index corrupted")

    monkeypatch.setattr(gateway.memory, "lookup_exact", boom)
    resp = await gateway.handle(make_payload())
    assert resp["choices"][0]["message"]["content"] == ANSWER_TEXT
    assert resp["preflight"]["action"] == "A5"


async def test_provider_failure_returns_error(gateway, failing_provider):
    resp = await gateway.handle(make_payload())
    assert "error" in resp
    row = gateway.logger.rows()[0]
    assert row["error"] is not None


async def test_retry_detection_flags_previous(gateway, provider_calls):
    session = "fixed-session"
    await gateway.handle(make_payload("Explain quantum entanglement basics"), session)
    await gateway.handle(make_payload("Explain quantum entanglement basics again"), session)
    rows = gateway.logger.rows()
    assert rows[0]["retry_flag"] == 1
    assert rows[1]["retry_flag"] == 0


async def test_streaming_passthrough(gateway, provider_calls):
    lines = [line async for line in gateway.handle_stream(make_payload())]
    assert any(ANSWER_TEXT in line for line in lines)
    assert lines[-1] == "data: [DONE]\n\n"
    row = gateway.logger.rows()[0]
    assert row["action"] == "A5" and row["tokens_out"] == 10


async def test_streaming_cache_hit(gateway, provider_calls):
    payload = make_payload("What is the capital of Spain?")
    await gateway.handle(payload)
    lines = [line async for line in gateway.handle_stream(payload)]
    assert len(provider_calls) == 1
    assert any(ANSWER_TEXT in line for line in lines)
    assert lines[-1] == "data: [DONE]\n\n"


async def test_ledger_warms_across_turns(gateway, provider_calls, settings):
    session = "conv-1"
    long_system = "You are a helpful assistant. " * 40
    p1 = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "First question, please answer."},
        ],
    }
    await gateway.handle(p1, session)
    p2 = {
        "model": "gpt-4o-mini",
        "messages": p1["messages"]
        + [
            {"role": "assistant", "content": ANSWER_TEXT},
            {"role": "user", "content": "A completely different follow-up question."},
        ],
    }
    await gateway.handle(p2, session)
    rows = gateway.logger.rows()
    assert rows[1]["tokens_in_hit"] > 0  # second turn hit the predicted warm prefix
