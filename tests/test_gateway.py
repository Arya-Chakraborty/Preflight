import pytest

from preflight.gateway import Gateway
from tests.conftest import ANSWER_TEXT, make_payload


@pytest.fixture
def gateway(settings):
    g = Gateway(settings)
    yield g
    g.close()


async def test_passthrough(gateway, provider_calls):
    resp = await gateway.handle(make_payload())
    assert resp["choices"][0]["message"]["content"] == ANSWER_TEXT
    assert resp["preflight"]["action"] == "A5"
    assert len(provider_calls) == 1
    row = gateway.logger.rows()[0]
    assert row["action"] == "A5"
    assert row["cost_realized"] > 0
    assert row["tokens_out"] == 10


async def test_a5_baseline_equals_realized(gateway, monkeypatch):
    """A raw passthrough saves nothing by definition: baseline must not be
    inflated by predicted output length (regression test for fake savings)."""
    import litellm

    async def no_usage_provider(model, messages, stream=False, **kwargs):
        return {
            "id": "r1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": ANSWER_TEXT},
                 "finish_reason": "stop"}
            ],
            # no usage: gateway accounts tokens locally for both realized and baseline
        }

    monkeypatch.setattr(litellm, "acompletion", no_usage_provider)
    await gateway.handle(make_payload())
    row = gateway.logger.rows()[0]
    assert row["action"] == "A5"
    assert abs(row["cost_baseline"] - row["cost_realized"]) < 1e-12


async def test_a1_baseline_reflects_answer_length(gateway, provider_calls):
    payload = make_payload("What is the capital of Italy?")
    await gateway.handle(payload)
    await gateway.handle(payload)
    a1_row = gateway.logger.rows()[1]
    assert a1_row["action"] == "A1"
    assert a1_row["cost_realized"] == 0
    assert a1_row["cost_baseline"] > 0  # savings measured against a real-length answer


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
    assert gateway.pfail.n_obs("A5") >= 1


async def test_spend_cap_rejects(gateway, provider_calls, settings):
    settings.spend_cap_usd = 0.0
    resp = await gateway.handle(make_payload("A unique spend-cap question?"))
    assert resp["error"]["type"] == "preflight_budget"


async def test_request_id_on_response(gateway, provider_calls):
    resp = await gateway.handle(make_payload())
    assert resp["preflight"]["request_id"]
    assert resp["preflight"]["action"] == "A5"


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


async def test_anthropic_outbound_has_cache_control(gateway, provider_calls):
    session = "anth-1"
    long_system = "You are a helpful assistant. " * 40
    p1 = {
        "model": "claude-haiku-3-5",
        "messages": [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "First question, please answer."},
        ],
    }
    await gateway.handle(p1, session)
    p2 = {
        "model": "claude-haiku-3-5",
        "messages": p1["messages"]
        + [
            {"role": "assistant", "content": ANSWER_TEXT},
            {"role": "user", "content": "A completely different follow-up question."},
        ],
    }
    await gateway.handle(p2, session)
    outbound = provider_calls[-1]["messages"]
    found = False
    for msg in outbound:
        content = msg.get("content")
        if isinstance(content, list):
            found = any(isinstance(b, dict) and "cache_control" in b for b in content)
            if found:
                break
    assert found
