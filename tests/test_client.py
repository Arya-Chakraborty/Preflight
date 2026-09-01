from preflight.client import PreflightClient
from tests.conftest import ANSWER_TEXT, make_payload


def test_library_mode_create_and_stats(settings, provider_calls):
    client = PreflightClient(settings)
    payload = make_payload("What is the speed of light?")
    resp = client.chat.completions.create(model=payload["model"], messages=payload["messages"])
    assert resp["choices"][0]["message"]["content"] == ANSWER_TEXT
    # Second identical call is a cache hit and must not touch the provider.
    resp2 = client.chat.completions.create(model=payload["model"], messages=payload["messages"])
    assert resp2["preflight"]["action"] == "A1"
    assert len(provider_calls) == 1
    assert client.stats()["requests"] == 2


def test_library_mode_rejects_stream(settings, provider_calls):
    client = PreflightClient(settings)
    payload = make_payload()
    try:
        client.chat.completions.create(
            model=payload["model"], messages=payload["messages"], stream=True
        )
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
