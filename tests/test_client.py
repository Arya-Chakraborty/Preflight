from preflight.client import PreflightClient
from tests.conftest import ANSWER_TEXT, make_payload


def test_library_mode_create_and_stats(settings, provider_calls):
    client = PreflightClient(settings)
    try:
        payload = make_payload("What is the speed of light?")
        resp = client.chat.completions.create(model=payload["model"], messages=payload["messages"])
        assert resp["choices"][0]["message"]["content"] == ANSWER_TEXT
        # Second identical call is a cache hit and must not touch the provider.
        resp2 = client.chat.completions.create(model=payload["model"], messages=payload["messages"])
        assert resp2["preflight"]["action"] == "A1"
        assert resp2["preflight"]["request_id"]
        assert len(provider_calls) == 1
        assert client.stats()["requests"] == 2
    finally:
        client.close()


def test_library_mode_stream(settings, provider_calls):
    client = PreflightClient(settings)
    try:
        payload = make_payload()
        chunks = list(
            client.chat.completions.create(
                model=payload["model"], messages=payload["messages"], stream=True
            )
        )
        text = "".join(
            (c.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            for c in chunks
            if isinstance(c, dict)
        )
        assert ANSWER_TEXT in text
    finally:
        client.close()
