from preflight.assembler.cache_hints import apply_cache_hints
from preflight.ledger.ledger import LedgerPrediction


def test_anthropic_breakpoint_on_last_warm_message():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "follow up"},
    ]
    pred = LedgerPrediction(warm_tokens=40, cold_tokens=10, warm_messages=2)
    out = apply_cache_hints(msgs, "anthropic", pred)
    assert msgs[0]["content"] == "sys"  # originals unmarked
    content = out[1]["content"]
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert out[2]["content"] == "hi"


def test_openai_gets_no_markers():
    msgs = [{"role": "user", "content": "hi"}]
    pred = LedgerPrediction(10, 0, 1)
    assert apply_cache_hints(msgs, "openai", pred) == msgs
