import time

from preflight.config import ProviderCacheRule
from preflight.ledger.ledger import PrefixLedger

RULE = ProviderCacheRule(min_prefix_tokens=5, read_mult=0.1, write_mult=1.0, ttl_s=300)

MSGS = [
    {"role": "system", "content": "You are a very helpful assistant with a long system prompt " * 5},
    {"role": "user", "content": "First question about something."},
]


def test_cold_session(tmp_path):
    ledger = PrefixLedger(tmp_path)
    pred = ledger.predict("s1", MSGS, "gpt-4o", RULE)
    assert pred.warm_tokens == 0 and pred.cold_tokens > 0


def test_warm_after_send(tmp_path):
    ledger = PrefixLedger(tmp_path)
    ledger.record_sent("s1", MSGS, "gpt-4o")
    grown = MSGS + [
        {"role": "assistant", "content": "Answer."},
        {"role": "user", "content": "Follow-up."},
    ]
    pred = ledger.predict("s1", grown, "gpt-4o", RULE)
    assert pred.warm_messages == 2
    assert pred.warm_tokens > 0
    assert pred.warm_tokens + pred.cold_tokens == ledger.predict("sX", grown, "gpt-4o", RULE).cold_tokens


def test_mutation_inside_prefix_forfeits_cache(tmp_path):
    """The core cache-aware claim: editing warm bytes re-prices the request."""
    ledger = PrefixLedger(tmp_path)
    ledger.record_sent("s1", MSGS, "gpt-4o")
    mutated = [dict(MSGS[0], content=MSGS[0]["content"] + " EDIT"), MSGS[1]]
    pred = ledger.predict("s1", mutated, "gpt-4o", RULE)
    assert pred.warm_tokens == 0 and pred.warm_messages == 0


def test_ttl_expiry(tmp_path):
    ledger = PrefixLedger(tmp_path)
    ledger.record_sent("s1", MSGS, "gpt-4o")
    rule = ProviderCacheRule(min_prefix_tokens=5, read_mult=0.1, write_mult=1.0, ttl_s=0)
    time.sleep(0.01)
    pred = ledger.predict("s1", MSGS, "gpt-4o", rule)
    assert pred.warm_tokens == 0


def test_min_prefix_threshold(tmp_path):
    ledger = PrefixLedger(tmp_path)
    tiny = [{"role": "user", "content": "hi"}]
    ledger.record_sent("s1", tiny, "gpt-4o")
    rule = ProviderCacheRule(min_prefix_tokens=1024, read_mult=0.1, write_mult=1.0, ttl_s=300)
    pred = ledger.predict("s1", tiny, "gpt-4o", rule)
    assert pred.warm_tokens == 0
