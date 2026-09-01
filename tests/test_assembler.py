import json

from preflight import tokens
from preflight.analyzer.features import Features
from preflight.assembler.assembler import Assembler
from preflight.assembler.compressor import Compressor, deterministic_clean
from preflight.assembler.grounding import GroundingHit
from preflight.ledger.ledger import LedgerPrediction
from preflight.memory.store import Match

X = Features(model="gpt-4o", provider="openai", total_tokens=1000)

MSGS = [
    {"role": "system", "content": "System prompt."},
    {"role": "user", "content": "Old question with     lots   of   spaces.\nrepeated line here that is long\nrepeated line here that is long"},
    {"role": "assistant", "content": "Old answer."},
    {"role": "user", "content": "New question?"},
]


def test_a3_preserves_warm_prefix_and_final_message(settings):
    asm = Assembler(settings)
    head_tokens = tokens.count_messages(MSGS[:2], X.model)
    pred = LedgerPrediction(warm_tokens=head_tokens, cold_tokens=22, warm_messages=2)
    cand = asm.build("A3", MSGS, X, pred)
    assert cand.messages[0] == MSGS[0]  # warm prefix byte-identical
    assert cand.messages[1] == MSGS[1]
    assert cand.messages[-1] == MSGS[-1]  # query intact
    assert cand.stats.warm_tokens == head_tokens
    assert cand.stats.warm_tokens + cand.stats.cold_tokens == cand.stats.total


def test_a3_compresses_tail(settings):
    asm = Assembler(settings)
    pred = LedgerPrediction(0, 1000, 0)
    cand = asm.build("A3", MSGS, X, pred)
    # the whitespace-heavy user message in the tail got cleaned
    assert "lots   of" not in cand.messages[1]["content"]


def test_a2_injects_before_final(settings):
    asm = Assembler(settings)
    match = Match("id1", 0.9, "Paris.", {"grounding": "France is in Europe."}, "h", 0.0)
    cand = asm.build("A2", MSGS, X, LedgerPrediction(0, 1000, 0), context_match=match)
    assert cand.messages[-2]["role"] == "system"
    assert "France is in Europe" in cand.messages[-2]["content"]
    assert cand.messages[-1] == MSGS[-1]


def test_a4_grounding_block(settings):
    asm = Assembler(settings)
    hits = [GroundingHit("Fact one.", 0.8, "doc.md")]
    cand = asm.build("A4", MSGS, X, LedgerPrediction(0, 1000, 0), grounding_hits=hits)
    assert any("Fact one." in (m.get("content") or "") for m in cand.messages)


def test_a2_injection_is_billed_cold(settings):
    asm = Assembler(settings)
    match = Match("id1", 0.9, "Paris.", {"grounding": "France is in Europe."}, "h", 0.0)
    head_tokens = tokens.count_messages(MSGS[:2], X.model)
    pred = LedgerPrediction(warm_tokens=head_tokens, cold_tokens=10, warm_messages=2)
    cand = asm.build("A2", MSGS, X, pred, context_match=match)
    assert cand.stats.warm_tokens == head_tokens
    assert cand.stats.cold_tokens > 10
    assert cand.messages[:2] == MSGS[:2]


def test_a2a3_injects_then_compresses(settings):
    asm = Assembler(settings)
    match = Match("id1", 0.7, "Paris.", {"grounding": "France is in Europe."}, "h", 0.0)
    cand = asm.build("A2A3", MSGS, X, LedgerPrediction(0, 1000, 0), context_match=match)
    assert cand.action == "A2A3"
    assert any("France is in Europe" in (m.get("content") or "") for m in cand.messages)
    assert cand.messages[-1] == MSGS[-1]


def test_skips_tool_role_compression(settings):
    asm = Assembler(settings)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "tool", "content": "tool output with     extra    spaces"},
        {"role": "user", "content": "q?"},
    ]
    cand = asm.build("A3", msgs, X, LedgerPrediction(0, 100, 0))
    assert cand.messages[1]["content"] == "tool output with     extra    spaces"


def test_json_bypass():
    payload = json.dumps(
        {"key": ["a repeated long array element value", "a repeated long array element value"]},
        indent=4,
    )
    compressor = Compressor(rate=0.3)
    out = compressor.compress(payload)
    parsed = json.loads(out)  # must remain valid JSON
    assert len(parsed["key"]) == 2  # dedup must not eat repeated array elements


def test_deterministic_clean_dedups():
    text = "a long duplicated line of tool output\n" * 3 + "unique"
    out = deterministic_clean(text)
    assert out.count("a long duplicated line") == 1
    assert "unique" in out
