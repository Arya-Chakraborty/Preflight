from preflight.analyzer.embeddings import HashingEmbedder
from preflight.memory.store import MemoryStore, conversation_hash

MSGS = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is the capital of France?"},
]


def _store(tmp_path):
    return MemoryStore(tmp_path, HashingEmbedder(128))


def test_exact_hit_and_ttl(tmp_path):
    store = _store(tmp_path)
    store.store_answer("m", MSGS, "What is the capital of France?", "Paris.")
    hit = store.lookup_exact("m", MSGS, ttl_s=3600)
    assert hit is not None and hit.answer_text == "Paris."
    assert store.lookup_exact("m", MSGS, ttl_s=-1) is None  # expired
    assert store.lookup_exact("other-model", MSGS, ttl_s=3600) is None


def test_semantic_hit(tmp_path):
    store = _store(tmp_path)
    store.store_answer("m", MSGS, "What is the capital of France?", "Paris.")
    match = store.lookup_semantic("What is the capital of France?", ttl_s=3600)
    assert match is not None
    assert match.similarity > 0.99
    assert match.conv_hash == conversation_hash(MSGS)
    weak = store.lookup_semantic("How do I bake sourdough bread at home?", ttl_s=3600)
    assert weak is None or weak.similarity < 0.8


def test_context_payload_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.store_answer(
        "m", MSGS, "q", "a", context={"grounding": "France facts", "action": "A5"}
    )
    match = store.lookup_semantic("q", ttl_s=3600)
    assert match.context["grounding"] == "France facts"


def test_t3_context_store_separate_from_answers(tmp_path):
    store = _store(tmp_path)
    store.store_answer("m", MSGS, "What is the capital of France?", "Paris.")
    store.store_context(
        "m",
        MSGS,
        "What is the capital of France?",
        {"grounding": "Europe notes", "tools": "search:france", "reasoning": "it's Paris"},
    )
    ctx = store.lookup_context("What is the capital of France?", ttl_s=3600)
    assert ctx is not None
    assert ctx.context["tools"] == "search:france"
    assert ctx.answer_text == ""
    # TTL: expired context is invisible
    assert store.lookup_context("What is the capital of France?", ttl_s=-1) is None
