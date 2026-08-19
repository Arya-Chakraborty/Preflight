import litellm
import pytest

from preflight.config import ProviderCacheRule, Settings

ANSWER_TEXT = "The answer is 42."


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
        data_dir=tmp_path / "pf",
        embedder="hashing",
        epsilon=0.0,
        audit_rate=0.0,
        min_tail_tokens=50,
        theta_high=0.95,
        theta_low=0.50,
        semantic_ttl_s=3600,
        cache_rules={
            "openai": ProviderCacheRule(min_prefix_tokens=10, read_mult=0.5, write_mult=1.0, ttl_s=300),
            "anthropic": ProviderCacheRule(min_prefix_tokens=10, read_mult=0.1, write_mult=1.25, ttl_s=300),
            "default": ProviderCacheRule(min_prefix_tokens=10, read_mult=0.5, write_mult=1.0, ttl_s=300),
        },
    )
    s.ensure_dirs()
    return s


@pytest.fixture
def provider_calls(monkeypatch):
    """Mock litellm.acompletion; returns the list of calls made."""
    calls: list[dict] = []

    async def fake_acompletion(model, messages, stream=False, **kwargs):
        calls.append({"model": model, "messages": messages, "stream": stream, **kwargs})
        if stream:
            async def agen():
                yield {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ANSWER_TEXT}, "finish_reason": None}],
                }
                yield {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                }

            return agen()
        return {
            "id": "r1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": ANSWER_TEXT}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    return calls


@pytest.fixture
def failing_provider(monkeypatch):
    async def fake_acompletion(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)


def make_payload(text="What is the capital of France?", system="You are helpful."):
    return {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    }
