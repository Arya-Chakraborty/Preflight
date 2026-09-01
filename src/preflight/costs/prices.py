"""Per-model price lookup: static table first, litellm's price map as fallback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_tok: float  # USD per input token
    output_per_tok: float  # USD per output token


# Static entries for common models (USD per token). Kept deliberately small;
# anything missing falls through to litellm's continuously-updated model_cost map.
_STATIC: dict[str, Price] = {
    "gpt-4o": Price(2.50e-6, 10.00e-6),
    "gpt-4o-mini": Price(0.15e-6, 0.60e-6),
    "claude-sonnet-4-20250514": Price(3.00e-6, 15.00e-6),
    "claude-haiku-3-5": Price(0.80e-6, 4.00e-6),
    "gemini/gemini-2.0-flash": Price(0.10e-6, 0.40e-6),
}

_DEFAULT = Price(3.00e-6, 15.00e-6)  # conservative frontier-model default


def listed_models() -> list[str]:
    return sorted(_STATIC)


def get_price(model: str) -> Price:
    if model in _STATIC:
        return _STATIC[model]
    try:
        import litellm

        info = litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/")[-1])
        if info and info.get("input_cost_per_token") is not None:
            return Price(
                float(info["input_cost_per_token"]),
                float(info.get("output_cost_per_token") or 0.0),
            )
    except Exception:
        pass
    return _DEFAULT


def provider_of(model: str) -> str:
    """Best-effort provider name for cache-rule lookup."""
    m = model.lower()
    if "/" in m:
        prefix = m.split("/", 1)[0]
        if prefix in ("openai", "anthropic", "gemini", "vertex_ai", "azure"):
            return "gemini" if prefix == "vertex_ai" else prefix
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    return "default"
