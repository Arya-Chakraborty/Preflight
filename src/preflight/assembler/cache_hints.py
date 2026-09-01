"""Provider cache-control markers applied at call time (not persisted in the ledger).

Anthropic bills prefix cache hits only when the client sets an explicit
`cache_control` breakpoint. OpenAI/Gemini implicit caching needs no markers.
Ledger hashing uses the unmarked messages so the next turn still matches.
"""

from __future__ import annotations

from preflight.ledger.ledger import LedgerPrediction


def apply_cache_hints(
    messages: list[dict],
    provider: str,
    pred: LedgerPrediction,
) -> list[dict]:
    if provider != "anthropic" or pred.warm_messages <= 0 or not messages:
        return messages
    idx = min(pred.warm_messages, len(messages)) - 1
    if idx < 0:
        return messages
    out = list(messages)
    out[idx] = _with_ephemeral_cache(out[idx])
    return out


def _with_ephemeral_cache(msg: dict) -> dict:
    tagged = dict(msg)
    content = tagged.get("content")
    marker = {"type": "ephemeral"}
    if isinstance(content, str):
        tagged["content"] = [
            {"type": "text", "text": content, "cache_control": marker},
        ]
        return tagged
    if isinstance(content, list) and content:
        blocks = [dict(b) if isinstance(b, dict) else b for b in content]
        last = blocks[-1]
        if isinstance(last, dict):
            last = dict(last)
            last["cache_control"] = marker
            blocks[-1] = last
        tagged["content"] = blocks
    return tagged
