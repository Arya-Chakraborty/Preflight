"""Local token counting with tiktoken (never trusts remote estimates)."""

from __future__ import annotations

from functools import lru_cache

import tiktoken

# Per-message structural overhead in the OpenAI chat format.
_MSG_OVERHEAD = 4


@lru_cache(maxsize=16)
def _encoding_for(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_text(text: str, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    return len(_encoding_for(model).encode(text, disallowed_special=()))


def count_message(message: dict, model: str = "gpt-4o") -> int:
    """Token count of one chat message, including structural overhead."""
    total = _MSG_OVERHEAD
    content = message.get("content")
    if isinstance(content, str):
        total += count_text(content, model)
    elif isinstance(content, list):  # multimodal parts; count text parts only
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                total += count_text(part.get("text", ""), model)
    if message.get("name"):
        total += count_text(str(message["name"]), model)
    if message.get("tool_calls"):
        total += count_text(str(message["tool_calls"]), model)
    return total


def count_messages(messages: list[dict], model: str = "gpt-4o") -> int:
    return sum(count_message(m, model) for m in messages) + 2  # priming overhead


def message_text(message: dict) -> str:
    """Plain-text view of a message's content."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""
