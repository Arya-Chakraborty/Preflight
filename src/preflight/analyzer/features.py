"""Per-request feature extraction: everything the decision engine sees."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from preflight import tokens

_MATH_RE = re.compile(r"(\d+\s*[-+*/^=]\s*\d+)|\b(solve|integral|derivative|equation|proof)\b", re.I)
_CODE_RE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|\bfunction\b|\bimport\s+\w+|;\s*$", re.M)
_MULTIPART_RE = re.compile(r"\b(first|second|then|finally|step \d|1\.|2\.|3\.)\b", re.I)


@dataclass
class Features:
    model: str = ""
    provider: str = "default"
    session_id: str = ""
    total_tokens: int = 0
    warm_prefix_tokens: int = 0  # ledger-predicted provider-cache hits
    tail_tokens: int = 0  # dynamic tokens after the warm prefix
    query_tokens: int = 0
    structure_fraction: float = 0.0  # share of content that is JSON/code
    difficulty: float = 0.0  # heuristic in [0,1]
    max_similarity: float = 0.0  # best T1/T2 match
    context_similarity: float = 0.0  # best T3 match in the reuse band
    conv_hash_match: bool = False
    grounding_score: float = 0.0  # best retrieval score from the grounding store
    matched_answer_id: str | None = None
    matched_context_id: str | None = None
    task_type: str = ""  # optional caller tag (chat / rag / agent-tool)

    def vector(self) -> list[float]:
        """Numeric form used by the estimators."""
        return [
            self.total_tokens / 1000.0,
            self.warm_prefix_tokens / 1000.0,
            self.tail_tokens / 1000.0,
            self.query_tokens / 1000.0,
            self.structure_fraction,
            self.difficulty,
            self.max_similarity,
            self.context_similarity,
            self.grounding_score,
        ]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def structure_fraction(text: str) -> float:
    """Approximate share of text that is structured (JSON or code)."""
    if not text:
        return 0.0
    stripped = text.strip()
    try:
        json.loads(stripped)
        return 1.0
    except (ValueError, TypeError):
        pass
    structured_chars = 0
    for block in re.findall(r"```.*?```", text, re.S):
        structured_chars += len(block)
    remainder = re.sub(r"```.*?```", "", text, flags=re.S)
    if _CODE_RE.search(remainder):
        structured_chars += int(0.3 * len(remainder))
    return min(structured_chars / max(len(text), 1), 1.0)


def difficulty_score(text: str) -> float:
    score = 0.0
    if _MATH_RE.search(text):
        score += 0.4
    if _CODE_RE.search(text):
        score += 0.3
    if _MULTIPART_RE.search(text):
        score += 0.2
    if len(text) > 2000:
        score += 0.1
    return min(score, 1.0)


def base_features(messages: list[dict], model: str, provider: str, session_id: str) -> Features:
    """Features computable without memory/ledger lookups (those are filled in later)."""
    query = tokens.message_text(messages[-1]) if messages else ""
    all_text = "\n".join(tokens.message_text(m) for m in messages)
    return Features(
        model=model,
        provider=provider,
        session_id=session_id,
        total_tokens=tokens.count_messages(messages, model),
        query_tokens=tokens.count_text(query, model),
        structure_fraction=structure_fraction(all_text),
        difficulty=difficulty_score(query),
    )
