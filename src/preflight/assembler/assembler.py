"""Candidate prompt construction per action.

Invariant enforced here: messages inside the ledger-matched warm prefix are
NEVER mutated (byte-identical), otherwise the provider cache discount is
forfeited. Compression and injection only touch the dynamic tail, and the final
user message is always left intact.
"""

from __future__ import annotations

from dataclasses import dataclass

from preflight import tokens
from preflight.analyzer.features import Features
from preflight.assembler.compressor import Compressor
from preflight.assembler.grounding import GroundingHit
from preflight.config import Settings
from preflight.costs.model import CandidateStats
from preflight.ledger.ledger import LedgerPrediction
from preflight.memory.store import Match

_CONTEXT_HEADER = (
    "Relevant context from a previous, similar request (may help answer faster; "
    "ignore anything inapplicable):"
)
_GROUNDING_HEADER = "Reference material (use if relevant):"


@dataclass
class Candidate:
    action: str
    messages: list[dict]
    stats: CandidateStats


class Assembler:
    def __init__(self, settings: Settings):
        self._s = settings
        self._compressor = Compressor(rate=settings.compression_rate)

    def build(
        self,
        action: str,
        messages: list[dict],
        x: Features,
        ledger_pred: LedgerPrediction,
        context_match: Match | None = None,
        grounding_hits: list[GroundingHit] | None = None,
    ) -> Candidate:
        if action == "A5":
            return self._finish("A5", messages, x, ledger_pred)
        if action == "A3":
            return self._compress_tail(messages, x, ledger_pred, action="A3")
        if action == "A2":
            return self._inject(
                "A2", messages, x, ledger_pred, self._context_block(context_match)
            )
        if action == "A4":
            return self._inject(
                "A4", messages, x, ledger_pred, self._grounding_block(grounding_hits or [])
            )
        if action == "A2A3":
            injected = self._inject(
                "A2", messages, x, ledger_pred, self._context_block(context_match)
            )
            return self._compress_tail(injected.messages, x, ledger_pred, action="A2A3")
        if action == "A4A3":
            injected = self._inject(
                "A4", messages, x, ledger_pred, self._grounding_block(grounding_hits or [])
            )
            return self._compress_tail(injected.messages, x, ledger_pred, action="A4A3")
        raise ValueError(f"Assembler cannot build action {action!r}")

    # ------------------------------------------------------------------ A3

    def _compress_tail(
        self,
        messages: list[dict],
        x: Features,
        pred: LedgerPrediction,
        action: str = "A3",
    ) -> Candidate:
        warm = pred.warm_messages
        head = messages[:warm]
        tail = messages[warm:]
        new_tail: list[dict] = []
        for i, msg in enumerate(tail):
            is_last = i == len(tail) - 1
            content = msg.get("content")
            role = msg.get("role")
            if (
                is_last
                or role in ("system", "tool")
                or msg.get("tool_calls")
                or not isinstance(content, str)
            ):
                new_tail.append(msg)
                continue
            compressed = self._compressor.compress(content)
            new_tail.append({**msg, "content": compressed})
        return self._finish(action, head + new_tail, x, pred)

    # --------------------------------------------------------------- A2/A4

    def _inject(
        self,
        action: str,
        messages: list[dict],
        x: Features,
        pred: LedgerPrediction,
        block: str,
    ) -> Candidate:
        if not block:
            return self._finish(action, messages, x, pred)
        # Inject just before the final user message, after the warm prefix -
        # position chosen so earlier bytes stay cache-stable.
        injected = {"role": "system", "content": block}
        new_messages = messages[:-1] + [injected, messages[-1]]
        return self._finish(action, new_messages, x, pred)

    def _context_block(self, match: Match | None) -> str:
        if match is None:
            return ""
        parts: list[str] = []
        ctx = match.context or {}
        for key in ("reasoning", "grounding", "tools"):
            if ctx.get(key):
                parts.append(str(ctx[key]))
        if not parts and match.answer_text:
            parts.append(f"A similar question was previously answered: {match.answer_text}")
        if not parts:
            return ""
        body = self._compressor.compress("\n\n".join(parts))
        return f"{_CONTEXT_HEADER}\n{body}"

    def _grounding_block(self, hits: list[GroundingHit]) -> str:
        if not hits:
            return ""
        body = "\n---\n".join(h.text.strip() for h in hits)
        return f"{_GROUNDING_HEADER}\n{body}"

    # -------------------------------------------------------------- shared

    def _finish(
        self, action: str, messages: list[dict], x: Features, pred: LedgerPrediction
    ) -> Candidate:
        total = tokens.count_messages(messages, x.model)
        if pred.warm_messages > 0:
            n_head = min(pred.warm_messages, len(messages))
            warm = tokens.count_messages(messages[:n_head], x.model)
            warm = min(warm, pred.warm_tokens) if pred.warm_tokens else warm
        else:
            warm = min(pred.warm_tokens, total)
        return Candidate(
            action=action,
            messages=messages,
            stats=CandidateStats(warm_tokens=warm, cold_tokens=max(total - warm, 0)),
        )
