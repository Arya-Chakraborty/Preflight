"""The cost formula from DESIGN.md section 2.3.

E[cost(a,x)] = p_in * T_miss + p_read * T_hit + p_write_premium * T_newprefix
             + p_out * E[T_out | x,a] + P[fail | x,a] * E[C_retry | x]
"""

from __future__ import annotations

from dataclasses import dataclass

from preflight.analyzer.features import Features
from preflight.config import Settings
from preflight.costs.estimators import FailureEstimator, OutputLenEstimator
from preflight.costs.prices import get_price


@dataclass
class CandidateStats:
    """Token accounting of one candidate prompt (built by the assembler)."""

    warm_tokens: int  # ledger-matched, billed at cached-read rate
    cold_tokens: int  # billed at full rate

    @property
    def total(self) -> int:
        return self.warm_tokens + self.cold_tokens


@dataclass
class Estimate:
    action: str
    expected_cost: float
    p_fail: float
    expected_out_tokens: float
    input_cost: float
    retry_term: float


class CostModel:
    def __init__(
        self,
        settings: Settings,
        outlen: OutputLenEstimator,
        pfail: FailureEstimator,
    ):
        self._s = settings
        self._outlen = outlen
        self._pfail = pfail

    def estimate(self, action: str, x: Features, stats: CandidateStats) -> Estimate:
        price = get_price(x.model)
        rule = self._s.cache_rule_for(x.provider)

        if action == "A1":
            p_fail = self._pfail.predict(x, "A1")
            retry = p_fail * self._retry_cost(x)
            return Estimate("A1", retry, p_fail, 0.0, 0.0, retry)

        input_cost = (
            price.input_per_tok * stats.cold_tokens * rule.write_mult
            + price.input_per_tok * rule.read_mult * stats.warm_tokens
        )
        out_tokens = self._outlen.predict(x, action)
        p_fail = self._pfail.predict(x, action)
        retry_term = p_fail * self._retry_cost(x)
        expected = input_cost + price.output_per_tok * out_tokens + retry_term
        return Estimate(action, expected, p_fail, out_tokens, input_cost, retry_term)

    def _retry_cost(self, x: Features) -> float:
        """E[C_retry]: a retry resends the grown context, so it costs more than the original."""
        price = get_price(x.model)
        out_tokens = self._outlen.predict(x, "A5")
        base = price.input_per_tok * x.total_tokens + price.output_per_tok * out_tokens
        return self._s.retry_cost_mult * base

    def realized_cost(
        self,
        model: str,
        provider: str,
        tokens_in_miss: int,
        tokens_in_hit: int,
        tokens_out: int,
    ) -> float:
        price = get_price(model)
        rule = self._s.cache_rule_for(provider)
        return (
            price.input_per_tok * tokens_in_miss
            + price.input_per_tok * rule.read_mult * tokens_in_hit
            + price.output_per_tok * tokens_out
        )
