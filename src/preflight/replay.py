"""Offline replay: re-run the decision pipeline over logged traffic.

No API calls are made. For each logged request we rebuild the feature vector,
ask the *current* policy what it would do, and compare its estimated cost with
the historically realized cost. Used for safe policy iteration and for the
counterfactual analysis in evaluation.
"""

from __future__ import annotations

import json
from collections import Counter

from preflight.analyzer.features import Features
from preflight.config import Settings
from preflight.costs.estimators import load_estimators
from preflight.costs.model import CandidateStats, CostModel
from preflight.outcomes.logger import OutcomeLogger
from preflight.policy.engine import choose, feasible_actions


def _stats_for(action: str, x: Features, settings: Settings) -> CandidateStats:
    """Approximate candidate token accounting without rebuilding prompts."""
    warm, tail = x.warm_prefix_tokens, x.tail_tokens
    if action == "A3":
        tail = int(tail * settings.compression_rate)
    elif action == "A2":
        tail += 200  # injected context block, compressed
    elif action == "A4":
        tail += 300  # injected grounding block
    return CandidateStats(warm_tokens=warm, cold_tokens=tail)


def replay_log(settings: Settings, limit: int = 500) -> dict:
    logger = OutcomeLogger(settings.data_dir)
    outlen, pfail = load_estimators(settings)
    cost_model = CostModel(settings, outlen, pfail)

    rows = logger.rows(limit=limit)
    shift: Counter[str] = Counter()
    realized_total = 0.0
    policy_total = 0.0
    n = 0
    for row in rows:
        try:
            feats_raw = json.loads(row["features_json"] or "{}")
            x = Features(**{
                k: v for k, v in feats_raw.items() if k in Features.__dataclass_fields__
            })
        except (TypeError, ValueError):
            continue
        feasible = feasible_actions(
            x,
            settings,
            has_context_match=x.context_similarity > 0,
            has_grounding=x.grounding_score > 0,
        )
        estimates = {
            a: cost_model.estimate(a, x, _stats_for(a, x, settings)) for a in feasible
        }
        decision = choose(estimates, x, settings)
        old_action = row["action"]
        shift[f"{old_action}->{decision.action}"] += 1
        realized_total += row["cost_realized"] or 0.0
        policy_total += estimates[decision.action].expected_cost
        n += 1

    return {
        "rows": n,
        "realized_usd": realized_total,
        "policy_usd": policy_total,
        "shift": dict(shift),
    }
