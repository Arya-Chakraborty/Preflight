"""Decision engine: Stage 1 feasibility filter, Stage 2 penalized cost
minimization, Stage 3 epsilon-greedy exploration.

Guarantees:
- A5 (raw passthrough) is always feasible: the system can never do worse than
  not existing.
- A1 is never chosen by exploration; cached-answer quality is audited by
  shadow-calling instead (see gateway.audit).
- Actions whose P[fail] exceeds tau are discarded (hard quality constraint).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from preflight.analyzer.features import Features
from preflight.config import Settings
from preflight.costs.model import Estimate


@dataclass
class Decision:
    action: str
    explored: bool
    estimates: dict[str, Estimate]
    feasible: list[str]


def feasible_actions(
    x: Features,
    settings: Settings,
    has_context_match: bool,
    has_grounding: bool,
) -> list[str]:
    """Stage 1: hard rules. A5 is always feasible."""
    valid: dict[str, bool] = {
        "A1": settings.enable_cache
        and x.max_similarity >= settings.theta_high
        and x.conv_hash_match,
        "A2": settings.enable_context_reuse
        and has_context_match
        and settings.theta_low <= x.context_similarity < settings.theta_high,
        "A3": settings.enable_compression and x.tail_tokens >= settings.min_tail_tokens,
        "A4": settings.enable_grounding and has_grounding and x.grounding_score > 0,
        "A5": True,
    }
    if settings.fixed_action:
        # Baseline mode: force the action when its preconditions hold, else raw.
        return [settings.fixed_action] if valid[settings.fixed_action] else ["A5"]
    return [a for a, ok in valid.items() if ok]


def choose(
    estimates: dict[str, Estimate],
    x: Features,
    settings: Settings,
    rng: random.Random | None = None,
) -> Decision:
    """Stages 2 and 3."""
    rng = rng or random.Random()
    feasible = list(estimates.keys())

    # Stage 2: hard tau constraint, then penalized score.
    admissible = {
        a: e for a, e in estimates.items() if e.p_fail <= settings.tau or a == "A5"
    }
    if not admissible:
        admissible = {"A5": estimates["A5"]}

    def score(e: Estimate) -> float:
        return e.expected_cost + settings.lambda_fail * e.p_fail

    best = min(admissible.values(), key=score).action

    # Stage 3: epsilon-greedy over non-A1 admissible actions.
    explored = False
    explorable = [a for a in admissible if a != "A1"]
    if (
        settings.epsilon > 0
        and not settings.fixed_action
        and len(explorable) > 1
        and rng.random() < settings.epsilon
    ):
        alternative = rng.choice([a for a in explorable if a != best] or [best])
        if alternative != best:
            best, explored = alternative, True

    return Decision(action=best, explored=explored, estimates=estimates, feasible=feasible)
