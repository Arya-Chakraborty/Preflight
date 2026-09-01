"""Decision engine: Stage 1 feasibility filter, Stage 2 penalized cost
minimization, Stage 3 exploration (ε-greedy and optional Thompson sampling).

Guarantees:
- A5 (raw passthrough) is always feasible: the system can never do worse than
  not existing.
- A1 is never chosen by exploration; cached-answer quality is audited by
  shadow-calling instead (see gateway.audit).
- Actions whose P[fail] exceeds tau are discarded (hard quality constraint).
- When uncertainty_n_min > 0 and the cheapest action has too few observations,
  the engine falls back to A5.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from preflight.analyzer.features import Features
from preflight.config import Settings, canonical_action
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
    if settings.enable_compose:
        valid["A2A3"] = valid["A2"] and valid["A3"]
        valid["A4A3"] = valid["A4"] and valid["A3"]
    if settings.fixed_action:
        # Baseline mode: force the action when its preconditions hold, else raw.
        return [settings.fixed_action] if valid.get(settings.fixed_action) else ["A5"]
    return [a for a, ok in valid.items() if ok]


def choose(
    estimates: dict[str, Estimate],
    x: Features,
    settings: Settings,
    rng: random.Random | None = None,
    obs_counts: dict[str, int] | None = None,
    fail_success: dict[str, tuple[float, float]] | None = None,
) -> Decision:
    """Stages 2 and 3."""
    rng = rng or random.Random()
    feasible = list(estimates.keys())

    # Stage 2: hard tau constraint, then penalized score.
    admissible = {
        a: e for a, e in estimates.items() if e.p_fail <= settings.tau or a == "A5"
    }
    if not admissible:
        admissible = {"A5": estimates["A5"]} if "A5" in estimates else estimates

    def score(e: Estimate) -> float:
        return e.expected_cost + settings.lambda_fail * e.p_fail

    if (
        settings.thompson_sampling
        and not settings.fixed_action
        and fail_success is not None
        and len(admissible) > 1
    ):
        best = _thompson_pick(admissible, settings, rng, fail_success)
    else:
        best = min(admissible.values(), key=score).action

    # Uncertainty guardrail: prefer A5 when the winner is under-observed.
    if (
        settings.uncertainty_fallback
        and settings.uncertainty_n_min > 0
        and not settings.fixed_action
        and obs_counts is not None
        and best not in ("A1", "A5")
        and obs_counts.get(canonical_action(best), 0) < settings.uncertainty_n_min
        and "A5" in admissible
    ):
        best = "A5"

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


def _thompson_pick(
    admissible: dict[str, Estimate],
    settings: Settings,
    rng: random.Random,
    fail_success: dict[str, tuple[float, float]],
) -> str:
    """Sample P[fail] from a Beta posterior and pick the lowest penalized cost."""
    best_a, best_s = "A5", float("inf")
    for a, e in admissible.items():
        fails, succ = fail_success.get(canonical_action(a), (0.0, 0.0))
        p_fail = rng.betavariate(fails + 1.0, succ + 1.0)
        s = e.expected_cost + settings.lambda_fail * p_fail
        if s < best_s:
            best_a, best_s = a, s
    return best_a
