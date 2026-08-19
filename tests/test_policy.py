import random

from preflight.analyzer.features import Features
from preflight.costs.model import Estimate
from preflight.policy.engine import choose, feasible_actions


def _x(**kw):
    base = dict(model="gpt-4o", provider="openai", total_tokens=3000, tail_tokens=3000)
    base.update(kw)
    return Features(**base)


def _est(action, cost, p_fail=0.05):
    return Estimate(action, cost, p_fail, 200, cost * 0.8, 0.0)


def test_a5_always_feasible(settings):
    acts = feasible_actions(_x(tail_tokens=0), settings, False, False)
    assert acts == ["A5"]


def test_a1_requires_similarity_and_conv_hash(settings):
    x = _x(max_similarity=0.99, conv_hash_match=True)
    assert "A1" in feasible_actions(x, settings, False, False)
    x2 = _x(max_similarity=0.99, conv_hash_match=False)
    assert "A1" not in feasible_actions(x2, settings, False, False)
    x3 = _x(max_similarity=0.90, conv_hash_match=True)  # below theta_high=0.95
    assert "A1" not in feasible_actions(x3, settings, False, False)


def test_a2_band(settings):
    x = _x(context_similarity=0.85)
    assert "A2" in feasible_actions(x, settings, True, False)
    assert "A2" not in feasible_actions(x, settings, False, False)  # no match object
    x_low = _x(context_similarity=0.3)
    assert "A2" not in feasible_actions(x_low, settings, True, False)


def test_a3_needs_min_tail(settings):
    assert "A3" in feasible_actions(_x(tail_tokens=100), settings, False, False)
    assert "A3" not in feasible_actions(_x(tail_tokens=10), settings, False, False)


def test_fixed_action_baseline(settings):
    settings.fixed_action = "A3"
    assert feasible_actions(_x(tail_tokens=100), settings, False, False) == ["A3"]
    assert feasible_actions(_x(tail_tokens=10), settings, False, False) == ["A5"]


def test_choose_argmin(settings):
    estimates = {"A5": _est("A5", 0.02), "A3": _est("A3", 0.01)}
    d = choose(estimates, _x(), settings)
    assert d.action == "A3" and not d.explored


def test_tau_constraint_discards_risky(settings):
    estimates = {"A5": _est("A5", 0.02, p_fail=0.05), "A3": _est("A3", 0.001, p_fail=0.9)}
    d = choose(estimates, _x(), settings)
    assert d.action == "A5"  # A3 cheaper but exceeds tau=0.25


def test_exploration(settings):
    settings.epsilon = 1.0
    estimates = {"A5": _est("A5", 0.02), "A3": _est("A3", 0.01)}
    d = choose(estimates, _x(), settings, rng=random.Random(0))
    assert d.explored and d.action == "A5"  # forced away from the argmin


def test_a1_never_explored(settings):
    settings.epsilon = 1.0
    estimates = {"A5": _est("A5", 0.02), "A1": _est("A1", 0.001, p_fail=0.01)}
    for seed in range(20):
        d = choose(estimates, _x(), settings, rng=random.Random(seed))
        assert not (d.explored and d.action == "A1")
