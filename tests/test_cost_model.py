from preflight.analyzer.features import Features
from preflight.costs.estimators import FailureEstimator, OutputLenEstimator
from preflight.costs.model import CandidateStats, CostModel
from preflight.costs.prices import get_price, provider_of


def _model(settings):
    return CostModel(settings, OutputLenEstimator(settings), FailureEstimator(settings))


def _features(**kw):
    base = dict(model="gpt-4o", provider="openai", total_tokens=6000)
    base.update(kw)
    return Features(**base)


def test_prices_known_and_provider():
    p = get_price("gpt-4o")
    assert p.input_per_tok == 2.50e-6 and p.output_per_tok == 10.00e-6
    assert provider_of("gpt-4o") == "openai"
    assert provider_of("claude-sonnet-4-20250514") == "anthropic"
    assert provider_of("gemini/gemini-2.0-flash") == "gemini"
    assert provider_of("some-local-model") == "default"


def test_warm_tokens_are_cheaper(settings):
    cm = _model(settings)
    x = _features()
    cold = cm.estimate("A5", x, CandidateStats(warm_tokens=0, cold_tokens=6000))
    warm = cm.estimate("A5", x, CandidateStats(warm_tokens=4000, cold_tokens=2000))
    assert warm.expected_cost < cold.expected_cost
    # openai read_mult=0.5 in test settings: warm input = (2000 + 0.5*4000) * p_in
    p_in = get_price("gpt-4o").input_per_tok
    assert abs(warm.input_cost - (2000 + 0.5 * 4000) * p_in) < 1e-12


def test_worked_example_compression_vs_raw(settings):
    """DESIGN.md 2.4: with a warm prefix, compressing only the tail wins;
    mutating the prefix (losing all warm tokens) loses to raw."""
    cm = _model(settings)
    x = _features(warm_prefix_tokens=4000, tail_tokens=2000)
    raw = cm.estimate("A5", x, CandidateStats(4000, 2000))
    good_compress = cm.estimate("A3", x, CandidateStats(4000, 1000))
    naive_compress = cm.estimate("A3", x, CandidateStats(0, 5000))
    assert good_compress.input_cost < raw.input_cost
    assert naive_compress.input_cost > raw.input_cost


def test_a1_cost_is_pure_risk(settings):
    cm = _model(settings)
    high_sim = _features(max_similarity=0.99)
    low_sim = _features(max_similarity=0.80)
    a1_high = cm.estimate("A1", high_sim, CandidateStats(0, 0))
    a1_low = cm.estimate("A1", low_sim, CandidateStats(0, 0))
    assert a1_high.input_cost == 0.0
    assert a1_high.expected_cost < a1_low.expected_cost


def test_realized_cost_split(settings):
    cm = _model(settings)
    p = get_price("gpt-4o")
    got = cm.realized_cost("gpt-4o", "openai", tokens_in_miss=1000, tokens_in_hit=2000, tokens_out=100)
    want = p.input_per_tok * 1000 + p.input_per_tok * 0.5 * 2000 + p.output_per_tok * 100
    assert abs(got - want) < 1e-12
