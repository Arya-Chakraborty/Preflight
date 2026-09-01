from preflight.analyzer.features import Features
from preflight.costs.estimators import FailureEstimator, OutputLenEstimator, refit_from_log
from preflight.gateway import Gateway
from tests.conftest import make_payload


def _x(**kw):
    base = dict(model="gpt-4o", provider="openai", total_tokens=1000)
    base.update(kw)
    return Features(**base)


def test_outlen_prior_then_running_mean(settings):
    est = OutputLenEstimator(settings)
    assert est.predict(_x(), "A5") == settings.prior_output_tokens
    for _ in range(6):
        est.observe(_x(), "A5", 500)
    assert abs(est.predict(_x(), "A5") - 500) < 1e-6


def test_outlen_persistence(settings):
    est = OutputLenEstimator(settings)
    for _ in range(6):
        est.observe(_x(), "A5", 321)
    reloaded = OutputLenEstimator(settings)
    assert abs(reloaded.predict(_x(), "A5") - 321) < 1e-6


def test_outlen_refit_learns_length_relation(settings):
    est = OutputLenEstimator(settings)
    xs, actions, ys = [], [], []
    for n in range(100):
        x = _x(total_tokens=n * 100)
        xs.append(x)
        actions.append("A5")
        ys.append(int(50 + 0.1 * n * 100))  # output grows with input
    mae = est.refit(xs, actions, ys)
    assert mae < 30
    small = est.predict(_x(total_tokens=100), "A5")
    large = est.predict(_x(total_tokens=9000), "A5")
    assert large > small


def test_pfail_a1_similarity_calibrated(settings):
    est = FailureEstimator(settings)
    assert est.predict(_x(max_similarity=1.0), "A1") == 0.0
    assert est.predict(_x(max_similarity=0.9), "A1") > 0.05


def test_pfail_prior_blend(settings):
    est = FailureEstimator(settings)
    p0 = est.predict(_x(), "A3")
    for _ in range(50):
        est.observe(_x(), "A3", failed=True)
    assert est.predict(_x(), "A3") > p0


def test_pfail_revise_to_failed_does_not_double_n(settings):
    est = FailureEstimator(settings)
    est.observe(_x(), "A5", failed=False)
    est.revise_to_failed("A5")
    assert est.n_obs("A5") == 1
    fails, succ = est.fail_success_counts("A5")
    assert fails == 1.0 and succ == 0.0


def test_canonical_composite_action(settings):
    est = FailureEstimator(settings)
    est.observe(_x(), "A2A3", failed=False)
    assert est.n_obs("A2") == 1


async def test_refit_from_log_end_to_end(settings, provider_calls):
    gateway = Gateway(settings)
    try:
        for i in range(5):
            await gateway.handle(make_payload(f"Unique question number {i} about topic {i}?"))
        report = refit_from_log(gateway.logger, settings)
        assert report["rows"] == 5
        assert "A5" in report["pfail"]
    finally:
        gateway.close()
