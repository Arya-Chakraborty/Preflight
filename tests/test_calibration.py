import json

import numpy as np
import pytest

from preflight.analyzer.features import Features
from preflight.calibration import (
    CURVE_FILE,
    CalibrationCurve,
    fit_curve,
    generate_pairs_live,
    run_calibration,
)
from preflight.costs.estimators import FailureEstimator
from preflight.memory.store import MemoryStore


def _noisy_pairs(rng, n=200):
    """Synthetic ground truth: risk drops sharply with similarity."""
    sims, labels = [], []
    for _ in range(n):
        s = rng.uniform(0.4, 1.0)
        true_risk = max(0.0, 1.5 * (0.92 - s))
        sims.append(s)
        labels.append(1 if rng.random() < true_risk else 0)
    return sims, labels


def test_fit_is_monotone_and_sane():
    rng = np.random.default_rng(0)
    sims, labels = _noisy_pairs(rng)
    curve = fit_curve(sims, labels, target_rate=0.05)
    assert all(a >= b for a, b in zip(curve.probs, curve.probs[1:]))  # non-increasing
    assert curve.predict(0.5) > curve.predict(0.99)
    assert 0.0 <= curve.predict(1.2) <= curve.predict(0.0) <= 1.0  # clamped at edges
    assert curve.recommended_theta is not None
    assert 0.8 <= curve.recommended_theta <= 1.0


def test_curve_roundtrip(tmp_path):
    curve = fit_curve([0.5, 0.7, 0.9, 0.95, 0.99, 1.0], [1, 1, 0, 0, 0, 0], 0.1)
    path = tmp_path / "c.json"
    curve.save(path)
    loaded = CalibrationCurve.load(path)
    assert loaded.sims == curve.sims and loaded.probs == curve.probs


def test_run_calibration_offline(settings, tmp_path):
    pairs = [
        {"query": f"q{i}", "cached_query": f"q{i}", "cached_answer": "a",
         "sim": s, "false_hit": f}
        for i, (s, f) in enumerate(
            [(0.5, 1), (0.6, 1), (0.7, 1), (0.85, 0), (0.9, 0), (0.96, 0), (0.99, 0)]
        )
    ]
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(json.dumps(pairs))
    report = run_calibration(settings, pairs_file=pairs_file, target_rate=0.05)
    assert report["pairs"] == 7
    assert report["recommended_theta"] is not None
    assert (settings.data_dir / CURVE_FILE).is_file()


def test_run_calibration_needs_enough_pairs(settings, tmp_path):
    pairs_file = tmp_path / "pairs.json"
    pairs_file.write_text(json.dumps([{"query": "q", "cached_query": "q",
                                       "cached_answer": "a", "sim": 0.9, "false_hit": 0}]))
    with pytest.raises(RuntimeError, match="at least 6"):
        run_calibration(settings, pairs_file=pairs_file)


def test_estimator_uses_calibrated_curve(settings):
    # Curve that disagrees strongly with the linear heuristic at sim=0.9:
    # heuristic says alpha*(1-0.9)=0.1; curve says 0.5.
    curve = fit_curve([0.85, 0.88, 0.9, 0.92, 0.97, 0.99], [1, 1, 0, 1, 0, 0], 0.01)
    curve.save(settings.data_dir / CURVE_FILE)
    est = FailureEstimator(settings)
    x = Features(model="m", provider="openai", max_similarity=0.9)
    assert abs(est.predict(x, "A1") - curve.predict(0.9)) < 1e-9


def test_generate_pairs_live_with_mock_llm(settings, monkeypatch):
    import litellm

    from preflight.analyzer.embeddings import HashingEmbedder

    memory = MemoryStore(settings.data_dir, HashingEmbedder(128))
    msgs_a = [{"role": "user", "content": "What is the capital of France?"}]
    msgs_b = [{"role": "user", "content": "How does photosynthesis work?"}]
    memory.store_answer("m", msgs_a, "What is the capital of France?", "Paris.")
    memory.store_answer("m", msgs_b, "How does photosynthesis work?", "Chlorophyll magic.")

    class _Resp:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]

    def fake_completion(model, messages, **kw):
        prompt = messages[0]["content"]
        if "Rewrite" in prompt:
            return _Resp("Could you tell me " + prompt.split("Question: ")[1].lower())
        if "meaningfully different" in prompt:
            return _Resp("What is the population of that place?")
        # judge: valid only when the new question closely resembles the cached one
        return _Resp("yes" if "capital" in prompt.split("New question:")[1][:80].lower()
                     and "capital" in prompt.lower() else "no")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    pairs = generate_pairs_live(settings, n=6, judge_model="mock")
    assert len(pairs) >= 6
    assert all(0.0 <= p["sim"] <= 1.0 and p["false_hit"] in (0, 1) for p in pairs)


def test_retry_seconds_parses_gemini_message():
    from preflight.calibration import _retry_seconds

    msg = 'Please retry in 10.608235171s.'
    assert 10.5 < _retry_seconds(Exception(msg), 2.0) < 12.0
    assert _retry_seconds(Exception("no hint"), 3.0) == 3.0


def test_asker_retries_rate_limit(monkeypatch):
    import litellm

    from preflight.calibration import RateLimitedAsker

    calls = {"n": 0}

    class RateLimitError(Exception):
        pass

    class _Resp:
        def __init__(self, text):
            self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]

    def fake_completion(model, messages, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError('429 Please retry in 0.01s.')
        return _Resp("rewritten question")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    asker = RateLimitedAsker(rpm=0, max_retries=5)
    assert asker.ask("mock", "hi") == "rewritten question"
    assert calls["n"] == 3
