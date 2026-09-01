"""Stochastic cost-model components, learned from the outcome log.

Cold start uses conservative priors; `preflight refit` upgrades to regressions
fit with plain numpy (closed-form ridge for output length, gradient-descent
logistic for failure). Parameters persist as JSON, never pickle.
"""

from __future__ import annotations

import json

import numpy as np

from preflight.analyzer.features import Features
from preflight.config import ACTIONS, Settings

_N_FEATURES = 9  # must match Features.vector()


def _design_row(x: Features, action: str) -> np.ndarray:
    onehot = [1.0 if a == action else 0.0 for a in ACTIONS]
    return np.array([1.0] + x.vector() + onehot, dtype=np.float64)


class OutputLenEstimator:
    """E[T_out | x, a]: per-(model, action) running means, optional ridge refit."""

    def __init__(self, settings: Settings):
        self._prior = float(settings.prior_output_tokens)
        self._path = settings.data_dir / "outlen_model.json"
        self._means: dict[str, list[float]] = {}  # key -> [sum, n]
        self._weights: list[float] | None = None
        self._load()

    def _key(self, model: str, action: str) -> str:
        return f"{model}|{action}"

    def predict(self, x: Features, action: str) -> float:
        if self._weights is not None:
            pred = float(_design_row(x, action) @ np.array(self._weights))
            if pred > 0:
                return pred
        s, n = self._means.get(self._key(x.model, action), (0.0, 0.0))
        return (s / n) if n >= 5 else self._prior

    def observe(self, x: Features, action: str, tokens_out: int) -> None:
        key = self._key(x.model, action)
        s, n = self._means.get(key, (0.0, 0.0))
        self._means[key] = [s + tokens_out, n + 1]
        self._save()

    def refit(self, xs: list[Features], actions: list[str], ys: list[int]) -> float:
        """Closed-form ridge regression; returns training MAE."""
        if len(ys) < 20:
            return float("nan")
        X = np.stack([_design_row(x, a) for x, a in zip(xs, actions)])
        y = np.array(ys, dtype=np.float64)
        lam = 1.0
        w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        self._weights = w.tolist()
        self._save()
        return float(np.mean(np.abs(X @ w - y)))

    def _save(self) -> None:
        self._path.write_text(json.dumps({"means": self._means, "weights": self._weights}))

    def _load(self) -> None:
        if self._path.is_file():
            data = json.loads(self._path.read_text())
            self._means = {k: list(v) for k, v in data.get("means", {}).items()}
            self._weights = data.get("weights")


class FailureEstimator:
    """P[fail | x, a]: per-action base rates blended with priors, optional logistic refit.

    A1 (cache-hit) risk is special-cased: it uses the measured isotonic
    calibration curve when `preflight calibrate` has produced one, and the
    linear alpha heuristic only as a cold-start fallback.
    """

    def __init__(self, settings: Settings):
        self._priors = dict(settings.prior_pfail)
        self._alpha = settings.false_hit_alpha
        self._path = settings.data_dir / "pfail_model.json"
        self._counts: dict[str, list[float]] = {}  # action -> [fails, n]
        self._weights: list[float] | None = None
        self._a1_curve = None
        self._load()
        self._load_a1_curve(settings)

    def _load_a1_curve(self, settings: Settings) -> None:
        try:
            from preflight.calibration import CURVE_FILE, CalibrationCurve

            self._a1_curve = CalibrationCurve.load(settings.data_dir / CURVE_FILE)
        except Exception:
            self._a1_curve = None

    def predict(self, x: Features, action: str) -> float:
        if action == "A1":
            # Never learned by exploration (we do not gamble on serving wrong
            # answers): measured calibration curve first, heuristic fallback.
            if self._a1_curve is not None:
                return float(self._a1_curve.predict(x.max_similarity))
            return float(np.clip(self._alpha * (1.0 - x.max_similarity), 0.0, 1.0))
        if self._weights is not None:
            z = float(_design_row(x, action) @ np.array(self._weights))
            return float(1.0 / (1.0 + np.exp(-z)))
        fails, n = self._counts.get(action, (0.0, 0.0))
        prior = self._priors.get(action, 0.05)
        # Beta-style blend: prior counts as 20 pseudo-observations.
        return float((fails + 20 * prior) / (n + 20))

    def observe(self, x: Features, action: str, failed: bool) -> None:
        fails, n = self._counts.get(action, (0.0, 0.0))
        self._counts[action] = [fails + (1.0 if failed else 0.0), n + 1]
        self._save()

    def refit(self, xs: list[Features], actions: list[str], ys: list[bool]) -> None:
        if len(ys) < 50 or len(set(ys)) < 2:
            return
        X = np.stack([_design_row(x, a) for x, a in zip(xs, actions)])
        y = np.array([1.0 if v else 0.0 for v in ys])
        w = np.zeros(X.shape[1])
        lr, lam = 0.1, 1e-3
        for _ in range(500):
            p = 1.0 / (1.0 + np.exp(-(X @ w)))
            grad = X.T @ (p - y) / len(y) + lam * w
            w -= lr * grad
        self._weights = w.tolist()
        self._save()

    def base_rates(self) -> dict[str, float]:
        out = {}
        for action in ACTIONS:
            fails, n = self._counts.get(action, (0.0, 0.0))
            out[action] = round((fails / n) if n else self._priors.get(action, 0.05), 4)
        return out

    def _save(self) -> None:
        self._path.write_text(json.dumps({"counts": self._counts, "weights": self._weights}))

    def _load(self) -> None:
        if self._path.is_file():
            data = json.loads(self._path.read_text())
            self._counts = {k: list(v) for k, v in data.get("counts", {}).items()}
            self._weights = data.get("weights")


def refit_from_log(logger, settings: Settings) -> dict:
    """Rebuild both estimators from the full outcome log (the `preflight refit` command)."""
    rows = logger.rows()
    xs: list[Features] = []
    actions: list[str] = []
    outs: list[int] = []
    fails: list[bool] = []
    for row in rows:
        try:
            feats = Features(**{
                k: v
                for k, v in json.loads(row["features_json"] or "{}").items()
                if k in Features.__dataclass_fields__
            })
        except (TypeError, ValueError):
            continue
        xs.append(feats)
        actions.append(row["action"])
        outs.append(row["tokens_out"] or 0)
        failed = bool(row["retry_flag"]) or (
            row["quality"] is not None and row["quality"] < 0.5
        )
        fails.append(failed)

    outlen = OutputLenEstimator(settings)
    pfail = FailureEstimator(settings)
    mae = outlen.refit(xs, actions, outs)
    pfail.refit(xs, actions, fails)
    return {"rows": len(xs), "outlen_mae": mae, "pfail": pfail.base_rates()}


def load_estimators(settings: Settings) -> tuple[OutputLenEstimator, FailureEstimator]:
    return OutputLenEstimator(settings), FailureEstimator(settings)
