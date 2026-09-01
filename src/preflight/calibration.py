"""Measured false-hit calibration for the semantic answer cache (action A1).

Replaces the linear heuristic P[false-hit | similarity] = alpha * (1 - sim)
with a curve fitted to judged evidence. Pipeline:

1. Build (new_query, cached_query, cached_answer) pairs spanning the similarity
   spectrum - LLM paraphrases (should be valid hits), related-but-different
   questions (should be invalid), and unrelated pairs (clearly invalid).
2. An LLM judge labels whether the cached answer actually answers the new query.
3. Isotonic regression (PAVA) fits a monotonically non-increasing risk curve
   over similarity - the shape is enforced, the values are measured.
4. theta_high is then derived: the lowest similarity whose measured risk stays
   under a target false-hit rate, instead of a hand-picked constant.

The fitted curve is persisted as JSON and picked up by FailureEstimator, so the
runtime policy prices A1 risk from evidence rather than assumption.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from preflight.config import Settings

CURVE_FILE = "a1_calibration.json"
PAIRS_FILE = "calibration_pairs.json"

_PARAPHRASE_PROMPT = (
    "Rewrite the following question using different wording but the exact same "
    "meaning. Reply with only the rewritten question.\n\nQuestion: {q}"
)
_RELATED_PROMPT = (
    "Write a question about the same topic as the following one, but asking for "
    "something meaningfully different, so that an answer to the original question "
    "would NOT correctly answer yours. Reply with only the new question.\n\n"
    "Question: {q}"
)
_JUDGE_PROMPT = (
    "You are grading a cached-answer system.\n\n"
    "New question: {q}\n\n"
    "Candidate answer (written for a possibly different question): {a}\n\n"
    "Does the candidate answer correctly and sufficiently answer the NEW "
    "question? Reply with exactly one word: yes or no."
)


@dataclass
class CalibrationCurve:
    """Non-increasing risk curve over similarity, linear-interpolated."""

    sims: list[float]  # ascending
    probs: list[float]  # non-increasing
    n_pairs: int = 0
    recommended_theta: float | None = None
    fitted_at: float = field(default_factory=time.time)

    def predict(self, sim: float) -> float:
        if not self.sims:
            return 0.0
        if sim <= self.sims[0]:
            return self.probs[0]
        if sim >= self.sims[-1]:
            return self.probs[-1]
        return float(np.interp(sim, self.sims, self.probs))

    def theta_for(self, target_rate: float) -> float | None:
        """Lowest similarity whose fitted risk is <= target_rate."""
        for s, p in zip(self.sims, self.probs):
            if p <= target_rate:
                return round(s, 4)
        return None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__))

    @classmethod
    def load(cls, path: Path) -> CalibrationCurve | None:
        if not path.is_file():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (ValueError, TypeError):
            return None


def _pava_nondecreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: least-squares non-decreasing fit."""
    vals: list[float] = []
    wts: list[float] = []
    counts: list[int] = []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        counts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]
            counts[-2] += counts[-1]
            vals[-2] = merged
            vals.pop()
            wts.pop()
            counts.pop()
    out = np.empty(len(y))
    i = 0
    for v, c in zip(vals, counts):
        out[i : i + c] = v
        i += c
    return out


def fit_curve(sims: list[float], false_hits: list[int], target_rate: float = 0.01) -> CalibrationCurve:
    """Fit a non-increasing P[false-hit | sim] curve via isotonic regression."""
    order = np.argsort(sims)
    s = np.asarray(sims, dtype=float)[order]
    y = np.asarray(false_hits, dtype=float)[order]
    # Risk must be non-increasing in similarity: fit non-decreasing on the
    # reversed sequence, then reverse back.
    fitted = _pava_nondecreasing(y[::-1], np.ones(len(y)))[::-1]
    curve = CalibrationCurve(
        sims=[round(float(v), 6) for v in s],
        probs=[round(float(v), 6) for v in fitted],
        n_pairs=len(s),
    )
    curve.recommended_theta = curve.theta_for(target_rate)
    return curve


# ------------------------------------------------------------------ pair generation


def _retry_seconds(exc: Exception, fallback: float) -> float:
    """Parse Gemini's RetryInfo delay when present; otherwise exponential backoff."""
    text = str(exc)
    match = re.search(r"[Pp]lease retry in ([0-9.]+)s", text)
    if match:
        return max(float(match.group(1)) + 0.5, fallback)
    match = re.search(r'"retryDelay":\s*"(\d+)s"', text)
    if match:
        return max(float(match.group(1)) + 0.5, fallback)
    return fallback


class RateLimitedAsker:
    """Serial LLM caller that stays under a requests-per-minute budget.

    Gemini free tier is 15 RPM; calibrate issues several calls per seed, so
    without pacing it dies mid-run. 429s are retried with the provider's delay.
    """

    def __init__(self, rpm: float = 15.0, max_retries: int = 8):
        self._min_interval = (60.0 / rpm) if rpm > 0 else 0.0
        self._max_retries = max_retries
        self._last = 0.0

    def _pace(self) -> None:
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)

    def ask(self, model: str, prompt: str) -> str:
        import litellm

        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            self._pace()
            try:
                resp = litellm.completion(
                    model=model, messages=[{"role": "user", "content": prompt}]
                )
                self._last = time.time()
                return (resp.choices[0].message.content or "").strip()
            except Exception as exc:
                self._last = time.time()
                last_exc = exc
                name = type(exc).__name__
                text = str(exc)
                is_rate = "RateLimit" in name or "429" in text or "RESOURCE_EXHAUSTED" in text
                if not is_rate or attempt == self._max_retries - 1:
                    raise
                sleep_for = _retry_seconds(exc, delay)
                time.sleep(sleep_for)
                delay = min(delay * 2, 60.0)
        raise last_exc or RuntimeError("LLM call failed")

    def judge(self, model: str, query: str, answer: str) -> int:
        verdict = self.ask(model, _JUDGE_PROMPT.format(q=query, a=answer)).lower()
        return 0 if verdict.startswith("yes") else 1


def generate_pairs_live(
    settings: Settings,
    n: int,
    judge_model: str,
    rpm: float = 0.0,
    progress=None,
) -> list[dict]:
    """Build labeled pairs from the answers already stored in semantic memory."""
    from preflight.analyzer.embeddings import build_embedder, cosine
    from preflight.memory.store import MemoryStore

    embedder = build_embedder(settings.embedder, settings.embedding_model, settings.hashing_dim)
    if embedder is None:
        raise RuntimeError("Calibration requires an embedder (settings.embedder != 'off').")
    memory = MemoryStore(settings.data_dir, embedder)
    entries = memory.sample_answers(max(n // 3 + 1, 2))
    if len(entries) < 2:
        raise RuntimeError(
            "Need at least 2 cached answers to calibrate. Run some live traffic first."
        )

    asker = RateLimitedAsker(rpm=rpm)
    pairs: list[dict] = []
    scratch = settings.data_dir / PAIRS_FILE
    try:
        for i, (query, answer) in enumerate(entries):
            if len(pairs) >= n:
                break
            variants = [
                asker.ask(judge_model, _PARAPHRASE_PROMPT.format(q=query)),
                asker.ask(judge_model, _RELATED_PROMPT.format(q=query)),
                entries[(i + 1) % len(entries)][0],
            ]
            for new_query in variants:
                if not new_query or len(pairs) >= n:
                    continue
                sim = cosine(embedder.embed(new_query), embedder.embed(query))
                pairs.append(
                    {
                        "query": new_query,
                        "cached_query": query,
                        "cached_answer": answer,
                        "sim": round(float(sim), 6),
                        "false_hit": asker.judge(judge_model, new_query, answer),
                    }
                )
                scratch.write_text(json.dumps(pairs, indent=1))
                if progress is not None:
                    progress(len(pairs), n)
    except Exception:
        if len(pairs) >= 6:
            scratch.write_text(json.dumps(pairs, indent=1))
            return pairs
        raise
    return pairs


def load_pairs_file(settings: Settings, path: Path) -> list[dict]:
    """Load externally-labeled pairs; (re)compute sims with the runtime embedder
    so the curve matches what the gateway will actually measure at serve time."""
    from preflight.analyzer.embeddings import build_embedder, cosine

    pairs = json.loads(Path(path).read_text())
    embedder = build_embedder(settings.embedder, settings.embedding_model, settings.hashing_dim)
    for p in pairs:
        if "sim" not in p and embedder is not None:
            p["sim"] = round(
                float(cosine(embedder.embed(p["query"]), embedder.embed(p["cached_query"]))), 6
            )
    return pairs


def run_calibration(
    settings: Settings,
    pairs_file: Path | None = None,
    n: int = 30,
    judge_model: str = "gemini/gemini-3.5-flash-lite",
    target_rate: float = 0.01,
    rpm: float = 15.0,
    progress=None,
) -> dict:
    if pairs_file is not None:
        pairs = load_pairs_file(settings, pairs_file)
    else:
        pairs = generate_pairs_live(
            settings, n, judge_model, rpm=rpm, progress=progress
        )
    if len(pairs) < 6:
        raise RuntimeError(f"Only {len(pairs)} pairs; need at least 6 for a meaningful fit.")

    curve = fit_curve([p["sim"] for p in pairs], [p["false_hit"] for p in pairs], target_rate)
    settings.ensure_dirs()
    curve.save(settings.data_dir / CURVE_FILE)
    (settings.data_dir / PAIRS_FILE).write_text(json.dumps(pairs, indent=1))
    return {
        "pairs": len(pairs),
        "false_hit_base_rate": round(float(np.mean([p["false_hit"] for p in pairs])), 4),
        "recommended_theta": curve.recommended_theta,
        "target_rate": target_rate,
        "curve_file": str(settings.data_dir / CURVE_FILE),
        "pairs_file": str(settings.data_dir / PAIRS_FILE),
    }
