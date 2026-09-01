"""Cache threshold ROC report: hit rate vs false-hit rate across thresholds.

Reads the judged pairs produced by `preflight calibrate` and sweeps the serving
threshold, producing the trade-off table that motivates a calibrated theta_high
over fixed-threshold caching (GPTCache-style).

    python benchmarks/cache_roc.py                    # uses ~/.preflight pairs
    python benchmarks/cache_roc.py --pairs my.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from preflight.calibration import CURVE_FILE, PAIRS_FILE, CalibrationCurve
from preflight.config import load_settings


def sweep(pairs: list[dict], thresholds: list[float]) -> list[dict]:
    rows = []
    n = len(pairs)
    for theta in thresholds:
        served = [p for p in pairs if p["sim"] >= theta]
        false_hits = sum(p["false_hit"] for p in served)
        rows.append(
            {
                "theta": round(theta, 3),
                "hit_rate": round(len(served) / n, 4) if n else 0.0,
                "false_hit_rate": round(false_hits / len(served), 4) if served else 0.0,
                "served": len(served),
                "false_hits": false_hits,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    settings = load_settings(args.config)
    pairs_path = args.pairs or (settings.data_dir / PAIRS_FILE)
    if not pairs_path.is_file():
        raise SystemExit(f"No pairs file at {pairs_path}. Run `preflight calibrate` first.")
    pairs = json.loads(pairs_path.read_text())

    thresholds = [round(0.50 + 0.025 * i, 3) for i in range(21)]
    rows = sweep(pairs, thresholds)

    curve = CalibrationCurve.load(settings.data_dir / CURVE_FILE)
    calibrated = curve.recommended_theta if curve else None

    print(f"\n{len(pairs)} judged pairs | calibrated theta_high = {calibrated}")
    print(f"{'theta':>7}{'hit rate':>10}{'false-hit rate':>16}{'served':>8}  note")
    print("-" * 55)
    for r in rows:
        note = ""
        if calibrated is not None and abs(r["theta"] - calibrated) < 0.0125:
            note = "<- calibrated"
        elif abs(r["theta"] - settings.theta_high) < 0.0125:
            note = "<- current config"
        print(
            f"{r['theta']:>7.3f}{r['hit_rate']:>10.1%}{r['false_hit_rate']:>16.1%}"
            f"{r['served']:>8}  {note}"
        )

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "cache_roc.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
