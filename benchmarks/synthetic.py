"""Synthetic-redundancy benchmark: compare Preflight's policy against fixed strategies.

Generates a query stream with controlled duplicate / paraphrase / long-context
rates, runs it through a fresh gateway per baseline, and reports full dollar
accounting from the outcome log.

Runs offline by default (--mock): the provider is simulated locally so the
harness needs no API keys and costs $0. Use --live to hit a real cheap model.

    python benchmarks/synthetic.py --requests 200 --duplicate-rate 0.3
    python benchmarks/synthetic.py --live --model gpt-4o-mini --requests 50
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import shutil
import tempfile
from pathlib import Path

import litellm

from preflight.config import ProviderCacheRule, Settings
from preflight.costs.estimators import refit_from_log
from preflight.gateway import Gateway

TOPICS = [
    "the causes of the French Revolution",
    "how photosynthesis works at the molecular level",
    "the difference between TCP and UDP",
    "how prompt caching reduces LLM API costs",
    "the plot of Hamlet act by act",
    "how gradient descent optimizes neural networks",
    "the economics of solar panel adoption",
    "why the sky appears blue during the day",
    "the history of the Silk Road trade routes",
    "how vaccines train the immune system",
    "the rules of chess for beginners",
    "what causes inflation in modern economies",
    "how DNS resolution works step by step",
    "the water cycle and its main stages",
    "how transformers use attention mechanisms",
    "the fall of the Roman Empire",
    "how compound interest grows savings",
    "what black holes are and how they form",
    "the process of cellular respiration",
    "how git branching and merging work",
]

PARAPHRASES = [
    "Can you explain {t}?",
    "Please describe {t} in detail.",
    "I want to understand {t}. Help me out.",
    "Give me an overview of {t}.",
    "Walk me through {t}.",
]

LONG_DOC = (
    "Background reference material follows. "
    + "This paragraph contains verbose, repetitive contextual filler that a good "
    "compressor should shrink substantially without losing the key facts. " * 60
)

BASELINES: dict[str, dict] = {
    "raw": dict(fixed_action="A5"),
    "cache_only": dict(enable_context_reuse=False, enable_compression=False, enable_grounding=False),
    "compress_always": dict(fixed_action="A3"),
    "ground_always": dict(fixed_action="A4"),
    "preflight_rules": dict(epsilon=0.0),
    "preflight_learned": dict(epsilon=0.10),
}


def generate_stream(n: int, duplicate_rate: float, paraphrase_rate: float,
                    long_context_rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    queries: list[str] = []
    payloads: list[dict] = []
    for _ in range(n):
        r = rng.random()
        if queries and r < duplicate_rate:
            text = rng.choice(queries)  # verbatim repeat -> exact/semantic cache hit
        elif queries and r < duplicate_rate + paraphrase_rate:
            base = rng.choice(queries)  # near-duplicate -> context-reuse band
            text = base + " Please keep it brief."
        else:
            topic = rng.choice(TOPICS)
            text = rng.choice(PARAPHRASES).format(t=topic)
        queries.append(text)
        messages = [{"role": "system", "content": "You are a concise, factual assistant."}]
        if rng.random() < long_context_rate:
            messages.append({"role": "user", "content": LONG_DOC})
            messages.append({"role": "assistant", "content": "Noted, I will use this reference."})
        messages.append({"role": "user", "content": text})
        payloads.append({"messages": messages})
    return payloads


def install_mock_provider(seed: int, fail_rate: float = 0.0) -> None:
    rng = random.Random(seed)

    async def fake_acompletion(model, messages, stream=False, **kwargs):
        blob = str(messages)
        grounded = "Reference material" in blob
        if fail_rate > 0 and not grounded and rng.random() < fail_rate:
            raise RuntimeError("simulated provider failure")
        words = 40 + rng.randrange(120)
        text = " ".join(f"tok{i}" for i in range(words))
        return {
            "id": "mock",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": text},
                 "finish_reason": "stop"}
            ],
            # usage omitted on purpose: the gateway then accounts tokens locally
            # via tiktoken + the prefix ledger, exercising the full cost model.
        }

    litellm.acompletion = fake_acompletion


async def run_baseline(name: str, overrides: dict, payloads: list[dict],
                       model: str, refit_midway: bool) -> dict:
    data_dir = Path(tempfile.mkdtemp(prefix=f"pf-bench-{name}-"))
    settings = Settings(
        data_dir=data_dir,
        embedder="hashing",
        audit_rate=0.0,
        min_tail_tokens=500,
        theta_high=0.95,
        theta_low=0.50,
        cache_rules={"default": ProviderCacheRule(min_prefix_tokens=64, read_mult=0.5,
                                                  write_mult=1.0, ttl_s=3600)},
        **overrides,
    )
    gateway = Gateway(settings)
    if settings.enable_grounding:
        for topic in TOPICS:
            gateway.grounding.add_text(f"Key facts about {topic}: (reference summary).")

    for i, payload in enumerate(payloads):
        await gateway.handle({"model": model, **payload}, session_id=f"user-{i % 8}")
        if refit_midway and i == len(payloads) // 2:
            refit_from_log(gateway.logger, settings)

    summary = gateway.logger.summary()
    shutil.rmtree(data_dir, ignore_errors=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--duplicate-rate", type=float, default=0.25)
    ap.add_argument("--paraphrase-rate", type=float, default=0.15)
    ap.add_argument("--long-context-rate", type=float, default=0.30)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live", action="store_true", help="use the real provider API")
    ap.add_argument("--fail-rate", type=float, default=0.0,
                    help="mock-only: fail ungrounded calls this often so A4 can look like insurance")
    ap.add_argument("--baselines", default="all", help="comma list or 'all'")
    args = ap.parse_args()

    if not args.live:
        install_mock_provider(args.seed, fail_rate=args.fail_rate)

    payloads = generate_stream(
        args.requests, args.duplicate_rate, args.paraphrase_rate,
        args.long_context_rate, args.seed,
    )
    names = list(BASELINES) if args.baselines == "all" else args.baselines.split(",")

    if "raw" not in names:
        names = ["raw"] + names  # reference denominator for fair cross-baseline savings

    results = []
    for name in names:
        summary = asyncio.run(
            run_baseline(name, BASELINES[name], payloads, args.model,
                         refit_midway=(name == "preflight_learned"))
        )
        actions = " ".join(f"{a}:{v['n']}" for a, v in sorted(summary["by_action"].items()))
        results.append({
            "baseline": name,
            "requests": summary["requests"],
            "realized_usd": round(summary["realized_usd"], 6),
            "mean_latency_ms": round(summary["mean_latency_ms"], 1),
            "actions": actions,
        })

    # Savings measured against the raw run's *realized* spend: every baseline
    # processed the identical stream, so this is the fair denominator.
    raw_spend = next(r["realized_usd"] for r in results if r["baseline"] == "raw")
    for r in results:
        r["savings_vs_raw_pct"] = round(
            100 * (1 - r["realized_usd"] / raw_spend) if raw_spend else 0.0, 1
        )

    header = f"{'baseline':<18}{'requests':>9}{'spend $':>12}{'vs raw':>9}  actions"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(f"{r['baseline']:<18}{r['requests']:>9}{r['realized_usd']:>12.4f}"
              f"{r['savings_vs_raw_pct']:>8.1f}%  {r['actions']}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "synthetic.csv"
    with open(out_file, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
