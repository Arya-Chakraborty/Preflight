# Preflight

**A local cost-optimizing gateway for frontier LLM APIs.**

Preflight sits between your application and OpenAI / Anthropic / Gemini and, for
every request, chooses the cheapest way to get an acceptable answer:

| Action | What it does |
|--------|--------------|
| A1 | Serve a semantically-cached answer — no API call at all |
| A2 | Reuse *context* from a similar past request as a head start for the new call |
| A3 | Compress the dynamic prompt tail (cache-aware: never mutates warm prefix bytes) |
| A4 | Inject grounding from a local document store to cut retry risk |
| A5 | Raw passthrough |

The choice is driven by an explicit dollar-level cost model — including
provider prompt-cache pricing predicted by a local **prefix ledger** — and a
policy that learns from your own traffic (contextual bandit over the outcome
log). Guarantee: on any internal failure Preflight degrades to a raw
passthrough, so it is never worse than not having it.

## Install

```bash
pip install "preflight-llm[all] @ git+https://github.com/aryachakraborty/preflight-llm"
# or minimal (deterministic compression + hashing embedder only):
pip install "preflight-llm @ git+https://github.com/aryachakraborty/preflight-llm"
```

Extras: `[memory]` = sentence-transformers embeddings, `[compression]` = LLMLingua-2.

## Quickstart (proxy mode)

```bash
export OPENAI_API_KEY=sk-...          # any provider keys litellm understands
preflight serve                        # listens on http://127.0.0.1:8411/v1
```

Point any OpenAI-compatible client at it — no code changes beyond the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8411/v1", api_key="unused")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is prompt caching?"}],
)
```

Streaming is supported transparently; cached answers are replayed as SSE chunks.

## Quickstart (library mode)

```python
import preflight

client = preflight.wrap()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "hello"}],
)
print(client.stats())
```

## CLI

```bash
preflight serve  [--config preflight.yaml]   # run the proxy
preflight stats                              # spend + action breakdown from the outcome log
preflight refit                              # retrain cost estimators from logged traffic
preflight replay                             # re-simulate logged traffic under the current policy
preflight ground add docs/                   # index documents for grounding (A4)
```

## Configuration

Copy [preflight.example.yaml](preflight.example.yaml) to `preflight.yaml`.
Every field is overridable via `PREFLIGHT_*` environment variables. Key dials:

- `theta_high` / `theta_low`: similarity thresholds for answer-cache and context-reuse.
- `lambda_fail` / `tau`: cost-quality trade-off (dollars per unit failure risk / hard ceiling).
- `epsilon`: bandit exploration rate (0 disables).
- `fixed_action`: force one action — used to build fixed-strategy baselines.

## How it works

See [DESIGN.md](DESIGN.md) for the full architecture: the cost formula, the
prefix ledger that predicts provider cache hits, the three-tier semantic memory,
and the three-stage decision procedure (feasibility filter, penalized cost
minimization, bandit learning).

## Benchmarks

`benchmarks/` contains a synthetic-redundancy harness that compares the learned
policy against fixed strategies (passthrough, cache-only, compress-always) with
full dollar accounting. See [benchmarks/README.md](benchmarks/README.md).

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

MIT license.
