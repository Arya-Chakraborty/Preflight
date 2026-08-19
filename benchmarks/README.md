# Preflight benchmarks

## Synthetic redundancy sweep (`synthetic.py`)

Compares six strategies on the same generated query stream with full dollar
accounting from the outcome log:

| Baseline | Meaning |
|---|---|
| `raw` | Every request forwarded unchanged (A5 always) |
| `cache_only` | Semantic answer cache + raw (GPTCache-style) |
| `compress_always` | Compress whenever feasible, else raw (llm-zip-style) |
| `ground_always` | Inject grounding whenever available, else raw |
| `preflight_rules` | Full action space, rule-based policy, no exploration |
| `preflight_learned` | Full action space + bandit exploration + mid-run refit |

The stream generator controls the three axes the paper's claims depend on:

- `--duplicate-rate`: verbatim repeats (exercises the answer cache)
- `--paraphrase-rate`: near-duplicates in the context-reuse band (exercises A2)
- `--long-context-rate`: requests carrying a compressible reference document (exercises A3)

### Run offline (default, $0, no keys needed)

```bash
python benchmarks/synthetic.py --requests 200 --duplicate-rate 0.25
```

The provider is mocked locally; token accounting still runs through the real
tokenizer, prefix ledger, and cost model, so relative comparisons are meaningful.

One caveat to read the mock results correctly: the simulated provider never
fails, so action A4's value proposition (spend grounding tokens now to avoid
retry costs later) cannot materialize, and the cold-start priors make the
policy buy insurance it never needs. On live traffic with real failures the
failure estimator learns actual per-action risk from retry flags; in the mock,
`cache_only` can therefore edge out the full policy. This is expected and is
exactly the estimator-bias phenomenon `preflight refit` exists to correct.

### Run live

```bash
export OPENAI_API_KEY=sk-...
python benchmarks/synthetic.py --live --model gpt-4o-mini --requests 50
```

Results are printed as a table and written to `results/synthetic.csv`.

## External benchmarks (evaluation roadmap)

For the paper-grade evaluation described in DESIGN.md section 5, point the
proxy at real workloads:

1. **tau-bench retail** (verifiable task rewards, agentic): run the agent with
   `base_url=http://127.0.0.1:8411/v1`, one Preflight config per baseline, and
   compare `preflight stats` outputs at equal task reward.
2. **LongBench-v2 RAG**: replay the query distribution through the proxy with
   documents indexed via `preflight ground add`.

Both benchmarks require their own harnesses/API budgets and are intentionally
kept out of this repository's test path.
