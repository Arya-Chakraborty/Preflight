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
# optional: simulated failures on ungrounded calls so A4 can look like insurance
python benchmarks/synthetic.py --requests 200 --fail-rate 0.3
```

The provider is mocked locally; token accounting still runs through the real
tokenizer, prefix ledger, and cost model, so relative comparisons are meaningful.

With `--fail-rate 0` the simulated provider never fails, so A4's insurance
value cannot materialize. Use `--fail-rate` (ungrounded calls fail; A4 prompts
contain `Reference material` and succeed) or live traffic plus `preflight refit`.

### Run live

```bash
export OPENAI_API_KEY=sk-...
python benchmarks/synthetic.py --live --model gpt-4o-mini --requests 50
```

Results are printed as a table and written to `results/synthetic.csv`.

## External benchmarks (evaluation roadmap)

For the paper-grade evaluation described in DESIGN.md section 5, see
[`benchmarks/external/README.md`](external/README.md) (tau-bench retail and
LongBench-v2 RAG). Those harnesses are intentionally kept out of CI.
