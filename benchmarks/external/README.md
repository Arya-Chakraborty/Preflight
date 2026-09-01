# External evaluation harnesses

These are **not** part of the unit-test CI. They spend real API dollars and
depend on third-party benchmark packages. Point the agent or replay script at
the Preflight proxy so every baseline shares the same dollar accounting.

Budget: keep a hard cap (`spend_cap_usd` in `preflight.yaml`) so a runaway
eval cannot blow the account. DESIGN.md §5 suggests $100–200 total.

## tau-bench retail

1. Install [tau-bench](https://github.com/sierra-research/tau-bench) and its retail env.
2. Start Preflight with one config per baseline (`fixed_action: A5`, cache-only, full policy, …):

   ```bash
   export OPENAI_API_KEY=sk-...
   preflight serve --config configs/raw.yaml
   ```

3. Point the agent at `http://127.0.0.1:8411/v1` (OpenAI-compatible). Set
   `x-preflight-session` to the task id so the prefix ledger stays per-task.
4. Compare `preflight stats` (or `GET /v1/preflight/stats`) at equal task reward.

## LongBench-v2 RAG

1. Index the corpus:

   ```bash
   preflight ground /path/to/longbench/docs
   ```

2. Replay LongBench queries through the proxy with the same `base_url`.
3. Report realized $ vs raw and quality delta (benchmark's own scores).

## Notes

- Do not check eval traces or API keys into this repo.
- Use `--fail-rate` on `benchmarks/synthetic.py` for a $0 stand-in of the
  grounding-as-insurance claim before spending on tau-bench.
