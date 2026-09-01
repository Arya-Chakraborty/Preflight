# Changelog

## 0.4.0

Operability pre-release: safe to leave a **single** process running on one `data_dir`.

- SQLite `meta.schema_version` (v1 stamp for existing v0.3 files) and a `migrate()` helper
  that refuses a schema newer than the running binary
- Exclusive `preflight.lock` on the data dir (second Gateway fails fast; advisory/`flock`,
  keep `data_dir` on a local disk)
- `GET /health` liveness vs `GET /ready` (disk, lock, sqlite, embedder). With `api_key`
  set, unauthenticated `/ready` returns only `{status, ok}`
- JSON request logs (`request_id`, action, $, latency); `import preflight` does not hijack
  logging (`configure_logging()` is entry-point only)
- FastAPI lifespan + `Gateway.close()`; uvicorn `timeout_graceful_shutdown=30`
- `preflight.prod.yaml`; refuse non-loopback bind without `api_key` and `spend_cap_usd`
- Warning when `embedder: auto` falls back to hashing
- Spend cap uses an in-memory running total (no full-table `SUM` per request)
- Request finish I/O runs off the event loop; auto-refit runs in a background thread
- Estimator model JSON written atomically (temp file + `fsync` + rename)
- API keys compared in constant time; `/v1/chat/completions` rejects malformed JSON (400)
  and oversized bodies (413)
- Dashboard works with `api_key` set (`/preflight?key=YOUR_KEY`)
- `X-Preflight-Request-Id` response header
- Config range validation for thresholds, rates, and spend caps
- `preflight refit` / `ground` / `calibrate` take the same exclusive `data_dir` lock
- Missing `--config` path is an error instead of silent defaults

## 0.3.0

Design-complete alpha: online failure learning, T3 context TTL, Anthropic
`cache_control`, A2/A4 cold billing, A2A3/A4A3 compose, library streaming,
optional auth and spend caps, stats dashboard.
