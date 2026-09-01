"""Prefix ledger: deterministic prediction of provider prompt-cache hits.

Provider-side caching bills a prompt prefix at a discount when the same bytes
were sent recently. Since *we* sent them, cache hits are predictable: we keep a
per-session hash chain over message boundaries with timestamps, and the longest
still-fresh match against a candidate prompt gives the warm-token count.

Mutating any byte inside a warm prefix forfeits the discount from that point on
- the cost model prices that automatically because the chain match stops at the
mutation point.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from preflight import tokens
from preflight.config import ProviderCacheRule
from preflight.db import connection

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prefix_chains (
    session_id TEXT PRIMARY KEY,
    chain_json TEXT NOT NULL,          -- list of [hash, cumulative_tokens]
    updated_at REAL NOT NULL
);
"""


def chain_of(messages: list[dict], model: str) -> list[tuple[str, int]]:
    """Rolling hash chain over message boundaries with cumulative token counts.

    Starts at 2 to match tokens.count_messages() priming overhead, so ledger
    totals and assembler candidate totals agree exactly.
    """
    chain: list[tuple[str, int]] = []
    h = hashlib.sha256()
    cum = 2
    for msg in messages:
        h.update(json.dumps(msg, sort_keys=True, default=str).encode())
        cum += tokens.count_message(msg, model)
        chain.append((h.hexdigest(), cum))
    return chain


@dataclass
class LedgerPrediction:
    warm_tokens: int  # billed at the cached-read rate
    cold_tokens: int  # billed at full input rate (and possibly cache-write premium)
    warm_messages: int = 0  # how many leading messages form the warm prefix


class PrefixLedger:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "ledger.sqlite3"
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        with connection(self._path) as conn:
            yield conn

    def predict(
        self,
        session_id: str,
        messages: list[dict],
        model: str,
        rule: ProviderCacheRule,
    ) -> LedgerPrediction:
        candidate = chain_of(messages, model)
        total = candidate[-1][1] if candidate else 0
        with self._conn() as conn:
            row = conn.execute(
                "SELECT chain_json, updated_at FROM prefix_chains WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None or (time.time() - row["updated_at"]) > rule.ttl_s:
            return LedgerPrediction(0, total, 0)

        stored = [tuple(x) for x in json.loads(row["chain_json"])]
        warm, warm_msgs = 0, 0
        for i, ((cand_hash, cand_cum), (stored_hash, _)) in enumerate(zip(candidate, stored)):
            if cand_hash != stored_hash:
                break
            warm, warm_msgs = cand_cum, i + 1
        if warm < rule.min_prefix_tokens:
            warm, warm_msgs = 0, 0
        return LedgerPrediction(warm, total - warm, warm_msgs)

    def record_sent(self, session_id: str, messages: list[dict], model: str) -> None:
        chain = chain_of(messages, model)
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO prefix_chains (session_id, chain_json, updated_at)
                   VALUES (?,?,?)
                   ON CONFLICT(session_id) DO UPDATE
                   SET chain_json = excluded.chain_json, updated_at = excluded.updated_at""",
                (session_id, json.dumps(chain), time.time()),
            )
