"""Outcome logger: the system's memory of consequences.

Every request writes one row: the action taken, exact token accounting split by
cache hit/miss, realized dollars, latency, and quality signals. This table is
simultaneously (a) the training data for the learned policy and (b) the raw
material for evaluation reports.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from preflight.db import connection, migrate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT,
    model TEXT,
    provider TEXT,
    action TEXT NOT NULL,
    explored INTEGER DEFAULT 0,
    tokens_in_miss INTEGER DEFAULT 0,
    tokens_in_hit INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_estimated REAL DEFAULT 0,
    cost_realized REAL DEFAULT 0,
    cost_baseline REAL DEFAULT 0,      -- what raw passthrough would have cost
    latency_ms REAL DEFAULT 0,
    quality REAL,                      -- judge/audit score in [0,1], NULL if unmeasured
    retry_flag INTEGER DEFAULT 0,      -- set retroactively when a follow-up looks like a retry
    error TEXT,
    features_json TEXT,
    payload_json TEXT,                 -- original request (for replay)
    response_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (ts);
CREATE INDEX IF NOT EXISTS idx_requests_session ON requests (session_id, ts);

CREATE TABLE IF NOT EXISTS audits (
    request_id TEXT NOT NULL,
    ts REAL NOT NULL,
    agreement REAL,                    -- similarity between cached and shadow answer
    shadow_cost REAL DEFAULT 0
);
"""


@dataclass
class Outcome:
    session_id: str
    model: str
    provider: str
    action: str
    explored: bool = False
    tokens_in_miss: int = 0
    tokens_in_hit: int = 0
    tokens_out: int = 0
    cost_estimated: float = 0.0
    cost_realized: float = 0.0
    cost_baseline: float = 0.0
    latency_ms: float = 0.0
    quality: float | None = None
    error: str | None = None
    features: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    response_text: str = ""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class OutcomeLogger:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "outcomes.sqlite3"
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            migrate(conn)

    @contextmanager
    def _conn(self):
        with connection(self._path) as conn:
            yield conn

    def log(self, o: Outcome) -> str:
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO requests
                   (id, ts, session_id, model, provider, action, explored,
                    tokens_in_miss, tokens_in_hit, tokens_out,
                    cost_estimated, cost_realized, cost_baseline, latency_ms,
                    quality, error, features_json, payload_json, response_text)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    o.request_id,
                    time.time(),
                    o.session_id,
                    o.model,
                    o.provider,
                    o.action,
                    int(o.explored),
                    o.tokens_in_miss,
                    o.tokens_in_hit,
                    o.tokens_out,
                    o.cost_estimated,
                    o.cost_realized,
                    o.cost_baseline,
                    o.latency_ms,
                    o.quality,
                    o.error,
                    json.dumps(o.features, default=str),
                    json.dumps(o.payload, default=str),
                    o.response_text,
                ),
            )
        return o.request_id

    def flag_retry(self, request_id: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE requests SET retry_flag = 1 WHERE id = ?", (request_id,))

    def log_audit(self, request_id: str, agreement: float, shadow_cost: float) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO audits (request_id, ts, agreement, shadow_cost) VALUES (?,?,?,?)",
                (request_id, time.time(), agreement, shadow_cost),
            )
            conn.execute(
                "UPDATE requests SET quality = ? WHERE id = ?",
                (agreement, request_id),
            )

    def set_quality(self, request_id: str, quality: float) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("UPDATE requests SET quality = ? WHERE id = ?", (quality, request_id))

    def get(self, request_id: str):
        with self._conn() as conn:
            return conn.execute("SELECT * FROM requests WHERE id = ?", (request_id,)).fetchone()

    def session_spend(self, session_id: str) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(cost_realized), 0) AS usd FROM requests WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return float(row["usd"])

    def last_in_session(self, session_id: str, n: int = 3) -> list[sqlite3.Row]:
        with self._conn() as conn:
            return conn.execute(
                "SELECT * FROM requests WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                (session_id, n),
            ).fetchall()

    def rows(self, limit: int | None = None) -> list[sqlite3.Row]:
        q = "SELECT * FROM requests ORDER BY ts ASC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as conn:
            return conn.execute(q).fetchall()

    def summary(self) -> dict:
        with self._conn() as conn:
            total = conn.execute(
                """SELECT COUNT(*) AS n,
                          COALESCE(SUM(cost_realized), 0) AS realized,
                          COALESCE(SUM(cost_baseline), 0) AS baseline,
                          COALESCE(AVG(latency_ms), 0) AS lat
                   FROM requests"""
            ).fetchone()
            by_action = conn.execute(
                """SELECT action, COUNT(*) AS n,
                          COALESCE(SUM(cost_realized), 0) AS usd,
                          COALESCE(AVG(tokens_out), 0) AS mean_out
                   FROM requests GROUP BY action"""
            ).fetchall()
        return {
            "requests": total["n"],
            "realized_usd": total["realized"],
            "baseline_usd": total["baseline"],
            "mean_latency_ms": total["lat"],
            "by_action": {
                r["action"]: {"n": r["n"], "usd": r["usd"], "mean_out": r["mean_out"]}
                for r in by_action
            },
        }

    def ping(self) -> None:
        from preflight.db import ping as _ping

        _ping(self._path)

    @property
    def path(self) -> Path:
        return self._path
