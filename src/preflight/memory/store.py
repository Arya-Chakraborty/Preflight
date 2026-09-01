"""Semantic memory: three tiers backed by SQLite + an in-process vector index.

T1 (exact):    hash of (model, full message list) -> stored answer.
T2 (semantic): embedding similarity >= theta_high, same conversation hash -> stored answer.
T3 (context):  similarity in [theta_low, theta_high) -> stored *supporting context*
               (reasoning, retrieved docs, tool outputs) reusable as grounding.

Vector search is brute-force cosine over a numpy matrix. At gateway scale
(tens of thousands of entries) this is single-digit milliseconds and avoids a
hard FAISS dependency; the matrix is rebuilt lazily after writes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preflight.analyzer.embeddings import Embedder

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,               -- 'answer' | 'context'
    created_at REAL NOT NULL,
    model TEXT,
    exact_key TEXT,
    conv_hash TEXT,
    query_text TEXT,
    answer_text TEXT,
    context_json TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_entries_exact ON entries (exact_key);
CREATE INDEX IF NOT EXISTS idx_entries_kind ON entries (kind, created_at);
"""


def exact_key(model: str, messages: list[dict]) -> str:
    return hashlib.sha256(
        (model + json.dumps(messages, sort_keys=True, default=str)).encode()
    ).hexdigest()


def conversation_hash(messages: list[dict]) -> str:
    """Hash of everything except the final user message (MeanCache-style context chain)."""
    return hashlib.sha256(
        json.dumps(messages[:-1], sort_keys=True, default=str).encode()
    ).hexdigest()


@dataclass
class Match:
    entry_id: str
    similarity: float
    answer_text: str
    context: dict
    conv_hash: str
    created_at: float


class MemoryStore:
    def __init__(self, data_dir: Path, embedder: Embedder | None):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "memory.sqlite3"
        self._embedder = embedder
        self._lock = threading.Lock()
        self._index_dirty = True
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        """Deterministically-closed connection (GC-based closing leaks fds)."""
        conn = sqlite3.connect(self._path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- writes

    def store_answer(
        self,
        model: str,
        messages: list[dict],
        query_text: str,
        answer_text: str,
        context: dict | None = None,
    ) -> str:
        entry_id = uuid.uuid4().hex
        emb = self._embedder.embed(query_text).tobytes() if self._embedder else None
        with self._lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO entries
                   (id, kind, created_at, model, exact_key, conv_hash,
                    query_text, answer_text, context_json, embedding)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry_id,
                    "answer",
                    time.time(),
                    model,
                    exact_key(model, messages),
                    conversation_hash(messages),
                    query_text,
                    answer_text,
                    json.dumps(context or {}, default=str),
                    emb,
                ),
            )
            self._index_dirty = True
        return entry_id

    # ---------------------------------------------------------------- reads

    def lookup_exact(self, model: str, messages: list[dict], ttl_s: int) -> Match | None:
        key = exact_key(model, messages)
        cutoff = time.time() - ttl_s
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM entries
                   WHERE exact_key = ? AND kind = 'answer' AND created_at >= ?
                   ORDER BY created_at DESC LIMIT 1""",
                (key, cutoff),
            ).fetchone()
        return self._row_to_match(row, 1.0) if row else None

    def lookup_semantic(self, query_text: str, ttl_s: int) -> Match | None:
        """Best embedding match among stored answers (caller applies thresholds)."""
        if self._embedder is None:
            return None
        best = self._nearest(query_text, kind="answer", ttl_s=ttl_s)
        return best

    def _nearest(self, query_text: str, kind: str, ttl_s: int) -> Match | None:
        self._ensure_index()
        if self._matrix is None or len(self._ids) == 0:
            return None
        q = self._embedder.embed(query_text)
        sims = self._matrix @ q
        order = np.argsort(-sims)
        cutoff = time.time() - ttl_s
        with self._conn() as conn:
            for idx in order[:10]:
                row = conn.execute(
                    "SELECT * FROM entries WHERE id = ?", (self._ids[int(idx)],)
                ).fetchone()
                if row and row["kind"] == kind and row["created_at"] >= cutoff:
                    return self._row_to_match(row, float(sims[int(idx)]))
        return None

    def _row_to_match(self, row: sqlite3.Row, sim: float) -> Match:
        return Match(
            entry_id=row["id"],
            similarity=sim,
            answer_text=row["answer_text"] or "",
            context=json.loads(row["context_json"] or "{}"),
            conv_hash=row["conv_hash"] or "",
            created_at=row["created_at"],
        )

    # ---------------------------------------------------------------- index

    def _ensure_index(self) -> None:
        with self._lock:
            if not self._index_dirty:
                return
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT id, embedding FROM entries WHERE embedding IS NOT NULL"
                ).fetchall()
            if not rows:
                self._ids, self._matrix = [], None
            else:
                self._ids = [r["id"] for r in rows]
                self._matrix = np.stack(
                    [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                )
            self._index_dirty = False

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def sample_answers(self, k: int) -> list[tuple[str, str]]:
        """Random (query, answer) pairs - the seed data for cache calibration."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT query_text, answer_text FROM entries
                   WHERE kind = 'answer' AND query_text != '' AND answer_text != ''
                   ORDER BY RANDOM() LIMIT ?""",
                (k,),
            ).fetchall()
        return [(r["query_text"], r["answer_text"]) for r in rows]
