"""Local grounding store for action A4: a small embedded RAG index.

Documents are added via `preflight ground <path>`; at request time the top
chunks above a relevance floor are offered to the decision engine, which treats
the added tokens as an investment against retry risk.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preflight.analyzer.embeddings import build_embedder
from preflight.config import Settings
from preflight.db import connection, migrate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT,
    text TEXT NOT NULL,
    created_at REAL NOT NULL,
    embedding BLOB NOT NULL
);
"""

_CHUNK_CHARS = 1200
_CHUNK_OVERLAP = 150


@dataclass
class GroundingHit:
    text: str
    score: float
    source: str


class GroundingStore:
    def __init__(self, settings: Settings):
        settings.ensure_dirs()
        self._path = settings.data_dir / "grounding.sqlite3"
        self._embedder = build_embedder(
            settings.embedder, settings.embedding_model, settings.hashing_dim
        )
        self._lock = threading.Lock()
        self._dirty = True
        self._ids: list[str] = []
        self._texts: dict[str, tuple[str, str]] = {}
        self._matrix: np.ndarray | None = None
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            migrate(conn)

    @contextmanager
    def _conn(self):
        with connection(self._path) as conn:
            yield conn

    def add_path(self, path: Path) -> int:
        files = [path] if path.is_file() else sorted(
            p for p in path.rglob("*") if p.suffix.lower() in (".txt", ".md", ".rst")
        )
        n = 0
        for f in files:
            try:
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            n += self.add_text(text, source=str(f))
        return n

    def add_text(self, text: str, source: str = "inline") -> int:
        if self._embedder is None:
            return 0
        step = max(_CHUNK_CHARS - _CHUNK_OVERLAP, 1)
        chunks = [text[i : i + _CHUNK_CHARS] for i in range(0, len(text), step)]
        with self._lock, self._conn() as conn:
            for chunk in chunks:
                if not chunk.strip():
                    continue
                emb = self._embedder.embed(chunk)
                conn.execute(
                    "INSERT INTO chunks (id, source, text, created_at, embedding) VALUES (?,?,?,?,?)",
                    (uuid.uuid4().hex, source, chunk, time.time(), emb.tobytes()),
                )
            self._dirty = True
        return len([c for c in chunks if c.strip()])

    def query(self, text: str, k: int = 3, floor: float = 0.3) -> list[GroundingHit]:
        if self._embedder is None:
            return []
        self._ensure_index()
        if self._matrix is None:
            return []
        q = self._embedder.embed(text)
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        hits = []
        for idx in order:
            score = float(sims[int(idx)])
            if score < floor:
                continue
            chunk_text, source = self._texts[self._ids[int(idx)]]
            hits.append(GroundingHit(chunk_text, score, source))
        return hits

    def _ensure_index(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            with self._conn() as conn:
                rows = conn.execute("SELECT id, source, text, embedding FROM chunks").fetchall()
            if not rows:
                self._ids, self._matrix, self._texts = [], None, {}
            else:
                self._ids = [r["id"] for r in rows]
                self._texts = {r["id"]: (r["text"], r["source"]) for r in rows}
                self._matrix = np.stack(
                    [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                )
            self._dirty = False

    def ping(self) -> None:
        from preflight.db import ping as _ping

        _ping(self._path)
