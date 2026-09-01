"""Shared SQLite helpers: WAL mode, deterministic close, schema versioning."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1


class SchemaTooNewError(RuntimeError):
    """On-disk schema is newer than this binary understands (would be a downgrade)."""


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write `text` to `path` atomically: temp file in the same dir, then rename.

    A crash mid-write can never leave a truncated/corrupt file behind; readers
    see either the old contents or the new contents, never a partial write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_META = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


@contextmanager
def connection(path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def ensure_meta(conn: sqlite3.Connection) -> None:
    conn.executescript(_META)


def schema_version(conn: sqlite3.Connection) -> int:
    ensure_meta(conn)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row["value"]) if row else 0


def set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    ensure_meta(conn)
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(version),),
    )


def migrate(
    conn: sqlite3.Connection,
    target: int = SCHEMA_VERSION,
    steps: dict[int, Callable[[sqlite3.Connection], None]] | None = None,
) -> int:
    """Bring `conn` to `target`. Version 0 (no meta / unstamped) is stamped to 1
    without running a step — that is the v0.3 on-disk layout. Later versions
    use `steps[n]` to go from n-1 to n.

    Refuses to touch a database whose schema is *newer* than `target`: that means
    an older binary opened data written by a newer one, and silently proceeding
    could corrupt it."""
    ensure_meta(conn)
    current = schema_version(conn)
    if current > target:
        raise SchemaTooNewError(
            f"on-disk schema version {current} is newer than supported {target}; "
            "upgrade preflight or point at a fresh data_dir"
        )
    if current == 0:
        set_schema_version(conn, 1)
        current = 1
    steps = steps or {}
    while current < target:
        nxt = current + 1
        fn = steps.get(nxt)
        if fn is None and nxt > 1:
            raise RuntimeError(f"no migration step for schema version {nxt}")
        if fn is not None:
            fn(conn)
        set_schema_version(conn, nxt)
        current = nxt
    return current


def ping(path: Path | str) -> None:
    with connection(path) as conn:
        conn.execute("SELECT 1")
